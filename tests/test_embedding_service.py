from types import SimpleNamespace
from typing import Self
from unittest.mock import MagicMock

import numpy as np
import pytest

from services.embedding_service import EmbeddingService


class _FakeAsyncSession:
    """Minimal async session stand-in backed by a plain dict, keyed like embedding_cache."""

    def __init__(self, store: dict):
        self._store = store

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def get(self, _model, key: str):
        return self._store.get(key)

    def add(self, obj) -> None:
        self._store[obj.cache_key] = obj

    async def execute(self, stmt) -> None:
        """Fake enough of `INSERT ... ON CONFLICT DO NOTHING` to back embed_cached's cache-miss path."""
        params = stmt.compile().params
        self._store.setdefault(params["cache_key"], SimpleNamespace(vector=params["vector"]))

    async def commit(self) -> None:
        pass


class _FakeSessionFactory:
    """Stand-in for async_sessionmaker: shares one in-memory store across sessions."""

    def __init__(self):
        self._store: dict = {}

    def __call__(self) -> _FakeAsyncSession:
        return _FakeAsyncSession(self._store)


@pytest.mark.asyncio
async def test_embed_cached_reuses_stored_vector_without_recomputing():
    model = MagicMock()
    model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    service = EmbeddingService(model=model, session_factory=_FakeSessionFactory())

    first = await service.embed_cached("candidate-1", "some resume text")
    second = await service.embed_cached("candidate-1", "some resume text")

    assert model.encode.call_count == 1
    np.testing.assert_array_equal(first, second)


@pytest.mark.asyncio
async def test_embed_cached_recomputes_for_different_keys():
    model = MagicMock()
    model.encode.side_effect = [
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.9, 0.8], dtype=np.float32),
    ]
    service = EmbeddingService(model=model, session_factory=_FakeSessionFactory())

    first = await service.embed_cached("candidate-1", "text a")
    second = await service.embed_cached("candidate-2", "text b")

    assert model.encode.call_count == 2
    assert not np.array_equal(first, second)
