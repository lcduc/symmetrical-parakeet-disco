import asyncio
import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from config.settings import settings
from models.orm_models import EmbeddingCache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Computes and caches text embeddings in the embedding_cache table.

    The embedding model is expensive to load, so it must be constructed once
    at app startup and injected here rather than reloaded per call.
    """

    def __init__(self, model: SentenceTransformer, session_factory: async_sessionmaker):
        self._model = model
        self._session_factory = session_factory

    async def embed_cached(self, key: str, text: str) -> np.ndarray:
        """Return the embedding for `text`, reusing the cached vector for `key` if present.

        The cache key is namespaced by `settings.embedding_model_name` so that
        switching embedding models invalidates old entries automatically
        instead of silently serving vectors from a different embedding space.
        """
        key = f"{settings.embedding_model_name}:{key}"
        async with self._session_factory() as session:
            cached = await session.get(EmbeddingCache, key)
            if cached is not None:
                logger.debug("Embedding cache hit for key %s", key)
                return np.array(cached.vector, dtype=np.float32)

            logger.info("Embedding cache miss for key %s; computing embedding", key)
            vector = await asyncio.to_thread(self._model.encode, text, normalize_embeddings=True)
            # ON CONFLICT DO NOTHING: two concurrent requests computing the same
            # not-yet-cached key would otherwise race on the cache_key primary
            # key and one would raise IntegrityError.
            stmt = (
                insert(EmbeddingCache)
                .values(cache_key=key, vector=vector.tolist())
                .on_conflict_do_nothing(index_elements=[EmbeddingCache.cache_key])
            )
            await session.execute(stmt)
            await session.commit()
            return vector
