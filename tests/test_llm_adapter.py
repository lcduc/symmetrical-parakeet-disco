from unittest.mock import patch

import pytest

from services.llm_adapter import (
    AnthropicAdapter,
    GoogleAdapter,
    OpenAIAdapter,
    build_llm_adapter,
)


def test_build_llm_adapter_uses_provider_default_model():
    with patch("services.llm_adapter.settings.llm_provider", "openai"), patch(
        "services.llm_adapter.settings.llm_model", None
    ):
        adapter = build_llm_adapter()

    assert isinstance(adapter, OpenAIAdapter)
    assert adapter._model == OpenAIAdapter.DEFAULT_MODEL


def test_build_llm_adapter_honors_llm_model_override():
    with patch("services.llm_adapter.settings.llm_provider", "anthropic"), patch(
        "services.llm_adapter.settings.llm_model", "claude-opus-4-6"
    ):
        adapter = build_llm_adapter()

    assert isinstance(adapter, AnthropicAdapter)
    assert adapter._model == "claude-opus-4-6"


def test_build_llm_adapter_dispatches_to_google():
    with patch("services.llm_adapter.settings.llm_provider", "google"), patch(
        "services.llm_adapter.settings.llm_model", None
    ):
        adapter = build_llm_adapter()

    assert isinstance(adapter, GoogleAdapter)
    assert adapter._model == GoogleAdapter.DEFAULT_MODEL


def test_build_llm_adapter_rejects_unknown_provider():
    with patch("services.llm_adapter.settings.llm_provider", "not_a_real_provider"), pytest.raises(ValueError):
        build_llm_adapter()


def test_build_llm_adapter_model_arg_overrides_settings_llm_model():
    with patch("services.llm_adapter.settings.llm_provider", "openai"), patch(
        "services.llm_adapter.settings.llm_model", "gpt-4.1-mini"
    ):
        adapter = build_llm_adapter(model="gpt-4o")

    assert adapter._model == "gpt-4o"


def test_build_llm_adapter_falls_back_to_settings_llm_model_when_arg_unset():
    with patch("services.llm_adapter.settings.llm_provider", "openai"), patch(
        "services.llm_adapter.settings.llm_model", "gpt-4.1-mini"
    ):
        adapter = build_llm_adapter(model=None)

    assert adapter._model == "gpt-4.1-mini"
