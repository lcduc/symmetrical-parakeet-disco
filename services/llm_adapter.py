import abc

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from config.settings import settings


class LLMAdapter(abc.ABC):
    """Provider-agnostic interface for LLM calls used by the extraction pipeline."""

    @abc.abstractmethod
    async def complete(self, prompt: str, max_tokens: int = 1024, temperature: float | None = None) -> str:
        """Return the model's plain-text response to `prompt`."""

    @abc.abstractmethod
    async def complete_structured(self, prompt: str) -> str:
        """Return the model's response to a prompt that asks for JSON output."""


class AnthropicAdapter(LLMAdapter):
    """LLMAdapter implementation backed by the Anthropic Messages API."""

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self._client = AsyncAnthropic(api_key=api_key or settings.llm_api_key)
        self._model = model

    async def complete(self, prompt: str, max_tokens: int = 1024, temperature: float | None = None) -> str:
        kwargs = {} if temperature is None else {"temperature": temperature}
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return next((block.text for block in response.content if block.type == "text"), "")

    async def complete_structured(self, prompt: str) -> str:
        return await self.complete(prompt, max_tokens=4096)


class OpenAIAdapter(LLMAdapter):
    """LLMAdapter implementation backed by the OpenAI Chat Completions API."""

    DEFAULT_MODEL = "gpt-4.1-mini"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self._client = AsyncOpenAI(api_key=api_key or settings.llm_api_key)
        self._model = model

    async def complete(self, prompt: str, max_tokens: int = 1024, temperature: float | None = None) -> str:
        kwargs = {} if temperature is None else {"temperature": temperature}
        response = await self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def complete_structured(self, prompt: str) -> str:
        return await self.complete(prompt, max_tokens=4096)


class GoogleAdapter(LLMAdapter):
    """LLMAdapter implementation backed by the Google Gemini API."""

    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self._client = genai.Client(api_key=api_key or settings.llm_api_key)
        self._model = model

    async def complete(self, prompt: str, max_tokens: int = 1024, temperature: float | None = None) -> str:
        config = genai_types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=temperature)
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    async def complete_structured(self, prompt: str) -> str:
        return await self.complete(prompt, max_tokens=4096)


_ADAPTERS_BY_PROVIDER: dict[str, type[LLMAdapter]] = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "google": GoogleAdapter,
}


def build_llm_adapter(model: str | None = None) -> LLMAdapter:
    """Instantiate the LLMAdapter configured by `settings.llm_provider`.

    Model resolution, most to least specific: the `model` argument (for a
    per-task override, e.g. `settings.extraction_llm_model`), then
    `settings.llm_model`, then the adapter's own built-in default.
    """
    adapter_cls = _ADAPTERS_BY_PROVIDER.get(settings.llm_provider)
    if adapter_cls is None:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r}")
    resolved_model = model or settings.llm_model
    return adapter_cls(model=resolved_model) if resolved_model else adapter_cls()
