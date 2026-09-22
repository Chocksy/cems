"""OpenAI-compatible LLM client for CEMS.

Defaults to OpenRouter as a unified gateway to any LLM provider (OpenAI,
Anthropic, Google, etc.), but any OpenAI-compatible endpoint works (Ollama,
vLLM, LiteLLM). OpenRouter-only extras (attribution headers, fast-provider
routing) are sent only when the endpoint is openrouter.ai.
"""

import logging
import os

from openai import OpenAI

from cems.config import CEMSConfig, is_openrouter_host

logger = logging.getLogger(__name__)

# Kept for backwards-compatible imports; the live value comes from CEMSConfig.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model mapping for OpenRouter (provider/model format)
OPENROUTER_MODELS = {
    # Default models - using Qwen3 for speed (via Cerebras/Groq when available)
    "default": "qwen/qwen3-32b",
    "fast": "qwen/qwen3-32b",  # ~2500 t/s on Cerebras
    "smart": "anthropic/claude-sonnet-4",
    # Explicit mappings from common model names
    "qwen3-32b": "qwen/qwen3-32b",
    "qwen3-8b": "qwen/qwen3-8b",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
    "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
    "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
}

# Fast inference providers (Cerebras ~3000 t/s, Groq ~900 t/s)
# Use these for speed-critical operations like query synthesis, reranking
FAST_PROVIDERS = ["cerebras", "groq", "sambanova"]


class OpenRouterClient:
    """OpenRouter LLM client for CEMS maintenance operations.

    This is the central LLM interface for all CEMS operations. Uses OpenRouter
    as a unified gateway to access any LLM provider.

    By default, routes requests through fast inference providers (Cerebras ~3000 t/s,
    Groq ~900 t/s) for speed. Falls back to other providers if unavailable.

    Example:
        client = OpenRouterClient()
        response = client.complete("Summarize these items: ...")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the LLM client.

        Args:
            api_key: API key. Defaults to CEMS_LLM_API_KEY, then OPENROUTER_API_KEY.
            model: Model name. Defaults to CEMS_LLM_MODEL or qwen/qwen3-32b.
            site_url: Attribution URL for the OpenRouter dashboard (OpenRouter only).
            site_name: Attribution name for the OpenRouter dashboard (OpenRouter only).
            base_url: OpenAI-compatible base URL. Defaults to CEMS_LLM_BASE_URL.
        """
        cfg = CEMSConfig()
        self.base_url = base_url or cfg.llm_base_url
        self.is_openrouter = is_openrouter_host(self.base_url)

        self.api_key = api_key or cfg.resolved_llm_api_key()
        if not self.api_key:
            raise ValueError(
                "LLM API key required. Set CEMS_LLM_API_KEY (or OPENROUTER_API_KEY "
                "when using OpenRouter), or pass api_key."
            )

        self.model = self._resolve_model(model or cfg.llm_model)
        self.site_url = site_url or os.getenv("CEMS_OPENROUTER_SITE_URL", "https://github.com/cems")
        self.site_name = site_name or os.getenv("CEMS_OPENROUTER_SITE_NAME", "CEMS Memory Server")

        client_kwargs: dict = {"base_url": self.base_url, "api_key": self.api_key}
        if self.is_openrouter:
            client_kwargs["default_headers"] = {
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            }
        self._client = OpenAI(**client_kwargs)

    def _resolve_model(self, model: str | None) -> str:
        """Resolve a model name to OpenRouter format.

        Args:
            model: Model name (can be standard or OpenRouter format)

        Returns:
            Model name in OpenRouter format (provider/model)
        """
        if model is None:
            return OPENROUTER_MODELS["default"]

        # Already in OpenRouter format
        if "/" in model:
            return model

        # Map known models
        if model in OPENROUTER_MODELS:
            return OPENROUTER_MODELS[model]

        # Assume it's already valid
        return model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,  # High default - let model use what it needs
        temperature: float = 0.3,
        model: str | None = None,
        fast_route: bool = True,
    ) -> str:
        """Generate a completion from the LLM.

        By default, routes through fast inference providers (Cerebras, Groq, SambaNova)
        for speed. Set fast_route=False for models not available on fast providers
        (e.g., Gemini, Claude).

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0-2)
            model: Override model for this call
            fast_route: Route through fast providers (default True). Disable for
                models only available from specific providers (e.g., Gemini).

        Returns:
            Generated text response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if fast_route and self.is_openrouter:
            # Route through fast providers (Cerebras ~3000 t/s, Groq ~900 t/s)
            kwargs["extra_body"] = {
                "provider": {
                    "order": FAST_PROVIDERS,
                    "allow_fallbacks": True,
                }
            }

        response = self._client.chat.completions.create(**kwargs)

        # Log details when response is empty (helps debug LLM issues)
        content = response.choices[0].message.content
        if not content:
            finish_reason = response.choices[0].finish_reason
            model_used = response.model if hasattr(response, 'model') else 'unknown'
            refusal = getattr(response.choices[0].message, 'refusal', None)
            logger.warning(
                f"[LLM] Empty response from {model_used}: "
                f"finish_reason={finish_reason}, refusal={refusal}, "
                f"content_was_none={content is None}"
            )

        return content or ""


# Module-level client (lazy initialization)
_client: OpenRouterClient | None = None


def get_client() -> OpenRouterClient:
    """Get the shared OpenRouter client instance.

    Returns:
        OpenRouterClient instance

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set
    """
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client
