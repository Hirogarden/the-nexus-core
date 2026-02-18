"""
The Nexus Core - LLM Adapter Layer

Provides a unified interface for all supported LLM providers.
Import the `llm` singleton to make inference calls anywhere in the system.

Usage:
    from nexus_core_llm_adapters import llm

    # Simple completion
    response = llm.complete("What is machine learning?")

    # With a system prompt
    response = llm.complete(
        "Explain neural networks",
        system_prompt="You are a concise technical expert."
    )

    # Get a processor callable for RecursiveLanguageModel
    processor = llm.as_processor(system_prompt="You are a helpful analyst.")

Supported providers (set LLM_PROVIDER in .env):
    none       - Stub responses, no API key required (default)
    openai     - Requires: pip install openai
    anthropic  - Requires: pip install anthropic
    ollama     - No extra packages needed, requires Ollama running locally
"""

import json
import urllib.request
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseLLMAdapter:
    """Shared interface for all LLM provider adapters."""

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a prompt and return the model's response as a string.

        Args:
            prompt:        The user-facing prompt or query.
            system_prompt: Optional instruction that shapes model behavior.
            context:       Optional dict passed through from the caller
                           (used for logging/tracing, not sent to the API).

        Returns:
            The model's text response.
        """
        raise NotImplementedError

    def as_processor(
        self, system_prompt: str = ""
    ) -> Callable[[str, Dict[str, Any]], str]:
        """
        Return a callable with signature (text, context) -> str,
        compatible with RecursiveLanguageModel's processor argument.

        Args:
            system_prompt: System instruction forwarded to every call.

        Returns:
            A closure that calls self.complete().
        """
        def processor(text: str, ctx: Dict[str, Any]) -> str:
            return self.complete(text, system_prompt=system_prompt, context=ctx)
        return processor

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Stub adapter (LLM_PROVIDER=none)
# ---------------------------------------------------------------------------

class StubAdapter(BaseLLMAdapter):
    """
    Zero-dependency placeholder that returns descriptive stub responses.
    Lets the whole system run end-to-end without any API keys.
    Set LLM_PROVIDER to a real provider in .env to get real responses.
    """

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        preview = prompt[:100].replace("\n", " ")
        ellipsis = "..." if len(prompt) > 100 else ""
        return (
            f"[Stub response - LLM_PROVIDER=none]\n"
            f"To enable real inference, set LLM_PROVIDER in your .env file.\n"
            f"Query received: {preview}{ellipsis}"
        )

    @property
    def provider_name(self) -> str:
        return "stub"


# ---------------------------------------------------------------------------
# OpenAI adapter (LLM_PROVIDER=openai)
# ---------------------------------------------------------------------------

class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI Chat Completions API adapter.
    Requires: pip install openai
    """

    def __init__(self, cfg) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The openai package is required for LLM_PROVIDER=openai.\n"
                "Install it with: pip install openai"
            )
        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.llm_model or "gpt-4o-mini"
        self._temperature = cfg.llm_temperature
        self._max_tokens = cfg.llm_max_tokens

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content

    @property
    def provider_name(self) -> str:
        return f"openai/{self._model}"


# ---------------------------------------------------------------------------
# Anthropic adapter (LLM_PROVIDER=anthropic)
# ---------------------------------------------------------------------------

class AnthropicAdapter(BaseLLMAdapter):
    """
    Anthropic Messages API adapter.
    Requires: pip install anthropic
    """

    def __init__(self, cfg) -> None:
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required for LLM_PROVIDER=anthropic.\n"
                "Install it with: pip install anthropic"
            )
        self._client = self._anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.llm_model or "claude-3-haiku-20240307"
        self._temperature = cfg.llm_temperature
        self._max_tokens = cfg.llm_max_tokens

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    @property
    def provider_name(self) -> str:
        return f"anthropic/{self._model}"


# ---------------------------------------------------------------------------
# Ollama adapter (LLM_PROVIDER=ollama)
# ---------------------------------------------------------------------------

class OllamaAdapter(BaseLLMAdapter):
    """
    Ollama local inference adapter.
    Zero extra dependencies - uses stdlib urllib only.
    Requires Ollama to be installed and running (https://ollama.com).

    Quick start:
        ollama pull llama3.2
        ollama serve          # starts on http://localhost:11434
    """

    def __init__(self, cfg) -> None:
        self._base_url = cfg.ollama_base_url.rstrip("/")
        self._model = cfg.llm_model or "llama3.2"
        self._temperature = cfg.llm_temperature
        self._max_tokens = cfg.llm_max_tokens

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read())
            return result.get("response", "")
        except OSError as exc:
            raise ConnectionError(
                f"Could not connect to Ollama at {self._base_url}.\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {exc}"
            ) from exc

    @property
    def provider_name(self) -> str:
        return f"ollama/{self._model}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_adapter(cfg=None) -> BaseLLMAdapter:
    """
    Return the adapter that matches the current configuration.

    Args:
        cfg: NexusConfig instance. Defaults to the module-level config singleton.

    Returns:
        The appropriate BaseLLMAdapter subclass instance.
    """
    if cfg is None:
        from nexus_core_config import config as cfg

    provider = cfg.llm_provider

    if provider == "openai":
        return OpenAIAdapter(cfg)
    elif provider == "anthropic":
        return AnthropicAdapter(cfg)
    elif provider == "ollama":
        return OllamaAdapter(cfg)
    else:
        return StubAdapter()


# ---------------------------------------------------------------------------
# Module-level singleton - import this in all other modules
# ---------------------------------------------------------------------------

llm: BaseLLMAdapter = get_adapter()
