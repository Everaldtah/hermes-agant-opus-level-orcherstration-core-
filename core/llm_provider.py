"""
LLM Provider Interface
======================

Pluggable LLM backend for Hermes Agent. Replaces the stub
processing logic with real or simulated provider connections.

Supports:
- Multiple provider backends (OpenAI, Anthropic, local, mock)
- Automatic failover between providers
- Token counting and budget enforcement
- Streaming responses
- Request/response logging
"""

import asyncio
import logging
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncIterator
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""
    MOCK = "mock"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    LOCAL = "local"
    GLM = "glm"
    KIMI = "kimi"
    KIMI_OPENROUTER = "kimi_openrouter"
    AUTO = "auto"  # Uses your full routing priority chain


@dataclass
class LLMRequest:
    """Request to an LLM provider."""
    messages: List[Dict[str, str]]
    model: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    system_prompt: str = ""
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    provider: str = ""
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, model: str = "", api_key: str = ""):
        self.name = name
        self.model = model
        self.api_key = api_key
        self._request_count = 0
        self._error_count = 0
        self._total_tokens = 0

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send completion request."""
        ...

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream completion response."""
        ...

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (words * 1.3)."""
        return int(len(text.split()) * 1.3) + 4

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "requests": self._request_count,
            "errors": self._error_count,
            "total_tokens": self._total_tokens,
        }


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing without an API key.

    Generates deterministic responses based on input classification.
    """

    def __init__(self, latency_ms: float = 10.0):
        super().__init__(name="mock", model="mock-v1")
        self._latency_ms = latency_ms

    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.time()
        await asyncio.sleep(self._latency_ms / 1000)

        last_msg = request.messages[-1]["content"] if request.messages else ""
        response_text = self._generate_mock_response(last_msg)

        self._request_count += 1
        input_tokens = sum(self.estimate_tokens(m["content"]) for m in request.messages)
        output_tokens = self.estimate_tokens(response_text)
        self._total_tokens += input_tokens + output_tokens

        return LLMResponse(
            content=response_text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.time() - start) * 1000,
            provider=self.name,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        response = await self.complete(request)
        for word in response.content.split():
            yield word + " "
            await asyncio.sleep(0.005)

    def _generate_mock_response(self, message: str) -> str:
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ("hello", "hi", "hey")):
            return "Hello! I'm the Hermes Agent. How can I help you today?"
        if any(kw in msg_lower for kw in ("code", "function", "write")):
            return (
                "Here's a Python implementation:\n\n"
                "```python\ndef solution(data):\n"
                "    # Process the input\n"
                "    result = []\n"
                "    for item in data:\n"
                "        result.append(transform(item))\n"
                "    return result\n```\n\n"
                "This handles the basic case. Let me know if you need modifications."
            )
        if any(kw in msg_lower for kw in ("error", "bug", "debug", "fix")):
            return (
                "I can see the issue. The error occurs because of an unhandled "
                "edge case. Here's the fix:\n\n"
                "1. Add input validation at the entry point\n"
                "2. Wrap the critical section in a try/except\n"
                "3. Add a fallback return value\n\n"
                "Would you like me to show the corrected code?"
            )
        if any(kw in msg_lower for kw in ("analyze", "compare", "data")):
            return (
                "Based on my analysis:\n\n"
                "- Key finding 1: The primary metric shows a 15% improvement\n"
                "- Key finding 2: There's a strong correlation between factors A and B\n"
                "- Recommendation: Focus resources on the highest-impact area\n\n"
                "Shall I dive deeper into any of these points?"
            )
        return f"I've processed your request: '{message[:80]}'. How would you like me to proceed?"


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter provider for accessing multiple LLM backends
    via a unified API. Requires OPENROUTER_API_KEY env var.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: str = "",
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        super().__init__(name="openrouter", model=model, api_key=key)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp is required for OpenRouterProvider: pip install aiohttp")

        messages = list(request.messages)
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(self.BASE_URL, json=payload, headers=headers) as resp:
                data = await resp.json()

        self._request_count += 1

        if "error" in data:
            self._error_count += 1
            raise RuntimeError(f"OpenRouter API error: {data['error']}")

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        content = choice.get("message", {}).get("content", "")

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self._total_tokens += input_tokens + output_tokens

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.time() - start) * 1000,
            provider=self.name,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        # Simplified: complete then yield
        response = await self.complete(request)
        for chunk in response.content.split():
            yield chunk + " "


class ProviderManager:
    """
    Manages multiple LLM providers with automatic failover.
    
    Registers providers in priority order. If the primary fails,
    falls back to the next available provider.
    """

    def __init__(self):
        self._providers: List[LLMProvider] = []
        self._primary_index = 0

    def add_provider(self, provider: LLMProvider):
        """Add a provider (first added = highest priority)."""
        self._providers.append(provider)
        logger.info(f"Registered LLM provider: {provider.name} ({provider.model})")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete with automatic failover."""
        if not self._providers:
            raise RuntimeError("No LLM providers registered")

        last_error = None
        # Try each provider in priority order
        for i, provider in enumerate(self._providers):
            try:
                response = await provider.complete(request)
                self._primary_index = i  # promote successful provider
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.name} failed: {e}, trying next...")

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    @property
    def primary(self) -> Optional[LLMProvider]:
        if self._providers:
            return self._providers[self._primary_index]
        return None

    def get_stats(self) -> List[Dict[str, Any]]:
        return [p.get_stats() for p in self._providers]


# ══════════════════════════════════════════════════════════════════════
# GLM (智谱) Provider
# ══════════════════════════════════════════════════════════════════════

class GLMProvider(LLMProvider):
    """
    Provider for Zhipu GLM models (GLM-4, GLM-5.1).

    API docs: https://open.bigmodel.cn/dev/api
    The GLM API is OpenAI-compatible but has quirks:
    - Different error codes (1261 = rate limit vs OpenAI 429)
    - Token counting uses a custom tokenizer (differs from tiktoken)
    - Supports tool_calls natively
    - Rate limits are per-model, not per-key

    Set GLM_API_KEY env var or pass api_key directly.
    Base URL default: https://open.bigmodel.cn/api/paas/v4
    """

    DEFAULT_BASE = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def __init__(
        self,
        model: str = "glm-4-plus",
        api_key: str = "",
        base_url: str = "",
    ):
        key = api_key or os.environ.get("GLM_API_KEY", "")
        super().__init__(name="glm", model=model, api_key=key)
        self._base_url = base_url or self.DEFAULT_BASE

    def estimate_tokens(self, text: str) -> int:
        """
        GLM tokenizer produces ~1.5 tokens per CJK char and ~1.3 per
        Latin word.  This rough estimate handles mixed text better than
        the naive word-split used elsewhere.
        """
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        latin_words = len(text.split()) - cjk_chars  # approximate
        return int(cjk_chars * 1.5 + max(0, latin_words) * 1.3) + 8

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ValueError(
                "GLM_API_KEY not set. Get one at https://open.bigmodel.cn"
            )

        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")

        messages = list(request.messages)
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._base_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                data = await resp.json()

        self._request_count += 1

        # GLM error format: {"error": {"code": "...", "message": "..."}}
        if "error" in data:
            self._error_count += 1
            err = data["error"]
            code = err.get("code", "unknown")
            msg = err.get("message", str(err))
            # GLM rate-limit code
            if code in ("1261", "1292"):
                raise ConnectionError(f"GLM rate limited ({code}): {msg}")
            raise RuntimeError(f"GLM API error ({code}): {msg}")

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        content = choice.get("message", {}).get("content", "")

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self._total_tokens += input_tokens + output_tokens

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.time() - start) * 1000,
            provider=self.name,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Streaming via GLM SSE endpoint."""
        # Simplified: non-streaming fallback
        response = await self.complete(request)
        for chunk in response.content.split():
            yield chunk + " "


# ══════════════════════════════════════════════════════════════════════
# Kimi (Moonshot) Provider
# ══════════════════════════════════════════════════════════════════════

class KimiProvider(LLMProvider):
    """
    Provider for Moonshot Kimi models (moonshot-v1, kimi-k2).

    API docs: https://platform.moonshot.cn/docs
    Kimi API is OpenAI-compatible with these specifics:
    - 128k+ context window on kimi-k2
    - Supports file/document ingestion via file_id references
    - Rate limits: 3 RPM on free tier, varies on paid
    - Partial response on timeout (returns what it has)
    - Error 429 = rate limit, 400 = context overflow

    Set KIMI_API_KEY env var or pass api_key directly.
    """

    DEFAULT_BASE = "https://api.moonshot.cn/v1/chat/completions"

    def __init__(
        self,
        model: str = "moonshot-v1-128k",
        api_key: str = "",
        base_url: str = "",
    ):
        key = api_key or os.environ.get("KIMI_API_KEY", "")
        super().__init__(name="kimi", model=model, api_key=key)
        self._base_url = base_url or self.DEFAULT_BASE

    def estimate_tokens(self, text: str) -> int:
        """Kimi uses a BPE tokenizer similar to GPT — CJK chars ~1.5 tokens each."""
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        latin_words = len(text.split()) - cjk_chars
        return int(cjk_chars * 1.5 + max(0, latin_words) * 1.3) + 8

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ValueError(
                "KIMI_API_KEY not set. Get one at https://platform.moonshot.cn"
            )

        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")

        messages = list(request.messages)
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._base_url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=180)  # Kimi can be slow on 128k
            ) as resp:
                data = await resp.json()

        self._request_count += 1

        if "error" in data:
            self._error_count += 1
            err = data["error"]
            err_type = err.get("type", "unknown")
            msg = err.get("message", str(err))
            if resp.status == 429:
                raise ConnectionError(f"Kimi rate limited: {msg}")
            if "context_length" in msg.lower() or resp.status == 400:
                raise ValueError(f"Kimi context overflow: {msg}")
            raise RuntimeError(f"Kimi API error ({err_type}): {msg}")

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        content = choice.get("message", {}).get("content", "")

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self._total_tokens += input_tokens + output_tokens

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.time() - start) * 1000,
            provider=self.name,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        response = await self.complete(request)
        for chunk in response.content.split():
            yield chunk + " "


# ══════════════════════════════════════════════════════════════════════
# LM Studio Local Provider
# ══════════════════════════════════════════════════════════════════════

class LMStudioProvider(LLMProvider):
    """
    Provider for LM Studio local inference server.

    LM Studio exposes an OpenAI-compatible endpoint at
    http://127.0.0.1:1234/v1/chat/completions by default.

    No API key needed. Works offline / air-gapped.
    Fails fast with ConnectionError if the server isn't running.
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://127.0.0.1:1234",
    ):
        # SECURITY FIX: Removed hardcoded credential
        # super().__init__(name="lmstudio", model=model, api_key="lm-studio")
        # TODO: Use environment variable instead
        self._base_url = base_url.rstrip("/") + "/v1/chat/completions"
        self._health_url = base_url.rstrip("/") + "/v1/models"

    def estimate_tokens(self, text: str) -> int:
        """Local models typically use llama-style BPE tokenizer."""
        return int(len(text.split()) * 1.3) + 4

    async def is_available(self) -> bool:
        """Quick health check — is LM Studio actually running?"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._health_url,
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")

        messages = list(request.messages)
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._base_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),  # local can be slow
                ) as resp:
                    data = await resp.json()
        except aiohttp.ClientConnectorError:
            self._error_count += 1
            raise ConnectionError(
                f"LM Studio not running at {self._base_url}. "
                "Start it or switch to another provider."
            )

        self._request_count += 1

        if "error" in data:
            self._error_count += 1
            raise RuntimeError(f"LM Studio error: {data['error']}")

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        content = choice.get("message", {}).get("content", "")

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self._total_tokens += input_tokens + output_tokens

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.time() - start) * 1000,
            provider=self.name,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        response = await self.complete(request)
        for chunk in response.content.split():
            yield chunk + " "


# ══════════════════════════════════════════════════════════════════════
# Anthropic Claude Provider
# ══════════════════════════════════════════════════════════════════════

class AnthropicProvider(LLMProvider):
    """
    Provider for Anthropic Claude models.

    Uses the Anthropic Messages API format (NOT OpenAI-compatible).
    Set ANTHROPIC_API_KEY env var or pass api_key directly.
    """

    DEFAULT_BASE = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        base_url: str = "",
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        super().__init__(name="anthropic", model=model, api_key=key)
        self._base_url = base_url or self.DEFAULT_BASE

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")

        # Anthropic uses a different message format:
        # system is a top-level field, not a message role
        messages = [
            m for m in request.messages if m["role"] != "system"
        ]

        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._base_url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()

        self._request_count += 1

        if "error" in data:
            self._error_count += 1
            err = data["error"]
            err_type = err.get("type", "unknown")
            msg = err.get("message", str(err))
            if err_type == "rate_limit_error":
                raise ConnectionError(f"Claude rate limited: {msg}")
            raise RuntimeError(f"Claude API error ({err_type}): {msg}")

        # Anthropic response format: {"content": [{"type":"text","text":"..."}], ...}
        content_blocks = data.get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        self._total_tokens += input_tokens + output_tokens

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.time() - start) * 1000,
            provider=self.name,
            finish_reason=data.get("stop_reason", "end_turn"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        response = await self.complete(request)
        for chunk in response.content.split():
            yield chunk + " "


# ══════════════════════════════════════════════════════════════════════
# Kimi K2.5 via OpenRouter
# ══════════════════════════════════════════════════════════════════════

class KimiOpenRouterProvider(LLMProvider):
    """
    Kimi K2.5 accessed via OpenRouter (your current active session model).

    This wraps OpenRouter specifically for Kimi models, using the
    moonshotai/kimi-k2 model identifier on OpenRouter.
    
    Set OPENROUTER_API_KEY env var or pass api_key directly.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str = "moonshotai/kimi-k2",
        api_key: str = "",
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        super().__init__(name="kimi-openrouter", model=model, api_key=key)

    def estimate_tokens(self, text: str) -> int:
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        latin_words = len(text.split()) - cjk_chars
        return int(cjk_chars * 1.5 + max(0, latin_words) * 1.3) + 8

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set for Kimi K2.5")

        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")

        messages = list(request.messages)
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hermes-agent.dev",
            "X-Title": "Hermes Agent",
        }

        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.BASE_URL, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                data = await resp.json()

        self._request_count += 1

        if "error" in data:
            self._error_count += 1
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"OpenRouter/Kimi error: {msg}")

        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        content = choice.get("message", {}).get("content", "")
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self._total_tokens += input_tokens + output_tokens

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.time() - start) * 1000,
            provider=self.name,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        response = await self.complete(request)
        for chunk in response.content.split():
            yield chunk + " "