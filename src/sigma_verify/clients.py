"""API clients for external model providers."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

# Error classification map — exception type patterns → error class
_ERROR_CLASSES = {
    "AuthenticationError": "auth-error",
    "PermissionDenied": "auth-error",
    "RateLimitError": "rate-limit",
    "ResourceExhausted": "rate-limit",
    "rate limit": "rate-limit",
    "quota": "rate-limit",
    "APITimeoutError": "timeout",
    "TimeoutException": "timeout",
    "Timeout": "timeout",
    "timed out": "timeout",
    "context_length": "token-limit",
    "max_tokens": "token-limit",
    "too many tokens": "token-limit",
    "ConnectionError": "network-error",
    "ConnectError": "network-error",
    "NetworkError": "network-error",
    "JSONDecodeError": "parse-error",
}


def classify_error(exc: Exception) -> str:
    """Classify an exception into a standard error class."""
    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()

    # Check exception type name first
    for pattern, error_class in _ERROR_CLASSES.items():
        if pattern in exc_type:
            return error_class

    # Check message content
    for pattern, error_class in _ERROR_CLASSES.items():
        if pattern.lower() in exc_msg:
            return error_class

    return "unknown-error"


@dataclass
class VerificationResult:
    """Structured result from an external model verification."""

    model: str
    provider: str
    assessment: str  # agree | disagree | partial | uncertain
    reasoning: str
    confidence: str  # high | medium | low
    counter_evidence: str  # empty if agrees
    provenance_tag: str  # |source:external-{provider}-{model}|
    status: str = "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "assessment": self.assessment,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "counter_evidence": self.counter_evidence,
            "provenance_tag": self.provenance_tag,
            "status": self.status,
        }

    def to_sigma(self) -> str:
        """Format as ΣComm-compatible string."""
        ce = f" |counter:{self.counter_evidence}" if self.counter_evidence else ""
        return (
            f"XVERIFY[{self.provider}:{self.model}]: "
            f"{self.assessment}({self.confidence}) "
            f"|{self.reasoning}"
            f"{ce} "
            f"{self.provenance_tag}"
        )


@dataclass
class VerificationError:
    """Structured failure result — verification was ATTEMPTED but FAILED."""

    model: str
    provider: str
    # auth-error | rate-limit | timeout | token-limit
    # | network-error | parse-error | unknown-error
    error_class: str
    error_detail: str
    tool_attempted: str  # verify_finding | cross_verify | challenge
    finding_brief: str  # truncated finding that was being verified
    provenance_tag: str  # |source:external-{provider}-{model}|

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "status": "failed",
            "error_class": self.error_class,
            "error_detail": self.error_detail,
            "tool_attempted": self.tool_attempted,
            "finding_brief": self.finding_brief,
            "provenance_tag": self.provenance_tag,
            "sigma": self.to_sigma(),
        }

    def to_sigma(self) -> str:
        """Format as ΣComm XVERIFY-FAIL string."""
        return (
            f"XVERIFY-FAIL[{self.provider}:{self.model}]: "
            f"{self.error_class} "
            f"|attempted:{self.tool_attempted} "
            f"|finding:{self.finding_brief[:80]} "
            f"|→ verification-gap"
        )


def _build_verify_prompt(finding: str, context: str) -> str:
    return (
        "You are an independent verification agent. Evaluate the following finding "
        "from an analyst team. Do NOT assume it is correct — assess it on its merits.\n\n"
        f"## Context\n{context}\n\n"
        f"## Finding to verify\n{finding}\n\n"
        "## Your task\n"
        "1. State whether you AGREE, DISAGREE, PARTIALLY AGREE, or are UNCERTAIN\n"
        "2. Provide your reasoning (2-4 sentences)\n"
        "3. Rate your confidence: HIGH, MEDIUM, or LOW\n"
        "4. If you disagree or partially agree, provide counter-evidence "
        "or an alternative interpretation\n\n"
        "Respond in JSON:\n"
        '{"assessment": "agree|disagree|partial|uncertain", '
        '"reasoning": "...", '
        '"confidence": "high|medium|low", '
        '"counter_evidence": "..."}'
    )


def _build_challenge_prompt(claim: str, evidence: str) -> str:
    return (
        "You are a devil's advocate. Your job is to find the strongest argument "
        "AGAINST the following claim, even if you ultimately agree with it.\n\n"
        f"## Claim\n{claim}\n\n"
        f"## Evidence presented for the claim\n{evidence}\n\n"
        "## Your task\n"
        "1. Identify the strongest counter-argument\n"
        "2. Find any logical gaps, unstated assumptions, or missing evidence\n"
        "3. Suggest what evidence would change the conclusion\n"
        "4. Rate how vulnerable the claim is: HIGH (likely wrong), "
        "MEDIUM (uncertain), LOW (probably right)\n\n"
        "Respond in JSON:\n"
        '{"counter_argument": "...", '
        '"logical_gaps": "...", '
        '"evidence_needed": "...", '
        '"vulnerability": "high|medium|low"}'
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from a model response, handling markdown fences and embedded JSON."""
    text = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if "```" in text:
        import re

        # Match content between ```json (or ```) and closing ```
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first { ... } block that looks like valid JSON
    # Handles cases where the model wraps JSON in prose
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    return {"raw_response": text, "parse_error": True}


# --- Ollama availability probe ---
#
# Ollama-backed providers need no API key, so key presence says nothing about
# whether they can actually be called. One cheap GET /api/tags per Ollama root
# URL (cached for the life of the process) tells us whether the server is up and
# which local models have been pulled. handle_init refreshes the cache.

OLLAMA_PROBE_TIMEOUT = 1.5  # seconds

# Opt-in: let init / server startup run `ollama serve` when Ollama is down.
AUTOSTART_OLLAMA_ENV = "SIGMA_VERIFY_AUTOSTART_OLLAMA"

_ollama_tags_cache: dict[str, list[str] | None] = {}

# Model descriptions shared by the Ollama-backed providers.
DESC_LOCAL = "local model, no API cost"
DESC_CLOUD_FREE = "Ollama cloud — free tier (requires `ollama signin`)"
DESC_CLOUD_PAID = "Ollama cloud — paid usage (requires `ollama signin` + credits)"


def ollama_root_url(base_url: str) -> str:
    """Strip the OpenAI-compatible /v1 suffix to get the Ollama server root."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def _fetch_ollama_tags(root_url: str) -> list[str] | None:
    """GET {root_url}/api/tags. Returns pulled model names, or None if unreachable."""
    try:
        req = urllib.request.Request(f"{root_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=OLLAMA_PROBE_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
    except Exception:
        return None
    return [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]


def probe_ollama(base_url: str, *, refresh: bool = False) -> list[str] | None:
    """Cached probe of the Ollama server behind base_url.

    Returns the list of locally available model names, or None when the server
    is not reachable. Tests monkeypatch _fetch_ollama_tags to stay offline.
    """
    root = ollama_root_url(base_url)
    if refresh or root not in _ollama_tags_cache:
        _ollama_tags_cache[root] = _fetch_ollama_tags(root)
    return _ollama_tags_cache[root]


def reset_ollama_probe() -> None:
    """Forget cached probe results (e.g. after starting Ollama)."""
    _ollama_tags_cache.clear()


def _model_pulled(model: str, tags: list[str]) -> bool:
    """True if model appears in the Ollama tags list (untagged names match :latest)."""
    if model in tags:
        return True
    return ":" not in model and f"{model}:latest" in tags


def is_cloud_model(model: str) -> bool:
    """Ollama cloud models carry a `cloud` tag, e.g. `glm-5.3:cloud`, `gpt-oss:120b-cloud`."""
    return ":" in model and model.rsplit(":", 1)[1].endswith("cloud")


class _OllamaClientBase:
    """Base class for all Ollama-backed providers (OpenAI-compatible API).

    Subclasses set class variables to configure identity and defaults:
        PROVIDER_NAME  — e.g. "llama", used in provenance tags and results
        ENV_PREFIX     — e.g. "LLAMA", drives env var names ({PREFIX}_MODEL, etc.)
        DEFAULT_MODEL  — e.g. "llama3.1:8b"
        DESCRIPTION    — quota message, one of DESC_LOCAL / DESC_CLOUD_FREE / DESC_CLOUD_PAID
        REASONING_NOTE — short note for init reporting, e.g. "local Ollama"
        PRESETS        — dict of provider presets (at minimum "ollama")
    """

    PROVIDER_NAME: str
    ENV_PREFIX: str
    DEFAULT_MODEL: str
    DESCRIPTION: str
    REASONING_NOTE: str = "Ollama"
    PRESETS: dict[str, dict[str, str]]
    DEFAULT_PROVIDER: str = "ollama"

    def __init__(self, model: str | None = None):
        provider = os.environ.get(f"{self.ENV_PREFIX}_PROVIDER", self.DEFAULT_PROVIDER)
        preset = self.PRESETS.get(provider, self.PRESETS[self.DEFAULT_PROVIDER])

        self.model = model or os.environ.get(f"{self.ENV_PREFIX}_MODEL", preset["model"])

        api_key_env = preset["api_key_env"]
        self._api_key = os.environ.get(
            f"{self.ENV_PREFIX}_API_KEY",
            os.environ.get(api_key_env, "") if api_key_env else "ollama",
        )
        self._base_url = os.environ.get(f"{self.ENV_PREFIX}_BASE_URL", preset["base_url"])
        # Presets without an API key env var talk to an Ollama server.
        self._uses_ollama = not api_key_env

    @property
    def is_cloud(self) -> bool:
        return self._uses_ollama and is_cloud_model(self.model)

    @property
    def unavailable_reason(self) -> str | None:
        """Why this provider can't be called right now, or None if it can."""
        if not self._api_key:
            return f"No API key configured for {self.PROVIDER_NAME}."
        if not self._uses_ollama:
            return None
        tags = probe_ollama(self._base_url)
        if tags is None:
            return (
                f"Ollama not running at {ollama_root_url(self._base_url)} "
                f"(start it with `ollama serve`, or set {AUTOSTART_OLLAMA_ENV}=1)."
            )
        if not self.is_cloud and not _model_pulled(self.model, tags):
            return f"Model {self.model} is not pulled — run `ollama pull {self.model}`."
        return None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def _get_client(self):
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    @staticmethod
    def _extract_content(message: Any) -> str:
        """Extract text from a chat completion message.

        Some Ollama thinking models (e.g. DeepSeek, GLM, Kimi) put their
        output in a 'reasoning' field instead of 'content'. Fall back
        to reasoning when content is empty.
        """
        content = message.content or ""
        if content:
            return content
        # Ollama thinking models: content is empty, output in reasoning
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            return reasoning
        # Last resort: check raw dict representation
        raw = getattr(message, "__dict__", {})
        return raw.get("reasoning", "") or raw.get("thinking", "") or ""

    def call(
        self,
        system: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int, int]:
        """General-purpose LLM call. Returns (text, tokens_in, tokens_out)."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (
            self._extract_content(response.choices[0].message),
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )

    def verify(self, finding: str, context: str) -> VerificationResult:
        client = self._get_client()
        prompt = _build_verify_prompt(finding, context)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise analytical verification agent. "
                        "Always respond in valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        parsed = _parse_json_response(self._extract_content(response.choices[0].message))
        return VerificationResult(
            model=self.model,
            provider=self.PROVIDER_NAME,
            assessment=parsed.get("assessment", "uncertain"),
            reasoning=parsed.get("reasoning", parsed.get("raw_response", "")),
            confidence=parsed.get("confidence", "low"),
            counter_evidence=parsed.get("counter_evidence", ""),
            provenance_tag=f"|source:external-{self.PROVIDER_NAME}-{self.model}|",
        )

    def challenge(self, claim: str, evidence: str, *, tier: str = "standard") -> dict[str, Any]:
        client = self._get_client()
        prompt = _build_challenge_prompt(claim, evidence)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise devil's advocate. Always respond in valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=2000,
        )
        parsed = _parse_json_response(self._extract_content(response.choices[0].message))
        parsed["model"] = self.model
        parsed["provider"] = self.PROVIDER_NAME
        parsed["tier"] = "standard"
        parsed["provenance_tag"] = f"|source:external-{self.PROVIDER_NAME}-{self.model}|"
        return parsed

    def check_quota(self) -> dict[str, str]:
        """Check remaining quota/budget for this provider."""
        reason = self.unavailable_reason
        if reason:
            return {
                "provider": self.PROVIDER_NAME,
                "status": "unavailable",
                "message": reason,
            }
        return {
            "provider": self.PROVIDER_NAME,
            "status": "ok",
            "message": (
                f"{self.PROVIDER_NAME.title()} via Ollama ({self.model}): {self.DESCRIPTION}."
            ),
        }


class OpenAIClient:
    """Wrapper for OpenAI API calls.

    Supports two tiers:
    - standard: chat completions API (gpt-5.4) — fast, cheap
    - reasoning: responses API (gpt-5.4-pro) — deep analytical thinking
    """

    STANDARD_MODEL = "gpt-5.4"
    REASONING_MODEL = "gpt-5.4-pro"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OPENAI_MODEL", self.STANDARD_MODEL)
        self.reasoning_model = os.environ.get("OPENAI_REASONING_MODEL", self.REASONING_MODEL)
        self._api_key = os.environ.get("OPENAI_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def call(
        self,
        system: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int, int]:
        """General-purpose LLM call. Returns (text, tokens_in, tokens_out)."""
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return (
            response.choices[0].message.content or "",
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )

    def _call_reasoning(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call a reasoning model via the responses API."""
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        response = client.responses.create(
            model=self.reasoning_model,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        return response.output_text or ""

    def verify(self, finding: str, context: str) -> VerificationResult:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise analytical verification agent. "
                        "Always respond in valid JSON."
                    ),
                },
                {"role": "user", "content": _build_verify_prompt(finding, context)},
            ],
            temperature=0.3,
            max_completion_tokens=1000,
        )
        parsed = _parse_json_response(response.choices[0].message.content or "")
        return VerificationResult(
            model=self.model,
            provider="openai",
            assessment=parsed.get("assessment", "uncertain"),
            reasoning=parsed.get("reasoning", parsed.get("raw_response", "")),
            confidence=parsed.get("confidence", "low"),
            counter_evidence=parsed.get("counter_evidence", ""),
            provenance_tag=f"|source:external-openai-{self.model}|",
        )

    def challenge(self, claim: str, evidence: str, *, tier: str = "reasoning") -> dict[str, Any]:
        prompt = _build_challenge_prompt(claim, evidence)

        if tier == "reasoning":
            try:
                full_prompt = (
                    "You are a precise devil's advocate. Always respond in valid JSON.\n\n" + prompt
                )
                raw = self._call_reasoning(full_prompt)
                parsed = _parse_json_response(raw)
                parsed["model"] = self.reasoning_model
                parsed["provider"] = "openai"
                parsed["tier"] = "reasoning"
                parsed["provenance_tag"] = f"|source:external-openai-{self.reasoning_model}|"
                return parsed
            except Exception:
                # Fall back to standard if reasoning model fails
                pass

        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise devil's advocate. Always respond in valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_completion_tokens=1000,
        )
        parsed = _parse_json_response(response.choices[0].message.content or "")
        parsed["model"] = self.model
        parsed["provider"] = "openai"
        parsed["tier"] = "standard"
        parsed["provenance_tag"] = f"|source:external-openai-{self.model}|"
        return parsed

    def check_quota(self) -> dict[str, str]:
        """Check remaining quota/budget for this provider."""
        if not self.available:
            return {
                "provider": "openai",
                "status": "unavailable",
                "message": "No OPENAI_API_KEY configured.",
            }
        return {
            "provider": "openai",
            "status": "ok",
            "message": (
                f"OpenAI ({self.model}): ~$0.002-0.01 per verification call (~1K tokens). "
                f"Reasoning model ({self.reasoning_model}): ~$0.01-0.05 per challenge call. "
                "Quota is credit-based — check platform.openai.com for remaining balance."
            ),
        }


class GeminiClient:
    """Wrapper for Google Gemini API calls.

    Supports two tiers:
    - standard: normal generation — fast, cheap
    - reasoning: thinking mode with thinking_budget — deep analytical thinking
    """

    DEFAULT_MODEL = "gemini-3.1-pro-preview"
    THINKING_BUDGET = 8192

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("GOOGLE_AI_MODEL", self.DEFAULT_MODEL)
        self._api_key = os.environ.get("GOOGLE_AI_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def call(
        self,
        system: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int, int]:
        """General-purpose LLM call. Returns (text, tokens_in, tokens_out)."""
        from google import genai

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=f"{system}\n\n{user_content}",
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", 0) if usage else 0
        tokens_out = getattr(usage, "candidates_token_count", 0) if usage else 0
        return (response.text or "", tokens_in, tokens_out)

    def _call_with_thinking(self, contents: str, max_tokens: int = 2000) -> str:
        """Call with thinking mode enabled for deeper reasoning."""
        from google import genai

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=self.THINKING_BUDGET,
                ),
            ),
        )
        return response.text or ""

    def verify(self, finding: str, context: str) -> VerificationResult:
        from google import genai

        client = genai.Client(api_key=self._api_key)
        prompt = _build_verify_prompt(finding, context)
        response = client.models.generate_content(
            model=self.model,
            contents=(
                "You are a precise analytical verification agent. "
                f"Always respond in valid JSON.\n\n{prompt}"
            ),
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2000,
            ),
        )
        parsed = _parse_json_response(response.text or "")
        return VerificationResult(
            model=self.model,
            provider="google",
            assessment=parsed.get("assessment", "uncertain"),
            reasoning=parsed.get("reasoning", parsed.get("raw_response", "")),
            confidence=parsed.get("confidence", "low"),
            counter_evidence=parsed.get("counter_evidence", ""),
            provenance_tag=f"|source:external-google-{self.model}|",
        )

    def challenge(self, claim: str, evidence: str, *, tier: str = "reasoning") -> dict[str, Any]:
        prompt = _build_challenge_prompt(claim, evidence)
        full_prompt = (
            "You are a precise devil's advocate. Always respond in valid JSON.\n\n" + prompt
        )

        if tier == "reasoning":
            try:
                raw = self._call_with_thinking(full_prompt)
                parsed = _parse_json_response(raw)
                parsed["model"] = self.model
                parsed["provider"] = "google"
                parsed["tier"] = "reasoning"
                parsed["provenance_tag"] = f"|source:external-google-{self.model}|"
                return parsed
            except Exception:
                # Fall back to standard if thinking mode fails
                pass

        from google import genai

        client = genai.Client(api_key=self._api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=2000,
            ),
        )
        parsed = _parse_json_response(response.text or "")
        parsed["model"] = self.model
        parsed["provider"] = "google"
        parsed["tier"] = "standard"
        parsed["provenance_tag"] = f"|source:external-google-{self.model}|"
        return parsed

    def check_quota(self) -> dict[str, str]:
        """Check remaining quota/budget for this provider."""
        if not self.available:
            return {
                "provider": "google",
                "status": "unavailable",
                "message": "No GOOGLE_AI_API_KEY configured.",
            }
        return {
            "provider": "google",
            "status": "warning",
            "message": (
                f"Gemini ({self.model}): daily per-model request limits apply "
                "(they vary by tier). Remaining quota can't be queried via the API — "
                "track usage manually. Thinking mode calls count against the same limit."
            ),
        }


_OLLAMA_URL = "http://localhost:11434/v1"
_OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _ollama_preset(model: str) -> dict[str, str]:
    return {"base_url": _OLLAMA_URL, "api_key_env": "", "model": model}


def _openrouter_preset(model: str) -> dict[str, str]:
    return {"base_url": _OPENROUTER_URL, "api_key_env": "OPENROUTER_API_KEY", "model": model}


class LlamaClient(_OllamaClientBase):
    """Meta Llama models via local Ollama or OpenRouter."""

    PROVIDER_NAME = "llama"
    ENV_PREFIX = "LLAMA"
    DEFAULT_MODEL = "llama3.1:8b"
    DESCRIPTION = DESC_LOCAL
    REASONING_NOTE = "local Ollama"
    PRESETS = {
        "ollama": _ollama_preset("llama3.1:8b"),
        "openrouter": _openrouter_preset("meta-llama/llama-4-maverick"),
    }


class GemmaClient(_OllamaClientBase):
    """Google Gemma models via local Ollama or OpenRouter."""

    PROVIDER_NAME = "gemma"
    ENV_PREFIX = "GEMMA"
    DEFAULT_MODEL = "gemma4:e4b"
    DESCRIPTION = DESC_LOCAL
    REASONING_NOTE = "local Ollama"
    PRESETS = {
        "ollama": _ollama_preset("gemma4:e4b"),
        "openrouter": _openrouter_preset("google/gemma-4-e4b"),
    }

    def check_quota(self) -> dict[str, str]:
        """Override: Gemma has provider-specific quota messages."""
        reason = self.unavailable_reason
        if reason:
            return {"provider": "gemma", "status": "unavailable", "message": reason}
        provider = os.environ.get("GEMMA_PROVIDER", self.DEFAULT_PROVIDER)
        if provider == "ollama":
            return {
                "provider": "gemma",
                "status": "ok",
                "message": f"Gemma via Ollama ({self.model}): {self.DESCRIPTION}.",
            }
        return {
            "provider": "gemma",
            "status": "ok",
            "message": (
                f"Gemma via {provider} ({self.model}): credit-based pricing. "
                "Check provider dashboard for remaining balance."
            ),
        }


class GptOssClient(_OllamaClientBase):
    """OpenAI gpt-oss open-weight MoE via Ollama cloud."""

    PROVIDER_NAME = "gpt-oss"
    ENV_PREFIX = "GPT_OSS"
    DEFAULT_MODEL = "gpt-oss:120b-cloud"
    DESCRIPTION = DESC_CLOUD_FREE
    REASONING_NOTE = "Ollama cloud, OpenAI open-weight"
    PRESETS = {
        "ollama": _ollama_preset("gpt-oss:120b-cloud"),
        "openrouter": _openrouter_preset("openai/gpt-oss-120b"),
    }


class NemotronClient(_OllamaClientBase):
    """NVIDIA Nemotron — Mamba-Transformer MoE hybrid via Ollama cloud."""

    PROVIDER_NAME = "nemotron"
    ENV_PREFIX = "NEMOTRON"
    DEFAULT_MODEL = "nemotron-3-super:cloud"
    DESCRIPTION = DESC_CLOUD_FREE
    REASONING_NOTE = "Ollama cloud, Mamba-Transformer hybrid"
    PRESETS = {
        "ollama": _ollama_preset("nemotron-3-super:cloud"),
        "openrouter": _openrouter_preset("nvidia/nemotron-3-super-120b-a12b"),
    }


class DeepSeekClient(_OllamaClientBase):
    """DeepSeek models — Chinese-lineage MoE via Ollama cloud."""

    PROVIDER_NAME = "deepseek"
    ENV_PREFIX = "DEEPSEEK"
    DEFAULT_MODEL = "deepseek-v4.1-flash:cloud"
    DESCRIPTION = DESC_CLOUD_PAID
    REASONING_NOTE = "Ollama cloud, DeepSeek AI"
    PRESETS = {"ollama": _ollama_preset("deepseek-v4.1-flash:cloud")}


class MistralClient(_OllamaClientBase):
    """Mistral Large — European-lineage MoE via Ollama cloud."""

    PROVIDER_NAME = "mistral"
    ENV_PREFIX = "MISTRAL"
    DEFAULT_MODEL = "mistral-large-3:675b-cloud"
    DESCRIPTION = DESC_CLOUD_PAID
    REASONING_NOTE = "Ollama cloud, Mistral AI"
    PRESETS = {"ollama": _ollama_preset("mistral-large-3:675b-cloud")}


class GlmClient(_OllamaClientBase):
    """Zhipu GLM models — Chinese-lineage reasoning MoE via Ollama cloud."""

    PROVIDER_NAME = "glm"
    ENV_PREFIX = "GLM"
    DEFAULT_MODEL = "glm-5.3:cloud"
    DESCRIPTION = DESC_CLOUD_PAID
    REASONING_NOTE = "Ollama cloud, Zhipu AI"
    PRESETS = {"ollama": _ollama_preset("glm-5.3:cloud")}


class KimiClient(_OllamaClientBase):
    """Moonshot Kimi models — agentic MoE via Ollama cloud."""

    PROVIDER_NAME = "kimi"
    ENV_PREFIX = "KIMI"
    DEFAULT_MODEL = "kimi-k3:cloud"
    DESCRIPTION = DESC_CLOUD_PAID
    REASONING_NOTE = "Ollama cloud, Moonshot AI"
    PRESETS = {"ollama": _ollama_preset("kimi-k3:cloud")}


class NemotronNanoClient(_OllamaClientBase):
    """NVIDIA Nemotron-3-Nano 4B — small Mamba-Transformer hybrid, local Ollama."""

    PROVIDER_NAME = "nemotron-nano"
    ENV_PREFIX = "NEMOTRON_NANO"
    DEFAULT_MODEL = "nemotron-3-nano:4b"
    DESCRIPTION = DESC_LOCAL
    REASONING_NOTE = "local Ollama, Mamba-Transformer hybrid"
    PRESETS = {"ollama": _ollama_preset("nemotron-3-nano:4b")}


class QwenLocalClient(_OllamaClientBase):
    """Alibaba Qwen 3.5 4B — small multilingual Transformer, local Ollama."""

    PROVIDER_NAME = "qwen-local"
    ENV_PREFIX = "QWEN_LOCAL"
    DEFAULT_MODEL = "qwen3.5:4b"
    DESCRIPTION = DESC_LOCAL
    REASONING_NOTE = "local Ollama, Alibaba DAMO"
    PRESETS = {"ollama": _ollama_preset("qwen3.5:4b")}


class AnthropicClient:
    """Wrapper for Anthropic Claude models.

    Uses the Anthropic SDK directly (not OpenAI-compatible). Excluded from every
    default provider selection: the callers of this server are Claude agents, and
    Claude verifying Claude is not cross-model verification. It is used only when
    named explicitly or when SIGMA_VERIFY_ALLOW_ANTHROPIC=1.
    """

    DEFAULT_MODEL = "claude-opus-5-5"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", self.DEFAULT_MODEL)
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def call(
        self,
        system: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, int, int]:
        """General-purpose LLM call. Returns (text, tokens_in, tokens_out)."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            temperature=temperature,
        )
        return (
            response.content[0].text if response.content else "",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

    def verify(self, finding: str, context: str) -> VerificationResult:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        prompt = _build_verify_prompt(finding, context)
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            system="You are a precise analytical verification agent. Always respond in valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = response.content[0].text if response.content else ""
        parsed = _parse_json_response(raw)
        return VerificationResult(
            model=self.model,
            provider="anthropic",
            assessment=parsed.get("assessment", "uncertain"),
            reasoning=parsed.get("reasoning", parsed.get("raw_response", "")),
            confidence=parsed.get("confidence", "low"),
            counter_evidence=parsed.get("counter_evidence", ""),
            provenance_tag=f"|source:external-anthropic-{self.model}|",
        )

    def challenge(self, claim: str, evidence: str, *, tier: str = "standard") -> dict[str, Any]:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        prompt = _build_challenge_prompt(claim, evidence)
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            system="You are a precise devil's advocate. Always respond in valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        raw = response.content[0].text if response.content else ""
        parsed = _parse_json_response(raw)
        parsed["model"] = self.model
        parsed["provider"] = "anthropic"
        parsed["tier"] = "standard"
        parsed["provenance_tag"] = f"|source:external-anthropic-{self.model}|"
        return parsed

    def check_quota(self) -> dict[str, str]:
        """Check remaining quota/budget for this provider."""
        if not self.available:
            return {
                "provider": "anthropic",
                "status": "unavailable",
                "message": "No ANTHROPIC_API_KEY configured.",
            }
        return {
            "provider": "anthropic",
            "status": "ok",
            "message": (
                f"Anthropic ({self.model}): per-token pricing; rate limits depend on "
                "your account tier. Excluded from default selections unless named "
                "explicitly or SIGMA_VERIFY_ALLOW_ANTHROPIC=1. "
                "Check console.anthropic.com for remaining credits."
            ),
        }


# All Ollama-backed client classes — for programmatic iteration in handlers
OLLAMA_CLIENTS: list[type[_OllamaClientBase]] = [
    LlamaClient,
    GemmaClient,
    NemotronNanoClient,
    QwenLocalClient,
    GptOssClient,
    NemotronClient,
    DeepSeekClient,
    MistralClient,
    GlmClient,
    KimiClient,
]

# Registry of all providers (Ollama + non-Ollama)
PROVIDERS: dict[str, type] = {
    "openai": OpenAIClient,
    "google": GeminiClient,
    **{cls.PROVIDER_NAME: cls for cls in OLLAMA_CLIENTS},
    "anthropic": AnthropicClient,
}
