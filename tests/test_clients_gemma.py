"""Tests for sigma_verify.clients — GemmaClient SDK-level mocks.

Tests Gemma client via Ollama (default) and OpenRouter presets,
verifying correct API call patterns and response parsing.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from sigma_verify.clients import GemmaClient, VerificationResult

# --- Helpers ---


def _chat_response(content: str, model: str = "gemma4:e4b") -> ChatCompletion:
    """Build a realistic OpenAI-compatible chat completion response."""
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )


def _valid_verify_json(**overrides) -> str:
    data = {
        "assessment": "agree",
        "reasoning": "Finding is well-supported by available evidence",
        "confidence": "high",
        "counter_evidence": "",
    }
    data.update(overrides)
    return json.dumps(data)


def _valid_challenge_json(**overrides) -> str:
    data = {
        "counter_argument": "Sample size insufficient for this claim",
        "logical_gaps": "No control group in cited study",
        "evidence_needed": "Longitudinal data over 5+ years",
        "vulnerability": "medium",
    }
    data.update(overrides)
    return json.dumps(data)


# --- Availability ---


class TestGemmaClientAvailability:
    def test_available_ollama_default(self):
        """Ollama preset sets api_key to 'ollama' — available when Ollama is up."""
        client = GemmaClient()
        assert client.available is True

    @patch.dict(os.environ, {"GEMMA_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "test-key"})
    def test_available_openrouter_with_key(self):
        assert GemmaClient().available is True

    @patch.dict(os.environ, {"GEMMA_PROVIDER": "openrouter"}, clear=True)
    def test_unavailable_openrouter_without_key(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("GEMMA_API_KEY", None)
        assert GemmaClient().available is False

    def test_default_model(self):
        client = GemmaClient()
        assert client.model == "gemma4:e4b"

    def test_custom_model(self):
        client = GemmaClient(model="gemma3:12b")
        assert client.model == "gemma3:12b"

    @patch.dict(os.environ, {"GEMMA_MODEL": "gemma3:27b"})
    def test_model_from_env(self):
        client = GemmaClient()
        assert client.model == "gemma3:27b"

    @patch.dict(os.environ, {"GEMMA_BASE_URL": "https://custom.api.com/v1"})
    def test_base_url_from_env(self):
        client = GemmaClient()
        assert client._base_url == "https://custom.api.com/v1"

    def test_default_base_url_ollama(self):
        client = GemmaClient()
        assert client._base_url == "http://localhost:11434/v1"

    @patch.dict(os.environ, {"GEMMA_API_KEY": "override-key"})
    def test_api_key_override(self):
        client = GemmaClient()
        assert client._api_key == "override-key"


class TestGemmaOllamaPreset:
    def test_ollama_defaults(self):
        client = GemmaClient()
        assert client._base_url == "http://localhost:11434/v1"
        assert client.model == "gemma4:e4b"
        assert client._api_key == "ollama"

    def test_ollama_available_without_key(self):
        client = GemmaClient()
        assert client.available is True

    @patch.dict(os.environ, {"GEMMA_MODEL": "gemma3:12b"})
    def test_ollama_model_override(self):
        client = GemmaClient()
        assert client.model == "gemma3:12b"
        assert client._base_url == "http://localhost:11434/v1"

    @patch.dict(os.environ, {"GEMMA_PROVIDER": "unknown-provider"})
    def test_unknown_provider_falls_back_to_ollama(self):
        client = GemmaClient()
        assert client._base_url == "http://localhost:11434/v1"


class TestGemmaOpenRouterPreset:
    @patch.dict(os.environ, {"GEMMA_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "test-key"})
    def test_openrouter_defaults(self):
        client = GemmaClient()
        assert client._base_url == "https://openrouter.ai/api/v1"
        assert client.model == "google/gemma-4-e4b"
        assert client._api_key == "test-key"


# --- Call / Verify / Challenge ---


class TestGemmaCall:
    @patch("openai.OpenAI")
    def test_call_uses_chat_completions(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response("response text")

        client = GemmaClient()
        text, tokens_in, tokens_out = client.call("system", "user content")

        assert text == "response text"
        assert tokens_in == 100
        assert tokens_out == 50
        call_args = mock_openai_cls.call_args
        assert call_args[1]["base_url"] == "http://localhost:11434/v1"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    @patch("openai.OpenAI")
    def test_call_passes_params(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response("ok")

        client = GemmaClient()
        client.call("sys", "user", temperature=0.2, max_tokens=512)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 512


class TestGemmaVerify:
    @patch("openai.OpenAI")
    def test_verify_parses_valid_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(
            _valid_verify_json(assessment="partial", confidence="medium")
        )

        client = GemmaClient()
        result = client.verify("Market is $5B", "Tech sector analysis")

        assert isinstance(result, VerificationResult)
        assert result.assessment == "partial"
        assert result.confidence == "medium"
        assert result.provider == "gemma"
        assert "gemma" in result.provenance_tag

    @patch("openai.OpenAI")
    def test_verify_handles_markdown_fences(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        fenced = "```json\n" + _valid_verify_json(assessment="disagree") + "\n```"
        mock_client.chat.completions.create.return_value = _chat_response(fenced)

        client = GemmaClient()
        result = client.verify("test", "")
        assert result.assessment == "disagree"

    @patch("openai.OpenAI")
    def test_verify_handles_unparseable(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(
            "The finding seems correct overall."
        )

        client = GemmaClient()
        result = client.verify("test", "")
        assert result.assessment == "uncertain"
        assert result.confidence == "low"

    @patch("openai.OpenAI")
    def test_verify_passes_correct_params(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(_valid_verify_json())

        client = GemmaClient()
        client.verify("test finding", "test context")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert "test finding" in call_kwargs["messages"][1]["content"]


class TestGemmaChallenge:
    @patch("openai.OpenAI")
    def test_challenge_parses_valid_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(
            _valid_challenge_json(vulnerability="high")
        )

        client = GemmaClient()
        result = client.challenge("AI replaces 50% of jobs", "McKinsey 2024")

        assert result["vulnerability"] == "high"
        assert result["provider"] == "gemma"
        assert result["tier"] == "standard"
        assert "gemma" in result["provenance_tag"]


# --- Quota ---


class TestGemmaQuota:
    def test_quota_ollama(self):
        client = GemmaClient()
        quota = client.check_quota()
        assert quota["provider"] == "gemma"
        assert quota["status"] == "ok"
        assert "local model" in quota["message"]

    @patch.dict(os.environ, {"GEMMA_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "key"})
    def test_quota_openrouter(self):
        client = GemmaClient()
        quota = client.check_quota()
        assert quota["provider"] == "gemma"
        assert quota["status"] == "ok"
        assert "openrouter" in quota["message"]

    @patch.dict(os.environ, {"GEMMA_PROVIDER": "openrouter"}, clear=True)
    def test_quota_unavailable(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("GEMMA_API_KEY", None)
        client = GemmaClient()
        quota = client.check_quota()
        assert quota["status"] == "unavailable"


# --- Error Handling ---


class TestGemmaErrors:
    @patch("openai.OpenAI")
    def test_verify_raises_on_rate_limit(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        class RateLimitError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = RateLimitError("rate limited")

        client = GemmaClient()
        with pytest.raises(RateLimitError):
            client.verify("test", "")

    @patch("openai.OpenAI")
    def test_call_handles_no_usage(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        response = _chat_response("text")
        object.__setattr__(response, "usage", None)
        mock_client.chat.completions.create.return_value = response

        client = GemmaClient()
        text, tokens_in, tokens_out = client.call("sys", "user")

        assert text == "text"
        assert tokens_in == 0
        assert tokens_out == 0
