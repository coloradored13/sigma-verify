"""Tests for sigma_verify.clients — NemotronClient via Ollama cloud.

Tests NVIDIA Nemotron-3-Super via Ollama cloud (OpenAI-compatible API).
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from sigma_verify.clients import NemotronClient, VerificationResult

# --- Helpers ---


def _chat_response(content: str, model: str = "nemotron-3-super:cloud") -> ChatCompletion:
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
        "reasoning": "Finding is well-supported",
        "confidence": "high",
        "counter_evidence": "",
    }
    data.update(overrides)
    return json.dumps(data)


def _valid_challenge_json(**overrides) -> str:
    data = {
        "counter_argument": "Sample size insufficient",
        "logical_gaps": "No control group",
        "evidence_needed": "Longitudinal data",
        "vulnerability": "medium",
    }
    data.update(overrides)
    return json.dumps(data)


# --- Availability & Presets ---


class TestNemotronAvailability:
    def test_available_ollama_default(self):
        """Ollama preset sets api_key to 'ollama' — available when Ollama is up."""
        assert NemotronClient().available is True

    def test_default_ollama_preset(self):
        client = NemotronClient()
        assert client.model == "nemotron-3-super:cloud"
        assert client._base_url == "http://localhost:11434/v1"
        assert client._api_key == "ollama"

    def test_custom_model(self):
        client = NemotronClient(model="custom/nemotron")
        assert client.model == "custom/nemotron"

    @patch.dict(os.environ, {"NEMOTRON_MODEL": "custom/model"})
    def test_model_from_env(self):
        client = NemotronClient()
        assert client.model == "custom/model"

    @patch.dict(os.environ, {"NEMOTRON_API_KEY": "override-key"})
    def test_api_key_override(self):
        client = NemotronClient()
        assert client._api_key == "override-key"

    @patch.dict(os.environ, {"NEMOTRON_BASE_URL": "https://custom.endpoint/v1"})
    def test_base_url_override(self):
        client = NemotronClient()
        assert client._base_url == "https://custom.endpoint/v1"

    @patch.dict(os.environ, {"NEMOTRON_PROVIDER": "unknown"})
    def test_unknown_provider_falls_back_to_ollama(self):
        client = NemotronClient()
        assert client._base_url == "http://localhost:11434/v1"


# --- API Calls ---


class TestNemotronCall:
    @patch("openai.OpenAI")
    def test_call_returns_text_and_tokens(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response("response")

        client = NemotronClient()
        text, tokens_in, tokens_out = client.call("system", "user content")

        assert text == "response"
        assert tokens_in == 100
        assert tokens_out == 50

    @patch("openai.OpenAI")
    def test_call_passes_params(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response("ok")

        client = NemotronClient()
        client.call("sys", "user", temperature=0.2, max_tokens=512)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 512

    @patch("openai.OpenAI")
    def test_call_handles_no_usage(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        resp = _chat_response("text")
        object.__setattr__(resp, "usage", None)
        mock_client.chat.completions.create.return_value = resp

        client = NemotronClient()
        text, tokens_in, tokens_out = client.call("sys", "user")
        assert text == "text"
        assert tokens_in == 0
        assert tokens_out == 0


# --- Verify ---


class TestNemotronVerify:
    @patch("openai.OpenAI")
    def test_verify_parses_valid_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(
            _valid_verify_json(assessment="partial", confidence="medium")
        )

        client = NemotronClient()
        result = client.verify("Market is $5B", "Tech analysis")

        assert isinstance(result, VerificationResult)
        assert result.assessment == "partial"
        assert result.confidence == "medium"
        assert result.provider == "nemotron"
        assert "nemotron" in result.provenance_tag

    @patch("openai.OpenAI")
    def test_verify_handles_markdown_fences(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        fenced = "```json\n" + _valid_verify_json(assessment="disagree") + "\n```"
        mock_client.chat.completions.create.return_value = _chat_response(fenced)

        client = NemotronClient()
        result = client.verify("test", "")
        assert result.assessment == "disagree"

    @patch("openai.OpenAI")
    def test_verify_handles_unparseable(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(
            "The finding looks correct."
        )

        client = NemotronClient()
        result = client.verify("test", "")
        assert result.assessment == "uncertain"
        assert result.confidence == "low"

    @patch("openai.OpenAI")
    def test_verify_passes_correct_params(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(_valid_verify_json())

        client = NemotronClient()
        client.verify("test finding", "test context")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert "test finding" in call_kwargs["messages"][1]["content"]


# --- Challenge ---


class TestNemotronChallenge:
    @patch("openai.OpenAI")
    def test_challenge_parses_valid_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(
            _valid_challenge_json(vulnerability="high")
        )

        client = NemotronClient()
        result = client.challenge("AI replaces 50% of jobs", "McKinsey 2024")

        assert result["vulnerability"] == "high"
        assert result["provider"] == "nemotron"
        assert result["tier"] == "standard"
        assert "nemotron" in result["provenance_tag"]


# --- Quota ---


class TestNemotronQuota:
    def test_quota_ollama_cloud(self):
        client = NemotronClient()
        quota = client.check_quota()
        assert quota["provider"] == "nemotron"
        assert quota["status"] == "ok"
        assert "free tier" in quota["message"]


# --- Errors ---


class TestNemotronErrors:
    @patch("openai.OpenAI")
    def test_verify_raises_on_rate_limit(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        class RateLimitError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = RateLimitError("rate limited")

        client = NemotronClient()
        with pytest.raises(RateLimitError):
            client.verify("test", "")
