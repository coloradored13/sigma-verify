"""Tests for _OllamaClientBase — shared behavior across all Ollama-backed providers.

Tests use a concrete test subclass to isolate base class logic.
Individual client tests only verify class-level configuration.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from sigma_verify.clients import VerificationResult, _OllamaClientBase

# --- Test subclass ---


class _TestClient(_OllamaClientBase):
    """Concrete subclass for testing base class behavior."""

    PROVIDER_NAME = "testprov"
    ENV_PREFIX = "TESTPROV"
    DEFAULT_MODEL = "test-model:latest"
    DESCRIPTION = "test provider, no cost"
    REASONING_NOTE = "test Ollama"
    PRESETS = {
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key_env": "",
            "model": "test-model:latest",
        },
        "alt": {
            "base_url": "https://alt.example.com/v1",
            "api_key_env": "ALT_API_KEY",
            "model": "test-model/alt",
        },
    }


# --- Helpers ---


def _chat_response(content: str, model: str = "test-model:latest") -> ChatCompletion:
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
        usage=CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


def _valid_verify_json(**overrides) -> str:
    data = {
        "assessment": "agree",
        "reasoning": "Well-supported finding",
        "confidence": "high",
        "counter_evidence": "",
    }
    data.update(overrides)
    return json.dumps(data)


def _valid_challenge_json(**overrides) -> str:
    data = {
        "counter_argument": "Insufficient sample",
        "logical_gaps": "No control group",
        "evidence_needed": "Longitudinal data",
        "vulnerability": "medium",
    }
    data.update(overrides)
    return json.dumps(data)


# --- Init & Availability ---


class TestOllamaBaseInit:
    def test_default_ollama_preset(self):
        client = _TestClient()
        assert client.model == "test-model:latest"
        assert client._base_url == "http://localhost:11434/v1"
        assert client._api_key == "ollama"

    def test_available_when_ollama_running(self):
        assert _TestClient().available is True

    def test_custom_model_param(self):
        client = _TestClient(model="custom:v2")
        assert client.model == "custom:v2"

    @patch.dict(os.environ, {"TESTPROV_MODEL": "env-model:latest"})
    def test_model_from_env(self):
        assert _TestClient().model == "env-model:latest"

    @patch.dict(os.environ, {"TESTPROV_API_KEY": "my-key"})
    def test_api_key_from_env(self):
        assert _TestClient()._api_key == "my-key"

    @patch.dict(os.environ, {"TESTPROV_BASE_URL": "https://custom.api/v1"})
    def test_base_url_from_env(self):
        assert _TestClient()._base_url == "https://custom.api/v1"

    @patch.dict(os.environ, {"TESTPROV_PROVIDER": "alt", "ALT_API_KEY": "alt-key"})
    def test_alternate_preset(self):
        client = _TestClient()
        assert client._base_url == "https://alt.example.com/v1"
        assert client.model == "test-model/alt"
        assert client._api_key == "alt-key"

    @patch.dict(os.environ, {"TESTPROV_PROVIDER": "unknown"})
    def test_unknown_provider_falls_back(self):
        client = _TestClient()
        assert client._base_url == "http://localhost:11434/v1"

    @patch.dict(os.environ, {"TESTPROV_PROVIDER": "alt"}, clear=True)
    def test_unavailable_without_key(self):
        os.environ.pop("ALT_API_KEY", None)
        os.environ.pop("TESTPROV_API_KEY", None)
        assert _TestClient().available is False


# --- Call ---


class TestOllamaBaseCall:
    @patch("openai.OpenAI")
    def test_call_uses_chat_completions(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response("hello")

        text, tok_in, tok_out = _TestClient().call("sys", "user")
        assert text == "hello"
        assert tok_in == 100
        assert tok_out == 50
        assert mock_cls.call_args[1]["base_url"] == "http://localhost:11434/v1"

    @patch("openai.OpenAI")
    def test_call_passes_params(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response("ok")

        _TestClient().call("sys", "user", temperature=0.2, max_tokens=512)
        kw = mock_client.chat.completions.create.call_args[1]
        assert kw["temperature"] == 0.2
        assert kw["max_tokens"] == 512

    @patch("openai.OpenAI")
    def test_call_handles_no_usage(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        resp = _chat_response("text")
        object.__setattr__(resp, "usage", None)
        mock_client.chat.completions.create.return_value = resp

        text, tok_in, tok_out = _TestClient().call("sys", "user")
        assert text == "text"
        assert tok_in == 0
        assert tok_out == 0


# --- Verify ---


class TestOllamaBaseVerify:
    @patch("openai.OpenAI")
    def test_verify_parses_valid_json(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(
            _valid_verify_json(assessment="partial", confidence="medium")
        )

        result = _TestClient().verify("Finding X", "Context Y")
        assert isinstance(result, VerificationResult)
        assert result.assessment == "partial"
        assert result.confidence == "medium"
        assert result.provider == "testprov"
        assert "|source:external-testprov-" in result.provenance_tag

    @patch("openai.OpenAI")
    def test_verify_handles_markdown_fences(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        fenced = "```json\n" + _valid_verify_json(assessment="disagree") + "\n```"
        mock_client.chat.completions.create.return_value = _chat_response(fenced)

        assert _TestClient().verify("test", "").assessment == "disagree"

    @patch("openai.OpenAI")
    def test_verify_handles_unparseable(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response("Looks correct.")

        result = _TestClient().verify("test", "")
        assert result.assessment == "uncertain"
        assert result.confidence == "low"

    @patch("openai.OpenAI")
    def test_verify_uses_low_temperature(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(_valid_verify_json())

        _TestClient().verify("finding", "context")
        assert mock_client.chat.completions.create.call_args[1]["temperature"] == 0.3


# --- Challenge ---


class TestOllamaBaseChallenge:
    @patch("openai.OpenAI")
    def test_challenge_parses_valid_json(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _chat_response(
            _valid_challenge_json(vulnerability="high")
        )

        result = _TestClient().challenge("claim", "evidence")
        assert result["vulnerability"] == "high"
        assert result["provider"] == "testprov"
        assert result["tier"] == "standard"
        assert "testprov" in result["provenance_tag"]


# --- Quota ---


class TestOllamaBaseQuota:
    def test_available_quota(self):
        quota = _TestClient().check_quota()
        assert quota["provider"] == "testprov"
        assert quota["status"] == "ok"
        assert "test provider, no cost" in quota["message"]

    @patch.dict(os.environ, {"TESTPROV_PROVIDER": "alt"}, clear=True)
    def test_unavailable_quota(self):
        os.environ.pop("ALT_API_KEY", None)
        os.environ.pop("TESTPROV_API_KEY", None)
        quota = _TestClient().check_quota()
        assert quota["status"] == "unavailable"


# --- Errors ---


class TestOllamaBaseErrors:
    @patch("openai.OpenAI")
    def test_verify_propagates_exceptions(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _TestClient().verify("test", "")

    @patch("openai.OpenAI")
    def test_challenge_propagates_exceptions(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _TestClient().challenge("claim", "evidence")
