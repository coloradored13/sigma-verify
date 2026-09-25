"""Tests for sigma_verify.clients — AnthropicClient SDK-level mocks.

Tests Claude models via the Anthropic SDK. Anthropic is opt-in only; see
test_anthropic_exclusion.py for the default-selection rules.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from sigma_verify.clients import AnthropicClient, VerificationResult

# --- Helpers ---


def _anthropic_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Build a mock Anthropic messages response."""
    mock = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    mock.content = [content_block]
    mock.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return mock


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


# --- Availability ---


class TestAnthropicAvailability:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    def test_available_with_key(self):
        assert AnthropicClient().available is True

    @patch.dict(os.environ, {}, clear=True)
    def test_unavailable_without_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        assert AnthropicClient().available is False

    def test_default_model(self):
        client = AnthropicClient()
        assert client.model == "claude-opus-5-5"

    def test_custom_model(self):
        client = AnthropicClient(model="claude-sonnet-4-6-20260320")
        assert client.model == "claude-sonnet-4-6-20260320"

    @patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-opus-5-5-custom"})
    def test_model_from_env(self):
        client = AnthropicClient()
        assert client.model == "claude-opus-5-5-custom"


# --- Call ---


class TestAnthropicCall:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_call_returns_text_and_tokens(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(
            "response text", input_tokens=200, output_tokens=80
        )

        client = AnthropicClient()
        text, tokens_in, tokens_out = client.call("system prompt", "user content")

        assert text == "response text"
        assert tokens_in == 200
        assert tokens_out == 80

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_call_passes_correct_params(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response("ok")

        client = AnthropicClient()
        client.call("sys", "user", temperature=0.2, max_tokens=512)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "sys"
        assert call_kwargs["messages"] == [{"role": "user", "content": "user"}]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 512

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_call_handles_empty_content(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = []
        mock_resp.usage = MagicMock(input_tokens=10, output_tokens=0)
        mock_client.messages.create.return_value = mock_resp

        client = AnthropicClient()
        text, tokens_in, tokens_out = client.call("sys", "user")
        assert text == ""
        assert tokens_in == 10


# --- Verify ---


class TestAnthropicVerify:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_verify_parses_valid_response(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(
            _valid_verify_json(assessment="partial", confidence="medium")
        )

        client = AnthropicClient()
        result = client.verify("Market is $5B", "Tech analysis")

        assert isinstance(result, VerificationResult)
        assert result.assessment == "partial"
        assert result.confidence == "medium"
        assert result.provider == "anthropic"
        assert "anthropic" in result.provenance_tag

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_verify_handles_markdown_fences(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        fenced = "```json\n" + _valid_verify_json(assessment="disagree") + "\n```"
        mock_client.messages.create.return_value = _anthropic_response(fenced)

        client = AnthropicClient()
        result = client.verify("test", "")
        assert result.assessment == "disagree"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_verify_handles_unparseable(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(
            "The finding is correct in my assessment."
        )

        client = AnthropicClient()
        result = client.verify("test", "")
        assert result.assessment == "uncertain"
        assert result.confidence == "low"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_verify_passes_correct_params(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(_valid_verify_json())

        client = AnthropicClient()
        client.verify("test finding", "test context")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert "test finding" in call_kwargs["messages"][0]["content"]


# --- Challenge ---


class TestAnthropicChallenge:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_challenge_parses_valid_response(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(
            _valid_challenge_json(vulnerability="high")
        )

        client = AnthropicClient()
        result = client.challenge("AI replaces 50% of jobs", "McKinsey 2024")

        assert result["vulnerability"] == "high"
        assert result["provider"] == "anthropic"
        assert result["tier"] == "standard"
        assert "anthropic" in result["provenance_tag"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_challenge_passes_correct_params(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(_valid_challenge_json())

        client = AnthropicClient()
        client.challenge("test claim", "test evidence")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert "test claim" in call_kwargs["messages"][0]["content"]


# --- Errors ---


class TestAnthropicErrors:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("anthropic.Anthropic")
    def test_verify_raises_on_rate_limit(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        class RateLimitError(Exception):
            pass

        mock_client.messages.create.side_effect = RateLimitError("rate limited")

        client = AnthropicClient()
        with pytest.raises(RateLimitError):
            client.verify("test", "")
