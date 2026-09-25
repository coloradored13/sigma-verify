"""Tests for sigma_verify.clients — SDK-level mocks testing full client flow.

These tests mock the actual OpenAI/Gemini SDK objects to verify that our
client code correctly builds requests, parses response shapes, and handles
SDK-level errors with proper error classification.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types as genai_types
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from sigma_verify.clients import GeminiClient, OpenAIClient, VerificationResult

# --- Helpers ---


def _openai_response(content: str, model: str = "gpt-4o") -> ChatCompletion:
    """Build a realistic OpenAI ChatCompletion response."""
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
        usage=None,
    )


def _gemini_response(content: str) -> genai_types.GenerateContentResponse:
    """Build a realistic Gemini GenerateContentResponse."""
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(parts=[genai_types.Part(text=content)]),
                finish_reason="STOP",
            )
        ]
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


# --- OpenAI Client Tests ---


class TestOpenAIClientVerify:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_verify_parses_valid_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(_valid_verify_json())

        client = OpenAIClient()
        result = client.verify("Market is $5B", "Tech sector analysis")

        assert isinstance(result, VerificationResult)
        assert result.assessment == "agree"
        assert result.confidence == "high"
        assert result.provider == "openai"
        assert result.model == "gpt-5.4"
        assert "|source:external-openai-gpt-5.4|" in result.provenance_tag

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_verify_parses_disagree_with_counter(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            _valid_verify_json(
                assessment="disagree",
                confidence="medium",
                counter_evidence="Industry data shows $3.2B not $5B",
            )
        )

        client = OpenAIClient()
        result = client.verify("Market is $5B", "")

        assert result.assessment == "disagree"
        assert result.confidence == "medium"
        assert "3.2B" in result.counter_evidence

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_verify_handles_markdown_fenced_json(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        fenced = "```json\n" + _valid_verify_json(assessment="partial") + "\n```"
        mock_client.chat.completions.create.return_value = _openai_response(fenced)

        client = OpenAIClient()
        result = client.verify("test", "")
        assert result.assessment == "partial"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_verify_handles_unparseable_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            "I think the finding is correct but I can't format JSON right now."
        )

        client = OpenAIClient()
        result = client.verify("test", "")
        # Should fall back gracefully
        assert result.assessment == "uncertain"
        assert result.confidence == "low"
        assert "can't format JSON" in result.reasoning

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_verify_passes_correct_params(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(_valid_verify_json())

        client = OpenAIClient(model="gpt-4o-mini")
        client.verify("test finding", "test context")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.3
        assert len(call_kwargs["messages"]) == 2
        assert "test finding" in call_kwargs["messages"][1]["content"]
        assert "test context" in call_kwargs["messages"][1]["content"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_verify_raises_on_rate_limit(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        class RateLimitError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = RateLimitError("rate limited")

        client = OpenAIClient()
        with pytest.raises(RateLimitError):
            client.verify("test", "")


class TestOpenAIClientChallenge:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_challenge_parses_valid_response(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(_valid_challenge_json())

        client = OpenAIClient()
        result = client.challenge("AI will replace 50% of jobs", "McKinsey 2024")

        assert result["counter_argument"] == "Sample size insufficient for this claim"
        assert result["vulnerability"] == "medium"
        assert result["provider"] == "openai"
        assert "|source:external-openai-gpt-5.4|" in result["provenance_tag"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_challenge_passes_correct_params(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(_valid_challenge_json())

        client = OpenAIClient()
        client.challenge("test claim", "test evidence")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert "test claim" in call_kwargs["messages"][1]["content"]
        assert "test evidence" in call_kwargs["messages"][1]["content"]


# --- Gemini Client Tests ---


class TestGeminiClientVerify:
    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_verify_parses_valid_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _gemini_response(
            _valid_verify_json(assessment="partial", confidence="medium")
        )

        client = GeminiClient()
        result = client.verify("CAGR is 12-16%", "VDR market analysis")

        assert isinstance(result, VerificationResult)
        assert result.assessment == "partial"
        assert result.confidence == "medium"
        assert result.provider == "google"
        assert "|source:external-google-gemini-3.1-pro-preview|" in result.provenance_tag

    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_verify_handles_markdown_fenced_json(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        fenced = "```json\n" + _valid_verify_json(assessment="disagree") + "\n```"
        mock_client.models.generate_content.return_value = _gemini_response(fenced)

        client = GeminiClient()
        result = client.verify("test", "")
        assert result.assessment == "disagree"

    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_verify_handles_unparseable_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _gemini_response(
            "The finding seems reasonable overall."
        )

        client = GeminiClient()
        result = client.verify("test", "")
        assert result.assessment == "uncertain"
        assert result.confidence == "low"

    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_verify_passes_correct_params(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _gemini_response(_valid_verify_json())

        client = GeminiClient(model="gemini-2.5-pro")
        client.verify("test finding", "test context")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"
        assert "test finding" in call_kwargs["contents"]
        assert "test context" in call_kwargs["contents"]

    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_verify_raises_on_quota(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        class ResourceExhausted(Exception):
            pass

        mock_client.models.generate_content.side_effect = ResourceExhausted("quota")

        client = GeminiClient()
        with pytest.raises(ResourceExhausted):
            client.verify("test", "")


class TestGeminiClientChallenge:
    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_challenge_parses_valid_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _gemini_response(
            _valid_challenge_json(vulnerability="high")
        )

        client = GeminiClient()
        result = client.challenge("Consolidation is inevitable", "3 acquisitions in 2024")

        assert result["vulnerability"] == "high"
        assert result["provider"] == "google"
        assert "|source:external-google-gemini-3.1-pro-preview|" in result["provenance_tag"]


# --- Client Availability ---


class TestClientAvailability:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_openai_available_with_key(self):
        assert OpenAIClient().available is True

    @patch.dict(os.environ, {}, clear=True)
    def test_openai_unavailable_without_key(self):
        # Clear any existing key
        os.environ.pop("OPENAI_API_KEY", None)
        assert OpenAIClient().available is False

    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test"})
    def test_gemini_available_with_key(self):
        assert GeminiClient().available is True

    @patch.dict(os.environ, {}, clear=True)
    def test_gemini_unavailable_without_key(self):
        os.environ.pop("GOOGLE_AI_API_KEY", None)
        assert GeminiClient().available is False

    def test_openai_default_model(self):
        client = OpenAIClient()
        assert client.model == "gpt-5.4"

    def test_gemini_default_model(self):
        client = GeminiClient()
        assert client.model == "gemini-3.1-pro-preview"

    def test_custom_model(self):
        assert OpenAIClient(model="gpt-4o-mini").model == "gpt-4o-mini"
        assert GeminiClient(model="gemini-2.5-pro").model == "gemini-2.5-pro"


# --- call() method tests ---


def _openai_response_with_usage(
    content: str, prompt_tokens: int = 100, completion_tokens: int = 50
) -> ChatCompletion:
    """Build an OpenAI response with usage metadata."""
    from openai.types import CompletionUsage

    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="gpt-5.4",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class TestOpenAIClientCall:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_call_returns_text_and_tokens(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response_with_usage(
            '{"probability": 0.75}', prompt_tokens=200, completion_tokens=80
        )

        client = OpenAIClient()
        text, tokens_in, tokens_out = client.call("System prompt", "User content")

        assert text == '{"probability": 0.75}'
        assert tokens_in == 200
        assert tokens_out == 80

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_call_passes_temperature_and_max_tokens(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response_with_usage("ok")

        client = OpenAIClient()
        client.call("sys", "user", temperature=0.2, max_tokens=1024)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_completion_tokens"] == 1024
        assert call_kwargs["messages"][0]["content"] == "sys"
        assert call_kwargs["messages"][1]["content"] == "user"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_call_handles_no_usage(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response("text")

        client = OpenAIClient()
        text, tokens_in, tokens_out = client.call("sys", "user")

        assert text == "text"
        assert tokens_in == 0
        assert tokens_out == 0


class TestGeminiClientCall:
    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_call_returns_text_and_tokens(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        response = _gemini_response('{"base_rate": 0.3}')
        response.usage_metadata = MagicMock(prompt_token_count=150, candidates_token_count=60)
        mock_client.models.generate_content.return_value = response

        client = GeminiClient()
        text, tokens_in, tokens_out = client.call("System prompt", "User content")

        assert text == '{"base_rate": 0.3}'
        assert tokens_in == 150
        assert tokens_out == 60

    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_call_passes_temperature_and_max_tokens(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _gemini_response("ok")

        client = GeminiClient()
        client.call("sys", "user", temperature=0.3, max_tokens=512)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert "sys" in call_kwargs["contents"]
        assert "user" in call_kwargs["contents"]

    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_call_handles_no_usage_metadata(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _gemini_response("text")

        client = GeminiClient()
        text, tokens_in, tokens_out = client.call("sys", "user")

        assert text == "text"
        assert tokens_in == 0
        assert tokens_out == 0


# --- Reasoning Tier Tests ---


class TestOpenAIReasoningTier:
    def test_default_reasoning_model(self):
        client = OpenAIClient()
        assert client.reasoning_model == "gpt-5.4-pro"

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "sk-test", "OPENAI_REASONING_MODEL": "gpt-5.4-pro-custom"},
    )
    def test_reasoning_model_from_env(self):
        client = OpenAIClient()
        assert client.reasoning_model == "gpt-5.4-pro-custom"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_challenge_uses_reasoning_by_default(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.responses.create.return_value = MagicMock(
            output_text=_valid_challenge_json(vulnerability="low")
        )

        client = OpenAIClient()
        result = client.challenge("test claim", "test evidence")

        # Should use responses API, not chat completions
        mock_client.responses.create.assert_called_once()
        mock_client.chat.completions.create.assert_not_called()
        assert result["vulnerability"] == "low"
        assert result["tier"] == "reasoning"
        assert result["model"] == "gpt-5.4-pro"
        assert "gpt-5.4-pro" in result["provenance_tag"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_challenge_standard_tier_uses_chat(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(_valid_challenge_json())

        client = OpenAIClient()
        result = client.challenge("test claim", "test evidence", tier="standard")

        # Should use chat completions, not responses
        mock_client.chat.completions.create.assert_called_once()
        mock_client.responses.create.assert_not_called()
        assert result["tier"] == "standard"
        assert result["model"] == "gpt-5.4"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_challenge_falls_back_on_reasoning_failure(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # Reasoning fails
        mock_client.responses.create.side_effect = Exception("model unavailable")
        # Standard succeeds
        mock_client.chat.completions.create.return_value = _openai_response(
            _valid_challenge_json(vulnerability="high")
        )

        client = OpenAIClient()
        result = client.challenge("test claim", "test evidence")

        # Should fall back to standard
        assert result["tier"] == "standard"
        assert result["vulnerability"] == "high"
        assert result["model"] == "gpt-5.4"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"})
    @patch("openai.OpenAI")
    def test_reasoning_passes_correct_model(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.responses.create.return_value = MagicMock(output_text=_valid_challenge_json())

        client = OpenAIClient()
        client.challenge("claim", "evidence")

        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.4-pro"


class TestGeminiReasoningTier:
    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_challenge_uses_thinking_by_default(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _gemini_response(
            _valid_challenge_json(vulnerability="low")
        )

        client = GeminiClient()
        result = client.challenge("test claim", "test evidence")

        assert result["tier"] == "reasoning"
        # Verify thinking_config was passed
        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == GeminiClient.THINKING_BUDGET

    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_challenge_standard_tier_no_thinking(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = _gemini_response(_valid_challenge_json())

        client = GeminiClient()
        result = client.challenge("test claim", "test evidence", tier="standard")

        assert result["tier"] == "standard"
        # Verify no thinking_config
        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert config.thinking_config is None

    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "AI-test-key"})
    @patch("google.genai.Client")
    def test_challenge_falls_back_on_thinking_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        # First call (thinking) fails, second (standard) succeeds
        mock_client.models.generate_content.side_effect = [
            Exception("thinking not supported"),
            _gemini_response(_valid_challenge_json(vulnerability="high")),
        ]

        client = GeminiClient()
        result = client.challenge("test claim", "test evidence")

        assert result["tier"] == "standard"
        assert result["vulnerability"] == "high"
