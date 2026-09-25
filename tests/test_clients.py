"""Tests for sigma_verify.clients — response parsing and result formatting."""

from sigma_verify.clients import (
    VerificationError,
    VerificationResult,
    _build_challenge_prompt,
    _build_verify_prompt,
    _parse_json_response,
    classify_error,
)


class TestParseJsonResponse:
    def test_plain_json(self):
        result = _parse_json_response('{"assessment": "agree"}')
        assert result == {"assessment": "agree"}

    def test_json_in_markdown_fences(self):
        text = '```json\n{"assessment": "disagree"}\n```'
        result = _parse_json_response(text)
        assert result == {"assessment": "disagree"}

    def test_json_in_plain_fences(self):
        text = '```\n{"assessment": "partial"}\n```'
        result = _parse_json_response(text)
        assert result == {"assessment": "partial"}

    def test_invalid_json_returns_raw(self):
        result = _parse_json_response("not json at all")
        assert result["parse_error"] is True
        assert result["raw_response"] == "not json at all"

    def test_empty_string(self):
        result = _parse_json_response("")
        assert result["parse_error"] is True

    def test_json_embedded_in_prose(self):
        text = (
            "Here is my analysis:\n"
            '{"assessment": "agree", "reasoning": "The finding is correct", '
            '"confidence": "high", "counter_evidence": ""}\n'
            "That is my conclusion."
        )
        result = _parse_json_response(text)
        assert result["assessment"] == "agree"
        assert result["confidence"] == "high"

    def test_json_with_nested_fences_and_whitespace(self):
        text = (
            "```json\n\n"
            '{"assessment": "partial", "reasoning": "mostly right", '
            '"confidence": "medium", "counter_evidence": "minor issue"}\n\n'
            "```"
        )
        result = _parse_json_response(text)
        assert result["assessment"] == "partial"

    def test_multiple_fences_picks_first(self):
        text = (
            '```json\n{"assessment": "agree"}\n```\n\nSome more text\n```\n{"other": "data"}\n```'
        )
        result = _parse_json_response(text)
        assert result["assessment"] == "agree"


class TestVerificationResult:
    def test_to_dict(self):
        r = VerificationResult(
            model="gpt-4o",
            provider="openai",
            assessment="agree",
            reasoning="Finding is well-supported",
            confidence="high",
            counter_evidence="",
            provenance_tag="|source:external-openai-gpt-4o|",
        )
        d = r.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["provider"] == "openai"
        assert d["assessment"] == "agree"

    def test_to_sigma_without_counter(self):
        r = VerificationResult(
            model="gpt-4o",
            provider="openai",
            assessment="agree",
            reasoning="Solid analysis",
            confidence="high",
            counter_evidence="",
            provenance_tag="|source:external-openai-gpt-4o|",
        )
        s = r.to_sigma()
        assert "XVERIFY[openai:gpt-4o]" in s
        assert "agree(high)" in s
        assert "|counter:" not in s

    def test_to_sigma_with_counter(self):
        r = VerificationResult(
            model="gemini-2.0-flash",
            provider="google",
            assessment="disagree",
            reasoning="Market size overestimated",
            confidence="medium",
            counter_evidence="Source data is from 2019",
            provenance_tag="|source:external-google-gemini-2.0-flash|",
        )
        s = r.to_sigma()
        assert "XVERIFY[google:gemini-2.0-flash]" in s
        assert "disagree(medium)" in s
        assert "|counter:Source data is from 2019" in s


class TestVerificationError:
    def test_to_dict_has_failed_status(self):
        e = VerificationError(
            model="gpt-4o",
            provider="openai",
            error_class="rate-limit",
            error_detail="quota exceeded",
            tool_attempted="verify_finding",
            finding_brief="Market size is $5B",
            provenance_tag="|source:external-openai-gpt-4o|",
        )
        d = e.to_dict()
        assert d["status"] == "failed"
        assert d["error_class"] == "rate-limit"
        assert "XVERIFY-FAIL" in d["sigma"]

    def test_to_sigma_format(self):
        e = VerificationError(
            model="gemini-2.0-flash",
            provider="google",
            error_class="timeout",
            error_detail="Request timed out",
            tool_attempted="cross_verify",
            finding_brief="CAGR estimate 12-16%",
            provenance_tag="|source:external-google-gemini-2.0-flash|",
        )
        s = e.to_sigma()
        assert "XVERIFY-FAIL[google:gemini-2.0-flash]" in s
        assert "timeout" in s
        assert "cross_verify" in s
        assert "verification-gap" in s

    def test_finding_brief_truncated(self):
        e = VerificationError(
            model="gpt-4o",
            provider="openai",
            error_class="token-limit",
            error_detail="too many tokens",
            tool_attempted="verify_finding",
            finding_brief="x" * 200,
            provenance_tag="|source:external-openai-gpt-4o|",
        )
        s = e.to_sigma()
        # Finding should be truncated to 80 chars in sigma output
        assert len(s) < 300


class TestClassifyError:
    def test_auth_error_by_type_name(self):
        class AuthenticationError(Exception):
            pass

        assert classify_error(AuthenticationError("bad key")) == "auth-error"

    def test_rate_limit_by_type_name(self):
        class RateLimitError(Exception):
            pass

        assert classify_error(RateLimitError("quota")) == "rate-limit"

    def test_timeout_by_message(self):
        assert classify_error(Exception("Request timed out after 30s")) == "timeout"

    def test_token_limit_by_message(self):
        assert classify_error(Exception("too many tokens for context")) == "token-limit"

    def test_network_error_by_type_name(self):
        class ConnectionError(Exception):
            pass

        assert classify_error(ConnectionError("refused")) == "network-error"

    def test_unknown_error_fallback(self):
        assert classify_error(Exception("something weird")) == "unknown-error"

    def test_quota_in_message(self):
        assert classify_error(Exception("quota limit reached")) == "rate-limit"


class TestPromptBuilders:
    def test_verify_prompt_contains_finding(self):
        prompt = _build_verify_prompt("Market is $5B", "Tech sector analysis")
        assert "Market is $5B" in prompt
        assert "Tech sector analysis" in prompt
        assert "AGREE" in prompt or "agree" in prompt.lower()

    def test_challenge_prompt_contains_claim(self):
        prompt = _build_challenge_prompt("AI will replace 50% of jobs", "McKinsey report 2024")
        assert "AI will replace 50% of jobs" in prompt
        assert "McKinsey report 2024" in prompt
        assert "devil" in prompt.lower()
