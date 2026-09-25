"""Tests for sigma_verify.handlers — tool handler logic with mocked API clients."""

import asyncio
import json
from unittest.mock import MagicMock, patch

from sigma_verify.clients import VerificationResult
from sigma_verify.handlers import (
    handle_challenge,
    handle_cross_verify,
    handle_get_models,
    handle_init,
    handle_verify_finding,
)


def _mock_openai_client(available=True):
    client = MagicMock()
    client.available = available
    client.model = "gpt-4o"
    client.verify.return_value = VerificationResult(
        model="gpt-4o",
        provider="openai",
        assessment="agree",
        reasoning="Finding is well-supported by data",
        confidence="high",
        counter_evidence="",
        provenance_tag="|source:external-openai-gpt-4o|",
    )
    client.challenge.return_value = {
        "counter_argument": "Sample size too small",
        "logical_gaps": "No control group",
        "evidence_needed": "Longitudinal study",
        "vulnerability": "medium",
        "model": "gpt-4o",
        "provider": "openai",
        "provenance_tag": "|source:external-openai-gpt-4o|",
    }
    return client


def _mock_gemini_client(available=True):
    client = MagicMock()
    client.available = available
    client.model = "gemini-2.0-flash"
    client.verify.return_value = VerificationResult(
        model="gemini-2.0-flash",
        provider="google",
        assessment="partial",
        reasoning="Partially supported but market size seems high",
        confidence="medium",
        counter_evidence="Industry reports show lower figures",
        provenance_tag="|source:external-google-gemini-2.0-flash|",
    )
    client.challenge.return_value = {
        "counter_argument": "Correlation not causation",
        "logical_gaps": "Assumes linear growth",
        "evidence_needed": "Multi-year data",
        "vulnerability": "high",
        "model": "gemini-2.0-flash",
        "provider": "google",
        "provenance_tag": "|source:external-google-gemini-2.0-flash|",
    }
    return client


class TestHandleInit:
    @patch("sigma_verify.handlers._get_clients")
    def test_ready_with_providers(self, mock_get):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = handle_init()
        assert result["status"] == "ready"
        assert "openai" in result["available_providers"]
        assert result["_state"] == "ready"

    @patch("sigma_verify.handlers._get_clients")
    def test_unconfigured_without_providers(self, mock_get):
        mock_get.return_value = {}
        result = handle_init()
        assert result["status"] == "no_providers"
        assert result["_state"] == "unconfigured"


class TestHandleGetModels:
    @patch("sigma_verify.handlers._get_clients")
    def test_lists_available_models(self, mock_get):
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "google": _mock_gemini_client(),
        }
        result = json.loads(handle_get_models())
        assert len(result["available"]) == 2

    @patch("sigma_verify.handlers._get_clients")
    def test_empty_when_no_keys(self, mock_get):
        mock_get.return_value = {}
        result = json.loads(handle_get_models())
        assert result["available"] == []


class TestHandleVerifyFinding:
    @patch("sigma_verify.handlers._get_clients")
    def test_verify_with_default_provider(self, mock_get):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = json.loads(handle_verify_finding(finding="Market is $5B"))
        assert result["assessment"] == "agree"
        assert result["provenance_tag"] == "|source:external-openai-gpt-4o|"

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_with_specific_provider(self, mock_get):
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "google": _mock_gemini_client(),
        }
        result = json.loads(
            handle_verify_finding(
                finding="Market is $5B",
                provider="google",
            )
        )
        assert result["assessment"] == "partial"
        assert "google" in result["provenance_tag"]

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_missing_finding(self, mock_get):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = json.loads(handle_verify_finding())
        assert "error" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_unknown_provider(self, mock_get):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = json.loads(
            handle_verify_finding(
                finding="test",
                provider="anthropic",
            )
        )
        assert "error" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_no_providers(self, mock_get):
        mock_get.return_value = {}
        result = json.loads(handle_verify_finding(finding="test"))
        assert "error" in result


class TestHandleCrossVerify:
    @patch("sigma_verify.handlers._get_clients")
    def test_cross_verify_multiple_providers(self, mock_get):
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "google": _mock_gemini_client(),
        }
        result = json.loads(handle_cross_verify(finding="Market is $5B"))
        assert result["count"] == 2
        assert result["agreement"] == "none"  # agree vs partial

    @patch("sigma_verify.handlers._get_clients")
    def test_cross_verify_unanimous(self, mock_get):
        oai = _mock_openai_client()
        gem = _mock_gemini_client()
        # Make both agree
        gem.verify.return_value = VerificationResult(
            model="gemini-2.0-flash",
            provider="google",
            assessment="agree",
            reasoning="Confirmed",
            confidence="high",
            counter_evidence="",
            provenance_tag="|source:external-google-gemini-2.0-flash|",
        )
        mock_get.return_value = {"openai": oai, "google": gem}
        result = json.loads(handle_cross_verify(finding="test"))
        assert result["agreement"] == "unanimous"

    @patch("sigma_verify.handlers._get_clients")
    def test_cross_verify_inside_running_event_loop(self, mock_get):
        """MCP-server context: the handler runs on the event-loop thread.

        asyncio.run() raises "cannot be called from a running event loop"
        there, which surfaced to agents as an opaque internal error on every
        cross_verify while this suite stayed green — no test called the
        handler the way the server does. This one does.
        """
        mock_get.return_value = {"openai": _mock_openai_client()}

        async def mcp_context():
            return json.loads(handle_cross_verify(finding="loop-context probe"))

        result = asyncio.run(mcp_context())
        assert result["count"] == 1
        assert "agreement" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_cross_verify_handles_api_error(self, mock_get):
        oai = _mock_openai_client()
        gem = _mock_gemini_client()
        gem.verify.side_effect = Exception("API rate limit")
        mock_get.return_value = {"openai": oai, "google": gem}
        result = json.loads(handle_cross_verify(finding="test"))
        assert result["count"] == 2
        # One success, one failure
        failures = [r for r in result["results"] if r.get("status") == "failed"]
        assert len(failures) == 1

    @patch("sigma_verify.handlers._get_clients")
    def test_cross_verify_missing_finding(self, mock_get):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = json.loads(handle_cross_verify())
        assert "error" in result


class TestHandleCrossVerifyFailureTracking:
    """Tests for structured error handling in cross-verification."""

    @patch("sigma_verify.handlers._get_clients")
    def test_partial_failure_shows_coverage_warning(self, mock_get):
        oai = _mock_openai_client()
        gem = _mock_gemini_client()
        gem.verify.side_effect = Exception("API rate limit")
        mock_get.return_value = {"openai": oai, "google": gem}
        result = json.loads(handle_cross_verify(finding="test"))
        assert result["coverage"] == "partial"
        assert "coverage_warning" in result
        assert result["succeeded"] == 1
        assert result["failed"] == 1

    @patch("sigma_verify.handlers._get_clients")
    def test_all_failures_shows_no_data(self, mock_get):
        oai = _mock_openai_client()
        gem = _mock_gemini_client()
        oai.verify.side_effect = Exception("timeout")
        gem.verify.side_effect = Exception("rate limit")
        mock_get.return_value = {"openai": oai, "google": gem}
        result = json.loads(handle_cross_verify(finding="test"))
        assert result["agreement"] == "no_data"
        assert result["succeeded"] == 0
        assert result["failed"] == 2

    @patch("sigma_verify.handlers._get_clients")
    def test_full_success_shows_full_coverage(self, mock_get):
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "google": _mock_gemini_client(),
        }
        result = json.loads(handle_cross_verify(finding="test"))
        assert result["coverage"] == "full"
        assert "coverage_warning" not in result

    @patch("sigma_verify.handlers._get_clients")
    def test_failure_result_has_xverify_fail_sigma(self, mock_get):
        oai = _mock_openai_client()
        gem = _mock_gemini_client()
        gem.verify.side_effect = Exception("API rate limit")
        mock_get.return_value = {"openai": oai, "google": gem}
        result = json.loads(handle_cross_verify(finding="market size $5B"))
        failures = [r for r in result["results"] if r.get("status") == "failed"]
        assert len(failures) == 1
        assert "XVERIFY-FAIL" in failures[0]["sigma"]
        assert "verification-gap" in failures[0]["sigma"]

    @patch("sigma_verify.handlers._get_clients")
    def test_failure_result_classifies_error(self, mock_get):
        oai = _mock_openai_client()
        gem = _mock_gemini_client()
        gem.verify.side_effect = Exception("API rate limit exceeded")
        mock_get.return_value = {"openai": oai, "google": gem}
        result = json.loads(handle_cross_verify(finding="test"))
        failures = [r for r in result["results"] if r.get("status") == "failed"]
        assert failures[0]["error_class"] == "rate-limit"


class TestHandleVerifyFailure:
    """Tests for structured error handling in single verification."""

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_api_error_returns_xverify_fail(self, mock_get):
        oai = _mock_openai_client()
        oai.verify.side_effect = Exception("AuthenticationError: invalid key")
        mock_get.return_value = {"openai": oai}
        result = json.loads(handle_verify_finding(finding="test"))
        assert result["status"] == "failed"
        assert result["error_class"] == "auth-error"
        assert "XVERIFY-FAIL" in result["sigma"]

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_timeout_classified(self, mock_get):
        oai = _mock_openai_client()
        oai.verify.side_effect = Exception("Request timed out after 30s")
        mock_get.return_value = {"openai": oai}
        result = json.loads(handle_verify_finding(finding="test"))
        assert result["error_class"] == "timeout"


class TestHandleChallenge:
    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_returns_counter(self, mock_get):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = json.loads(handle_challenge(claim="AI will dominate"))
        assert "counter_argument" in result
        assert "provenance_tag" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_missing_claim(self, mock_get):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = json.loads(handle_challenge())
        assert "error" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_api_error_returns_xverify_fail(self, mock_get):
        oai = _mock_openai_client()
        oai.challenge.side_effect = Exception("RateLimitError: quota exceeded")
        mock_get.return_value = {"openai": oai}
        result = json.loads(handle_challenge(claim="test claim"))
        assert result["status"] == "failed"
        assert result["error_class"] == "rate-limit"
        assert "XVERIFY-FAIL" in result["sigma"]

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_passes_tier_to_client(self, mock_get):
        oai = _mock_openai_client()
        mock_get.return_value = {"openai": oai}
        handle_challenge(claim="test claim", tier="standard")
        oai.challenge.assert_called_once_with("test claim", "", tier="standard")

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_defaults_to_reasoning_tier(self, mock_get):
        oai = _mock_openai_client()
        mock_get.return_value = {"openai": oai}
        handle_challenge(claim="test claim")
        oai.challenge.assert_called_once_with("test claim", "", tier="reasoning")


class TestHandleVerifyFindingModelOverride:
    """Cover line 160: client.model = model override."""

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_with_model_override(self, mock_get):
        oai = _mock_openai_client()
        mock_get.return_value = {"openai": oai}
        handle_verify_finding(finding="test", model="gpt-4-turbo")
        assert oai.model == "gpt-4-turbo"


class TestHandleCrossVerifyNoProviders:
    """Cover line 193: cross_verify with no clients."""

    @patch("sigma_verify.handlers._get_clients")
    def test_cross_verify_no_providers(self, mock_get):
        mock_get.return_value = {}
        result = json.loads(handle_cross_verify(finding="test"))
        assert "error" in result


class TestHandleCrossVerifyPartialAgreement:
    """Cover line 54: partial agreement in _format_cross_results."""

    @patch("sigma_verify.handlers._get_clients")
    def test_partial_agreement_three_providers(self, mock_get):
        oai = _mock_openai_client()  # agree
        gem = _mock_gemini_client()  # partial
        oai2 = _mock_openai_client()  # agree (same as oai)
        mock_get.return_value = {"openai": oai, "google": gem, "openai2": oai2}
        result = json.loads(handle_cross_verify(finding="test"))
        assert result["agreement"] == "partial"
        assert result["count"] == 3


class TestHandleChallengeProviderVariants:
    """Cover lines 232-246: challenge with no clients, specific/invalid
    provider, and model override."""

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_no_providers(self, mock_get):
        mock_get.return_value = {}
        result = json.loads(handle_challenge(claim="test"))
        assert "error" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_specific_provider(self, mock_get):
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "google": _mock_gemini_client(),
        }
        result = json.loads(handle_challenge(claim="test", provider="google"))
        assert "counter_argument" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_invalid_provider(self, mock_get):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = json.loads(handle_challenge(claim="test", provider="anthropic"))
        assert "error" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_model_override(self, mock_get):
        oai = _mock_openai_client()
        mock_get.return_value = {"openai": oai}
        handle_challenge(claim="test", model="gpt-4-turbo")
        assert oai.model == "gpt-4-turbo"


class TestHandleInitReasoningModels:
    @patch("sigma_verify.handlers._get_clients")
    def test_init_reports_reasoning_models(self, mock_get):
        from sigma_verify.clients import GeminiClient, OpenAIClient

        oai = _mock_openai_client()
        oai.__class__ = OpenAIClient
        oai.reasoning_model = "gpt-5.4-pro"
        gem = _mock_gemini_client()
        gem.__class__ = GeminiClient
        mock_get.return_value = {"openai": oai, "google": gem}
        result = handle_init()
        assert "reasoning_models" in result
        assert result["reasoning_models"]["openai"] == "gpt-5.4-pro"
        assert "thinking mode" in result["reasoning_models"]["google"]


# --- Ollama pre-flight ---


def _mock_gemma_client(available=True):
    client = MagicMock()
    client.available = available
    client.model = "gemma4:e4b"
    client._base_url = "http://localhost:11434/v1"
    client.verify.return_value = VerificationResult(
        model="gemma4:e4b",
        provider="gemma",
        assessment="agree",
        reasoning="Finding checks out",
        confidence="medium",
        counter_evidence="",
        provenance_tag="|source:external-gemma-gemma4:e4b|",
    )
    client.challenge.return_value = {
        "counter_argument": "Alternative interpretation possible",
        "logical_gaps": "Limited sample",
        "evidence_needed": "Broader dataset",
        "vulnerability": "medium",
        "model": "gemma4:e4b",
        "provider": "gemma",
        "provenance_tag": "|source:external-gemma-gemma4:e4b|",
    }
    client.check_quota.return_value = {
        "provider": "gemma",
        "status": "ok",
        "message": "Gemma via Ollama (gemma4:e4b): local model, no API cost.",
    }
    return client


class TestHasOllamaClients:
    def test_detects_ollama_client(self):
        from sigma_verify.handlers import _has_ollama_clients

        clients = {"gemma": _mock_gemma_client()}
        assert _has_ollama_clients(clients) is True

    def test_no_ollama_when_all_remote(self):
        from sigma_verify.handlers import _has_ollama_clients

        clients = {"openai": _mock_openai_client(), "google": _mock_gemini_client()}
        assert _has_ollama_clients(clients) is False

    def test_mixed_clients(self):
        from sigma_verify.handlers import _has_ollama_clients

        clients = {"openai": _mock_openai_client(), "gemma": _mock_gemma_client()}
        assert _has_ollama_clients(clients) is True


class TestIsOllamaRunning:
    def test_running_when_reachable(self, monkeypatch):
        from sigma_verify import clients
        from sigma_verify.handlers import _is_ollama_running

        monkeypatch.setattr(clients, "_fetch_ollama_tags", lambda root: [])
        assert _is_ollama_running(refresh=True) is True

    def test_not_running_when_unreachable(self, monkeypatch):
        from sigma_verify import clients
        from sigma_verify.handlers import _is_ollama_running

        monkeypatch.setattr(clients, "_fetch_ollama_tags", lambda root: None)
        assert _is_ollama_running(refresh=True) is False


class TestEnsureOllama:
    @patch("sigma_verify.handlers._is_ollama_running", return_value=True)
    def test_already_running(self, mock_check):
        from sigma_verify.handlers import _ensure_ollama

        result = _ensure_ollama()
        assert result["status"] == "ok"

    @patch("sigma_verify.handlers._start_ollama", return_value=True)
    @patch("sigma_verify.handlers._is_ollama_running", return_value=False)
    def test_started_successfully(self, mock_check, mock_start, monkeypatch):
        from sigma_verify.handlers import _ensure_ollama

        monkeypatch.setenv("SIGMA_VERIFY_AUTOSTART_OLLAMA", "1")

        result = _ensure_ollama()
        assert result["status"] == "started"

    @patch("sigma_verify.handlers._start_ollama", return_value=False)
    @patch("sigma_verify.handlers._is_ollama_running", return_value=False)
    def test_failed_to_start(self, mock_check, mock_start, monkeypatch):
        from sigma_verify.handlers import _ensure_ollama

        monkeypatch.setenv("SIGMA_VERIFY_AUTOSTART_OLLAMA", "1")

        result = _ensure_ollama()
        assert result["status"] == "unavailable"


class TestInitWithOllama:
    @patch("sigma_verify.handlers._ensure_ollama")
    @patch("sigma_verify.handlers._get_clients")
    def test_init_includes_ollama_status(self, mock_get, mock_ensure):
        mock_ensure.return_value = {"status": "ok", "message": "Ollama is running."}
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "gemma": _mock_gemma_client(),
        }
        result = handle_init()
        assert result["ollama"]["status"] == "ok"
        assert "gemma" in result["available_providers"]

    @patch("sigma_verify.handlers._ensure_ollama")
    @patch("sigma_verify.handlers._get_clients")
    def test_init_drops_ollama_clients_when_unavailable(self, mock_get, mock_ensure):
        mock_ensure.return_value = {
            "status": "unavailable",
            "message": "Ollama is not running.",
        }
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "gemma": _mock_gemma_client(),
        }
        result = handle_init()
        assert "gemma" not in result["available_providers"]
        assert "openai" in result["available_providers"]

    @patch("sigma_verify.handlers._ollama_configured", return_value=False)
    @patch("sigma_verify.handlers._get_clients")
    def test_init_no_ollama_status_when_no_ollama_providers_configured(self, mock_get, _):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = handle_init()
        assert "ollama" not in result


class TestPreFlightWithOllama:
    @patch("sigma_verify.handlers._ensure_ollama")
    @patch("sigma_verify.handlers._has_ollama_clients", return_value=True)
    def test_preflight_includes_ollama(self, mock_has, mock_ensure):
        from sigma_verify.handlers import _pre_flight_check

        mock_ensure.return_value = {"status": "ok", "message": "Ollama is running."}
        clients = {
            "openai": _mock_openai_client(),
            "gemma": _mock_gemma_client(),
        }
        result = _pre_flight_check(clients)
        assert result["ollama"]["status"] == "ok"

    @patch("sigma_verify.handlers._ensure_ollama")
    @patch("sigma_verify.handlers._has_ollama_clients", return_value=True)
    def test_preflight_ollama_free(self, mock_has, mock_ensure):
        from sigma_verify.handlers import _pre_flight_check

        mock_ensure.return_value = {"status": "ok", "message": "Ollama is running."}
        clients = {"gemma": _mock_gemma_client()}
        result = _pre_flight_check(clients)
        assert result["cost_estimate_usd"] == 0.0

    @patch("sigma_verify.handlers._has_ollama_clients", return_value=False)
    def test_preflight_no_ollama_check_for_remote_only(self, mock_has):
        from sigma_verify.handlers import _pre_flight_check

        clients = {"openai": _mock_openai_client()}
        result = _pre_flight_check(clients)
        assert "ollama" not in result


# --- Provider selection in cross_verify ---


class TestCrossVerifyProviderSelection:
    @patch("sigma_verify.handlers._get_clients")
    def test_filter_to_requested_providers(self, mock_get):
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "google": _mock_gemini_client(),
            "gemma": _mock_gemma_client(),
        }
        result = json.loads(handle_cross_verify(finding="test", providers="openai,gemma"))
        assert result["count"] == 2
        providers_used = {r["provider"] for r in result["results"]}
        assert providers_used == {"openai", "gemma"}

    @patch("sigma_verify.handlers._get_clients")
    def test_empty_providers_means_all(self, mock_get):
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "google": _mock_gemini_client(),
        }
        result = json.loads(handle_cross_verify(finding="test", providers=""))
        assert result["count"] == 2

    @patch("sigma_verify.handlers._get_clients")
    def test_unavailable_requested_in_preflight(self, mock_get):
        mock_get.return_value = {
            "openai": _mock_openai_client(),
        }
        result = json.loads(handle_cross_verify(finding="test", providers="openai,nonexistent"))
        assert result["count"] == 1
        assert "nonexistent" in result["pre_flight"]["unavailable_requested"]

    @patch("sigma_verify.handlers._get_clients")
    def test_all_requested_unavailable_returns_error(self, mock_get):
        mock_get.return_value = {"openai": _mock_openai_client()}
        result = json.loads(handle_cross_verify(finding="test", providers="nonexistent"))
        assert "error" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_whitespace_in_providers_handled(self, mock_get):
        mock_get.return_value = {
            "openai": _mock_openai_client(),
            "google": _mock_gemini_client(),
        }
        result = json.loads(handle_cross_verify(finding="test", providers="openai, google"))
        assert result["count"] == 2
