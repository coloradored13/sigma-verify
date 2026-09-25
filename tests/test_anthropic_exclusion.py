"""Anthropic is excluded from every default provider selection.

Callers of this server are Claude agents; Claude verifying Claude is not
cross-model verification. Anthropic is used only when named explicitly or when
SIGMA_VERIFY_ALLOW_ANTHROPIC=1.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from sigma_verify.clients import AnthropicClient, VerificationResult
from sigma_verify.handlers import (
    handle_challenge,
    handle_cross_verify,
    handle_get_models,
    handle_init,
    handle_verify_finding,
)


def _mock_client(provider: str, model: str):
    client = MagicMock()
    client.available = True
    client.model = model
    client.verify.return_value = VerificationResult(
        model=model,
        provider=provider,
        assessment="agree",
        reasoning="ok",
        confidence="high",
        counter_evidence="",
        provenance_tag=f"|source:external-{provider}-{model}|",
    )
    client.challenge.return_value = {
        "counter_argument": "weak",
        "model": model,
        "provider": provider,
        "provenance_tag": f"|source:external-{provider}-{model}|",
    }
    return client


def _anthropic():
    return _mock_client("anthropic", "claude-opus-5-5")


def _openai():
    return _mock_client("openai", "gpt-5.4")


@pytest.fixture
def allow_anthropic(monkeypatch):
    monkeypatch.setenv("SIGMA_VERIFY_ALLOW_ANTHROPIC", "1")


def test_default_model():
    assert AnthropicClient().model == "claude-opus-5-5"


class TestCrossVerify:
    @patch("sigma_verify.handlers._get_clients")
    def test_default_excludes_anthropic(self, mock_get):
        anth = _anthropic()
        mock_get.return_value = {"anthropic": anth, "openai": _openai()}
        result = json.loads(handle_cross_verify(finding="x"))
        assert result["count"] == 1
        assert {r["provider"] for r in result["results"]} == {"openai"}
        anth.verify.assert_not_called()

    @patch("sigma_verify.handlers._get_clients")
    def test_explicit_name_includes_anthropic(self, mock_get):
        mock_get.return_value = {"anthropic": _anthropic(), "openai": _openai()}
        result = json.loads(handle_cross_verify(finding="x", providers="anthropic,openai"))
        assert {r["provider"] for r in result["results"]} == {"anthropic", "openai"}

    @patch("sigma_verify.handlers._get_clients")
    def test_env_opt_in_includes_anthropic(self, mock_get, allow_anthropic):
        mock_get.return_value = {"anthropic": _anthropic(), "openai": _openai()}
        result = json.loads(handle_cross_verify(finding="x"))
        assert result["count"] == 2

    @patch("sigma_verify.handlers._get_clients")
    def test_only_anthropic_available_is_an_error(self, mock_get):
        mock_get.return_value = {"anthropic": _anthropic()}
        result = json.loads(handle_cross_verify(finding="x"))
        assert "error" in result
        assert "SIGMA_VERIFY_ALLOW_ANTHROPIC" in result["error"]


class TestSingleProviderTools:
    @patch("sigma_verify.handlers._get_clients")
    def test_verify_default_skips_anthropic_even_if_first(self, mock_get):
        anth = _anthropic()
        mock_get.return_value = {"anthropic": anth, "openai": _openai()}
        result = json.loads(handle_verify_finding(finding="x"))
        assert result["provider"] == "openai"
        anth.verify.assert_not_called()

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_explicit_anthropic_allowed(self, mock_get):
        mock_get.return_value = {"anthropic": _anthropic(), "openai": _openai()}
        result = json.loads(handle_verify_finding(finding="x", provider="anthropic"))
        assert result["provider"] == "anthropic"

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_only_anthropic_errors(self, mock_get):
        mock_get.return_value = {"anthropic": _anthropic()}
        result = json.loads(handle_verify_finding(finding="x"))
        assert "error" in result

    @patch("sigma_verify.handlers._get_clients")
    def test_verify_env_opt_in(self, mock_get, allow_anthropic):
        mock_get.return_value = {"anthropic": _anthropic()}
        result = json.loads(handle_verify_finding(finding="x"))
        assert result["provider"] == "anthropic"

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_default_skips_anthropic(self, mock_get):
        anth = _anthropic()
        mock_get.return_value = {"anthropic": anth, "openai": _openai()}
        result = json.loads(handle_challenge(claim="x"))
        assert result["provider"] == "openai"
        anth.challenge.assert_not_called()

    @patch("sigma_verify.handlers._get_clients")
    def test_challenge_explicit_anthropic_allowed(self, mock_get):
        mock_get.return_value = {"anthropic": _anthropic(), "openai": _openai()}
        result = json.loads(handle_challenge(claim="x", provider="anthropic"))
        assert result["provider"] == "anthropic"

    @patch("sigma_verify.handlers._get_clients")
    def test_error_path_reports_selected_provider(self, mock_get):
        oai = _openai()
        oai.verify.side_effect = RuntimeError("boom")
        mock_get.return_value = {"anthropic": _anthropic(), "openai": oai}
        result = json.loads(handle_verify_finding(finding="x"))
        assert result["status"] == "failed"
        assert result["provider"] == "openai"


class TestInitAndModels:
    @patch("sigma_verify.handlers._get_clients")
    def test_init_excludes_anthropic_from_available(self, mock_get):
        mock_get.return_value = {"anthropic": _anthropic(), "openai": _openai()}
        result = handle_init()
        assert result["available_providers"] == ["openai"]
        assert "anthropic" not in result["models"]
        assert result["excluded_by_default"] == ["anthropic"]
        assert "SIGMA_VERIFY_ALLOW_ANTHROPIC" in result["excluded_note"]

    @patch("sigma_verify.handlers._get_clients")
    def test_init_only_anthropic_is_unconfigured(self, mock_get):
        mock_get.return_value = {"anthropic": _anthropic()}
        result = handle_init()
        assert result["status"] == "no_providers"
        assert result["_state"] == "unconfigured"
        assert result["excluded_by_default"] == ["anthropic"]

    @patch("sigma_verify.handlers._get_clients")
    def test_init_env_opt_in(self, mock_get, allow_anthropic):
        mock_get.return_value = {"anthropic": _anthropic(), "openai": _openai()}
        result = handle_init()
        assert set(result["available_providers"]) == {"anthropic", "openai"}
        assert "excluded_by_default" not in result

    @patch("sigma_verify.handlers._get_clients")
    def test_get_models_flags_default_selection(self, mock_get):
        mock_get.return_value = {"anthropic": _anthropic(), "openai": _openai()}
        models = {m["provider"]: m for m in json.loads(handle_get_models())["available"]}
        assert models["anthropic"]["default_selection"] is False
        assert models["openai"]["default_selection"] is True

    @pytest.mark.parametrize("value", ["0", "", "no"])
    @patch("sigma_verify.handlers._get_clients")
    def test_env_values_that_do_not_opt_in(self, mock_get, value, monkeypatch):
        monkeypatch.setenv("SIGMA_VERIFY_ALLOW_ANTHROPIC", value)
        mock_get.return_value = {"anthropic": _anthropic(), "openai": _openai()}
        assert handle_init()["available_providers"] == ["openai"]
