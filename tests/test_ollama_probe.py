"""Tests for the Ollama availability probe and Ollama-backed provider availability."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from sigma_verify import clients, handlers
from sigma_verify.clients import (
    OLLAMA_PROBE_TIMEOUT,
    GemmaClient,
    GptOssClient,
    LlamaClient,
    NemotronClient,
    _fetch_ollama_tags,  # the real function — conftest patches the module attribute
    is_cloud_model,
    ollama_root_url,
    probe_ollama,
    reset_ollama_probe,
)
from sigma_verify.handlers import handle_init
from sigma_verify.machine import build_machine


def _set_tags(monkeypatch, tags):
    """Make the probe report `tags` (None = Ollama unreachable). Counts fetches."""
    calls = []

    def fake(root):
        calls.append(root)
        return None if tags is None else list(tags)

    monkeypatch.setattr(clients, "_fetch_ollama_tags", fake)
    reset_ollama_probe()
    return calls


def _urlopen_response(payload: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestFetchOllamaTags:
    @patch("sigma_verify.clients.urllib.request.urlopen")
    def test_parses_model_names(self, mock_urlopen):
        mock_urlopen.return_value = _urlopen_response(
            {"models": [{"name": "llama3.1:8b"}, {"name": "gemma4:e4b"}]}
        )
        assert _fetch_ollama_tags("http://localhost:11434") == ["llama3.1:8b", "gemma4:e4b"]

    @patch("sigma_verify.clients.urllib.request.urlopen")
    def test_hits_api_tags_with_short_timeout(self, mock_urlopen):
        mock_urlopen.return_value = _urlopen_response({"models": []})
        _fetch_ollama_tags("http://localhost:11434")
        req = mock_urlopen.call_args.args[0]
        assert req.full_url == "http://localhost:11434/api/tags"
        assert req.get_method() == "GET"
        assert mock_urlopen.call_args.kwargs["timeout"] == OLLAMA_PROBE_TIMEOUT
        assert OLLAMA_PROBE_TIMEOUT <= 2

    @patch("sigma_verify.clients.urllib.request.urlopen", side_effect=OSError("refused"))
    def test_unreachable_returns_none(self, _):
        assert _fetch_ollama_tags("http://localhost:11434") is None

    @patch("sigma_verify.clients.urllib.request.urlopen")
    def test_non_json_returns_none(self, mock_urlopen):
        resp = _urlopen_response({})
        resp.read.return_value = b"<html>not ollama</html>"
        mock_urlopen.return_value = resp
        assert _fetch_ollama_tags("http://localhost:11434") is None

    def test_real_urlopen_reaches_blocked_socket_as_none(self):
        # conftest blocks TCP connects; the probe must swallow that, not raise.
        assert _fetch_ollama_tags("http://127.0.0.1:9") is None


class TestProbeHelpers:
    @pytest.mark.parametrize(
        "base_url,root",
        [
            ("http://localhost:11434/v1", "http://localhost:11434"),
            ("http://localhost:11434/v1/", "http://localhost:11434"),
            ("http://gpu-box:11434", "http://gpu-box:11434"),
        ],
    )
    def test_root_url(self, base_url, root):
        assert ollama_root_url(base_url) == root

    @pytest.mark.parametrize(
        "model,cloud",
        [
            ("gpt-oss:120b-cloud", True),
            ("glm-5.3:cloud", True),
            ("mistral-large-3:675b-cloud", True),
            ("llama3.1:8b", False),
            ("gemma4:e4b", False),
            ("cloudy-model", False),
            ("meta-llama/llama-4-maverick", False),
        ],
    )
    def test_is_cloud_model(self, model, cloud):
        assert is_cloud_model(model) is cloud


class TestProbeCache:
    def test_cached_per_root(self, monkeypatch):
        calls = _set_tags(monkeypatch, ["llama3.1:8b"])
        probe_ollama("http://localhost:11434/v1")
        probe_ollama("http://localhost:11434/v1")
        LlamaClient().available
        GemmaClient().available
        assert calls == ["http://localhost:11434"]

    def test_refresh_refetches(self, monkeypatch):
        calls = _set_tags(monkeypatch, [])
        probe_ollama("http://localhost:11434/v1")
        probe_ollama("http://localhost:11434/v1", refresh=True)
        assert len(calls) == 2

    def test_reset_clears(self, monkeypatch):
        calls = _set_tags(monkeypatch, [])
        probe_ollama("http://localhost:11434/v1")
        reset_ollama_probe()
        probe_ollama("http://localhost:11434/v1")
        assert len(calls) == 2

    def test_separate_roots_probed_separately(self, monkeypatch):
        calls = _set_tags(monkeypatch, [])
        probe_ollama("http://localhost:11434/v1")
        probe_ollama("http://gpu-box:11434/v1")
        assert calls == ["http://localhost:11434", "http://gpu-box:11434"]


class TestOllamaDown:
    def test_local_provider_unavailable(self, monkeypatch):
        _set_tags(monkeypatch, None)
        client = LlamaClient()
        assert client.available is False
        assert "Ollama not running at http://localhost:11434" in client.unavailable_reason

    def test_cloud_provider_unavailable(self, monkeypatch):
        _set_tags(monkeypatch, None)
        assert GptOssClient().available is False
        assert NemotronClient().available is False

    def test_quota_reports_reason(self, monkeypatch):
        _set_tags(monkeypatch, None)
        quota = LlamaClient().check_quota()
        assert quota["status"] == "unavailable"
        assert "Ollama not running" in quota["message"]
        gemma = GemmaClient().check_quota()
        assert gemma["status"] == "unavailable"
        assert "Ollama not running" in gemma["message"]

    @patch.dict(os.environ, {"LLAMA_BASE_URL": "http://gpu-box:11434/v1"})
    def test_reason_names_configured_server(self, monkeypatch):
        _set_tags(monkeypatch, None)
        assert "http://gpu-box:11434" in LlamaClient().unavailable_reason


class TestOllamaUp:
    def test_local_model_not_pulled(self, monkeypatch):
        _set_tags(monkeypatch, ["gemma4:e4b"])
        client = LlamaClient()
        assert client.available is False
        assert "ollama pull llama3.1:8b" in client.unavailable_reason
        assert GemmaClient().available is True

    def test_cloud_model_available_without_local_tag(self, monkeypatch):
        _set_tags(monkeypatch, [])
        client = GptOssClient()
        assert client.is_cloud is True
        assert client.available is True
        assert client.unavailable_reason is None

    def test_untagged_model_matches_latest(self, monkeypatch):
        _set_tags(monkeypatch, ["mistral:latest"])
        assert LlamaClient(model="mistral").available is True
        assert LlamaClient(model="mistral:7b").available is False

    def test_quota_ok_mentions_signin_for_cloud(self, monkeypatch):
        _set_tags(monkeypatch, [])
        quota = GptOssClient().check_quota()
        assert quota["status"] == "ok"
        assert "ollama signin" in quota["message"]


class TestNonOllamaPresetSkipsProbe:
    @patch.dict(os.environ, {"LLAMA_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or"})
    def test_openrouter_does_not_probe(self, monkeypatch):
        def boom(root):
            raise AssertionError("probe must not run for non-Ollama presets")

        monkeypatch.setattr(clients, "_fetch_ollama_tags", boom)
        reset_ollama_probe()
        client = LlamaClient()
        assert client.available is True
        assert client.is_cloud is False

    @patch.dict(os.environ, {"LLAMA_PROVIDER": "openrouter"})
    def test_openrouter_without_key_reason(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert "No API key" in LlamaClient().unavailable_reason


class TestInitReporting:
    def test_ollama_down_drops_ollama_providers_with_reasons(self, monkeypatch):
        _set_tags(monkeypatch, None)
        result = handle_init()
        assert result["ollama"]["status"] == "unavailable"
        assert result["status"] == "no_providers"
        assert "Ollama not running" in result["unavailable_providers"]["llama"]
        assert "Ollama not running" in result["unavailable_providers"]["gpt-oss"]

    def test_unpulled_local_models_reported(self, monkeypatch):
        _set_tags(monkeypatch, ["llama3.1:8b"])
        result = handle_init()
        assert "llama" in result["available_providers"]
        assert "gemma" not in result["available_providers"]
        assert "ollama pull gemma4:e4b" in result["unavailable_providers"]["gemma"]

    def test_cloud_providers_flagged_for_signin(self, monkeypatch):
        _set_tags(monkeypatch, [])
        result = handle_init()
        assert "gpt-oss" in result["ollama_cloud"]["providers"]
        assert "llama" not in result["ollama_cloud"]["providers"]
        assert "ollama signin" in result["ollama_cloud"]["note"]

    def test_init_refreshes_stale_probe(self, monkeypatch):
        _set_tags(monkeypatch, None)
        assert LlamaClient().available is False  # caches "down"
        monkeypatch.setattr(clients, "_fetch_ollama_tags", lambda root: ["llama3.1:8b"])
        result = handle_init()
        assert "llama" in result["available_providers"]

    def test_missing_api_keys_reported(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_AI_API_KEY", raising=False)
        result = handle_init()
        assert "OPENAI_API_KEY" in result["unavailable_providers"]["openai"]
        assert "GOOGLE_AI_API_KEY" in result["unavailable_providers"]["google"]


class TestAutostartOptIn:
    """`ollama serve` runs only when SIGMA_VERIFY_AUTOSTART_OLLAMA is set."""

    @pytest.fixture
    def start_calls(self, monkeypatch):
        calls = []

        def fake_start():
            calls.append(1)
            return False

        monkeypatch.setattr(handlers, "_start_ollama", fake_start)
        _set_tags(monkeypatch, None)
        return calls

    def test_off_by_default_in_init(self, start_calls):
        result = handle_init()
        assert start_calls == []
        assert result["ollama"]["status"] == "unavailable"
        assert "SIGMA_VERIFY_AUTOSTART_OLLAMA=1" in result["ollama"]["message"]

    def test_off_by_default_at_server_startup(self, start_calls):
        build_machine()
        assert start_calls == []

    def test_provider_reason_mentions_opt_in(self, start_calls):
        reason = LlamaClient().unavailable_reason
        assert "Ollama not running at http://localhost:11434" in reason
        assert "or set SIGMA_VERIFY_AUTOSTART_OLLAMA=1" in reason

    @pytest.mark.parametrize("value", ["0", "", "no", "false"])
    def test_other_values_do_not_opt_in(self, start_calls, monkeypatch, value):
        monkeypatch.setenv("SIGMA_VERIFY_AUTOSTART_OLLAMA", value)
        handle_init()
        assert start_calls == []

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
    def test_opt_in_runs_start(self, start_calls, monkeypatch, value):
        monkeypatch.setenv("SIGMA_VERIFY_AUTOSTART_OLLAMA", value)
        result = handle_init()
        assert start_calls == [1]
        assert result["ollama"]["status"] == "unavailable"  # stub reports start failure

    def test_opt_in_at_server_startup(self, start_calls, monkeypatch):
        monkeypatch.setenv("SIGMA_VERIFY_AUTOSTART_OLLAMA", "1")
        build_machine()
        assert start_calls == [1]

    def test_opt_in_successful_start_enables_providers(self, monkeypatch):
        tags = {"value": None}
        monkeypatch.setattr(clients, "_fetch_ollama_tags", lambda root: tags["value"])
        reset_ollama_probe()

        def fake_start():
            tags["value"] = ["llama3.1:8b"]  # server comes up
            # like the real _start_ollama, the readiness poll refreshes the probe
            return handlers._is_ollama_running(refresh=True)

        monkeypatch.setattr(handlers, "_start_ollama", fake_start)
        monkeypatch.setenv("SIGMA_VERIFY_AUTOSTART_OLLAMA", "1")
        result = handle_init()
        assert result["ollama"]["status"] == "started"
        assert "llama" in result["available_providers"]

    def test_running_ollama_never_starts(self, monkeypatch):
        calls = []
        monkeypatch.setattr(handlers, "_start_ollama", lambda: calls.append(1) or True)
        monkeypatch.setenv("SIGMA_VERIFY_AUTOSTART_OLLAMA", "1")
        _set_tags(monkeypatch, [])
        handle_init()
        assert calls == []
