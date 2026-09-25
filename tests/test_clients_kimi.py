"""Tests for KimiClient — class-var configuration only.
All behavior tested via test_clients_ollama_base.py."""

from sigma_verify.clients import DESC_CLOUD_PAID, KimiClient, _OllamaClientBase


class TestKimiClientConfig:
    def test_is_ollama_base(self):
        assert isinstance(KimiClient(), _OllamaClientBase)

    def test_provider_name(self):
        assert KimiClient.PROVIDER_NAME == "kimi"

    def test_env_prefix(self):
        assert KimiClient.ENV_PREFIX == "KIMI"

    def test_default_model(self):
        assert KimiClient().model == "kimi-k3:cloud"

    def test_default_base_url(self):
        assert KimiClient()._base_url == "http://localhost:11434/v1"

    def test_available_when_ollama_running(self):
        assert KimiClient().available is True

    def test_description(self):
        assert KimiClient.DESCRIPTION == DESC_CLOUD_PAID
