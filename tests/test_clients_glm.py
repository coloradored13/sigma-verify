"""Tests for GlmClient — class-var configuration only.
All behavior tested via test_clients_ollama_base.py."""

from sigma_verify.clients import DESC_CLOUD_PAID, GlmClient, _OllamaClientBase


class TestGlmClientConfig:
    def test_is_ollama_base(self):
        assert isinstance(GlmClient(), _OllamaClientBase)

    def test_provider_name(self):
        assert GlmClient.PROVIDER_NAME == "glm"

    def test_env_prefix(self):
        assert GlmClient.ENV_PREFIX == "GLM"

    def test_default_model(self):
        assert GlmClient().model == "glm-5.3:cloud"

    def test_default_base_url(self):
        assert GlmClient()._base_url == "http://localhost:11434/v1"

    def test_available_when_ollama_running(self):
        assert GlmClient().available is True

    def test_description(self):
        assert GlmClient.DESCRIPTION == DESC_CLOUD_PAID
