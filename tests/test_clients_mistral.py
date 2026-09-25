"""Tests for MistralClient — class-var configuration only.
All behavior tested via test_clients_ollama_base.py."""

from sigma_verify.clients import DESC_CLOUD_PAID, MistralClient, _OllamaClientBase


class TestMistralClientConfig:
    def test_is_ollama_base(self):
        assert isinstance(MistralClient(), _OllamaClientBase)

    def test_provider_name(self):
        assert MistralClient.PROVIDER_NAME == "mistral"

    def test_env_prefix(self):
        assert MistralClient.ENV_PREFIX == "MISTRAL"

    def test_default_model(self):
        assert MistralClient().model == "mistral-large-3:675b-cloud"

    def test_default_base_url(self):
        assert MistralClient()._base_url == "http://localhost:11434/v1"

    def test_available_when_ollama_running(self):
        assert MistralClient().available is True

    def test_is_cloud(self):
        assert MistralClient().is_cloud is True

    def test_description(self):
        assert MistralClient.DESCRIPTION == DESC_CLOUD_PAID
