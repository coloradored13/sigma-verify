"""Tests for NemotronNanoClient and QwenLocalClient — class-var configuration only.
All behavior tested via test_clients_ollama_base.py."""

from sigma_verify.clients import NemotronNanoClient, QwenLocalClient, _OllamaClientBase


class TestNemotronNanoConfig:
    def test_is_ollama_base(self):
        assert isinstance(NemotronNanoClient(), _OllamaClientBase)

    def test_provider_name(self):
        assert NemotronNanoClient.PROVIDER_NAME == "nemotron-nano"

    def test_default_model(self):
        assert NemotronNanoClient().model == "nemotron-3-nano:4b"

    def test_default_base_url(self):
        assert NemotronNanoClient()._base_url == "http://localhost:11434/v1"

    def test_available_when_ollama_running(self):
        assert NemotronNanoClient().available is True

    def test_description(self):
        assert "local model" in NemotronNanoClient.DESCRIPTION


class TestQwenLocalConfig:
    def test_is_ollama_base(self):
        assert isinstance(QwenLocalClient(), _OllamaClientBase)

    def test_provider_name(self):
        assert QwenLocalClient.PROVIDER_NAME == "qwen-local"

    def test_default_model(self):
        assert QwenLocalClient().model == "qwen3.5:4b"

    def test_default_base_url(self):
        assert QwenLocalClient()._base_url == "http://localhost:11434/v1"

    def test_available_when_ollama_running(self):
        assert QwenLocalClient().available is True

    def test_description(self):
        assert "local model" in QwenLocalClient.DESCRIPTION
