"""Shared fixtures: keep the suite offline and independent of the host environment."""

import socket

import pytest

from sigma_verify import clients as clients_mod
from sigma_verify import handlers as handlers_mod

# Every local default model plus the test-only model used in test_clients_ollama_base.
FAKE_OLLAMA_TAGS = [
    cls.DEFAULT_MODEL
    for cls in clients_mod.OLLAMA_CLIENTS
    if not clients_mod.is_cloud_model(cls.DEFAULT_MODEL)
] + ["test-model:latest"]

_ENV_TO_CLEAR = ["SIGMA_VERIFY_ALLOW_ANTHROPIC", "SIGMA_VERIFY_AUTOSTART_OLLAMA"] + [
    f"{cls.ENV_PREFIX}_{suffix}"
    for cls in clients_mod.OLLAMA_CLIENTS
    for suffix in ("PROVIDER", "MODEL", "BASE_URL", "API_KEY")
]

_real_connect = socket.socket.connect


def _guarded_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError(f"Network access blocked in tests: {address!r}")
    return _real_connect(self, address)


@pytest.fixture(autouse=True)
def offline_ollama(monkeypatch):
    """Pretend Ollama is running with every default local model pulled.

    Tests that need a different Ollama state monkeypatch
    sigma_verify.clients._fetch_ollama_tags themselves.
    """
    for var in _ENV_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(clients_mod, "_fetch_ollama_tags", lambda root: list(FAKE_OLLAMA_TAGS))
    monkeypatch.setattr(handlers_mod, "_start_ollama", lambda: False)
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    clients_mod.reset_ollama_probe()
    yield
    clients_mod.reset_ollama_probe()
