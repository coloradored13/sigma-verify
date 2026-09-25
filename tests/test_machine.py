"""Tests for sigma_verify.machine — state machine definition and integration."""

from sigma_verify.machine import build_machine


class TestBuildMachine:
    def test_builds_without_error(self):
        m = build_machine()
        assert m.name == "sigma_verify"

    def test_has_gateway(self):
        m = build_machine()
        assert m._gateway_def is not None
        assert m._gateway_def.name == "init"

    def test_has_all_actions(self):
        m = build_machine()
        expected = {
            "get_models",
            "verify_finding",
            "cross_verify",
            "challenge",
            "check_quotas",
        }
        assert set(m._action_defs.keys()) == expected

    def test_verify_requires_finding(self):
        m = build_machine()
        assert "finding" in m._action_defs["verify_finding"].required

    def test_challenge_requires_claim(self):
        m = build_machine()
        assert "claim" in m._action_defs["challenge"].required

    def test_cross_verify_requires_finding(self):
        m = build_machine()
        assert "finding" in m._action_defs["cross_verify"].required

    def test_get_models_available_from_all_states(self):
        m = build_machine()
        assert m._action_from_states["get_models"] == "*"

    def test_verify_only_from_ready(self):
        m = build_machine()
        assert "ready" in m._action_from_states["verify_finding"]


class TestMachineClosureInvocation:
    """Exercise closure bodies in machine.py (lines 92-109) by calling
    handlers through the state machine with mocked clients."""

    def test_init_closure(self):
        from unittest.mock import patch

        m = build_machine()
        gw = m.get_gateway()
        with patch("sigma_verify.handlers._get_clients", return_value={}):
            result = gw.handler()
        assert result["status"] == "no_providers"

    def test_get_models_closure(self):
        from unittest.mock import patch

        m = build_machine()
        handler = m._get_handler("get_models")
        with patch("sigma_verify.handlers._get_clients", return_value={}):
            import json

            result = json.loads(handler())
        assert result["available"] == []

    def test_verify_closure(self):
        from unittest.mock import MagicMock, patch

        from sigma_verify.clients import VerificationResult

        m = build_machine()
        handler = m._get_handler("verify_finding")
        client = MagicMock()
        client.available = True
        client.model = "gpt-4o"
        client.verify.return_value = VerificationResult(
            model="gpt-4o",
            provider="openai",
            assessment="agree",
            reasoning="ok",
            confidence="high",
            counter_evidence="",
            provenance_tag="|source:external-openai-gpt-4o|",
        )
        with patch("sigma_verify.handlers._get_clients", return_value={"openai": client}):
            import json

            result = json.loads(handler(finding="test"))
        assert result["assessment"] == "agree"

    def test_cross_verify_closure(self):
        from unittest.mock import MagicMock, patch

        from sigma_verify.clients import VerificationResult

        m = build_machine()
        handler = m._get_handler("cross_verify")
        client = MagicMock()
        client.available = True
        client.model = "gpt-4o"
        client.verify.return_value = VerificationResult(
            model="gpt-4o",
            provider="openai",
            assessment="agree",
            reasoning="ok",
            confidence="high",
            counter_evidence="",
            provenance_tag="|source:external-openai-gpt-4o|",
        )
        with patch("sigma_verify.handlers._get_clients", return_value={"openai": client}):
            import json

            result = json.loads(handler(finding="test"))
        assert result["count"] == 1

    def test_challenge_closure(self):
        from unittest.mock import MagicMock, patch

        m = build_machine()
        handler = m._get_handler("challenge")
        client = MagicMock()
        client.available = True
        client.model = "gpt-4o"
        client.challenge.return_value = {
            "counter_argument": "weak",
            "model": "gpt-4o",
            "provider": "openai",
            "provenance_tag": "|source:external|",
        }
        with patch("sigma_verify.handlers._get_clients", return_value={"openai": client}):
            import json

            result = json.loads(handler(claim="test claim"))
        assert result["status"] == "success"
