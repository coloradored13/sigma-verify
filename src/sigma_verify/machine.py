"""State machine definition for ΣVerify — cross-model verification as MCP tools."""

from __future__ import annotations

import sys

from hateoas_agent import StateMachine

from .handlers import (
    handle_challenge,
    handle_check_quotas,
    handle_cross_verify,
    handle_get_models,
    handle_init,
    handle_verify_finding,
)


def build_machine() -> StateMachine:
    """Build the ΣVerify state machine.

    Gateway semantics: the MCP layer does not advertise ``ready``-state
    actions (verify_finding, cross_verify, challenge) until a session calls
    the ``init`` gateway and receives ``_state: "ready"``. This is a
    hateoas-agent contract. Agents that skip ``init`` see only the gateway
    tool at MCP ``tools/list`` time.

    At startup the server also runs ``handle_init`` once, logs the result to
    stderr for operator visibility, and sets ``_initial_state_hint`` so a
    hateoas-agent version that honors it can start in ``ready``.

    Future state-gated tools MUST authorize in the handler as well; do not
    rely on MCP tool advertisement as an authorization signal.

    States:
        unconfigured — no providers available, only get_models/check_quotas
        ready — at least one provider available, all tools unlocked
    """
    vm = StateMachine("sigma_verify", gateway_name="init")

    # --- Gateway ---
    vm.gateway(
        description=(
            "Initialize ΣVerify. Reports which non-Claude model providers are "
            "available (OpenAI, Google, Ollama local and cloud models) and why "
            "any others are not."
        ),
        params={},
    )

    # --- Actions ---
    vm.action(
        "get_models",
        description="List available external models and their status",
        from_states="*",
    )

    vm.action(
        "verify_finding",
        description=(
            "Verify a finding using an external model. Returns structured assessment "
            "(agree/disagree/partial/uncertain) with reasoning and provenance tag."
        ),
        from_states=["ready"],
        params={
            "finding": "string",
            "context": "string",
            "provider": "string",
            "model": "string",
        },
        required=["finding"],
    )

    vm.action(
        "cross_verify",
        description=(
            "Verify a finding across available external models simultaneously. "
            "Returns comparison of assessments with agreement level. "
            "Optionally specify providers to use a subset (comma-separated). "
            "Anthropic is excluded unless named explicitly."
        ),
        from_states=["ready"],
        params={
            "finding": "string",
            "context": "string",
            "providers": "string (comma-separated provider names, default: all non-Anthropic)",
        },
        required=["finding"],
    )

    vm.action(
        "challenge",
        description=(
            "Ask an external model to play devil's advocate against a specific claim. "
            "Returns counter-arguments, logical gaps, and vulnerability assessment. "
            "Defaults to reasoning tier (deeper analysis) — set tier='standard' for faster/cheaper."
        ),
        from_states=["ready"],
        params={
            "claim": "string",
            "evidence": "string",
            "provider": "string",
            "model": "string",
            "tier": "string (reasoning|standard, default: reasoning)",
        },
        required=["claim"],
    )

    vm.action(
        "check_quotas",
        description=(
            "Check quota and budget constraints for all providers (available and unavailable). "
            "Returns informational summary of rate limits, pricing, and known constraints. "
            "Run before batch verification to surface potential issues."
        ),
        from_states="*",
    )

    # --- Register handlers ---

    @vm.on_gateway
    def _init():
        return handle_init()

    @vm.on_action("get_models")
    def _get_models():
        return handle_get_models()

    @vm.on_action("verify_finding")
    def _verify(finding="", context="", provider="", model=""):
        return handle_verify_finding(finding, context, provider, model)

    @vm.on_action("cross_verify")
    def _cross(finding="", context="", providers=""):
        return handle_cross_verify(finding, context, providers)

    @vm.on_action("challenge")
    def _challenge(claim="", evidence="", provider="", model="", tier="reasoning"):
        return handle_challenge(claim, evidence, provider, model, tier)

    @vm.on_action("check_quotas")
    def _check_quotas():
        return handle_check_quotas()

    # Startup probe for operator visibility. handle_init starts `ollama serve` only
    # when SIGMA_VERIFY_AUTOSTART_OLLAMA=1 and the server isn't running.
    try:
        _probe = handle_init()
        _status = _probe.get("status", "unknown")
        _providers = _probe.get("available_providers", [])
        print(
            f"ΣVerify auto-ready: {_status} |providers={','.join(_providers) or 'none'}",
            file=sys.stderr,
        )
        if _probe.get("_state") == "ready":
            vm._initial_state_hint = "ready"
    except Exception as exc:  # noqa: BLE001 — probe must not break server startup
        print(f"ΣVerify auto-ready: probe-failed |error={exc}", file=sys.stderr)

    return vm
