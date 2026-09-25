"""Handlers for sigma-verify MCP tools."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import subprocess
import time
from typing import Any

from .clients import (
    AUTOSTART_OLLAMA_ENV,
    OLLAMA_CLIENTS,
    AnthropicClient,
    GeminiClient,
    OpenAIClient,
    VerificationError,
    VerificationResult,
    _OllamaClientBase,
    classify_error,
    probe_ollama,
)

# Union type for all client classes
_AnyClient = OpenAIClient | GeminiClient | _OllamaClientBase | AnthropicClient

# Timeout per individual provider call (seconds)
CALL_TIMEOUT = 90  # seconds per provider call

OLLAMA_URL = "http://localhost:11434"
OLLAMA_STARTUP_TIMEOUT = 10  # seconds to wait after starting ollama

# Claude verifying Claude is not cross-model verification, so Anthropic is left
# out of every default selection. Naming it explicitly, or setting this env var
# to 1, opts back in.
ALLOW_ANTHROPIC_ENV = "SIGMA_VERIFY_ALLOW_ANTHROPIC"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _anthropic_allowed() -> bool:
    return _env_flag(ALLOW_ANTHROPIC_ENV)


def _autostart_allowed() -> bool:
    """Starting `ollama serve` on the user's machine is opt-in."""
    return _env_flag(AUTOSTART_OLLAMA_ENV)


def _default_pool(clients: dict[str, _AnyClient]) -> dict[str, _AnyClient]:
    """Providers eligible for default (unnamed) selection."""
    if _anthropic_allowed():
        return dict(clients)
    return {name: c for name, c in clients.items() if name != "anthropic"}


def _is_ollama_running(refresh: bool = False) -> bool:
    """Check if the local Ollama server is reachable (shares the clients' probe cache)."""
    return probe_ollama(OLLAMA_URL, refresh=refresh) is not None


def _ollama_configured() -> bool:
    """True if any Ollama-backed provider is configured to use an Ollama server."""
    return any(cls()._uses_ollama for cls in OLLAMA_CLIENTS)


def _start_ollama() -> bool:
    """Attempt to start Ollama and wait for it to become available."""
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False

    deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if _is_ollama_running(refresh=True):
            return True
        time.sleep(0.5)
    return False


def _ensure_ollama(refresh: bool = False) -> dict[str, str]:
    """Check Ollama, starting it only if SIGMA_VERIFY_AUTOSTART_OLLAMA is set.

    Returns status dict for init / pre-flight reporting.
    """
    if _is_ollama_running(refresh=refresh):
        return {"status": "ok", "message": "Ollama is running."}

    if not _autostart_allowed():
        return {
            "status": "unavailable",
            "message": (
                f"Ollama is not running at {OLLAMA_URL}. Start it with `ollama serve` "
                f"(or set {AUTOSTART_OLLAMA_ENV}=1)."
            ),
        }

    if _start_ollama():
        return {
            "status": "started",
            "message": "Ollama was not running — started automatically.",
        }

    return {
        "status": "unavailable",
        "message": (
            f"Ollama is not running at {OLLAMA_URL} and could not be started. "
            "Start it with `ollama serve` (or the Ollama app)."
        ),
    }


def _has_ollama_clients(clients: dict[str, _AnyClient]) -> bool:
    """Check if any client is backed by Ollama (localhost:11434)."""
    for client in clients.values():
        base_url = getattr(client, "_base_url", "")
        if "localhost:11434" in base_url:
            return True
    return False


def _get_clients() -> dict[str, _AnyClient]:
    """Instantiate all configured clients."""
    clients: dict[str, _AnyClient] = {}

    # Non-Ollama clients
    oai = OpenAIClient()
    if oai.available:
        clients["openai"] = oai
    gem = GeminiClient()
    if gem.available:
        clients["google"] = gem

    # All Ollama-backed clients
    for cls in OLLAMA_CLIENTS:
        instance = cls()
        if instance.available:
            clients[cls.PROVIDER_NAME] = instance

    # Optional non-Ollama
    anth = AnthropicClient()
    if anth.available:
        clients["anthropic"] = anth

    return clients


def _format_result(result: VerificationResult) -> str:
    """Format a single verification result."""
    return json.dumps(result.to_dict(), indent=2)


def _format_cross_results(
    results: list[dict[str, Any]],
    pre_flight: dict[str, Any] | None = None,
) -> str:
    """Format cross-verification results with comparison and failure tracking."""
    successes = [r for r in results if r.get("status") != "failed"]
    failures = [r for r in results if r.get("status") == "failed"]

    output: dict[str, Any] = {
        "results": results,
        "count": len(results),
        "succeeded": len(successes),
        "failed": len(failures),
    }

    # Include pre-flight info
    if pre_flight:
        output["pre_flight"] = pre_flight

    # Compute agreement only among successful results
    if successes:
        assessments = [r.get("assessment", "unknown") for r in successes]
        if len(set(assessments)) == 1:
            output["agreement"] = "unanimous"
        elif len(set(assessments)) == len(assessments):
            output["agreement"] = "none"
        else:
            output["agreement"] = "partial"
    else:
        output["agreement"] = "no_data"

    # Coverage warning
    if failures:
        output["coverage"] = "partial"
        output["coverage_warning"] = (
            f"{len(failures)}/{len(results)} providers failed. "
            "Verification is incomplete — do NOT treat as full cross-model validation."
        )
        output["failure_sigma"] = [r.get("sigma", "") for r in failures if r.get("sigma")]
    else:
        output["coverage"] = "full"

    return json.dumps(output, indent=2)


async def _call_provider_async(
    name: str,
    client: _AnyClient,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    tool_name: str,
    finding_brief: str,
) -> dict[str, Any]:
    """Call a single provider method with timeout, returning a result dict.

    Wraps the synchronous client method in run_in_executor with an
    asyncio.wait_for timeout. On timeout or exception, returns a
    VerificationError dict instead of raising.
    """
    loop = asyncio.get_running_loop()

    try:
        method = getattr(client, method_name)
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: method(*args, **kwargs)),
            timeout=CALL_TIMEOUT,
        )
        # verify() returns VerificationResult, challenge() returns dict
        if isinstance(result, VerificationResult):
            return result.to_dict()
        return result
    except asyncio.TimeoutError:
        error = VerificationError(
            model=client.model,
            provider=name,
            error_class="timeout",
            error_detail=f"Provider call timed out after {CALL_TIMEOUT}s",
            tool_attempted=tool_name,
            finding_brief=finding_brief[:80],
            provenance_tag=f"|source:external-{name}-{client.model}|",
        )
        return error.to_dict()
    except Exception as e:
        error = VerificationError(
            model=client.model,
            provider=name,
            error_class=classify_error(e),
            error_detail=str(e),
            tool_attempted=tool_name,
            finding_brief=finding_brief[:80],
            provenance_tag=f"|source:external-{name}-{client.model}|",
        )
        return error.to_dict()


def _run_concurrent(
    clients: dict[str, _AnyClient],
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    tool_name: str = "cross_verify",
    finding_brief: str = "",
) -> list[dict[str, Any]]:
    """Run a provider method concurrently across all clients.

    Bridge from sync handler context to async execution. Two calling
    contexts exist and need different loop handling:

    - CLI / tests: no event loop on this thread — asyncio.run() owns one.
    - MCP server: the handler is invoked on the event-loop thread, where
      asyncio.run() raises "cannot be called from a running event loop".
      That crash surfaced to agents as an opaque "internal error" on every
      cross_verify. In that case
      the whole gather runs on a fresh thread with its own loop; provider
      calls already execute in worker threads via run_in_executor, so this
      adds one coordinating thread, not per-provider threads.
    """
    if kwargs is None:
        kwargs = {}

    async def _gather_all() -> list[dict[str, Any]]:
        tasks = [
            _call_provider_async(name, client, method_name, args, kwargs, tool_name, finding_brief)
            for name, client in clients.items()
        ]
        return list(await asyncio.gather(*tasks))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_gather_all())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _gather_all()).result()


def _unavailable_reasons() -> dict[str, str]:
    """Why each unavailable provider can't be used (for init reporting)."""
    reasons: dict[str, str] = {}
    if not OpenAIClient().available:
        reasons["openai"] = "OPENAI_API_KEY not set."
    if not GeminiClient().available:
        reasons["google"] = "GOOGLE_AI_API_KEY not set."
    for cls in OLLAMA_CLIENTS:
        reason = cls().unavailable_reason
        if reason:
            reasons[cls.PROVIDER_NAME] = reason
    return reasons


def handle_init() -> dict[str, Any]:
    """Gateway: check which providers are available."""
    # Re-probe Ollama (auto-starting it only if opted in) if any provider uses it.
    ollama_status = None
    if _ollama_configured():
        ollama_status = _ensure_ollama(refresh=True)

    clients = _get_clients()
    if ollama_status is not None and ollama_status["status"] == "unavailable":
        # Drop Ollama-backed clients that can't reach the server
        clients = {
            name: c
            for name, c in clients.items()
            if "localhost:11434" not in getattr(c, "_base_url", "")
        }

    pool = _default_pool(clients)
    excluded = sorted(set(clients) - set(pool))
    available = list(pool.keys())
    models = {name: c.model for name, c in pool.items()}
    unavailable = _unavailable_reasons()

    if not available:
        result: dict[str, Any] = {
            "status": "no_providers",
            "message": (
                "No cross-model providers available. Set OPENAI_API_KEY and/or "
                "GOOGLE_AI_API_KEY, or run Ollama (`ollama serve`) with a pulled local "
                "model or `ollama signin` for cloud models. Without providers, "
                "findings simply stay untagged."
            ),
            "unavailable_providers": unavailable,
            "ollama": ollama_status,
            "_state": "unconfigured",
        }
        if excluded:
            result["excluded_by_default"] = excluded
        return result

    # Report reasoning models
    reasoning = {}
    cloud: list[str] = []
    for name, c in pool.items():
        if isinstance(c, OpenAIClient):
            reasoning[name] = c.reasoning_model
        elif isinstance(c, GeminiClient):
            reasoning[name] = f"{c.model} (thinking mode)"
        elif isinstance(c, _OllamaClientBase):
            reasoning[name] = f"{c.model} (standard only — {c.REASONING_NOTE})"
            if c.is_cloud:
                cloud.append(name)
        elif isinstance(c, AnthropicClient):
            reasoning[name] = f"{c.model} (extended thinking available)"

    result = {
        "status": "ready",
        "available_providers": available,
        "models": models,
        "reasoning_models": reasoning,
        "unavailable_providers": unavailable,
        "message": f"ΣVerify ready. Providers: {', '.join(available)}",
        "_state": "ready",
    }
    if cloud:
        result["ollama_cloud"] = {
            "providers": cloud,
            "note": (
                "Ollama cloud models require `ollama signin`; paid-tier models also "
                "need credits. Availability here only confirms Ollama is running — "
                "a missing signin surfaces as an auth-error on the first call."
            ),
        }
    if excluded:
        result["excluded_by_default"] = excluded
        result["excluded_note"] = (
            "Anthropic is excluded from default selections (Claude verifying Claude "
            f"is not cross-model). Name it explicitly or set {ALLOW_ANTHROPIC_ENV}=1."
        )
    if ollama_status is not None:
        result["ollama"] = ollama_status
    return result


def handle_get_models() -> str:
    """List available models and their status."""
    clients = _get_clients()
    if not clients:
        return json.dumps(
            {
                "available": [],
                "message": (
                    "No providers available. Set OPENAI_API_KEY and/or GOOGLE_AI_API_KEY, "
                    "or run Ollama. Call init for per-provider reasons."
                ),
            }
        )

    pool = _default_pool(clients)
    models = []
    for name, client in clients.items():
        models.append(
            {
                "provider": name,
                "model": client.model,
                "available": client.available,
                "default_selection": name in pool,
            }
        )
    return json.dumps({"available": models}, indent=2)


def _select_provider(clients: dict[str, _AnyClient], provider: str) -> tuple[str, _AnyClient] | str:
    """Pick the named provider, or the first default-eligible one.

    Returns (name, client), or a JSON error string.
    """
    if provider:
        if provider in clients:
            return provider, clients[provider]
        return json.dumps(
            {"error": f"Provider '{provider}' not available. Available: {list(clients.keys())}"}
        )
    pool = _default_pool(clients)
    if not pool:
        return json.dumps(
            {
                "error": (
                    "No default providers available. Anthropic is excluded unless named "
                    f"explicitly (provider='anthropic') or {ALLOW_ANTHROPIC_ENV}=1."
                )
            }
        )
    return next(iter(pool.items()))


def handle_verify_finding(
    finding: str = "",
    context: str = "",
    provider: str = "",
    model: str = "",
) -> str:
    """Verify a finding using a specific external model."""
    if not finding:
        return json.dumps({"error": "finding is required"})

    clients = _get_clients()
    if not clients:
        return json.dumps({"error": "No providers available"})

    selected = _select_provider(clients, provider)
    if isinstance(selected, str):
        return selected
    name, client = selected

    # Override model if specified
    if model:
        client.model = model

    try:
        result = client.verify(finding, context)
        return _format_result(result)
    except Exception as e:
        error = VerificationError(
            model=client.model,
            provider=name,
            error_class=classify_error(e),
            error_detail=str(e),
            tool_attempted="verify_finding",
            finding_brief=finding[:80],
            provenance_tag=f"|source:external-{name}-{client.model}|",
        )
        return json.dumps(error.to_dict(), indent=2)


def _pre_flight_check(
    clients: dict[str, _AnyClient],
) -> dict[str, Any]:
    """Pre-flight quota check and cost estimate before cross-verification.

    Returns dict with:
      - ollama: Ollama status (if local providers are in use)
      - quota_warnings: list of provider warnings
      - cost_estimate_usd: estimated cost for this call
      - providers_called: count of providers that will be called
    """
    # Check Ollama if any local providers are in the client list
    ollama_status = None
    if _has_ollama_clients(clients):
        ollama_status = _ensure_ollama()
        if ollama_status["status"] == "unavailable":
            # Remove Ollama-backed clients that can't reach the server
            clients = {
                name: c
                for name, c in clients.items()
                if "localhost:11434" not in getattr(c, "_base_url", "")
            }

    # Rough per-call cost estimates (~1K tokens in + out) for per-token API providers.
    # Ollama-backed calls are not estimated: local models are free, and cloud usage
    # is billed (or free-tier limited) by Ollama on the user's account.
    cost_map = {
        "openai": 0.005,
        "google": 0.002,
        "anthropic": 0.003,
    }

    warnings = []
    total_cost = 0.0

    for name, client in clients.items():
        quota = client.check_quota()
        if quota["status"] == "warning":
            warnings.append(
                {
                    "provider": name,
                    "message": quota["message"],
                }
            )
        # Ollama-backed providers are not estimated (see above)
        if "localhost:11434" in getattr(client, "_base_url", ""):
            continue
        total_cost += cost_map.get(name, 0.005)

    result: dict[str, Any] = {
        "quota_warnings": warnings,
        "cost_estimate_usd": round(total_cost, 4),
        "providers_called": len(clients),
    }
    if ollama_status is not None:
        result["ollama"] = ollama_status
    return result


def handle_cross_verify(
    finding: str = "",
    context: str = "",
    providers: str = "",
) -> str:
    """Verify a finding across available models concurrently and compare.

    providers: comma-separated list of provider names to use (default: all available
               except Anthropic). Example: "llama,gemma,openai" to use only those 3.
               Anthropic is included only when named here or opted in via env.
    """
    if not finding:
        return json.dumps({"error": "finding is required"})

    clients = _get_clients()
    if not clients:
        return json.dumps({"error": "No providers available"})

    # Filter to requested providers if specified
    missing: set[str] = set()
    if providers:
        requested = {p.strip() for p in providers.split(",")}
        missing = requested - set(clients.keys())
        clients = {k: v for k, v in clients.items() if k in requested}
        if not clients:
            return json.dumps(
                {
                    "error": (
                        f"None of the requested providers are available. "
                        f"Requested: {sorted(requested)}, "
                        f"Available: {sorted(_get_clients().keys())}"
                    )
                }
            )
    else:
        clients = _default_pool(clients)
        if not clients:
            return json.dumps(
                {
                    "error": (
                        "No default providers available. Anthropic is excluded unless "
                        f"named explicitly or {ALLOW_ANTHROPIC_ENV}=1."
                    )
                }
            )

    # Pre-flight quota check and cost estimate
    pre_flight = _pre_flight_check(clients)
    if missing:
        pre_flight["unavailable_requested"] = sorted(missing)

    results = _run_concurrent(
        clients,
        method_name="verify",
        args=(finding, context),
        tool_name="cross_verify",
        finding_brief=finding,
    )

    return _format_cross_results(results, pre_flight=pre_flight)


def handle_check_quotas() -> str:
    """Check quota/budget status for all providers (including unavailable)."""
    all_clients: dict[str, _AnyClient] = {}
    all_clients["openai"] = OpenAIClient()
    all_clients["google"] = GeminiClient()
    for cls in OLLAMA_CLIENTS:
        all_clients[cls.PROVIDER_NAME] = cls()
    all_clients["anthropic"] = AnthropicClient()

    quotas = []
    for name, client in all_clients.items():
        quotas.append(client.check_quota())

    available_count = sum(1 for q in quotas if q["status"] != "unavailable")
    warning_count = sum(1 for q in quotas if q["status"] == "warning")

    summary: dict[str, Any] = {
        "quotas": quotas,
        "providers_available": available_count,
        "providers_total": len(quotas),
        "warnings": warning_count,
    }

    if available_count == 0:
        summary["message"] = (
            "No providers configured. Set API keys to enable cross-model verification."
        )
    elif warning_count > 0:
        summary["message"] = (
            f"{available_count}/{len(quotas)} providers available, "
            f"{warning_count} with quota warnings. Review before batch runs."
        )
    else:
        summary["message"] = (
            f"{available_count}/{len(quotas)} providers available. No quota warnings."
        )

    return json.dumps(summary, indent=2)


def handle_challenge(
    claim: str = "",
    evidence: str = "",
    provider: str = "",
    model: str = "",
    tier: str = "reasoning",
) -> str:
    """Ask an external model to challenge a specific claim.

    tier: "reasoning" (default) uses reasoning models for deeper analysis,
          "standard" uses the regular chat model.
    """
    if not claim:
        return json.dumps({"error": "claim is required"})

    clients = _get_clients()
    if not clients:
        return json.dumps({"error": "No providers available"})

    selected = _select_provider(clients, provider)
    if isinstance(selected, str):
        return selected
    name, client = selected

    if model:
        client.model = model

    try:
        result = client.challenge(claim, evidence, tier=tier)
        result["status"] = "success"
        return json.dumps(result, indent=2)
    except Exception as e:
        error = VerificationError(
            model=client.model,
            provider=name,
            error_class=classify_error(e),
            error_detail=str(e),
            tool_attempted="challenge",
            finding_brief=claim[:80],
            provenance_tag=f"|source:external-{name}-{client.model}|",
        )
        return json.dumps(error.to_dict(), indent=2)
