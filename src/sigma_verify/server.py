"""ΣVerify MCP Server — cross-model verification for Claude Code agents.

Exposes OpenAI, Google, and Ollama-served (local and cloud) models as MCP
tools so agents can check findings against a model other than Claude.

Usage:
    sigma-verify                    # reads API keys / overrides from env vars
"""

from __future__ import annotations

from .machine import build_machine


def main():
    machine = build_machine()

    from hateoas_agent.mcp_server import serve

    serve(machine, name="sigma-verify")


if __name__ == "__main__":
    main()
