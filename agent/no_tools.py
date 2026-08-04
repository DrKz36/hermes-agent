"""Fail-closed policy helpers for the CLI ``--no-tools`` mode.

The mode is deliberately an internal runtime policy, not an environment
variable.  It is enabled only after argparse has accepted the explicit CLI
flag and is consulted by the agent bootstrap, agent construction, and the
provider dispatch seams.
"""

from __future__ import annotations

import os
import sys
from contextvars import ContextVar
from typing import Any, Mapping, Sequence


class NoToolsError(RuntimeError):
    """Base class for deterministic ``--no-tools`` policy failures."""


class NoToolsConflictError(NoToolsError):
    """Raised when ``--no-tools`` conflicts with another CLI policy."""


class NoToolsBootstrapError(NoToolsError):
    """Raised when a tool-capable bootstrap path ran too early."""


class NoToolsInvariantError(NoToolsError):
    """Raised when an agent no longer has an empty tool surface."""


class NoToolsPayloadError(NoToolsInvariantError):
    """Raised when a provider request would carry a tool-control field."""


_NO_TOOLS_BOOTSTRAP: ContextVar[bool] = ContextVar(
    "hermes_no_tools_bootstrap", default=False
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FORBIDDEN_PAYLOAD_FIELDS = frozenset(
    {
        "tools",
        "toolchoice",
        "functions",
        "functioncall",
        "paralleltoolcalls",
        "toolconfig",
        "functioncallingconfig",
    }
)
_NESTED_PROVIDER_OPTIONS = frozenset(
    {
        "extrabody",
        "additionalmodelrequestfields",
        "inferenceconfig",
        "provideroptions",
        "providerconfig",
    }
)


def _enabled_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _normalise_field_name(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def validate_no_tools_cli_args(args: Any) -> None:
    """Reject CLI combinations that would weaken the governance contract."""
    if not bool(getattr(args, "no_tools", False)):
        return

    conflicts: list[str] = []
    if bool(getattr(args, "safe_mode", False)):
        conflicts.append("--safe-mode")
    if _enabled_env("HERMES_SAFE_MODE"):
        conflicts.append("HERMES_SAFE_MODE")
    if bool(getattr(args, "ignore_rules", False)):
        conflicts.append("--ignore-rules")
    if _enabled_env("HERMES_IGNORE_RULES"):
        conflicts.append("HERMES_IGNORE_RULES")
    if bool(getattr(args, "tui", False)) or _enabled_env("HERMES_TUI"):
        conflicts.append("TUI")
    if getattr(args, "oneshot", None):
        conflicts.append("--oneshot")
    if getattr(args, "resume", None) or getattr(args, "continue_last", None):
        conflicts.append("--resume/--continue")
    if getattr(args, "profile", None):
        conflicts.append("--profile")
    if getattr(args, "image", None):
        conflicts.append("--image")
    if bool(getattr(args, "worktree", False)):
        conflicts.append("--worktree")
    if bool(getattr(args, "yolo", False)):
        conflicts.append("--yolo")
    toolsets = getattr(args, "toolsets", None)
    if toolsets:
        conflicts.append("--toolsets")
    if getattr(args, "skills", None):
        conflicts.append("--skills")

    command = getattr(args, "command", None)
    if command not in {None, "chat"}:
        conflicts.append(f"command={command}")

    if conflicts:
        raise NoToolsConflictError(
            "--no-tools is limited to a fresh, text-only classic chat with "
            "governance rules enabled; incompatible: " + ", ".join(conflicts)
        )


def activate_no_tools_bootstrap() -> None:
    """Mark the current launch as no-tools before importing the agent runtime.

    ``model_tools`` has a module-level extension-discovery side effect.  If it
    or any equivalent extension/MCP/hook bootstrap is already imported, this
    process cannot prove that discovery did not precede the explicit policy.
    Fail before a provider request instead of attempting to clean shared
    global state belonging to another session.
    """
    loaded: list[str] = []
    if "model_tools" in sys.modules:
        # model_tools performs its own registry/bootstrap work at import time.
        loaded.append("model_tools")

    plugin_module = sys.modules.get("hermes_cli.plugins")
    plugin_manager = getattr(plugin_module, "_plugin_manager", None)
    if bool(getattr(plugin_manager, "_discovered", False)):
        loaded.append("hermes_cli.plugins")

    mcp_module = sys.modules.get("tools.mcp_tool")
    if bool(getattr(mcp_module, "_mcp_tool_server_names", None)):
        loaded.append("tools.mcp_tool")

    hook_module = sys.modules.get("agent.shell_hooks")
    if bool(getattr(hook_module, "_registered", None)):
        loaded.append("agent.shell_hooks")

    provider_registry = sys.modules.get("providers")
    if provider_registry is not None and bool(
        getattr(provider_registry, "_discovered", False)
    ):
        loaded.append("providers (profile registry)")
    if loaded:
        raise NoToolsBootstrapError(
            "--no-tools was activated after extension-capable bootstrap "
            f"({', '.join(loaded)}); refusing an unprovable tool surface"
        )
    _NO_TOOLS_BOOTSTRAP.set(True)


def bootstrap_no_tools_from_argv(argv: Sequence[str] | None = None) -> bool:
    """Arm the internal policy before the full CLI parser imports extensions.

    The normal argparse validation remains authoritative and runs immediately
    afterwards.  This deliberately tiny lexical pass exists only because the
    full parser registers optional command modules before it can hand us a
    Namespace.  It has no environment-variable side channel and cannot make
    an invalid invocation runnable: argparse still rejects incompatible flags
    or non-chat commands before any agent dispatch.
    """
    tokens = sys.argv[1:] if argv is None else argv
    if "--no-tools" not in tokens:
        return False
    activate_no_tools_bootstrap()
    return True


def prepare_no_tools_cli_bootstrap(args: Any) -> None:
    """Validate then activate the mode at the CLI's post-parse boundary."""
    if not bool(getattr(args, "no_tools", False)):
        return
    validate_no_tools_cli_args(args)
    activate_no_tools_bootstrap()


def no_tools_bootstrap_active() -> bool:
    """Whether the current import/initialisation path is policy-constrained."""
    return _NO_TOOLS_BOOTSTRAP.get()


def assert_no_tools_agent_invariant(agent: Any, *, phase: str) -> None:
    """Ensure a no-tools agent has no schema, names, memory, or MCP refresh."""
    if not bool(getattr(agent, "no_tools", False)):
        return

    violations: list[str] = []
    if getattr(agent, "enabled_toolsets", None):
        violations.append("enabled_toolsets")
    if getattr(agent, "disabled_toolsets", None):
        violations.append("disabled_toolsets")
    if getattr(agent, "tools", None):
        violations.append("tools")
    if getattr(agent, "valid_tool_names", None):
        violations.append("valid_tool_names")
    if getattr(agent, "_context_engine_tool_names", None):
        violations.append("context_engine_tools")
    if getattr(agent, "_memory_store", None) is not None:
        violations.append("memory_store")
    if getattr(agent, "_memory_manager", None) is not None:
        violations.append("memory_manager")
    if not bool(getattr(agent, "_skip_mcp_refresh", False)):
        violations.append("mcp_refresh")
    if getattr(agent, "_fallback_chain", None):
        violations.append("fallback_chain")

    if violations:
        raise NoToolsInvariantError(
            f"--no-tools invariant failed during {phase}: {', '.join(violations)}"
        )


def _find_forbidden_payload_fields(payload: Mapping[str, Any]) -> list[str]:
    found: list[str] = []

    def visit(mapping: Mapping[str, Any], path: str) -> None:
        for key, value in mapping.items():
            normalized = _normalise_field_name(key)
            item_path = f"{path}.{key}" if path else str(key)
            if normalized in _FORBIDDEN_PAYLOAD_FIELDS:
                found.append(item_path)
            if normalized in _NESTED_PROVIDER_OPTIONS and isinstance(value, Mapping):
                visit(value, item_path)

    visit(payload, "")
    return found


def assert_no_tools_payload(agent: Any, payload: Mapping[str, Any], *, phase: str) -> None:
    """Fail before network when a no-tools request carries tool controls."""
    if not bool(getattr(agent, "no_tools", False)):
        return
    assert_no_tools_agent_invariant(agent, phase=phase)
    fields = _find_forbidden_payload_fields(payload)
    if fields:
        raise NoToolsPayloadError(
            f"--no-tools payload rejected during {phase}: {', '.join(fields)}"
        )


def assert_no_tools_model_response(agent: Any, assistant_message: Any) -> None:
    """Refuse a provider response that nevertheless attempts a tool call."""
    if not bool(getattr(agent, "no_tools", False)):
        return
    assert_no_tools_agent_invariant(agent, phase="provider response")
    if isinstance(assistant_message, Mapping):
        tool_calls = assistant_message.get("tool_calls") or assistant_message.get("function_call")
    else:
        tool_calls = getattr(assistant_message, "tool_calls", None) or getattr(
            assistant_message, "function_call", None
        )
    if tool_calls:
        raise NoToolsInvariantError(
            "--no-tools provider response contained a tool call"
        )
