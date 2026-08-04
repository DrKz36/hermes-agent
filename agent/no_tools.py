"""Fail-closed policy for the machine-only ``hermes chat --no-tools`` path.

This policy intentionally owns a very small surface:

``hermes chat --no-tools -Q -q <prompt>``

It is not a general "hide tool output" option and does not make the
interactive CLI safe.  The policy must be armed before agent-tool discovery,
then remains attached to the synchronous agent instance through provider
dispatch.
"""

from __future__ import annotations

import os
import sys
from contextvars import ContextVar
from typing import Any, Mapping, Sequence


class NoToolsError(RuntimeError):
    """Base error for deterministic ``--no-tools`` policy failures."""


class NoToolsConflictError(NoToolsError):
    """The CLI invocation is outside the machine-only no-tools contract."""


class NoToolsBootstrapError(NoToolsError):
    """An extension-capable bootstrap ran before the policy was armed."""


class NoToolsInvariantError(NoToolsError):
    """The agent gained a capability that the policy forbids."""


class NoToolsPayloadError(NoToolsInvariantError):
    """A provider payload would carry a tool-control field."""


_BOOTSTRAP_ACTIVE: ContextVar[bool] = ContextVar(
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
def _enabled_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _normalise_field_name(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def argv_requests_no_tools(argv: Sequence[str] | None = None) -> bool:
    """Return whether raw argv selects or abbreviates ``--no-tools``.

    ``argparse`` accepts unique long-option prefixes by default.  Recognise
    those prefixes here too so an abbreviated spelling reaches the lexical
    rejection path *before* optional parser/plugin discovery.  It is still
    rejected by :func:`validate_no_tools_raw_argv`; this helper only chooses
    the safe bootstrap boundary.
    """
    tokens = sys.argv[1:] if argv is None else argv
    return any(
        token == "--no-tools"
        or token.startswith("--no-tools=")
        or (token.startswith("--no-") and "--no-tools".startswith(token))
        for token in tokens
    )


def argv_requests_profile(argv: Sequence[str] | None = None) -> bool:
    """Return whether the raw invocation supplies Hermes' pre-parse profile flag."""
    tokens = sys.argv[1:] if argv is None else argv
    return any(
        token in {"--profile", "-p"} or token.startswith("--profile=")
        for token in tokens
    )


def reject_no_tools_profile_override(argv: Sequence[str] | None = None) -> None:
    """Reject a profile override before it can redirect ``HERMES_HOME``.

    ``--profile`` is consumed before argparse by :mod:`hermes_cli.main`, so
    this check belongs at that same early boundary.  A pre-selected default
    home is not reclassified as a CLI profile override here.
    """
    if argv_requests_no_tools(argv) and argv_requests_profile(argv):
        raise NoToolsConflictError(
            "--no-tools is limited to the selected home; --profile is not allowed"
        )


def validate_no_tools_environment() -> None:
    """Reject process state that would widen the machine-only envelope."""
    conflicts: list[str] = []
    if _enabled_env("HERMES_SAFE_MODE"):
        conflicts.append("HERMES_SAFE_MODE")
    if _enabled_env("HERMES_IGNORE_RULES"):
        conflicts.append("HERMES_IGNORE_RULES")
    if _enabled_env("HERMES_TUI"):
        conflicts.append("HERMES_TUI")
    if os.environ.get("HERMES_KANBAN_TASK") or os.environ.get(
        "HERMES_KANBAN_GOAL_MODE"
    ):
        conflicts.append("kanban environment")
    if conflicts:
        raise NoToolsConflictError(
            "--no-tools rejects incompatible environment: " + ", ".join(conflicts)
        )


def validate_no_tools_cli_args(args: Any) -> None:
    """Require exactly the synchronous machine chat envelope.

    Model/provider selection remains the responsibility of the already selected
    Hermes home.  Keeping this envelope small is deliberate: it prevents a
    future option from quietly selecting an interactive, image, resume, skill,
    or extension-capable path.
    """
    if not bool(getattr(args, "no_tools", False)):
        return

    validate_no_tools_environment()
    conflicts: list[str] = []
    if getattr(args, "command", None) != "chat":
        conflicts.append("command must be chat")
    if not bool(getattr(args, "quiet", False)):
        conflicts.append("requires -Q/--quiet")
    if not isinstance(getattr(args, "query", None), str) or not args.query:
        conflicts.append("requires -q/--query")
    if bool(getattr(args, "safe_mode", False)):
        conflicts.append("--safe-mode")
    if bool(getattr(args, "ignore_rules", False)):
        conflicts.append("--ignore-rules")
    if bool(getattr(args, "ignore_user_config", False)):
        conflicts.append("--ignore-user-config")
    if bool(getattr(args, "tui", False)) or bool(getattr(args, "cli", False)):
        conflicts.append("TUI/interactive interface selector")
    if getattr(args, "image", None):
        conflicts.append("--image")
    if getattr(args, "resume", None) or getattr(args, "continue_last", None):
        conflicts.append("--resume/--continue")
    if getattr(args, "toolsets", None):
        conflicts.append("--toolsets")
    if getattr(args, "skills", None):
        conflicts.append("--skills")
    if bool(getattr(args, "worktree", False)) or bool(getattr(args, "w", False)):
        conflicts.append("--worktree")
    if bool(getattr(args, "yolo", False)) or bool(getattr(args, "accept_hooks", False)):
        conflicts.append("yolo/hooks")
    if bool(getattr(args, "checkpoints", False)) or bool(getattr(args, "pass_session_id", False)):
        conflicts.append("checkpoints/session prompt")
    if getattr(args, "source", None):
        conflicts.append("--source")
    if bool(getattr(args, "verbose", False)) or getattr(args, "reasoning", None):
        conflicts.append("verbose/reasoning override")
    if getattr(args, "max_turns", None) is not None:
        conflicts.append("--max-turns")
    if getattr(args, "model", None) or getattr(args, "provider", None):
        conflicts.append("model/provider override")
    if getattr(args, "oneshot", None):
        conflicts.append("--oneshot")
    if conflicts:
        raise NoToolsConflictError(
            "--no-tools supports only `hermes chat --no-tools -Q -q <prompt>`; "
            "incompatible: " + ", ".join(conflicts)
        )


def validate_no_tools_raw_argv(argv: Sequence[str] | None = None) -> None:
    """Reject a root/global spelling or an unparsed extension flag.

    Argparse's inherited flags deliberately make ``--no-tools`` available at
    several parser levels.  The public policy has one spelling, however: the
    first CLI token must be ``chat`` and every later token must belong to the
    small machine-only envelope.  Keeping this lexical check separate from the
    parsed-argument validation makes a new parser option fail closed until it
    is consciously admitted here.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not argv_requests_no_tools(tokens):
        return
    if not tokens or tokens[0] != "chat":
        raise NoToolsConflictError(
            "--no-tools is available only as `hermes chat --no-tools -Q -q <prompt>`"
        )

    seen_no_tools = 0
    seen_quiet = 0
    seen_query = 0
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--no-tools":
            seen_no_tools += 1
            index += 1
            continue
        if token == "-Q":
            seen_quiet += 1
            index += 1
            continue
        if token == "-q":
            seen_query += 1
            if (
                index + 1 >= len(tokens)
                or not tokens[index + 1]
                or tokens[index + 1].startswith("-")
            ):
                raise NoToolsConflictError("--no-tools requires a non-empty -q/--query")
            index += 2
            continue
        if token.startswith("-"):
            raise NoToolsConflictError(
                f"--no-tools rejects unapproved option: {token}"
            )
        raise NoToolsConflictError(
            "--no-tools accepts its prompt only through -q/--query"
        )

    if seen_no_tools != 1 or seen_quiet != 1 or seen_query != 1:
        raise NoToolsConflictError(
            "--no-tools requires exactly one --no-tools, -Q/--quiet, and -q/--query"
        )


def no_tools_raw_query(argv: Sequence[str] | None = None) -> str:
    """Return the single prompt after validating the machine-only argv.

    This lets the CLI enter its constrained path without constructing the full
    command parser, whose optional subcommands can import extension modules.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    validate_no_tools_raw_argv(tokens)
    validate_no_tools_environment()
    for index, token in enumerate(tokens):
        if token == "-q":
            return str(tokens[index + 1])
    # ``validate_no_tools_raw_argv`` guarantees this branch is unreachable.
    raise NoToolsConflictError("--no-tools requires a non-empty -q/--query")


def activate_no_tools_bootstrap() -> None:
    """Arm the policy before the agent runtime imports capability registries."""
    if _BOOTSTRAP_ACTIVE.get():
        return

    loaded: list[str] = []
    if "model_tools" in sys.modules:
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
    provider_module = sys.modules.get("providers")
    if bool(getattr(provider_module, "_discovered", False)):
        loaded.append("providers")
    if loaded:
        raise NoToolsBootstrapError(
            "--no-tools was armed after extension-capable bootstrap: "
            + ", ".join(loaded)
        )
    _BOOTSTRAP_ACTIVE.set(True)


def bootstrap_no_tools_from_argv(argv: Sequence[str] | None = None) -> bool:
    """Arm the internal policy before optional parser extensions can load."""
    if not argv_requests_no_tools(argv):
        return False
    activate_no_tools_bootstrap()
    return True


def prepare_no_tools_cli_bootstrap(args: Any) -> None:
    """Validate parsed args and arm the policy at the command boundary."""
    if not bool(getattr(args, "no_tools", False)):
        return
    validate_no_tools_raw_argv()
    validate_no_tools_cli_args(args)
    activate_no_tools_bootstrap()


def no_tools_bootstrap_active() -> bool:
    """Whether this synchronous launch is under the no-tools policy."""
    return _BOOTSTRAP_ACTIVE.get()


def assert_no_tools_agent_invariant(agent: Any, *, phase: str) -> None:
    """Fail closed if a constrained agent contains an executable capability."""
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
            f"--no-tools invariant failed during {phase}: " + ", ".join(violations)
        )


def _find_forbidden_payload_fields(payload: Mapping[str, Any]) -> list[str]:
    found: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized = _normalise_field_name(key)
                item_path = f"{path}.{key}" if path else str(key)
                if normalized in _FORBIDDEN_PAYLOAD_FIELDS:
                    found.append(item_path)
                visit(nested, item_path)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(payload, "")
    return found


def assert_no_tools_payload(agent: Any, payload: Mapping[str, Any], *, phase: str) -> None:
    """Stop before network if the final provider payload contains tool controls."""
    if not bool(getattr(agent, "no_tools", False)):
        return
    assert_no_tools_agent_invariant(agent, phase=phase)
    fields = _find_forbidden_payload_fields(payload)
    if fields:
        raise NoToolsPayloadError(
            f"--no-tools payload rejected during {phase}: " + ", ".join(fields)
        )


def assert_no_tools_model_response(agent: Any, assistant_message: Any) -> None:
    """Reject a provider tool call rather than interpreting or retrying it."""
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
        raise NoToolsInvariantError("--no-tools provider response contained a tool call")
