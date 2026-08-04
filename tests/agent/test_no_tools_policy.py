"""Fail-closed unit coverage for the constrained no-tools policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.no_tools import (
    NoToolsConflictError,
    NoToolsInvariantError,
    NoToolsPayloadError,
    assert_no_tools_model_response,
    assert_no_tools_payload,
    validate_no_tools_environment,
)


def _agent() -> SimpleNamespace:
    return SimpleNamespace(
        no_tools=True,
        enabled_toolsets=[],
        disabled_toolsets=[],
        tools=[],
        valid_tool_names=set(),
        _context_engine_tool_names=set(),
        _memory_store=None,
        _memory_manager=None,
        _skip_mcp_refresh=True,
        _fallback_chain=[],
    )


@pytest.mark.parametrize(
    "payload, path",
    [
        ({"model": "fixture", "tools": []}, "tools"),
        (
            {
                "model": "fixture",
                "extra_body": {"parallel_tool_calls": False},
            },
            "extra_body.parallel_tool_calls",
        ),
        (
            {
                "model": "fixture",
                "provider_config": [{"function_call": {"name": "x"}}],
            },
            "provider_config[0].function_call",
        ),
    ],
)
def test_no_tools_payload_guard_recursively_rejects_provider_tool_fields(payload, path):
    with pytest.raises(NoToolsPayloadError) as exc_info:
        assert_no_tools_payload(_agent(), payload, phase="offline test")
    assert path in str(exc_info.value)


def test_no_tools_response_guard_rejects_tool_call_before_interpretation():
    response = SimpleNamespace(tool_calls=[{"id": "fixture"}], function_call=None)

    with pytest.raises(NoToolsInvariantError, match="provider response contained a tool call"):
        assert_no_tools_model_response(_agent(), response)


def test_no_tools_agent_mutation_fails_before_payload_dispatch():
    agent = _agent()
    agent.tools = [{"type": "function", "function": {"name": "hostile"}}]

    with pytest.raises(NoToolsInvariantError, match="tools"):
        assert_no_tools_payload(agent, {"model": "fixture"}, phase="offline test")


@pytest.mark.parametrize(
    "name, value",
    [
        ("HERMES_SAFE_MODE", "1"),
        ("HERMES_IGNORE_RULES", "true"),
        ("HERMES_TUI", "yes"),
        ("HERMES_KANBAN_TASK", "fixture-task"),
    ],
)
def test_no_tools_rejects_environment_that_widens_the_contract(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(NoToolsConflictError):
        validate_no_tools_environment()
