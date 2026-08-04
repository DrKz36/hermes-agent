"""CLI parsing and startup boundaries for ``hermes chat --no-tools``."""

from __future__ import annotations

import os
import sys
import types
from argparse import Namespace

import pytest


def _args(**overrides):
    values = {
        "no_tools": True,
        "command": "chat",
        "safe_mode": False,
        "ignore_rules": False,
        "tui": False,
        "oneshot": None,
        "toolsets": None,
        "skills": None,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture(autouse=True)
def _clean_no_tools_env(monkeypatch):
    for name in (
        "HERMES_IGNORE_RULES",
        "HERMES_SAFE_MODE",
        "HERMES_TUI",
        "HERMES_KANBAN_TASK",
    ):
        monkeypatch.delenv(name, raising=False)


def test_flag_is_available_before_or_after_chat_subcommand():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    assert parser.parse_args(["--no-tools", "chat"]).no_tools is True
    assert parser.parse_args(["chat", "--no-tools"]).no_tools is True


@pytest.mark.parametrize(
    ("overrides", "env_name"),
    [
        ({"safe_mode": True}, None),
        ({"ignore_rules": True}, None),
        ({"toolsets": "terminal"}, None),
        ({"skills": ["untrusted"]}, None),
        ({"oneshot": "fixture"}, None),
        ({"resume": "old-session"}, None),
        ({"profile": "other-home"}, None),
        ({"image": "private.png"}, None),
        ({"worktree": True}, None),
        ({"yolo": True}, None),
        ({"tui": True}, None),
        ({}, "HERMES_IGNORE_RULES"),
        ({}, "HERMES_SAFE_MODE"),
    ],
)
def test_no_tools_rejects_weaker_or_tool_capable_modes(monkeypatch, overrides, env_name):
    from agent.no_tools import NoToolsConflictError, validate_no_tools_cli_args

    if env_name:
        monkeypatch.setenv(env_name, "1")
    with pytest.raises(NoToolsConflictError):
        validate_no_tools_cli_args(_args(**overrides))


def test_prepare_agent_startup_returns_before_plugins_mcp_or_hooks(monkeypatch):
    import hermes_cli.main as main_mod
    import providers
    from agent.no_tools import no_tools_bootstrap_active

    calls = []
    monkeypatch.setattr(providers, "_discovered", False)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: calls.append("plugin")),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=lambda *_a, **_k: calls.append("hook")),
    )

    main_mod._prepare_agent_startup(_args())

    assert no_tools_bootstrap_active() is True
    assert calls == []
    assert "model_tools" not in sys.modules
    mcp_module = sys.modules.get("tools.mcp_tool")
    assert not getattr(mcp_module, "_mcp_tool_server_names", None)


def test_cmd_chat_no_tools_skips_skill_sync_and_kanban_bootstrap(monkeypatch):
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser
    import providers

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(["chat", "--no-tools", "-q", "fixture"])
    captured = {}
    calls = []
    monkeypatch.setattr(providers, "_discovered", False)
    fake_cli = types.ModuleType("cli")

    def fake_main(**kwargs):
        captured.update(kwargs)

    fake_cli.main = fake_main
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(
        main_mod,
        "_sync_bundled_skills_for_startup",
        lambda: calls.append("skills"),
    )
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: calls.append("kanban"))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "fixture-kanban-task")

    main_mod.cmd_chat(args)

    assert captured["no_tools"] is True
    assert captured["ignore_rules"] is False
    assert calls == []
