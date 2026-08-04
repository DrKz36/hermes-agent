"""Offline contracts for the explicit CLI ``--no-tools`` policy."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

def _empty_no_tools_agent():
    return SimpleNamespace(
        no_tools=True,
        tools=[],
        valid_tool_names=set(),
        _context_engine_tool_names=set(),
        _memory_store=None,
        _memory_manager=None,
        _skip_mcp_refresh=True,
        api_mode="chat_completions",
    )


def test_bootstrap_imports_no_registry_or_agentic_tool_modules(tmp_path):
    """The policy must be active before run_agent can import tool modules."""
    probe = r'''
from types import SimpleNamespace
import sys
import types

from agent.no_tools import bootstrap_no_tools_from_argv

plugins = types.ModuleType("hermes_cli.plugins")
plugins.discover_plugins = lambda: (_ for _ in ()).throw(
    AssertionError("extension discovery must not run")
)
sys.modules["hermes_cli.plugins"] = plugins

assert bootstrap_no_tools_from_argv(["--no-tools", "chat"])
import run_agent  # noqa: F401

# Importing the classic CLI after the internal bootstrap must not read a
# project/home dotenv before the constrained runtime resolves its explicit
# provider path.
import hermes_cli.env_loader as env_loader
env_loader.load_hermes_dotenv = lambda **_kwargs: (_ for _ in ()).throw(
    AssertionError("dotenv load must stay outside --no-tools bootstrap")
)
import cli  # noqa: F401
assert cli.get_tool_definitions() == []
assert cli.get_toolset_for_tool("hostile") is None

forbidden = {
    "model_tools",
    "tools.terminal_tool",
    "tools.browser_tool",
}
loaded = forbidden.intersection(sys.modules)
assert not loaded, sorted(loaded)
assert run_agent._loaded_env_paths == ()
print("no_tools_bootstrap=PASS")
'''
    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_path / "fresh-home")
    env.pop("HERMES_IGNORE_RULES", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, (
        f"bootstrap probe failed rc={result.returncode}\n"
        f"stdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
    )
    assert result.stdout.strip().endswith("no_tools_bootstrap=PASS")


def test_no_tools_bootstrap_is_inert_without_the_explicit_flag(tmp_path):
    """The normal chat path retains its existing startup behavior."""
    probe = r'''
from agent.no_tools import bootstrap_no_tools_from_argv, no_tools_bootstrap_active

assert bootstrap_no_tools_from_argv(["chat", "-q", "fixture"]) is False
assert no_tools_bootstrap_active() is False
print("no_tools_absent=PASS")
'''
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tmp_path),
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, (
        f"normal-path probe failed rc={result.returncode}\n"
        f"stdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
    )
    assert result.stdout.strip().endswith("no_tools_absent=PASS")


def test_main_arms_no_tools_before_dotenv_or_optional_tool_startup(tmp_path):
    """The console entry module itself cannot create an unsafe pre-parse gap."""
    probe = r'''
import os
import pathlib
import sys
import tempfile
import types

with tempfile.TemporaryDirectory() as root:
    home = pathlib.Path(root) / "fresh-home"
    home.mkdir()
    hostile_plugin = home / "plugins" / "hostile"
    hostile_plugin.mkdir(parents=True)
    marker = pathlib.Path(root) / "hostile-plugin-imported"
    (hostile_plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('unexpected', encoding='utf-8')\n",
        encoding="utf-8",
    )
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HOME"] = root
    os.environ["USERPROFILE"] = root
    os.environ.pop("HERMES_IGNORE_RULES", None)

    fake_env = types.ModuleType("hermes_cli.env_loader")
    fake_env.load_hermes_dotenv = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("dotenv must not run before no-tools validation")
    )
    sys.modules["hermes_cli.env_loader"] = fake_env

    sys.argv = ["hermes", "chat", "--no-tools", "-q", "fixture"]
    import hermes_cli.main as main_mod

    captured = {}
    fake_cli = types.ModuleType("cli")
    fake_cli.main = lambda **kwargs: captured.update(kwargs)
    sys.modules["cli"] = fake_cli
    main_mod._has_any_provider_configured = lambda: True
    main_mod._termux_should_prefetch_update_check = lambda: False
    main_mod.main()
    assert captured["no_tools"] is True
    assert captured["ignore_rules"] is False

    forbidden = {
        "model_tools",
        "tools.terminal_tool",
        "tools.browser_tool",
        "tools.mcp_tool",
        "agent.shell_hooks",
        "hermes_cli.plugins",
    }
    loaded = forbidden.intersection(sys.modules)
    assert not loaded, sorted(loaded)
    assert not marker.exists()

print("main_no_tools_import=PASS")
'''
    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_path / "fresh-home")
    env.pop("HERMES_IGNORE_RULES", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, (
        f"main bootstrap probe failed rc={result.returncode}\n"
        f"stdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
    )
    assert result.stdout.strip().endswith("main_no_tools_import=PASS")


def test_no_tools_provider_profile_registry_does_not_discover_extensions(tmp_path):
    """Provider routing keeps the built-in Nous path without loading profiles."""
    probe = r'''
import os
import pathlib
import tempfile

from agent.no_tools import bootstrap_no_tools_from_argv

with tempfile.TemporaryDirectory() as root:
    home = pathlib.Path(root) / "fresh-home"
    plugin = home / "plugins" / "model-providers" / "hostile"
    plugin.mkdir(parents=True)
    marker = pathlib.Path(root) / "hostile-provider-imported"
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('unexpected', encoding='utf-8')\n",
        encoding="utf-8",
    )
    os.environ["HERMES_HOME"] = str(home)
    os.environ.pop("HERMES_IGNORE_RULES", None)

    assert bootstrap_no_tools_from_argv(["--no-tools", "chat"])
    import providers
    assert providers.get_provider_profile("hostile") is None
    assert providers.list_providers() == []

    # The auth module's dynamic provider registry must honor the same guard.
    import hermes_cli.auth as auth
    assert "hostile" not in auth.PROVIDER_REGISTRY
    assert auth.resolve_provider("nous") == "nous"
    assert not marker.exists()

print("no_tools_provider_profiles=PASS")
'''
    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_path / "fresh-home")
    env.pop("HERMES_IGNORE_RULES", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, (
        f"provider-profile probe failed rc={result.returncode}\n"
        f"stdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
    )
    assert result.stdout.strip().endswith("no_tools_provider_profiles=PASS")


def test_agent_keeps_rules_but_has_no_memory_or_tool_surface(tmp_path):
    """A fresh isolated home keeps governance but has no agentic surface."""
    probe = r'''
import os
import pathlib
import sys
import tempfile
from unittest.mock import patch

from agent.no_tools import bootstrap_no_tools_from_argv

prior_cwd = os.getcwd()
with tempfile.TemporaryDirectory() as root:
    root_path = pathlib.Path(root)
    home = root_path / "fresh-home"
    workspace = root_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    (home / "SOUL.md").write_text("NO_TOOLS_SOUL_SENTINEL", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("NO_TOOLS_RULE_SENTINEL", encoding="utf-8")
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HOME"] = root
    os.environ["USERPROFILE"] = root
    os.environ["HERMES_KANBAN_TASK"] = "fixture-kanban-task"
    os.environ.pop("HERMES_IGNORE_RULES", None)
    os.chdir(workspace)

    assert bootstrap_no_tools_from_argv(["--no-tools", "chat"])
    from run_agent import AIAgent

    config = {
        "agent": {"environment_probe": True},
        "memory": {"memory_enabled": True},
    }
    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch(
            "agent.context_compressor.get_model_context_length",
            side_effect=AssertionError("no metadata probe is allowed"),
        ),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="fixture/model",
            api_key="fixture-only",
            base_url="https://provider.invalid/v1",
            provider="nous",
            quiet_mode=True,
            no_tools=True,
            skip_context_files=False,
            skip_memory=False,
        )
        prompt = agent._build_system_prompt()
        payload = agent._build_api_kwargs(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "public fixture"},
            ]
        )

    assert "NO_TOOLS_RULE_SENTINEL" in prompt
    assert "NO_TOOLS_SOUL_SENTINEL" in prompt
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent._context_engine_tool_names == set()
    assert agent._memory_store is None
    assert agent._memory_manager is None
    assert agent._skip_mcp_refresh is True
    assert agent._environment_probe is False
    assert agent._fallback_chain == []
    assert agent._kanban_worker_guidance == ""
    assert agent.context_compressor.context_length == 256_000
    assert "tools" not in payload
    assert "tool_choice" not in payload
    agent._touch_activity("no-tools fixture")
    assert "model_tools" not in sys.modules
    assert "tools.terminal_tool" not in sys.modules
    assert "tools.browser_tool" not in sys.modules
    assert "tools.kanban_tools" not in sys.modules
    assert "tools.mcp_tool" not in sys.modules
    assert "hermes_cli.plugins" not in sys.modules
    provider_module = sys.modules.get("providers")
    assert not getattr(provider_module, "_discovered", False)
    import logging
    logging.shutdown()
    os.chdir(prior_cwd)

print("no_tools_agent=PASS")
'''
    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_path / "fresh-home")
    env.pop("HERMES_IGNORE_RULES", None)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, (
        f"agent bootstrap probe failed rc={result.returncode}\n"
        f"stdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
    )
    assert result.stdout.strip().endswith("no_tools_agent=PASS")


def test_payload_and_response_violations_stop_before_a_client_call():
    """A late schema mutation or tool response is a non-retryable local error."""
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    from agent.no_tools import (
        NoToolsInvariantError,
        NoToolsPayloadError,
        assert_no_tools_model_response,
        assert_no_tools_payload,
    )

    agent = _empty_no_tools_agent()
    calls = []

    def fake_client(_reason):
        calls.append(_reason)
        raise AssertionError("provider client must not be constructed")

    assert_no_tools_payload(agent, {"model": "fixture", "messages": []}, phase="test")
    with pytest.raises(NoToolsPayloadError):
        assert_no_tools_payload(
            agent,
            {"model": "fixture", "extra_body": {"tool_choice": "auto"}},
            phase="test",
        )
    with pytest.raises(NoToolsPayloadError):
        _dispatch_nonstreaming_api_request(
            agent,
            {"model": "fixture", "messages": [], "tools": []},
            make_client=fake_client,
        )
    with pytest.raises(NoToolsInvariantError):
        assert_no_tools_model_response(
            agent,
            SimpleNamespace(tool_calls=[{"id": "fixture"}], function_call=None),
        )
    agent.tools.append({"type": "function", "function": {"name": "hostile"}})
    with pytest.raises(NoToolsInvariantError):
        _dispatch_nonstreaming_api_request(
            agent,
            {"model": "fixture", "messages": []},
            make_client=fake_client,
        )

    assert calls == []


def test_late_extension_or_provider_profile_bootstrap_fails_closed(monkeypatch):
    """A process with prior extension discovery cannot be retrofitted safely."""
    from agent.no_tools import NoToolsBootstrapError, activate_no_tools_bootstrap

    monkeypatch.setitem(sys.modules, "providers", SimpleNamespace(_discovered=True))
    with pytest.raises(NoToolsBootstrapError, match="providers"):
        activate_no_tools_bootstrap()


def test_memory_and_mcp_guards_cannot_reinject_a_schema():
    """Defensive late paths preserve the empty snapshot instead of rebuilding it."""
    from agent.memory_manager import inject_memory_provider_tools
    from tools.mcp_tool import refresh_agent_mcp_tools

    agent = _empty_no_tools_agent()
    assert inject_memory_provider_tools(agent) == 0
    assert refresh_agent_mcp_tools(agent) == set()


def test_no_tools_never_upgrades_iteration_limit_to_summary_dispatch():
    """A constrained chat cannot spend a second request on a summary."""
    from agent.chat_completion_helpers import handle_max_iterations
    from agent.no_tools import NoToolsInvariantError

    with pytest.raises(NoToolsInvariantError):
        handle_max_iterations(_empty_no_tools_agent(), [], 1)


def test_no_tools_rejects_a_late_fallback_injection():
    """A mutated fallback chain cannot create a second provider attempt."""
    from agent.chat_completion_helpers import try_activate_fallback

    agent = _empty_no_tools_agent()
    agent._fallback_chain = [{"provider": "other", "model": "fixture"}]
    agent._fallback_index = 0
    assert try_activate_fallback(agent) is False
    assert agent._fallback_index == 0
