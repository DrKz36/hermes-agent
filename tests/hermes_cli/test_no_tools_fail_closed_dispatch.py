"""Offline failure-path coverage for the machine-only no-tools transport.

The happy-path integration test proves one actual fake Nous dispatch.  These
probes prove that an injected payload field is stopped *before* that dispatch,
and that a provider tool call stops after its one response without retrying or
falling back.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_probe(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "fresh-home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    (home / "SOUL.md").write_text("SOUL_SENTINEL_NO_TOOLS", encoding="utf-8")
    (workspace / "AGENTS.md").write_text(
        "AGENTS_SENTINEL_NO_TOOLS", encoding="utf-8"
    )

    env = dict(os.environ)
    env.update(
        {
            "HERMES_HOME": str(home),
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
            "NO_TOOLS_FAILURE_MODE": mode,
            "NO_PROXY": "*",
            "PYTHONPATH": os.pathsep.join(
                item
                for item in (str(REPO_ROOT), env.get("PYTHONPATH", ""))
                if item
            ),
        }
    )
    for name in (
        "HERMES_IGNORE_RULES",
        "HERMES_SAFE_MODE",
        "HERMES_TUI",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_GOAL_MODE",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        env.pop(name, None)

    probe = r'''
import contextlib
import io
import logging
import os
import socket
import sys
import types

workspace = sys.argv[-1]
os.chdir(workspace)
sys.argv = ["hermes", "chat", "--no-tools", "-Q", "-q", "public fixture"]
mode = os.environ["NO_TOOLS_FAILURE_MODE"]


def _network_forbidden(*_args, **_kwargs):
    raise AssertionError("network access attempted by offline no-tools test")


socket.create_connection = _network_forbidden
socket.socket.connect = _network_forbidden
logging.disable(logging.CRITICAL)

import hermes_cli.main as main_mod
import hermes_cli.runtime_provider as runtime_provider
import cli
import run_agent
import agent.chat_completion_helpers as completion_helpers
from agent.no_tools import NoToolsInvariantError, NoToolsPayloadError

MODEL = "deepseek/deepseek-v4-flash-0731"
captured = {"runtime_calls": 0, "dispatches": 0}


def _runtime(*, requested, explicit_api_key=None, explicit_base_url=None):
    assert requested == "nous", requested
    captured["runtime_calls"] += 1
    return {
        "api_key": "fixture-only-not-a-credential",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "provider": "nous",
        "requested_provider": "nous",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "source": "offline-test",
    }


runtime_provider.resolve_runtime_provider = _runtime


class FakeCompletions:
    def create(self, **kwargs):
        captured["dispatches"] += 1
        message = types.SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                types.SimpleNamespace(
                    id="hostile",
                    function=types.SimpleNamespace(
                        name="hostile", arguments="{}"
                    ),
                )
            ],
            function_call=None,
            refusal=None,
        )
        return types.SimpleNamespace(
            id="fixture-response",
            model=MODEL,
            choices=[types.SimpleNamespace(index=0, finish_reason="tool_calls", message=message)],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )


class FakeOpenAI:
    def __init__(self, **_kwargs):
        self.chat = types.SimpleNamespace(completions=FakeCompletions())

    def close(self):
        return None


run_agent.OpenAI = FakeOpenAI
original_build = completion_helpers.build_api_kwargs


def _mutate_build(*args, **kwargs):
    payload = original_build(*args, **kwargs)
    if mode == "payload":
        payload["tools"] = []
    return payload


completion_helpers.build_api_kwargs = _mutate_build
original_init_agent = cli.HermesCLI._init_agent


def _capture_agent(self, *args, **kwargs):
    result = original_init_agent(self, *args, **kwargs)
    captured["agent"] = self.agent
    return result


cli.HermesCLI._init_agent = _capture_agent
cli.CLI_CONFIG["model"].update({"default": MODEL, "provider": "nous"})
cli.CLI_CONFIG["agent"].update({"max_turns": 99, "fallback_providers": []})
cli.CLI_CONFIG["fallback_providers"] = []
cli.CLI_CONFIG.pop("fallback_model", None)

stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    try:
        main_mod.main()
    except SystemExit as exc:
        assert exc.code == 2
        assert isinstance(exc.__cause__, (NoToolsPayloadError, NoToolsInvariantError))
        captured["error"] = type(exc.__cause__).__name__
    else:
        raise AssertionError("no-tools policy violation unexpectedly completed")

assert captured["runtime_calls"] == 1, captured
assert captured["agent"].max_iterations == 1, captured["agent"].max_iterations
assert captured["agent"]._api_max_retries == 1, captured["agent"]._api_max_retries
assert captured["agent"]._fallback_chain == [], captured["agent"]._fallback_chain
assert captured["agent"]._session_db is None, captured["agent"]._session_db
assert stdout.getvalue() == "", stdout.getvalue()
assert stderr.getvalue().startswith("Error: --no-tools "), stderr.getvalue()
if mode == "payload":
    assert captured["error"] == "NoToolsPayloadError"
    assert captured["dispatches"] == 0
else:
    assert captured["error"] == "NoToolsInvariantError"
    assert captured["dispatches"] == 1
print(f"no_tools_{mode}_fail_closed=PASS")
'''
    return subprocess.run(
        [sys.executable, "-c", probe, str(workspace)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("mode", ("payload", "response"))
def test_machine_no_tools_policy_failure_never_retries_or_falls_back(tmp_path, mode):
    result = _run_probe(tmp_path, mode)
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout={result.stdout[-4000:]}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert result.stdout.strip().endswith(f"no_tools_{mode}_fail_closed=PASS")
