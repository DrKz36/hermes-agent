"""Offline transport proof for the constrained ``--no-tools`` chat path.

The public contract is intentionally narrow.  This test does not replace the
agent with a canned ``run_conversation`` result: it drives the real CLI,
``AIAgent`` construction, prompt assembly, Chat Completions transport, and
the final dispatch seam.  Only the OpenAI SDK client is fake, so no network,
credential, or real Hermes home is required.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_probe(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "fresh-home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    (home / "SOUL.md").write_text("SOUL_SENTINEL_NO_TOOLS", encoding="utf-8")
    (home / "config.yaml").write_text(
        "terminal:\n  env_type: ssh\n  ssh_key: fixture-terminal-value\n"
        "auxiliary:\n  vision:\n    provider: fixture\n"
        "    api_key: fixture-auxiliary-value\n",
        encoding="utf-8",
    )
    (workspace / "AGENTS.md").write_text(
        "AGENTS_SENTINEL_NO_TOOLS", encoding="utf-8"
    )

    env = dict(os.environ)
    env.update(
        {
            "HERMES_HOME": str(home),
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
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
        "TERMINAL_SSH_KEY",
        "TERMINAL_ENV",
        "AUXILIARY_VISION_PROVIDER",
        "AUXILIARY_VISION_API_KEY",
    ):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-c", source, str(workspace)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_machine_no_tools_real_cli_agent_and_nous_dispatch_are_offline(tmp_path):
    """Prove the actual one-shot path reaches one fake Nous dispatch cleanly."""
    probe = r'''
import contextlib
import copy
import io
import logging
import os
import socket
import sys
import threading
import types
import uuid
from datetime import datetime as RealDateTime

workspace = sys.argv[-1]
os.chdir(workspace)
sys.argv = [
    "hermes", "chat", "--no-tools", "-Q", "-q", "public fixture"
]

# This is defense in depth for the test.  Any attempted socket connection is
# a test failure; the sole provider request below is handled by FakeOpenAI.
def _network_forbidden(*_args, **_kwargs):
    raise AssertionError("network access attempted by offline no-tools test")

socket.create_connection = _network_forbidden
socket.socket.connect = _network_forbidden
logging.disable(logging.CRITICAL)

import hermes_cli.main as main_mod
import hermes_cli.runtime_provider as runtime_provider
import cli
import run_agent
from agent.no_tools import no_tools_bootstrap_active
import agent.chat_completion_helpers as completion_helpers

assert no_tools_bootstrap_active()
assert os.environ.get("HERMES_IGNORE_RULES") is None
assert os.environ.get("TERMINAL_SSH_KEY") is None, os.environ.get("TERMINAL_SSH_KEY")
assert os.environ.get("TERMINAL_ENV") is None, os.environ.get("TERMINAL_ENV")
assert os.environ.get("AUXILIARY_VISION_PROVIDER") is None, os.environ.get("AUXILIARY_VISION_PROVIDER")
assert os.environ.get("AUXILIARY_VISION_API_KEY") is None, os.environ.get("AUXILIARY_VISION_API_KEY")

MODEL = "deepseek/deepseek-v4-flash-0731"
BASE_URL = "https://inference-api.nousresearch.com/v1"
captured = {"dispatches": 0, "client_inits": [], "payloads": []}


def _runtime(*, requested, explicit_api_key=None, explicit_base_url=None):
    assert requested == "nous", requested
    assert explicit_api_key is None
    assert explicit_base_url is None
    return {
        "api_key": "fixture-only-not-a-credential",
        "base_url": BASE_URL,
        "provider": "nous",
        "requested_provider": "nous",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "source": "offline-test",
    }


runtime_provider.resolve_runtime_provider = _runtime


def _contains_forbidden_field(value):
    forbidden = {
        "tools", "toolchoice", "functions", "functioncall",
        "paralleltoolcalls", "toolconfig", "functioncallingconfig",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = "".join(ch for ch in str(key).lower() if ch.isalnum())
            if normalized in forbidden:
                return True
            if _contains_forbidden_field(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_field(item) for item in value)
    return False


class FakeCompletions:
    def create(self, **kwargs):
        assert no_tools_bootstrap_active(), "policy lost before fake dispatch"
        assert not _contains_forbidden_field(kwargs), kwargs
        captured["dispatches"] += 1
        captured["payloads"].append(copy.deepcopy(kwargs))
        message = types.SimpleNamespace(
            role="assistant",
            content="NOUS_ISOLATED_OK",
            tool_calls=None,
            function_call=None,
            refusal=None,
        )
        return types.SimpleNamespace(
            id="fixture-response",
            model=MODEL,
            choices=[types.SimpleNamespace(index=0, finish_reason="stop", message=message)],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )


class FakeOpenAI:
    def __init__(self, **kwargs):
        # Hermes supplies an httpx client carrying an RLock.  Retain the
        # observable routing fields without attempting to serialize that
        # implementation detail.
        captured["client_inits"].append(
            {key: value for key, value in kwargs.items() if key != "http_client"}
        )
        self.chat = types.SimpleNamespace(completions=FakeCompletions())

    def close(self):
        return None


run_agent.OpenAI = FakeOpenAI
original_build = completion_helpers.build_api_kwargs


def _capture_build(*args, **kwargs):
    payload = original_build(*args, **kwargs)
    captured["built_payload"] = copy.deepcopy(payload)
    return payload


completion_helpers.build_api_kwargs = _capture_build
original_init_agent = cli.HermesCLI._init_agent


def _capture_agent(self, *args, **kwargs):
    result = original_init_agent(self, *args, **kwargs)
    captured["agent"] = self.agent
    return result


cli.HermesCLI._init_agent = _capture_agent
cli.CLI_CONFIG["model"].update({"default": MODEL, "provider": "nous"})
cli.CLI_CONFIG["agent"].update({"max_turns": 1, "fallback_providers": []})
cli.CLI_CONFIG["display"].update({"streaming": False})
cli.CLI_CONFIG["fallback_providers"] = []
cli.CLI_CONFIG.pop("fallback_model", None)


class FixedDateTime(RealDateTime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 2, 3, 4, 5, tzinfo=tz)


cli.datetime = FixedDateTime
cli.uuid.uuid4 = lambda: uuid.UUID("abcdefabcdefabcdefabcdefabcdefab")

thread_starts = []
_original_thread_start = threading.Thread.start


def _capture_thread_start(self, *args, **kwargs):
    thread_starts.append(getattr(self, "name", ""))
    return _original_thread_start(self, *args, **kwargs)


threading.Thread.start = _capture_thread_start

stdout = io.StringIO()
stderr = io.StringIO()
try:
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            main_mod.main()
        except SystemExit as exc:
            if exc.code != 0:
                raise AssertionError(
                    f"no-tools command exited {exc.code}; "
                    f"stdout={stdout.getvalue()!r}; stderr={stderr.getvalue()!r}"
                ) from exc
finally:
    threading.Thread.start = _original_thread_start

agent = captured["agent"]
assert stdout.getvalue().encode("utf-8") == b"NOUS_ISOLATED_OK\n", repr(stdout.getvalue())
assert stderr.getvalue().encode("utf-8") == (
    b"\nsession_id: 20260102_030405_abcdef\n"
), repr(stderr.getvalue())
assert captured["dispatches"] == 1
assert captured["built_payload"] == captured["payloads"][0]
assert captured["payloads"][0]["model"] == MODEL
assert not _contains_forbidden_field(captured["payloads"][0])
assert "product=hermes-agent" in captured["payloads"][0]["extra_body"]["tags"]
assert "conversation=20260102_030405_abcdef" in captured["payloads"][0]["extra_body"]["tags"]
assert agent.no_tools is True
assert agent.provider == "nous"
assert agent.requested_provider == "nous"
assert agent.model == MODEL
assert agent.tools == []
assert agent.valid_tool_names == set()
assert agent._context_engine_tool_names == set()
assert agent._skip_mcp_refresh is True
assert agent._memory_store is None
assert agent._memory_manager is None
assert agent._session_db is None
assert agent._fallback_chain == []
assert agent._api_max_retries == 1
assert agent.max_iterations == 1
assert thread_starts == [], thread_starts
assert "AGENTS_SENTINEL_NO_TOOLS" in agent._cached_system_prompt
assert "SOUL_SENTINEL_NO_TOOLS" in agent._cached_system_prompt
assert not ("model_tools" in sys.modules)
assert not ("tools.terminal_tool" in sys.modules)
assert not ("tools.browser_tool" in sys.modules)
assert not ("tools.mcp_tool" in sys.modules)
plugin_module = sys.modules.get("hermes_cli.plugins")
assert not bool(getattr(getattr(plugin_module, "_plugin_manager", None), "_discovered", False))
assert not ("agent.shell_hooks" in sys.modules)
print("real_no_tools_nous_dispatch=PASS")
'''
    result = _run_probe(tmp_path, probe)
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout={result.stdout[-4000:]}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert result.stdout.strip().endswith("real_no_tools_nous_dispatch=PASS")
