"""Machine-only boundaries for ``hermes chat --no-tools -Q -q``.

These probes intentionally run in fresh Python processes.  The no-tools
bootstrap is process-scoped by design, so a normal pytest process cannot
meaningfully prove that an invalid interactive invocation stayed ahead of all
extension imports.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IMPORTS = {
    "model_tools",
    "tools.terminal_tool",
    "tools.browser_tool",
    "tools.mcp_tool",
    "agent.shell_hooks",
    "hermes_cli.plugins",
}


def _run_probe(
    tmp_path: Path, source: str, *, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    home = tmp_path / "fresh-home"
    home.mkdir()
    env.update(
        {
            "HERMES_HOME": str(home),
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
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
    ):
        env.pop(name, None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_rejected_interactive_form_never_starts_runtime_or_imports_tools(tmp_path):
    """Missing ``-Q`` fails before prewarm/process-loop/bootstrap imports."""
    probe = r'''
import sys
import threading
sys.argv = ["hermes", "chat", "--no-tools", "-q", "public fixture"]
import hermes_cli.main as main_mod
startup_calls = []
thread_starts = []
_real_thread_start = threading.Thread.start

def _record_thread_start(self, *args, **kwargs):
    thread_starts.append(self.name)
    return _real_thread_start(self, *args, **kwargs)

threading.Thread.start = _record_thread_start
main_mod._prepare_agent_startup = lambda *_a, **_k: startup_calls.append("startup")
try:
    main_mod.main()
except SystemExit as exc:
    assert exc.code == 2, exc.code
else:
    raise AssertionError("interactive no-tools form was accepted")

forbidden = {
    "model_tools",
    "tools.terminal_tool",
    "tools.browser_tool",
    "tools.mcp_tool",
    "agent.shell_hooks",
    "hermes_cli.plugins",
}
assert not startup_calls, startup_calls
assert not thread_starts, thread_starts
assert "cli" not in sys.modules
assert "run_agent" not in sys.modules
assert not (forbidden & set(sys.modules)), forbidden & set(sys.modules)
print("rejected_interactive_inert=PASS")
'''
    result = _run_probe(tmp_path, probe)
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout={result.stdout[-2000:]}\n"
        f"stderr={result.stderr[-2000:]}"
    )
    assert result.stdout.strip().endswith("rejected_interactive_inert=PASS")


def test_exact_machine_form_reaches_only_the_synchronous_cli_entry(tmp_path):
    """The permitted spelling forwards no interactive or extension options."""
    probe = r'''
import sys
import types

captured = {}
fake_cli = types.ModuleType("cli")
fake_cli.main = lambda **kwargs: captured.update(kwargs)
sys.modules["cli"] = fake_cli
sys.argv = [
    "hermes", "chat", "--no-tools", "-Q", "-q", "public fixture"
]

import hermes_cli.main as main_mod
main_mod._has_any_provider_configured = lambda: True
main_mod._termux_should_prefetch_update_check = lambda: False
main_mod.main()

assert captured == {
    "query": "public fixture",
    "quiet": True,
    "no_tools": True,
}
forbidden = {
    "model_tools",
    "tools.terminal_tool",
    "tools.browser_tool",
    "tools.mcp_tool",
    "agent.shell_hooks",
    "hermes_cli.plugins",
}
assert not (forbidden & set(sys.modules)), forbidden & set(sys.modules)
print("machine_entry_only=PASS")
'''
    result = _run_probe(tmp_path, probe)
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout={result.stdout[-2000:]}\n"
        f"stderr={result.stderr[-2000:]}"
    )
    assert result.stdout.strip().endswith("machine_entry_only=PASS")


def test_rules_disabling_environment_rejects_before_cli_import(tmp_path):
    """A pre-existing governance bypass cannot reach the one-shot runtime."""
    probe = r'''
import sys
sys.argv = ["hermes", "chat", "--no-tools", "-Q", "-q", "public fixture"]
import hermes_cli.main as main_mod
try:
    main_mod.main()
except SystemExit as exc:
    assert exc.code == 2, exc.code
else:
    raise AssertionError("HERMES_IGNORE_RULES was accepted")

assert "cli" not in sys.modules
assert "run_agent" not in sys.modules
print("rules_env_rejected_preimport=PASS")
'''
    result = _run_probe(
        tmp_path, probe, extra_env={"HERMES_IGNORE_RULES": "1"}
    )
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout={result.stdout[-2000:]}\n"
        f"stderr={result.stderr[-2000:]}"
    )
    assert result.stdout.strip().endswith("rules_env_rejected_preimport=PASS")


def test_abbreviated_no_tools_is_rejected_before_cli_import(tmp_path):
    """Argparse long-option abbreviations cannot widen the public contract."""
    probe = r'''
import sys
sys.argv = ["hermes", "chat", "--no-to", "-Q", "-q", "public fixture"]
import hermes_cli.main as main_mod
try:
    main_mod.main()
except SystemExit as exc:
    assert exc.code == 2, exc.code
else:
    raise AssertionError("abbreviated --no-tools form was accepted")

assert "cli" not in sys.modules
assert "run_agent" not in sys.modules
assert "model_tools" not in sys.modules
assert "hermes_cli.plugins" not in sys.modules
print("abbreviated_no_tools_rejected_preimport=PASS")
'''
    result = _run_probe(tmp_path, probe)
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout={result.stdout[-2000:]}\n"
        f"stderr={result.stderr[-2000:]}"
    )
    assert result.stdout.strip().endswith(
        "abbreviated_no_tools_rejected_preimport=PASS"
    )


def test_raw_contract_rejects_global_spelling_and_unapproved_options():
    """Argparse inheritance cannot silently widen the public surface."""
    from agent.no_tools import NoToolsConflictError, validate_no_tools_raw_argv

    invalid = (
        ["--no-tools", "chat", "-Q", "-q", "fixture"],
        ["chat", "--no-tools", "-Q", "-q", "fixture", "--toolsets", "all"],
        ["chat", "--no-tools", "-Q", "-q", "fixture", "--safe-mode"],
        ["chat", "--no-tools", "-Q", "-q", "fixture", "--ignore-rules"],
        ["chat", "--no-tools", "--quiet", "-q", "fixture"],
        ["chat", "--no-tools", "-Q", "--query", "fixture"],
        ["chat", "--no-tools", "-Q", "-q", "--safe-mode"],
        ["chat", "--no-tools", "-Q", "fixture"],
        ["chat", "--no-tools", "-q", "fixture"],
    )
    for argv in invalid:
        try:
            validate_no_tools_raw_argv(argv)
        except NoToolsConflictError:
            continue
        raise AssertionError(f"raw no-tools contract accepted {argv!r}")
