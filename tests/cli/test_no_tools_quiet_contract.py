"""Byte-exact quiet-output contract for the explicit no-tools chat path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli as cli_mod


def test_quiet_no_tools_success_preserves_machine_output_bytes(monkeypatch, capsys):
    """A fake offline provider result emits only payload stdout and session stderr."""
    import providers

    route = {
        "model": "fixture/model",
        "runtime": {},
        "signature": ("fixture/model",),
    }

    class FakeAgent:
        session_id = "no-tools-session"
        quiet_mode = False
        suppress_status_output = False
        stream_delta_callback = object()
        tool_gen_callback = object()

        def run_conversation(self, **_kwargs):
            return {"final_response": "NOUS_ISOLATED_OK", "failed": False}

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.session_id = "no-tools-session"
            self.agent = None
            self.conversation_history = []
            self.provider = "nous"
            self.model = "fixture/model"
            self.requested_provider = "nous"
            self._active_agent_route_signature = route["signature"]

        def _claim_active_session(self, *_args, **_kwargs):
            return True

        def _release_active_session(self):
            return None

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, _query):
            return route

        def _init_agent(self, **_kwargs):
            self.agent = FakeAgent()
            return True

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_k: None)
    monkeypatch.setattr(providers, "_discovered", False)

    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(query="public fixture", quiet=True, no_tools=True)

    captured = capsys.readouterr()
    assert exit_info.value.code == 0
    assert captured.out.encode("utf-8") == b"NOUS_ISOLATED_OK\n"
    assert captured.err.encode("utf-8") == b"\nsession_id: no-tools-session\n"
