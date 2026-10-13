# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


try:
    import relax.core.controller as controller_module

    Controller = controller_module.Controller
except (ImportError, AssertionError) as exc:
    pytest.skip(f"relax.core.controller requires the full runtime environment: {exc}", allow_module_level=True)


def test_global_restart_signals_failure_from_any_phase():
    controller = object.__new__(Controller)
    controller._restart_done_event = threading.Event()
    controller._restart_error = None

    def fail_before_reinit():
        raise RuntimeError("ray init failed")

    controller._run_global_restart = fail_before_reinit

    controller._global_restart()

    assert controller._restart_done_event.is_set()
    assert isinstance(controller._restart_error, RuntimeError)
    assert str(controller._restart_error) == "ray init failed"


def test_global_restart_signals_success_on_same_event():
    controller = object.__new__(Controller)
    restart_done_event = threading.Event()
    controller._restart_done_event = restart_done_event
    controller._restart_error = None
    controller._run_global_restart = lambda: None

    controller._global_restart()

    assert controller._restart_done_event is restart_done_event
    assert restart_done_event.is_set()
    assert controller._restart_error is None


def test_restart_cycle_ack_is_published_after_old_state_is_cleared():
    controller = object.__new__(Controller)
    controller._restart_done_event = threading.Event()
    controller._restart_done_event.set()
    controller._restart_mode = "local"
    controller._restart_error = None
    controller._restarting = True

    class StartNextCycleOnAck:
        def set(self):
            assert controller._restarting is False
            assert controller._restart_mode is None
            controller._restarting = True
            controller._restart_mode = "global"

    controller._restart_consumed_event = StartNextCycleOnAck()

    restart_mode, restart_error = controller._consume_restart_cycle()

    assert restart_mode == "local"
    assert restart_error is None
    assert controller._restarting is True
    assert controller._restart_mode == "global"


@pytest.mark.parametrize("stopped_routers", [0, 1])
def test_global_restart_stops_owned_router_before_deleting_serve(monkeypatch, stopped_routers):
    events = []
    stop_ref = object()
    stop_remote = Mock(side_effect=lambda: events.append("stop router") or stop_ref)
    manager = SimpleNamespace(stop_owned_routers=SimpleNamespace(remote=stop_remote))
    service = SimpleNamespace(
        _stop_heartbeat_thread=Mock(),
        get_rollout_manager=Mock(return_value=manager),
    )
    config = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=3000)
    controller = object.__new__(Controller)
    controller.config = config
    controller.runtime_env = {}
    controller._global_restart_count = 0
    controller._max_global_restart = 1
    controller._health_manager = SimpleNamespace(_checker=None, stop=Mock())
    controller._cancel_pending_tasks = Mock()
    controller.serve_dict = {controller_module.ROLES.rollout: service}
    controller._metrics_service_enabled = False
    controller._autoscaler_config = None
    controller._shutdown_agentic_rollout_services = Mock()
    controller._restart_error = None

    def get(ref, *, timeout):
        assert ref is stop_ref and timeout == 30
        return stopped_routers

    monkeypatch.setattr(controller_module, "recovery_load_path", Mock())
    monkeypatch.setattr(controller_module, "run", lambda value: value)
    monkeypatch.setattr(controller_module.ray, "get", get)
    monkeypatch.setattr(controller_module.ray, "shutdown", Mock())
    monkeypatch.setattr(controller_module.ray, "init", Mock())
    monkeypatch.setattr(controller_module.serve, "delete", lambda role: events.append(f"delete {role}"))
    monkeypatch.setattr(controller_module.serve, "shutdown", Mock())
    monkeypatch.setattr(controller_module.serve, "start", Mock())
    monkeypatch.setattr(controller_module.tq, "close", Mock())
    monkeypatch.setattr(controller_module, "shutdown_async_loop", Mock())
    monkeypatch.setattr(controller_module.time, "sleep", Mock())
    monkeypatch.setattr(Controller, "__init__", lambda self, config, runtime_env: None)

    controller._run_global_restart()

    stop_remote.assert_called_once_with()
    assert events.index("stop router") < events.index(f"delete {controller_module.ROLES.rollout}")
    if stopped_routers:
        assert config.sglang_router_ip is None
        assert config.sglang_router_port is None
    else:
        assert config.sglang_router_ip == "127.0.0.1"
        assert config.sglang_router_port == 3000
