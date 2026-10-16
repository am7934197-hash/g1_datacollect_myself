"""Native-RTC protocol helpers for the deployed LeRobot PolicyServer.

This file is intentionally dependency-light so it can be copied beside the
remote ``lerobot/async_inference/policy_server.py``.  The server must still use
its own observation preprocessor and action postprocessor.

Required PolicyServer integration, inside ``_predict_action_chunk``:

1. Read ``timed_observation.rtc_context`` with :func:`rtc_kwargs`.
2. Convert ``prev_chunk_left_over`` from ``absolute_postprocessed`` back to the
   policy's normalized/relative action space.  For relative checkpoints this
   is mandatory; use the same inverse/re-anchoring steps as LeRobot's RTC real
   robot example.
3. Call ``policy.predict_action_chunk(preprocessed, **kwargs)``.
4. Postprocess the returned chunk for robot execution.
5. Serialize the TimedAction list with :func:`response_envelope`.

Do not acknowledge ``applied=True`` unless steps 2-3 actually ran with an
initialized RTC processor.  The robot client deliberately fails closed.

During ``SendPolicyInstructions`` the server must also map
``RemotePolicyConfig.rtc_config`` to LeRobot's ``RTCConfig`` and run::

    from lerobot.policies.rtc import RTCConfig
    policy.config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=remote_cfg.rtc_config["execution_horizon"],
    )
    policy.init_rtc_processor()
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def rtc_kwargs(
    timed_observation: Any,
    *,
    prefix_to_policy_space: Optional[Callable[[Any], Any]] = None,
) -> tuple[dict, dict]:
    """Build official LeRobot RTC kwargs and response metadata."""
    context = dict(getattr(timed_observation, "rtc_context", None) or {})
    if not context.get("enabled", False):
        return {}, {"applied": False}

    previous = context.get("prev_chunk_left_over")
    semantics = context.get("prev_chunk_semantics")
    if previous and semantics == "absolute_postprocessed":
        if prefix_to_policy_space is None:
            raise RuntimeError(
                "native RTC received an absolute/postprocessed prefix but the "
                "PolicyServer did not provide prefix_to_policy_space"
            )
        previous = prefix_to_policy_space(previous)

    inference_delay = max(0, int(context.get("inference_delay", 0) or 0))
    execution_horizon = int(context.get("execution_horizon", 0) or 0)
    if execution_horizon <= 0:
        raise ValueError("native RTC execution_horizon must be positive")

    kwargs = {
        "inference_delay": inference_delay,
        "prev_chunk_left_over": previous,
        "execution_horizon": execution_horizon,
    }
    meta = {
        "applied": True,
        "session_id": context.get("session_id"),
        "infer_seq": context.get("infer_seq"),
        "inference_delay_frames": inference_delay,
        "execution_horizon": execution_horizon,
        "prefix_frames_used": min(len(previous) if previous is not None else 0,
                                  execution_horizon),
    }
    return kwargs, meta


def response_envelope(timed_actions: list, rtc_meta: dict) -> dict:
    """Payload to pickle into ``services_pb2.Actions.data``."""
    return {"actions": timed_actions, "rtc": dict(rtc_meta)}
