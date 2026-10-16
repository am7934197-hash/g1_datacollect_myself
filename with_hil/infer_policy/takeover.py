"""HIL episode latch: one-shot takeover, success/failure end, sidecar JSON.

Takeover is a one-way AUTO -> INTERVENTION latch for the current episode.
There is no hold/release path back to policy.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import inference_logger as logger


class EpisodeControl:
    """Per-execute episode state. Thread-safe for stdin / HTTP / loop."""

    def __init__(
        self,
        run_id: str,
        instruction: str,
        sidecar_dir: str = "logs/hil_episodes",
        default_success_on_timeout: int = 0,
        collector_policy_id: Optional[str] = None,
        takeover_grace_s: float = 0.0,
    ) -> None:
        self.run_id = run_id
        self.instruction = instruction
        self.sidecar_dir = sidecar_dir
        self.default_success_on_timeout = int(default_success_on_timeout)
        self.collector_policy_id = collector_policy_id
        self.takeover_grace_s = max(0.0, float(takeover_grace_s))
        self.schema_version = "evo_rl_compatible_raw_v1"
        self.started_at = time.time()
        self.ended_at: Optional[float] = None
        self.takeover_at: Optional[float] = None
        self.intervention_data_start_at: Optional[float] = None
        self.takeover_step: Optional[int] = None
        self.policy_motion_complete_at: Optional[float] = None
        self.episode_success: Optional[int] = None
        self.end_reason: Optional[str] = None
        self.sidecar_path: Optional[str] = None
        self.capture_result = None
        self.current_step = 0
        self._intervened = False
        self._finished = False
        self._lock = threading.RLock()

    @contextmanager
    def command_guard(self):
        """Serialize a final SDK send with takeover/finish acknowledgement."""
        with self._lock:
            yield

    @property
    def intervened(self) -> bool:
        with self._lock:
            return self._intervened

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def mode(self) -> str:
        with self._lock:
            if self._finished:
                return "FINISHED"
            if self.policy_motion_complete_at is not None:
                return "AWAITING_LABEL"
            if self._intervened:
                return "INTERVENTION"
            return "AUTO"

    def set_step(self, step: int) -> None:
        with self._lock:
            self.current_step = int(step)

    def request_takeover(
        self, step: Optional[int] = None, grace_s: Optional[float] = None
    ) -> bool:
        """Latch AUTO -> INTERVENTION. Repeat presses are ignored."""
        with self._lock:
            if self._finished:
                logger.info(
                    f"Takeover ignored: episode already finished run_id={self.run_id}"
                )
                return False
            if self._intervened:
                logger.info(
                    f"Takeover ignored: already INTERVENTION run_id={self.run_id}"
                )
                return False
            self._intervened = True
            self.takeover_at = time.time()
            effective_grace_s = self.takeover_grace_s if grace_s is None else max(0.0, float(grace_s))
            self.takeover_grace_s = effective_grace_s
            self.intervention_data_start_at = self.takeover_at + effective_grace_s
            self.takeover_step = self.current_step if step is None else int(step)
            logger.warning(
                f"TAKEOVER START run_id={self.run_id} step={self.takeover_step}"
            )
            return True

    def mark_policy_motion_complete(self) -> None:
        with self._lock:
            if self.policy_motion_complete_at is None:
                self.policy_motion_complete_at = time.time()

    def finish(self, success: int, reason: str) -> bool:
        """Idempotent episode end. First caller wins."""
        with self._lock:
            if self._finished:
                return False
            self._finished = True
            self.episode_success = int(success)
            self.end_reason = str(reason)
            self.ended_at = time.time()
            logger.warning(
                f"EPISODE END run_id={self.run_id} success={self.episode_success} "
                f"reason={self.end_reason} intervened={self._intervened}"
            )
            return True

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "instruction": self.instruction,
                "collector_policy_id": self.collector_policy_id,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "takeover_at": self.takeover_at,
                "takeover_grace_s": self.takeover_grace_s,
                "intervention_data_start_at": self.intervention_data_start_at,
                "takeover_step": self.takeover_step,
                "policy_motion_complete_at": self.policy_motion_complete_at,
                "intervened": self._intervened,
                "episode_success": self.episode_success,
                "end_reason": self.end_reason,
                "capture": self.capture_result,
            }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            if self._finished:
                mode = "FINISHED"
            elif self.policy_motion_complete_at is not None:
                mode = "AWAITING_LABEL"
            elif self._intervened:
                mode = "INTERVENTION"
            else:
                mode = "AUTO"
            return {
                "mode": mode,
                "intervened": self._intervened,
                "run_id": self.run_id,
                "episode_success": self.episode_success,
                "end_reason": self.end_reason,
                "takeover_at": self.takeover_at,
                "takeover_grace_s": self.takeover_grace_s,
                "intervention_data_start_at": self.intervention_data_start_at,
                "takeover_step": self.takeover_step,
                "policy_motion_complete_at": self.policy_motion_complete_at,
                "current_step": self.current_step,
            }

    def write_sidecar(self) -> str:
        payload = self.to_dict()
        os.makedirs(self.sidecar_dir, exist_ok=True)
        path = os.path.join(self.sidecar_dir, f"{self.run_id}.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
        with self._lock:
            self.sidecar_path = path
        logger.info(f"HIL sidecar written: {path}")
        return path


def run_continuous_collection(
    run_episode: Callable[[], dict],
    reset_between: Callable[[], tuple[bool, str]],
    should_stop: Callable[[], bool],
    update_state: Callable[..., None],
    max_episodes: Optional[int] = None,
    reset_before: Optional[Callable[[], tuple[bool, str]]] = None,
    reset_after: Optional[Callable[[], tuple[bool, str]]] = None,
) -> dict:
    """Run labeled HIL episodes until stopped, capped, or reset fails."""
    completed = 0
    episode_results = []
    limit = int(max_episodes or 0)

    if reset_before is not None:
        update_state(
            phase="INITIAL_RESET",
            episode_index=1,
            environment_reset_remaining_s=None,
        )
        reset_ok, reset_message = reset_before()
        if not reset_ok:
            message = f"Initial robot reset failed: {reset_message}"
            update_state(error=message)
            return {
                "success": False,
                "message": message,
                "details": {
                    "episodes_completed": 0,
                    "episodes": [],
                    "initial_reset": False,
                },
            }
        update_state(phase="STARTING_EPISODE", episode_index=1)

    while not should_stop():
        index = completed + 1
        update_state(phase="RUNNING_EPISODE", episode_index=index)
        result = run_episode()
        details = dict(result.get("details") or {})
        episode_summary = {
            "episode_index": index,
            "run_id": details.get("run_id"),
            "episode_success": details.get("episode_success"),
            "end_reason": details.get("end_reason"),
            "intervened": details.get("intervened"),
            "sidecar_path": details.get("sidecar_path"),
        }
        episode_results.append(episode_summary)
        episode_results = episode_results[-20:]
        update_state(last_episode=episode_summary)

        if should_stop():
            break
        if details.get("end_reason") not in ("success_key", "failure_key"):
            underlying = str(result.get("message") or "").strip()
            reason = str(details.get("end_reason") or "").strip()
            cause = underlying if underlying and underlying != "VLA loop completed" else reason
            message = (
                "Continuous collection stopped because episode did not end with s/f: "
                f"{reason or cause}"
            )
            if cause and cause != reason:
                message += f" ({cause})"
            update_state(error=message)
            return {
                "success": False,
                "message": message,
                "details": {
                    "episodes_completed": completed,
                    "episodes": episode_results,
                },
            }

        completed += 1
        update_state(episodes_completed=completed)
        if limit > 0 and completed >= limit:
            if reset_after is not None:
                update_state(
                    phase="FINAL_RESET",
                    environment_reset_remaining_s=None,
                )
                reset_ok, reset_message = reset_after()
                if not reset_ok:
                    message = f"Final robot reset failed: {reset_message}"
                    update_state(error=message)
                    return {
                        "success": False,
                        "message": message,
                        "details": {
                            "episodes_completed": completed,
                            "episodes": episode_results,
                            "final_reset": False,
                        },
                    }
            return {
                "success": True,
                "message": "Continuous collection reached max_episodes",
                "details": {
                    "episodes_completed": completed,
                    "episodes": episode_results,
                    "final_reset": reset_after is not None,
                },
            }

        reset_ok, reset_message = reset_between()
        if not reset_ok:
            if should_stop():
                break
            update_state(error=reset_message)
            return {
                "success": False,
                "message": reset_message,
                "details": {
                    "episodes_completed": completed,
                    "episodes": episode_results,
                },
            }
        update_state(phase="STARTING_EPISODE", episode_index=completed + 1)

    return {
        "success": True,
        "message": "Continuous collection stopped",
        "details": {"episodes_completed": completed, "episodes": episode_results},
    }
