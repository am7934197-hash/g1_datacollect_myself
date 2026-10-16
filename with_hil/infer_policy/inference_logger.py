"""
统一日志模块 — 控制台输出 + 结构化推理日志 + 可选图像保存。

其他文件只需:
    import inference_logger as logger
    logger.info("...")
    logger.warning("...")
    logger.error("...")
    logger.debug("...")
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

# ============================================================================
# 控制台 logger (所有模块共用)
# ============================================================================

_console = logging.getLogger("galbot_vla")
_console.setLevel(logging.INFO)
_console.propagate = False  # 不向 root 传播，避免重复输出

_stdout = logging.StreamHandler()
_stdout.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
_console.addHandler(_stdout)

info = _console.info
warning = _console.warning
error = _console.error
debug = _console.debug
exception = _console.exception
set_level = _console.setLevel

# ============================================================================
# JSON 序列化工具
# ============================================================================

_JSON_DUMPS = json.dumps
_JSON_SEP = (",", ":")


def _fast_dumps(obj, default=None):
    return _JSON_DUMPS(obj, ensure_ascii=False, separators=_JSON_SEP, default=default)


def _json_default(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, bytes):
        return v.hex()
    return str(v)


def _truncate_state(state_dict: dict, max_vals: int = 8) -> dict:
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, (list, np.ndarray)) and len(v) > max_vals * 2:
            out[k] = list(v[:max_vals]) + ["..."] + list(v[-max_vals:])
        else:
            out[k] = v
    return out


# ============================================================================
# 推理文件日志 + 图像保存
# ============================================================================

class VLAInferenceLogger:
    """将每一步推理的 state/actions/耗时写入结构化日志，可选保存图像。"""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.enabled = cfg.get("enabled", True)
        if not self.enabled:
            return

        self._log_dir = os.path.abspath(cfg.get("log_dir", "logs"))
        self._max_size = int(cfg.get("max_log_size_mb", 100)) * 1024 * 1024
        self._log_data = cfg.get("log_data", True)
        self._save_images = cfg.get("save_images", False)
        self._image_dir = os.path.abspath(cfg.get("image_dir", os.path.join(self._log_dir, "images")))
        self._max_image_runs = int(cfg.get("max_image_runs", 5))

        self._current_log_path: Optional[str] = None
        self._log_seq: int = 0
        self._current_run_id: Optional[str] = None
        self._lock = threading.Lock()

        os.makedirs(self._log_dir, exist_ok=True)
        if self._save_images:
            os.makedirs(self._image_dir, exist_ok=True)
        self._rotate()

    # -- Run management --

    def start_run(self, run_id: str, instruction: str):
        if not self.enabled:
            return
        with self._lock:
            self._current_run_id = run_id
            self._write({
                "event": "run_start",
                "run_id": run_id,
                "instruction": instruction,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            })

    def log_camera_geometry(self, run_id: str, geometry: dict):
        """Persist native HxW vs 224 letterbox for the first observation of a run."""
        if not self.enabled or not geometry:
            return
        with self._lock:
            self._write({
                "event": "camera_geometry",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "cameras": geometry,
            })
        if self._save_images:
            step_dir = os.path.join(self._image_dir, run_id)
            os.makedirs(step_dir, exist_ok=True)
            path = os.path.join(step_dir, "camera_geometry.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(geometry, f, ensure_ascii=False, indent=2, default=_json_default)
            except Exception as e:
                warning(f"[InferenceLog] camera_geometry write error: {e}")

    def end_run(self, run_id: str):
        if not self.enabled:
            return
        with self._lock:
            self._write({
                "event": "run_end",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            })
            self._current_run_id = None

    # -- Step logging --

    def log_obs(self, run_id: str, step: int, obs: dict, elapsed_ms: float):
        if not self.enabled:
            return
        with self._lock:
            record = {
                "run_id": run_id,
                "step": step,
                "obs_elapsed_ms": elapsed_ms,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            }
            if self._log_data:
                state_dict = {k: v for k, v in obs.items()
                              if not isinstance(v, np.ndarray) and k != "task"}
                record["state"] = _truncate_state(state_dict)
            self._write(record)

        if self._save_images:
            image_keys = {k for k, v in obs.items() if isinstance(v, np.ndarray)}
            self._save_step_images(run_id, step, obs, image_keys)
        if self._save_images and step == 0:
            self._cleanup_old_runs()

    def log_step(self, run_id: str, instruction: str, step: int,
                 obs: dict, actions: list, elapsed_ms: float,
                 selected_action_count: Optional[int] = None,
                 action_delta_summary: Optional[dict] = None):
        """Persist the complete postprocessed policy chunk.

        ``actions`` are the absolute targets received from PolicyServer, after
        its checkpoint postprocessor.  Keeping all 16 dimensions and all chunk
        frames is intentional: truncating to five 8-D samples hid the left arm
        and made relative/absolute-action failures impossible to diagnose.
        """
        if not self.enabled:
            return
        with self._lock:
            record = {
                "run_id": run_id,
                "step": step,
                "elapsed_ms": elapsed_ms,
                "num_actions": len(actions),
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            }
            if selected_action_count is not None:
                record["selected_action_count"] = int(selected_action_count)
            if action_delta_summary is not None:
                record["action_delta_summary"] = action_delta_summary
            if self._log_data:
                record["instruction"] = instruction
                state_dict = {k: v for k, v in obs.items()
                              if not isinstance(v, np.ndarray) and k != "task"}
                record["state"] = _truncate_state(state_dict)
                record["postprocessed_actions"] = [
                    list(a.get("action", [])) for a in actions if "action" in a
                ]
            self._write(record)
            if self._save_images:
                self._save_policy_output(
                    run_id,
                    step,
                    record.get("postprocessed_actions", []),
                    selected_action_count,
                    action_delta_summary,
                )

    # -- Internals --

    def _write(self, record: dict):
        self._rotate()
        try:
            line = _fast_dumps(record, default=_json_default)
            with open(self._current_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            warning(f"[InferenceLog] Write error: {e}")

    def _rotate(self):
        if (self._current_log_path is None
                or not os.path.exists(self._current_log_path)
                or os.path.getsize(self._current_log_path) >= self._max_size):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_seq += 1
            fname = f"inference_{ts}_{self._log_seq:03d}.log"
            self._current_log_path = os.path.join(self._log_dir, fname)

    def _save_step_images(self, run_id: str, step: int, obs: dict, image_keys: set):
        step_dir = os.path.join(self._image_dir, run_id)
        os.makedirs(step_dir, exist_ok=True)

        state_dict = {k: v for k, v in obs.items() if k not in image_keys and k != "task"}
        state_path = os.path.join(step_dir, f"step_{step:04d}.json")
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, ensure_ascii=False, default=_json_default, indent=2)
        except Exception:
            pass

        for key in sorted(image_keys):
            img_arr = obs[key]
            if img_arr is None or img_arr.size == 0:
                continue
            try:
                Image.fromarray(img_arr.astype(np.uint8)).save(
                    os.path.join(step_dir, f"step_{step:04d}_{key}.jpg"), quality=85)
            except Exception:
                pass

    def _save_policy_output(self, run_id: str, step: int, actions: list,
                            selected_action_count: Optional[int],
                            action_delta_summary: Optional[dict]):
        """Augment step JSON without changing its existing flat state keys."""
        step_dir = os.path.join(self._image_dir, run_id)
        state_path = os.path.join(step_dir, f"step_{step:04d}.json")
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            payload = {}
        payload["_policy_output"] = {
            "action_semantics": "absolute_postprocessed",
            "received_action_count": len(actions),
            "selected_action_count": selected_action_count,
            "action_delta_summary": action_delta_summary,
            "actions": actions,
        }
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, default=_json_default, indent=2)
        except Exception as exc:
            warning(f"[InferenceLog] policy output write error: {exc}")

    def _cleanup_old_runs(self):
        if self._max_image_runs <= 0:
            return
        try:
            entries = sorted(os.listdir(self._image_dir))
            while len(entries) > self._max_image_runs:
                oldest = entries.pop(0)
                old_path = os.path.join(self._image_dir, oldest)
                if os.path.isdir(old_path):
                    import shutil
                    shutil.rmtree(old_path, ignore_errors=True)
        except Exception:
            pass
