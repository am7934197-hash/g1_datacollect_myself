"""
Galbot VLA Client Server — VLA-Only Mode.

职责: 获取观测 → VLA 推理 → 执行动作。无导航、无放置、无任务流水线。

配置: vla_config.yaml（任务参数；固定硬件/schema/安全默认值由 g1_defaults.py 展开）
用法: python server.py [--config vla_config.yaml]
"""

from __future__ import annotations

import os
import json
import signal
import sys
import time
import yaml
import threading
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict

import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

from robot_controller import RobotController
from vla_client import VLAClient
from motion_processing import RTCController
import inference_logger as logger
from inference_logger import VLAInferenceLogger
from image_preprocessor import ImagePreprocessor
from g1_defaults import (
    STATE_NAMES, TRAINING_FPS, TRAINING_LETTERBOX_PAD_TBLR,
    expand_config, validate_checkpoint_schema,
    format_checkpoint_report, inspect_checkpoint_files, resolve_checkpoint_path,
)
from takeover import EpisodeControl, run_continuous_collection
from hil_capture import CaptureBridge


# ------------------------------------------------------------------
# 配置加载
# ------------------------------------------------------------------

def load_config(config_path: str = "vla_config.yaml") -> dict:
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return expand_config(yaml.safe_load(f))


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "vla_config.yaml"
if len(sys.argv) > 1 and sys.argv[1] == "--config":
    CONFIG_PATH = sys.argv[2]
if not os.path.isabs(CONFIG_PATH):
    CONFIG_PATH = os.path.join(_SCRIPT_DIR, CONFIG_PATH)
CONFIG_PATH = os.path.abspath(CONFIG_PATH)

CFG = load_config(CONFIG_PATH)

ROBOT_CFG = CFG.get("robot", {})
VLA_CFG = {**CFG, **(CFG.get("grpc", {}))}  # 展平给 VLAClient
SERVER_CFG = CFG.get("server", {})
HIL_CFG = CFG.get("hil") or {}


HIL_SESSION_CFG = dict(HIL_CFG.get("continuous") or {})
HIL_SESSION_CFG["_path"] = CONFIG_PATH + "#robot.reset_pose"

REQUIRED_CAMERAS = [
    key.rsplit(".", 1)[-1]
    for key, feature in (CFG.get("observation_features") or {}).items()
    if (feature or {}).get("dtype") == "image"
]
REQUIRED_STATE_NAMES = list(
    ((CFG.get("observation_features") or {}).get("observation.state") or {}).get("names")
    or ROBOT_CFG.get("state_feature_names")
    or []
)


def _log_runtime_identity():
    logger.info(f"script={os.path.abspath(__file__)}")
    logger.info(f"config={CONFIG_PATH}")
    logger.info(f"profile={CFG.get('profile')}")
    logger.info(f"pretrained_name_or_path={CFG.get('pretrained_name_or_path')}")
    logger.info(f"observation_features={list((CFG.get('observation_features') or {}).keys())}")
    logger.info(f"robot.state_feature_names={ROBOT_CFG.get('state_feature_names')}")
    logger.info(f"observation_mapping={CFG.get('observation_mapping')}")
    logger.info(
        "hil takeover_key=%s success_key=%s failure_key=%s sidecar_dir=%s"
        % (
            HIL_CFG.get("takeover_key", "i"),
            HIL_CFG.get("success_key", "s"),
            HIL_CFG.get("failure_key", "f"),
            HIL_CFG.get("sidecar_dir", "logs/hil_episodes"),
        )
    )
    logger.info(
        "continuous config=%s enabled=%s max_episodes=%s reset_time_s=%s"
        % (
            CONFIG_PATH + "#controls/dataset",
            HIL_SESSION_CFG.get("enabled", False),
            HIL_SESSION_CFG.get("max_episodes", 1),
            HIL_SESSION_CFG.get("reset_time_s", 10.0),
        )
    )


def validate_runtime_config():
    """Fail before robot motion when config and checkpoint semantics disagree."""
    features = CFG.get("observation_features") or {}
    expected_image_features = {f"observation.images.{name}" for name in REQUIRED_CAMERAS}
    configured_image_features = {
        key for key, value in features.items() if (value or {}).get("dtype") == "image"
    }
    if not REQUIRED_CAMERAS or configured_image_features != expected_image_features:
        raise ValueError(
            "observation image schema mismatch: "
            f"configured={sorted(configured_image_features)}, "
            f"expected={sorted(expected_image_features)}"
        )

    robot_state_names = list(ROBOT_CFG.get("state_feature_names") or [])
    if robot_state_names != STATE_NAMES or REQUIRED_STATE_NAMES != STATE_NAMES:
        raise ValueError(
            "23-D state order mismatch with the approved checkpoint schema: "
            f"observation={REQUIRED_STATE_NAMES}, robot={robot_state_names}, expected={STATE_NAMES}"
        )

    expected_action_names = list(CFG.get("profile_action_names") or [])
    action_dim = int(CFG.get("profile_action_dim", 0) or 0)
    robot_action_dim = int(ROBOT_CFG.get("vla_action_dim", 0) or 0)
    if action_dim not in (16, 23) or len(expected_action_names) != action_dim:
        raise ValueError(
            f"profile action schema is invalid: dim={action_dim}, names={expected_action_names}"
        )
    if robot_action_dim != action_dim:
        raise ValueError(
            f"robot.vla_action_dim={robot_action_dim} does not match profile action_dim={action_dim}"
        )

    control = ROBOT_CFG.get("control") or {}
    if control.get("action_semantics", "absolute") != "absolute":
        raise ValueError(
            "PI05 policy postprocessor returns absolute joint targets; "
            "robot.control.action_semantics must be 'absolute'"
        )
    configured_fps = float(CFG.get("fps", 0) or 0)
    control_fps = float(control.get("fps", 0) or 0)
    if abs(configured_fps - TRAINING_FPS) > 1e-6 or abs(control_fps - TRAINING_FPS) > 1e-6:
        raise ValueError(
            f"training/control frequency mismatch: training={TRAINING_FPS}, "
            f"top_level={configured_fps}, robot.control={control_fps}"
        )
    control_dt = float(control.get("dt", 0) or 0)
    if control_dt <= 0 or abs(control_dt - 1.0 / TRAINING_FPS) > 0.002:
        raise ValueError(
            f"robot.control.dt={control_dt} is incompatible with {TRAINING_FPS} Hz training data"
        )

    image_shapes = {
        tuple(int(v) for v in (feature or {}).get("shape", ())[:2])
        for feature in features.values()
        if (feature or {}).get("dtype") == "image"
    }
    if len(image_shapes) != 1 or any(len(shape) != 2 for shape in image_shapes):
        raise ValueError(f"all configured camera features must share one HWC size, got {image_shapes}")
    configured_image_hw = next(iter(image_shapes))
    image_cfg = CFG.get("image_preprocess") or {}
    target_hw = tuple(int(x) for x in image_cfg.get("target_size", ()))
    if (
        not image_cfg.get("enabled", False)
        or not image_cfg.get("keep_aspect_ratio", False)
        or target_hw != configured_image_hw
    ):
        raise ValueError(
            "image_preprocess must letterbox native frames: enabled=true, "
            "keep_aspect_ratio=true, "
            f"target_size={list(configured_image_hw)}; "
            f"got {image_cfg}"
        )

    rtc_cfg = CFG.get("rtc") or {}
    if rtc_cfg.get("enabled", False):
        if HIL_CFG.get("shadow_policy_during_intervention", True):
            raise ValueError(
                "hil.shadow_policy_during_intervention currently requires rtc.enabled=false; "
                "RTC may still have an in-flight inference when takeover is latched"
            )
        chunk_size = int(CFG.get("actions_per_chunk", 0) or 0)
        threshold = int(rtc_cfg.get("pre_infer_threshold", -1))
        if chunk_size <= 0 or not 0 < threshold < chunk_size:
            raise ValueError(
                f"RTC threshold must be inside the action chunk: threshold={threshold}, chunk={chunk_size}"
            )
        execution_horizon = int(rtc_cfg.get("execution_horizon", 0) or 0)
        inference_delay = int(rtc_cfg.get("inference_delay", 0) or 0)
        if not 0 < execution_horizon <= chunk_size:
            raise ValueError(
                "RTC execution_horizon must be inside the action chunk: "
                f"horizon={execution_horizon}, chunk={chunk_size}"
            )
        if not 0 <= inference_delay <= execution_horizon:
            raise ValueError(
                "RTC inference_delay must satisfy 0 <= delay <= execution_horizon: "
                f"delay={inference_delay}, horizon={execution_horizon}"
            )
        if int(rtc_cfg.get("warmup_inferences", 0) or 0) < 1:
            raise ValueError("RTC requires warmup_inferences >= 1 so compile output is never executed")
    interpolation = control.get("interpolation") or {}
    if interpolation.get("enabled", False):
        input_hz = float(interpolation.get("input_hz", CFG.get("fps", 30)))
        output_hz = float(interpolation.get("output_hz", 250))
        if input_hz <= 0 or output_hz <= 0:
            raise ValueError(
                f"invalid interpolation rates: input_hz={input_hz}, output_hz={output_hz}"
            )
        if output_hz < input_hz:
            logger.warning(
                f"interpolation downsamples policy frames: input_hz={input_hz}, "
                f"output_hz={output_hz}"
            )

    enabled_groups = {
        group.get("name")
        for group in (ROBOT_CFG.get("joint_groups") or [])
        if group.get("enabled", True)
    }
    if "head" in enabled_groups:
        raise ValueError(
            "robot.joint_groups.head must remain disabled: training head pose is fixed and "
            "the policy must not drive the head cameras"
        )

    if HIL_SESSION_CFG.get("enabled", False):
        reset_pose = HIL_SESSION_CFG.get("reset_pose") or {}
        expected_sizes = {
            str(group.get("name")): len(group.get("indices") or [])
            for group in (ROBOT_CFG.get("joint_groups") or [])
        }
        missing = [name for name in expected_sizes if name not in reset_pose]
        bad_sizes = {
            name: {"expected": size, "actual": len(reset_pose.get(name) or [])}
            for name, size in expected_sizes.items()
            if name in reset_pose and len(reset_pose.get(name) or []) != size
        }
        if missing or bad_sizes:
            raise ValueError(
                "invalid HIL session reset_pose: "
                f"missing={missing}, bad_sizes={bad_sizes}, config={HIL_SESSION_CFG.get('_path')}"
            )
        if float(HIL_SESSION_CFG.get("reset_time_s", 10.0)) < 0:
            raise ValueError("hil.continuous.reset_time_s must be >= 0")

    checkpoint_dir = resolve_checkpoint_path(CFG.get("pretrained_name_or_path"))
    CFG["pretrained_name_or_path"] = checkpoint_dir
    VLA_CFG["pretrained_name_or_path"] = checkpoint_dir
    file_info = inspect_checkpoint_files(checkpoint_dir)
    logger.info(format_checkpoint_report(file_info))
    checkpoint_config_path = os.path.join(checkpoint_dir, "config.json")
    if file_info["local_readable"] and os.path.isfile(checkpoint_config_path):
        with open(checkpoint_config_path, "r", encoding="utf-8") as f:
            checkpoint_cfg = json.load(f)
        validate_checkpoint_schema(CFG, checkpoint_cfg)

    logger.info(
        "Runtime config validation passed "
        f"({len(REQUIRED_CAMERAS)} cameras, 23-D state, {action_dim}-D action, absolute execution)"
    )


def validate_observation(obs: Optional[dict]) -> bool:
    """Reject obs that PolicyServer cannot consume. Never send a bad request."""
    if not obs:
        logger.error("observation validation failed: obs is None")
        return False

    missing = []
    for cam in REQUIRED_CAMERAS:
        img = obs.get(f"observation.images.{cam}", obs.get(cam))
        if img is None:
            missing.append(cam)
            continue
        shape = getattr(img, "shape", None)
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[-1] != 3:
            logger.error(f"camera '{cam}' bad ndarray: type={type(img)} shape={shape} dtype={getattr(img, 'dtype', None)}")
            missing.append(cam)
            continue
        feature = (CFG.get("observation_features") or {}).get(f"observation.images.{cam}") or {}
        configured_shape = tuple(int(v) for v in feature.get("shape", ()))
        expected_hw = configured_shape[:2]
        h, w = int(img.shape[0]), int(img.shape[1])
        if len(expected_hw) != 2 or (h, w) != expected_hw:
            logger.error(
                f"camera '{cam}' shape={shape}, expected HWC {configured_shape} after preprocessing"
            )
            missing.append(cam)
            continue
        if img.dtype != np.uint8:
            logger.error(f"camera '{cam}' dtype={img.dtype}, expected uint8, shape={shape}")
            missing.append(cam)
            continue
        if img.size == 0 or not np.any(img):
            logger.error(f"camera '{cam}' is empty/all-zero, refusing to send, shape={shape}")
            missing.append(cam)
            continue
        image_cfg = CFG.get("image_preprocess") or {}
        if image_cfg.get("enforce_training_geometry", True):
            from image_preprocessor import count_letterbox_black_bar_rows

            top, bottom = count_letterbox_black_bar_rows(img)
            expected_top, expected_bottom, _, _ = TRAINING_LETTERBOX_PAD_TBLR
            tolerance = int(image_cfg.get("black_bar_tolerance_px", 4))
            if abs(top - expected_top) > tolerance or abs(bottom - expected_bottom) > tolerance:
                logger.error(
                    f"camera '{cam}' letterbox bars top/bottom={top}/{bottom}, expected "
                    f"{expected_top}/{expected_bottom}±{tolerance}; refusing geometry that differs "
                    "from the configured training geometry"
                )
                missing.append(cam)
                continue

    if len(REQUIRED_STATE_NAMES) != 23:
        logger.error(
            f"state_feature_names length={len(REQUIRED_STATE_NAMES)}, expected 23: {REQUIRED_STATE_NAMES}"
        )
        logger.error(f"obs.keys()={list(obs.keys())}")
        return False

    finite_count = 0
    for name in REQUIRED_STATE_NAMES:
        if name not in obs:
            missing.append(name)
            continue
        try:
            val = float(obs[name])
        except (TypeError, ValueError):
            logger.error(f"state '{name}' is not numeric: {obs[name]!r}")
            missing.append(name)
            continue
        if not np.isfinite(val):
            logger.error(f"state '{name}' is not finite: {val}")
            missing.append(name)
            continue
        finite_count += 1

    missing = list(dict.fromkeys(missing))
    if missing or finite_count != 23:
        logger.error(
            f"observation validation failed, missing={missing}, "
            f"finite_state={finite_count}/23, obs.keys()={list(obs.keys())}"
        )
        return False
    return True


def summarize_postprocessed_action_delta(obs: dict, actions: list) -> dict:
    """Compare absolute PolicyServer targets with the observation state.

    Arm radians and gripper model units are reported separately.  A single
    max over all 16 dimensions is misleading because grippers use 0..100 while
    arm joints use radians.
    """
    state = np.asarray([obs[name] for name in REQUIRED_STATE_NAMES[:16]], dtype=np.float32)
    vectors = []
    invalid_actions = 0
    for item in actions:
        value = item.get("action") if isinstance(item, dict) else None
        arr = np.asarray(value, dtype=np.float32).reshape(-1) if value is not None else np.array([])
        if arr.size not in (16, 23) or not np.all(np.isfinite(arr)):
            invalid_actions += 1
            continue
        # Diagnostics concern only the commanded arms/grippers.  A legacy
        # 23-D place action additionally contains disabled leg/head targets.
        vectors.append(arr[:16])
    if not vectors:
        return {
            "action_semantics": "absolute_postprocessed",
            "valid_action_count": 0,
            "invalid_action_count": invalid_actions,
        }

    action_array = np.stack(vectors)
    abs_delta = np.abs(action_array - state[None, :])
    arm_indices = list(range(7)) + list(range(8, 15))
    arm_delta = abs_delta[:, arm_indices]
    gripper_delta = abs_delta[:, [7, 15]]
    gripper_scale = float((ROBOT_CFG.get("control") or {}).get("gripper_action_scale", 0.0012))
    return {
        "action_semantics": "absolute_postprocessed",
        "valid_action_count": len(vectors),
        "invalid_action_count": invalid_actions,
        "arm_abs_delta_rad": {
            "max": float(np.max(arm_delta)),
            "median": float(np.median(arm_delta)),
            "p95": float(np.quantile(arm_delta, 0.95)),
            "first_frame_max": float(np.max(arm_delta[0])),
            "last_frame_max": float(np.max(arm_delta[-1])),
            "per_joint_max": [float(x) for x in np.max(arm_delta, axis=0)],
        },
        "gripper_abs_delta_model": {
            "right_max": float(np.max(gripper_delta[:, 0])),
            "left_max": float(np.max(gripper_delta[:, 1])),
        },
        "gripper_abs_delta_m": {
            "right_max": float(np.max(gripper_delta[:, 0]) * gripper_scale),
            "left_max": float(np.max(gripper_delta[:, 1]) * gripper_scale),
        },
    }


def log_inferred_actions(run_id: str, instruction: str, step: int, obs: dict,
                         actions: list, elapsed_ms: float,
                         selected_action_count: Optional[int] = None,
                         *, is_intervention: bool = False,
                         execution_allowed: bool = True,
                         inference_kind: str = "control") -> dict:
    """Log the complete received chunk plus a concise hold/motion diagnostic."""
    summary = summarize_postprocessed_action_delta(obs, actions)
    arm = summary.get("arm_abs_delta_rad") or {}
    logger.info(
        "Policy action delta: semantics=absolute_postprocessed received=%d selected=%s "
        "arm_rad max=%.6f median=%.6f p95=%.6f first=%.6f last=%.6f "
        "gripper_model=%s gripper_m=%s",
        len(actions),
        selected_action_count,
        float(arm.get("max", float("nan"))),
        float(arm.get("median", float("nan"))),
        float(arm.get("p95", float("nan"))),
        float(arm.get("first_frame_max", float("nan"))),
        float(arm.get("last_frame_max", float("nan"))),
        summary.get("gripper_abs_delta_model"),
        summary.get("gripper_abs_delta_m"),
    )
    inference_logger.log_step(
        run_id=run_id,
        instruction=instruction,
        step=step,
        obs=obs,
        actions=actions,
        elapsed_ms=elapsed_ms,
        selected_action_count=selected_action_count,
        action_delta_summary=summary,
    )
    if capture is not None and capture.directory and os.path.basename(capture.directory) == run_id:
        try:
            with _archive_log_lock, open(
                os.path.join(capture.directory, "policy.jsonl"), "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps({
                    "run_id": run_id, "step": step, "received_at": time.time(),
                    "actions": actions, "selected_action_count": selected_action_count,
                    "is_intervention": bool(is_intervention),
                    "state": "INTERVENTION" if is_intervention else "AUTO",
                    "action_source": "human" if is_intervention else "policy",
                    "execution_allowed": bool(execution_allowed),
                    "policy_action_valid": bool(actions),
                    "inference_kind": inference_kind,
                    "collector_policy_id": (
                        "human" if is_intervention else str(
                            HIL_CFG.get("collector_policy_id")
                            or CFG.get("pretrained_name_or_path")
                        )
                    ),
                    "semantics": "complementary_info.policy_action; never execution confirmation",
                }, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n")
        except Exception:
            episode = _get_episode()
            if episode:
                episode.finish(0, "capture_fault")
            raise
    return summary


# ------------------------------------------------------------------
# 全局状态
# ------------------------------------------------------------------

robot: Optional[RobotController] = None
vla: Optional[VLAClient] = None
_stop_event = threading.Event()  # 用于中断正在运行的推理循环
_session_stop_event = threading.Event()  # 结束整次连续 /execute 会话
_recovery_in_progress = threading.Event()  # 右方向键恢复期间拒绝新的 /execute
_recovery_guard = threading.Lock()
_episode_lock = threading.Lock()
_current_episode: Optional[EpisodeControl] = None
_execute_lock = threading.Lock()
capture: Optional[CaptureBridge] = None
_archive_log_lock = threading.Lock()
_session_state_lock = threading.Lock()
_session_state = {
    "active": False,
    "phase": "IDLE",
    "episode_index": None,
    "episodes_completed": 0,
    "episode_limit": None,
    "environment_reset_remaining_s": None,
    "last_episode": None,
    "error": None,
}
inference_logger = VLAInferenceLogger(CFG.get("inference_log", {}))
image_preprocessor = ImagePreprocessor(CFG.get("image_preprocess", {}))
logger.info(
    "image_preprocess enabled=%s target_size=[%s, %s] keep_aspect_ratio=%s "
    "crop_to_training_aspect=%s training_hw=%s "
    "emulate_data_collection=%s scheme=%s cameras=%s zoom=%s out_wh=%s "
    "(client letterbox 224 uint8; PolicyServer /255 + resize_with_pad + *2-1)"
    % (
        image_preprocessor.enabled,
        image_preprocessor._target_h,
        image_preprocessor._target_w,
        image_preprocessor._keep_ratio,
        image_preprocessor._crop_to_training,
        list(image_preprocessor._training_hw),
        image_preprocessor._emulate_dc,
        image_preprocessor._emulate_scheme,
        sorted(image_preprocessor._emulate_cameras),
        image_preprocessor._emulate_zoom,
        list(image_preprocessor._emulate_out_wh),
    )
)

# ------------------------------------------------------------------
# 生命周期
# ------------------------------------------------------------------

def _shutdown_robot_with_timeout(robot: RobotController, timeout: float = 5.0):
    """在独立线程中关闭机器人，超时后强制退出，防止 SDK 阻塞。"""
    result = {"done": False}

    def _do_shutdown():
        try:
            robot.shutdown()
            result["done"] = True
        except Exception as e:
            logger.warning(f"Robot shutdown error: {e}")

    t = threading.Thread(target=_do_shutdown, daemon=True)
    t.start()
    t.join(timeout)
    if not result["done"]:
        logger.warning(f"Robot shutdown timed out after {timeout}s, forcing exit")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot, vla, capture
    _log_runtime_identity()
    validate_runtime_config()

    # ---- 初始化机器人 ----
    robot = RobotController(ROBOT_CFG)
    if robot.is_available():
        robot.init()
        time.sleep(2.0)
        if not (HIL_CFG.get("capture") or {}).get("enabled", False):
            robot.set_init_pose()
        else:
            logger.info("HIL capture enabled: startup does not command initial pose")
    else:
        logger.warning("Robot SDK not available, running in simulation mode")

    # ---- 初始化 VLA 客户端 ----
    vla = VLAClient(VLA_CFG)
    if vla.connect():
        auto_setup = VLA_CFG.get("auto_setup_policy", True)
        force_setup = VLA_CFG.get("force_setup_policy", False)
        if auto_setup:
            if force_setup:
                logger.info("force_setup_policy=true, sending setup with configured model path")
                if not vla.setup_policy():
                    logger.error("Policy setup failed; /execute will remain disabled")
            elif vla.check_policy_ready():
                logger.info("Policy already loaded on remote, skipping setup")
            else:
                logger.info("Policy not loaded, triggering setup...")
                if not vla.setup_policy():
                    logger.error("Policy setup failed; /execute will remain disabled")
        logger.info("Connected to PolicyServer")
    else:
        logger.warning("Failed to connect to PolicyServer")

    if (HIL_CFG.get("capture") or {}).get("enabled", False):
        capture = CaptureBridge(HIL_CFG["capture"], on_key=_physical_takeover)
        capture.start()

    yield

    # ---- 清理（带超时，防止 SDK 阻塞导致进程无法退出） ----
    logger.info("Shutting down...")
    _session_stop_event.set()
    _stop_event.set()
    _persist_episode_on_process_exit("stop")
    if capture is not None:
        capture.close()
    if robot is not None:
        _shutdown_robot_with_timeout(robot, timeout=5.0)
    if vla is not None:
        vla.disconnect()
    logger.info("Cleanup done, exiting")
    os._exit(0)  # 强制退出，绕过 SDK 残留的非 daemon 线程


app = FastAPI(title="Galbot VLA Client", lifespan=lifespan)


# ------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------

class VLARequest(BaseModel):
    instruction: Optional[str] = None
    max_steps: Optional[int] = None
    continuous: Optional[bool] = None
    max_episodes: Optional[int] = None


class VLAResponse(BaseModel):
    success: bool
    message: str
    details: Optional[dict] = None


# ------------------------------------------------------------------
# VLA 推理循环
# ------------------------------------------------------------------

# Robot 操作锁（RTC 模式下主线程下发动作 vs 异步线程获取观测）
_robot_lock = threading.Lock()


def _get_episode() -> Optional[EpisodeControl]:
    with _episode_lock:
        return _current_episode


def _append_episode_event(episode: EpisodeControl, event: str, **fields) -> None:
    """Append a sparse control event beside the native streams."""
    if capture is None or not capture.directory:
        return
    if os.path.basename(capture.directory) != episode.run_id:
        return
    payload = {
        "schema_version": episode.schema_version,
        "event": event,
        "run_id": episode.run_id,
        "wall_time": time.time(),
        "monotonic_ns": time.monotonic_ns(),
        "mode": episode.mode,
        "is_intervention": episode.intervened,
        **fields,
    }
    try:
        with _archive_log_lock:
            with open(os.path.join(capture.directory, "events.jsonl"), "a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    except Exception:
        episode.finish(0, "capture_fault")
        raise


def _physical_takeover(event):
    episode = _get_episode()
    if episode is not None and not episode.finished:
        if event["status"] in (1, 3):
            # A physical button press is already the start of manual control;
            # only the terminal/API takeover uses the preparation window.
            if not episode.intervened and episode.request_takeover(grace_s=0.0):
                _append_episode_event(
                    episode, "intervention_start", source="physical_button",
                    key=event["key"], status=event["status"], source_ns=event["source_ns"],
                )
        logger.info("HIL physical key=%s status=%s source_ns=%s",
                    event["key"], event["status"], event["source_ns"])


def _begin_episode(run_id: str, instruction: str) -> EpisodeControl:
    global _current_episode
    sidecar_dir = HIL_CFG.get("sidecar_dir", "logs/hil_episodes")
    if not os.path.isabs(sidecar_dir):
        sidecar_dir = os.path.join(_SCRIPT_DIR, sidecar_dir)
    episode = EpisodeControl(
        run_id=run_id,
        instruction=instruction,
        sidecar_dir=sidecar_dir,
        default_success_on_timeout=int(HIL_CFG.get("default_success_on_timeout", 0) or 0),
        collector_policy_id=str(HIL_CFG.get("collector_policy_id") or CFG.get("pretrained_name_or_path")),
        takeover_grace_s=float(HIL_CFG.get("takeover_grace_s", 0.0) or 0.0),
    )
    with _episode_lock:
        _current_episode = episode
    return episode


def _clear_episode(episode: Optional[EpisodeControl]) -> None:
    global _current_episode
    with _episode_lock:
        if _current_episode is episode:
            _current_episode = None


def _finalize_episode(episode: EpisodeControl, *, default_reason: str = "timeout") -> Optional[str]:
    if not episode.finished:
        success = episode.default_success_on_timeout if default_reason == "timeout" else 0
        episode.finish(success, default_reason)
    try:
        if capture is not None and capture.directory:
            directory = capture.directory
            try:
                _append_episode_event(
                    episode, "episode_end", episode_success=episode.episode_success,
                    end_reason=episode.end_reason,
                )
            except Exception as exc:
                logger.error("Failed to append HIL episode_end event: %s", exc)
            try:
                result = capture.end()
                episode.capture_result = {"directory": directory, "result": result}
            except Exception as exc:
                episode.capture_result = {"directory": directory, "error": str(exc)}
                logger.error("HIL archive finalization failed: %s", exc)
        return episode.write_sidecar()
    finally:
        _clear_episode(episode)


def _hil_should_stop_chunk(episode: Optional[EpisodeControl]) -> bool:
    if _stop_event.is_set():
        return True
    if episode is None:
        return False
    if capture is not None and not capture.status()["ready"]:
        episode.finish(0, "capture_fault")
        return True
    return episode.intervened or episode.finished


def _await_episode_label_after_motion(episode: EpisodeControl, step: int) -> None:
    """Keep capture/episode active after policy motion reaches max_steps."""
    if episode.finished or not HIL_CFG.get("await_label_after_max_steps", True):
        return
    episode.mark_policy_motion_complete()
    _append_episode_event(episode, "policy_motion_complete", step=step)
    _reset_policy_motion_state("policy max_steps reached")
    logger.warning(
        "POLICY MOTION STOPPED run_id=%s step=%s; waiting for success/failure label",
        episode.run_id,
        step,
    )
    while not episode.finished:
        episode.set_step(step)
        if capture is not None and not capture.status()["ready"]:
            episode.finish(0, "capture_fault")
            break
        if _stop_event.is_set():
            episode.finish(0, "stop")
            break
        time.sleep(0.05)


def _reset_policy_motion_state(reason: str) -> None:
    if robot is None:
        return
    try:
        robot.reset_action_filter()
        interpolator = getattr(robot, "interpolator", None)
        if interpolator is not None and hasattr(interpolator, "reset"):
            interpolator.reset()
        logger.info(f"Reset action filter / interpolator after {reason}")
    except Exception as exc:
        logger.warning(f"Failed to reset action filter / interpolator after {reason}: {exc}")


def _request_takeover(source: str, step: Optional[int] = None) -> dict:
    episode = _get_episode()
    if episode is None:
        logger.warning(f"Takeover from {source} ignored: no active episode")
        return {"success": False, "message": "No active episode"}
    ok = episode.request_takeover(step)
    if ok:
        _append_episode_event(
            episode, "intervention_start", source=source, step=episode.takeover_step,
            takeover_grace_s=episode.takeover_grace_s,
            intervention_data_start_at=episode.intervention_data_start_at,
        )
        if episode.takeover_grace_s > 0:
            logger.warning(
                "POLICY STOPPED; operator preparation window %.1fs. "
                "These frames remain raw but will be excluded from training.",
                episode.takeover_grace_s,
            )

            def _announce_intervention_data_start() -> None:
                deadline = time.monotonic() + episode.takeover_grace_s
                last_reported_second = None
                while _get_episode() is episode and not episode.finished:
                    remaining = max(0.0, deadline - time.monotonic())
                    reported_second = int(remaining + 0.999)
                    if reported_second != last_reported_second and reported_second > 0:
                        logger.warning(
                            "TAKEOVER PREPARATION: human demonstration starts in %ss",
                            reported_second,
                        )
                        last_reported_second = reported_second
                    if remaining <= 0:
                        _append_episode_event(
                            episode, "intervention_data_start", source=source,
                            wall_time=episode.intervention_data_start_at,
                        )
                        logger.warning(
                            "TAKEOVER PREPARATION FINISHED: start human demonstration now"
                        )
                        return
                    time.sleep(min(0.1, remaining))

            threading.Thread(
                target=_announce_intervention_data_start,
                name="takeover-grace-timer", daemon=True,
            ).start()
    return {
        "success": True,
        "latched": ok,
        "intervened": episode.intervened,
        "run_id": episode.run_id,
        "message": "TAKEOVER START" if ok else "Already in INTERVENTION or episode finished",
    }


def _request_episode_finish(success: int, reason: str, source: str) -> dict:
    episode = _get_episode()
    if episode is None:
        logger.warning(f"Episode {reason} from {source} ignored: no active episode")
        return {"success": False, "message": "No active episode"}
    first = episode.finish(success, reason)
    if first:
        _append_episode_event(
            episode, "outcome_label", source=source,
            episode_success=int(success), end_reason=reason,
        )
    return {
        "success": True,
        "accepted": first,
        "run_id": episode.run_id,
        "episode_success": episode.episode_success,
        "end_reason": episode.end_reason,
        "message": f"Episode marked {reason}" if first else "Episode already finished",
    }


def _hil_key_byte(name: str, default: str) -> bytes:
    key = str(HIL_CFG.get(name, default) or default)
    if not key:
        key = default
    return key.encode("utf-8")[:1]


def _persist_episode_on_process_exit(reason: str = "stop") -> None:
    episode = _get_episode()
    if episode is None:
        return
    if not episode.finished:
        episode.finish(0, reason)
    try:
        _finalize_episode(episode, default_reason=reason)
    except Exception as exc:
        logger.warning(f"Failed to write sidecar on process exit: {exc}")
    _clear_episode(episode)


def _request_interactive_recovery(source: str) -> bool:
    """Stop policy execution and restore the configured initial pose safely.

    The recovery runs in a worker because it may wait for an in-flight command
    chunk to release ``_robot_lock``.  The process and PolicyServer connection
    remain alive so the next task can be started normally after recovery.
    Right-arrow is independent of the HIL latch: during INTERVENTION it must
    not send set_joint_commands / set_recovery_pose.
    """
    episode = _get_episode()
    if episode is not None and episode.intervened and not episode.finished:
        logger.warning(
            f"Right-arrow recovery ignored during INTERVENTION from {source}: "
            "would fight teleop on wbcs/target"
        )
        return False

    with _recovery_guard:
        if _recovery_in_progress.is_set():
            logger.warning(f"Recovery already in progress; ignoring duplicate request from {source}")
            return False
        _recovery_in_progress.set()

    logger.warning(f"Interactive recovery requested from {source}: stopping VLA and returning to initial pose")
    _session_stop_event.set()
    _stop_event.set()
    if vla is not None:
        try:
            vla.reset()
        except Exception as exc:
            logger.warning(f"Policy reset during recovery failed: {exc}")

    def _recover():
        try:
            if robot is None or robot.robot is None:
                logger.error("Recovery skipped: robot is not initialized")
                return
            # An executing chunk observes _stop_event and releases this lock.
            # No VLA command can interleave with the return-to-home command.
            with _robot_lock:
                robot.reset_action_filter()
                ok = robot.set_recovery_pose()
            if ok:
                logger.warning("Interactive recovery complete: initial pose reached, grippers fully open")
            else:
                logger.error("Interactive recovery failed while sending the initial pose")
        except Exception as exc:
            logger.exception(f"Interactive recovery crashed: {exc}")
        finally:
            _recovery_in_progress.clear()

    threading.Thread(target=_recover, name="interactive-recovery", daemon=True).start()
    return True


def _obs_meta() -> dict:
    if robot is None:
        now = time.time()
        return {"obs_capture_ts": now, "state_ts": now, "image_ts": {}}
    return dict(getattr(robot, "last_obs_meta", None) or {})


def _capture_and_preprocess():
    obs = robot.get_observation(features=CFG.get("observation_features"))
    if obs is None:
        return None, _obs_meta()
    obs = image_preprocessor.process(obs)
    return obs, _obs_meta()


def _set_session_state(**updates) -> None:
    with _session_state_lock:
        _session_state.update(updates)


def _get_session_state() -> dict:
    with _session_state_lock:
        return dict(_session_state)


def _wait_for_takeover_release(timeout_s: float) -> bool:
    """Do not fight the master arms while a physical takeover key is held."""
    if capture is None:
        return True
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        held = dict((capture.latest or {}).get("held") or {})
        if not any(held.values()):
            return True
        if _session_stop_event.is_set():
            return False
        if time.monotonic() >= deadline:
            logger.error("Cannot reset robot: takeover key still held after %.1fs", timeout_s)
            return False
        _set_session_state(phase="WAITING_TAKEOVER_RELEASE")
        time.sleep(0.1)


def _reset_between_collection_episodes(
    wait_for_environment: bool = True,
) -> tuple[bool, str]:
    """Reset policy and robot, optionally reserving time for manual scene reset."""
    if robot is None or robot.robot is None:
        return False, "Robot is not initialized"
    _recovery_in_progress.set()
    try:
        _set_session_state(phase="RESETTING_POLICY", environment_reset_remaining_s=None)
        if vla is None:
            return False, "PolicyServer client is not initialized"
        if not vla.reset():
            if bool((CFG.get("rtc") or {}).get("enabled", False)):
                return False, "PolicyServer reset failed"
            # Synchronous inference sends a complete fresh observation for every
            # request and has no client-side RTC session to preserve. Some
            # PolicyServer deployments expose ResetPolicy in protobuf but fail
            # the RPC at runtime. Continue safely here; the next episode's
            # warmup remains a fail-closed connectivity/inference check before
            # any robot command can be sent.
            logger.warning(
                "PolicyServer reset failed in synchronous mode; continuing with "
                "fresh observations. Next-episode warmup must pass before motion."
            )
        _reset_policy_motion_state("continuous episode boundary")

        release_timeout = float(
            HIL_SESSION_CFG.get("wait_takeover_release_s", 10.0) or 10.0
        )
        if not _wait_for_takeover_release(release_timeout):
            return False, "Takeover button was not released before robot reset"
        if _session_stop_event.is_set():
            return False, "Continuous session stopped"

        pose = HIL_SESSION_CFG.get("reset_pose") or {}
        settle_s = float(HIL_SESSION_CFG.get("pose_settle_s", 8.0) or 8.0)
        retries = int(HIL_SESSION_CFG.get("pose_retries", 2) or 2)
        _set_session_state(phase="RESETTING_ROBOT")
        logger.warning(
            "CONTINUOUS HIL: restoring reset pose from %s",
            HIL_SESSION_CFG.get("_path"),
        )
        with _robot_lock:
            robot.reset_action_filter()
            gripper_cycle = HIL_SESSION_CFG.get("reset_gripper_cycle") or {}
            if gripper_cycle.get("enabled", False):
                open_width = float(gripper_cycle.get("open_width", 0.12))
                logger.warning(
                    "CONTINUOUS HIL: opening both grippers to %.3fm before reset pose",
                    open_width,
                )
                opened = robot.set_named_pose(
                    {
                        "right_gripper": [open_width],
                        "left_gripper": [open_width],
                    },
                    timeout=float(gripper_cycle.get("open_settle_s", 2.0)),
                    retries=int(gripper_cycle.get("retries", 2)),
                    label="continuous_reset_open_grippers",
                    verify=True,
                )
                if not opened:
                    return False, "Robot failed to open grippers before reset pose"
            ok = robot.set_named_pose(
                pose,
                timeout=settle_s,
                retries=retries,
                label="continuous_episode_reset",
                verify=True,
            )
        if not ok:
            return False, "Robot failed to reach configured collection reset pose"

        if not wait_for_environment:
            logger.warning(
                "CONTINUOUS HIL: final robot reset complete; collection finished"
            )
            _set_session_state(environment_reset_remaining_s=None)
            return True, "final reset complete"

        reset_seconds = max(
            0.0, float(HIL_SESSION_CFG.get("reset_time_s", 10.0) or 0.0)
        )
        deadline = time.monotonic() + reset_seconds
        logger.warning(
            "CONTINUOUS HIL: robot reset complete; %.1fs to restore the environment",
            reset_seconds,
        )
        last_reported_second = None
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            reported_second = int(remaining + 0.999)
            if reported_second != last_reported_second:
                logger.warning(
                    "CONTINUOUS HIL: restore environment, next episode in %ss",
                    reported_second,
                )
                last_reported_second = reported_second
            _set_session_state(
                phase="ENVIRONMENT_RESET",
                environment_reset_remaining_s=round(remaining, 1),
            )
            if remaining <= 0:
                break
            if _session_stop_event.wait(min(0.1, remaining)):
                return False, "Continuous session stopped"
        _set_session_state(environment_reset_remaining_s=None)
        return True, "reset complete"
    finally:
        _recovery_in_progress.clear()


def _warmup_policy(instruction: str, warmup_n: int) -> dict:
    """Run compile/warmup inferences and DISCARD every returned chunk.

    The first torch.compile packet is ~2.9s. It must finish before motion starts.
    """
    if warmup_n <= 0:
        return {"success": True, "warmup_inferences": 0, "warmup_ms": []}
    logger.info(f"WARMUP: discarding {warmup_n} inference packet(s) before any robot motion")
    times_ms = []
    for i in range(warmup_n):
        with _robot_lock:
            obs, meta = _capture_and_preprocess()
        if obs is None or not validate_observation(obs):
            return {
                "success": False,
                "warmup_inferences": i,
                "warmup_ms": times_ms,
                "message": "Warmup observation invalid",
            }
        obs["task"] = instruction
        t0 = time.perf_counter()
        actions = vla.infer(obs, instruction, timestep=-1 - i)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)
        n_act = len(actions) if actions else 0
        logger.info(
            f"WARMUP infer {i + 1}/{warmup_n}: {elapsed_ms:.0f}ms actions={n_act} "
            f"DISCARDED (not queued, robot not moving) obs_ts={meta.get('obs_capture_ts')}"
        )
        if not actions:
            return {
                "success": False,
                "warmup_inferences": i + 1,
                "warmup_ms": times_ms,
                "message": "Warmup inference returned no actions",
            }
    return {"success": True, "warmup_inferences": warmup_n, "warmup_ms": times_ms}


def _run_shadow_inference(
    episode: EpisodeControl, instruction: str, timestep: int, run_id: str
) -> None:
    """Record counterfactual policy output during human control without executing it."""
    if episode.finished or not episode.intervened:
        return
    try:
        t0_obs = time.perf_counter()
        with _robot_lock:
            obs, _ = _capture_and_preprocess()
        obs_elapsed_ms = (time.perf_counter() - t0_obs) * 1000
        if obs is None:
            logger.warning("HIL shadow inference skipped: observation is None")
            return
        obs["task"] = instruction
        if not validate_observation(obs):
            logger.warning("HIL shadow inference skipped: observation schema invalid")
            return
        inference_logger.log_obs(
            run_id=run_id, step=timestep, obs=obs, elapsed_ms=obs_elapsed_ms
        )
        t0 = time.perf_counter()
        actions = vla.infer(obs, instruction, timestep=timestep)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if episode.finished:
            return
        if not actions:
            logger.warning("HIL shadow inference returned no actions at step %s", timestep)
            _append_episode_event(episode, "shadow_policy_empty", step=timestep)
            return
        log_inferred_actions(
            run_id, instruction, timestep, obs, actions, elapsed_ms,
            selected_action_count=0, is_intervention=True,
            execution_allowed=False, inference_kind="intervention_shadow",
        )
        logger.info(
            "HIL shadow policy recorded step=%s actions=%s; execution disabled",
            timestep, len(actions),
        )
    except Exception as exc:
        logger.warning("HIL shadow inference failed (teleop remains active): %s", exc)
        if not episode.finished:
            try:
                _append_episode_event(
                    episode, "shadow_policy_error", step=timestep, error=str(exc)
                )
            except Exception:
                logger.exception("Failed to record shadow inference error")


def _run_one_episode_owned(instruction: str, max_steps: int) -> dict:
    try:
        return _run_vla_owned(instruction, max_steps)
    finally:
        episode = _get_episode()
        if episode is not None:
            _finalize_episode(episode, default_reason="stop")


def run_vla(
    instruction: str,
    max_steps: int = 2000,
    continuous: Optional[bool] = None,
    max_episodes: Optional[int] = None,
) -> dict:
    if not _execute_lock.acquire(blocking=False):
        return {"success": False, "message": "Episode already running", "details": None}
    use_continuous = (
        bool(HIL_SESSION_CFG.get("enabled", False))
        if continuous is None else bool(continuous)
    )
    episode_limit = int(
        HIL_SESSION_CFG.get("max_episodes", 1)
        if max_episodes is None else max_episodes
    )
    try:
        if not use_continuous:
            return _run_one_episode_owned(instruction, max_steps)

        _session_stop_event.clear()
        _set_session_state(
            active=True, phase="STARTING_EPISODE", episode_index=1,
            episodes_completed=0, episode_limit=episode_limit,
            environment_reset_remaining_s=None,
            last_episode=None, error=None,
        )

        def _run_episode_for_session() -> dict:
            logger.warning(
                "CONTINUOUS HIL: starting episode %s",
                _get_session_state().get("episode_index"),
            )
            return _run_one_episode_owned(instruction, max_steps)

        return run_continuous_collection(
            run_episode=_run_episode_for_session,
            reset_between=_reset_between_collection_episodes,
            should_stop=_session_stop_event.is_set,
            update_state=_set_session_state,
            max_episodes=episode_limit,
            reset_before=_reset_between_collection_episodes,
            reset_after=lambda: _reset_between_collection_episodes(
                wait_for_environment=False
            ),
        )
    finally:
        try:
            episode = _get_episode()
            if episode is not None:
                _finalize_episode(episode, default_reason="stop")
        finally:
            if use_continuous and vla is not None:
                try:
                    vla.reset()
                except Exception as exc:
                    logger.warning("Final continuous-session policy reset failed: %s", exc)
            _set_session_state(
                active=False, phase="IDLE", episode_index=None,
                environment_reset_remaining_s=None,
            )
            _execute_lock.release()


def _run_vla_owned(instruction: str, max_steps: int = 2000) -> dict:
    """VLA 推理入口，根据 rtc.enabled 自动选择同步/RTC 模式。"""
    if robot is None or vla is None:
        return {"success": False, "message": "Server not initialized", "details": None}
    if _recovery_in_progress.is_set():
        return {"success": False, "message": "Recovery in progress; wait for initial pose", "details": None}
    if not vla.connected:
        return {"success": False, "message": "PolicyServer is not connected", "details": None}
    if not vla.policy_ready:
        return {"success": False, "message": "Policy setup has not completed successfully", "details": None}

    if capture is not None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        episode = _begin_episode(run_id, instruction)
        root = (HIL_CFG.get("capture") or {}).get("directory", "logs/hil_raw")
        if not os.path.isabs(root):
            root = os.path.join(_SCRIPT_DIR, root)
        try:
            capture.begin(os.path.join(root, run_id), {
                "schema_version": episode.schema_version,
                "run_id": run_id, "instruction": instruction, "started_at": episode.started_at,
                "checkpoint": CFG.get("pretrained_name_or_path"),
                "collector_policy_id": episode.collector_policy_id,
                "robot_config": ROBOT_CFG, "observation_features": CFG.get("observation_features"),
                "note": "Raw native protobuf units; target messages are submissions, not execution acknowledgements.",
                "evo_rl_mapping": {
                    "action": "resample wbc_info.joint_info.command_pos as the executed command",
                    "complementary_info.policy_action": "align policy.jsonl actions to frames",
                    "complementary_info.is_intervention": "events.jsonl intervention_start latch",
                    "complementary_info.state": "AUTO before latch, INTERVENTION after latch",
                    "complementary_info.collector_policy_id": "policy checkpoint before latch; human after latch",
                    "episode_success": "sidecar episode_success/outcome_label event",
                },
            })
            _append_episode_event(
                episode, "episode_start", instruction=instruction,
                collector_policy_id=episode.collector_policy_id,
            )
        except Exception as exc:
            episode.finish(0, "capture_not_ready")
            return {"success": False, "message": str(exc), "details": {"run_id": run_id}}

    rtc_cfg = CFG.get("rtc", {})
    # Compile/warmup must be the first operation after read-only diagnostics.
    # In particular, do not move the grippers before the multi-second compile
    # packet has completed and been discarded.
    warmup = _warmup_policy(instruction, int(rtc_cfg.get("warmup_inferences", 0) or 0))
    if not warmup.get("success", True):
        return {
            "success": False,
            "message": warmup.get("message", "Policy warmup failed"),
            "details": warmup,
        }

    if robot.robot is not None and capture is None:
        if not robot.set_training_start_grippers():
            return {
                "success": False,
                "message": "Failed to set training-start gripper widths (0.0 m) after warmup",
                "details": {"warmup": warmup},
            }

    if rtc_cfg.get("enabled", False):
        result = _run_vla_rtc(instruction, max_steps, rtc_cfg)
    else:
        result = _run_vla_sync(instruction, max_steps)

    details = result.get("details")
    if details is None:
        details = {}
        result["details"] = details
    details["warmup"] = warmup
    return result


def _attach_hil_details(details: dict, episode: EpisodeControl) -> dict:
    details.update({
        "run_id": episode.run_id,
        "sidecar_path": episode.sidecar_path,
        "episode_success": episode.episode_success,
        "end_reason": episode.end_reason,
        "intervened": episode.intervened,
        "takeover_at": episode.takeover_at,
        "takeover_step": episode.takeover_step,
        "mode": episode.mode,
        "capture": episode.capture_result,
    })
    return details


def _run_vla_sync(instruction: str, max_steps: int) -> dict:
    """同步推理模式（原有逻辑）。HIL: takeover 闩锁后停发，s/f 结束 episode。"""
    execute_count = CFG.get("execute_actions_per_chunk", 10)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    control = ROBOT_CFG.get("control", {})
    dt = control.get("dt", 0.033)
    episode = _get_episode() or _begin_episode(run_id, instruction)
    run_id = episode.run_id
    takeover_cleaned = False
    command_failed = False
    result = None
    step = 0
    total_actions = 0
    shadow_sequence = 0
    shadow_interval = max(0.1, float(HIL_CFG.get("shadow_policy_interval_s", 1.0) or 1.0))
    shadow_enabled = bool(HIL_CFG.get("shadow_policy_during_intervention", True))

    inference_logger.start_run(run_id, instruction)

    try:
        if hasattr(vla, "update_instruction"):
            vla.update_instruction(instruction)
            time.sleep(0.2)

        _stop_event.clear()
        robot.reset_action_filter()
        if hasattr(robot, "_pump_cameras"):
            logger.info("Pumping cameras on execute thread...")
            robot._pump_cameras(duration=5.0, interval=0.1)

        while step < max_steps:
            episode.set_step(step)
            _hil_should_stop_chunk(episode)
            if episode.finished:
                break
            if episode.intervened:
                if not takeover_cleaned:
                    _reset_policy_motion_state("TAKEOVER START")
                    takeover_cleaned = True
                if shadow_enabled:
                    shadow_sequence += 1
                    _run_shadow_inference(episode, instruction, step + shadow_sequence, run_id)
                    time.sleep(shadow_interval)
                else:
                    time.sleep(dt)
                continue
            if _stop_event.is_set():
                reason = "recovery" if _recovery_in_progress.is_set() else "stop"
                logger.info(f"Stop requested at step {step}, ending inference ({reason})")
                episode.finish(0, reason)
                vla.reset()
                break

            # ---- 1. 获取观测 ----
            t0_obs = time.perf_counter()
            obs = robot.get_observation(features=CFG.get("observation_features"))
            elapsed_obs_ms = (time.perf_counter() - t0_obs) * 1000

            if episode.finished or episode.intervened or _stop_event.is_set():
                continue

            if obs is None:
                logger.warning(f"Failed to get observation at step {step}, retrying...")
                time.sleep(dt)
                continue

            # ---- 1b. 图像预处理 ----
            t0_prep = time.perf_counter()
            obs = image_preprocessor.process(obs)
            elapsed_prep_ms = (time.perf_counter() - t0_prep) * 1000

            inference_logger.log_obs(run_id=run_id, step=step, obs=obs, elapsed_ms=elapsed_obs_ms)
            logger.info(f"Get obs: {elapsed_obs_ms:.0f}ms | preprocess: {elapsed_prep_ms:.0f}ms")

            obs["task"] = instruction

            if not validate_observation(obs):
                logger.warning(f"Skip infer at step {step}: observation schema invalid")
                time.sleep(dt)
                continue

            # ---- 2. VLA 推理 ----
            t0 = time.perf_counter()
            actions = vla.infer(obs, instruction, timestep=step)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if episode.finished:
                break
            if episode.intervened:
                logger.info(f"Takeover during infer at step {step}, discarding returned actions")
                if actions:
                    log_inferred_actions(
                        run_id, instruction, step, obs, actions, elapsed_ms,
                        selected_action_count=0, is_intervention=True,
                        execution_allowed=False, inference_kind="takeover_during_inference",
                    )
                continue
            if _stop_event.is_set():
                reason = "recovery" if _recovery_in_progress.is_set() else "stop"
                logger.info(f"Stop requested while inferring at step {step}, discarding returned actions")
                episode.finish(0, reason)
                vla.reset()
                break

            if not actions:
                logger.info(f"No actions returned at step {step}, ending")
                episode.finish(0, "stop")
                break

            # ---- 3. 提取动作向量 ----
            n = min(execute_count, len(actions))
            action_vectors = [a["action"] for a in actions[:n] if "action" in a]

            log_inferred_actions(
                run_id, instruction, step, obs, actions, elapsed_ms,
                selected_action_count=len(action_vectors),
                is_intervention=False, execution_allowed=True, inference_kind="control",
            )

            # ---- 4. 下发动作 ----
            with _robot_lock:
                executed = robot.execute_action_chunk(
                    action_vectors,
                    dt=dt,
                    stop_flag=lambda: _hil_should_stop_chunk(episode),
                    command_guard=episode.command_guard,
                )
            if episode.finished or episode.intervened:
                continue
            if not executed:
                logger.error(f"Action execution failed at step {step}; stopping run")
                episode.finish(0, "stop")
                command_failed = True
                break
            total_actions += len(action_vectors)

            logger.info(f"Step {step}: {len(action_vectors)} actions, {elapsed_ms:.0f}ms")
            step += n

        if step >= max_steps and not episode.finished:
            _await_episode_label_after_motion(episode, step)

        details = {
            "run_id": run_id,
            "steps": step,
            "total_actions_executed": total_actions,
            "max_steps": max_steps,
        }
        if command_failed:
            result = {
                "success": False,
                "message": "Robot rejected or failed to execute action chunk",
                "details": details,
            }
        else:
            result = {
                "success": True,
                "message": "VLA loop completed",
                "details": details,
            }
        return result
    finally:
        sidecar = _finalize_episode(episode, default_reason="timeout")
        inference_logger.end_run(run_id)
        if result is not None:
            _attach_hil_details(result.setdefault("details", {}), episode)
        logger.info(f"HIL episode finalized sidecar={sidecar}")


def _run_vla_rtc(instruction: str, max_steps: int, rtc_cfg: dict) -> dict:
    """Pi0.5 model-native RTC asynchronous inference mode.

    流程:
      1. 首次推理（warmup 已在 run_vla 丢弃 compile 首包）→ fill buffer
      2. 主线程按 30 Hz 单调时钟逐帧取 buffer → 下发（sleep 不含处理时间叠加）
      3. buffer 剩余 ≤ pre_infer_threshold 时，后台线程异步 trigger
      4. Send the unexecuted prefix to the remote RTC processor
      5. Replace the future queue after dropping frames stale since capture

    HIL: intervention 后停止从 buffer 下发、停止 async infer，直到 episode 结束。
    不要在接管后 rtc.reset() 再跑 policy。
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    control = ROBOT_CFG.get("control", {})
    fps = float(control.get("fps", CFG.get("fps", 30)) or 30)
    period = 1.0 / fps
    dt = float(control.get("dt", period))
    episode = _get_episode() or _begin_episode(run_id, instruction)
    run_id = episode.run_id
    takeover_cleaned = False
    result = None
    step = 0
    async_thread = None
    failure_message = None

    inference_logger.start_run(run_id, instruction)
    rtc = RTCController(rtc_cfg)

    try:
        if hasattr(vla, "update_instruction"):
            vla.update_instruction(instruction)
            time.sleep(0.2)

        _stop_event.clear()
        robot.reset_action_filter()

        logger.info("RTC: running initial inference (post-warmup, this chunk WILL be executed)...")
        t0_first = time.perf_counter()
        with _robot_lock:
            obs, meta = _capture_and_preprocess()
        if obs is None:
            logger.error("RTC: initial observation is None")
            episode.finish(0, "stop")
            result = {"success": False, "message": "Invalid observation", "details": {"run_id": run_id}}
            return result
        inference_logger.log_obs(run_id=run_id, step=0, obs=obs, elapsed_ms=(time.perf_counter() - t0_first) * 1000)
        obs["task"] = instruction

        if not validate_observation(obs):
            logger.error("RTC: initial observation schema invalid")
            episode.finish(0, "stop")
            result = {"success": False, "message": "Invalid observation", "details": {"run_id": run_id}}
            return result

        initial_rtc_context = {
            "enabled": True,
            "session_id": rtc.session_id,
            "infer_seq": 0,
            "prev_chunk_left_over": None,
            "prev_chunk_semantics": "absolute_postprocessed",
            "inference_delay": 0,
            "execution_horizon": int(rtc_cfg.get("execution_horizon", 10)),
        }
        try:
            initial_result = vla.infer_rtc(
                obs, instruction, timestep=0, rtc_context=initial_rtc_context
            )
        except RuntimeError as exc:
            logger.error("RTC: PolicyServer native-RTC handshake failed: %s", exc)
            episode.finish(0, "stop")
            result = {
                "success": False,
                "message": str(exc),
                "details": {"run_id": run_id},
            }
            return result
        actions, initial_server_meta = initial_result or (None, {})
        if episode.finished:
            result = {"success": True, "message": "Episode finished during initial inference", "details": {"run_id": run_id}}
            return result
        if episode.intervened:
            logger.info("RTC: takeover during initial inference; discarding actions")
        elif _stop_event.is_set():
            logger.info("RTC: stop requested while initial inference was running; discarding actions")
            reason = "recovery" if _recovery_in_progress.is_set() else "stop"
            episode.finish(0, reason)
            vla.reset()
            result = {"success": True, "message": "VLA stopped for recovery", "details": {"run_id": run_id}}
            return result
        elif not actions:
            logger.error("RTC: initial inference returned no actions")
            episode.finish(0, "stop")
            result = {"success": False, "message": "Initial inference failed", "details": {"run_id": run_id}}
            return result
        else:
            action_vectors = [a["action"] for a in actions if "action" in a]
            obs_age_s = time.time() - float(meta.get("obs_capture_ts") or time.time())
            stale_frames = int(round(max(0.0, obs_age_s) * fps))
            if stale_frames > 0:
                if stale_frames >= len(action_vectors):
                    logger.error(
                        f"RTC: initial chunk entirely stale obs_age={obs_age_s * 1000:.0f}ms "
                        f"stale_frames={stale_frames} chunk={len(action_vectors)}"
                    )
                    episode.finish(0, "stop")
                    result = {
                        "success": False,
                        "message": "Initial chunk entirely stale after inference latency",
                        "details": {"run_id": run_id},
                    }
                    return result
                logger.warning(
                    f"RTC: initial chunk obs_age={obs_age_s * 1000:.0f}ms -> skip first {stale_frames} actions "
                    f"(do not execute chunk[0] against a {stale_frames}-frame-old state)"
                )
                action_vectors = action_vectors[stale_frames:]
            log_inferred_actions(
                run_id, instruction, 0, obs, actions,
                elapsed_ms=(time.perf_counter() - t0_first) * 1000,
                selected_action_count=len(action_vectors),
            )
            fusion0 = rtc.add_chunk(action_vectors)
            fusion0["skipped"] = stale_frames
            fusion0["start_index"] = stale_frames
            fusion0["mode"] = "replace_skip_stale" if stale_frames else "replace"
            first_infer_ms = (time.perf_counter() - t0_first) * 1000
            initial_delay_frames = max(0, int(round(first_infer_ms / 1000.0 * fps)))
            rtc.set_inference_delay_frames(
                int(initial_server_meta.get("inference_delay_frames", initial_delay_frames))
            )
            logger.info(
                f"RTC: initial inference done in {first_infer_ms:.0f}ms, buffer={rtc.buffer_remaining()} frames "
                f"fusion={fusion0}"
            )

        next_t = time.perf_counter()

        while step < max_steps:
            episode.set_step(step)
            _hil_should_stop_chunk(episode)
            if episode.finished:
                break
            if episode.intervened:
                if not takeover_cleaned:
                    rtc.close_session()
                    _reset_policy_motion_state("TAKEOVER START")
                    takeover_cleaned = True
                time.sleep(period)
                continue
            if _stop_event.is_set():
                rtc.close_session()
                reason = "recovery" if _recovery_in_progress.is_set() else "stop"
                logger.info(f"RTC: stop requested at step {step}")
                episode.finish(0, reason)
                vla.reset()
                break

            if rtc.new_chunk_ready:
                if episode.intervened or episode.finished:
                    rtc.consume_new_chunk()
                    logger.info("RTC: discarding returned chunk after HIL intervention/finish")
                else:
                    report, chunk_meta = rtc.consume_new_chunk()
                    if (chunk_meta or {}).get("fatal_error"):
                        failure_message = str(chunk_meta["fatal_error"])
                        episode.finish(0, "stop")
                        break
                    logger.info(
                        "RTC: installed native chunk step=%s buffer=%s mode=%s "
                        "skipped=%s prefix_frames_used=%s",
                        step, rtc.buffer_remaining(), (report or {}).get("mode"),
                        (report or {}).get("skipped"),
                        (report or {}).get("prefix_frames_used"),
                    )
                if async_thread is not None:
                    async_thread.join(timeout=1.0)
                    async_thread = None

            action = rtc.get_next_action()
            if action is None:
                if rtc.infer_in_progress:
                    rtc.record_hold_frame()
                    logger.warning("RTC: awaiting native chunk at step=%s; holding target", step)
                    time.sleep(period)
                    continue
                logger.warning(f"RTC: buffer empty at step {step}, ending")
                failure_message = "RTC action buffer underflow"
                episode.finish(0, "stop")
                break

            with _robot_lock:
                executed = robot.execute_action_chunk(
                    [action],
                    dt=dt,
                    skip_interp=True,
                    sleep=False,
                    stop_flag=lambda: _hil_should_stop_chunk(episode),
                    command_guard=episode.command_guard,
                )
            if episode.finished or episode.intervened:
                continue
            if not executed:
                logger.error(f"RTC: action execution failed at step {step}; stopping")
                failure_message = f"Robot rejected VLA action at RTC step {step}"
                episode.finish(0, "stop")
                _stop_event.set()
                break

            logger.info(f"RTC step {step}: buffer_remaining={rtc.buffer_remaining()}")
            step += 1

            if rtc.should_trigger_inference() and not episode.intervened and not episode.finished:
                logger.info(
                    f"RTC: triggering async inference at step {step} "
                    f"(remaining={rtc.buffer_remaining()})"
                )
                infer_ticket = rtc.mark_inference_started()
                async_thread = threading.Thread(
                    target=_rtc_async_infer,
                    args=(rtc, instruction, step, run_id, rtc_cfg, infer_ticket),
                    daemon=True,
                )
                async_thread.start()

            next_t += period
            remaining = next_t - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_t = time.perf_counter()

        if async_thread is not None and async_thread.is_alive():
            async_thread.join(timeout=1.0)

        if step >= max_steps and not episode.finished:
            _await_episode_label_after_motion(episode, step)

        result = {
            "success": failure_message is None,
            "message": failure_message or "VLA RTC loop completed",
            "details": {
                "run_id": run_id,
                "steps": step,
                "total_actions_executed": step,
                "max_steps": max_steps,
            },
        }
        return result
    finally:
        rtc.close_session()
        sidecar = _finalize_episode(episode, default_reason="timeout")
        inference_logger.end_run(run_id)
        if result is not None:
            _attach_hil_details(result.setdefault("details", {}), episode)
        logger.info(f"HIL episode finalized sidecar={sidecar}")


def _rtc_async_infer_legacy(rtc, instruction, current_step, run_id):
    """后台线程：获取观测 → VLA 推理 → 返回结果。"""
    try:
        episode = _get_episode()
        if episode is not None and (episode.intervened or episode.finished):
            logger.info("RTC async: skip infer after HIL intervention/finish")
            rtc.mark_inference_done([])
            return
        t0 = time.perf_counter()
        with _robot_lock:
            obs, meta = _capture_and_preprocess()
        if obs is None:
            logger.error("RTC async: observation is None, skip infer")
            rtc.mark_inference_done([])
            return
        obs_elapsed_ms = (time.perf_counter() - t0) * 1000
        obs["task"] = instruction
        if not validate_observation(obs):
            logger.error("RTC async: observation schema invalid, skip infer")
            rtc.mark_inference_done([])
            return

        inference_logger.log_obs(run_id=run_id, step=current_step, obs=obs, elapsed_ms=obs_elapsed_ms)

        t_infer = time.perf_counter()
        actions = vla.infer(obs, instruction, timestep=current_step)
        infer_elapsed_ms = (time.perf_counter() - t_infer) * 1000

        if _stop_event.is_set() or (
            episode is not None and (episode.intervened or episode.finished)
        ):
            logger.info("RTC async: stop/takeover while inferring; discarding stale chunk")
            rtc.mark_inference_done([])
            return

        if actions:
            action_vectors = [a["action"] for a in actions if "action" in a]
            log_inferred_actions(
                run_id, instruction, current_step, obs, actions, infer_elapsed_ms,
                selected_action_count=len(action_vectors),
            )
            rtc.mark_inference_done(action_vectors)
            logger.info(
                f"RTC async infer done: obs={obs_elapsed_ms:.0f}ms, infer={infer_elapsed_ms:.0f}ms, "
                f"actions={len(action_vectors)}"
            )
        else:
            logger.warning("RTC async inference returned no actions")
            rtc.mark_inference_done([])
    except Exception as e:
        logger.error(f"RTC async inference error: {e}")
        rtc.mark_inference_done([])


def _rtc_async_infer(rtc, instruction, current_step, run_id, rtc_cfg, infer_ticket):
    """Capture a prefix snapshot and request model-native RTC remotely."""
    session_id = infer_ticket["session_id"]
    try:
        episode = _get_episode()
        if episode is not None and (episode.intervened or episode.finished):
            rtc.mark_inference_done([], session_id=session_id)
            return

        t0 = time.perf_counter()
        with _robot_lock:
            obs, _ = _capture_and_preprocess()
        capture_meta = rtc.mark_observation_captured(session_id)
        if obs is None or capture_meta is None:
            rtc.mark_inference_done([], session_id=session_id)
            return
        obs_elapsed_ms = (time.perf_counter() - t0) * 1000
        obs["task"] = instruction
        if not validate_observation(obs):
            rtc.mark_inference_done([], session_id=session_id)
            return

        inference_logger.log_obs(
            run_id=run_id, step=current_step, obs=obs, elapsed_ms=obs_elapsed_ms
        )
        previous_actions = capture_meta.pop("previous_actions")
        inference_delay = int(capture_meta.get("inference_delay", 0))
        execution_horizon = int(rtc_cfg.get("execution_horizon", 10))
        if inference_delay > execution_horizon:
            raise RuntimeError(
                "measured RTC inference delay exceeds execution_horizon: "
                f"delay={inference_delay}, horizon={execution_horizon}"
            )
        if len(previous_actions) < inference_delay:
            raise RuntimeError(
                "RTC prefix is shorter than the measured inference delay: "
                f"prefix={len(previous_actions)}, delay={inference_delay}; "
                "increase pre_infer_threshold"
            )
        rtc_context = {
            **infer_ticket,
            **capture_meta,
            "enabled": True,
            "prev_chunk_left_over": [action.tolist() for action in previous_actions],
            "prev_chunk_semantics": "absolute_postprocessed",
            "inference_delay": inference_delay,
            "execution_horizon": execution_horizon,
        }

        t_infer = time.perf_counter()
        rtc_result = vla.infer_rtc(
            obs, instruction, timestep=current_step, rtc_context=rtc_context
        )
        infer_elapsed_ms = (time.perf_counter() - t_infer) * 1000

        if _stop_event.is_set() or not rtc.is_session_active(session_id) or (
            episode is not None and (episode.intervened or episode.finished)
        ):
            logger.info("RTC async: session stopped while inferring; discarding chunk")
            return

        if not rtc_result:
            rtc.mark_inference_done([], session_id=session_id)
            return
        actions, server_meta = rtc_result
        if not actions:
            rtc.mark_inference_done([], meta=server_meta, session_id=session_id)
            return

        action_vectors = [a["action"] for a in actions if "action" in a]
        measured_delay = max(
            0,
            rtc.trajectory_step - int(capture_meta["obs_capture_trajectory_step"]),
        )
        response_meta = {
            **capture_meta,
            **dict(server_meta or {}),
            "inference_delay_frames": int(
                (server_meta or {}).get("inference_delay_frames", measured_delay)
            ),
        }
        log_inferred_actions(
            run_id, instruction, current_step, obs, actions, infer_elapsed_ms,
            selected_action_count=len(action_vectors),
        )
        rtc.mark_inference_done(
            action_vectors, meta=response_meta, session_id=session_id
        )
        logger.info(
            "RTC native infer done: obs=%.0fms infer=%.0fms actions=%s "
            "prefix=%s horizon=%s delay=%s",
            obs_elapsed_ms, infer_elapsed_ms, len(action_vectors),
            response_meta.get("prefix_frames_used"),
            rtc_context["execution_horizon"],
            response_meta["inference_delay_frames"],
        )
    except Exception as exc:
        logger.error("RTC native async inference error: %s", exc)
        rtc.mark_inference_done(
            [], meta={"fatal_error": str(exc)}, session_id=session_id
        )


# ------------------------------------------------------------------
# HTTP Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "robot_available": bool(robot and robot.is_available() and robot.robot is not None),
        "policy_server_connected": vla.connected if vla else False,
        "policy_ready": vla.policy_ready if vla else False,
        "configured_checkpoint": getattr(vla, "configured_checkpoint", None) if vla else None,
        "required_checkpoint": CFG.get("pretrained_name_or_path"),
        "profile": CFG.get("profile"),
        "profile_cameras": list(REQUIRED_CAMERAS),
        "profile_action_dim": CFG.get("profile_action_dim"),
        "configured_num_episodes": HIL_SESSION_CFG.get("max_episodes"),
        "configured_max_steps": CFG.get("max_steps"),
        "training_fps": TRAINING_FPS,
        "image_preprocess": CFG.get("image_preprocess", {}),
        "instruction": CFG.get("instruction", ""),
    }


@app.post("/camera_snapshot")
def camera_snapshot():
    """Capture four raw camera frames for mapping/exposure/focus diagnosis."""
    if robot is None or robot.robot is None:
        return {"success": False, "message": "Robot camera service is not initialized"}

    with _robot_lock:
        obs = robot.get_observation(features=CFG.get("observation_features"))
    if obs is None:
        return {"success": False, "message": "Failed to capture all four cameras"}

    diag_cfg = CFG.get("camera_diagnostics") or {}
    snapshot_dir = diag_cfg.get("snapshot_dir", "logs/camera_snapshots")
    if not os.path.isabs(snapshot_dir):
        snapshot_dir = os.path.join(_SCRIPT_DIR, snapshot_dir)
    run_dir = os.path.join(snapshot_dir, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    os.makedirs(run_dir, exist_ok=False)

    cameras = {}
    for name in REQUIRED_CAMERAS:
        image = obs.get(name)
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            cameras[name] = {"error": "missing or invalid frame"}
            continue
        image = np.asarray(image, dtype=np.uint8)
        path = os.path.join(run_dir, f"{name}.jpg")
        Image.fromarray(image).save(path, quality=95)

        gray = image.astype(np.float32).mean(axis=2)
        # Mean absolute neighboring-pixel difference: simple dependency-free
        # sharpness indicator, useful for spotting continuous autofocus hunting.
        sharpness = 0.5 * (
            float(np.abs(np.diff(gray, axis=0)).mean())
            + float(np.abs(np.diff(gray, axis=1)).mean())
        )
        cameras[name] = {
            "path": path,
            "shape": list(image.shape),
            "mean_brightness": float(gray.mean()),
            "contrast_std": float(gray.std()),
            "dark_fraction": float((gray <= 2).mean()),
            "saturated_fraction": float((gray >= 253).mean()),
            "sharpness": sharpness,
        }

    return {
        "success": all("error" not in value for value in cameras.values()),
        "snapshot_dir": run_dir,
        "camera_mapping": (ROBOT_CFG.get("observation_camera_map") or {}),
        "head_state": {
            "head_joint1": obs.get("head_joint1"),
            "head_joint2": obs.get("head_joint2"),
        },
        "cameras": cameras,
    }


@app.post("/execute", response_model=VLAResponse)
def execute(req: Optional[VLARequest] = None):
    """Run one episode or a YAML-configured continuous HIL collection session."""
    req = req or VLARequest()
    instruction = req.instruction or CFG.get("instruction", "")
    result = run_vla(
        instruction=instruction,
        max_steps=req.max_steps or int(CFG.get("max_steps", 300)),
        continuous=req.continuous,
        max_episodes=req.max_episodes,
    )
    return VLAResponse(**result)


@app.post("/takeover")
def takeover():
    """Latch AUTO -> INTERVENTION for the active episode (not hold/release)."""
    return _request_takeover("POST /takeover")


@app.post("/episode/success")
def episode_success():
    return _request_episode_finish(1, "success_key", "POST /episode/success")


@app.post("/episode/failure")
def episode_failure():
    return _request_episode_finish(0, "failure_key", "POST /episode/failure")


@app.get("/hil/status")
def hil_status():
    episode = _get_episode()
    if episode is None:
        return {
            "mode": "idle",
            "intervened": False,
            "run_id": None,
            "episode_success": None,
            "capture": capture.status() if capture else {"enabled": False},
            "session": _get_session_state(),
        }
    return dict(
        episode.status(),
        capture=capture.status() if capture else {"enabled": False},
        session=_get_session_state(),
    )


@app.post("/reset")
def reset():
    """重置 VLA 策略状态。"""
    if vla is None:
        return {"success": False, "message": "VLA not initialized"}
    ok = vla.reset()
    return {"success": ok}


@app.post("/reconnect")
def reconnect():
    """重连 PolicyServer。"""
    if vla is None:
        return {"success": False, "message": "VLA not initialized"}
    if vla.connected:
        vla.disconnect()
    ok = vla.connect()
    if ok:
        ok = vla.setup_policy()
    return {
        "success": ok,
        "connected": vla.connected,
        "policy_ready": vla.policy_ready,
    }
    
@app.post("/stop")
def stop():
    _session_stop_event.set()  # 结束连续采集，禁止自动开始下一条
    _stop_event.set()          # 中断正在运行的 run_vla 循环
    _request_episode_finish(0, "stop", "POST /stop")
    if vla is not None and not _execute_lock.locked():
        vla.reset()            # 重置远端 PolicyServer 模型状态
    return {"success": True, "message": "Continuous session stop signal sent"}


@app.post("/shutdown")
def shutdown():
    """Return first, then release robot resources and terminate the process."""
    logger.warning("Shutdown requested via API")
    _session_stop_event.set()
    _stop_event.set()
    _persist_episode_on_process_exit("stop")

    def _graceful_exit():
        # Give the HTTP worker enough time to flush the success response.
        time.sleep(0.3)
        if robot is not None:
            # Never destroy the SDK while an action chunk is still being sent.
            with _robot_lock:
                _shutdown_robot_with_timeout(robot, timeout=5.0)
        if vla is not None:
            vla.disconnect()
        logger.info("API shutdown complete")
        os._exit(0)

    threading.Thread(target=_graceful_exit, daemon=True).start()
    return {"success": True, "message": "Shutdown scheduled"}

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def start_stdin_monitor() -> None:
    """Listen on this process's terminal for HIL keys, Ctrl+C, and right-arrow.

    Takeover / success / failure are single ASCII keys from hil.*_key.
    Right-arrow is usually ESC [ C; some terminals send ESC O C.
    The Galbot SDK ignores SIGINT in C, so Ctrl+C is detected as stdin 0x03.
    uvicorn may reset terminal flags, so raw mode is re-applied each loop.
    """
    import select
    import termios

    def _stdin_monitor():
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            logger.warning("stdin is not a TTY; HIL keys and right-arrow recovery are disabled")
            return

        takeover_key = _hil_key_byte("takeover_key", "i")
        success_key = _hil_key_byte("success_key", "s")
        failure_key = _hil_key_byte("failure_key", "f")
        raw = termios.tcgetattr(fd)
        raw[0] &= ~(termios.IXON | getattr(termios, "IXOFF", 0))
        # Keep output processing unchanged; disabling OPOST corrupts concurrent
        # log rendering in integrated terminals.
        raw[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
        raw[6][termios.VMIN] = 1
        raw[6][termios.VTIME] = 0
        # Fallback for SDK components that temporarily restore ICANON: make the
        # takeover key an alternate line delimiter, so it is delivered without
        # Enter even during that canonical-mode race.
        raw[6][termios.VEOL] = takeover_key
        escape_sequence = b""
        escape_deadline = 0.0
        logger.warning(
            "HIL keys on THIS terminal: "
            f"takeover={takeover_key.decode('latin1')!r} (one-shot latch), "
            f"success={success_key.decode('latin1')!r}, "
            f"failure={failure_key.decode('latin1')!r}; "
            "right-arrow restores initial pose (disabled during INTERVENTION)"
        )

        while True:
            try:
                # SDK/uvicorn startup may restore canonical input behind this
                # thread. Reapply the complete raw profile unconditionally so
                # every HIL key is delivered without waiting for Enter.
                termios.tcsetattr(fd, termios.TCSANOW, raw)

                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    if escape_sequence and time.monotonic() >= escape_deadline:
                        escape_sequence = b""
                    continue

                ch = os.read(fd, 1)
                if ch == b"\x03":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    logger.warning("Ctrl+C detected via stdin, shutting down")
                    _session_stop_event.set()
                    _stop_event.set()
                    _persist_episode_on_process_exit("stop")
                    if vla is not None:
                        vla.disconnect()
                    os._exit(0)

                if ch == b"\x1b":
                    escape_sequence = ch
                    escape_deadline = time.monotonic() + 0.2
                    continue
                if escape_sequence:
                    escape_sequence += ch
                    if escape_sequence in (b"\x1b[C", b"\x1bOC"):
                        escape_sequence = b""
                        _request_interactive_recovery("terminal right-arrow key")
                    elif escape_sequence not in (b"\x1b[", b"\x1bO"):
                        escape_sequence = b""
                    continue

                folded = ch.lower()
                if folded == takeover_key.lower():
                    _request_takeover("stdin")
                elif folded == success_key.lower():
                    _request_episode_finish(1, "success_key", "stdin")
                elif folded == failure_key.lower():
                    _request_episode_finish(0, "failure_key", "stdin")
            except Exception:
                time.sleep(0.2)

    threading.Thread(target=_stdin_monitor, name="stdin-monitor", daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    start_stdin_monitor()
    host = SERVER_CFG.get("host", "0.0.0.0")
    port = SERVER_CFG.get("port", 8005)
    logger.info(f"Starting VLA server on {host}:{port}")
    logger.info(f"Instruction: '{CFG.get('instruction', '')}'")
    _log_runtime_identity()
    uvicorn.run(app, host=host, port=port)
    os._exit(0)
