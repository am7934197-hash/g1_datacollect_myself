"""G1 HIL runtime defaults and concise task-config expansion.

The public YAML contains only task/session knobs.  Robot safety limits,
checkpoint schemas, SDK wiring and stable implementation choices live here so
they are reviewed and tested as code.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict


STATE_NAMES = [
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
    "right_arm_joint7", "right_arm_gripper",
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
    "left_arm_joint7", "left_arm_gripper",
    "leg_joint1", "leg_joint2", "leg_joint3", "leg_joint4", "leg_joint5",
    "head_joint1", "head_joint2",
]

TRAINING_FPS = 30.0
TRAINING_LETTERBOX_PAD_TBLR = (28, 28, 0, 0)
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)

ARM_GRIPPER_ACTION_NAMES = list(STATE_NAMES[:16])
FULL_ACTION_NAMES = list(STATE_NAMES)

PROFILE_SPECS = {
    "g1_pick_3cam_16d": {
        "cameras": ("head_right", "left_arm", "right_arm"),
        "action_names": ARM_GRIPPER_ACTION_NAMES,
    },
    "g1_place_4cam_23d": {
        "cameras": ("head_left", "head_right", "left_arm", "right_arm"),
        "action_names": FULL_ACTION_NAMES,
    },
}

JOINT_GROUPS = [
    {"name": "right_arm", "indices": list(range(0, 7)), "enabled": True},
    {"name": "right_gripper", "indices": [7], "is_gripper": True, "enabled": True},
    {"name": "left_arm", "indices": list(range(8, 15)), "enabled": True},
    {"name": "left_gripper", "indices": [15], "is_gripper": True, "enabled": True},
    {"name": "leg", "indices": list(range(16, 21)), "enabled": False},
    {"name": "head", "indices": [21, 22], "enabled": False},
]

JOINT_LIMITS = [
    [-1.83, 0.11], [0.29, 1.6081], [-0.84, 1.17], [1.178, 2.47],
    [-0.76, 1.04], [-0.7354, 0.8227], [-1.5382, 0.61], [0.0, 100.0],
    [0.0, 1.45], [-1.6081, -0.27], [-0.77, 0.74], [-2.38, -1.248],
    [-1.16, 0.81], [-0.64, 0.7354], [-1.26, 1.5382], [0.0, 100.0],
    [0.0, 0.9374], [0.0, 2.5847], [0.0, 2.3262],
    [-1.5906, 1.5906], [-0.1645, 0.1645],
    [-1.5208, 1.5208], [-0.2143, 0.4936],
]

RESET_POSE_SIZES = {
    "right_arm": 7,
    "right_gripper": 1,
    "left_arm": 7,
    "left_gripper": 1,
    "leg": 5,
    "head": 2,
}

BASE_CONFIG: Dict[str, Any] = {
    "auto_setup_policy": True,
    "force_setup_policy": True,
    "policy_type": "pi05",
    "device": "cuda",
    "inference_optimization": {
        "torch_dtype": "bfloat16",
        "torch_compile": True,
        "use_cuda_graph": False,
    },
    "actions_per_chunk": 50,
    "fps": 30,
    "obs_queue_timeout": 1.0,
    "action_response_timeout": 30.0,
    "observation_transport": {"image_encoding": "png", "png_compress_level": 3},
    "observation_mapping": {},
    "robot": {
        "sensor_set": [
            "HEAD_LEFT_CAMERA", "HEAD_RIGHT_CAMERA",
            "LEFT_ARM_CAMERA", "RIGHT_ARM_CAMERA",
        ],
        "sensor_warmup_seconds": 15.0,
        "camera_rgb_retries": 8,
        "camera_rgb_retry_delay": 0.3,
        "observation_camera_map": {
            "head_left": "head_left", "head_right": "head_right",
            "left_arm": "left_arm", "right_arm": "right_arm",
        },
        "head_camera_roi": {
            "enabled": False,
            "expected_size": [960, 1280],
            "head_left": [118, 784, 243, 1123],
        },
        "reference_frame": "base_link",
        "end_effector_frame": "right_arm",
        "state_feature_names": STATE_NAMES,
        "joint_groups": JOINT_GROUPS,
        "init_pose_verify": {
            "joint_tolerance_rad": 0.03,
            "gripper_tolerance_model": 3.0,
        },
        "recovery": {"gripper_open_width": 0.12, "settle_seconds": 8.0, "retries": 2},
        "control": {
            "action_semantics": "absolute",
            "fps": 30,
            "dt": 0.033,
            "set_joint_time_from_start": 0.0,
            "gripper_action_scale": 0.0012,
            "max_joint_delta_per_step": 0.08,
            "gripper_max_delta_per_step": 20.0,
            "joint_limits": JOINT_LIMITS,
            "action_filter": {
                "enabled": False,
                "type": "lowpass",
                "lowpass": {"alpha": 0.3},
                "moving_average": {"window_size": 5},
                "one_euro": {
                    "freq": 30.0,
                    "min_cutoff": 1.0,
                    "beta": 0.01,
                    "d_cutoff": 1.0,
                },
            },
            "interpolation": {
                "enabled": True, "method": "linear",
                "input_hz": 30, "output_hz": 250,
            },
        },
    },
    "rtc": {
        "enabled": False,
        "pre_infer_threshold": 32,
        "execution_horizon": 10,
        "inference_delay": 0,
        "max_steps_per_observation": 0,
        "warmup_inferences": 1,
    },
    "hil": {
        "shadow_policy_during_intervention": True,
        "await_label_after_max_steps": True,
        "default_success_on_timeout": 0,
        "capture": {"enabled": True, "python": "/usr/bin/python3"},
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8005,
        "vla_inference_log_file": "vla_inference_results.jsonl",
        "save_observation_images": False,
    },
    "inference_log": {
        "enabled": True,
        "max_log_size_mb": 100,
        "log_data": True,
        "save_images": True,
        "max_image_runs": 5,
    },
    "image_preprocess": {
        "enabled": True,
        "target_size": [224, 224],
        "keep_aspect_ratio": True,
        "crop_to_training_aspect": False,
        "training_hw": [480, 640],
        "enforce_training_geometry": False,
        "black_bar_tolerance_px": 4,
        "emulate_data_collection": {
            "enabled": False,
            "scheme": "via_640",
            "output_wh": [640, 480],
            "cameras": {
                "head_left": {"undistort_type": 1, "zoom_factor": 1.0},
                "head_right": {"undistort_type": 1, "zoom_factor": 1.0},
                "left_arm": {
                    "zoom_factor": 1.0,
                    "crop_to_training_aspect": False,
                },
                "right_arm": {
                    "zoom_factor": 1.0,
                    "crop_to_training_aspect": False,
                },
            },
        },
    },
}

_ALLOWED_KEYS = {
    "": {
        "profile", "policy", "dataset", "robot", "controls", "server",
        "actions_per_chunk", "fps", "obs_queue_timeout",
        "action_response_timeout", "inference_optimization", "rtc",
        "observation_transport", "image_preprocess",
    },
    "policy": {
        "server_address", "checkpoint", "instruction", "execute_actions_per_chunk",
        "max_steps", "type", "device", "auto_setup", "force_setup",
    },
    "dataset": {"root", "reset_time_s", "num_episodes"},
    "robot": {"reset_pose", "reset_gripper_cycle", "control"},
    "controls": {
        "takeover_key", "success_key", "failure_key", "continuous",
        "shadow_inference", "shadow_policy_interval_s", "takeover_grace_s",
    },
    "server": {"port"},
    "inference_optimization": {"torch_dtype", "torch_compile", "use_cuda_graph"},
    "rtc": {
        "enabled", "pre_infer_threshold",
        "execution_horizon", "inference_delay", "max_steps_per_observation",
        "warmup_inferences",
    },
    "robot.control": {
        "action_semantics", "fps", "dt", "set_joint_time_from_start",
        "gripper_action_scale", "max_joint_delta_per_step",
        "gripper_max_delta_per_step", "action_filter", "interpolation",
    },
    "robot.control.action_filter": {
        "enabled", "type", "lowpass", "moving_average", "one_euro",
    },
    "robot.control.action_filter.lowpass": {"alpha"},
    "robot.control.action_filter.moving_average": {"window_size"},
    "robot.control.action_filter.one_euro": {
        "freq", "min_cutoff", "beta", "d_cutoff",
    },
    "robot.control.interpolation": {"enabled", "method", "input_hz", "output_hz"},
    "robot.reset_gripper_cycle": {
        "enabled", "open_width", "open_settle_s", "retries",
    },
    "observation_transport": {"image_encoding", "png_compress_level"},
    "image_preprocess": {
        "enabled", "target_size", "keep_aspect_ratio", "crop_to_training_aspect",
        "training_hw", "enforce_training_geometry", "black_bar_tolerance_px",
    },
}


def _image_feature() -> dict:
    return {
        "dtype": "image",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channels"],
    }


def _observation_features(cameras) -> dict:
    features = {
        f"observation.images.{name}": _image_feature() for name in cameras
    }
    features["observation.state"] = {
        "dtype": "float32", "shape": [23], "names": list(STATE_NAMES)
    }
    return features


def _check_keys(raw: dict) -> None:
    unknown = set(raw) - _ALLOWED_KEYS[""]
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    for section in (
        "policy", "dataset", "robot", "controls", "server",
        "inference_optimization", "rtc", "observation_transport", "image_preprocess",
    ):
        value = raw.get(section, {})
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be a mapping")
        unknown = set(value) - _ALLOWED_KEYS[section]
        if unknown:
            raise ValueError(f"unknown {section} config keys: {sorted(unknown)}")

    robot = raw.get("robot") or {}
    control = robot.get("control") or {}
    if not isinstance(control, dict):
        raise ValueError("robot.control must be a mapping")
    paths = {
        "robot.control": control,
        "robot.reset_gripper_cycle": robot.get("reset_gripper_cycle") or {},
        "robot.control.action_filter": control.get("action_filter") or {},
        "robot.control.action_filter.lowpass":
            ((control.get("action_filter") or {}).get("lowpass") or {}),
        "robot.control.action_filter.moving_average":
            ((control.get("action_filter") or {}).get("moving_average") or {}),
        "robot.control.action_filter.one_euro":
            ((control.get("action_filter") or {}).get("one_euro") or {}),
        "robot.control.interpolation": control.get("interpolation") or {},
    }
    for path, value in paths.items():
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        unknown = set(value) - _ALLOWED_KEYS[path]
        if unknown:
            raise ValueError(f"unknown {path} config keys: {sorted(unknown)}")


def _deep_update(target: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _validate_control(control: dict) -> None:
    if control.get("action_semantics") != "absolute":
        raise ValueError("robot.control.action_semantics must be absolute")
    positive = (
        "fps", "dt", "gripper_action_scale", "max_joint_delta_per_step",
        "gripper_max_delta_per_step",
    )
    for key in positive:
        if float(control.get(key, 0)) <= 0:
            raise ValueError(f"robot.control.{key} must be > 0")
    if float(control.get("set_joint_time_from_start", 0)) < 0:
        raise ValueError("robot.control.set_joint_time_from_start must be >= 0")

    action_filter = control.get("action_filter") or {}
    filter_type = str(action_filter.get("type", "none"))
    if filter_type not in {"none", "lowpass", "moving_average", "one_euro"}:
        raise ValueError(f"unknown robot.control.action_filter.type: {filter_type}")
    alpha = float((action_filter.get("lowpass") or {}).get("alpha", 0.3))
    if not 0 < alpha <= 1:
        raise ValueError("robot.control.action_filter.lowpass.alpha must be in (0, 1]")
    window = int((action_filter.get("moving_average") or {}).get("window_size", 5))
    if window < 1:
        raise ValueError("robot.control.action_filter.moving_average.window_size must be >= 1")
    one_euro = action_filter.get("one_euro") or {}
    for key in ("freq", "min_cutoff", "d_cutoff"):
        if float(one_euro.get(key, 0)) <= 0:
            raise ValueError(f"robot.control.action_filter.one_euro.{key} must be > 0")
    if float(one_euro.get("beta", 0)) < 0:
        raise ValueError("robot.control.action_filter.one_euro.beta must be >= 0")

    interpolation = control.get("interpolation") or {}
    method = str(interpolation.get("method", "linear"))
    if method not in {"linear", "cubic", "quintic"}:
        raise ValueError(f"unknown robot.control.interpolation.method: {method}")
    for key in ("input_hz", "output_hz"):
        if float(interpolation.get(key, 0)) <= 0:
            raise ValueError(f"robot.control.interpolation.{key} must be > 0")


def _required_text(section: dict, key: str) -> str:
    value = str(section.get(key) or "").strip()
    if not value:
        raise ValueError(f"policy.{key} is required")
    return value


def _validate_reset_pose(pose: dict) -> dict:
    if not isinstance(pose, dict):
        raise ValueError("robot.reset_pose must be a mapping")
    missing = sorted(set(RESET_POSE_SIZES) - set(pose))
    extra = sorted(set(pose) - set(RESET_POSE_SIZES))
    bad = {}
    for name, size in RESET_POSE_SIZES.items():
        if name not in pose:
            continue
        values = pose[name]
        actual = len(values) if isinstance(values, (list, tuple)) else None
        if actual != size:
            bad[name] = {"expected": size, "actual": actual}
    if missing or extra or bad:
        raise ValueError(
            f"invalid robot.reset_pose: missing={missing}, extra={extra}, bad_sizes={bad}"
        )
    groups = {group["name"]: group for group in JOINT_GROUPS}
    normalized = {}
    for name, values in pose.items():
        normalized[name] = []
        for value, index in zip(values, groups[name]["indices"]):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"robot.reset_pose.{name} contains non-finite value {number}")
            normalized[name].append(number)
    return normalized


def expand_config(raw: dict) -> dict:
    """Expand the concise public YAML into the legacy runtime CFG shape."""
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    _check_keys(raw)
    profile_name = str(raw.get("profile") or "").strip()
    if profile_name not in PROFILE_SPECS:
        raise ValueError(
            f"unknown profile {profile_name!r}; expected one of {sorted(PROFILE_SPECS)}"
        )
    policy = raw.get("policy") or {}
    dataset = raw.get("dataset") or {}
    robot = raw.get("robot") or {}
    controls = raw.get("controls") or {}
    service = raw.get("server") or {}
    checkpoint = _required_text(policy, "checkpoint")
    server_address = _required_text(policy, "server_address")
    instruction = _required_text(policy, "instruction")
    pose = _validate_reset_pose(robot.get("reset_pose"))
    reset_gripper_cycle = robot.get("reset_gripper_cycle") or {}
    open_width = float(reset_gripper_cycle.get("open_width", 0.12))
    open_settle_s = float(reset_gripper_cycle.get("open_settle_s", 2.0))
    open_retries = int(reset_gripper_cycle.get("retries", 2))
    if not 0.0 <= open_width <= 0.12:
        raise ValueError("robot.reset_gripper_cycle.open_width must be in 0..0.12 m")
    if open_settle_s < 0:
        raise ValueError("robot.reset_gripper_cycle.open_settle_s must be >= 0")
    if open_retries < 1:
        raise ValueError("robot.reset_gripper_cycle.retries must be >= 1")
    root = str(dataset.get("root") or "logs").strip()
    if not root:
        raise ValueError("dataset.root must not be empty")
    reset_time = float(dataset.get("reset_time_s", 10.0))
    if reset_time < 0:
        raise ValueError("dataset.reset_time_s must be >= 0")
    num_episodes = int(dataset.get("num_episodes", 1))
    if num_episodes < 1:
        raise ValueError("dataset.num_episodes must be >= 1")

    spec = PROFILE_SPECS[profile_name]
    actions_per_chunk = int(raw.get("actions_per_chunk", BASE_CONFIG["actions_per_chunk"]))
    if actions_per_chunk < 1:
        raise ValueError("actions_per_chunk must be >= 1")
    execute_count = int(policy.get("execute_actions_per_chunk", 25))
    if not 0 < execute_count <= actions_per_chunk:
        raise ValueError(
            "policy.execute_actions_per_chunk must be in "
            f"1..actions_per_chunk ({actions_per_chunk})"
        )
    max_steps = int(policy.get("max_steps", 300))
    if max_steps < 1:
        raise ValueError("policy.max_steps must be >= 1")
    fps = float(raw.get("fps", BASE_CONFIG["fps"]))
    if fps <= 0:
        raise ValueError("fps must be > 0")
    obs_queue_timeout = float(
        raw.get("obs_queue_timeout", BASE_CONFIG["obs_queue_timeout"])
    )
    action_response_timeout = float(
        raw.get("action_response_timeout", BASE_CONFIG["action_response_timeout"])
    )
    if obs_queue_timeout <= 0 or action_response_timeout <= 0:
        raise ValueError("inference timeouts must be > 0")
    shadow_interval = float(controls.get("shadow_policy_interval_s", 1.0))
    if shadow_interval <= 0:
        raise ValueError("controls.shadow_policy_interval_s must be > 0")
    takeover_grace_s = float(controls.get("takeover_grace_s", 0.0))
    if takeover_grace_s < 0:
        raise ValueError("controls.takeover_grace_s must be >= 0")
    keys = {
        "takeover_key": str(controls.get("takeover_key", "i")),
        "success_key": str(controls.get("success_key", "s")),
        "failure_key": str(controls.get("failure_key", "f")),
    }
    if any(len(value.encode("utf-8")) != 1 for value in keys.values()):
        raise ValueError("controls takeover/success/failure keys must each be one ASCII byte")
    if len({value.lower() for value in keys.values()}) != 3:
        raise ValueError("controls takeover/success/failure keys must be distinct")
    port = int(service.get("port", 8005))
    if not 1 <= port <= 65535:
        raise ValueError("server.port must be in 1..65535")

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["inference_optimization"].update(raw.get("inference_optimization") or {})
    cfg["rtc"].update(raw.get("rtc") or {})
    cfg["observation_transport"].update(raw.get("observation_transport") or {})
    cfg["image_preprocess"].update(raw.get("image_preprocess") or {})
    cfg.update({
        "profile": profile_name,
        "profile_action_names": list(spec["action_names"]),
        "profile_action_dim": len(spec["action_names"]),
        "policy_type": str(policy.get("type", BASE_CONFIG["policy_type"])),
        "device": str(policy.get("device", BASE_CONFIG["device"])),
        "auto_setup_policy": bool(policy.get("auto_setup", BASE_CONFIG["auto_setup_policy"])),
        "force_setup_policy": bool(policy.get("force_setup", BASE_CONFIG["force_setup_policy"])),
        "policy_server_address": server_address,
        "pretrained_name_or_path": checkpoint,
        "instruction": instruction,
        "execute_actions_per_chunk": execute_count,
        "max_steps": max_steps,
        "actions_per_chunk": actions_per_chunk,
        "fps": fps,
        "obs_queue_timeout": obs_queue_timeout,
        "action_response_timeout": action_response_timeout,
        "observation_features": _observation_features(spec["cameras"]),
    })
    cfg["robot"].update({
        "vla_action_dim": len(spec["action_names"]),
        "init_joint_positions": pose,
    })
    _deep_update(cfg["robot"]["control"], robot.get("control") or {})
    _validate_control(cfg["robot"]["control"])
    cfg["hil"].update({
        **keys,
        "takeover_grace_s": takeover_grace_s,
        "sidecar_dir": os.path.join(root, "hil_episodes"),
        "shadow_policy_during_intervention": bool(
            controls.get("shadow_inference", True)
        ),
        "shadow_policy_interval_s": shadow_interval,
        "collector_policy_id": f"{profile_name}:{checkpoint}",
        "continuous": {
            "enabled": bool(controls.get("continuous", True)),
            "max_episodes": num_episodes,
            "reset_time_s": reset_time,
            "pose_settle_s": 8.0,
            "pose_retries": 2,
            "wait_takeover_release_s": 10.0,
            "reset_pose": copy.deepcopy(pose),
            "reset_gripper_cycle": {
                "enabled": bool(reset_gripper_cycle.get("enabled", False)),
                "open_width": open_width,
                "open_settle_s": open_settle_s,
                "retries": open_retries,
            },
        },
    })
    cfg["hil"]["capture"]["directory"] = os.path.join(root, "hil_raw")
    cfg["server"]["port"] = port
    cfg["inference_log"].update({
        "log_dir": root,
        "image_dir": os.path.join(root, "images"),
    })
    cfg["camera_diagnostics"] = {
        "snapshot_dir": os.path.join(root, "camera_snapshots")
    }
    return cfg


def validate_checkpoint_schema(config: dict, checkpoint: dict) -> None:
    """Validate an expanded profile against a readable checkpoint config."""
    configured = config.get("observation_features") or {}
    checkpoint_inputs = checkpoint.get("input_features") or {}
    if set(configured) != set(checkpoint_inputs):
        raise ValueError(
            "checkpoint/config input_features mismatch: "
            f"checkpoint={sorted(checkpoint_inputs)}, configured={sorted(configured)}"
        )
    for key, feature in configured.items():
        checkpoint_feature = checkpoint_inputs.get(key) or {}
        if feature.get("dtype") == "image":
            chw = list(checkpoint_feature.get("shape") or [])
            hwc = list(feature.get("shape") or [])
            expected_hwc = [chw[1], chw[2], chw[0]] if len(chw) == 3 else []
            if hwc != expected_hwc:
                raise ValueError(
                    f"checkpoint/config image shape mismatch for {key}: "
                    f"checkpoint CHW={chw}, configured HWC={hwc}"
                )
        elif key == "observation.state":
            if list(checkpoint_feature.get("shape") or []) != [23]:
                raise ValueError("checkpoint observation.state must have shape [23]")

    action_dim = int(config.get("profile_action_dim", 0) or 0)
    output_shape = list(
        (((checkpoint.get("output_features") or {}).get("action") or {}).get("shape") or [])
    )
    if output_shape != [action_dim]:
        raise ValueError(
            f"checkpoint action shape={output_shape}, profile action_dim={action_dim}"
        )
    action_names = list(checkpoint.get("action_feature_names") or [])
    expected_names = list(config.get("profile_action_names") or [])
    if action_names and action_names != expected_names:
        raise ValueError(
            f"checkpoint action_feature_names mismatch: expected={expected_names}, got={action_names}"
        )
    if not checkpoint.get("use_relative_actions", False):
        raise ValueError("checkpoint must use relative arm actions")
    if list(checkpoint.get("relative_exclude_joints") or []) != ["gripper"]:
        raise ValueError("checkpoint must exclude only gripper from relative actions")
    normalization = checkpoint.get("normalization_mapping") or {}
    if normalization.get("STATE") != "QUANTILES" or normalization.get("ACTION") != "QUANTILES":
        raise ValueError(
            f"checkpoint STATE/ACTION normalization must both be QUANTILES; got {normalization}"
        )


def resolve_checkpoint_path(raw: str) -> str:
    path = str(raw or "").strip()
    if not path:
        raise ValueError("Checkpoint path is empty")
    return os.path.abspath(os.path.expanduser(path))


def inspect_checkpoint_files(checkpoint_dir: str) -> Dict[str, Any]:
    missing = []
    present = []
    for name in REQUIRED_CHECKPOINT_FILES:
        full = os.path.join(checkpoint_dir, name)
        if os.path.isfile(full):
            present.append(os.path.abspath(full))
        else:
            missing.append(full)
    local_readable = not missing
    return {
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "required_files": list(REQUIRED_CHECKPOINT_FILES),
        "present": present,
        "missing": missing,
        "local_readable": local_readable,
        "load_mode": "local" if local_readable else "remote_pinned",
    }


def format_checkpoint_report(info: Dict[str, Any]) -> str:
    if not info["local_readable"]:
        return (
            f"checkpoint={info['checkpoint_dir']} "
            "load_mode=remote_pinned (loaded by PolicyServer)"
        )

    lines = [
        f"resolved_checkpoint={info['checkpoint_dir']}",
        f"load_mode={info['load_mode']}",
        "required_files:",
    ]
    for name in info["required_files"]:
        full = os.path.join(info["checkpoint_dir"], name)
        status = "PRESENT" if full in info["present"] else "MISSING_LOCALLY"
        lines.append(f"  {status} {full}")
    return "\n".join(lines)
