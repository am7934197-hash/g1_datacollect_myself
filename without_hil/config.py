"""Configuration and stable schemas for G1 SFT collection."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Union

import yaml


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

PROFILE_SPECS = {
    "g1_pick_3cam_16d": {
        "cameras": ["head_right", "left_arm", "right_arm"],
        "action_names": STATE_NAMES[:16],
    },
    "g1_place_4cam_23d": {
        "cameras": ["head_left", "head_right", "left_arm", "right_arm"],
        "action_names": STATE_NAMES,
    },
}

GROUPS = {
    "right_arm": list(range(0, 7)),
    "right_gripper": [7],
    "left_arm": list(range(8, 15)),
    "left_gripper": [15],
    "leg": list(range(16, 21)),
    "head": [21, 22],
}
GRIPPER_GROUPS = {"right_gripper", "left_gripper"}
# reset_pose 中跳过范围校验的关节组（仅左右臂）。
POSE_UNRESTRICTED_GROUPS = {"right_arm", "left_arm"}

JOINT_LIMITS = [
    [-1.83, 0.11], [0.29, 1.6081], [-0.84, 1.17], [1.178, 2.47],
    [-0.76, 1.04], [-0.7354, 0.8227], [-1.5382, 0.61], [0.0, 100.0],
    [0.0, 1.45], [-1.6081, -0.27], [-0.77, 0.74], [-2.38, -1.248],
    [-1.16, 0.81], [-0.64, 0.7354], [-1.26, 1.5382], [0.0, 100.0],
    [0.0, 0.9374], [0.0, 2.5847], [0.0, 2.3262],
    [-1.5906, 1.5906], [-0.1645, 0.1645],
    [-1.5208, 1.5208], [-0.2143, 0.4936],
]


def _image_feature() -> dict:
    return {"dtype": "image", "shape": [480, 640, 3],
            "names": ["height", "width", "channels"]}


def _resolve(base: Path, value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (base / path).resolve())


def _validate_pose(pose: dict) -> None:
    if set(pose) != set(GROUPS):
        raise ValueError(f"robot.reset_pose must contain exactly {sorted(GROUPS)}")
    for name, indices in GROUPS.items():
        values = pose[name]
        if not isinstance(values, list) or len(values) != len(indices):
            raise ValueError(f"robot.reset_pose.{name} must have {len(indices)} values")
        for value, index in zip(values, indices):
            if not isinstance(value, (int, float)):
                raise ValueError(f"robot.reset_pose.{name} contains a non-number")
            # 左右臂不做范围校验：手动示教的起始位姿可能超出数据派生的软限位。
            # 仍对 leg/head/夹爪保留限位，避免误填导致躯干/头部危险动作。
            if name in POSE_UNRESTRICTED_GROUPS:
                continue
            lower, upper = JOINT_LIMITS[index]
            # YAML poses are often copied from six-decimal telemetry. Permit
            # only the corresponding tiny rounding error, not a real overrun.
            epsilon = 1e-5
            if not lower - epsilon <= float(value) <= upper + epsilon:
                raise ValueError(
                    f"robot.reset_pose.{name} value {value} outside [{lower}, {upper}]"
                )


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    required = {"profile", "task", "dataset", "capture", "robot"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    profile = raw.get("profile")
    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown profile {profile!r}; choose {sorted(PROFILE_SPECS)}")
    instruction = str((raw.get("task") or {}).get("instruction") or "").strip()
    if not instruction:
        raise ValueError("task.instruction must not be empty")
    dataset = raw.get("dataset") or {}
    num_episodes = int(dataset.get("num_episodes", 10))
    fps = float(dataset.get("output_fps", 30))
    skew = float(dataset.get("max_skew_s", 0.10))
    environment_reset_s = float(dataset.get("environment_reset_s", 10.0))
    if num_episodes <= 0:
        raise ValueError("dataset.num_episodes must be positive")
    if fps <= 0 or skew <= 0:
        raise ValueError("dataset.output_fps and max_skew_s must be positive")
    if environment_reset_s < 0:
        raise ValueError("dataset.environment_reset_s must be non-negative")
    if not fps.is_integer():
        raise ValueError("dataset.output_fps must be an integer")
    capture = raw.get("capture") or {}
    if int(capture.get("max_queue", 2048)) <= 0:
        raise ValueError("capture.max_queue must be positive")
    if float(capture.get("topic_timeout_s", 2.0)) <= 0:
        raise ValueError("capture.topic_timeout_s must be positive")
    robot = raw.get("robot") or {}
    _validate_pose(robot.get("reset_pose") or {})
    for key in ("reset_timeout_s", "reset_retries", "reset_max_speed_rad_s",
                "joint_tolerance_rad",
                "gripper_tolerance_model", "gripper_action_scale"):
        if float(robot.get(key, 0)) <= 0:
            raise ValueError(f"robot.{key} must be positive")
    if float(robot.get("sdk_init_settle_s", 2.0)) < 0:
        raise ValueError("robot.sdk_init_settle_s must be non-negative")
    max_width = float(robot.get("gripper_max_width_m", 0.12))
    feedback_max = float(
        robot.get("gripper_feedback_max_position_m", max_width / 2.0)
    )
    open_width = float(robot.get("gripper_open_width_m", max_width))
    if max_width <= 0 or not 0 < open_width <= max_width:
        raise ValueError(
            "robot.gripper_open_width_m must be positive and no greater than "
            "robot.gripper_max_width_m"
        )
    if feedback_max <= 0:
        raise ValueError("robot.gripper_feedback_max_position_m must be positive")
    if float(robot.get("gripper_open_settle_s", 2.0)) < 0:
        raise ValueError("robot.gripper_open_settle_s must be non-negative")
    if not isinstance(robot.get("verify_gripper_position", False), bool):
        raise ValueError("robot.verify_gripper_position must be true or false")
    if not isinstance(robot.get("verify_gripper_open_position", True), bool):
        raise ValueError("robot.verify_gripper_open_position must be true or false")

    cfg = copy.deepcopy(raw)
    base = config_path.parent
    cfg["config_path"] = str(config_path)
    cfg["profile_spec"] = copy.deepcopy(PROFILE_SPECS[profile])
    cfg["state_names"] = list(STATE_NAMES)
    cfg["dataset"]["raw_root"] = _resolve(base, dataset.get("raw_root", "data/raw"))
    cfg["dataset"]["episode_root"] = _resolve(base, dataset.get("episode_root", "data/episodes"))
    cfg["dataset"]["lerobot_root"] = _resolve(base, dataset.get("lerobot_root", "data/lerobot"))
    cfg["dataset"]["output_fps"] = int(fps)
    cfg["dataset"]["num_episodes"] = num_episodes
    cfg["dataset"]["max_skew_s"] = skew
    cfg["dataset"]["environment_reset_s"] = environment_reset_s
    cfg["capture"]["max_queue"] = int(capture.get("max_queue", 2048))
    cfg["capture"]["topic_timeout_s"] = float(capture.get("topic_timeout_s", 2.0))
    cfg["robot"]["sdk_init_settle_s"] = float(robot.get("sdk_init_settle_s", 2.0))
    cfg["robot"]["verify_gripper_position"] = bool(
        robot.get("verify_gripper_position", False)
    )
    cfg["robot"]["gripper_open_width_m"] = open_width
    cfg["robot"]["gripper_feedback_max_position_m"] = feedback_max
    cfg["robot"]["gripper_open_settle_s"] = float(
        robot.get("gripper_open_settle_s", 2.0)
    )
    cfg["robot"]["verify_gripper_open_position"] = bool(
        robot.get("verify_gripper_open_position", True)
    )
    return cfg


def manifest_metadata(cfg: dict, run_id: str, started_at: float) -> dict:
    spec = cfg["profile_spec"]
    features = {
        f"observation.images.{camera}": _image_feature() for camera in spec["cameras"]
    }
    features["observation.state"] = {
        "dtype": "float32", "shape": [23], "names": list(STATE_NAMES),
    }
    return {
        "schema_version": "galbot_sft_raw_v1",
        "run_id": run_id,
        "profile": cfg["profile"],
        "instruction": cfg["task"]["instruction"],
        "started_at": started_at,
        "observation_features": features,
        "robot_config": {
            "state_feature_names": list(STATE_NAMES),
            "action_feature_names": list(spec["action_names"]),
            "action_dim": len(spec["action_names"]),
        },
    }
