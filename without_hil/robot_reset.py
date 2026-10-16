"""Minimal Galbot SDK reset path; no inference or camera ownership."""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from config import GRIPPER_GROUPS, GROUPS

try:
    import galbot_sdk.g1 as gm
    from galbot_sdk.g1 import GalbotRobot, G1JointGroup
except ImportError:
    gm = GalbotRobot = G1JointGroup = None

LOG = logging.getLogger("sft_collect")


class RobotResetter:
    def __init__(self, config: dict):
        self.cfg = config
        self.robot: Optional[object] = None
        self.last_verification = None
        self.last_open_verification = None

    def init(self) -> None:
        if GalbotRobot is None:
            raise RuntimeError("galbot_sdk.g1 is unavailable")
        self.robot = GalbotRobot()
        if not self.robot.init(set()):
            raise RuntimeError("GalbotRobot.init returned False")
        # init() may return before the motion channel is fully connected. The
        # vendor examples also wait briefly before issuing the first command.
        settle_s = float(self.cfg.get("sdk_init_settle_s", 2.0))
        if settle_s > 0:
            LOG.info("等待机器人控制通信稳定：%.1f 秒", settle_s)
            time.sleep(settle_s)

    def close(self) -> None:
        robot, self.robot = self.robot, None
        if robot is None:
            return
        try:
            robot.request_shutdown()
        except Exception as exc:
            LOG.warning("请求关闭机器人接口失败：%s", exc)
        try:
            robot.destroy()
        except Exception as exc:
            LOG.warning("销毁机器人接口失败：%s", exc)

    def _move(self) -> bool:
        if self.robot is None:
            raise RuntimeError("robot resetter is not initialized")
        pose = self.cfg["reset_pose"]
        body_groups = ["right_arm", "left_arm", "leg", "head"]
        body_positions = [
            float(value)
            for name in body_groups
            for value in pose[name]
        ]
        status = self.robot.set_joint_positions(
            body_positions,
            body_groups,
            [],
            True,
            float(self.cfg["reset_max_speed_rad_s"]),
            float(self.cfg["reset_timeout_s"]),
        )
        LOG.info("发送全身复位命令，关节组=%s，状态=%s", body_groups, status)
        if status != gm.ControlStatus.SUCCESS:
            return False

        scale = float(self.cfg["gripper_action_scale"])
        velocity = float(self.cfg.get("gripper_velocity_mps", 0.1))
        effort = float(self.cfg.get("gripper_effort_n", 10.0))
        for name in ("right_gripper", "left_gripper"):
            width_m = float(pose[name][0]) * scale
            gripper_status = self.robot.set_gripper_command(
                getattr(G1JointGroup, name), width_m, velocity, effort, False
            )
            LOG.info("发送复位夹爪命令，夹爪=%s，宽度=%.6f 米，状态=%s",
                     name, width_m, gripper_status)
            if gripper_status != gm.ControlStatus.SUCCESS:
                return False
        time.sleep(float(self.cfg.get("gripper_settle_s", 2.0)))
        return True

    def _state(self) -> np.ndarray:
        groups = [getattr(G1JointGroup, name) for name in GROUPS]
        states = self.robot.get_joint_states(groups, [])
        values = np.asarray([item.position for item in states], dtype=np.float32).reshape(-1)
        if values.size != 23:
            raise RuntimeError(f"get_joint_states returned {values.size} values, expected 23")
        # set_gripper_command() accepts total opening width (0..0.12 m), while
        # get_joint_states() reports one finger's travel (about 0..0.06 m).
        # Keep these two physical quantities separate when mapping feedback to
        # the training 0..100 range.
        feedback_max = float(self.cfg.get("gripper_feedback_max_position_m", 0.06))
        values[7] = np.clip(values[7] / feedback_max * 100.0, 0.0, 100.0)
        values[15] = np.clip(values[15] / feedback_max * 100.0, 0.0, 100.0)
        return values

    def open_grippers(self) -> bool:
        """Fully open both grippers and verify their feedback before reset."""
        if self.robot is None:
            raise RuntimeError("robot resetter is not initialized")
        max_width = float(self.cfg.get("gripper_max_width_m", 0.12))
        width_m = float(self.cfg.get("gripper_open_width_m", max_width))
        velocity = float(self.cfg.get("gripper_velocity_mps", 0.1))
        effort = float(self.cfg.get("gripper_effort_n", 10.0))
        command_ok = True
        command_status = {}
        for name in ("right_gripper", "left_gripper"):
            status = self.robot.set_gripper_command(
                getattr(G1JointGroup, name), width_m, velocity, effort, False
            )
            command_status[name] = str(status)
            command_ok = command_ok and status == gm.ControlStatus.SUCCESS
            LOG.info(
                "发送完全张开命令，夹爪=%s，宽度=%.6f 米，状态=%s",
                name, width_m, status,
            )
        time.sleep(float(self.cfg.get("gripper_open_settle_s", 2.0)))

        verification = {
            "success": bool(command_ok),
            "command_status": command_status,
            "target_width_m": width_m,
            "groups": {},
        }
        if command_ok and bool(self.cfg.get("verify_gripper_open_position", True)):
            try:
                measured = self._state()
                target_model = width_m / max_width * 100.0
                tolerance = float(self.cfg["gripper_tolerance_model"])
                for name, index in (("right_gripper", 7), ("left_gripper", 15)):
                    value = float(measured[index])
                    error = abs(value - target_model)
                    ok = error <= tolerance
                    verification["success"] = verification["success"] and ok
                    verification["groups"][name] = {
                        "success": ok,
                        "target_model": target_model,
                        "measured_model": value,
                        "abs_error": error,
                        "tolerance": tolerance,
                    }
            except Exception as exc:
                verification["success"] = False
                verification["error"] = str(exc)
        self.last_open_verification = verification
        log = LOG.info if verification["success"] else LOG.error
        log("夹爪完全张开校验结果：%s", verification)
        return bool(verification["success"])

    def verify(self) -> bool:
        measured = self._state()
        result = {"success": True, "groups": {}}
        verify_grippers = bool(self.cfg.get("verify_gripper_position", False))
        for name, indices in GROUPS.items():
            target = np.asarray(self.cfg["reset_pose"][name], dtype=np.float32)
            error = np.abs(measured[indices] - target)
            tolerance = float(
                self.cfg["gripper_tolerance_model"] if name in GRIPPER_GROUPS
                else self.cfg["joint_tolerance_rad"]
            )
            within_tolerance = bool(np.all(error <= tolerance))
            # Fully closed grippers commonly settle at a non-zero encoder
            # reading. The SDK command status is authoritative by default;
            # feedback remains visible as a diagnostic. Body joints are always
            # verified strictly.
            position_verified = name not in GRIPPER_GROUPS or verify_grippers
            ok = within_tolerance if position_verified else True
            result["success"] = result["success"] and ok
            result["groups"][name] = {
                "success": ok,
                "within_tolerance": within_tolerance,
                "position_verified": position_verified,
                "max_abs_error": float(np.max(error)),
                "tolerance": tolerance,
                "target": target.tolist(),
                "measured": measured[indices].tolist(),
            }
        self.last_verification = result
        return bool(result["success"])

    def reset(self) -> bool:
        retries = int(self.cfg["reset_retries"])
        for attempt in range(1, retries + 1):
            try:
                before = self._state()
                LOG.info("复位前机器人状态：%s", before.tolist())
            except Exception as exc:
                LOG.warning("无法读取复位前机器人状态：%s", exc)
            moved = self._move()
            LOG.info("机器人复位尝试 %d/%d，运动命令结果=%s", attempt, retries, moved)
            if moved:
                try:
                    if self.verify():
                        LOG.info("机器人复位到位校验通过：%s", self.last_verification)
                        return True
                except Exception as exc:
                    LOG.error("机器人复位到位校验失败：%s", exc)
            if attempt < retries:
                time.sleep(1.0)
        LOG.error("机器人未到达配置的复位姿态：%s", self.last_verification)
        return False
