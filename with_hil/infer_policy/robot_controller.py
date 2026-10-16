"""
Robot controller module for Galbot G1.
Encapsulates galbot SDK operations: initialization, navigation, camera, motion, gripper.
"""

from __future__ import annotations

import math
import select
import threading
import time
from contextlib import nullcontext
from typing import Callable, Optional

try:
    import galbot_sdk.g1 as gm
    from galbot_sdk.g1 import GalbotNavigation
    from galbot_sdk.g1 import GalbotRobot
    from galbot_sdk.g1 import GalbotMotion
    from galbot_sdk.g1 import G1JointGroup, SensorType, JointCommand
    from galbot_sdk.g1 import G1ControllerName, ControlStatus
except ImportError:
    gm = None
    GalbotNavigation = None
    GalbotRobot = None
    GalbotMotion = None
    G1JointGroup = None
    SensorType = None
    JointCommand = None
    G1ControllerName = None
    ControlStatus = None

import time
import base64
import io
from typing import Sequence, List, Dict, Any, Optional

import numpy as np
from PIL import Image

import inference_logger as logger

from motion_processing import create_action_filter, create_interpolator

try:
    import cv2
except ImportError:
    cv2 = None


def interruptible_sleep(seconds: float, stop_flag: Optional[Callable[[], bool]] = None) -> bool:
    """Sleep for specified seconds, but can be interrupted by signals or stop flag.

    Returns False if interrupted/stopped, True if completed normally.
    """
    if seconds <= 0:
        return True

    start = time.perf_counter()
    remaining = seconds

    while remaining > 0:
        try:
            select.select([], [], [], min(remaining, 0.05))
        except select.error:
            pass

        if stop_flag is not None and stop_flag():
            return False

        elapsed = time.perf_counter() - start
        remaining = seconds - elapsed

    return True


def decode_compressed_image(compressed_image: dict, camera_info: dict = None) -> np.ndarray:
    """Decode compressed image from galbot SDK format."""
    if cv2 is None:
        raise ImportError("opencv-python is required for image decoding")

    image_data = compressed_image["data"]
    fmt = compressed_image["format"]
    if fmt == "rgb8":
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode RGB image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif fmt == "16UC1":
        depth_img = np.frombuffer(image_data, dtype=np.uint16).copy()
        if not camera_info:
            depth_img = depth_img.reshape((720, 1280))
        else:
            depth_img = depth_img.reshape((camera_info["height"], camera_info["width"]))
        depth_scale = compressed_image.get("depth_scale", 1000.0)
        return depth_img.astype(np.float32) / depth_scale
    else:
        raise ValueError(f"Unsupported image format: {fmt}")


class RobotController:
    """High-level controller for Galbot G1 robot."""

    def __init__(self, config: dict):
        self.cfg = config
        self.robot: Optional[GalbotRobot] = None
        self.nav: Optional[GalbotNavigation] = None
        self.motion: Optional[GalbotMotion] = None
        self.action_filter = create_action_filter(
            self._control_cfg().get("action_filter", {}), dim=23
        )
        self.interpolator = create_interpolator(
            self._control_cfg().get("interpolation", {}), dim=23
        )
        action_semantics = self._control_cfg().get("action_semantics", "absolute")
        if action_semantics != "absolute":
            raise ValueError(
                "robot.control.action_semantics must be 'absolute': the PI05 "
                "checkpoint postprocessor already converts relative predictions "
                "back to absolute joint targets"
            )
        self._joint_group_map = {
            "left_arm": G1JointGroup.left_arm,
            "right_arm": G1JointGroup.right_arm,
            "left_gripper": G1JointGroup.left_gripper,
            "right_gripper": G1JointGroup.right_gripper,
            "head": G1JointGroup.head,
        }
        # Try to add optional joint groups that may not exist in all SDK versions
        try:
            self._joint_group_map["leg"] = G1JointGroup.leg
        except AttributeError:
            pass
        try:
            self._joint_group_map["chassis"] = G1JointGroup.chassis
        except AttributeError:
            pass

        # head_camera_roi: warn once per camera when size mismatches / box is invalid
        self._head_roi_warned = set()
        roi_cfg = self.cfg.get("head_camera_roi") or {}
        if roi_cfg.get("enabled"):
            logger.info(
                "head_camera_roi enabled: expected_size=%s head_left=%s head_right=%s"
                % (
                    roi_cfg.get("expected_size"),
                    roi_cfg.get("head_left"),
                    roi_cfg.get("head_right"),
                )
            )
        else:
            logger.info("head_camera_roi disabled: no crop; native frames go to ImagePreprocessor")
        self.last_obs_meta = {
            "obs_capture_ts": None,
            "state_ts": None,
            "image_ts": {},
        }
        self.last_measured_state = None
        self.last_init_pose_verification = None

    def is_available(self) -> bool:
        return gm is not None

    def init(self) -> bool:
        if not self.is_available():
            logger.warning("galbot_sdk not available, running in mock mode")
            return False

        self.robot = GalbotRobot()
        self.nav = GalbotNavigation()
        self.motion = GalbotMotion()

        sensor_set = self._configured_sensor_set()
        ok_robot = self.robot.init(set(sensor_set))
        ok_motion = self.motion.init()
        ok_nav = False
        try:
            ok_nav = bool(self.nav.init())
        except Exception as e:
            logger.warning(f"Nav init raised, continuing without map navigation: {e}")
            ok_nav = False

        logger.info(f"Robot init: {ok_robot}, Motion init: {ok_motion}, Nav init: {ok_nav}")
        if not ok_robot:
            return False

        # Warm up sensors. Instead of a plain sleep, we continuously poll the
        # RGB cameras in the main thread: the galbot SDK camera stream tends to
        # stop producing data if it is not accessed for a few seconds, and
        # background-thread polling does not keep it alive for this SDK version.
        warmup = self.cfg.get("sensor_warmup_seconds", 5.0)
        if warmup > 0:
            logger.info(f"Warming up sensors for {warmup}s with camera pumping...")
            self._pump_cameras(duration=warmup)

        if not ok_motion:
            logger.warning("motion init failed, continuing without motion services")

        if not ok_nav:
            logger.warning("Nav init failed; map navigation disabled.")
        elif not self.nav.is_localized():
            logger.warning(
                "Not localized. Map navigate_to_goal will fail until "
                "localization_server has a valid pose."
            )

        return bool(ok_robot)

    def _pump_cameras(self, duration: float, interval: float = 0.2):
        """Wait like the official SDK example, then poll until frames arrive.

        Galbot examples do: robot.init(sensors); time.sleep(5); get_rgb_data().
        Immediate polling right after init often returns empty and never recovers.
        Head cameras are slower (~10s).
        """
        if self.robot is None:
            return
        camera_map = self._active_camera_sensor_map()
        if not camera_map:
            logger.warning("No cameras configured for pumping")
            return

        settle = min(5.0, max(0.0, duration * 0.5))
        if settle > 0:
            logger.info(f"Waiting {settle:.0f}s for camera streams to start (SDK pattern)...")
            time.sleep(settle)

        end_time = time.perf_counter() + max(0.0, duration - settle)
        first_success = {name: False for name in camera_map}
        logged_empty = set()
        while time.perf_counter() < end_time:
            pending = [n for n, ok in first_success.items() if not ok]
            if not pending:
                break
            for name in pending:
                sensor = camera_map[name]
                try:
                    rgb_data = self.robot.get_rgb_data(sensor)
                    if rgb_data:
                        img = decode_compressed_image(rgb_data)
                        logger.info(f"Camera '{name}' first frame: shape={img.shape}")
                        first_success[name] = True
                    elif name not in logged_empty:
                        logger.info(
                            f"Camera '{name}' get_rgb_data empty "
                            f"(type={type(rgb_data).__name__}, value={rgb_data!r})"
                        )
                        logged_empty.add(name)
                except Exception as e:
                    logger.warning(f"Camera pump error for '{name}': {e}")
            if not all(first_success.values()):
                time.sleep(interval)

        failed = [name for name, ok in first_success.items() if not ok]
        if failed:
            logger.warning(f"cameras that did not produce data during warmup: {failed}")
        else:
            logger.info("All configured cameras produced at least one frame")

    def _configured_sensor_set(self) -> list:
        sensor_set_cfg = self.cfg.get("sensor_set", [])
        sensor_set = []
        for name in sensor_set_cfg:
            sensor = getattr(SensorType, name, None)
            if sensor is not None:
                sensor_set.append(sensor)
            else:
                logger.warning(f"unknown sensor '{name}'")
        return sensor_set

    def _head_frame_wh(self, camera_name: str) -> Optional[tuple[int, int]]:
        img = self.get_rgb_image(camera_name)
        if img is None:
            return None
        height, width = img.shape[:2]
        return (int(width), int(height))

    def _head_cameras_match_expected(self, expect_w: int, expect_h: int) -> bool:
        camera_map = self._active_camera_sensor_map()
        names = [name for name in ("head_left", "head_right") if name in camera_map]
        if not names:
            logger.warning("no head cameras in active set; skip size check")
            return True
        for name in names:
            wh = self._head_frame_wh(name)
            if wh is None:
                logger.warning(f"head camera '{name}' empty after capture restart")
                return False
            if wh != (expect_w, expect_h):
                logger.warning(
                    f"head camera '{name}' size {wh[0]}x{wh[1]}, expected {expect_w}x{expect_h}"
                )
                return False
            logger.info(f"head camera '{name}' {wh[0]}x{wh[1]} matches {expect_w}x{expect_h}")
        return True

    def reacquire_cameras_after_capture_restart(
        self,
        expected_head_wh: tuple[int, int],
        warmup_seconds: Optional[float] = None,
    ) -> None:
        """Resubscribe the existing GalbotRobot after front_head_camera_capture restarts.

        Does not construct a second GalbotRobot and does not call request_shutdown().
        """
        if self.robot is None:
            raise RuntimeError("robot not initialized; cannot reacquire cameras")
        expect_w, expect_h = int(expected_head_wh[0]), int(expected_head_wh[1])
        warmup = float(
            self.cfg.get("sensor_warmup_seconds", 5.0) if warmup_seconds is None else warmup_seconds
        )
        logger.info(
            f"Reacquiring cameras after capture restart; expect head {expect_w}x{expect_h}"
        )
        self._pump_cameras(duration=warmup)
        if self._head_cameras_match_expected(expect_w, expect_h):
            return
        logger.warning("head size mismatch after pump; re-init sensors on existing GalbotRobot")
        if SensorType is None:
            raise RuntimeError("SDK SensorType unavailable")
        ok = self.robot.init(set(self._configured_sensor_set()))
        if not ok:
            raise RuntimeError("robot.init after capture restart returned False")
        self._pump_cameras(duration=warmup)
        if not self._head_cameras_match_expected(expect_w, expect_h):
            raise RuntimeError(
                f"head camera did not reach {expect_w}x{expect_h} after capture restart"
            )

    def shutdown(self):
        """Shutdown the robot and its subsystems."""
        logger.info("shutdown() entered")

        if self.nav is not None:
            try:
                logger.info("stopping navigation...")
                self.nav.stop_navigation()
                logger.info("navigation stopped")
            except Exception as e:
                logger.error(f"stop_navigation error: {e}")
            self.nav = None

        self.motion = None

        if self.robot is not None:
            robot = self.robot
            self.robot = None
            try:
                logger.info("calling request_shutdown()...")
                robot.request_shutdown()
                logger.info("request_shutdown() returned")
            except Exception as e:
                logger.error(f"request_shutdown error: {e}")
            try:
                logger.info("calling destroy()...")
                robot.destroy()
                logger.info("destroy() returned")
            except Exception as e:
                logger.error(f"destroy error: {e}")
        logger.info("shutdown() finished")

    def reset_action_filter(self):
        """Reset the action filter state. Call this at the start of each new inference run."""
        self.action_filter.reset()

    def get_current_pose(self) -> Optional[List[float]]:
        if self.nav is None:
            return None
        try:
            return list(self.nav.get_current_pose())
        except Exception as e:
            logger.error(f"get_current_pose error: {e}")
            return None

    def navigate_to_goal(self, goal_pose: Sequence[float], timeout: float = 30.0, retry: int = 3) -> bool:
        if self.nav is None:
            logger.warning("Navigation not available")
            return False
        try:
            if not self.nav.is_localized():
                logger.warning("Not localized; refuse navigate_to_goal")
                return False
            if not self._ensure_chassis_pose_ctrl():
                logger.warning("navigate_to_goal aborted: chassis pose controller not ready")
                return False
            goal = [float(v) for v in list(goal_pose)[:7]]
            cur_pose = self.get_current_pose()
            logger.info(f"Current pose: {cur_pose}")
            logger.info(f"Goal pose: {goal}")

            for attempt in range(retry, 0, -1):
                status = self.nav.navigate_to_goal(
                    goal, enable_collision_check=False, is_blocking=False, timeout=timeout
                )
                start = time.time()
                reached = False
                while time.time() - start < timeout:
                    time.sleep(0.5)
                    if self.nav.check_goal_arrival():
                        reached = True
                        break
                if reached:
                    logger.info("Navigation goal reached")
                    try:
                        self.nav.stop_navigation()
                    except Exception:
                        pass
                    return True
                logger.warning(
                    f"Navigation attempt failed, retry left: {attempt - 1}, status={status}"
                )
                try:
                    self.nav.stop_navigation()
                except Exception:
                    pass
                time.sleep(0.5)

            return False
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            try:
                self.nav.stop_navigation()
            except Exception:
                pass
            return False

    @staticmethod
    def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def _read_odom_xyyaw(self) -> Optional[Dict[str, float]]:
        if self.robot is None:
            return None
        try:
            raw = self.robot.get_odom()
        except Exception as e:
            logger.error(f"get_odom error: {e}")
            return None
        if raw is None:
            return None
        if isinstance(raw, dict):
            pos = list(raw.get("position") or [0, 0, 0])
            ori = list(raw.get("orientation") or [0, 0, 0, 1])
        else:
            pos = list(getattr(raw, "position", [0, 0, 0]))
            ori = list(getattr(raw, "orientation", [0, 0, 0, 1]))
        yaw = self._yaw_from_quat(*[float(v) for v in ori[:4]])
        return {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "yaw": yaw,
            "yaw_deg": math.degrees(yaw),
        }

    def _ensure_chassis_pose_ctrl(self) -> bool:
        if self.robot is None or G1ControllerName is None or ControlStatus is None:
            logger.warning("Chassis pose controller unavailable")
            return False
        target = str(G1ControllerName.CHASSIS_POSE_CTRL)
        active = None
        try:
            active = self.robot.get_active_controller("chassis")
            logger.info(f"Active chassis controller: {active}")
        except Exception as e:
            logger.warning(f"get_active_controller(chassis) failed: {e}")

        if active is not None and target in str(active):
            logger.info("Chassis already on %s, skip switch", target)
            return True

        status = self.robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
        if status == ControlStatus.SUCCESS:
            return True

        logger.warning(f"switch_controller CHASSIS_POSE_CTRL failed: {status}; recovering")
        try:
            self.robot.reload_controller("chassis")
            time.sleep(0.5)
            self.robot.acquire_controller(G1ControllerName.CHASSIS_POSE_CTRL)
            time.sleep(0.5)
            self.robot.start_controller("chassis")
            time.sleep(0.5)
            status = self.robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
        except Exception as e:
            logger.error(f"chassis controller recovery error: {e}")
            return False
        if status != ControlStatus.SUCCESS:
            logger.error(f"chassis controller still FAULT after recovery: {status}")
            return False
        return True

    def move_relative_base(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw_deg: float = 0.0,
        timeout: float = 20.0,
        speed_mps: float = 0.15,
        yaw_deg_s: float = 30.0,
    ) -> bool:
        """Move chassis relative to the current heading using set_base_pose.

        +X is forward, +Y is left, yaw_deg positive is left turn.
        Unspecified axes should be passed as 0. Does not use the map navigator.
        speed_mps controls interpolation duration: interp_s = dist / speed_mps.
        """
        if self.robot is None:
            logger.warning("move_relative_base: robot not initialized")
            return False
        if G1ControllerName is None or ControlStatus is None:
            logger.warning("move_relative_base: SDK imports unavailable")
            return False
        if abs(x) < 1e-9 and abs(y) < 1e-9 and abs(yaw_deg) < 1e-9:
            logger.info("move_relative_base: zero displacement, skip")
            return True

        if not self._ensure_chassis_pose_ctrl():
            return False

        before = self._read_odom_xyyaw()
        if before is None:
            logger.warning("move_relative_base: cannot read odom")
            return False

        speed = float(speed_mps)
        if speed <= 0:
            logger.warning("move_relative_base: speed_mps must be > 0, falling back to 0.15")
            speed = 0.15
        yaw_rate = float(yaw_deg_s) if float(yaw_deg_s) > 0 else 30.0

        dtheta = math.radians(float(yaw_deg))
        yaw0 = before["yaw"]
        goal_x = before["x"] + float(x) * math.cos(yaw0) - float(y) * math.sin(yaw0)
        goal_y = before["y"] + float(x) * math.sin(yaw0) + float(y) * math.cos(yaw0)
        goal_yaw = yaw0 + dtheta
        dist = math.hypot(float(x), float(y))
        interp_s = min(float(timeout), max(1.0, dist / speed + abs(float(yaw_deg)) / yaw_rate))

        logger.info(
            "move_relative_base relative x=%.3f y=%.3f yaw=%.1fdeg speed=%.3fm/s; "
            "odom from (%.3f, %.3f, %.1fdeg) to (%.3f, %.3f, %.1fdeg) interp=%.1fs",
            x, y, yaw_deg, speed,
            before["x"], before["y"], before["yaw_deg"],
            goal_x, goal_y, math.degrees(goal_yaw), interp_s,
        )
        try:
            status = self.robot.set_base_pose(
                float(goal_x),
                float(goal_y),
                float(goal_yaw),
                "odom",
                "odom",
                float(interp_s),
                True,
                float(timeout),
            )
        except Exception as e:
            logger.error(f"set_base_pose error: {e}")
            return False
        if status != ControlStatus.SUCCESS:
            logger.warning(f"set_base_pose failed: {status}")
            return False

        after = self._read_odom_xyyaw()
        if after is not None:
            logger.info(
                "move_relative_base done odom=(%.3f, %.3f, %.1fdeg) delta=(%+.3f, %+.3f, %+.1fdeg)",
                after["x"], after["y"], after["yaw_deg"],
                after["x"] - before["x"], after["y"] - before["y"],
                after["yaw_deg"] - before["yaw_deg"],
            )
        return True

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    @property
    def _CAMERA_SENSOR_MAP(self) -> dict:
        if SensorType is None:
            return {}
        return {
            "head_left": SensorType.HEAD_LEFT_CAMERA,
            "head_right": SensorType.HEAD_RIGHT_CAMERA,
            "left_arm": SensorType.LEFT_ARM_CAMERA,
            "right_arm": SensorType.RIGHT_ARM_CAMERA,
        }

    def _active_camera_sensor_map(self) -> dict:
        """Only cameras listed in observation_camera_map / sensor_set."""
        all_cams = self._CAMERA_SENSOR_MAP
        cam_map = self.cfg.get("observation_camera_map") or {}
        if cam_map:
            names = set(cam_map.values()) | set(cam_map.keys())
            return {n: all_cams[n] for n in names if n in all_cams}
        sensor_set = self.cfg.get("sensor_set") or []
        wanted = set()
        for item in sensor_set:
            key = str(item).replace("HEAD_LEFT_CAMERA", "head_left").replace(
                "HEAD_RIGHT_CAMERA", "head_right"
            ).replace("LEFT_ARM_CAMERA", "left_arm").replace(
                "RIGHT_ARM_CAMERA", "right_arm"
            )
            if key in all_cams:
                wanted.add(key)
            raw = str(item).lower()
            for name in all_cams:
                if name.replace("_", "") in raw.replace("_", ""):
                    wanted.add(name)
        return {n: all_cams[n] for n in wanted} if wanted else all_cams

    def get_rgb_image(self, camera_name: str = "left_arm") -> Optional[np.ndarray]:
        if self.robot is None:
            return None
        try:
            sensor_type = self._CAMERA_SENSOR_MAP.get(camera_name, SensorType.HEAD_RIGHT_CAMERA)
            rgb_data = None
            retries = self.cfg.get("camera_rgb_retries", 3)
            retry_delay = self.cfg.get("camera_rgb_retry_delay", 0.2)
            for attempt in range(retries):
                rgb_data = self.robot.get_rgb_data(sensor_type)
                if rgb_data:
                    break
                if attempt < retries - 1:
                    logger.warning(f"get_rgb_image({camera_name}) empty, retrying {attempt + 1}/{retries}...")
                    time.sleep(retry_delay)
            if not rgb_data:
                logger.warning(f"get_rgb_image({camera_name}) failed after {retries} attempts")
                return None
            return decode_compressed_image(rgb_data)
        except Exception as e:
            logger.error(f"get_rgb_image({camera_name}) error: {e}")
            return None

    _HEAD_ROI_CAMERAS = ("head_left", "head_right")

    def _apply_head_camera_roi(self, img: np.ndarray, camera_name: str) -> np.ndarray:
        """Crop SDK head-camera originals to align FOV with training views.

        Only ``head_left`` / ``head_right`` are eligible. Wrist cameras are never
        cropped, even if present in config. Applied after ``get_rgb_image()``
        decode and before ``_resize_to_feature_shape()``.

        If the decoded size is not exactly ``expected_size`` (default 960×1280,
        i.e. 1280×960 W×H), log a warning and return the image unchanged.
        """
        if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
            return img
        if camera_name not in self._HEAD_ROI_CAMERAS:
            return img

        roi_cfg = self.cfg.get("head_camera_roi") or {}
        if not roi_cfg.get("enabled"):
            return img

        box = roi_cfg.get(camera_name)
        if not box:
            return img

        h, w = int(img.shape[0]), int(img.shape[1])
        expected = roi_cfg.get("expected_size") or [960, 1280]
        try:
            exp_h, exp_w = int(expected[0]), int(expected[1])
        except (TypeError, ValueError, IndexError):
            exp_h, exp_w = 960, 1280

        if h != exp_h or w != exp_w:
            warn_key = f"{camera_name}:size"
            if warn_key not in self._head_roi_warned:
                logger.warning(
                    f"head_camera_roi skip {camera_name}: got {w}x{h}, "
                    f"expected {exp_w}x{exp_h}; crop skipped"
                )
                self._head_roi_warned.add(warn_key)
            return img

        try:
            y0, y1, x0, x1 = (int(v) for v in box)
        except (TypeError, ValueError):
            warn_key = f"{camera_name}:box"
            if warn_key not in self._head_roi_warned:
                logger.warning(
                    f"head_camera_roi skip {camera_name}: invalid box {box!r}"
                )
                self._head_roi_warned.add(warn_key)
            return img

        if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
            warn_key = f"{camera_name}:bounds"
            if warn_key not in self._head_roi_warned:
                logger.warning(
                    f"head_camera_roi skip {camera_name}: box [{y0}:{y1}, {x0}:{x1}] "
                    f"out of bounds for {w}x{h}"
                )
                self._head_roi_warned.add(warn_key)
            return img

        cropped = img[y0:y1, x0:x1]
        applied_key = f"{camera_name}:applied"
        if applied_key not in self._head_roi_warned:
            logger.info(
                f"head_camera_roi {camera_name}: {w}x{h} -> "
                f"img[{y0}:{y1}, {x0}:{x1}] shape={cropped.shape}"
            )
            self._head_roi_warned.add(applied_key)
        return cropped

    def get_depth_image(self, camera_name: str = "left_arm_depth_camera") -> Optional[np.ndarray]:
        if self.robot is None:
            return None
        try:
            sensor_type = SensorType.LEFT_ARM_DEPTH_CAMERA if "left" in camera_name else SensorType.RIGHT_ARM_DEPTH_CAMERA
            depth_data = self.robot.get_depth_data(sensor_type)
            if not depth_data:
                return None
            return decode_compressed_image(depth_data)
        except Exception as e:
            logger.error(f"get_depth_image error: {e}")
            return None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_joint_state(self) -> Optional[np.ndarray]:
        """Get current joint positions in PI05 model order (23 DOF).

        Order: right_arm(7) + right_gripper(1) + left_arm(7) + left_gripper(1) + leg(5) + head(2).
        Matches JOINT_GROUPS / DIM_MAPS from convert_galbot_mcap_to_lerobot_odom.py.

        Gripper values at index 7 (right_gripper) and 15 (left_gripper) are
        converted to VLA scale (0=closed, 100=open).
        """
        if self.robot is None:
            return None
        try:
            joint_groups = [
                G1JointGroup.right_arm,
                G1JointGroup.right_gripper,
                G1JointGroup.left_arm,
                G1JointGroup.left_gripper,
                G1JointGroup.leg,
                G1JointGroup.head,
            ]
            joint_states = self.robot.get_joint_states(joint_groups, [])
            if joint_states:
                positions = [js.position for js in joint_states]
                arr = np.asarray(positions, dtype=np.float32).reshape(-1)
                if arr.size != 23:
                    logger.error(f"get_joint_states returned {arr.size} values, expected 23")
                    return None
                # Grippers at index 7 (right_gripper) and 15 (left_gripper)
                arr[7] = self._gripper_state_to_vla_value(arr[7])
                arr[15] = self._gripper_state_to_vla_value(arr[15])
                return arr
        except Exception as e:
            logger.error(f"get_joint_states error: {e}")
        return None

    def get_vla_state(self) -> Optional[np.ndarray]:
        """Return full 23-DOF state vector in PI05 model order."""
        return self.get_joint_state()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def set_end_effector_pose(self, pose: Sequence[float], timeout: float = 5.0) -> bool:
        if self.motion is None:
            return False
        try:
            status = self.motion.set_end_effector_pose(
                target_pose=list(pose),
                end_effector_frame=self.cfg.get("end_effector_frame", "left_arm"),
                reference_frame=self.cfg.get("reference_frame", "base_link"),
                enable_collision_check=self.cfg.get("motion_collision_check", False),
                is_blocking=True,
                timeout=timeout,
                params=gm.Parameter()
            )
            ok = status == gm.MotionStatus.SUCCESS
            logger.info(f"set_end_effector_pose status={status}, ok={ok}")
            return ok
        except Exception as e:
            logger.error(f"set_end_effector_pose error: {e}")
            return False

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def _gripper_velocity(self, velocity: Optional[float] = None) -> float:
        return float(self.cfg.get("gripper_velocity", 0.15) if velocity is None else velocity)

    def _gripper_effort(self, effort: Optional[float] = None) -> float:
        return float(self.cfg.get("gripper_effort", 10.0) if effort is None else effort)

    def _gripper_width_command_scale(self) -> float:
        return float(self.cfg.get("gripper_width_command_scale", 1.0))

    def _control_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("control", {})

    def _set_joint_time_from_start(self) -> float:
        return float(
            self._control_cfg().get(
                "set_joint_time_from_start",
                self.cfg.get("set_joint_time_from_start", 0.0),
            )
        )

    def _gripper_action_scale(self) -> float:
        return float(self._control_cfg().get("gripper_action_scale", 0.0012))

    def _arm_max_delta_per_step(self, fallback: float) -> float:
        return float(self._control_cfg().get("max_joint_delta_per_step", fallback))

    def _gripper_max_delta_per_step(self) -> float:
        return float(self._control_cfg().get("gripper_max_delta_per_step", 20.0))

    def _joint_limits(self) -> Optional[np.ndarray]:
        """Return (23, 2) array of [lower, upper] per joint, or None if not configured."""
        limits_cfg = self._control_cfg().get("joint_limits")
        if not limits_cfg or len(limits_cfg) != 23:
            return None
        return np.array(limits_cfg, dtype=np.float32)

    def _clip_to_joint_limits(self, action: np.ndarray) -> np.ndarray:
        """Clip action to absolute joint limits. No-op if limits not configured."""
        limits = self._joint_limits()
        if limits is None:
            return action
        return np.clip(action, limits[:, 0], limits[:, 1])

    def _gripper_max_width(self) -> float:
        return float(self.cfg.get("gripper_max_width", 0.12))

    def _gripper_command_position(self, value: float, scale: float) -> float:
        position = float(value) * float(scale)
        min_width = float(self.cfg.get("gripper_min_width", 0.0))
        max_width = self._gripper_max_width()
        return float(np.clip(position, min_width, max_width))

    def _gripper_state_to_vla_value(self, position: float) -> float:
        """Return gripper state in the VLA 0~100 unit range."""
        position = float(position)
        max_width = self._gripper_max_width()
        if abs(position) <= max_width * 1.5:
            return float(np.clip(position / max_width * 100.0, 0.0, 100.0))
        return position

    # ------------------------------------------------------------------
    # Initial Pose
    # ------------------------------------------------------------------

    def _send_joint_group_positions(self, positions_map: dict) -> bool:
        """Send positions to multiple joint groups in one set_joint_commands call.

        Args:
            positions_map: {group_name: [position_values, ...]}
                Gripper groups are detected via joint_groups.is_gripper in config
                and auto-converted via _gripper_command_position.

        Returns:
            True on SUCCESS, False otherwise.
        """
        if self.robot is None or JointCommand is None:
            return False

        joint_groups_cfg = self.cfg.get("joint_groups", [])
        is_gripper_map = {g["name"]: g.get("is_gripper", False) for g in joint_groups_cfg}

        all_commands: list = []
        all_groups: list = []

        for name, positions in positions_map.items():
            try:
                g_enum = getattr(G1JointGroup, name)
            except AttributeError:
                logger.warning(f"G1JointGroup.{name} not available, skipping")
                continue

            is_gripper = is_gripper_map.get(name, False)
            for pos in positions:
                cmd = JointCommand()
                if is_gripper:
                    cmd.position = self._gripper_command_position(float(pos), self._gripper_width_command_scale())
                    cmd.velocity = self._gripper_velocity()
                    cmd.effort = self._gripper_effort()
                else:
                    cmd.position = float(pos)
                    cmd.velocity = 0.0
                    cmd.effort = 0.0
                all_commands.append(cmd)
            all_groups.append(g_enum)

        if not all_commands:
            return False

        time_from_start = self._set_joint_time_from_start()
        status = self.robot.set_joint_commands(all_commands, all_groups, [], time_from_start)
        return status == gm.ControlStatus.SUCCESS

    def _set_pose_with_retry(self, positions_map: dict, timeout: float, retries: int, label: str) -> bool:
        """Send a named joint pose and wait for the controller to settle."""
        for attempt in range(retries, 0, -1):
            ok = self._send_joint_group_positions(positions_map)
            logger.info(f"{label} attempt={retries - attempt + 1}/{retries} ok={ok}")
            if ok:
                time.sleep(timeout)
                return True
            if attempt > 1:
                time.sleep(1.0)
        return False

    def set_init_pose(self, timeout: float = 8.0, retries: int = 2) -> bool:
        """Set and read back the configured training initial pose."""
        init_cfg = self.cfg.get("init_joint_positions")
        if not init_cfg:
            logger.info("No init_joint_positions configured")
            return True  # not an error
        for attempt in range(1, max(1, retries) + 1):
            sent = self._send_joint_group_positions(init_cfg)
            logger.info(f"set_init_pose attempt={attempt}/{retries} command_ok={sent}")
            if sent:
                time.sleep(timeout)
                if self.verify_init_pose(init_cfg):
                    return True
            if attempt < retries:
                time.sleep(1.0)
        return False

    def set_named_pose(
        self,
        positions_map: dict,
        timeout: float = 8.0,
        retries: int = 2,
        label: str = "set_named_pose",
        verify: bool = True,
    ) -> bool:
        """Set a complete pose supplied by a collection/session config.

        This uses the same joint-command path as policy execution so the server
        keeps sole ownership of the SDK between episodes.
        """
        if not positions_map:
            logger.error("%s: empty positions_map", label)
            return False
        attempts = max(1, int(retries))
        for attempt in range(1, attempts + 1):
            sent = self._send_joint_group_positions(positions_map)
            logger.info(
                "%s attempt=%s/%s command_ok=%s", label, attempt, attempts, sent
            )
            if sent:
                time.sleep(max(0.0, float(timeout)))
                if not verify or self.verify_init_pose(positions_map):
                    return True
            if attempt < attempts:
                time.sleep(1.0)
        return False

    def verify_init_pose(self, positions_map: Optional[dict] = None) -> bool:
        """Verify training-pose arrival against fresh 23-D SDK feedback."""
        positions_map = positions_map or self.cfg.get("init_joint_positions") or {}
        state = self.get_joint_state()
        if state is None:
            logger.error("Initial pose verification failed: joint state unavailable")
            self.last_init_pose_verification = {"success": False, "reason": "state unavailable"}
            return False

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        groups = {g["name"]: g for g in self.cfg.get("joint_groups", [])}
        verify_cfg = self.cfg.get("init_pose_verify") or {}
        joint_tolerance = float(verify_cfg.get("joint_tolerance_rad", 0.03))
        gripper_tolerance = float(verify_cfg.get("gripper_tolerance_model", 3.0))
        result = {"success": True, "groups": {}}

        for name, values in positions_map.items():
            group = groups.get(name)
            if group is None:
                result["success"] = False
                result["groups"][name] = {"success": False, "reason": "missing joint group"}
                continue
            indices = list(group.get("indices") or [])
            target = np.asarray(values, dtype=np.float32).reshape(-1)
            if len(indices) != target.size:
                result["success"] = False
                result["groups"][name] = {"success": False, "reason": "dimension mismatch"}
                continue
            if group.get("is_gripper", False):
                target = np.asarray(
                    [
                        self._gripper_state_to_vla_value(
                            self._gripper_command_position(float(value), self._gripper_width_command_scale())
                        )
                        for value in target
                    ],
                    dtype=np.float32,
                )
                tolerance = gripper_tolerance
                unit = "model_0_100"
            else:
                tolerance = joint_tolerance
                unit = "rad"
            measured = state[indices]
            errors = np.abs(measured - target)
            group_ok = bool(np.all(errors <= tolerance))
            result["success"] = bool(result["success"] and group_ok)
            result["groups"][name] = {
                "success": group_ok,
                "max_abs_error": float(np.max(errors)),
                "tolerance": tolerance,
                "unit": unit,
                "target": [float(x) for x in target],
                "measured": [float(x) for x in measured],
            }

        self.last_init_pose_verification = result
        log = logger.info if result["success"] else logger.error
        log(f"Initial pose read-back verification: {result}")
        return bool(result["success"])

    def training_start_gripper_positions(self) -> dict:
        """Gripper model values from ``init_joint_positions`` (training frame 0).

        Training observations start at model value 0, which maps to 0.0 m.
        Recovery may leave the hardware at ``gripper_open_width`` (0.12 m / VLA 100);
        callers should apply this map before the first policy observation.
        """
        init_cfg = self.cfg.get("init_joint_positions") or {}
        gripper_map = {}
        for name in ("right_gripper", "left_gripper"):
            values = init_cfg.get(name)
            if not values:
                logger.warning(
                    f"No init_joint_positions.{name}; cannot match training-start gripper"
                )
                continue
            gripper_map[name] = [float(v) for v in values]
        return gripper_map

    def set_training_start_grippers(self, timeout: float = 2.0, retries: int = 2) -> bool:
        """Close both grippers to the training-episode start width before /execute."""
        gripper_map = self.training_start_gripper_positions()
        if not gripper_map:
            logger.error("Cannot set training-start grippers: missing init_joint_positions entries")
            return False
        logger.info(
            "Setting grippers to training start (physical m, not recovery open 0.12): %s"
            % gripper_map
        )
        return self._set_pose_with_retry(
            gripper_map, timeout, retries, "set_training_start_grippers"
        )

    def set_recovery_pose(self) -> bool:
        """Restore the initial pose and fully open both grippers.

        Keep recovery on the same set_joint_commands path as policy execution.
        This is the control path verified to work while the server owns the SDK.
        """
        init_cfg = self.cfg.get("init_joint_positions")
        if not init_cfg:
            logger.error("Cannot recover: no init_joint_positions configured")
            return False

        recovery_cfg = self.cfg.get("recovery", {})
        settle_seconds = float(recovery_cfg.get("settle_seconds", 8.0))
        retries = max(1, int(recovery_cfg.get("retries", 2)))
        open_width = float(recovery_cfg.get("gripper_open_width", self._gripper_max_width()))

        recovery_pose = {name: list(values) for name, values in init_cfg.items()}
        for name in ("right_gripper", "left_gripper"):
            if name in recovery_pose:
                recovery_pose[name] = [open_width] * len(recovery_pose[name])
            else:
                logger.warning(f"Recovery pose has no '{name}' entry; cannot force it open")

        logger.warning(
            f"Recovery via working set_joint_commands path: "
            f"initial pose + grippers fully open ({open_width:.3f} m)"
        )
        return self._set_pose_with_retry(
            recovery_pose,
            settle_seconds,
            retries,
            "set_recovery_pose",
        )

    # ------------------------------------------------------------------
    # Joint Commands
    # ------------------------------------------------------------------

    def _build_execute_group_spec(self):
        """Build joint_groups + model→cmd index mapping from vla_config.yaml.

        读取 ``robot.joint_groups``，格式与 replay_config.yaml 一致:
            [{name, indices, is_gripper?, enabled?}, ...]

        model→cmd 恒等映射: 未启用组对应的 model 索引设为 -1 跳过。

        Returns:
            (joint_groups, model_to_cmd, gripper_cmd_indices, total_cmd_joints)
        """
        MODEL_DIM = 23
        joint_groups_cfg = self.cfg.get("joint_groups", [])
        if not joint_groups_cfg:
            # 兜底: 未配置则不下发任何关节
            logger.warning("No joint_groups configured, action execution disabled")
            return [], np.full(MODEL_DIM, -1, dtype=int), [], 0

        model_to_cmd = np.full(MODEL_DIM, -1, dtype=int)
        gripper_cmd_indices: list[int] = []
        joint_groups: list = []

        cmd_idx = 0
        for g in joint_groups_cfg:
            if not g.get("enabled", True):
                continue

            name = g["name"]
            try:
                g_enum = getattr(G1JointGroup, name)
            except AttributeError:
                logger.warning(f"G1JointGroup.{name} not available, skipping")
                continue

            indices = g["indices"]
            is_gripper = g.get("is_gripper", False)

            for local_j, model_j in enumerate(indices):
                cmd_j = cmd_idx + local_j
                model_to_cmd[model_j] = cmd_j
                if is_gripper:
                    gripper_cmd_indices.append(cmd_j)

            joint_groups.append(g_enum)
            cmd_idx += len(indices)

        return joint_groups, model_to_cmd, gripper_cmd_indices, cmd_idx

    def _expand_model_action(self, action: Sequence[float], current_state: np.ndarray) -> np.ndarray:
        """Expand the configured model action prefix to the full 23-D robot state.

        The new pick policy predicts a 16-D arms/grippers prefix, while the
        legacy place policy predicts all 23 dimensions. Missing dimensions are
        copied from the latest measured state.
        """
        full_dim = 23
        model_dim = int(self.cfg.get("vla_action_dim", full_dim))
        if not 0 < model_dim <= full_dim:
            raise ValueError(f"invalid vla_action_dim={model_dim}; expected 1..{full_dim}")

        action_arr = np.asarray(action, dtype=np.float32)
        state_arr = np.asarray(current_state, dtype=np.float32).reshape(-1)
        if action_arr.ndim != 1 or action_arr.size not in (model_dim, full_dim):
            raise ValueError(
                f"action dim mismatch: got shape={action_arr.shape}, "
                f"expected {model_dim} or {full_dim}"
            )
        if state_arr.size != full_dim:
            raise ValueError(f"current state dim mismatch: got {state_arr.size}, expected {full_dim}")

        if action_arr.size == full_dim:
            return action_arr.copy()
        expanded = state_arr.copy()
        expanded[:model_dim] = action_arr
        return expanded

    def execute_action_chunk(self, actions: List[List[float]], dt: float, max_delta: float = 0.3,
                              skip_interp: bool = False,
                              sleep: bool = True,
                              stop_flag: Optional[Callable[[], bool]] = None,
                              command_guard=None):
        """Execute configured model actions via 23-D set_joint_commands.

        Full order: right_arm(7) + right_gripper(1) + left_arm(7) +
        left_gripper(1) + leg(5) + head(2). Every policy output is expanded and
        safety-checked in 23-D, while disabled joint groups are not sent.
        Grippers at model index 7 (right) and 15 (left) receive special position scaling.

        Config ``robot.joint_groups`` defines which groups + their model indices.

        Two-phase design:
          1) 对所有 action 做 delta 限幅 + 关节限位 + 滤波 → safe_actions
          2) 插值升频 (30→250Hz) → 逐帧下发 SDK

        Args:
            skip_interp: 如果为 True，跳过插值阶段，每帧直接下发 (RTC 模式用)。
        """
        if self.robot is None or JointCommand is None:
            return False

        MODEL_DIM = 23
        joint_groups, model_to_cmd, gripper_cmd_indices, total_cmd = self._build_execute_group_spec()

        if total_cmd == 0:
            logger.warning("No joint groups enabled for execution")
            return False

        joint_commands = [JointCommand() for _ in range(total_cmd)]
        gripper_velocity = self._gripper_velocity()
        gripper_effort = self._gripper_effort()
        gripper_action_scale = self._gripper_action_scale()
        arm_max_delta = self._arm_max_delta_per_step(max_delta)
        gripper_max_delta = self._gripper_max_delta_per_step()
        gripper_model_indices = frozenset([7, 15])  # right_gripper, left_gripper in 23-dim

        # --- Get current 23-dim state for delta limiting ---
        current_pos = self.get_joint_state()
        if current_pos is None:
            logger.error("Cannot execute action chunk: current joint state is unavailable")
            return False
        current_pos = np.asarray(current_pos, dtype=np.float32).reshape(-1)
        if current_pos.size != MODEL_DIM or not np.all(np.isfinite(current_pos)):
            logger.error(
                f"Cannot execute action chunk: invalid state shape={current_pos.shape}, "
                f"finite={np.all(np.isfinite(current_pos))}"
            )
            return False
        interpolation_start = current_pos.copy()

        # ================================================================
        # Phase 1: Compute per-action safe targets
        # ================================================================
        safe_actions = []
        for action in actions:
            try:
                action_arr = self._expand_model_action(action, current_pos)
            except ValueError as exc:
                logger.warning(str(exc))
                return False
            if not np.all(np.isfinite(action_arr)):
                logger.error(f"Refusing non-finite model action: {action_arr}")
                return False

            # Delta limiting (23-dim model space)
            delta = action_arr - current_pos
            clipped_delta = np.empty(MODEL_DIM, dtype=np.float32)
            for j in range(MODEL_DIM):
                if j in gripper_model_indices:
                    clipped_delta[j] = np.clip(delta[j], -gripper_max_delta, gripper_max_delta)
                else:
                    clipped_delta[j] = np.clip(delta[j], -arm_max_delta, arm_max_delta)
            safe_action = current_pos + clipped_delta

            # Absolute joint limits
            safe_action = self._clip_to_joint_limits(safe_action)

            # Action smoothing filter
            safe_action = self.action_filter(safe_action)

            safe_actions.append(safe_action)
            current_pos = safe_action

        if not safe_actions:
            return False

        # ================================================================
        # Phase 2: Interpolate (30Hz → 250Hz) then send
        # ================================================================
        interp_cfg = self._control_cfg().get("interpolation", {})
        interp_enabled = interp_cfg.get("enabled", False) and not skip_interp

        if interp_enabled:
            t0_interp = time.perf_counter()
            # The first model action belongs one model period in the future.
            # Start interpolation at the measured robot state so action[0] is
            # not applied as an instantaneous jump.
            waypoints = np.vstack([interpolation_start, np.stack(safe_actions)])
            interp_actions = self.interpolator.interpolate(waypoints)  # (M, 23)
            elapsed_interp_ms = (time.perf_counter() - t0_interp) * 1000
            logger.info(
                f"Interpolation: current+{len(safe_actions)} targets→{len(interp_actions)} frames, "
                f"{elapsed_interp_ms:.2f}ms"
            )
            send_dt = 1.0 / float(interp_cfg.get("output_hz", 250))
        else:
            interp_actions = [np.asarray(a, dtype=np.float32) for a in safe_actions]
            send_dt = dt

        command_failed = False
        for i, interp_action in enumerate(interp_actions):
            if stop_flag is not None and stop_flag():
                logger.warning(f"Action chunk interrupted at interpolated step {i}/{len(interp_actions)}")
                break
            # Interpolators (especially cubic/quintic) can overshoot waypoint
            # limits.  Clip every command, not only the low-frequency targets.
            interp_action = self._clip_to_joint_limits(
                np.asarray(interp_action, dtype=np.float32)
            )
            # Map 23-dim → joint_commands
            for model_j in range(MODEL_DIM):
                cmd_j = model_to_cmd[model_j]
                if cmd_j < 0:
                    continue
                cmd = joint_commands[cmd_j]
                if cmd_j in gripper_cmd_indices:
                    cmd.position = self._gripper_command_position(float(interp_action[model_j]), gripper_action_scale)
                    cmd.velocity = gripper_velocity
                    cmd.effort = gripper_effort
                else:
                    cmd.position = float(interp_action[model_j])
                    cmd.velocity = 0.0
                    cmd.effort = 0.0

            if i == 0:
                logger.info(
                    "SDK command 23d gripper_m=%s first6=%s"
                    % (
                        {
                            "right_arm_gripper": float(interp_action[7]) * gripper_action_scale,
                            "left_arm_gripper": float(interp_action[15]) * gripper_action_scale,
                        },
                        [float(x) for x in interp_action[:6]],
                    )
                )

            time_from_start = self._set_joint_time_from_start()
            try:
                with command_guard() if command_guard else nullcontext():
                    if stop_flag is not None and stop_flag():
                        break
                    status = self.robot.set_joint_commands(
                        joint_commands,
                        joint_groups,
                        [],
                        time_from_start,
                    )
                if status != gm.ControlStatus.SUCCESS:
                    logger.warning(f"set_joint_commands status={status} at interp step {i}/{len(interp_actions)}")
                    command_failed = True
                    if status == gm.ControlStatus.FAULT:
                        logger.error(
                            "Controller FAULT at interp step "
                            f"{i}/{len(interp_actions)}; stop commanding this chunk"
                        )
                        break
            except Exception as e:
                logger.error(f"set_joint_commands error at interp step {i}: {e}")
                command_failed = True
                break

            if sleep and send_dt > 0:
                time.sleep(send_dt)

        return not command_failed

    # ------------------------------------------------------------------
    # Observation (for VLA)
    # ------------------------------------------------------------------

    def _feature_shape(self, out_key: str, features: Optional[dict], default=(480, 640, 3)) -> tuple:
        """Return (H, W, C) for a camera feature, falling back to default."""
        if not features:
            return tuple(default)
        feat = features.get(f"observation.images.{out_key}") or features.get(out_key)
        if not feat:
            return tuple(default)
        shape = feat.get("shape")
        if not shape or len(shape) < 2:
            return tuple(default)
        h, w = int(shape[0]), int(shape[1])
        c = int(shape[2]) if len(shape) > 2 else int(default[2] if len(default) > 2 else 3)
        return (h, w, c)

    def _resize_to_feature_shape(self, img: np.ndarray, out_key: str, features: Optional[dict]) -> np.ndarray:
        """按 observation_features 中对应 key 的 shape 直接拉伸图像（无黑边填充）。

        shape 形如 [height, width, channels]（如 [480, 640, 3]）。
        尺寸一致时原样返回；与配置不一致时拉伸到目标尺寸。
        """
        if not features or cv2 is None:
            return img
        feat = features.get(f"observation.images.{out_key}") or features.get(out_key)
        if not feat:
            return img
        shape = feat.get("shape")
        if not shape or len(shape) < 2:
            return img
        target_h, target_w = int(shape[0]), int(shape[1])
        if target_h <= 0 or target_w <= 0:
            return img
        if img.shape[0] == target_h and img.shape[1] == target_w:
            return img
        interp = cv2.INTER_AREA if img.shape[0] * img.shape[1] > target_h * target_w else cv2.INTER_LINEAR
        return cv2.resize(img, (target_w, target_h), interpolation=interp)

    def get_observation(self, features: dict = None) -> Optional[Dict[str, Any]]:
        """获取 VLA 推理所需的完整观测字典。

        自动根据 observation_camera_map / observation_features 决定读取哪些摄像头，
        VLA 不用的摄像头不读取（节省带宽和计算）。
        所有 key 使用平键格式，与远端 PolicyServer 期望一致。

        Returns dict，例如:
          {
            "head_left":       ndarray(H, W, 3),       # uint8 RGB, native SDK size
            "left_arm":        ndarray(H, W, 3),
            "right_arm":       ndarray(H, W, 3),
            "right_arm_joint1": float_val,              # 23 个独立 state key
            "right_arm_joint2": float_val,
            ...
            "head_joint2":     float_val,
          }
        """
        cam_map = self.cfg.get("observation_camera_map")

        if not cam_map and features:
            # 自动推导: 从 observation_features 中找 dtype=="image" 的 key，
            # 提取末端摄像头名（如 "observation.images.head_left" → "head_left"），
            # 在 _CAMERA_SENSOR_MAP 中匹配 SDK 摄像头。
            cam_map = {}
            for feat_key, feat_info in features.items():
                if feat_info.get("dtype") == "image":
                    cam_name = feat_key.rsplit(".", 1)[-1] if "." in feat_key else feat_key
                    if cam_name in self._CAMERA_SENSOR_MAP:
                        cam_map[cam_name] = cam_name
                    else:
                        logger.warning(f"no SDK camera for feature '{feat_key}', skipping")

        if not cam_map:
            cam_map = {name: name for name in self._CAMERA_SENSOR_MAP}

        # ---- 1. 采集图像 ----
        obs: Dict[str, Any] = {}
        t0 = time.perf_counter()
        capture_wall = time.time()
        image_ts = {}
        missing = []
        for out_key, sdk_cam in cam_map.items():
            img = self.get_rgb_image(sdk_cam)
            image_ts[out_key] = time.time()
            if img is None:
                logger.warning(f"failed to get image from camera '{sdk_cam}'")
                missing.append(sdk_cam)
                continue
            # Native SDK resolution; ImagePreprocessor letterboxes to 224. Do not stretch.
            obs[out_key] = img
        if missing:
            logger.warning(f"missing cameras {missing}, skip this observation")
            return None
        t_images = (time.perf_counter() - t0) * 1000

        # ---- 2. 采集 23 维关节状态，拆成独立平键 ----
        state_names = self.cfg.get("state_feature_names") or []
        if len(state_names) != 23:
            logger.error(
                f"robot.state_feature_names length={len(state_names)}, expected 23: {state_names}"
            )
            return None
        state = self.get_vla_state()
        state_ts = time.time()
        t_state = (time.perf_counter() - t0) * 1000 - t_images
        self.last_obs_meta = {
            "obs_capture_ts": capture_wall,
            "state_ts": state_ts,
            "image_ts": image_ts,
        }
        if state is None:
            logger.error("get_vla_state returned None")
            return None
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size != 23:
            logger.error(f"get_vla_state size={state.size}, expected 23")
            return None
        if not np.all(np.isfinite(state)):
            logger.error(f"get_vla_state has non-finite values: {state}")
            return None
        for i, name in enumerate(state_names):
            obs[name] = float(state[i])

        t_total = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[RobotController] obs timing: images={t_images:.0f}ms, state={t_state:.0f}ms, "
            f"total={t_total:.0f}ms"
        )
        logger.info(
            f"[RobotController] state names={state_names} shape={state.shape} "
            f"head={state[:5].tolist()} obs.keys={list(obs.keys())}"
        )
        return obs

    # ------------------------------------------------------------------
    # Replay methods
    # ------------------------------------------------------------------

    def replay_init(self) -> bool:
        """Lightweight initialization for replay mode (no sensor warmup needed)."""
        if not self.is_available():
            logger.warning("galbot_sdk not available, running in mock mode")
            return False

        self.robot = GalbotRobot()
        self.nav = GalbotNavigation()
        self.motion = GalbotMotion()

        sensor_set_cfg = self.cfg.get("sensor_set", [])
        sensor_set = []
        for name in sensor_set_cfg:
            sensor = getattr(SensorType, name, None)
            if sensor is not None:
                sensor_set.append(sensor)
            else:
                logger.warning(f"unknown sensor '{name}'")

        ok_robot = self.robot.init(set(sensor_set))
        ok_motion = self.motion.init()
        ok_nav = self.nav.init()

        logger.info(f"Replay init: Robot={ok_robot}, Motion={ok_motion}, Nav={ok_nav}")
        if not ok_robot:
            return False
        if not ok_nav:
            logger.warning("Nav init failed, odom replay disabled")

        # Ensure localization for odom replay
        if ok_nav and not self.nav.is_localized():
            logger.info("Not localized, attempting relocalization...")
            self.nav.relocalize([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            for _ in range(20):
                time.sleep(0.5)
                if self.nav.is_localized():
                    break

        return ok_robot

    def _resolve_joint_groups(
        self, group_names: list
    ) -> tuple[list, list[str]]:
        """Resolve group name strings to G1JointGroup enums.

        Returns (groups, group_names_for_api) — the second list may contain
        string names for groups not found in the SDK (e.g. 'chassis').
        """
        groups = []
        api_names = []
        for name in group_names:
            g = self._joint_group_map.get(name)
            if g is not None:
                groups.append(g)
                api_names.append(name)
            else:
                logger.warning(f"Skipping unsupported joint group: {name}")
        return groups, api_names

    def replay_joint_state(
        self,
        group_slices: list,
        state_vector: np.ndarray,
        delta_limit: float = 0.3,
        gripper_action_scale: float = 0.0012,
    ) -> bool:
        """Set all joint groups from a full state/action vector for replay.

        一次 set_joint_commands 同时下发所有关节组（而非逐组串行调用）。

        Args:
            group_slices: List of dicts [{name, indices, is_gripper}, ...]
                where indices is a 1-D numpy int array.
            state_vector: Full joint state/action vector matching group_slices
            delta_limit: Max per-step delta for arm joints (radians)
            gripper_action_scale: Scale factor for gripper values (0~100 → 0~0.12)
        """
        if self.robot is None or JointCommand is None:
            logger.warning("Robot not available for replay")
            return False

        if len(state_vector) == 0:
            return False

        target = np.asarray(state_vector, dtype=np.float32)

        # Clip to absolute joint limits
        target = self._clip_to_joint_limits(target)

        time_from_start = self._set_joint_time_from_start()

        all_commands: list = []
        all_groups: list = []

        for gs in group_slices:
            name = gs["name"]
            indices = gs["indices"]  # np.ndarray
            is_gripper = gs.get("is_gripper", False)
            is_joint = gs.get("is_joint", True)

            if not is_joint:
                continue

            g_enum = self._joint_group_map.get(name)
            if g_enum is None:
                continue

            positions = target[indices]
            for pos in positions:
                cmd = JointCommand()
                if is_gripper:
                    cmd.position = self._gripper_command_position(
                        float(pos), gripper_action_scale
                    )
                    cmd.velocity = self._gripper_velocity()
                    cmd.effort = self._gripper_effort()
                else:
                    cmd.position = float(pos)
                    cmd.velocity = 0.0
                    cmd.effort = 0.0
                all_commands.append(cmd)
            all_groups.append(g_enum)

        if not all_commands:
            return False

        try:
            status = self.robot.set_joint_commands(
                all_commands,
                all_groups,
                [],
                time_from_start,
            )
            if status != gm.ControlStatus.SUCCESS:
                logger.warning(f"replay_joint_state status={status}")
                return False
        except Exception as e:
            logger.error(f"replay_joint_state error: {e}")
            return False

        return True

    def replay_odom_move(
        self,
        odom_delta: np.ndarray,
        odom_scale: float = 1.0,
        method: str = "move_straight_to",
    ) -> bool:
        """Move chassis based on odometry delta [dx, dy, dtheta].

        Args:
            odom_delta: [dx, dy, dtheta] relative movement in meters/radians
            odom_scale: Scale factor applied to the odometry delta
            method: Navigation method — "move_straight_to" (default) or
                    "navigate_to_goal"
        """
        if method == "move_straight_to":
            return self._replay_odom_move_straight(odom_delta, odom_scale)
        else:
            return self._replay_odom_navigate_to_goal(odom_delta, odom_scale)

    def _compute_map_goal(self, odom_delta: np.ndarray, odom_scale: float):
        """Compute absolute map-frame goal from map-frame odom delta.

        The odom_delta comes from observation.odom which is already in map frame.
        Simply add the delta to current pose to get the goal.

        cur_pose format: [x, y, z, qx, qy, qz, qw]

        Returns (goal_pose, dx, dy, dtheta) or (None, ...) if skipped.
        """
        if self.nav is None:
            return None, 0, 0, 0

        dx = float(odom_delta[0]) * odom_scale
        dy = float(odom_delta[1]) * odom_scale
        dtheta = float(odom_delta[2]) * odom_scale

        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dtheta) < 1e-6:
            return None, dx, dy, dtheta

        cur_pose = self.get_current_pose()
        if cur_pose is None:
            logger.warning("Cannot get current pose for odom move")
            return None, dx, dy, dtheta

        goal_x = cur_pose[0] + dx
        goal_y = cur_pose[1] + dy

        qx, qy, qz, qw = cur_pose[3], cur_pose[4], cur_pose[5], cur_pose[6]
        cur_theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        goal_theta = cur_theta + dtheta

        new_qw = np.cos(goal_theta / 2.0)
        new_qz = np.sin(goal_theta / 2.0)

        goal = [
            goal_x,
            goal_y,
            cur_pose[2],
            0.0,
            0.0,
            new_qz,
            new_qw,
        ]
        return goal, dx, dy, dtheta

    def _replay_odom_move_straight(
        self,
        odom_delta: np.ndarray,
        odom_scale: float = 1.0,
    ) -> bool:
        """Chassis movement via move_straight_to (straight-line path).

        Switches controller to CHASSIS_POSE_CTRL, calls move_straight_to,
        waits, then stops navigation.

        move_straight_to receives RELATIVE position [dx, dy, dtheta]
        in robot's local coordinate frame.

        observation.odom is in map frame, so we need to transform
        the delta from map frame to robot local frame.
        """
        if self.robot is None or self.nav is None:
            return False
        if G1ControllerName is None or ControlStatus is None:
            logger.warning("move_straight_to: SDK imports unavailable")
            return False

        dx = float(odom_delta[0]) * odom_scale
        dy = float(odom_delta[1]) * odom_scale
        dtheta = float(odom_delta[2]) * odom_scale

        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dtheta) < 1e-6:
            return True

        cur_pose = self.get_current_pose()
        if cur_pose is None:
            logger.warning("Cannot get current pose for odom move")
            return False

        qx, qy, qz, qw = cur_pose[3], cur_pose[4], cur_pose[5], cur_pose[6]
        theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        local_dx = dx * cos_t + dy * sin_t
        local_dy = -dx * sin_t + dy * cos_t

        try:
            # Switch to chassis pose controller
            res = self.robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
            if res != ControlStatus.SUCCESS:
                logger.warning(f"switch_controller CHASSIS_POSE_CTRL failed: {res}")
                return False

            nav_timeout = self.cfg.get("replay", {}).get("odom_nav_timeout", 0.1)
            
            self.nav.move_straight_to([local_dx, local_dy, dtheta], is_blocking=False, timeout=nav_timeout)

            return True
        except Exception as e:
            logger.error(f"replay_odom_move_straight error: {e}")
            try:
                self.nav.stop_navigation()
            except Exception:
                pass
            return False

    def _replay_odom_navigate_to_goal(
        self,
        odom_delta: np.ndarray,
        odom_scale: float = 1.0,
    ) -> bool:
        """Chassis movement via navigate_to_goal (planned path, collision-aware).

        Kept as fallback / alternative to move_straight_to.
        """
        if self.nav is None:
            return False

        goal, dx, dy, dtheta = self._compute_map_goal(odom_delta, odom_scale)
        if goal is None:
            return True

        try:
            nav_timeout = self.cfg.get("replay", {}).get("odom_nav_timeout", 1.0)
            self.nav.navigate_to_goal(
                goal,
                enable_collision_check=False,
                is_blocking=False,
                timeout=nav_timeout,
            )
            start = time.time()
            reached = False
            while time.time() - start < nav_timeout:
                interruptible_sleep(0.05)
                if self.nav.check_goal_arrival():
                    reached = True
                    break
            try:
                self.nav.stop_navigation()
            except Exception:
                pass
            if not reached:
                logger.info(f"Odom move incomplete: delta=[{dx:.4f}, {dy:.4f}, {dtheta:.4f}]")
            return True
        except Exception as e:
            logger.error(f"replay_odom_navigate error: {e}")
            try:
                self.nav.stop_navigation()
            except Exception:
                pass
            return False

    def replay_get_robot_joint_groups(self) -> list:
        """Return list of joint group names that are supported by the SDK.

        Used by the replay script to determine which parts of the state vector
        can be sent to the robot.
        """
        return list(self._joint_group_map.keys())
