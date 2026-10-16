"""Restart front_head_camera_capture with working or data-collection para_dir.

Restarting the head driver can knock down the wrist RealSense capture
processes. After a head switch, start left/right arm capture if they are
missing. Do not use ``ps -C`` (comm is truncated to 15 chars) and do not
``pkill -f`` (it can kill the caller).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

CAPTURE_BIN = "/data/galbot/bin/front_head_camera_capture"
PROC_NAME = "front_head_camera_capture"
PARA = {
    "working": "/userdata/user_config/base",
    "data_collection": "/userdata/user_config/data_collection",
}
EXPECTED_WH = {
    "working": (1280, 960),
    "data_collection": (640, 480),
}
# Wrist cameras stay on the working (1280x720) driver. ImagePreprocessor
# center-crops 16:9 to 4:3. Do not follow the head into data_collection.
ARM_PARA = PARA["working"]
ARM_CAPTURES = (
    ("/data/galbot/bin/left_arm_camera_capture", "left_arm"),
    ("/data/galbot/bin/right_arm_camera_capture", "right_arm"),
)
LOG_DIR = Path("/tmp/head_capture_switch")


def _bin_pids(binary: str) -> list[int]:
    """Match a capture binary via /proc cmdline."""
    pids = []
    marker = (binary + "\x00").encode()
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if cmdline.startswith(marker):
            pids.append(int(entry.name))
    return pids


def _pids() -> list[int]:
    """Match the head capture binary via /proc cmdline."""
    return _bin_pids(CAPTURE_BIN)


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
    except OSError:
        return ""


def _log_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def stop_capture(timeout_s: float = 6.0) -> None:
    pids = _pids()
    if not pids:
        print("no capture process")
        return
    for pid in pids:
        print(f"SIGTERM pid={pid} {_cmdline(pid)}")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            print(f"  {exc}")
    deadline = time.time() + timeout_s
    while time.time() < deadline and _pids():
        time.sleep(0.2)
    for pid in _pids():
        print(f"SIGKILL pid={pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.5)
    leftover = _pids()
    if leftover:
        raise RuntimeError(f"capture still running: {leftover}")
    # V4L2 device needs a moment after the process dies.
    time.sleep(3.0)
    print("capture stopped")


def start_capture(mode: str, retries: int = 3) -> int:
    if mode not in PARA:
        raise ValueError(f"unknown head capture mode '{mode}'")
    para = PARA[mode]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for attempt in range(1, retries + 1):
        log_path = LOG_DIR / f"{mode}_{attempt}.log"
        log_f = open(log_path, "wb")
        proc = subprocess.Popen(
            [CAPTURE_BIN, para],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd="/tmp",
        )
        print(f"start {mode} attempt={attempt} pid={proc.pid} para={para}")
        time.sleep(6.0)
        text = _log_text(log_path)
        alive = proc.poll() is None
        crashed = "V4L2 camera initialization failed" in text
        if alive and not crashed:
            print(f"  capture up, log={log_path}")
            return proc.pid
        last_err = text[-1200:]
        print(f"  start failed alive={alive} crashed={crashed}")
        if proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(4.0)
    raise RuntimeError(f"could not start {mode} capture\n{last_err}")


def _start_arm_capture(binary: str, name: str, retries: int = 3) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    last_err = ""
    env = os.environ.copy()
    lib = "/data/galbot/lib"
    env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    for attempt in range(1, retries + 1):
        log_path = LOG_DIR / f"{name}_{attempt}.log"
        log_f = open(log_path, "wb")
        proc = subprocess.Popen(
            [binary, ARM_PARA],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd="/tmp",
            env=env,
        )
        print(f"start {name} attempt={attempt} pid={proc.pid} para={ARM_PARA}")
        time.sleep(6.0)
        text = _log_text(log_path)
        alive = proc.poll() is None
        crashed = "V4L2 camera initialization failed" in text or "No device connected" in text
        if alive and not crashed:
            print(f"  {name} capture up, log={log_path}")
            return proc.pid
        last_err = text[-1200:]
        print(f"  {name} start failed alive={alive} crashed={crashed}")
        if proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(4.0)
    raise RuntimeError(f"could not start {name} capture\n{last_err}")


def ensure_arm_captures() -> dict[str, list[int]]:
    """Start wrist capture processes if a head-camera restart knocked them down."""
    running: dict[str, list[int]] = {}
    for binary, name in ARM_CAPTURES:
        pids = _bin_pids(binary)
        if pids:
            print(f"{name} capture already running pids={pids}")
            running[name] = pids
            continue
        print(f"{name} capture missing; restarting")
        _start_arm_capture(binary, name)
        running[name] = _bin_pids(binary)
        if not running[name]:
            raise RuntimeError(f"{name} capture exited immediately after restart")
    return running


def current_head_mode() -> str | None:
    """Return working / data_collection from the running head capture cmdline."""
    pids = _pids()
    if not pids:
        return None
    cmd = _cmdline(pids[0])
    dc = PARA["data_collection"]
    working = PARA["working"]
    if dc in cmd:
        return "data_collection"
    if working in cmd:
        return "working"
    return None


def switch_to(mode: str) -> int:
    stop_capture()
    pid = start_capture(mode)
    ensure_arm_captures()
    return pid


def ensure_mode(mode: str, settle_s: float = 3.0) -> int:
    """Switch head capture only if it is not already on ``mode``.

    Restarting capture after GalbotMotion.init() can unmatch
    singorix/wbcs/target_server. Call this before Motion.init, and skip the
    SIGTERM restart when the driver is already correct.
    """
    if mode not in PARA:
        raise ValueError(f"unknown head capture mode '{mode}'")
    current = current_head_mode()
    pids = _pids()
    if current == mode and pids:
        print(f"head capture already {mode} pids={pids}; skip restart")
        ensure_arm_captures()
        return pids[0]
    print(f"head capture mode={current!r} -> {mode}")
    pid = switch_to(mode)
    if settle_s > 0:
        time.sleep(settle_s)
    return pid
