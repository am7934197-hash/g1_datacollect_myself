#!/usr/bin/env python3
"""Interactive, inference-free G1 SFT episode collector."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pathlib import Path
import select
import shutil
import sys
import termios
import time
import tty
from typing import Optional

from config import load_config, manifest_metadata
from raw_capture import CaptureBridge
from robot_reset import RobotResetter

LOG = logging.getLogger("sft_collect")


class State(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    RECORDING = "RECORDING"
    WAITING_NEXT = "WAITING_NEXT"
    RESETTING = "RESETTING"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class SessionMachine:
    def __init__(self):
        self.state = State.STARTING

    def transition(self, target: State) -> None:
        allowed = {
            State.STARTING: {State.READY, State.ERROR, State.STOPPED},
            State.READY: {State.RECORDING, State.RESETTING, State.STOPPED, State.ERROR},
            State.RECORDING: {State.WAITING_NEXT, State.ERROR},
            State.WAITING_NEXT: {State.RESETTING, State.STOPPED, State.ERROR},
            State.RESETTING: {State.READY, State.ERROR, State.STOPPED},
            State.ERROR: {State.STOPPED},
            State.STOPPED: set(),
        }
        if target not in allowed[self.state]:
            raise RuntimeError(f"invalid state transition {self.state} -> {target}")
        self.state = target


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def cbreak_stdin():
    if not sys.stdin.isatty():
        raise RuntimeError("interactive collection requires a TTY")
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def normalize_key(value: str) -> str:
    if value in ("\x1b[C", "\x1bOC"):
        return "RIGHT"
    if value in ("\x1b[D", "\x1bOD"):
        return "LEFT"
    return value.lower()


class TerminalKeyDecoder:
    """Decode terminal keys even when an escape sequence arrives in pieces."""

    def __init__(self):
        self.buffer = ""

    def feed(self, value: str) -> list:
        self.buffer += value
        keys = []
        while self.buffer:
            arrow = None
            if self.buffer.startswith(("\x1b[C", "\x1bOC")):
                arrow = "RIGHT"
            elif self.buffer.startswith(("\x1b[D", "\x1bOD")):
                arrow = "LEFT"
            if arrow is not None:
                keys.append(arrow)
                self.buffer = self.buffer[3:]
                continue
            if self.buffer[0] == "\x1b":
                # ESC, ESC[ and ESCO may be the beginning of a fragmented
                # cursor-key sequence. Keep them until later input arrives.
                if self.buffer in ("\x1b", "\x1b[", "\x1bO"):
                    break
                keys.append(normalize_key(self.buffer[0]))
                self.buffer = self.buffer[1:]
                continue
            keys.append(normalize_key(self.buffer[0]))
            self.buffer = self.buffer[1:]
        return keys


class Collector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.machine = SessionMachine()
        self.capture = CaptureBridge(cfg["capture"])
        self.resetter = RobotResetter(cfg["robot"])
        self.pending: Optional[Path] = None
        self.run_id: Optional[str] = None
        self.started_at: Optional[float] = None
        self.completed = 0

    @property
    def raw_root(self) -> Path:
        return Path(self.cfg["dataset"]["raw_root"])

    def initialize(self) -> None:
        self.raw_root.mkdir(parents=True, exist_ok=True)
        Path(self.cfg["dataset"]["episode_root"]).mkdir(parents=True, exist_ok=True)
        (self.raw_root / ".in_progress").mkdir(parents=True, exist_ok=True)
        self.capture.start()
        deadline = time.monotonic() + 30.0
        while not self.capture.status()["ready"] and time.monotonic() < deadline:
            status = self.capture.status()
            if status.get("error"):
                raise RuntimeError(status["error"])
            LOG.info("正在等待相机和机器人状态话题：%s", status.get("ages", {}))
            time.sleep(1.0)
        if not self.capture.status()["ready"]:
            raise RuntimeError(f"采集话题未就绪：{self.capture.status()}")
        LOG.info("相机和机器人状态话题已就绪")
        self.resetter.init()
        self.machine.transition(State.READY)
        LOG.info("启动复位：正在将机器人移动到配置的初始姿态")
        self._reset()
        LOG.info("启动复位完成，机器人已到达初始姿态")
        self._environment_reset_countdown()
        self.start_episode()

    def start_episode(self) -> None:
        if self.machine.state != State.READY:
            return
        now = time.time()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.pending = self.raw_root / ".in_progress" / self.run_id
        metadata = manifest_metadata(self.cfg, self.run_id, now)
        self.capture.begin(self.pending, metadata)
        self.started_at = now
        self.machine.transition(State.RECORDING)
        LOG.info(
            "开始采集第 %d/%d 条数据，编号：%s；按右方向键保存，按左方向键丢弃重采",
            self.completed + 1, self.cfg["dataset"]["num_episodes"], self.run_id,
        )

    def _finish_capture(self) -> dict:
        reply = self.capture.end()
        result = (reply or {}).get("result") or {}
        if result.get("error") or not result.get("finalized"):
            raise RuntimeError(f"原始数据归档不完整：{result}")
        return result

    def _environment_reset_countdown(self) -> None:
        seconds = float(self.cfg["dataset"].get("environment_reset_s", 10.0))
        if seconds <= 0:
            LOG.info("环境整理倒计时已关闭，立即开始下一条数据采集")
            return
        LOG.info("请整理任务环境，倒计时结束后将自动开始采集")
        remaining = seconds
        while remaining > 0:
            LOG.info("环境整理倒计时：%d 秒", int(remaining + 0.999))
            interval = min(1.0, remaining)
            time.sleep(interval)
            remaining -= interval
        LOG.info("环境整理倒计时结束")

    def _reset(self, *, open_grippers_first: bool = False) -> None:
        self.machine.transition(State.RESETTING)
        if open_grippers_first:
            LOG.info("正在完全张开左右夹爪")
            if not self.resetter.open_grippers():
                self.machine.transition(State.ERROR)
                raise RuntimeError("复位前左右夹爪未能完全张开")
            LOG.info("左右夹爪已完全张开")
        LOG.info("正在复位机器人到配置姿态")
        if not self.resetter.reset():
            self.machine.transition(State.ERROR)
            raise RuntimeError("机器人自动复位失败，采集已停止")
        self.run_id = None
        self.pending = None
        self.started_at = None
        self.machine.transition(State.READY)

    def reset_and_start(self) -> None:
        if self.machine.state not in (State.READY, State.WAITING_NEXT):
            return
        self._reset(open_grippers_first=self.machine.state == State.WAITING_NEXT)
        LOG.info("机器人复位完成")
        self._environment_reset_countdown()
        self.start_episode()

    def reset_and_stop(self) -> None:
        if self.machine.state != State.WAITING_NEXT:
            return
        self._reset(open_grippers_first=True)
        self.machine.transition(State.STOPPED)
        LOG.info("已完成 %d 条数据采集，最终复位完成，程序退出", self.completed)

    def save_episode(self) -> None:
        if self.machine.state != State.RECORDING:
            return
        ended_at = time.time()
        result = self._finish_capture()
        destination = self.raw_root / self.run_id
        os.replace(self.pending, destination)
        sidecar = {
            "schema_version": "galbot_sft_episode_v1",
            "run_id": self.run_id,
            "profile": self.cfg["profile"],
            "instruction": self.cfg["task"]["instruction"],
            "started_at": self.started_at,
            "ended_at": ended_at,
            "episode_success": 1,
            "end_reason": "success_key",
            "raw_dir": str(destination),
            "capture": result,
        }
        episode_root = Path(self.cfg["dataset"]["episode_root"])
        atomic_json(episode_root / f"{self.run_id}.json", sidecar)
        self.completed += 1
        LOG.info("第 %d 条数据采集完成并已保存：%s", self.completed, destination)
        self.machine.transition(State.WAITING_NEXT)
        if self.completed >= int(self.cfg["dataset"]["num_episodes"]):
            self.reset_and_stop()
        else:
            LOG.info(
                "已保存 %d/%d 条数据，准备张开夹爪并复位",
                self.completed, self.cfg["dataset"]["num_episodes"],
            )

    def discard_episode(self) -> None:
        if self.machine.state != State.RECORDING:
            return
        self._finish_capture()
        discarded = self.pending
        shutil.rmtree(discarded)
        LOG.info("当前数据已丢弃：%s", discarded)
        self.machine.transition(State.WAITING_NEXT)
        LOG.info("准备张开夹爪并复位，然后重新采集第 %d 条数据", self.completed + 1)

    def run(self) -> int:
        self.initialize()
        LOG.info(
            "纯人工采集已启动；配置=%s，任务=%s；右方向键保存，左方向键丢弃重采",
            self.cfg["profile"], self.cfg["task"]["instruction"],
        )
        decoder = TerminalKeyDecoder()
        with cbreak_stdin():
            while self.machine.state not in (State.ERROR, State.STOPPED):
                if self.machine.state == State.RECORDING and not self.capture.status()["ready"]:
                    self.machine.transition(State.ERROR)
                    raise RuntimeError(f"采集话题异常：{self.capture.status()}")
                readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not readable:
                    continue
                chunk = os.read(sys.stdin.fileno(), 64).decode(errors="ignore")
                for key in decoder.feed(chunk):
                    if key == "RIGHT" and self.machine.state == State.RECORDING:
                        self.save_episode()
                        if self.machine.state == State.WAITING_NEXT:
                            self.reset_and_start()
                    elif key == "LEFT" and self.machine.state == State.RECORDING:
                        self.discard_episode()
                        if self.machine.state == State.WAITING_NEXT:
                            self.reset_and_start()
                    elif key == "q" and self.machine.state in (State.READY, State.WAITING_NEXT):
                        self.machine.transition(State.STOPPED)
                    else:
                        LOG.warning("当前阶段 %s 忽略按键 %r", self.machine.state.value, key)
        return 0

    def close(self) -> None:
        try:
            self.capture.close()
        finally:
            self.resetter.close()


def preflight(cfg: dict, seconds: float) -> int:
    bridge = CaptureBridge(cfg["capture"])
    try:
        print(
            "仅执行预检：检查话题健康状态，不复位、不录制，键盘操作无效。",
            flush=True,
        )
        bridge.start()
        deadline = time.monotonic() + 30.0
        while not bridge.status()["ready"] and time.monotonic() < deadline:
            if bridge.status().get("error"):
                raise RuntimeError(bridge.status()["error"])
            time.sleep(0.5)
        if not bridge.status()["ready"]:
            raise RuntimeError(f"采集话题未就绪：{bridge.status()}")
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            status = bridge.status()
            if not status["ready"]:
                raise RuntimeError(f"采集话题健康检查失败：{status}")
            print(json.dumps(status, ensure_ascii=False), flush=True)
            time.sleep(1.0)
        return 0
    finally:
        bridge.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="collect_config.yaml")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.preflight:
        return preflight(cfg, args.seconds)
    collector = Collector(cfg)
    try:
        return collector.run()
    except KeyboardInterrupt:
        LOG.warning("采集被中断；当前未完成数据保留在 .in_progress 目录")
        return 130
    except Exception:
        LOG.exception("采集器发生错误并停止")
        return 1
    finally:
        collector.close()


if __name__ == "__main__":
    raise SystemExit(main())
