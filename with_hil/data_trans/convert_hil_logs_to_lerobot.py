#!/usr/bin/env python3
"""Convert G1 HIL protobuf archives to a LeRobot v3 dataset.

The converter consumes the pair written by ``infer_policy``:

* ``logs/hil_raw/<run_id>/``: timestamped protobuf streams, policy JSONL, events.
* ``logs/hil_episodes/<run_id>.json``: final success/failure and intervention metadata.

It never imports the Galbot SDK or OpenCV. Protobuf classes are built from each
episode's ``schema.pb`` and JPEG frames are encoded by GStreamer or ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


RECORD_HEADER = struct.Struct("<QQI")
CAMERA_STREAMS = ("head_left", "head_right", "left_arm", "right_arm")
NORMAL_END_REASONS = {"success_key", "failure_key"}
DEFAULT_LOGS_ROOT = Path(__file__).resolve().parents[1] / "infer_policy" / "logs"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "lerobot_g1_hil"

MODEL_TO_WBC = {
    # Model/checkpoint feature names omit ``joint1`` while WBCInfo uses it.
    "right_arm_gripper": "right_gripper_joint1",
    "left_arm_gripper": "left_gripper_joint1",
    # Retain aliases used by early capture manifests.
    "right_gripper": "right_gripper_joint1",
    "left_gripper": "left_gripper_joint1",
}


@dataclass(frozen=True)
class RecordRef:
    timestamp: float
    payload_offset: int
    payload_size: int


@dataclass
class EpisodeSource:
    run_id: str
    sidecar_path: Path
    raw_dir: Path
    sidecar: dict
    manifest: dict

    @property
    def instruction(self) -> str:
        return str(self.sidecar.get("instruction") or self.manifest.get("instruction") or "")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def index_pbstream(path: Path) -> List[RecordRef]:
    """Index record receive timestamps without retaining large payloads."""
    refs: List[RecordRef] = []
    with path.open("rb") as stream:
        while True:
            header = stream.read(RECORD_HEADER.size)
            if not header:
                break
            if len(header) != RECORD_HEADER.size:
                raise ValueError(f"{path}: truncated record header")
            wall_ns, _mono_ns, size = RECORD_HEADER.unpack(header)
            payload_offset = stream.tell()
            stream.seek(size, os.SEEK_CUR)
            if stream.tell() > path.stat().st_size:
                raise ValueError(f"{path}: truncated protobuf payload")
            refs.append(RecordRef(wall_ns / 1e9, payload_offset, size))
    if not refs:
        raise ValueError(f"{path}: no records")
    return refs


def read_payload(path: Path, ref: RecordRef, stream=None) -> bytes:
    owns_stream = stream is None
    stream = stream or path.open("rb")
    try:
        stream.seek(ref.payload_offset)
        payload = stream.read(ref.payload_size)
        if len(payload) != ref.payload_size:
            raise ValueError(f"{path}: truncated protobuf payload at {ref.payload_offset}")
        return payload
    finally:
        if owns_stream:
            stream.close()


def build_message_classes(schema_path: Path, type_names: Iterable[str]) -> Dict[str, Any]:
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.ParseFromString(schema_path.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    pending = list(descriptor_set.file)
    while pending:
        previous = len(pending)
        for file_descriptor in pending[:]:
            try:
                pool.Add(file_descriptor)
                pending.remove(file_descriptor)
            except Exception:
                pass
        if len(pending) == previous:
            raise ValueError(
                f"{schema_path}: unresolved protobuf dependencies: "
                f"{[item.name for item in pending]}"
            )
    factory = message_factory.MessageFactory(pool)
    classes = {}
    for type_name in type_names:
        descriptor = pool.FindMessageTypeByName(type_name)
        if hasattr(factory, "GetPrototype"):
            classes[type_name] = factory.GetPrototype(descriptor)
        else:  # protobuf 5+
            classes[type_name] = message_factory.GetMessageClass(descriptor)
    return classes


def decode_message(message_class, payload: bytes):
    message = message_class()
    message.ParseFromString(payload)
    return message


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if source_times.size == 0:
        raise ValueError("cannot align an empty timestamp sequence")
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, source_times.size - 1)
    left = np.clip(right - 1, 0, source_times.size - 1)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(source_times[right] - target_times)
    indices = np.where(choose_left, left, right).astype(np.int64)
    skew = np.abs(source_times[indices] - target_times)
    return indices, skew


def discover_episodes(logs_root: Path, include_unlabeled: bool = False) -> List[EpisodeSource]:
    sidecar_dir = logs_root / "hil_episodes"
    raw_root = logs_root / "hil_raw"
    episodes: List[EpisodeSource] = []
    for sidecar_path in sorted(sidecar_dir.glob("*.json")):
        try:
            sidecar = read_json(sidecar_path)
        except (OSError, ValueError) as exc:
            logging.warning("skip invalid sidecar %s: %s", sidecar_path, exc)
            continue
        run_id = str(sidecar.get("run_id") or sidecar_path.stem)
        if not include_unlabeled and sidecar.get("end_reason") not in NORMAL_END_REASONS:
            continue
        raw_dir = raw_root / run_id
        captured = ((sidecar.get("capture") or {}).get("directory"))
        if not raw_dir.is_dir() and captured:
            candidate = Path(captured)
            if candidate.is_dir():
                raw_dir = candidate
        manifest_path = raw_dir / "manifest.json"
        if not manifest_path.is_file():
            logging.warning("skip %s: missing %s", run_id, manifest_path)
            continue
        manifest = read_json(manifest_path)
        if manifest.get("error") or not manifest.get("finalized", False):
            logging.warning("skip %s: incomplete capture error=%r", run_id, manifest.get("error"))
            continue
        episodes.append(EpisodeSource(run_id, sidecar_path, raw_dir, sidecar, manifest))
    return episodes


def profile_cameras(manifest: dict, camera_mode: str) -> List[str]:
    if camera_mode == "all":
        return list(CAMERA_STREAMS)
    if camera_mode != "profile":
        requested = [item.strip() for item in camera_mode.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(CAMERA_STREAMS))
        if unknown:
            raise ValueError(f"unknown cameras: {unknown}; available={CAMERA_STREAMS}")
        return requested
    features = manifest.get("observation_features") or {}
    cameras = [
        key.rsplit(".", 1)[-1]
        for key, value in features.items()
        if (value or {}).get("dtype") == "image"
    ]
    return cameras or ["head_right", "left_arm", "right_arm"]


def model_names(manifest: dict, action_dim_mode: str) -> Tuple[List[str], List[str]]:
    features = manifest.get("observation_features") or {}
    state_names = list(((features.get("observation.state") or {}).get("names")) or [])
    if not state_names:
        state_names = list(((manifest.get("robot_config") or {}).get("state_feature_names")) or [])
    if len(state_names) != 23:
        raise ValueError(f"expected 23 state names, got {len(state_names)}")
    if action_dim_mode == "profile":
        action_dim = int((manifest.get("robot_config") or {}).get("vla_action_dim", 0) or 0)
    else:
        action_dim = int(action_dim_mode)
    if action_dim not in (16, 23):
        raise ValueError(f"action dimension must be 16 or 23, got {action_dim}")
    return state_names, state_names[:action_dim]


def raw_name(model_name: str) -> str:
    return MODEL_TO_WBC.get(model_name, model_name)


def clip_model_gripper_actions(
    actions: np.ndarray,
    action_names: Sequence[str],
) -> np.ndarray:
    """Keep WBC gripper commands in model units and enforce their 0..100 range."""
    gripper_indices = [
        index
        for index, name in enumerate(action_names)
        if raw_name(name) in ("right_gripper_joint1", "left_gripper_joint1")
    ]
    if not gripper_indices:
        return actions
    clipped = np.asarray(actions, dtype=np.float32).copy()
    clipped[:, gripper_indices] = np.clip(clipped[:, gripper_indices], 0.0, 100.0)
    return clipped


def sustained_motion_bounds(
    actions: np.ndarray,
    states: np.ndarray,
    action_names: Sequence[str],
    fps: int,
    action_epsilon: float,
    state_epsilon: float,
    gripper_epsilon: float,
    window_s: float,
    minimum_active_frames: int,
    padding_s: float,
) -> Tuple[int, int]:
    """Return a conservative [start, end) span containing sustained edge motion.

    Only leading and trailing inactivity are removed. Internal pauses remain in
    place so temporally separated observations/actions are never spliced.
    """
    frame_count = len(actions)
    if frame_count < 2:
        return 0, frame_count
    arm_indices = np.asarray(
        [index for index, name in enumerate(action_names) if "gripper" not in name.lower()],
        dtype=np.int64,
    )
    gripper_indices = np.asarray(
        [index for index, name in enumerate(action_names) if "gripper" in name.lower()],
        dtype=np.int64,
    )
    action_delta = np.abs(np.diff(actions, axis=0))
    state_delta = np.abs(np.diff(states[:, : actions.shape[1]], axis=0))
    active = np.zeros(frame_count - 1, dtype=bool)
    if arm_indices.size:
        active |= np.max(action_delta[:, arm_indices], axis=1) > action_epsilon
        active |= np.max(state_delta[:, arm_indices], axis=1) > state_epsilon
    if gripper_indices.size:
        active |= np.max(action_delta[:, gripper_indices], axis=1) > gripper_epsilon

    window_frames = max(1, int(round(window_s * fps)))
    minimum_active_frames = max(1, min(int(minimum_active_frames), window_frames))
    score = np.convolve(
        active.astype(np.int32), np.ones(window_frames, dtype=np.int32), mode="same"
    )
    sustained = np.flatnonzero(score >= minimum_active_frames)
    if not sustained.size:
        logging.warning("no sustained motion detected; retaining all %d frames", frame_count)
        return 0, frame_count
    padding_frames = max(0, int(round(padding_s * fps)))
    start = max(0, int(sustained[0]) + 1 - padding_frames)
    end = min(frame_count, int(sustained[-1]) + 2 + padding_frames)
    return start, end


def load_wbc_arrays(
    path: Path,
    refs: Sequence[RecordRef],
    message_class,
    state_names: Sequence[str],
    action_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    states = np.empty((len(refs), len(state_names)), dtype=np.float32)
    actions = np.empty((len(refs), len(action_names)), dtype=np.float32)
    with path.open("rb") as stream:
        for row, ref in enumerate(refs):
            message = decode_message(message_class, read_payload(path, ref, stream))
            joint = message.joint_info
            names = list(joint.names)
            state_values = list(joint.state_pos)
            # WBCInfo reports gripper command_pos in controller/model units
            # (0..100).  gripper_action_scale is applied only to the final
            # JointCommand sent to the SDK; never apply that scale here.
            command_values = list(joint.command_pos)
            if len(names) != len(state_values) or len(names) != len(command_values):
                raise ValueError(f"{path}: inconsistent WBC joint arrays at record {row}")
            lookup = {name: index for index, name in enumerate(names)}
            try:
                states[row] = [state_values[lookup[raw_name(name)]] for name in state_names]
                actions[row] = [command_values[lookup[raw_name(name)]] for name in action_names]
            except KeyError as exc:
                raise ValueError(f"{path}: missing WBC joint {exc.args[0]!r}") from exc
    return states, actions


def load_policy_actions(path: Path, action_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    samples = []
    for row in read_jsonl(path):
        for item in row.get("actions") or []:
            action = item.get("action")
            timestamp = item.get("timestamp")
            if timestamp is None or not isinstance(action, list) or len(action) != action_dim:
                continue
            samples.append((float(timestamp), np.asarray(action, dtype=np.float32)))
    if not samples:
        return np.empty(0, dtype=np.float64), np.empty((0, action_dim), dtype=np.float32)
    # Replans can predict the same future timestamp. Keep the last logged value.
    by_time = {timestamp: action for timestamp, action in samples}
    ordered = sorted(by_time.items())
    return (
        np.asarray([item[0] for item in ordered], dtype=np.float64),
        np.stack([item[1] for item in ordered]).astype(np.float32),
    )


def intervention_time(source: EpisodeSource) -> Optional[float]:
    value = source.sidecar.get("intervention_data_start_at")
    if value is not None:
        return float(value)
    value = source.sidecar.get("takeover_at")
    if value is not None:
        return float(value)
    for event in read_jsonl(source.raw_dir / "events.jsonl"):
        if event.get("event") == "intervention_start":
            return float(event["wall_time"])
    return None


def build_alignment(
    source: EpisodeSource,
    cameras: Sequence[str],
    fps: int,
    max_skew_s: float,
) -> Tuple[np.ndarray, Dict[str, List[RecordRef]], Dict[str, np.ndarray]]:
    stream_names = ["wbc_info"] + list(cameras)
    refs = {
        name: index_pbstream(source.raw_dir / f"{name}.pbstream")
        for name in stream_names
    }
    started = float(source.sidecar.get("started_at") or source.manifest.get("started_at"))
    ended = float(source.sidecar.get("ended_at") or source.manifest.get("ended_at"))
    t0 = max([started] + [items[0].timestamp for items in refs.values()])
    t1 = min([ended] + [items[-1].timestamp for items in refs.values()])
    if t1 <= t0:
        raise ValueError(f"{source.run_id}: streams do not overlap ({t0}..{t1})")
    grid = t0 + np.arange(int(math.floor((t1 - t0) * fps)) + 1) / float(fps)
    aligned: Dict[str, np.ndarray] = {}
    valid = np.ones(grid.size, dtype=bool)
    for name, items in refs.items():
        times = np.asarray([item.timestamp for item in items], dtype=np.float64)
        indices, skew = nearest_indices(times, grid)
        aligned[name] = indices
        valid &= skew <= max_skew_s
    if not valid.any():
        raise ValueError(f"{source.run_id}: no frames satisfy max_skew_s={max_skew_s}")
    dropped = int((~valid).sum())
    if dropped:
        logging.warning("%s: dropping %d/%d frames with excessive skew", source.run_id, dropped, grid.size)
    grid = grid[valid]
    aligned = {name: indices[valid] for name, indices in aligned.items()}
    return grid, refs, aligned


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write LeRobot parquet files. "
            "Install data_trans/requirements.txt in a conversion environment."
        ) from exc
    return pa, pq


def fixed_list_array(pa, values: np.ndarray):
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), values.shape[1]
    )


def write_frame_parquet(path: Path, frame: dict) -> None:
    pa, pq = require_pyarrow()
    arrays = {}
    for key, value in frame.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            arrays[key] = fixed_list_array(pa, value)
        elif isinstance(value, np.ndarray):
            arrays[key] = pa.array(value.tolist())
        else:
            arrays[key] = pa.array(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.table(arrays), tmp, compression="zstd")
    os.replace(tmp, path)


def write_rows_parquet(path: Path, rows: Sequence[dict]) -> None:
    pa, pq = require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(list(rows)), tmp, compression="zstd")
    os.replace(tmp, path)


def encode_video(
    source_path: Path,
    refs: Sequence[RecordRef],
    selected_indices: Sequence[int],
    image_class,
    output_path: Path,
    fps: int,
    width: int,
    height: int,
    ffmpeg: str,
    video_backend: str = "auto",
    gst_launch: str = "gst-launch-1.0",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.stem + ".tmp.mp4")
    ffmpeg_command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-hwaccel", "none",
        "-f", "image2pipe", "-framerate", str(fps), "-vcodec", "mjpeg", "-i", "pipe:0",
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        ),
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-g", str(fps), str(tmp),
    ]
    gst_command = [
        gst_launch, "-q", "fdsrc", "fd=0", "!",
        f"image/jpeg,framerate={fps}/1", "!", "jpegparse", "!", "jpegdec", "!",
        "videoconvert", "!", "videoscale", "add-borders=true", "!",
        f"video/x-raw,width={width},height={height},framerate={fps}/1,pixel-aspect-ratio=1/1",
        "!", "x264enc", "speed-preset=veryfast", "tune=zerolatency",
        f"key-int-max={fps}", "!", "h264parse", "!", "mp4mux", "!",
        "filesink", f"location={tmp}",
    ]
    commands = {
        "ffmpeg": [ffmpeg_command],
        "gstreamer": [gst_command],
        "auto": [gst_command, ffmpeg_command],
    }[video_backend]
    errors = []
    for command in commands:
        tmp.unlink(missing_ok=True)
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            errors.append(f"{command[0]}: {exc}")
            continue
        write_error = None
        try:
            with source_path.open("rb") as stream:
                for index in selected_indices:
                    ref = refs[int(index)]
                    message = decode_message(image_class, read_payload(source_path, ref, stream))
                    process.stdin.write(bytes(message.data))
            process.stdin.close()
        except Exception as exc:
            write_error = exc
            if process.stdin:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        code = process.wait()
        if code == 0 and write_error is None and tmp.is_file():
            os.replace(tmp, output_path)
            return
        backend = Path(command[0]).name
        errors.append(f"{backend}: {write_error or ''} {stderr.strip()}".strip())
        tmp.unlink(missing_ok=True)
    raise RuntimeError(f"video encoding failed for {source_path}: {'; '.join(errors)}")


def numeric_stats(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def merge_stats(episode_stats: Sequence[dict], feature: str) -> dict:
    parts = [item[feature] for item in episode_stats]
    counts = np.asarray([part["count"][0] for part in parts], dtype=np.float64)
    total = counts.sum()
    means = [np.asarray(part["mean"], dtype=np.float64) for part in parts]
    variances = [np.square(np.asarray(part["std"], dtype=np.float64)) for part in parts]
    mean = sum(count * item for count, item in zip(counts, means)) / total
    second = sum(
        count * (variance + np.square(item))
        for count, variance, item in zip(counts, variances, means)
    ) / total
    return {
        "min": np.min([part["min"] for part in parts], axis=0).tolist(),
        "max": np.max([part["max"] for part in parts], axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(second - np.square(mean), 0)).tolist(),
        "count": [int(total)],
    }


def feature_metadata(
    state_names: Sequence[str],
    action_names: Sequence[str],
    policy_action_dim: int,
    cameras: Sequence[str],
    fps: int,
    width: int,
    height: int,
) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": [len(state_names)], "names": list(state_names)},
        "action": {"dtype": "float32", "shape": [len(action_names)], "names": list(action_names)},
        "complementary_info.policy_action": {
            "dtype": "float32", "shape": [policy_action_dim], "names": list(action_names)
        },
        "complementary_info.policy_action_valid": {"dtype": "bool", "shape": [1], "names": None},
        "complementary_info.is_intervention": {"dtype": "bool", "shape": [1], "names": None},
        "complementary_info.state": {"dtype": "string", "shape": [1], "names": None},
        "complementary_info.collector_policy_id": {"dtype": "string", "shape": [1], "names": None},
        "episode_success": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features


def convert_episode(source: EpisodeSource, episode_index: int, global_start: int, args) -> dict:
    cameras = profile_cameras(source.manifest, args.cameras)
    state_names, action_names = model_names(source.manifest, args.action_dim)
    grid, refs, aligned = build_alignment(source, cameras, args.fps, args.max_skew_s)

    # Policy stops at takeover_at, but the configurable preparation window is
    # neither policy control nor a human demonstration. Keep it in the raw
    # archive for auditability and omit it from the training dataset.
    takeover_at = source.sidecar.get("takeover_at")
    intervention_start = intervention_time(source)
    omitted_pause_s = 0.0
    if takeover_at is not None and intervention_start is not None:
        pause_start = float(takeover_at)
        pause_end = float(intervention_start)
        if pause_end > pause_start:
            keep = (grid < pause_start) | (grid >= pause_end)
            omitted_pause_s = pause_end - pause_start
            grid = grid[keep]
            aligned = {name: indices[keep] for name, indices in aligned.items()}
            if not grid.size:
                raise ValueError(f"{source.run_id}: no aligned frames remain after takeover grace window")

    # A plain supervised policy must not imitate the autonomous prefix that led
    # to a human takeover.  For SFT materialization, keep successful autonomous
    # episodes intact, but start intervened episodes at the first human frame.
    # Filtering the alignment here also ensures that every camera is cut at the
    # exact same frame as state/action data and that no action chunk can cross
    # the AUTO -> INTERVENTION boundary.
    takeover = intervention_start
    is_intervention = np.zeros(grid.size, dtype=bool) if takeover is None else grid >= takeover
    source_frame_start = 0
    if args.intervention_sft_only and is_intervention.any():
        source_frame_start = int(np.flatnonzero(is_intervention)[0])
        keep = is_intervention
        grid = grid[keep]
        aligned = {name: indices[keep] for name, indices in aligned.items()}
        is_intervention = is_intervention[keep]
        if not grid.size:
            raise ValueError(f"{source.run_id}: no intervention frames remain for SFT")

    type_names = [source.manifest["topics"]["wbc_info"]["type"]]
    type_names += [source.manifest["topics"][name]["type"] for name in cameras]
    classes = build_message_classes(source.raw_dir / "schema.pb", type_names)
    wbc_type = source.manifest["topics"]["wbc_info"]["type"]
    all_states, all_actions = load_wbc_arrays(
        source.raw_dir / "wbc_info.pbstream", refs["wbc_info"], classes[wbc_type],
        state_names, action_names,
    )
    all_actions = clip_model_gripper_actions(all_actions, action_names)
    states = all_states[aligned["wbc_info"]]
    actions = all_actions[aligned["wbc_info"]]

    trim_start = 0
    trim_end = len(grid)
    if args.trim_static_edges:
        trim_start, trim_end = sustained_motion_bounds(
            actions,
            states,
            action_names,
            args.fps,
            args.trim_action_epsilon,
            args.trim_state_epsilon,
            args.trim_gripper_epsilon,
            args.trim_window_s,
            args.trim_min_active_frames,
            args.trim_padding_s,
        )
        original_segment_frames = len(grid)
        grid = grid[trim_start:trim_end]
        aligned = {name: indices[trim_start:trim_end] for name, indices in aligned.items()}
        states = states[trim_start:trim_end]
        actions = actions[trim_start:trim_end]
        is_intervention = is_intervention[trim_start:trim_end]
        source_frame_start += trim_start
        logging.info(
            "trim run_id=%s prefix=%d suffix=%d kept=%d",
            source.run_id,
            trim_start,
            original_segment_frames - trim_end,
            len(grid),
        )

    policy_times, policy_values = load_policy_actions(source.raw_dir / "policy.jsonl", len(action_names))
    policy_action = np.zeros_like(actions)
    policy_valid = np.zeros(grid.size, dtype=bool)
    if policy_times.size:
        policy_indices, policy_skew = nearest_indices(policy_times, grid)
        policy_valid = policy_skew <= args.policy_max_skew_s
        policy_action[policy_valid] = policy_values[policy_indices[policy_valid]]

    collector_policy = str(source.manifest.get("collector_policy_id") or source.manifest.get("checkpoint") or "policy")
    collector_ids = np.asarray(
        ["human_teleop" if value else collector_policy for value in is_intervention], dtype=object
    )
    states_text = np.asarray(
        ["INTERVENTION" if value else "AUTO" for value in is_intervention], dtype=object
    )
    success = int(source.sidecar.get("episode_success", 0) or 0)
    frame_count = grid.size
    output_grid = grid.copy()
    if omitted_pause_s and takeover is not None:
        output_grid[grid >= takeover] -= omitted_pause_s
    relative_time = (output_grid - output_grid[0]).astype(np.float32)
    frame = {
        "observation.state": states,
        "action": actions,
        "complementary_info.policy_action": policy_action,
        "complementary_info.policy_action_valid": policy_valid,
        "complementary_info.is_intervention": is_intervention,
        "complementary_info.state": states_text,
        "complementary_info.collector_policy_id": collector_ids,
        "episode_success": np.full(frame_count, success, dtype=np.int64),
        "timestamp": relative_time,
        "frame_index": np.arange(frame_count, dtype=np.int64),
        "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
        "index": np.arange(global_start, global_start + frame_count, dtype=np.int64),
        "task_index": np.full(frame_count, args.task_index, dtype=np.int64),
        "subtask_index": np.zeros(frame_count, dtype=np.int64),
    }

    data_path = args.output_dir / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
    write_frame_parquet(data_path, frame)
    for camera in cameras:
        stream_path = source.raw_dir / f"{camera}.pbstream"
        output_path = (
            args.output_dir / "videos" / f"observation.images.{camera}" /
            "chunk-000" / f"file-{episode_index:03d}.mp4"
        )
        image_type = source.manifest["topics"][camera]["type"]
        encode_video(
            stream_path, refs[camera], aligned[camera], classes[image_type], output_path,
            args.fps, args.width, args.height, args.ffmpeg,
            args.video_backend, args.gst_launch,
        )

    stats = {
        "observation.state": numeric_stats(states),
        "action": numeric_stats(actions),
        "complementary_info.policy_action": numeric_stats(policy_action),
    }
    return {
        "run_id": source.run_id,
        "episode_index": episode_index,
        "frames": frame_count,
        "instruction": source.instruction,
        "episode_success": success,
        "end_reason": source.sidecar.get("end_reason"),
        "intervened": bool(is_intervention.any()),
        "takeover_frame": int(np.argmax(is_intervention)) if is_intervention.any() else None,
        "source_frame_start": source_frame_start,
        "trimmed_prefix_frames": trim_start,
        "trimmed_suffix_frames": original_segment_frames - trim_end if args.trim_static_edges else 0,
        "sft_segment": "intervention" if is_intervention.all() else "autonomous",
        "cameras": list(cameras),
        "state_names": list(state_names),
        "action_names": list(action_names),
        "stats": stats,
        "duration_s": float(relative_time[-1]) if frame_count else 0.0,
    }


def finalize_dataset(output_dir: Path, progress: dict, args) -> None:
    episodes = list(progress.get("episodes") or [])
    if not episodes:
        return
    task_names = list(progress.get("tasks") or [])
    first = episodes[0]
    cameras = first["cameras"]
    state_names = first["state_names"]
    action_names = first["action_names"]
    for episode in episodes:
        if episode["cameras"] != cameras or episode["state_names"] != state_names or episode["action_names"] != action_names:
            raise ValueError("mixed profile schemas cannot share one LeRobot dataset")

    total_frames = sum(int(item["frames"]) for item in episodes)
    features = feature_metadata(
        state_names, action_names, len(action_names), cameras,
        args.fps, args.width, args.height,
    )
    info = {
        "codebase_version": "v3.0",
        "robot_type": "galbot_g1",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(task_names),
        "total_subtasks": 1,
        "total_videos": len(episodes) * len(cameras),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/file-{episode_file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_file_index:03d}.mp4",
        "features": features,
    }
    write_json(output_dir / "meta" / "info.json", info)
    stats = {
        feature: merge_stats([item["stats"] for item in episodes], feature)
        for feature in ("observation.state", "action", "complementary_info.policy_action")
    }
    write_json(output_dir / "meta" / "stats.json", stats)
    write_rows_parquet(
        output_dir / "meta" / "tasks.parquet",
        [{"task_index": index, "task": task} for index, task in enumerate(task_names)],
    )
    write_rows_parquet(
        output_dir / "meta" / "subtasks.parquet",
        [{"subtask_index": 0, "subtask": "complete task"}],
    )
    cursor = 0
    episode_rows = []
    for item in episodes:
        length = int(item["frames"])
        row = {
            "episode_index": int(item["episode_index"]),
            "tasks": [item["instruction"]],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": int(item["episode_index"]),
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "episode_success": int(item["episode_success"]),
            "intervened": bool(item["intervened"]),
            "source_run_id": item["run_id"],
        }
        for camera in cameras:
            key = f"videos/observation.images.{camera}"
            row[f"{key}/chunk_index"] = 0
            row[f"{key}/file_index"] = int(item["episode_index"])
            row[f"{key}/from_timestamp"] = 0.0
            row[f"{key}/to_timestamp"] = length / float(args.fps)
        episode_rows.append(row)
        cursor += length
    write_rows_parquet(
        output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        episode_rows,
    )
    write_json(
        output_dir / "conversion_report.json",
        {
            "source_logs": str(args.logs_root),
            "output_dir": str(output_dir),
            "fps": args.fps,
            "episodes": [
                {key: value for key, value in item.items() if key != "stats"}
                for item in episodes
            ],
            "failed": progress.get("failed") or [],
        },
    )


def inspect_sources(episodes: Sequence[EpisodeSource], args) -> int:
    rows = []
    for source in episodes:
        try:
            cameras = profile_cameras(source.manifest, args.cameras)
            state_names, action_names = model_names(source.manifest, args.action_dim)
            grid, _refs, _aligned = build_alignment(source, cameras, args.fps, args.max_skew_s)
            rows.append({
                "run_id": source.run_id,
                "success": source.sidecar.get("episode_success"),
                "intervened": source.sidecar.get("intervened"),
                "frames": int(grid.size),
                "duration_s": round(float(grid[-1] - grid[0]), 3),
                "cameras": cameras,
                "state_dim": len(state_names),
                "action_dim": len(action_names),
                "instruction": source.instruction,
            })
        except Exception as exc:
            rows.append({"run_id": source.run_id, "error": str(exc)})
    print(json.dumps({"logs_root": str(args.logs_root), "episodes": rows}, ensure_ascii=False, indent=2))
    return 1 if any("error" in item for item in rows) else 0


def validate_common_schema(episodes: Sequence[EpisodeSource], args) -> None:
    """Reject mixed pick/place schemas before creating partial output."""
    signatures: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]], List[str]] = {}
    for source in episodes:
        cameras = tuple(profile_cameras(source.manifest, args.cameras))
        state_names, action_names = model_names(source.manifest, args.action_dim)
        signature = (cameras, tuple(state_names), tuple(action_names))
        signatures.setdefault(signature, []).append(source.run_id)
    if len(signatures) > 1:
        details = [
            {
                "cameras": list(signature[0]),
                "state_dim": len(signature[1]),
                "action_dim": len(signature[2]),
                "run_ids": run_ids,
            }
            for signature, run_ids in signatures.items()
        ]
        raise ValueError(
            "source contains incompatible profile schemas; convert each schema to a "
            f"separate output directory: {json.dumps(details, ensure_ascii=False)}"
        )


def convert(args) -> None:
    episodes = discover_episodes(args.logs_root, args.include_unlabeled)
    if args.source_dataset_filter is not None:
        _, pq = require_pyarrow()
        episode_meta = (
            args.source_dataset_filter / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        if not episode_meta.is_file():
            raise FileNotFoundError(
                f"source dataset filter is missing episode metadata: {episode_meta}"
            )
        allowed = pq.read_table(episode_meta, columns=["source_run_id"])["source_run_id"].to_pylist()
        by_run_id = {source.run_id: source for source in episodes}
        missing = [run_id for run_id in allowed if run_id not in by_run_id]
        if missing:
            raise ValueError(
                f"{len(missing)} filtered run ids are missing from raw logs; first={missing[:5]}"
            )
        episodes = [by_run_id[run_id] for run_id in allowed]
    if args.successful_only:
        episodes = [
            source for source in episodes
            if int(source.sidecar.get("episode_success", 0) or 0) == 1
        ]
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise RuntimeError(f"no eligible episodes found under {args.logs_root}")
    if args.inspect:
        raise SystemExit(inspect_sources(episodes, args))
    validate_common_schema(episodes, args)
    if args.video_backend == "ffmpeg" and shutil.which(args.ffmpeg) is None:
        raise RuntimeError(f"ffmpeg executable not found: {args.ffmpeg}")
    if args.video_backend == "gstreamer" and shutil.which(args.gst_launch) is None:
        raise RuntimeError(f"GStreamer executable not found: {args.gst_launch}")
    if args.video_backend == "auto" and not any(
        shutil.which(item) for item in (args.ffmpeg, args.gst_launch)
    ):
        raise RuntimeError("neither ffmpeg nor GStreamer is available for video encoding")
    require_pyarrow()

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    progress_path = args.output_dir / "progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else {
        "version": 1, "episodes": [], "failed": [], "tasks": []
    }
    done = {item["run_id"] for item in progress["episodes"]}
    total_frames = sum(int(item["frames"]) for item in progress["episodes"])

    for source in episodes:
        if source.run_id in done:
            logging.info("skip completed %s", source.run_id)
            continue
        instruction = source.instruction
        if instruction not in progress["tasks"]:
            progress["tasks"].append(instruction)
        args.task_index = progress["tasks"].index(instruction)
        episode_index = len(progress["episodes"])
        logging.info("convert episode=%d run_id=%s", episode_index, source.run_id)
        try:
            result = convert_episode(source, episode_index, total_frames, args)
            progress["episodes"].append(result)
            total_frames += int(result["frames"])
            write_json(progress_path, progress)
        except Exception as exc:
            logging.exception("failed run_id=%s", source.run_id)
            progress["failed"].append({"run_id": source.run_id, "error": str(exc)})
            write_json(progress_path, progress)
            if args.fail_fast:
                raise
    finalize_dataset(args.output_dir, progress, args)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--cameras", default="profile",
        help="profile (default), all, or comma-separated head_left,head_right,left_arm,right_arm",
    )
    parser.add_argument("--action-dim", choices=("profile", "16", "23"), default="profile")
    parser.add_argument("--max-skew-s", type=float, default=0.10)
    parser.add_argument("--policy-max-skew-s", type=float, default=0.10)
    parser.add_argument("--include-unlabeled", action="store_true")
    parser.add_argument(
        "--source-dataset-filter",
        type=Path,
        help=(
            "only convert source_run_id values present in this LeRobot dataset's "
            "meta/episodes table, preserving that table's order"
        ),
    )
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="only convert episodes whose operator label is episode_success=1",
    )
    parser.add_argument(
        "--intervention-sft-only",
        action="store_true",
        help=(
            "for intervened episodes, discard the autonomous prefix and keep only "
            "frames at or after intervention_data_start_at"
        ),
    )
    parser.add_argument(
        "--trim-static-edges",
        action="store_true",
        help="remove only leading/trailing inactivity while retaining internal pauses",
    )
    parser.add_argument("--trim-action-epsilon", type=float, default=1e-3)
    parser.add_argument("--trim-state-epsilon", type=float, default=5e-4)
    parser.add_argument("--trim-gripper-epsilon", type=float, default=0.5)
    parser.add_argument("--trim-window-s", type=float, default=0.5)
    parser.add_argument("--trim-min-active-frames", type=int, default=3)
    parser.add_argument("--trim-padding-s", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inspect", action="store_true", help="validate and summarize without writing")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--gst-launch", default="gst-launch-1.0")
    parser.add_argument(
        "--video-backend", choices=("auto", "ffmpeg", "gstreamer"), default="auto",
        help="video encoder; auto tries GStreamer then ffmpeg",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    args.logs_root = args.logs_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.source_dataset_filter is not None:
        args.source_dataset_filter = args.source_dataset_filter.expanduser().resolve()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("fps, width and height must be positive")
    if args.max_skew_s <= 0 or args.policy_max_skew_s <= 0:
        parser.error("alignment skew limits must be positive")
    if args.trim_window_s <= 0 or args.trim_padding_s < 0:
        parser.error("trim window must be positive and trim padding must be non-negative")
    if args.trim_min_active_frames <= 0:
        parser.error("trim minimum active frames must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
