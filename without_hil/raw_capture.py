"""Read-only Embosa recorder used by the normal G1 SFT collector.

The native runtime is isolated in system Python so its protobuf 3.x modules do
not collide with the collector/converter environment.  This module never sends
robot commands and never changes camera or swallows configuration.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time
import uuid

PREFIX = "SFT_JSON "
TOPICS = {
    "target": ("singorix/wbcs/target", "galbot.singorix_proto.singorix_target_pb2", "SingoriXTarget"),
    "wbc_info": ("singorix/wbcs/wbc_info", "galbot.singorix_proto.singorix_info_pb2", "WBCInfo"),
    "head_left": ("/front_head_camera/left_color/image_raw", "galbot.sensor_proto.image_pb2", "CompressedImage"),
    "head_right": ("/front_head_camera/right_color/image_raw", "galbot.sensor_proto.image_pb2", "CompressedImage"),
    "left_arm": ("/left_arm_camera/color/image_raw", "galbot.sensor_proto.image_pb2", "CompressedImage"),
    "right_arm": ("/right_arm_camera/color/image_raw", "galbot.sensor_proto.image_pb2", "CompressedImage"),
}
REQUIRED = set(TOPICS) - {"target"}  # Target is legitimately silent while idle.
RECORD_HEADER = struct.Struct("<QQI")  # receive wall ns, monotonic ns, payload bytes


def read_records(path):
    """Iterate an archive; reject truncated records rather than silently dropping."""
    with open(path, "rb") as stream:
        while True:
            header = stream.read(RECORD_HEADER.size)
            if not header:
                return
            if len(header) != RECORD_HEADER.size:
                raise ValueError("truncated record header")
            wall, mono, size = RECORD_HEADER.unpack(header)
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError("truncated protobuf payload")
            yield wall, mono, payload


class RawRecorder:
    def __init__(self, directory, metadata, max_queue=256):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.counts = {key: 0 for key in TOPICS}
        self.error = None
        self.closed = False
        self.lock = threading.Lock()
        self.queue = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def put(self, key, payload, wall, mono):
        with self.lock:
            if self.closed or self.error:
                return
            try:
                self.queue.put_nowait((key, payload, wall, mono))
            except queue.Full:
                self.error = "capture queue overflow; archive incomplete"

    def _write(self):
        streams = {}
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                key, payload, wall, mono = item
                if key not in streams:
                    streams[key] = open(self.directory / (key + ".pbstream"), "wb")
                streams[key].write(RECORD_HEADER.pack(wall, mono, len(payload)))
                streams[key].write(payload)
                self.counts[key] += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            for stream in streams.values():
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                except Exception as exc:
                    self.error = str(exc)

    def close(self):
        with self.lock:
            self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.error = "capture writer did not finish"
        result = dict(self.metadata, counts=self.counts.copy(), error=self.error,
                      finalized=not self.thread.is_alive(), ended_at=time.time(),
                      format="protobuf records: little-endian <QQI + payload; wall_ns, monotonic_ns, byte_length",
                      topics={k: {"topic": v[0], "type": v[1].rsplit(".", 1)[0] + "." + v[2]}
                              for k, v in TOPICS.items()})
        (self.directory / "manifest.json").write_text(json.dumps(result, indent=2))
        return result


class CaptureBridge:
    """Parent-process control. The child uses native serialized subscriptions."""
    def __init__(self, config):
        self.config = config
        self.process = None
        self.latest = {}
        self.last_heartbeat = 0.0
        self.error = None
        self.pending = {}
        self.lock = threading.Lock()
        self.directory = None

    def start(self):
        env = os.environ.copy()
        # Vendor-generated protobuf files predate protobuf 4; isolate this from VLA.
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        self.process = subprocess.Popen(
            [self.config.get("python", "/usr/bin/python3"), "-u", __file__, "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1, env=env)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if not line.startswith(PREFIX):
                    continue
                event = json.loads(line[len(PREFIX):])
                if event["kind"] == "heartbeat":
                    self.latest = event
                    self.last_heartbeat = time.monotonic()
                elif event["kind"] == "fatal":
                    self.error = event["error"]
                elif event.get("id"):
                    with self.lock:
                        slot = self.pending.get(event["id"])
                    if slot:
                        slot[1].update(event)
                        slot[0].set()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.error = self.error or "capture process disconnected"

    def status(self):
        result = dict(self.latest)
        result["error"] = self.error or result.get("error")
        ages = result.get("ages", {})
        heartbeat_age = time.monotonic() - self.last_heartbeat
        timeout = float(self.config.get("topic_timeout_s", 2.0))
        result["ready"] = bool(
            self.process and self.process.poll() is None and not result["error"]
            and heartbeat_age < timeout
            and all(ages.get(k, 999) + heartbeat_age < timeout for k in REQUIRED))
        result["directory"] = self.directory
        return result

    def request(self, command, **kwargs):
        token = uuid.uuid4().hex
        slot = (threading.Event(), {})
        with self.lock:
            self.pending[token] = slot
        try:
            self.process.stdin.write(json.dumps(dict(command=command, id=token, **kwargs)) + "\n")
            self.process.stdin.flush()
            if not slot[0].wait(15):
                self.error = "capture command timeout: " + command
                raise RuntimeError(self.error)
            if slot[1].get("error"):
                raise RuntimeError(slot[1]["error"])
            return slot[1]
        finally:
            with self.lock:
                self.pending.pop(token, None)

    def begin(self, directory, metadata):
        if not self.status()["ready"]:
            raise RuntimeError("SFT capture not ready: " + str(self.status()))
        self.request(
            "begin", directory=str(directory), metadata=metadata,
            max_queue=int(self.config.get("max_queue", 2048)),
            topic_timeout_s=float(self.config.get("topic_timeout_s", 2.0)),
        )
        self.directory = str(directory)

    def end(self):
        if not self.directory:
            return None
        try:
            return self.request("end")
        finally:
            self.directory = None

    def close(self):
        if self.process:
            try:
                self.end()
            finally:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)


def worker():
    """Native runtime lives only here; importing this module is hardware-free."""
    version = "%s.%s" % sys.version_info[:2]
    paths = sorted(Path("/data/galbot/lib").glob("python" + version + ".*"))
    # System protobuf 3.6.1 cannot load the installed generated descriptors.
    # Use the robot's bundled 3.20.2 only inside this isolated worker.
    sys.path[:0] = [str(p) for p in paths] + [
        "/data/galbot/lib/reception/python3/site-packages",
        "/data/galbot/lib/python3/site-packages",
    ]
    import embosa_python as embosa
    from google.protobuf.descriptor_pb2 import FileDescriptorSet

    output_lock = threading.Lock()
    state_lock = threading.RLock()
    state = {"recorder": None, "seen": {}, "counts": {}, "error": None}

    def emit(kind, **fields):
        with output_lock:
            print(PREFIX + json.dumps(dict(kind=kind, **fields)), flush=True)

    classes = {k: getattr(importlib.import_module(v[1]), v[2]) for k, v in TOPICS.items()}
    descriptor_set = FileDescriptorSet()
    seen_descriptors = set()

    def collect_descriptor(d):
        if d.name in seen_descriptors:
            return
        for dependency in d.dependencies:
            collect_descriptor(dependency)
        seen_descriptors.add(d.name)
        d.CopyToProto(descriptor_set.file.add())

    for cls in classes.values():
        collect_descriptor(cls.DESCRIPTOR.file)

    def callback(key, message):
        wall, mono = time.time_ns(), time.monotonic_ns()
        try:
            with state_lock:
                state["seen"][key] = time.monotonic()
                state["counts"][key] = state["counts"].get(key, 0) + 1
                recorder = state["recorder"]
                if recorder:
                    recorder.put(key, message.SerializeToString(), wall, mono)
        except Exception as exc:
            state["error"] = str(exc)

    if embosa.GetState() == embosa.STATE_UNINITIALIZED:
        embosa.EmbosaInit()
    node = embosa.CreateNode("sft_capture_" + str(os.getpid()))
    readers = []
    for key, spec in TOPICS.items():
        qos = embosa.Qos()
        if key in ("head_left", "head_right", "left_arm", "right_arm"):
            qos.intra_core_qos.transport_type = embosa.LARGE_DATA_TRANSPORT
        readers.append(node.CreateSerializationReader(
            classes[key], spec[0], lambda msg, k=key: callback(k, msg), qos))

    def heartbeat():
        while True:
            with state_lock:
                recorder = state["recorder"]
                emit("heartbeat", ages={k: time.monotonic() - v for k, v in state["seen"].items()},
                     counts=dict(state["counts"]),
                     error=state["error"] or (recorder.error if recorder else None))
            time.sleep(0.2)

    threading.Thread(target=heartbeat, daemon=True).start()
    for line in sys.stdin:
        command = json.loads(line)
        try:
            if command["command"] == "begin":
                with state_lock:
                    if state["recorder"]:
                        raise RuntimeError("capture already active")
                    if state["error"]:
                        raise RuntimeError("capture worker has an error")
                    timeout = float(command.get("topic_timeout_s", 2.0))
                    if any(time.monotonic() - state["seen"].get(k, 0) >= timeout for k in REQUIRED):
                        raise RuntimeError("required capture topics missing or stale")
                    recorder = RawRecorder(
                        command["directory"], command["metadata"],
                        max_queue=int(command.get("max_queue", 2048)),
                    )
                    (recorder.directory / "schema.pb").write_bytes(descriptor_set.SerializeToString())
                    state["recorder"] = recorder
                emit("reply", id=command["id"])
            elif command["command"] == "end":
                with state_lock:
                    recorder, state["recorder"] = state["recorder"], None
                result = recorder.close() if recorder else None
                emit("reply", id=command["id"], result=result)
            else:
                raise ValueError("unknown capture command")
        except Exception as exc:
            emit("reply", id=command["id"], error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", default="data/preflight")
    args = parser.parse_args()
    if args.worker:
        try:
            worker()
        except Exception as exc:
            print(PREFIX + json.dumps({"kind": "fatal", "error": str(exc)}), flush=True)
            raise
        return
    if not args.preflight:
        parser.error("use --preflight (read-only; no VLA or motion)")
    bridge = CaptureBridge({})
    try:
        bridge.start()
        deadline = time.monotonic() + 20
        while not bridge.status()["ready"] and time.monotonic() < deadline:
            if bridge.status().get("error"):
                raise RuntimeError(bridge.status()["error"])
            print("Waiting for topics:", bridge.status(), flush=True)
            time.sleep(2)
        directory = Path(args.output).resolve() / uuid.uuid4().hex
        bridge.begin(directory, {"kind": "read_only_preflight", "started_at": time.time()})
        print("Recording to", directory, flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            if not bridge.status()["ready"]:
                raise RuntimeError("capture became unhealthy: " + str(bridge.status()))
            time.sleep(0.2)
        result = bridge.end()
        print(json.dumps(result, indent=2))
        if result["result"]["error"]:
            raise RuntimeError(result["result"]["error"])
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
