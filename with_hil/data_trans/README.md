# G1 HIL → LeRobot v3

`convert_hil_logs_to_lerobot.py` converts the raw archives produced by
`hil/infer_policy` into one resumable LeRobot v3 dataset.

## Data mapping

- `observation.state`: WBC measured `state_pos`, reordered to the 23-D model state.
- `action`: WBC `command_pos`, meaning the command actually present in the controller
  stream. Its gripper fields are already the unscaled model values (`0..100`) and
  are written unchanged. `gripper_action_scale` applies only at the final SDK send
  boundary; the SDK's `0..0.12` metre values must never enter this field. The profile
  selects 16-D pick or 23-D place actions.
- `complementary_info.policy_action`: policy prediction aligned by timestamp.
- `complementary_info.is_intervention`: false before takeover and true afterwards.
- `episode_success`: the operator's `s=1` / `f=0` label.
- Videos: profile cameras by default, H.264 at 640×480 and 30 FPS.

Episodes ended by `stop` or `recovery` are skipped by default. Both normally labelled
success and failure episodes are retained.

## Setup

Use a separate conversion environment; the robot inference environment does not ship
with `pyarrow`:

```bash
cd g1_datacollect_myself/with_hil/data_trans
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Video encoding defaults to `--video-backend auto`: it tries GStreamer
(`jpegdec`, `x264enc`, `mp4mux`) first, then falls back to `ffmpeg`. This order
works around the bundled ffmpeg failing to initialize NVIDIA libraries on the G1.

## Validate source data

This does not write output and does not require `pyarrow`:

```bash
python3 convert_hil_logs_to_lerobot.py --inspect
```

## Convert

```bash
.venv/bin/python convert_hil_logs_to_lerobot.py \
  --logs-root ../infer_policy/logs \
  --output-dir ./lerobot_g1_hil
```

The default camera set comes from each episode's profile. For all four recorded
cameras, pass `--cameras all`. To restart conversion from scratch, add `--overwrite`.
Otherwise `progress.json` lets the command resume completed episodes.

Pick (3-camera/16-D action) and place (4-camera/23-D action) captures must use
different output directories. The converter checks this before writing data.

For a quick one-episode conversion:

```bash
.venv/bin/python convert_hil_logs_to_lerobot.py --limit 1 --fail-fast
```
