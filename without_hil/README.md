# G1 无模型纯人工数据采集

本项目用于 G1 机器人纯人工遥操作示教采集。它不连接 VLA、PolicyServer 或其他外部模型，
只负责订阅机器人原始话题、管理每条 episode、控制条目之间的复位，并将成功数据离线转换为
LeRobot v3 数据集。

训练数据语义：

- `observation.state` 来自 `WBCInfo.joint_info.state_pos`。
- `action` 来自 `WBCInfo.joint_info.command_pos`。
- 四路相机和 WBC 原始消息按实际发布速度完整保存为 protobuf 流。
- `target` 只保存为可选诊断流，不作为训练 action。
- 不生成 HIL 的 intervention、policy action 或 collector policy 字段。

## 1. 真机运行前准备

1. 机器人已进入 `DATA_COLLECTION_MODE`。
2. 四路相机、`wbc_info` 话题正常发布。
3. 银河主从臂或现有遥操作程序已启动，人工可以控制机器人。
4. 急停已释放，机器人周围没有障碍物。
5. 已检查并修改 `collect_config.yaml` 中的任务指令和复位姿态。

> **安全警告：** 正式采集会主动控制机器人完成全身复位和夹爪动作。配置文件中的
> `robot.reset_pose` 只是一套示例值，上机前必须根据当前机器人和任务重新确认。

本程序不会修改 `swallows`、不会切换相机模式、不会启动模型，也不会自动上传数据。

## 2. 快速使用

进入项目目录：

```bash
cd g1_datacollect_myself/without_hil
```

先做只读预检：

```bash
./run_collect.sh --preflight --seconds 10
```

预检只检查相机和 WBC 话题，不录制、不初始化机器人运动接口、不发送复位命令。四路相机和
`wbc_info` 必须持续有数据；机器人空闲时 `target` 没有消息是正常现象。

预检通过后开始正式采集：

```bash
./run_collect.sh
```

启动后的自动流程为：

```text
检查话题
  → 机器人首次复位
  → 环境整理倒计时
  → 自动开始采集第 1 条
  → 操作者按方向键结束当前条
  → 完全张开左右夹爪
  → 机器人复位
  → 环境整理倒计时
  → 自动开始下一条
```

达到 `dataset.num_episodes` 后，程序会再次张开夹爪、执行最终复位，然后自动退出。最终复位
后没有下一条任务，因此不会再等待环境整理倒计时。

## 3. 采集按键

按键必须在运行 `./run_collect.sh` 的终端中输入。

| 按键 | 生效阶段 | 行为 |
|---|---|---|
| 右方向键 | 正在采集 | 保存当前条，张开夹爪、复位、倒计时并开始下一条 |
| 左方向键 | 正在采集 | 丢弃当前条，不增加完成计数；张开夹爪、复位后重采当前编号 |

不需要按 `r` 开始。首次复位和环境倒计时完成后，终端会自动显示：

```text
开始采集第 1/10 条数据，编号：...
```

录制过程中 `q`、`s`、`d` 均无效。按 `Ctrl+C` 会停止程序；尚未完成的片段保留在
`data/raw/.in_progress/`，不会进入训练集。

## 4. 配置说明

正式运行前编辑：

```bash
vim collect_config.yaml
```

### 4.1 任务与采集数量

```yaml
profile: g1_pick_3cam_16d

task:
  instruction: "Left arm pick up ..."

dataset:
  num_episodes: 10
  output_fps: 15
  max_skew_s: 0.10
  environment_reset_s: 10.0
```

- `profile: g1_pick_3cam_16d`：训练集使用 `head_right` 和双腕相机，action 为 16 维。
- `profile: g1_place_4cam_23d`：训练集使用四路相机，action 为 23 维。
- `task.instruction`：写入当前批次所有 episode 的任务指令。
- `num_episodes`：本次运行需要成功保存的条数；左键丢弃的数据不计数。
- `environment_reset_s`：首次复位和条目之间留给操作者整理物体/场景的倒计时秒数；设为
  `0` 可关闭倒计时。

程序重启后，本次完成计数会重新从 0 开始，不会根据目录中已有数据自动续算。例如已经成功
保存 1 条、总目标仍为 10 条时，重启前可把 `num_episodes` 临时改为 `9`。

### 4.2 `output_fps` 与相机频率

如果希望最终 LeRobot 数据集为 15 Hz，修改：

```yaml
dataset:
  output_fps: 15
```

只需要修改这一项即可控制离线转换输出的 FPS。它不会改变相机硬件或 DDS 话题的发布频率，
也不会让原始采集进程丢帧。原始 `.pbstream` 始终按各话题实际发布速度完整保存，转换时再按
15 Hz 时间网格选取最近的一帧。

如果希望相机话题本身只发布 15 Hz，需要修改机器人上游相机服务；本项目不负责修改相机
硬件或发布配置。

### 4.3 `max_skew_s` 是什么

```yaml
dataset:
  max_skew_s: 0.10
```

`max_skew_s` 是离线转换时允许的最大时间对齐误差，不是相机频率。转换器会以
`output_fps` 建立统一时间网格，然后分别寻找时间上最近的相机帧和 WBC 数据：

- `0.10` 表示最近数据与目标时间点最多允许相差 0.10 秒，即 100 ms。
- 任一必需相机或 WBC 的最近数据超过这个误差，转换会拒绝该时间对齐，避免使用过旧数据。
- 数值过大可能接受较旧帧；数值过小则可能因为正常抖动或偶发延迟导致转换失败。

把 `output_fps` 从 30 改成 15 时，通常不需要修改 `max_skew_s: 0.10`。如果话题稳定且希望
更严格的同步质量，可以尝试 `0.05`；修改后先运行 `--inspect` 检查所有 episode 是否满足
对齐要求。

### 4.4 数据路径

```yaml
dataset:
  raw_root: data/raw
  episode_root: data/episodes
  lerobot_root: data/lerobot
```

相对路径均以 `collect_config.yaml` 所在目录为基准解析。不同 profile 默认转换到不同子目录，
不会混合为一个训练集。

### 4.5 机器人复位与夹爪

```yaml
robot:
  reset_pose:
    right_arm: [...]
    right_gripper: [0.0]
    left_arm: [...]
    left_gripper: [0.0]
    leg: [...]
    head: [...]

  reset_timeout_s: 45.0
  reset_retries: 2
  reset_max_speed_rad_s: 0.4
  joint_tolerance_rad: 0.03

  gripper_max_width_m: 0.12
  gripper_feedback_max_position_m: 0.06
  gripper_open_width_m: 0.12
  gripper_open_settle_s: 2.0
  verify_gripper_open_position: true
```

- 手臂、腿和头使用弧度；reset pose 中的夹爪使用训练数据的 `0..100` 标度。
- 每条保存或丢弃后，左右夹爪先张开到 `gripper_open_width_m`，反馈校验通过后才进行全身
  复位。张开或复位失败时程序停止，不会自动开始下一条。
- `gripper_max_width_m=0.12` 是发送给 SDK 的总开口宽度。
- `gripper_feedback_max_position_m=0.06` 是 `get_joint_states()` 返回的单侧夹指最大行程。
  两者含义不同，不应设置成相同值。
- 全闭夹爪常出现非零反馈，因此 `verify_gripper_position` 默认关闭；手臂、腿、头仍严格检查
  到位。完全张开阶段由 `verify_gripper_open_position` 单独严格校验。

## 5. 数据输出

成功保存一条数据后目录结构为：

```text
data/
├── raw/<run_id>/
│   ├── head_left.pbstream
│   ├── head_right.pbstream
│   ├── left_arm.pbstream
│   ├── right_arm.pbstream
│   ├── wbc_info.pbstream
│   ├── target.pbstream        # 该话题有数据时才出现
│   ├── schema.pb
│   └── manifest.json
└── episodes/<run_id>.json
```

原始记录格式为小端 `<QQI`：接收 wall time ns、monotonic time ns、protobuf payload 长度，
后面紧跟原始 protobuf bytes。源消息时间戳和厂商原始单位均保留。

数据只会先写入：

```text
data/raw/.in_progress/<run_id>/
```

写入队列完全排空且 manifest 无错误后，目录才会原子移动到 `data/raw/<run_id>/`。话题失联、
队列溢出、写盘失败、夹爪张开失败或机器人复位失败都会停止会话。

## 6. 转换为 LeRobot v3

转换依赖与真机采集依赖分开。首次使用时创建环境：

```bash
cd g1_datacollect_myself/without_hil
./setup_env.sh
.venv/bin/pip install -r requirements-convert.txt
```

先检查来源数据，不写转换结果：

```bash
.venv/bin/python convert_to_lerobot.py --inspect
```

按 `collect_config.yaml` 的 profile、`output_fps` 和 `max_skew_s` 执行转换：

```bash
.venv/bin/python convert_to_lerobot.py
```

常用覆盖参数：

```bash
# 临时按 15 Hz 转换，不修改 YAML
.venv/bin/python convert_to_lerobot.py --fps 15

# 指定 profile
.venv/bin/python convert_to_lerobot.py --profile g1_place_4cam_23d

# 指定输出目录
.venv/bin/python convert_to_lerobot.py \
  --fps 15 \
  --output-dir data/lerobot/pick_15hz

# 删除目标转换目录并从头转换
.venv/bin/python convert_to_lerobot.py --overwrite
```

转换结果包含 H.264 视频、帧级 Parquet、LeRobot v3 metadata、统计信息、
`conversion_report.json` 和用于断点续转的 `progress.json`。视频后端可通过
`--video-backend auto|gstreamer|ffmpeg` 指定。

## 7. 常见问题

### 修改 `output_fps` 后，原始相机会变成对应频率吗？

不会。它只影响离线 LeRobot 转换。原始 protobuf 仍全速采集。

### 改为 15 Hz 后需要同步修改 `max_skew_s` 吗？

通常不需要。建议先保留 `0.10`，运行 `--inspect`；如果所有话题稳定且希望更严格同步，再
尝试 `0.05`。

### 夹爪已经张开，但为什么以前会显示校验失败？

SDK 的张开命令使用 0.12 m 总开口，状态反馈约 0.06 m 是单侧夹指行程。当前实现已分别使用
`gripper_max_width_m` 和 `gripper_feedback_max_position_m` 换算，避免把正常张开误判为 50%。

### 程序异常退出后 `.in_progress` 中的数据怎么办？

这些目录是未完成片段，不会被转换器当作成功 episode。确认无保留价值后再人工清理；程序
不会自动删除，避免误删故障现场。

## 8. 离线测试

```bash
cd g1_datacollect_myself/without_hil
/usr/bin/python3 -m unittest discover -s tests -v
```

离线测试不连接机器人。代码测试通过不能替代真机验收；修改 reset pose、夹爪参数或 profile
后，应重新执行话题预检和短 episode 验证。
