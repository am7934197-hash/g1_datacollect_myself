# HIL infer_policy（TM01 物理接管 + 持续原始数据采集）

## 实机预检（先做这个）

保持数采模式、VLA 未启动，主臂保持平稳。在机器人终端运行：

```bash
cd g1_datacollect_myself/with_hil/infer_policy
python3 hil_capture.py --preflight --seconds 30
```

该命令只订阅消息，没有 SDK 运动调用，也不加载 VLA。等待出现 `Recording to` 后，
分别按住、松开左右接管键。输出应出现 `key8L`、`key8R` 的 1/3/2 事件。
结束时查看输出 manifest：四路相机、Joy、WBCInfo 的 counts 均应非零，error 应为 null。
target 在无遥操目标时允许为零。缺流/失联会报错，不应绕过检查启动推理。

已根据本机 2026-09-18 的日志确认左右接管键映射；原生订阅链路及 VLA 并发接管仍需实机预检/联调。

## 接入行为

原始采集默认启用。任一接管键收到 1（按下）或 3（持续按住）
即闩锁停止本轮全部模型关节命令，2（松开）只结束物理按住状态，不恢复 VLA。
Joy 的纯 axes 消息不代表松开。保留 i/s/f 和 HTTP 操作。
SDK 发送与接管闩锁使用同一互斥锁；接管确认后不再发送新命令。
这不能撤销已经进入 SDK 的一次调用，也不能证明底层模型/遥操仲裁没有瞬时重叠。

启用采集时，server 启动不自动回初始姿态，/execute 不自动闭合夹爪；起始姿态需由操作者准备。
启动 episode 前需松开两侧接管键，所有必需话题在最近 2 秒内有数据。
episode 开始前启动录制（包括 warmup），结束时排空写入队列；人工阶段持续录制。
采集子进程退出、话题失联、队列溢出或写入失败会禁止后续模型动作。
不自动修改系统 swallows 配置，也不触发系统数据上传。

## 原始数据格式

每轮保存至 `logs/hil_raw/{run_id}/`，不自动清理：

- `joy/target/wbc_info/head_left/head_right/left_arm/right_arm.pbstream`：七路原始 protobuf。
- `schema.pb`：包含依赖的 FileDescriptorSet，可离线恢复消息定义。
- `manifest.json`：话题类型、接收条数、模型/机器人配置、写入异常和结束状态。
- `policy.jsonl`：模型输出、是否接管、是否允许执行、collector policy ID。接管后按配置低频保存 shadow policy action，但绝不下发。
- `events.jsonl`：episode 开始、接管闩锁、policy 运动结束、s/f 结果及最终结束事件。
- `logs/hil_episodes/{run_id}.json`：任务结果、接管时间、原始数据目录及采集结果。

每条 pbstream 记录为小端 `<QQI`（接收 wall_ns、monotonic_ns、payload 长度），随后为
原始 protobuf bytes。`hil_capture.read_records()` 可读取并检查截断记录。
源时间戳、关节名称、目标配置和压缩图像均保留在原始消息里。
数据保持厂商原始单位；特别是原始夹爪值不能直接当米使用。
target 是提交目标，WBCInfo 提供 target/command/state，应共同分析，不能把提交等同于执行确认。
当前产物是用于后续对齐、质检和转换的原始归档，不是已经转换好的 LeRobot/RECAP 训练集；
未进行强制 30 Hz 重采样，也不保证 DDS 上游没有丢帧。

归档现在按 Evo-RL 字段语义保存转换所需来源：LeRobot `action` 应从
`wbc_info.joint_info.command_pos` 重采样，`complementary_info.policy_action` 来自 `policy.jsonl`，
`complementary_info.is_intervention/state/collector_policy_id` 由 `events.jsonl` 的接管闩锁
生成，`episode_success` 来自 s/f 标注。不能把 policy 输出或 `target` 直接冒充已执行动作。
接管期间默认每 1 秒做一次 shadow inference；它只记录“模型当时会怎么做”，
`execution_allowed=false`、`selected_action_count=0`，不会调用机器人动作接口。
该安全路径使用代码默认值 `rtc.enabled=false`；启动校验会拒绝把 shadow inference
与仍可能存在后台请求的 RTC 模式同时启用。

子进程使用系统 Python、本机 Embosa 和随机器人提供的 protobuf 3.20.2；
与 VLA 进程的 protobuf 环境隔离。若该目录缺失，预检会报错，不替换系统依赖。

离线验证：

```bash
python3 -m unittest tests.test_hil_config -v
python3 -m unittest discover -s tests -p test_hil_capture.py -v
```

从 `infer_client/infer_policy` 拷出的 VLA 闭环，加上 Human-in-the-Loop 闩锁。不改 `infer_client`。

人工命令仍由银河主从臂写 `wbcs/target`。本进程负责 policy 下发、一次接管后停发、
episode 成败标注；独立采集子进程持续记录图像、按键、目标和状态。

## 启动

当前单模型配置连接本机 `127.0.0.1:30014`。先保持以下 SSH 隧道运行；
`30017/30018` 是 behavior tree 双模型配置使用的端口，不能用于本服务：

```bash
ssh -p 50017 -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:30014:127.0.0.1:30014 zhangyuqi@112.65.216.193
```

```bash
cd g1_datacollect_myself/with_hil/infer_policy
./run_server.sh
```

不要直接运行 `python server.py`。机器人默认 shell 将 `LD_LIBRARY_PATH` 设为
`/data/galbot/lib`，会让系统 NumPy 加载到不兼容的 BLAS，并报
`undefined symbol: cblas_sgemm`。启动脚本使用项目本地 `.venv`，并只为本进程调整
动态库查找顺序，不修改系统 Python 或全局环境。服务所在终端必须保持运行。

默认 HTTP `0.0.0.0:8005`。`vla_config.yaml` 的 `controls.continuous` 默认启用连续采集，
`dataset.num_episodes` 决定一次采集多少条；一次 `POST /execute` 会保持连接并连续创建
episode，达到数量后自动返回：

```bash
curl -sS -X POST 127.0.0.1:8005/execute
```

每条 episode 按 `s` 或 `f` 后会先完成原始数据和 sidecar 落盘，然后重置 PolicyServer，
确认物理接管键已经松开，再把机器人恢复到 `vla_config.yaml` 的
`robot.reset_pose`。机器人到位并
通过关节反馈校验后，终端倒计时 10 秒供操作者摆回物体，随后自动开始下一条 episode。
复位运动和这 10 秒不会录入前一条或下一条 episode。

## 按键（必须打在 **本进程所在终端**）

| 键 | 作用 |
|----|------|
| `i` | 立即停止 Policy 并一次闩锁。`takeover_grace_s` 准备期仍保存原始数据，但转换训练集时自动剔除 |
| `s` | 当前 episode 记为成功；自动复位、倒计时并开始下一条 |
| `f` | 当前 episode 记为失败；自动复位、倒计时并开始下一条 |
| 右方向键 | **独立**恢复初始姿态。AUTO 时与原 infer 相同；INTERVENTION 期间只打 warning，不发 `set_joint_commands` |

HTTP 与键盘等价：

- `POST /takeover`
- `POST /episode/success`、`POST /episode/failure`
- `GET /hil/status`：除 episode 状态外，`session.phase` 显示运行、机器人复位或环境倒计时
- `POST /stop`：结束当前 episode 和整个连续 `/execute` 会话

接管后必须再按 `s`/`f`（或 `/stop`）才会结束；只按 `i` 不会自动标成功。未接管、policy 自己做完也可以直接按 `s`。

## Sidecar

每条 episode 写：

`logs/hil_episodes/{run_id}.json`

`run_id` 为 `YYYYMMDD_HHMMSS`。字段：

```json
{
  "run_id": "20260916_143000",
  "instruction": "...",
  "started_at": 1.0,
  "ended_at": 2.0,
  "takeover_at": 1.3,
  "takeover_step": 42,
  "intervened": true,
  "episode_success": 1,
  "end_reason": "success_key"
}
```

`end_reason`：`success_key` / `failure_key` / `timeout` / `stop` / `recovery`。

- 跑到 `max_steps` 后 policy 停止运动，但 episode 和采集保持活动，直到操作者标记
  `success` / `failure`；此时应在另一个终端调用标注接口
- `takeover_at` 为空表示整段都是 policy
- 不改 swallows / MCAP；本阶段不产出 LeRobot 列，原始归档见上文

`vla_config.yaml` 只保留日常任务参数。连续采集开关位于 `controls.continuous`，
`dataset.num_episodes` 是本次采集条数，`dataset.reset_time_s` 对应 Evo-RL 的环境整理时间，
复位目标位于 `robot.reset_pose`。
固定相机/schema、SDK 映射、关节限位、插值和日志默认值在 `g1_defaults.py`。
YAML 保留推理调优参数，包括 PolicyServer setup、action chunk、FPS、通信超时、图像传输、
图像预处理、PyTorch 优化和 RTC；
修改后需要重启服务。启用 `rtc.enabled` 时必须同时设置
`controls.shadow_inference: false`，避免接管时仍有异步推理请求在途。
`robot.control` 保留动作滤波、插值、单步限幅和夹爪缩放等执行参数；真机
`joint_limits` 仍固定在 `g1_defaults.py` 中。

切换模型时修改 `profile`、`policy.checkpoint`、`policy.server_address` 和任务指令后重启：

- `g1_pick_3cam_16d`：head_right + 双腕相机，16 维双臂/夹爪动作
- `g1_place_4cam_23d`：双头相机 + 双腕相机，23 维完整动作

checkpoint 文件在本机可读时，启动会按 profile 校验完整 schema；远端路径不可读时会明确
告警，并由 PolicyServer 加载该路径。未知 profile、旧版顶层字段或错误位姿维度会在机器人
运动前直接报错。

临时只跑一条可在请求中覆盖 YAML：

```bash
curl -sS -X POST 127.0.0.1:8005/execute \
  -H 'Content-Type: application/json' \
  -d '{"max_steps":300,"continuous":false}'
```

请求中的 `"max_episodes":2` 可以临时覆盖 YAML，完成两条后让请求自动返回。

## 验收

1. `cd g1_datacollect_myself/with_hil/infer_policy && ./run_server.sh`
2. `POST /execute`，双臂按 policy 动
3. 按一次 `i`：日志 `TAKEOVER START`，policy 停发；松开 `i` 后仍停发；银河主从臂可拖到结束
4. INTERVENTION 期间按右方向键：只 warning，机械臂不被 recovery 抢走
5. 按 `s` 或 `f`：当前 sidecar 写入 `episode_success`，机器人自动回 YAML 位姿
6. 观察终端 10 秒倒计时；结束后新的 run_id 开始推理和采集
7. 从另一个终端 `curl -sS -X POST 127.0.0.1:8005/stop`，连续会话结束，原 `/execute` 请求返回
