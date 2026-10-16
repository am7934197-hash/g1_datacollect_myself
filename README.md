# G1 数据采集工具

本仓库包含两套相互独立的 G1 数据采集程序，只保留运行所需的源码、配置、启动脚本和数据
转换工具，不包含日志、原始数据、LeRobot 数据集、视频、模型权重或虚拟环境。

## 目录说明

```text
g1_datacollect_myself/
├── with_hil/
│   ├── infer_policy/   # VLA 自动执行 + 人工接管采集
│   └── data_trans/     # HIL 原始日志转 LeRobot v3
└── without_hil/        # 从第一帧开始纯人工遥操作采集 + LeRobot 转换
```

## 有 HIL 的采集

HIL 版本先由 VLA 控制机器人，操作者触发接管后停止模型动作并继续记录人工遥操作。数据包含
policy action、intervention 状态和 episode 成败标签。

```bash
cd with_hil/infer_policy
```

首次使用前修改 `vla_config.yaml`，至少配置：

- PolicyServer 地址和 checkpoint 路径；
- profile 与任务指令；
- 机器人 reset pose；
- 采集条数与环境整理时间。

先执行只读预检：

```bash
python3 hil_capture.py --preflight --seconds 30
```

再启动服务：

```bash
./run_server.sh
```

启动采集请求：

```bash
curl -sS -X POST 127.0.0.1:8005/execute
```

HIL 的接管键、成功/失败标注、连续采集及转换命令详见
[`with_hil/infer_policy/README.md`](with_hil/infer_policy/README.md) 和
[`with_hil/data_trans/README.md`](with_hil/data_trans/README.md)。

## 无 HIL 的纯人工采集

无 HIL 版本不连接任何模型。机器人首次复位和环境倒计时完成后自动开始录制，整条数据都由
人工遥操作产生。

```bash
cd without_hil
```

上机前修改 `collect_config.yaml` 中的任务指令、采集数量、输出 FPS 和 reset pose。

先执行只读预检：

```bash
./run_collect.sh --preflight --seconds 10
```

开始采集：

```bash
./run_collect.sh
```

- 右方向键：保存当前条；张开夹爪、复位、倒计时后开始下一条。
- 左方向键：丢弃当前条；张开夹爪、复位后重新采集当前编号。

完整配置、15 Hz 输出、时间对齐和 LeRobot 转换说明见
[`without_hil/README.md`](without_hil/README.md)。

## 数据与隐私约定

仓库的 `.gitignore` 会排除以下内容：

- `logs/`、`data/`、`outputs/` 和 `.in_progress/`；
- protobuf 流、MCAP、Parquet、视频和转换报告；
- checkpoint、模型权重、虚拟环境和 Python 缓存。

提交前仍建议运行：

```bash
git status --short
git ls-files
```

确认没有将现场数据、日志或凭据加入提交。

