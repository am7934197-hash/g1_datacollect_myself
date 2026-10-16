"""
动作滤波与关节插值 — 平滑模型输出并升频到 SDK 控制频率。

用法:
    from motion_processing import create_action_filter, create_interpolator
    flt = create_action_filter(cfg, dim=23)
    filtered = flt(raw_action)

支持:
    - none:            直通，不做滤波
    - lowpass:         一阶指数平滑 (EMA)
    - moving_average:  滑动窗口均值
    - one_euro:        1€ 自适应滤波（速度高时响应快，静止时平滑强）
"""

from __future__ import annotations

import threading
import time
import uuid
import numpy as np
from typing import Optional


# ============================================================================
# 滤波器基类
# ============================================================================

class BaseFilter:
    dim: int

    def __init__(self, dim: int = 23):
        self.dim = dim

    def reset(self):
        """重置内部状态（新一轮推理开始时调用）。"""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


# ============================================================================
# 一阶低通滤波 (EMA)
# ============================================================================

class LowPassFilter(BaseFilter):
    """Exponential Moving Average — 一阶指数平滑。

    y[t] = alpha * x[t] + (1 - alpha) * y[t-1]

    alpha 越小 → 越平滑但响应慢；alpha 越大 → 响应快但平滑弱。
    """

    def __init__(self, dim: int = 23, alpha: float = 0.3):
        super().__init__(dim)
        self.alpha = float(alpha)
        self._state: Optional[np.ndarray] = None

    def reset(self):
        self._state = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = x.astype(np.float32).copy()
            return self._state.copy()
        self._state = self.alpha * x + (1.0 - self.alpha) * self._state
        return self._state.copy()


# ============================================================================
# 滑动均值滤波
# ============================================================================

class MovingAverageFilter(BaseFilter):
    """滑动窗口均值 — 每个关节取最近 window_size 步的平均值。

    window_size 越大 → 越平滑但延迟越高。
    """

    def __init__(self, dim: int = 23, window_size: int = 5):
        super().__init__(dim)
        self.window_size = max(1, int(window_size))
        self._buffer = np.zeros((self.window_size, dim), dtype=np.float32)
        self._idx = 0
        self._count = 0

    def reset(self):
        self._buffer.fill(0)
        self._idx = 0
        self._count = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._buffer[self._idx] = x
        self._idx = (self._idx + 1) % self.window_size
        if self._count < self.window_size:
            self._count += 1
        return self._buffer[:self._count].mean(axis=0).astype(np.float32)


# ============================================================================
# 1€ 自适应滤波
# ============================================================================

class OneEuroFilter(BaseFilter):
    """1€ Filter — 自适应截止频率低通滤波。

    参考文献: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass
    Filter for Noisy Input in Interactive Systems", CHI 2012.

    原理:
        - 用信号变化速度（dx/dt）自适应调整截止频率
        - 静止/慢速时截止频率低 → 强去抖
        - 快速移动时截止频率高 → 低延迟响应

    参数:
        freq:       数据采样率 (Hz)，默认 30
        min_cutoff: 最小截止频率 (Hz)，静止时的平滑强度，越小越平滑，推荐 0.5~2
        beta:       速度增益，控制响应灵敏度，0 表示固定截止频率，推荐 0.005~0.05
        d_cutoff:   速度估计的截止频率 (Hz)，通常 1.0
    """

    def __init__(self, dim: int = 23, freq: float = 30.0,
                 min_cutoff: float = 1.0, beta: float = 0.01,
                 d_cutoff: float = 1.0):
        super().__init__(dim)
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._first_time = True

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._first_time = True

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        """Compute smoothing factor from cutoff frequency and sample rate."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / freq
        return float(te / (te + tau))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)

        if self._first_time:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros(self.dim, dtype=np.float32)
            self._first_time = False
            return x.copy()

        # Estimate derivative (velocity)
        dx = (x - self._x_prev) * self.freq
        alpha_d = self._alpha(self.d_cutoff, self.freq)
        self._dx_prev = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        # Adaptive cutoff based on signal speed
        speed = np.abs(self._dx_prev)
        cutoff = self.min_cutoff + self.beta * speed  # per-joint adaptive cutoff
        alpha = self._alpha(cutoff, self.freq)

        # Apply low-pass with adaptive alpha
        self._x_prev = alpha * x + (1.0 - alpha) * self._x_prev
        return self._x_prev.copy()


# ============================================================================
# 工厂函数
# ============================================================================

def create_action_filter(cfg: dict, dim: int = 23) -> BaseFilter:
    """从配置字典创建动作滤波器。

    cfg 格式 (对应 yaml 中 robot.control.action_filter):
        enabled: true
        type: "lowpass"          # none / lowpass / moving_average / one_euro
        lowpass:
          alpha: 0.3
        moving_average:
          window_size: 5
        one_euro:
          freq: 30.0
          min_cutoff: 1.0
          beta: 0.01
          d_cutoff: 1.0
    """
    enabled = cfg.get("enabled", False)
    filter_type = cfg.get("type", "none")

    if not enabled or filter_type == "none":
        return BaseFilter(dim)

    if filter_type == "lowpass":
        lp_cfg = cfg.get("lowpass", {})
        return LowPassFilter(dim=dim, alpha=lp_cfg.get("alpha", 0.3))

    if filter_type == "moving_average":
        ma_cfg = cfg.get("moving_average", {})
        return MovingAverageFilter(dim=dim, window_size=ma_cfg.get("window_size", 5))

    if filter_type == "one_euro":
        oe_cfg = cfg.get("one_euro", {})
        return OneEuroFilter(
            dim=dim,
            freq=oe_cfg.get("freq", 30.0),
            min_cutoff=oe_cfg.get("min_cutoff", 1.0),
            beta=oe_cfg.get("beta", 0.01),
            d_cutoff=oe_cfg.get("d_cutoff", 1.0),
        )

    raise ValueError(f"Unknown action filter type: {filter_type}")


# Joint trajectory interpolation

def _output_time_grid(total_time: float, output_dt: float) -> np.ndarray:
    """Build a near-output_dt grid that always reaches the final waypoint."""
    if total_time <= 0:
        return np.array([0.0], dtype=np.float64)
    intervals = max(int(np.ceil(total_time / output_dt)), 1)
    return np.linspace(0.0, total_time, intervals + 1, dtype=np.float64)

def _tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas 算法求解三对角线性方程组 Ax=d, 其中:
    a = sub-diagonal (n-1 个)
    b = main diagonal (n 个)
    c = super-diagonal (n-1 个)
    d = right-hand side (n 个)
    """
    n = len(d)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.array([d[0] / b[0]], dtype=np.float64)
    cp = np.empty(n - 1, dtype=np.float64)
    dp = np.empty(n, dtype=np.float64)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])

    x = np.empty(n, dtype=np.float64)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ============================================================================
# 插值基类
# ============================================================================

class BaseInterpolator:
    dim: int

    def __init__(self, dim: int = 23):
        self.dim = dim

    def interpolate(self, actions: np.ndarray) -> np.ndarray:
        """actions: (N, dim), 返回插值后 (M, dim)"""
        return actions

    def reset(self):
        pass


# ============================================================================
# 线性插值
# ============================================================================

class LinearInterpolator(BaseInterpolator):
    """逐关节线性插值 (np.interp)，速度在控制点处不连续。"""

    def __init__(self, dim: int = 23, input_dt: float = 1.0 / 30, output_dt: float = 1.0 / 250):
        super().__init__(dim)
        self.input_dt = float(input_dt)
        self.output_dt = float(output_dt)

    def interpolate(self, actions: np.ndarray) -> np.ndarray:
        N = len(actions)
        if N < 2:
            return actions.astype(np.float32)

        t_in = np.arange(N, dtype=np.float64) * self.input_dt
        total_time = (N - 1) * self.input_dt
        t_out = _output_time_grid(total_time, self.output_dt)
        M = len(t_out)

        result = np.empty((M, self.dim), dtype=np.float32)
        for j in range(self.dim):
            result[:, j] = np.interp(t_out, t_in, actions[:, j].astype(np.float64))
        return result


# ============================================================================
# 三次样条插值 (Natural Cubic Spline)
# ============================================================================

class CubicInterpolator(BaseInterpolator):
    """Natural cubic spline — C2 连续（位置、速度、加速度）。无冲击。"""

    def __init__(self, dim: int = 23, input_dt: float = 1.0 / 30, output_dt: float = 1.0 / 250):
        super().__init__(dim)
        self.input_dt = float(input_dt)
        self.output_dt = float(output_dt)

    def interpolate(self, actions: np.ndarray) -> np.ndarray:
        N = len(actions)
        if N < 2:
            return actions.astype(np.float32)
        if N == 2:
            # 退化为线性
            lin = LinearInterpolator(self.dim, self.input_dt, self.output_dt)
            return lin.interpolate(actions)

        t = np.arange(N, dtype=np.float64) * self.input_dt
        h = np.diff(t)  # (N-1,)

        total_time = (N - 1) * self.input_dt
        t_out = _output_time_grid(total_time, self.output_dt)
        M = len(t_out)

        # --- 构建三对角系统 A·M = B ---
        n_inner = N - 2
        a_sub = h[:-1].astype(np.float64).copy()     # sub
        b_diag = (2.0 * (h[:-1] + h[1:])).astype(np.float64)  # main
        c_sup = h[1:].astype(np.float64).copy()      # super

        result = np.empty((M, self.dim), dtype=np.float32)

        for j in range(self.dim):
            y = actions[:, j].astype(np.float64)
            B = np.empty(n_inner, dtype=np.float64)
            for i in range(n_inner):
                B[i] = 6.0 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])

            M_inner = _tridiag_solve(a_sub, b_diag, c_sup, B)
            M_vec = np.zeros(N, dtype=np.float64)
            M_vec[1:N - 1] = M_inner

            # --- 多项式系数 (power form, per segment) ---
            # P(t) = a + b*(t-t_i) + c*(t-t_i)^2 + d*(t-t_i)^3
            coeffs_a = np.empty(N - 1, dtype=np.float64)
            coeffs_b = np.empty(N - 1, dtype=np.float64)
            coeffs_c = np.empty(N - 1, dtype=np.float64)
            coeffs_d = np.empty(N - 1, dtype=np.float64)
            for i in range(N - 1):
                hi = h[i]
                coeffs_a[i] = y[i]
                coeffs_c[i] = M_vec[i] / 2.0
                coeffs_d[i] = (M_vec[i + 1] - M_vec[i]) / (6.0 * hi)
                coeffs_b[i] = (y[i + 1] - y[i]) / hi - hi * (2.0 * M_vec[i] + M_vec[i + 1]) / 6.0

            # --- 求值 ---
            seg_indices = np.clip(np.searchsorted(t, t_out, side='right') - 1, 0, N - 2)
            for k in range(M):
                si = seg_indices[k]
                dt = t_out[k] - t[si]
                result[k, j] = float(coeffs_a[si] + dt * (coeffs_b[si] + dt * (coeffs_c[si] + dt * coeffs_d[si])))

        return result


# ============================================================================
# 五次 Hermite 样条插值
# ============================================================================

class QuinticInterpolator(BaseInterpolator):
    """Piecewise quintic Hermite — C2 连续，匹配位置/速度/加速度。

    速度与加速度由中心差分估计，端点用前/后向差分。
    """

    def __init__(self, dim: int = 23, input_dt: float = 1.0 / 30, output_dt: float = 1.0 / 250):
        super().__init__(dim)
        self.input_dt = float(input_dt)
        self.output_dt = float(output_dt)

    def interpolate(self, actions: np.ndarray) -> np.ndarray:
        N = len(actions)
        if N < 3:
            # 点数太少，用线性
            lin = LinearInterpolator(self.dim, self.input_dt, self.output_dt)
            return lin.interpolate(actions)

        t = np.arange(N, dtype=np.float64) * self.input_dt
        h = np.diff(t)
        total_time = (N - 1) * self.input_dt
        t_out = _output_time_grid(total_time, self.output_dt)
        M = len(t_out)

        result = np.empty((M, self.dim), dtype=np.float32)

        for j in range(self.dim):
            y = actions[:, j].astype(np.float64)

            # 估计速度 (中心差分)
            v = np.zeros(N, dtype=np.float64)
            v[0] = (y[1] - y[0]) / h[0]  # forward
            for i in range(1, N - 1):
                v[i] = (y[i + 1] - y[i - 1]) / (h[i - 1] + h[i])
            v[N - 1] = (y[N - 1] - y[N - 2]) / h[N - 2]  # backward

            # 估计加速度 (中心差分 of velocities)
            a = np.zeros(N, dtype=np.float64)
            a[0] = (v[1] - v[0]) / h[0]
            for i in range(1, N - 1):
                a[i] = (v[i + 1] - v[i - 1]) / (h[i - 1] + h[i])
            a[N - 1] = (v[N - 1] - v[N - 2]) / h[N - 2]

            # --- 逐段求系数 ---
            # P(t)=c0 + c1*τ + c2*τ² + c3*τ³ + c4*τ⁴ + c5*τ⁵,  τ = t - t_i
            coeffs = np.zeros((N - 1, 6), dtype=np.float64)
            for i in range(N - 1):
                hi = h[i]
                hi2 = hi * hi
                hi3 = hi2 * hi
                hi4 = hi3 * hi
                hi5 = hi4 * hi

                # 已知: P(0)=y_i, P'(0)=v_i, P''(0)=a_i, P(hi)=y_{i+1}, P'(hi)=v_{i+1}, P''(hi)=a_{i+1}
                c0 = y[i]
                c1 = v[i]
                c2 = a[i] / 2.0

                # 3 变量线性方程组求 c3, c4, c5:
                #  c3*hi3 + c4*hi4 + c5*hi5 = y[i+1] - (c0 + c1*hi + c2*hi2)
                #  3*c3*hi2 + 4*c4*hi3 + 5*c5*hi4 = v[i+1] - (c1 + 2*c2*hi)
                #  6*c3*hi + 12*c4*hi2 + 20*c5*hi3 = a[i+1] - 2*c2
                rhs1 = y[i + 1] - (c0 + c1 * hi + c2 * hi2)
                rhs2 = v[i + 1] - (c1 + 2.0 * c2 * hi)
                rhs3 = a[i + 1] - 2.0 * c2

                # 矩阵 [hi3 hi4 hi5; 3hi2 4hi3 5hi4; 6hi 12hi2 20hi3] * [c3 c4 c5]^T = [rhs1 rhs2 rhs3]^T
                # Cramer's rule
                det = (hi3 * (4 * hi3 * 20 * hi3 - 5 * hi4 * 12 * hi2)
                       - hi4 * (3 * hi2 * 20 * hi3 - 5 * hi4 * 6 * hi)
                       + hi5 * (3 * hi2 * 12 * hi2 - 4 * hi3 * 6 * hi))

                if abs(det) > 1e-30:
                    inv_det = 1.0 / det
                    # cofactor matrix for c3
                    c3 = ((rhs1 * (4 * hi3 * 20 * hi3 - 5 * hi4 * 12 * hi2)
                           + hi4 * (rhs3 * 5 * hi4 - rhs2 * 20 * hi3)
                           + hi5 * (rhs2 * 12 * hi2 - rhs3 * 4 * hi3)) * inv_det)
                    # cofactor matrix for c4
                    c4 = ((hi3 * (rhs2 * 20 * hi3 - rhs3 * 5 * hi4)
                           + rhs1 * (5 * hi4 * 6 * hi - 3 * hi2 * 20 * hi3)
                           + hi5 * (rhs3 * 3 * hi2 - rhs2 * 6 * hi)) * inv_det)
                    # cofactor matrix for c5
                    c5 = ((hi3 * (4 * hi3 * rhs3 - 12 * hi2 * rhs2)
                           + hi4 * (6 * hi * rhs2 - 3 * hi2 * rhs3)
                           + rhs1 * (3 * hi2 * 12 * hi2 - 4 * hi3 * 6 * hi)) * inv_det)
                else:
                    c3 = c4 = c5 = 0.0

                coeffs[i] = [c0, c1, c2, c3, c4, c5]

            # --- 求值 ---
            seg_indices = np.clip(np.searchsorted(t, t_out, side='right') - 1, 0, N - 2)
            for k in range(M):
                si = seg_indices[k]
                dt = t_out[k] - t[si]
                c = coeffs[si]
                dt2 = dt * dt
                result[k, j] = float(c[0] + dt * (c[1] + dt * (c[2] + dt * (c[3] + dt * (c[4] + dt * c[5])))))

        return result


# ============================================================================
# Passthrough
# ============================================================================

class PassthroughInterpolator(BaseInterpolator):
    """不插值，直接返回原数据。"""

    def interpolate(self, actions: np.ndarray) -> np.ndarray:
        return actions.astype(np.float32)


# ============================================================================
# 工厂函数
# ============================================================================

def create_interpolator(cfg: dict, dim: int = 23) -> BaseInterpolator:
    """从配置字典创建插值器。

    cfg 格式 (对应 yaml 中 robot.control.interpolation):
        enabled: true
        method: "linear"          # linear / cubic / quintic
        input_hz: 30              # 模型输出频率
        output_hz: 250            # SDK 控制频率
    """
    enabled = cfg.get("enabled", False)
    if not enabled:
        return PassthroughInterpolator(dim)

    method = cfg.get("method", "linear")
    input_hz = float(cfg.get("input_hz", 30))
    output_hz = float(cfg.get("output_hz", 250))
    input_dt = 1.0 / input_hz
    output_dt = 1.0 / output_hz

    if method == "linear":
        return LinearInterpolator(dim, input_dt, output_dt)
    elif method == "cubic":
        return CubicInterpolator(dim, input_dt, output_dt)
    elif method == "quintic":
        return QuinticInterpolator(dim, input_dt, output_dt)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


# Retained only for log/replay compatibility; runtime uses RTCController below.

class _LegacyBlendingRTCController:
    """异步动作 buffer 控制器，支持加权和多项式两种融合方式。"""

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self._fusion_window = int(config.get("fusion_window", 10))
        self._fusion_method = str(config.get("fusion_method", "weighted"))
        self._pre_infer_threshold = int(config.get("pre_infer_threshold", 15))

        self._action_buffer: list = []     # list of np.ndarray (dim,)
        self._exec_ptr: int = 0
        self._lock = threading.Lock()
        self._new_chunk: Optional[list] = None
        self._new_chunk_meta: Optional[dict] = None
        self._new_chunk_ready = threading.Event()
        self._infer_in_progress = False
        self._infer_start_exec_ptr: int = 0  # exec_ptr 快照，用于计算推理延迟帧偏移
        self._infer_seq: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """清空 buffer，每轮新推理开始前调用。"""
        with self._lock:
            self._action_buffer = []
            self._exec_ptr = 0
            self._new_chunk = None
            self._new_chunk_meta = None
            self._new_chunk_ready.clear()
            self._infer_in_progress = False
            self._infer_start_exec_ptr = 0
            self._infer_seq = 0

    def add_chunk(self, actions: list) -> dict:
        """首次填充或融合新 chunk 到 buffer。

        根据推理延迟跳过新 chunk 中已过期的帧：
          frames_elapsed = exec_ptr - _infer_start_exec_ptr
          新 chunk 的前 frames_elapsed 帧对应推理期间已被旧 buffer 覆盖的时间段，丢弃。

        Returns a fusion report:
          mode: replace | fuse | drop_expired
          skipped: how many new-chunk frames were dropped as stale
          start_index: first kept new-chunk index (0 if none skipped)
          frames_elapsed: exec_ptr delta during inference
        """
        original_len = len(actions)
        actions = [np.asarray(a, dtype=np.float32) for a in actions]
        report = {
            "mode": "replace",
            "skipped": 0,
            "start_index": 0,
            "frames_elapsed": 0,
            "original_len": original_len,
            "kept": original_len,
        }
        with self._lock:
            if not self._action_buffer or self._exec_ptr >= len(self._action_buffer):
                self._action_buffer = actions
                self._exec_ptr = 0
                report["mode"] = "replace"
                return report

            remaining = self._action_buffer[self._exec_ptr:]
            frames_elapsed = self._exec_ptr - self._infer_start_exec_ptr
            report["frames_elapsed"] = int(frames_elapsed)
            if frames_elapsed > 0:
                if frames_elapsed >= len(actions):
                    report["mode"] = "drop_expired"
                    report["skipped"] = len(actions)
                    report["kept"] = 0
                    report["start_index"] = original_len
                    return report
                report["skipped"] = int(frames_elapsed)
                report["start_index"] = int(frames_elapsed)
                actions = actions[frames_elapsed:]

            if self._fusion_method == "polynomial":
                fused = self._fuse_quintic_hermite(remaining, actions)
            else:
                fused = self._fuse_weighted(remaining, actions)
            self._action_buffer = self._action_buffer[:self._exec_ptr] + fused
            report["mode"] = "fuse"
            report["kept"] = len(actions)
            return report

    def get_next_action(self) -> Optional[np.ndarray]:
        """取下一帧动作，buffer 空返回 None。"""
        with self._lock:
            if self._exec_ptr >= len(self._action_buffer):
                return None
            action = self._action_buffer[self._exec_ptr]
            self._exec_ptr += 1
            return action.copy()

    def buffer_remaining(self) -> int:
        """返回 buffer 中剩余未执行的帧数。"""
        with self._lock:
            return max(0, len(self._action_buffer) - self._exec_ptr)

    def should_trigger_inference(self) -> bool:
        """是否应该触发下一轮异步推理。"""
        return (self.buffer_remaining() <= self._pre_infer_threshold
                and not self._infer_in_progress)

    @property
    def infer_in_progress(self) -> bool:
        return self._infer_in_progress

    @property
    def new_chunk_ready(self) -> bool:
        return self._new_chunk_ready.is_set()

    def mark_inference_started(self):
        """异步推理线程开始时调用，记录当前 exec_ptr 用于计算延迟偏移。"""
        self._infer_in_progress = True
        with self._lock:
            self._infer_start_exec_ptr = self._exec_ptr

    def mark_inference_done(self, new_actions: list, meta: Optional[dict] = None):
        """异步推理线程完成时调用。"""
        self._new_chunk = [np.asarray(a, dtype=np.float32) for a in new_actions]
        self._new_chunk_meta = meta or {}
        self._new_chunk_ready.set()
        self._infer_in_progress = False

    def consume_new_chunk(self):
        """主线程消费异步推理结果，融合入 buffer。

        Returns (fusion_report, meta) or (None, None) if nothing is ready.
        """
        if not self._new_chunk_ready.is_set():
            return None, None
        self._new_chunk_ready.clear()
        chunk = self._new_chunk
        meta = self._new_chunk_meta or {}
        self._new_chunk = None
        self._new_chunk_meta = None
        # A failed/empty inference must not erase the still-valid tail of the
        # previous buffer. Report it and let the caller decide whether to stop.
        if chunk:
            return self.add_chunk(chunk), meta
        return {
            "mode": "empty",
            "skipped": 0,
            "start_index": 0,
            "original_len": 0,
            "kept": 0,
        }, meta

    # ------------------------------------------------------------------
    # Fusion methods
    # ------------------------------------------------------------------

    def _fuse_weighted(self, old_chunk, new_chunk):
        """加权线性过渡，权重从 old→new 线性变化。

        overlap = min(fusion_window, len(old), len(new))
        fused[i] = (1-t)*old[i] + t*new[i],  t = i/overlap
        """
        overlap = min(self._fusion_window, len(old_chunk), len(new_chunk))
        result = []
        for i in range(overlap):
            t = i / max(overlap, 1)
            w_old = 1.0 - t
            w_new = t
            result.append(w_old * old_chunk[i] + w_new * new_chunk[i])
        if len(new_chunk) > overlap:
            result.extend(new_chunk[overlap:])
        return result

    def _fuse_quintic_hermite(self, old_chunk, new_chunk):
        """五次 Hermite 样条融合，保证 C2 连续（位置/速度/加速度匹配）。

        基函数:
          h00(t) = 1 - 10t³ + 15t⁴ - 6t⁵   (h(0)=1, 其他零阶/一阶/二阶=0)
          h01(t) = 10t³ - 15t⁴ + 6t⁵       (h(1)=1, 其他零阶/一阶/二阶=0)

        fused[i] = h00(t) * old[i] + h01(t) * new[i]
        """
        overlap = min(self._fusion_window, len(old_chunk), len(new_chunk))
        old_arr = np.asarray(old_chunk, dtype=np.float64)
        new_arr = np.asarray(new_chunk, dtype=np.float64)
        dim = old_arr.shape[1]

        result = np.empty((max(overlap, len(new_chunk)), dim), dtype=np.float64)

        if overlap > 0:
            t_vals = np.linspace(0.0, 1.0, overlap, endpoint=False)  # 0, 1/N, 2/N, ...
            t2 = t_vals * t_vals
            t3 = t2 * t_vals
            t4 = t3 * t_vals
            t5 = t4 * t_vals
            h00 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5   # (overlap,)
            h01 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5          # (overlap,)
            for d in range(dim):
                result[:overlap, d] = h00 * old_arr[:overlap, d] + h01 * new_arr[:overlap, d]

        if len(new_chunk) > overlap:
            result[overlap:len(new_chunk)] = new_arr[overlap:]

        return [result[i].astype(np.float32) for i in range(len(new_chunk))]


class RTCController:
    """Robot-side queue for model-native Real-Time Chunking.

    Continuity is generated by the policy RTC processor during flow matching.
    This class never blends chunks locally: it snapshots the unexecuted prefix
    and replaces the future queue after compensating for inference latency.
    """

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self._pre_infer_threshold = int(config.get("pre_infer_threshold", 15))
        self._max_steps_per_observation = int(config.get("max_steps_per_observation", 0))
        self._inference_delay_frames = max(0, int(config.get("inference_delay", 0)))
        if self._max_steps_per_observation < 0:
            raise ValueError("RTC max_steps_per_observation must be >= 0")

        self._action_buffer: list = []
        self._exec_ptr = 0
        self._lock = threading.Lock()
        self._new_chunk: Optional[list] = None
        self._new_chunk_meta: dict = {}
        self._new_chunk_ready = threading.Event()
        self._infer_in_progress = False
        self._control_step = 0
        self._trajectory_step = 0
        self._last_install_trajectory_step = 0
        self._last_action: Optional[np.ndarray] = None
        self._session_id = uuid.uuid4().hex
        self._session_active = True
        self._infer_seq = 0

    def reset(self):
        with self._lock:
            self._action_buffer = []
            self._exec_ptr = 0
            self._new_chunk = None
            self._new_chunk_meta = {}
            self._new_chunk_ready.clear()
            self._infer_in_progress = False
            self._control_step = 0
            self._trajectory_step = 0
            self._last_install_trajectory_step = 0
            self._last_action = None
            self._session_id = uuid.uuid4().hex
            self._session_active = True
            self._infer_seq = 0

    def add_chunk(self, actions: list, skip_frames: int = 0) -> dict:
        """Replace the future queue, dropping frames stale since capture."""
        original_len = len(actions)
        actions = _normalize_rtc_actions(actions)
        skip_frames = max(0, min(int(skip_frames), len(actions)))
        actions = actions[skip_frames:]
        report = {
            "mode": "replace_rtc",
            "skipped": skip_frames,
            "start_index": skip_frames,
            "frames_elapsed": skip_frames,
            "original_len": original_len,
            "kept": len(actions),
        }
        with self._lock:
            self._action_buffer = actions
            self._exec_ptr = 0
            self._last_install_trajectory_step = self._trajectory_step
        return report

    def get_next_action(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._observation_budget_exhausted():
                return None
            if self._exec_ptr >= len(self._action_buffer):
                return None
            action = self._action_buffer[self._exec_ptr]
            self._exec_ptr += 1
            self._control_step += 1
            self._trajectory_step += 1
            self._last_action = action.copy()
            return action.copy()

    def record_hold_frame(self) -> None:
        with self._lock:
            self._control_step += 1

    def last_action(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last_action is None else self._last_action.copy()

    @property
    def control_step(self) -> int:
        with self._lock:
            return self._control_step

    @property
    def trajectory_step(self) -> int:
        with self._lock:
            return self._trajectory_step

    def frames_since_last_chunk_install(self) -> int:
        with self._lock:
            return max(0, self._trajectory_step - self._last_install_trajectory_step)

    def buffer_remaining(self) -> int:
        with self._lock:
            return max(0, len(self._action_buffer) - self._exec_ptr)

    def should_trigger_inference(self) -> bool:
        with self._lock:
            remaining = max(0, len(self._action_buffer) - self._exec_ptr)
            trigger = (
                self._observation_budget_exhausted() or remaining == 0
                if self._max_steps_per_observation
                else remaining <= self._pre_infer_threshold
            )
            return (
                trigger
                and not self._infer_in_progress
                and not self._new_chunk_ready.is_set()
            )

    def _observation_budget_exhausted(self) -> bool:
        return bool(self._max_steps_per_observation) and (
            self._trajectory_step - self._last_install_trajectory_step
            >= self._max_steps_per_observation
        )

    @property
    def infer_in_progress(self) -> bool:
        with self._lock:
            return self._infer_in_progress

    @property
    def new_chunk_ready(self) -> bool:
        return self._new_chunk_ready.is_set()

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._session_id

    def is_session_active(self, session_id: Optional[str] = None) -> bool:
        with self._lock:
            return self._session_active and (
                session_id is None or session_id == self._session_id
            )

    def close_session(self) -> None:
        with self._lock:
            self._session_active = False
            self._infer_in_progress = False

    def mark_inference_started(self) -> dict:
        with self._lock:
            self._infer_in_progress = True
            self._infer_seq += 1
            return {
                "session_id": self._session_id,
                "infer_seq": self._infer_seq,
                "infer_start_step": self._control_step,
                "infer_start_time": time.monotonic(),
            }

    def mark_observation_captured(self, session_id: str) -> Optional[dict]:
        with self._lock:
            if not self._session_active or session_id != self._session_id:
                return None
            return {
                "obs_capture_step": self._control_step,
                "obs_capture_control_step": self._control_step,
                "obs_capture_trajectory_step": self._trajectory_step,
                "obs_capture_time": time.monotonic(),
                "previous_actions": [
                    action.copy() for action in self._action_buffer[self._exec_ptr:]
                ],
                "previous_action_start_index": self._exec_ptr,
                "inference_delay": self._inference_delay_frames,
            }

    def set_inference_delay_frames(self, frames: int) -> None:
        with self._lock:
            self._inference_delay_frames = max(0, int(frames))

    def mark_inference_done(
        self,
        new_actions: list,
        meta: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        actions = _normalize_rtc_actions(new_actions)
        with self._lock:
            if not self._session_active or (
                session_id is not None and session_id != self._session_id
            ):
                return False
            self._new_chunk = actions
            self._new_chunk_meta = dict(meta or {})
            if "inference_delay_frames" in self._new_chunk_meta:
                self._inference_delay_frames = max(
                    0, int(self._new_chunk_meta["inference_delay_frames"])
                )
            self._infer_in_progress = False
            self._new_chunk_ready.set()
            return True

    def wait_for_new_chunk(self, timeout: float) -> bool:
        return self._new_chunk_ready.wait(max(0.0, float(timeout)))

    def consume_new_chunk(self):
        """Install a model-native RTC chunk; no action-space blending."""
        with self._lock:
            if not self._new_chunk_ready.is_set():
                return None, None
            self._new_chunk_ready.clear()
            chunk = self._new_chunk
            self._new_chunk = None
            meta = self._new_chunk_meta
            self._new_chunk_meta = {}

        if chunk:
            observation_step = int(
                meta.get("obs_capture_trajectory_step", self.trajectory_step)
            )
            local_delay_frames = max(0, self.trajectory_step - observation_step)
            report = self.add_chunk(chunk, skip_frames=local_delay_frames)
            report["server_inference_delay_frames"] = int(
                meta.get("inference_delay_frames", 0) or 0
            )
            report["prefix_frames_used"] = int(meta.get("prefix_frames_used", 0) or 0)
            return report, meta
        return ({
            "mode": "empty",
            "skipped": 0,
            "start_index": 0,
            "frames_elapsed": 0,
            "original_len": 0,
            "kept": 0,
        }, meta)


def _normalize_rtc_actions(actions: list) -> list:
    normalized = [np.asarray(action, dtype=np.float32).reshape(-1) for action in actions]
    if not normalized:
        return []
    dim = normalized[0].size
    for index, action in enumerate(normalized[1:], start=1):
        if action.size != dim:
            raise ValueError(
                "action dim mismatch inside RTC chunk: "
                f"frame 0 has {dim}, frame {index} has {action.size}"
            )
    return normalized
