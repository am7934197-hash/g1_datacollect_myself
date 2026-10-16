"""Client-side image preprocessing for per-checkpoint input geometry."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

TRAINING_IMAGE_HW = (480, 640)  # H, W; 4:3 source used to build the 224 videos


def letterbox_content_size(
    src_h: int, src_w: int, target_h: int = 224, target_w: int = 224
) -> Tuple[int, int]:
    """Content HxW after long-edge INTER_LINEAR scale into target, before padding."""
    scale = min(target_h / src_h, target_w / src_w)
    new_h = max(1, min(target_h, int(round(src_h * scale))))
    new_w = max(1, min(target_w, int(round(src_w * scale))))
    return new_h, new_w


def letterbox_pad_hw(
    src_h: int, src_w: int, target_h: int = 224, target_w: int = 224
) -> Tuple[int, int, int, int]:
    """Return (pad_top, pad_bottom, pad_left, pad_right) for a native HxW frame."""
    new_h, new_w = letterbox_content_size(src_h, src_w, target_h, target_w)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    return pad_top, target_h - new_h - pad_top, pad_left, target_w - new_w - pad_left


def crop_to_training_aspect(
    img: np.ndarray,
    training_hw: Tuple[int, int] = TRAINING_IMAGE_HW,
    rtol: float = 0.02,
) -> np.ndarray:
    """Center-crop to the training 4:3 aspect. Does not stretch.

    Wrist cameras on this robot are 720×1280 (16:9). Training videos are
    480×640 (4:3). Cropping 1280→960 keeps height 720, then letterbox
    produces the same 168×224 content + 28px bars as 480×640.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or native_matches_training_aspect(h, w, training_hw, rtol):
        return img
    train_h, train_w = training_hw
    target_aspect = train_h / train_w
    native_aspect = h / w
    if native_aspect < target_aspect:
        new_w = max(1, min(w, int(round(h / target_aspect))))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    new_h = max(1, min(h, int(round(w * target_aspect))))
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, :]


def native_matches_training_aspect(
    src_h: int, src_w: int, training_hw: Tuple[int, int] = TRAINING_IMAGE_HW, rtol: float = 0.02
) -> bool:
    """True when native aspect matches the 480x640 (4:3) training videos."""
    if src_h <= 0 or src_w <= 0:
        return False
    train_h, train_w = training_hw
    native_aspect = src_h / src_w
    train_aspect = train_h / train_w
    return abs(native_aspect - train_aspect) <= rtol * train_aspect


def describe_camera_geometry(
    native_img: np.ndarray,
    processed_img: Optional[np.ndarray] = None,
    training_hw: Tuple[int, int] = TRAINING_IMAGE_HW,
    crop_to_training: bool = False,
) -> dict:
    """Native HxW vs the 224 letterbox that PolicyServer will see."""
    h, w = int(native_img.shape[0]), int(native_img.shape[1])
    c = int(native_img.shape[2]) if native_img.ndim == 3 else 1
    cropped = crop_to_training_aspect(native_img, training_hw) if crop_to_training else native_img
    ch, cw = int(cropped.shape[0]), int(cropped.shape[1])
    pad_top, pad_bottom, pad_left, pad_right = letterbox_pad_hw(ch, cw)
    report = {
        "native_shape": [h, w, c],
        "native_dtype": str(native_img.dtype),
        "aspect_h_over_w": round(h / w, 4) if w else None,
        "matches_training_4_3": native_matches_training_aspect(h, w, training_hw),
        "crop_to_training_aspect": bool(crop_to_training),
        "cropped_shape": [ch, cw, c],
        "training_hw": list(training_hw),
        "expected_letterbox_pad": {
            "top": pad_top,
            "bottom": pad_bottom,
            "left": pad_left,
            "right": pad_right,
        },
    }
    if processed_img is not None and isinstance(processed_img, np.ndarray) and processed_img.ndim == 3:
        top_bar, bottom_bar = count_letterbox_black_bar_rows(processed_img)
        report["processed_shape"] = [int(x) for x in processed_img.shape]
        report["processed_black_bar_rows"] = {"top": top_bar, "bottom": bottom_bar}
    return report


def count_letterbox_black_bar_rows(img: np.ndarray, threshold: int = 2) -> Tuple[int, int]:
    """Count near-black rows from the top and bottom of an HWC uint8 image."""
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[0] == 0:
        return 0, 0
    row_max = img.max(axis=(1, 2))
    top = 0
    for value in row_max:
        if int(value) <= threshold:
            top += 1
        else:
            break
    bottom = 0
    for value in row_max[::-1]:
        if int(value) <= threshold:
            bottom += 1
        else:
            break
    return top, bottom


class ImagePreprocessor:
    """Letterbox to target_size before transport. PI05 路径下应 enabled=true。

    Usage:
        pp = ImagePreprocessor({"enabled": True, "target_size": [224, 224]})
        obs = pp.process(obs)
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.enabled = cfg.get("enabled", False)
        self._target_h, self._target_w = cfg.get("target_size", [224, 224])
        self._keep_ratio = cfg.get("keep_aspect_ratio", True)
        # This remains opt-in at the utility level so callers must state their
        # geometry contract. The production config pins it on: ready_224 was
        # built from 640x480 content and has 28px top/bottom bars at 224.
        self._crop_to_training = bool(cfg.get("crop_to_training_aspect", False))
        train_hw = cfg.get("training_hw", list(TRAINING_IMAGE_HW))
        self._training_hw = (int(train_hw[0]), int(train_hw[1]))
        emulate_cfg = cfg.get("emulate_data_collection") or {}
        self._emulate_dc = bool(emulate_cfg.get("enabled", False))
        self._emulate_specs = parse_camera_specs(emulate_cfg) if self._emulate_dc else {}
        self._emulate_cameras: Set[str] = set(self._emulate_specs)
        self._emulate_zoom = {
            name: float(spec["zoom_factor"]) for name, spec in self._emulate_specs.items()
        }
        self._emulate_out_wh = next(
            (spec["out_wh"] for spec in self._emulate_specs.values()),
            (640, 480),
        )
        self._emulate_scheme = next(
            (spec.get("scheme", "via_640") for spec in self._emulate_specs.values()),
            "via_640",
        )

    def process(self, obs: dict) -> dict:
        """处理 obs 中所有图像 ndarray。未启用时原样返回。"""
        if not obs or not self.enabled:
            return obs
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                img = v
                spec = self._emulate_specs.get(k) if self._emulate_dc else None
                if spec:
                    if spec.get("crop_to_training_aspect"):
                        img = crop_to_training_aspect(img, self._training_hw)
                    img = emulate_fov(
                        img,
                        spec["zoom_factor"],
                        spec["out_wh"],
                        spec.get("scheme", self._emulate_scheme),
                    )
                obs[k] = self._resize(img)
        return obs

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """按长边等比缩进 target，居中黑边 → (target_h, target_w, C) uint8。"""
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)

        if self._crop_to_training:
            img = crop_to_training_aspect(img, self._training_hw)
            h, w = img.shape[:2]

        if self._keep_ratio:
            scale = min(self._target_h / h, self._target_w / w)
            new_h = max(1, min(self._target_h, int(round(h * scale))))
            new_w = max(1, min(self._target_w, int(round(w * scale))))
            resized = self._interp(img, new_w, new_h)
            pad_h0 = (self._target_h - new_h) // 2
            pad_w0 = (self._target_w - new_w) // 2
            padded = np.zeros((self._target_h, self._target_w, 3), dtype=np.uint8)
            padded[pad_h0:pad_h0 + new_h, pad_w0:pad_w0 + new_w] = resized
            return padded
        return self._interp(img, self._target_w, self._target_h)

    def _interp(self, img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        if cv2 is not None:
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        from PIL import Image
        return np.array(
            Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR),
            dtype=np.uint8,
        )


# Optional data-collection camera-geometry emulation

DEFAULT_ZOOM_FACTOR = 1.0  # on already-rectified working frames; vendor RAW zoom=3 is not this
DEFAULT_OUT_WH = (640, 480)  # color_width, color_height
DEFAULT_SCHEME = "via_640"
VALID_SCHEMES = ("via_640", "crop_only")
HEAD_CAMERAS = ("head_left", "head_right")
WRIST_CAMERAS = ("left_arm", "right_arm")


def _interp(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    from PIL import Image
    return np.array(
        Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR),
        dtype=np.uint8,
    )


def normalize_scheme(value: Any) -> str:
    scheme = str(value or DEFAULT_SCHEME).strip()
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"emulate_data_collection.scheme must be one of {VALID_SCHEMES}, got {value!r}")
    return scheme


def center_crop_zoom(
    img: np.ndarray,
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
) -> np.ndarray:
    """Center-crop by zoom_factor. Does not resize. zoom=1 returns img."""
    if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
        raise ValueError("img must be an HWC ndarray")
    zoom = float(zoom_factor)
    if zoom <= 0:
        raise ValueError(f"zoom_factor must be > 0, got {zoom_factor}")
    if abs(zoom - 1.0) < 1e-9:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    crop_w = max(1, min(w, int(round(w / zoom))))
    crop_h = max(1, min(h, int(round(h / zoom))))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def resize_to_wh(img: np.ndarray, out_wh: Tuple[int, int] = DEFAULT_OUT_WH) -> np.ndarray:
    h, w = int(img.shape[0]), int(img.shape[1])
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"out_wh must be positive, got {out_wh}")
    if (h, w) == (out_h, out_w):
        return img
    return _interp(img, out_w, out_h)


def zoom_crop_to_size(
    img: np.ndarray,
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
    out_wh: Tuple[int, int] = DEFAULT_OUT_WH,
) -> np.ndarray:
    """Center-crop by zoom_factor then resize to ``out_wh`` (W, H)."""
    if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
        raise ValueError("img must be an HWC ndarray")
    h, w = int(img.shape[0]), int(img.shape[1])
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"out_wh must be positive, got {out_wh}")
    if (h, w) == (out_h, out_w):
        return img
    cropped = center_crop_zoom(img, zoom_factor)
    return resize_to_wh(cropped, out_wh)


def emulate_fov(
    img: np.ndarray,
    zoom_factor: float,
    out_wh: Tuple[int, int],
    scheme: str = DEFAULT_SCHEME,
) -> np.ndarray:
    """Crop to data-collection FOV. via_640 also resizes to out_wh; crop_only does not."""
    scheme = normalize_scheme(scheme)
    h, w = int(img.shape[0]), int(img.shape[1])
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if (h, w) == (out_h, out_w):
        return img
    cropped = center_crop_zoom(img, zoom_factor)
    if scheme == "crop_only":
        return cropped
    return resize_to_wh(cropped, out_wh)


def parse_camera_specs(emulate_cfg: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    """Per-camera zoom / crop settings.

    Heads use zoom=1 on already-rectified working frames. Wrists crop 16:9 to
    4:3 then optionally resize to 640x480. A plain camera list keeps the global
    zoom_factor for those names only.
    """
    cfg = emulate_cfg or {}
    out_wh = cfg.get("output_wh", list(DEFAULT_OUT_WH))
    out_wh = (int(out_wh[0]), int(out_wh[1]))
    global_zoom = float(cfg.get("zoom_factor", DEFAULT_ZOOM_FACTOR))
    scheme = normalize_scheme(cfg.get("scheme", DEFAULT_SCHEME))
    cameras = cfg.get("cameras")
    specs: Dict[str, Dict[str, Any]] = {}

    def head_spec(zoom: float = DEFAULT_ZOOM_FACTOR) -> Dict[str, Any]:
        return {
            "zoom_factor": float(zoom),
            "out_wh": out_wh,
            "crop_to_training_aspect": False,
            "undistort_type": int(cfg.get("undistort_type", 1)),
            "scheme": scheme,
        }

    def wrist_spec() -> Dict[str, Any]:
        return {
            "zoom_factor": 1.0,
            "out_wh": out_wh,
            "crop_to_training_aspect": True,
            "undistort_type": 0,
            "scheme": scheme,
        }

    if cameras is None:
        for name in HEAD_CAMERAS:
            specs[name] = head_spec(global_zoom)
        for name in WRIST_CAMERAS:
            specs[name] = wrist_spec()
        return specs

    if isinstance(cameras, dict):
        for name, raw in cameras.items():
            key = str(name)
            item = dict(raw) if isinstance(raw, dict) else {}
            spec = wrist_spec() if key in WRIST_CAMERAS else head_spec(global_zoom)
            if "zoom_factor" in item:
                spec["zoom_factor"] = float(item["zoom_factor"])
            if "crop_to_training_aspect" in item:
                spec["crop_to_training_aspect"] = bool(item["crop_to_training_aspect"])
            if "undistort_type" in item:
                spec["undistort_type"] = int(item["undistort_type"])
            if "output_wh" in item:
                wh = item["output_wh"]
                spec["out_wh"] = (int(wh[0]), int(wh[1]))
            if "scheme" in item:
                spec["scheme"] = normalize_scheme(item["scheme"])
            specs[key] = spec
        return specs

    for name in cameras:
        key = str(name)
        specs[key] = head_spec(global_zoom) if key in HEAD_CAMERAS else wrist_spec()
        if key not in HEAD_CAMERAS and key not in WRIST_CAMERAS:
            specs[key] = head_spec(global_zoom)
    return specs




# PI0.5 tensor-path diagnostics

def count_bars(img: np.ndarray, threshold: float = 2.0) -> Tuple[int, int, int, int]:
    """Return (top, bottom, left, right) near-black margins."""
    if img.ndim == 3 and img.shape[0] == 3:
        hwc = np.transpose(img, (1, 2, 0))
    else:
        hwc = img
    if hwc.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(hwc)) <= 1.5 else 1.0
        vis = np.clip(np.asarray(hwc) * scale, 0, 255).astype(np.uint8)
    else:
        vis = hwc
    row_max = vis.max(axis=(1, 2))
    col_max = vis.max(axis=(0, 2))
    top = 0
    for value in row_max:
        if int(value) <= threshold:
            top += 1
        else:
            break
    bottom = 0
    for value in row_max[::-1]:
        if int(value) <= threshold:
            bottom += 1
        else:
            break
    left = 0
    for value in col_max:
        if int(value) <= threshold:
            left += 1
        else:
            break
    right = 0
    for value in col_max[::-1]:
        if int(value) <= threshold:
            right += 1
        else:
            break
    return top, bottom, left, right


def _interp(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    from PIL import Image
    return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))


def resize_with_pad_uint8(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """OpenPI/lerobot resize_with_pad geometry on HWC uint8."""
    h, w = int(img.shape[0]), int(img.shape[1])
    ratio = max(w / float(width), h / float(height))
    resized_h = max(1, int(h / ratio))
    resized_w = max(1, int(w / ratio))
    resized = _interp(img, resized_w, resized_h)
    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    padded = np.zeros((height, width, 3), dtype=np.uint8)
    padded[pad_h0:pad_h0 + resized_h, pad_w0:pad_w0 + resized_w] = resized
    return padded


def preprocess_images_pi05(images_hwc_uint8: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Run the documented PI05 image path once and return tensors + checks.

    Input: camera_key -> HWC uint8 RGB in 0-255.
    Output per camera:
      input_uint8, after_div255, after_resize, after_pm1 CHW float32
    """
    report: Dict[str, Any] = {"cameras": {}, "div255_applied": True, "pm1_applied": True}
    for key, image in images_hwc_uint8.items():
        arr = np.asarray(image)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"{key}: expected HWC uint8 RGB, got shape={arr.shape} dtype={arr.dtype}"
            )
        in_min = int(arr.min())
        in_max = int(arr.max())
        if in_min < 0 or in_max > 255:
            raise ValueError(f"{key}: uint8 range {in_min}-{in_max} is invalid")

        as_float = arr.astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        resized_again = False
        if (h, w) != (224, 224):
            padded_u8 = resize_with_pad_uint8(arr, 224, 224)
            as_float = padded_u8.astype(np.float32) / 255.0
            resized_again = True
        else:
            padded_u8 = arr
        pm1 = as_float * 2.0 - 1.0
        chw = np.transpose(pm1, (2, 0, 1))
        top, bottom, left, right = count_bars(padded_u8)
        # RGB check: channel 0 is red. A red-dominant patch in the non-bar region.
        content = padded_u8[top:224 - bottom if bottom else 224, left:224 - right if right else 224]
        rgb_means = [float(content[:, :, c].mean()) for c in range(3)] if content.size else [0, 0, 0]
        report["cameras"][key] = {
            "input_shape": list(arr.shape),
            "input_dtype": str(arr.dtype),
            "input_min": in_min,
            "input_max": in_max,
            "div255_min": float(as_float.min()),
            "div255_max": float(as_float.max()),
            "tensor_shape": list(chw.shape),
            "tensor_dtype": str(chw.dtype),
            "tensor_min": float(chw.min()),
            "tensor_max": float(chw.max()),
            "resized_again": resized_again,
            "black_bars_tblr": [top, bottom, left, right],
            "content_hw": [int(content.shape[0]), int(content.shape[1])] if content.size else [0, 0],
            "content_channel_means_rgb": rgb_means,
            "layout": "CHW",
            "color": "RGB",
        }
    return report
