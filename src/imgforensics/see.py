"""Seeing tools: give a vision LLM pixels it would otherwise lose to downscaling.

Every view can carry a coordinate grid labelled in ORIGINAL image pixels, so the
model can name regions precisely in follow-up calls.
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import clamp_box

ENHANCE_MODES = ("highpass", "contrast", "channels", "saturation", "noise", "recompress")


def draw_grid(view: np.ndarray, x0: int, y0: int, scale: float, step: int) -> np.ndarray:
    """Overlay a grid every `step` original pixels, labelled with original coords."""
    out = view.copy()
    H, W = out.shape[:2]
    font_scale = max(0.35, min(0.6, W / 1400))
    gx = (x0 // step + 1) * step
    while (gx - x0) * scale < W:
        px = int((gx - x0) * scale)
        cv2.line(out, (px, 0), (px, H), (255, 255, 0), 1)
        cv2.putText(out, str(gx), (px + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
        gx += step
    gy = (y0 // step + 1) * step
    while (gy - y0) * scale < H:
        py = int((gy - y0) * scale)
        cv2.line(out, (0, py), (W, py), (0, 255, 255), 1)
        cv2.putText(out, str(gy), (2, py - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)
        gy += step
    return out


def overview(img: np.ndarray, max_side: int = 1400, grid_step: int | None = None) -> np.ndarray:
    """Whole image, resized to fit max_side, with a coordinate grid."""
    H, W = img.shape[:2]
    s = min(1.0, max_side / max(H, W))
    view = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    step = grid_step or _nice_step(max(H, W) / 10)
    return draw_grid(view, 0, 0, s, step)


def zoom(img: np.ndarray, x: int, y: int, w: int, h: int, scale: float | None = None,
         grid: bool = True, max_side: int = 1200) -> np.ndarray:
    """Crop at full resolution and upscale with nearest-neighbour (no invented detail)."""
    x, y, w, h = clamp_box(img, x, y, w, h)
    crop = img[y:y + h, x:x + w]
    if scale is None:
        scale = max(1.0, min(8.0, max_side / max(w, h)))
    view = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    if grid:
        view = draw_grid(view, x, y, scale, _nice_step(max(w, h) / 6))
    return view


def enhance(img: np.ndarray, mode: str, x: int = 0, y: int = 0, w: int | None = None,
            h: int | None = None, max_side: int = 1200) -> np.ndarray:
    """Alternative renderings that expose seams, texture breaks and colour oddities.

    highpass   - fine detail only; regenerated areas often look smoother or differently textured
    contrast   - local contrast (CLAHE) to reveal hidden structure in flat areas
    channels   - R | G | B side by side; colour fringing and channel mismatches
    saturation - HSV saturation as greyscale; pasted colour stands out
    noise      - amplified denoising residual; sensor noise should look uniform
    recompress - |image - JPEG(q=90)| amplified; areas with a different history react differently
    """
    if mode not in ENHANCE_MODES:
        raise ValueError(f"mode must be one of {ENHANCE_MODES}")
    H, W = img.shape[:2]
    x, y, w, h = clamp_box(img, x, y, w or W - x, h or H - y)
    crop = img[y:y + h, x:x + w]

    if mode == "highpass":
        f = crop.astype(np.float32)
        hp = f - cv2.GaussianBlur(f, (0, 0), 2.0)
        out = np.clip(128 + hp * 4, 0, 255).astype(np.uint8)
    elif mode == "contrast":
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif mode == "channels":
        b, g, r = cv2.split(crop)
        out = cv2.cvtColor(np.hstack([r, g, b]), cv2.COLOR_GRAY2BGR)
    elif mode == "saturation":
        s = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1]
        out = cv2.applyColorMap(s, cv2.COLORMAP_INFERNO)
    elif mode == "noise":
        den = cv2.fastNlMeansDenoisingColored(crop, None, 5, 5, 7, 21)
        res = crop.astype(np.float32) - den.astype(np.float32)
        out = np.clip(128 + res * 8, 0, 255).astype(np.uint8)
    else:  # recompress
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        diff = np.abs(crop.astype(np.int16) - rec.astype(np.int16)).sum(2).astype(np.float32)
        out = cv2.applyColorMap(np.clip(diff * 6, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

    s = min(4.0, max_side / max(out.shape[:2]))
    interp = cv2.INTER_NEAREST if s >= 1 else cv2.INTER_AREA
    return cv2.resize(out, (int(out.shape[1] * s), int(out.shape[0] * s)), interpolation=interp)


def _nice_step(raw: float) -> int:
    for s in (10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if s >= raw:
            return s
    return 1000
