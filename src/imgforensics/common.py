"""Shared helpers: loading, region handling, result format, PNG encoding."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
import numpy as np

# Reliability levels every tool reports alongside its value.
# "none" means the test does not apply to this image (e.g. no straight lines).
RELIABILITY = ("none", "low", "medium", "high")


@dataclass
class Result:
    """Uniform tool output. The LLM should weigh `value` by `reliability`."""
    tool: str
    applicable: bool
    reliability: str                 # one of RELIABILITY
    leaning: str                     # "camera", "generated", "edited", "neutral"
    value: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float):
        return round(o, 4)
    return o


def load(path: str) -> np.ndarray:
    """Load as BGR uint8. Raises if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def clamp_box(img: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty region {x},{y},{w},{h} for image {W}x{H}")
    return x0, y0, x1 - x0, y1 - y0


def poly_mask(shape: tuple[int, ...], polygons: list[list[list[float]]]) -> np.ndarray:
    """Polygons as [[x,y],...] lists -> uint8 mask (255 inside)."""
    m = np.zeros(shape[:2], np.uint8)
    for p in polygons:
        cv2.fillPoly(m, [np.asarray(p, np.int32)], 255)
    return m


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def gray_f(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def phone_focal_px(w: int, h: int, equiv_mm: float = 26.0) -> float:
    """Focal length in pixels for a 35mm-equivalent focal length (diagonal-based)."""
    return equiv_mm / 43.27 * float(np.hypot(w, h))
