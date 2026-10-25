"""Render a physically correct synthetic floor through a known pinhole camera.
Ground truth focal length lets the geometry tests be checked exactly."""
from __future__ import annotations

import math

import cv2
import numpy as np


def render_floor(W=896, H=1195, f=900.0, cam_h=1.5, pitch_deg=12.0, yaw_deg=30.0,
                 tile_m=0.6, seed=0, chips=True):
    rng = np.random.default_rng(seed)
    # top-down floor texture: 20 m x 20 m at 4 mm/px
    s = 0.004
    T = 5000
    tex = np.full((T, T, 3), (200, 196, 190), np.uint8)
    if chips:
        for _ in range(60000):
            cx, cy = rng.integers(0, T, 2)
            ax, ay = rng.uniform(4, 10, 2)
            ang = rng.uniform(0, 180)
            col = tuple(int(c) for c in rng.choice([(60, 60, 70), (40, 90, 150), (110, 120, 120), (70, 110, 170)]))
            cv2.ellipse(tex, (int(cx), int(cy)), (int(ax), int(ay)), ang, 0, 360, col, -1)
    step = int(tile_m / s)
    for k in range(0, T, step):
        cv2.line(tex, (k, 0), (k, T), (90, 90, 90), 6)
        cv2.line(tex, (0, k), (T, k), (90, 90, 90), 6)
    # floor coords: X = (u - T/2)*s, Z = 0.5 + (T - v)*s  (v=T is nearest)
    A = np.array([[s, 0, -T / 2 * s], [0, -s, 0.5 + T * s], [0, 0, 1]])
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    R = Rx @ Ry
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    # world floor point (X, cam_h, Z) in camera-y-down coords -> image
    M = np.array([[1, 0, 0], [0, 0, cam_h], [0, 1, 0]])
    Hm = K @ R @ M @ A
    img = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_AREA, borderValue=(235, 235, 235))
    # horizon: image of floor directions at infinity -> line through K R (1,0,0) and K R (0,0,1)
    v1, v2 = K @ R @ np.array([1, 0, 0]), K @ R @ np.array([0, 0, 1])
    hor = np.cross(v1, v2)
    hor = hor / math.hypot(hor[0], hor[1])
    return img, dict(f=f, horizon=hor.tolist())
