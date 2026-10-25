"""Frequency-domain statistics.

Natural photos have a power spectrum falling roughly as 1/f^2 (slope about -2 in log-log).
Decoder upsampling in generators can leave periodic peaks at fixed fractions of the
resolution. WhatsApp resizing blurs these, so this is a weak, supporting signal.
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import Result, gray_f


def spectral_analysis(img: np.ndarray, max_side: int = 1024) -> Result:
    g = gray_f(img)
    H, W = g.shape
    s = min(1.0, max_side / max(H, W))
    if s < 1:
        g = cv2.resize(g, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    n = min(g.shape)
    g = g[:n, :n] - g[:n, :n].mean()
    win = np.outer(np.hanning(n), np.hanning(n))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2 + 1e-9
    yy, xx = np.indices(P.shape)
    r = np.hypot(xx - n // 2, yy - n // 2) / n            # cycles per pixel
    band = (r > 0.03) & (r < 0.35)
    rb = np.round(r[band] * 200).astype(int)
    radial = np.bincount(rb, np.log(P[band])) / np.maximum(np.bincount(rb), 1)
    freqs = np.arange(len(radial)) / 200
    ok = np.bincount(rb) > 0
    slope, icpt = np.polyfit(np.log(freqs[ok]), radial[ok], 1)

    # residual peaks after removing the radial trend; ignore the axes and JPEG 1/8 harmonics
    model = icpt + slope * np.log(np.maximum(r, 1e-3))
    res = np.log(P) - model
    res[np.abs(xx - n // 2) < 3] = 0
    res[np.abs(yy - n // 2) < 3] = 0
    res[~band] = 0
    for k in range(1, 8):
        for cyc in (k / 8,):
            for sgn in (-1, 1):
                c = int(n // 2 + sgn * cyc * n)
                res[:, max(c - 2, 0):c + 3] = 0
                res[max(c - 2, 0):c + 3, :] = 0
    med, mad = np.median(res[band]), 1.4826 * np.median(np.abs(res[band] - np.median(res[band])))
    z = (res - med) / (mad + 1e-9)
    loc = (z == cv2.dilate(z.astype(np.float32), np.ones((9, 9), np.uint8))) & (z > 6)
    peaks = [dict(fx=float((x - n // 2) / n), fy=float((y - n // 2) / n), z=float(z[y, x]))
             for y, x in zip(*np.nonzero(loc)) if y >= n // 2][:10]
    peaks.sort(key=lambda p: -p["z"])
    val = dict(slope=float(slope), peaks=peaks, analysed_size=n)
    notes = ["Weak signal after resizing/recompression; never decisive on its own.",
             "Textures with regular patterns (fabric, tiles, screens) also create peaks."]
    if len(peaks) >= 4:
        return Result("spectral_analysis", True, "low", "generated", val,
                      f"{len(peaks)} off-axis periodic peaks not explained by JPEG - possible upsampling "
                      "artefacts (or a regular pattern in the scene).", notes)
    if -2.8 < slope < -1.6:
        return Result("spectral_analysis", True, "low", "camera", val,
                      f"Spectral slope {slope:.2f} is in the natural-image range; no periodic peaks.", notes)
    return Result("spectral_analysis", True, "low", "neutral", val,
                  f"Spectral slope {slope:.2f} is outside the usual range; inconclusive.", notes)
