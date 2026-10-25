"""Consistency tools: a real photo is ONE capture, so every region should share one
"process signature". Regions that disagree are candidates for editing.

Signals per tile:
  noise      - residual noise in the flattest pixels of the tile, normalised for brightness
               (sensor noise grows with brightness; we fit that curve from the image itself)
  fine_ratio - fine-scale vs mid-scale detail; regenerated areas often have a different
               balance (too smooth, or synthetic micro-texture)
  saturation - mean HSV saturation relative to similar-brightness tiles
These are RELATIVE measures, so uniform WhatsApp recompression mostly cancels out.
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import Result, clamp_box, gray_f

_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _tile_stats(gray: np.ndarray, sat: np.ndarray, gmag: np.ndarray, x: int, y: int, t: int) -> dict:
    g = gray[y:y + t, x:x + t]
    m = gmag[y:y + t, x:x + t]
    resp = np.abs(cv2.filter2D(g, -1, _NOISE_K))[1:-1, 1:-1]
    mm = m[1:-1, 1:-1]
    flat = mm <= np.percentile(mm, 40)
    noise = float(np.sqrt(np.pi / 2) * resp[flat].mean() / 6.0) if flat.any() else np.nan
    fine = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    mid = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 3.0)
    fine_ratio = float(np.std(fine) / (np.std(mid) + 1e-3))
    return dict(x=x, y=y, lum=float(g.mean()), noise=noise, fine_ratio=fine_ratio,
                sat=float(sat[y:y + t, x:x + t].mean()), detail=float(np.percentile(m, 90)))


def _qbin(a: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    return np.clip(np.searchsorted(edges, a, side="right") - 1, 0, bins - 1)


def _robust_z(values: np.ndarray, lum: np.ndarray, detail: np.ndarray,
              floor: float, lum_bins: int = 6, det_bins: int = 3) -> np.ndarray:
    """z-score of log(value) against tiles of similar brightness AND similar texture level,
    so flat walls are compared with flat walls, not with fabric."""
    v = np.log(np.maximum(values, floor))
    z = np.full(v.shape, np.nan)
    idx = _qbin(lum, lum_bins) * det_bins + _qbin(detail, det_bins)
    gmad = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))) + 1e-6
    for b in range(lum_bins * det_bins):
        sel = (idx == b) & np.isfinite(v)
        if sel.sum() < 6:
            continue
        med = np.median(v[sel])
        mad = max(1.4826 * np.median(np.abs(v[sel] - med)), 0.5 * gmad, 0.15)  # log units; guards flat-noise images
        z[sel] = (v[sel] - med) / mad
    return z


def region_stats_map(img: np.ndarray, tile: int = 32, z_thresh: float = 3.0):
    """Returns (Result, heatmap_bgr). Heatmap: red = stands out from the rest of the image."""
    H, W = img.shape[:2]
    gray = gray_f(img)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gmag = cv2.magnitude(gx, gy)

    tiles = [_tile_stats(gray, sat, gmag, x, y, tile)
             for y in range(0, H - tile + 1, tile) for x in range(0, W - tile + 1, tile)]
    lum = np.array([t["lum"] for t in tiles])
    det = np.array([t["detail"] for t in tiles])
    # clipped or textureless tiles carry no noise information (JPEG zeroes noise in flat areas)
    usable = (lum > 20) & (lum < 235) & (det > np.percentile(det, 10))
    floors = {"noise": 0.5, "fine_ratio": 0.05, "sat": 4.0}
    zs = {}
    for key in ("noise", "fine_ratio", "sat"):
        vals = np.array([t[key] for t in tiles], float)
        z = np.full(len(tiles), np.nan)
        z[usable] = _robust_z(vals[usable], lum[usable], det[usable], floors[key])
        zs[key] = z

    stack = np.abs(np.vstack([zs[k] for k in zs]))
    score = np.where(np.isfinite(stack).any(0), np.nanmax(np.nan_to_num(stack, nan=0.0), axis=0), 0.0)
    gh, gw = H // tile, W // tile
    grid = score[: gh * gw].reshape(gh, gw)

    # merge flagged tiles into regions
    flagged = (grid >= z_thresh).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(flagged, connectivity=8)
    regions = []
    for i in range(1, n):
        cells = np.argwhere(lbl == i)
        ids = [r * gw + c for r, c in cells]
        mean_z = {k: float(np.nanmean(zs[k][ids])) for k in zs}
        top = max(mean_z, key=lambda k: abs(mean_z[k]) if np.isfinite(mean_z[k]) else 0)
        regions.append(dict(
            box=[int(st[i, 0] * tile), int(st[i, 1] * tile), int(st[i, 2] * tile), int(st[i, 3] * tile)],
            tiles=int(st[i, 4]), max_z=float(grid[lbl == i].max()),
            mean_z=mean_z, strongest_signal=top,
            direction=("higher" if mean_z[top] > 0 else "lower") + f" {top} than similar-brightness areas",
        ))
    regions.sort(key=lambda r: -r["max_z"] * r["tiles"])

    frac = float((grid >= z_thresh).mean())
    n_usable = int(usable.sum())
    rel = "none" if n_usable < 30 else ("low" if n_usable < 120 else "medium")
    if not regions:
        leaning, interp = "neutral", "No region clearly disagrees with the rest of the image."
    elif frac > 0.25:
        leaning = "neutral"
        interp = ("Many regions disagree - the image is globally heterogeneous (strong content variation, "
                  "heavy processing, or fully synthetic). Not a localised-edit signal.")
    else:
        leaning = "edited"
        interp = (f"{len(regions)} localised region(s) disagree with the rest. Zoom into them; content-driven "
                  "causes (skin vs fabric, motion blur, bokeh) must be ruled out before calling it an edit.")

    heat = _heatmap(img, grid, tile, z_thresh)
    res = Result("region_stats_map", n_usable >= 30, rel, leaning,
                 dict(tile=tile, z_thresh=z_thresh, flagged_fraction=frac, usable_tiles=n_usable,
                      regions=regions[:12]),
                 interp,
                 ["Relative test: compares the image against itself, so uniform recompression cancels.",
                  "Different materials legitimately differ in texture; treat a flag as 'look here', not proof."])
    return res, heat


def _heatmap(img, grid, tile, z_thresh):
    H, W = img.shape[:2]
    norm = np.clip(grid / (2 * z_thresh), 0, 1)
    up = cv2.resize((norm * 255).astype(np.uint8), (grid.shape[1] * tile, grid.shape[0] * tile),
                    interpolation=cv2.INTER_NEAREST)
    full = np.zeros((H, W), np.uint8)
    full[: up.shape[0], : up.shape[1]] = up
    col = cv2.applyColorMap(full, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.55, col, 0.45, 0)
    flagged = (full >= 127).astype(np.uint8)
    cnts, _ = cv2.findContours(flagged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
    return out


def compare_regions(img: np.ndarray, boxes: list[list[int]], names: list[str] | None = None) -> Result:
    """Measure several regions side by side to test a hypothesis such as
    'the chocolate stain differs from the surrounding shirt'.
    boxes: [[x, y, w, h], ...]. Put a reference region of similar brightness/material first."""
    gray = gray_f(img)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    gmag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    names = names or [f"region_{i}" for i in range(len(boxes))]
    stats = []
    for name, b in zip(names, boxes):
        x, y, w, h = clamp_box(img, *b)
        t = 16
        cells = [_tile_stats(gray, sat, gmag, xx, yy, t)
                 for yy in range(y, y + h - t + 1, t) for xx in range(x, x + w - t + 1, t)]
        if not cells:
            stats.append(dict(name=name, box=[x, y, w, h], error="region smaller than 16px"))
            continue
        agg = {k: float(np.nanmedian([c[k] for c in cells])) for k in ("lum", "noise", "fine_ratio", "sat", "detail")}
        spread = {k + "_iqr": float(np.subtract(*np.nanpercentile([c[k] for c in cells], [75, 25])))
                  for k in ("noise", "fine_ratio")}
        stats.append(dict(name=name, box=[x, y, w, h], cells=len(cells), **agg, **spread))
    diffs = []
    ref = stats[0]
    for s in stats[1:]:
        if "error" in s or "error" in ref:
            continue
        d = {}
        for k in ("noise", "fine_ratio", "sat", "detail"):
            d[k + "_ratio_vs_ref"] = float(s[k] / (ref[k] + 1e-6))
        d["lum_diff"] = s["lum"] - ref["lum"]
        diffs.append(dict(name=s["name"], **d))
    big = [d["name"] for d in diffs
           if (abs(np.log(d["noise_ratio_vs_ref"] + 1e-6)) > np.log(1.6)
               or abs(np.log(d["fine_ratio_ratio_vs_ref"] + 1e-6)) > np.log(1.4))
           and abs(d["lum_diff"]) < 40]
    interp = ("Regions " + ", ".join(big) + " differ from the reference in noise/fine-detail at similar brightness."
              if big else "No region differs strongly from the reference at similar brightness.")
    return Result("compare_regions", True, "low" if len(boxes) < 3 else "medium",
                  "edited" if big else "neutral", dict(regions=stats, vs_reference=diffs), interp,
                  ["Only meaningful when regions have similar brightness and similar material.",
                   "Ratios beyond ~1.6x (noise) or ~1.4x (fine detail) are worth a closer look."])
