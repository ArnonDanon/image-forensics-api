"""Physics tools: test whether the image is consistent with a real lens + sensor
looking at a real 3D scene. Each tool says when it does NOT apply.

Coordinates are original image pixels, origin top-left.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .common import Result, gray_f, phone_focal_px, poly_mask

# --------------------------------------------------------------------------- aspect ratio
CAMERA_RATIOS = {"4:3": 4 / 3, "3:2": 1.5, "16:9": 16 / 9}
SCREEN_RATIOS = {"19.5:9 screenshot": 19.5 / 9, "20:9 screenshot": 20 / 9}
# Ratios that cameras do not produce but generator presets do.
GENERATOR_RATIOS = {
    "1:1 (most generators' default)": 1.0,
    "SDXL 1152x896": 1152 / 896,
    "SDXL 1216x832": 1216 / 832,
    "SDXL/DALL-E 1344x768 / 1792x1024": 1.75,
    "SDXL 1536x640": 1536 / 640,
    "Midjourney 16:9 1456x816": 1456 / 816,
    "Midjourney 4:3 1232x928": 1232 / 928,
}


def aspect_ratio_match(w: int, h: int) -> Result:
    r = max(w, h) / min(w, h)
    tol = 1.5 / min(w, h) + 0.002            # rounding after resize (e.g. WhatsApp 1600px)
    def hits(table):
        return [k for k, v in table.items() if abs(r - v) <= tol * v]
    cam, gen, scr = hits(CAMERA_RATIOS), hits(GENERATOR_RATIOS), hits(SCREEN_RATIOS)
    val = dict(width=w, height=h, ratio=r, tolerance=tol, camera_match=cam, generator_match=gen, screenshot_match=scr)
    if gen and not cam:
        sq = gen == ["1:1 (most generators' default)"]
        return Result("aspect_ratio_match", True, "low" if sq else "medium", "generated", val,
                      f"Ratio {r:.4f} matches a generator preset ({', '.join(gen)}) and no camera sensor ratio.",
                      ["Cropping can produce any ratio; a square image may simply be a crop."])
    if cam:
        return Result("aspect_ratio_match", True, "low", "camera", val,
                      f"Ratio {r:.4f} is a standard camera ratio ({', '.join(cam)}).",
                      ["Weak: many generators (Gemini, Imagen, GPT-image) also offer 4:3, 3:2 and 16:9,"
                       " and AI edits keep the original photo's ratio."])
    if scr:
        return Result("aspect_ratio_match", True, "low", "neutral", val,
                      "Phone-screen ratio: likely a screenshot; the content inside could be anything.")
    return Result("aspect_ratio_match", True, "none", "neutral", val,
                  f"Ratio {r:.4f} matches nothing known - probably cropped. Not informative.")


# --------------------------------------------------------------------------- highlight clipping
def clipping_test(img: np.ndarray, min_area: int = 25) -> Result:
    """Real sensors saturate: blown highlights become FLAT plateaus, and the colour channels
    clip at different brightness, leaving a ring of partially clipped pixels around them.
    Generators tend to render highlights as smooth roll-offs that never truly flatten."""
    H, W = img.shape[:2]
    mx = img.max(axis=2)
    bright = (mx >= 235)
    if bright.mean() < 0.0005:
        return Result("clipping_test", False, "none", "neutral", dict(bright_fraction=float(bright.mean())),
                      "No bright highlights - test does not apply.")
    full = (img.min(axis=2) >= 250).astype(np.uint8)      # all channels clipped
    n, lbl, st, _ = cv2.connectedComponentsWithStats(full, connectivity=8)
    lum = gray_f(img)
    plateaus = []
    for i in range(1, n):
        if st[i, 4] < min_area:
            continue
        m = (lbl == i)
        core = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        flat_std = float(lum[core].std()) if core.sum() > 5 else float(lum[m].std())
        ring = (cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~m
        rp = img[ring]
        nclip = (rp >= 250).sum(axis=1)
        partial = float(((nclip == 1) | (nclip == 2)).mean()) if len(rp) else 0.0
        plateaus.append(dict(box=[int(v) for v in st[i, :4]], area=int(st[i, 4]),
                             flatness_std=flat_std, partial_clip_ring=partial))
    plateaus.sort(key=lambda p: -p["area"])
    bright_area = int((mx >= 245).sum())
    clipped_area = int(sum(p["area"] for p in plateaus))
    val = dict(bright_px=bright_area, clipped_plateau_px=clipped_area,
               plateau_fraction_of_bright=clipped_area / max(bright_area, 1), plateaus=plateaus[:10])
    if not plateaus and bright_area > 400:
        return Result("clipping_test", True, "low", "generated", val,
                      "Bright highlights exist but none are truly clipped flat plateaus - a smooth roll-off "
                      "more typical of rendering than of a sensor.",
                      ["Well-exposed real photos (or HDR phone shots) can also avoid clipping."])
    if plateaus:
        flat = [p for p in plateaus if p["flatness_std"] < 1.5]
        ring = np.mean([p["partial_clip_ring"] for p in plateaus[:5]])
        if flat and ring > 0.1:
            return Result("clipping_test", True, "medium", "camera", val,
                          f"{len(flat)} flat clipped plateau(s) with per-channel clipping around them "
                          f"(ring {ring:.0%}) - consistent with sensor saturation.",
                          ["Some generators now output clipped whites (sky, lamps); treat as supporting evidence."])
        return Result("clipping_test", True, "low", "neutral", val,
                      "Clipped areas exist but lack the per-channel clipping ring typical of sensors.")
    return Result("clipping_test", True, "none", "neutral", val, "Too few bright pixels to judge.")


# --------------------------------------------------------------------------- dynamic range
def dynamic_range_test(img: np.ndarray, tile: int = 16) -> Result:
    """A single exposure cannot keep full detail in deep shadows AND bright highlights.
    Measures texture in the darkest and brightest tiles relative to mid-tones."""
    g = gray_f(img)
    gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    H, W = g.shape
    rows = []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            rows.append((g[y:y + tile, x:x + tile].mean(), np.percentile(gm[y:y + tile, x:x + tile], 75)))
    a = np.array(rows)
    lum, det = a[:, 0], a[:, 1]
    lo, hi = np.percentile(lum, [5, 95])
    if hi - lo < 120:
        return Result("dynamic_range_test", False, "none", "neutral", dict(p5=lo, p95=hi),
                      "Scene has low contrast - test does not apply.")
    dark, light = det[lum <= lo], det[(lum >= hi) & (lum < 245)]
    mid = det[(lum > np.percentile(lum, 40)) & (lum < np.percentile(lum, 60))]
    rd, rl = float(np.median(dark) / (np.median(mid) + 1e-3)), float(np.median(light) / (np.median(mid) + 1e-3)) if len(light) else 0.0
    val = dict(p5_luminance=lo, p95_luminance=hi, shadow_detail_vs_mid=rd, highlight_detail_vs_mid=rl)
    if rd > 0.9 and rl > 0.9:
        return Result("dynamic_range_test", True, "low", "generated", val,
                      "Shadows and highlights both keep as much detail as mid-tones - unusual for one exposure.",
                      ["Phone HDR merges exposures and can produce this too."])
    return Result("dynamic_range_test", True, "low", "camera" if min(rd, rl) < 0.5 else "neutral", val,
                  "Detail falls off in shadows and/or highlights as expected for a single exposure."
                  if min(rd, rl) < 0.5 else "Detail distribution is unremarkable.")


# --------------------------------------------------------------------------- vanishing points
def _segments(img: np.ndarray, min_len_frac: float = 0.04) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector()
    segs = lsd.detect(g)[0]
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4)
    L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[L >= min_len_frac * np.hypot(*g.shape)]


def _hline(s):
    return np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0])


def _consistent(segs: np.ndarray, vp: np.ndarray, deg: float) -> np.ndarray:
    """Angle between each segment and the line from its midpoint to the VP (homogeneous vp)."""
    mid = np.c_[(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2]
    d = np.c_[segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]]
    if abs(vp[2]) > 1e-9:
        t = vp[:2] / vp[2] - mid
    else:                                        # VP at infinity: direction only
        t = np.tile(vp[:2], (len(segs), 1))
    cos = np.abs((d * t).sum(1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1))) < deg


def _refine(segs: np.ndarray) -> np.ndarray:
    """Least-squares VP: smallest singular vector of stacked normalised lines."""
    Ls = np.array([_hline(s) for s in segs])
    Ls /= np.linalg.norm(Ls[:, :2], axis=1, keepdims=True)
    w = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    _, _, Vt = np.linalg.svd(Ls * w[:, None])
    return Vt[-1]


def vanishing_points(img: np.ndarray, max_vps: int = 3, deg: float = 1.5, iters: int = 1500,
                     seed: int = 0) -> Result:
    """Detect straight segments, find up to 3 vanishing points by RANSAC, then check what
    they imply about the camera: focal length and principal point. A real uncropped photo has
    its principal point near the image centre and a plausible focal length."""
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    segs = _segments(img)
    total = len(segs)
    if total < 20:
        return Result("vanishing_points", False, "none", "neutral", dict(segments=total),
                      "Too few straight lines - test does not apply.")
    remaining = segs.copy()
    vps = []
    for _ in range(max_vps):
        if len(remaining) < 8:
            break
        best, best_in = None, None
        lines = np.array([_hline(s) for s in remaining])
        for _ in range(iters):
            i, j = rng.choice(len(remaining), 2, replace=False)
            vp = np.cross(lines[i], lines[j])
            if np.linalg.norm(vp) < 1e-9:
                continue
            inl = _consistent(remaining, vp, deg)
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = vp, inl
        if best_in is None or best_in.sum() < 8:
            break
        vp = _refine(remaining[best_in])
        inl = _consistent(remaining, vp, deg)
        ang = np.degrees(np.arctan2(remaining[inl][:, 3] - remaining[inl][:, 1],
                                    remaining[inl][:, 2] - remaining[inl][:, 0])) % 180
        vertical = float(np.median(np.minimum(np.abs(ang - 90), 180)) < 20)
        fin = abs(vp[2]) > 1e-9 * np.linalg.norm(vp[:2])
        pt = (vp[:2] / vp[2]).tolist() if fin else None
        vps.append(dict(point=pt, direction=None if fin else (vp[:2] / np.linalg.norm(vp[:2])).tolist(),
                        inliers=int(inl.sum()), mostly_vertical=bool(vertical), _h=vp))
        remaining = remaining[~inl]

    explained = 1 - len(remaining) / total
    out = dict(segments=total, explained_fraction=explained,
               vps=[{k: v for k, v in d.items() if k != "_h"} for d in vps])
    pp = np.array([W / 2, H / 2])
    notes = ["Cropping moves the principal point; a crop alone can fail the centre check.",
             "Curved or organic scenes have no reliable lines; the test then reports 'none'."]

    finite = [d for d in vps if d["point"] is not None]
    # horizon from the non-vertical VPs
    horiz = [d for d in vps if not d["mostly_vertical"]]
    if len(horiz) >= 2:
        hl = np.cross(horiz[0]["_h"], horiz[1]["_h"])
        hl = hl / (np.hypot(hl[0], hl[1]) + 1e-12)
        if hl[1] < 0:
            hl = -hl
        out["horizon_line"] = hl.tolist()                 # a*x + b*y + c = 0
        out["horizon_y_at_centre"] = float(-(hl[0] * W / 2 + hl[2]) / hl[1]) if abs(hl[1]) > 1e-9 else None

    # focal from pairs of finite VPs assumed orthogonal, principal point = image centre
    focals = []
    for a in range(len(finite)):
        for b in range(a + 1, len(finite)):
            v1, v2 = np.array(finite[a]["point"]) - pp, np.array(finite[b]["point"]) - pp
            f2 = -float(v1 @ v2)
            focals.append(dict(pair=[a, b], f_px=math.sqrt(f2) if f2 > 0 else None, f2=f2))
    out["focal_estimates"] = focals
    f26 = phone_focal_px(W, H, 26)
    out["phone_main_focal_px"] = f26
    # principal point from three finite, mutually orthogonal VPs = orthocentre of the triangle
    if len(finite) == 3:
        A, B, C = (np.array(d["point"]) for d in finite)
        M = np.array([C - B, C - A])
        rhs = np.array([(C - B) @ A, (C - A) @ B])
        try:
            orth = np.linalg.solve(M, rhs)
            off = float(np.linalg.norm(orth - pp) / np.hypot(W, H))
            out["principal_point"] = orth.tolist()
            out["principal_point_offset_frac"] = off
        except np.linalg.LinAlgError:
            pass

    good_f = [f["f_px"] for f in focals if f["f_px"]]
    impossible = [f for f in focals if f["f_px"] is None]
    rel = "high" if len(vps) >= 3 and min(d["inliers"] for d in vps) >= 15 else \
          "medium" if len(vps) >= 2 and explained > 0.4 else "low"
    if len(vps) < 2:
        return Result("vanishing_points", True, "low", "neutral", out,
                      "Only one consistent vanishing direction - cannot test camera geometry.", notes)
    msgs, leaning = [], "neutral"
    if good_f:
        eq = [f / f26 * 26 for f in good_f]
        out["focal_35mm_equiv"] = eq
        plaus = [e for e in eq if 12 <= e <= 130]
        msgs.append(f"VP pairs imply focal ≈ {', '.join(f'{e:.0f}' for e in eq)} mm-equiv")
        leaning = "camera" if plaus else "generated"
    if impossible and not good_f:
        msgs.append("vanishing points are not compatible with ANY real focal length")
        leaning = "generated"
    if "principal_point_offset_frac" in out:
        off = out["principal_point_offset_frac"]
        msgs.append(f"principal point {off:.0%} of diagonal from centre")
        if off > 0.2:
            leaning = "generated" if leaning != "camera" else "neutral"
    msgs.append(f"{explained:.0%} of straight segments explained by {len(vps)} VPs")
    note_pairs = "Focal assumes the two VP directions are perpendicular in 3D (true for most buildings/rooms)."
    return Result("vanishing_points", True, rel, leaning, out, "; ".join(msgs) + ".", notes + [note_pairs])


# --------------------------------------------------------------------------- floor texture
def floor_test(img: np.ndarray, floor_polygons: list[list[list[float]]],
               horizon_line: list[float] | None = None, exclude_boxes: list[list[int]] | None = None,
               diff_thresh: float = 20.0) -> Result:
    """Random texture on a flat floor (terrazzo, gravel, carpet speckle) must be squashed by
    perspective by an exact, predictable amount. From the squash and the horizon we solve for
    the focal length; a real photo gives a focal length a real camera has.

    floor_polygons: regions of visible textured floor ([[x,y],...]).
    horizon_line: [a,b,c] with a*x+b*y+c=0. If omitted, taken from vanishing_points()."""
    H, W = img.shape[:2]
    if horizon_line is None:
        vp = vanishing_points(img)
        horizon_line = vp.value.get("horizon_line")
        if horizon_line is None:
            return Result("floor_test", False, "none", "neutral", {},
                          "No horizon available (pass horizon_line, or the image needs two line directions).")
    a, b, c = horizon_line
    nrm = math.hypot(a, b)
    a, b, c = a / nrm, b / nrm, c / nrm
    if b < 0:
        a, b, c = -a, -b, -c
    mask = poly_mask(img.shape, floor_polygons)
    for bx in exclude_boxes or []:
        x, y, w, h = bx
        mask[y:y + h, x:x + w] = 0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    bg = np.stack([cv2.medianBlur(lab[..., i], 31) for i in range(3)], -1).astype(np.float32)
    d = np.sqrt(((lab.astype(np.float32) - bg) ** 2).sum(-1))
    bw = ((d > diff_thresh) & (mask > 0)).astype(np.uint8)
    n, lbl, st, cen = cv2.connectedComponentsWithStats(bw)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 30 or area > 1500:
            continue
        sub = mask[max(y - 1, 0):y + h + 1, max(x - 1, 0):x + w + 1]
        if (sub == 0).any():
            continue
        ys, xs = np.nonzero(lbl[y:y + h, x:x + w] == i)
        C = np.cov(np.vstack([xs, ys]))
        blobs.append((cen[i][0], cen[i][1], C / np.trace(C)))
    if len(blobs) < 15:
        return Result("floor_test", False, "none", "neutral", dict(blobs=len(blobs)),
                      "Not enough floor texture elements (need ~15+).")
    X = np.array([p[0] for p in blobs]); Y = np.array([p[1] for p in blobs])
    D = a * X + b * Y + c
    if (D <= 0).mean() > 0.1:
        return Result("floor_test", False, "none", "neutral", dict(blobs=len(blobs)),
                      "Floor region is not below the horizon - wrong region or wrong horizon.")
    Cs = [p[2] for p in blobs]

    def ratio_of(idx):
        S = sum(Cs[i] for i in idx)
        ev = np.linalg.eigvalsh(S)
        return float(np.sqrt(ev[0] / ev[1]))

    rng = np.random.default_rng(0)
    allidx = np.arange(len(blobs))
    ratio = ratio_of(allidx)
    boots = [ratio_of(rng.choice(allidx, len(allidx))) for _ in range(500)]
    lo, hi = np.percentile(boots, [5, 95])
    cx, cy = W / 2, H / 2
    d0 = a * cx + b * cy + c

    def pred(f):
        rr = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + f ** 2)
        return float(np.mean(D * f / (math.sqrt(f * f + d0 * d0) * rr)))
    fs = np.arange(100, 6000, 5.0)
    P = np.array([pred(f) for f in fs])

    def f_for(r):
        return float(fs[np.argmin(np.abs(P - r))])
    f_est, f_lo, f_hi = f_for(ratio), f_for(hi), f_for(lo)
    f26 = phone_focal_px(W, H, 26)
    eq = lambda f: f / f26 * 26
    val = dict(blobs=len(blobs), foreshortening_ratio=ratio, ratio_ci90=[lo, hi],
               implied_focal_px=f_est, focal_ci90_px=[f_lo, f_hi],
               implied_focal_35mm_equiv=eq(f_est), focal_ci90_35mm_equiv=[eq(f_lo), eq(f_hi)],
               horizon_line=[a, b, c])
    width = eq(f_hi) - eq(f_lo)
    rel = "low" if len(blobs) < 40 or width > 30 else "medium"
    ok = eq(f_hi) >= 12 and eq(f_lo) <= 130
    lean = "camera" if ok else "generated"
    interp = (f"Floor texture squash implies a {eq(f_est):.0f} mm-equiv lens "
              f"(90% range {eq(f_lo):.0f}-{eq(f_hi):.0f}); "
              + ("a real camera can have this." if ok else "no normal camera has this - floor geometry is inconsistent."))
    return Result("floor_test", True, rel, lean, val, interp,
                  ["Assumes texture elements are randomly oriented on a flat floor.",
                   "Compression blur rounds small chips and biases the focal estimate slightly low.",
                   "Only tests the floor: an AI edit elsewhere in a real photo will pass."])
