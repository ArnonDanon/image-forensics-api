"""Run every automatic test and assemble an evidence ledger.

The report deliberately does NOT output a final verdict: it collects weighted evidence for
the caller (an LLM or a human) who must also inspect the image itself.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import physics, spectral
from .common import Result
from .consistency import region_stats_map

WEIGHT = {"none": 0.0, "low": 1.0, "medium": 2.0, "high": 3.0}


def full_report(img: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    H, W = img.shape[:2]
    t0 = time.perf_counter()
    results: list[Result] = []
    timings = {}

    def run(name, fn):
        t = time.perf_counter()
        try:
            out = fn()
        except Exception as e:  # a failing test must not break the report
            out = Result(name, False, "none", "neutral", {}, f"error: {e}")
        timings[name] = round(time.perf_counter() - t, 3)
        return out

    results.append(run("aspect_ratio_match", lambda: physics.aspect_ratio_match(W, H)))
    results.append(run("clipping_test", lambda: physics.clipping_test(img)))
    results.append(run("dynamic_range_test", lambda: physics.dynamic_range_test(img)))
    results.append(run("vanishing_points", lambda: physics.vanishing_points(img)))
    results.append(run("spectral_analysis", lambda: spectral.spectral_analysis(img)))
    heat_holder = {}

    def _cons():
        r, heat = region_stats_map(img)
        heat_holder["heat"] = heat
        return r
    results.append(run("region_stats_map", _cons))

    tally = {"camera": 0.0, "generated": 0.0, "edited": 0.0, "neutral": 0.0}
    for r in results:
        if r.applicable:
            tally[r.leaning] += WEIGHT[r.reliability]
    scene_says_camera = tally["camera"] > 2 * tally["generated"]
    summary = {
        "evidence_weight": tally,
        "scene_level": ("Scene geometry/optics look like a real camera capture." if scene_says_camera else
                        "Scene-level tests do not clearly support a real camera capture."),
        "caveat": ("Scene-level tests cannot detect AI edits inside a real photo (the edit inherits the "
                   "original geometry). Inspect the regions flagged by region_stats_map and zoom into "
                   "hands, faces, text and object boundaries before deciding."),
    }
    report = dict(image=dict(width=W, height=H), summary=summary,
                  results=[r.to_dict() for r in results],
                  timings_s=timings, total_s=round(time.perf_counter() - t0, 3))
    return report, heat_holder.get("heat", img)
