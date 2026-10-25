import numpy as np

from imgforensics import physics
from synthetic import render_floor


def test_vanishing_points_recover_focal():
    img, gt = render_floor(chips=False)
    r = physics.vanishing_points(img)
    assert r.applicable
    fs = [e["f_px"] for e in r.value["focal_estimates"] if e["f_px"]]
    assert fs, r.value
    assert abs(max(fs, key=lambda f: -abs(f - gt["f"])) - gt["f"]) / gt["f"] < 0.1


import pytest


@pytest.mark.parametrize("f,pitch,yaw", [(900, 12, 30), (700, 20, 10), (1200, 8, 45), (600, 25, 0)])
def test_floor_test_recovers_focal_with_true_horizon(f, pitch, yaw):
    img, gt = render_floor(f=f, pitch_deg=pitch, yaw_deg=yaw)
    H, W = img.shape[:2]
    poly = [[[0, int(H * 0.6)], [W, int(H * 0.6)], [W, H], [0, H]]]
    r = physics.floor_test(img, poly, horizon_line=gt["horizon"])
    assert r.applicable, r.interpretation
    est = r.value["implied_focal_px"]
    assert abs(est - f) / f < 0.12, r.value


def test_aspect_ratio_tables():
    assert physics.aspect_ratio_match(4000, 3000).leaning == "camera"
    assert physics.aspect_ratio_match(1216, 832).leaning == "generated"
    assert physics.aspect_ratio_match(1600, 1095).value["generator_match"]   # 1216x832 after WhatsApp resize
