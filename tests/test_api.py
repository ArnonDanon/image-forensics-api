import cv2
from fastapi.testclient import TestClient

from imgforensics import api
from imgforensics.store import ImageStore
from synthetic import render_floor


def _client(tmp_path):
    api.get_store.cache_clear()
    api.app.dependency_overrides[api.get_store] = lambda: ImageStore(tmp_path)
    return TestClient(api.app)


def test_end_to_end(tmp_path):
    c = _client(tmp_path)
    img, _ = render_floor()
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    r = c.post("/images", files={"file": ("a.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 200
    iid = r.json()["image_id"]

    rep = c.get(f"/images/{iid}/report").json()
    assert {x["tool"] for x in rep["results"]} >= {"vanishing_points", "region_stats_map", "clipping_test"}
    for x in rep["results"]:
        assert x["reliability"] in ("none", "low", "medium", "high")

    for path in ("overview.png", "zoom.png?x=10&y=10&w=100&h=80", "enhance.png?mode=noise&x=0&y=600&w=300&h=300",
                 "consistency.png"):
        r = c.get(f"/images/{iid}/{path}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", path

    r = c.post(f"/images/{iid}/compare_regions", json={"boxes": [[0, 800, 200, 200], [500, 800, 200, 200]]})
    assert r.status_code == 200

    assert c.get("/images/doesnotexist/report").status_code == 404
    assert c.post("/images", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400
