"""REST API. Run: uvicorn imgforensics.api:app --host 0.0.0.0 --port 8000
Interactive docs at /docs."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import __version__, physics, see, spectral
from .common import png_bytes
from .consistency import compare_regions, region_stats_map
from .report import full_report
from .store import ImageStore

app = FastAPI(title="Image Forensics API", version=__version__,
              description="Deterministic measurements that help an LLM or analyst decide whether an image "
                          "is a real camera photo, AI-generated, or AI-edited. Every result carries "
                          "`applicable`, `reliability` and `leaning`; no endpoint returns a final verdict.")


@lru_cache
def get_store() -> ImageStore:
    return ImageStore()


def _img(image_id: str, store: ImageStore = Depends(get_store)):
    try:
        return store.get(image_id)
    except KeyError:
        raise HTTPException(404, f"unknown image_id {image_id}")


def _png(arr) -> Response:
    return Response(png_bytes(arr), media_type="image/png")


_reports: dict[str, dict] = {}


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/images", summary="Upload an image; returns its id (sha256-based, idempotent)")
async def upload(file: UploadFile = File(...), store: ImageStore = Depends(get_store)):
    data = await file.read()
    try:
        image_id = store.put(data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    h, w = store.get(image_id).shape[:2]
    return {"image_id": image_id, "width": w, "height": h}


@app.get("/images/{image_id}/report", summary="Run every automatic test; evidence ledger, no verdict")
def report(image_id: str, img=Depends(_img)):
    if image_id not in _reports:
        rep, _ = full_report(img)
        _reports[image_id] = rep
    return _reports[image_id]


# ---- seeing
@app.get("/images/{image_id}/overview.png", summary="Whole image with a coordinate grid")
def overview(img=Depends(_img), max_side: int = Query(1400, ge=200, le=4000), grid_step: int | None = None):
    return _png(see.overview(img, max_side, grid_step))


@app.get("/images/{image_id}/zoom.png", summary="Full-resolution crop, nearest-neighbour upscaled")
def zoom(img=Depends(_img), x: int = 0, y: int = 0, w: int = 256, h: int = 256,
         scale: float | None = Query(None, gt=0, le=16), grid: bool = True):
    try:
        return _png(see.zoom(img, x, y, w, h, scale, grid))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/images/{image_id}/enhance.png", summary="Alternative renderings that expose seams and texture breaks")
def enhance(img=Depends(_img), mode: Literal[see.ENHANCE_MODES] = "highpass",  # type: ignore[valid-type]
            x: int = 0, y: int = 0, w: int | None = None, h: int | None = None):
    try:
        return _png(see.enhance(img, mode, x, y, w, h))
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---- consistency
@app.get("/images/{image_id}/consistency")
def consistency(img=Depends(_img), tile: int = Query(32, ge=8, le=256), z_thresh: float = 3.0):
    return region_stats_map(img, tile, z_thresh)[0].to_dict()


@app.get("/images/{image_id}/consistency.png", summary="Heatmap: red = disagrees with the rest of the image")
def consistency_png(img=Depends(_img), tile: int = Query(32, ge=8, le=256), z_thresh: float = 3.0):
    return _png(region_stats_map(img, tile, z_thresh)[1])


class CompareBody(BaseModel):
    boxes: list[list[int]] = Field(..., description="[[x,y,w,h],...]; the FIRST is the reference")
    names: list[str] | None = None


@app.post("/images/{image_id}/compare_regions")
def compare(body: CompareBody, img=Depends(_img)):
    if len(body.boxes) < 2:
        raise HTTPException(400, "need at least 2 boxes")
    return compare_regions(img, body.boxes, body.names).to_dict()


# ---- physics
@app.get("/images/{image_id}/aspect_ratio")
def aspect(img=Depends(_img)):
    h, w = img.shape[:2]
    return physics.aspect_ratio_match(w, h).to_dict()


@app.get("/images/{image_id}/clipping")
def clipping(img=Depends(_img)):
    return physics.clipping_test(img).to_dict()


@app.get("/images/{image_id}/dynamic_range")
def dyn_range(img=Depends(_img)):
    return physics.dynamic_range_test(img).to_dict()


@app.get("/images/{image_id}/vanishing_points")
def vps(img=Depends(_img)):
    return physics.vanishing_points(img).to_dict()


class FloorBody(BaseModel):
    floor_polygons: list[list[list[float]]] = Field(..., description="[[[x,y],...],...] visible textured floor")
    horizon_line: list[float] | None = Field(None, description="[a,b,c] with a*x+b*y+c=0; default from vanishing points")
    exclude_boxes: list[list[int]] | None = None


@app.post("/images/{image_id}/floor_test")
def floor(body: FloorBody, img=Depends(_img)):
    return physics.floor_test(img, body.floor_polygons, body.horizon_line, body.exclude_boxes).to_dict()


@app.get("/images/{image_id}/spectral")
def spectrum(img=Depends(_img)):
    return spectral.spectral_analysis(img).to_dict()
