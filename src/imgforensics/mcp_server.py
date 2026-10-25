"""MCP server exposing the forensic tools to an LLM.

Run (stdio):   python -m imgforensics.mcp_server
Run (HTTP):    python -m imgforensics.mcp_server --http --port 8001
"""
from __future__ import annotations

import argparse
import json

from mcp.server.mcpserver import Image, MCPServer

from . import physics, see, spectral
from .common import png_bytes
from .consistency import compare_regions as _compare, region_stats_map
from .report import full_report
from .store import ImageStore

INSTRUCTIONS = """\
You are deciding whether an image is a real camera photo, fully AI-generated, or a real photo with
AI edits. These tools MEASURE; you JUDGE. Recommended workflow:

1. load_image -> image_id.
2. run_report: every automatic test. Note `reliability` (none/low/medium/high) and `leaning`.
3. overview, then consistency_map: zoom into flagged regions AND into hands, faces, teeth, eyes,
   text, jewellery, object boundaries, reflections and corners (watermarks, logos).
4. Use enhance (highpass / noise / saturation / channels) on suspicious crops, and compare_regions
   to test a specific hypothesis against a reference of similar brightness and material.
5. floor_test when there is visible textured floor; vanishing_points when there are straight lines.

Rules for your verdict:
- Scene-level tests (geometry, optics) passing does NOT rule out a local AI edit.
- A single 'low' result is never enough. Require several independent findings that agree.
- Answer with: verdict (camera | ai_generated | ai_edited | cannot_determine), confidence,
  edited_regions (boxes, if any), and an evidence list citing the tool or crop for each point.
- Prefer cannot_determine over a guess.
"""

mcp = MCPServer("image-forensics", instructions=INSTRUCTIONS)
store = ImageStore()


def _png(arr) -> Image:
    return Image(data=png_bytes(arr), format="png")


@mcp.tool()
def load_image(path: str) -> dict:
    """Load an image file from disk into the store. Returns image_id, width, height."""
    image_id = store.put_path(path)
    h, w = store.get(image_id).shape[:2]
    return {"image_id": image_id, "width": w, "height": h}


@mcp.tool()
def run_report(image_id: str) -> dict:
    """Run every automatic test (aspect ratio, clipping, dynamic range, vanishing points,
    spectrum, region consistency). Returns an evidence ledger, not a verdict."""
    return full_report(store.get(image_id))[0]


@mcp.tool()
def overview(image_id: str, max_side: int = 1400):
    """The whole image with a coordinate grid labelled in original pixels."""
    return _png(see.overview(store.get(image_id), max_side))


@mcp.tool()
def zoom(image_id: str, x: int, y: int, w: int, h: int, scale: float | None = None):
    """Full-resolution crop of [x,y,w,h], upscaled without interpolation, with coordinate grid.
    Use for hands, faces, text, edges, corners - anything small."""
    return _png(see.zoom(store.get(image_id), x, y, w, h, scale))


@mcp.tool()
def enhance(image_id: str, mode: str, x: int = 0, y: int = 0, w: int | None = None, h: int | None = None):
    """Alternative rendering of a region. mode: highpass | contrast | channels | saturation |
    noise | recompress. Seams, pasted colour and texture breaks often only show up here."""
    return _png(see.enhance(store.get(image_id), mode, x, y, w, h))


@mcp.tool()
def consistency_map(image_id: str, tile: int = 32, z_thresh: float = 3.0):
    """Which regions disagree with the rest of the image in noise, fine detail or saturation
    (compared with areas of similar brightness and texture). Returns JSON and a heatmap."""
    res, heat = region_stats_map(store.get(image_id), tile, z_thresh)
    return [json.dumps(res.to_dict(), indent=1), _png(heat)]


@mcp.tool()
def compare_regions(image_id: str, boxes: list[list[int]], names: list[str] | None = None) -> dict:
    """Compare regions [[x,y,w,h],...]; the FIRST is the reference. Use similar brightness and
    material, e.g. a clean part of a shirt vs a suspicious stain on the same shirt."""
    return _compare(store.get(image_id), boxes, names).to_dict()


@mcp.tool()
def vanishing_points(image_id: str) -> dict:
    """Straight-line geometry: vanishing points, horizon, implied focal length and principal
    point. A real uncropped photo gives a plausible lens and a centred principal point."""
    return physics.vanishing_points(store.get(image_id)).to_dict()


@mcp.tool()
def floor_test(image_id: str, floor_polygons: list[list[list[float]]],
               horizon_line: list[float] | None = None, exclude_boxes: list[list[int]] | None = None) -> dict:
    """Perspective squash of random floor texture -> implied focal length. floor_polygons are
    [[x,y],...] outlines of visible textured floor; exclude feet, objects and watermarks."""
    return physics.floor_test(store.get(image_id), floor_polygons, horizon_line, exclude_boxes).to_dict()


@mcp.tool()
def clipping_test(image_id: str) -> dict:
    """Sensor saturation signature in highlights (flat plateaus + per-channel clipping ring)."""
    return physics.clipping_test(store.get(image_id)).to_dict()


@mcp.tool()
def spectral_analysis(image_id: str) -> dict:
    """Power-spectrum slope and periodic peaks (possible upsampling traces). Weak signal."""
    return spectral.spectral_analysis(store.get(image_id)).to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    if a.http:
        mcp.run("streamable-http", host=a.host, port=a.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
