# image-forensics-api

Deterministic measurements that help an LLM (or a human analyst) decide whether an image is a
**real camera photo**, **fully AI-generated**, or a **real photo with AI edits** - including images
laundered through WhatsApp and similar apps.

The design splits the job:

- **The tools measure.** Classical code only: geometry, optics, sensor physics, statistics. No
  learned models. Every result reports whether it applies, how reliable it is, and which way it leans.
- **The caller judges.** An LLM with vision (or a person) combines the measurements with what it
  sees in zoomed crops - hands, text, anatomy, physics - and returns a verdict with evidence, or
  `cannot_determine`.

No endpoint returns a final verdict on its own. That is deliberate: no single deterministic test is
reliable enough on laundered images, and scene-level tests cannot see local edits.

## Result format

Every tool returns the same shape:

```json
{
  "tool": "vanishing_points",
  "applicable": true,
  "reliability": "high",          // none | low | medium | high
  "leaning": "camera",            // camera | generated | edited | neutral
  "value": { "...": "measurements" },
  "interpretation": "VP pairs imply focal ≈ 23, 24, 25 mm-equiv; principal point 1% of diagonal from centre; ...",
  "notes": ["when this test can mislead"]
}
```

## Tools

| Group | Tool | What it measures |
|---|---|---|
| Seeing | `overview`, `zoom`, `enhance` | Full-resolution crops with a pixel-coordinate grid; high-pass, noise residual, saturation, per-channel, local contrast and recompression views |
| Consistency | `consistency_map` | Regions whose noise, fine detail or saturation disagree with areas of similar brightness and texture (heatmap + boxes) |
| | `compare_regions` | Tests a hypothesis: "is this stain different from the clean shirt next to it?" |
| Physics | `vanishing_points` | Line segments → up to 3 vanishing points → implied focal length and principal point. A real uncropped photo gives a real lens and a centred principal point |
| | `floor_test` | Perspective squash of random floor texture → implied focal length (exact Jacobian model) |
| | `clipping_test` | Sensor saturation: flat plateaus with per-channel clipping rings |
| | `dynamic_range_test` | Detail kept in shadows and highlights at once (implausible for one exposure) |
| | `aspect_ratio_match` | Camera sensor ratios vs generator presets (survives resizing) |
| Statistics | `spectral_analysis` | Power-spectrum slope and periodic peaks (possible upsampling traces) |
| All | `report` | Runs every automatic test, returns an evidence ledger |

## Run it

```bash
pip install -e ".[api,mcp,dev]"

# REST API (docs at http://localhost:8000/docs)
uvicorn imgforensics.api:app --port 8000

# MCP server for an LLM (stdio)
python -m imgforensics.mcp_server
# or over HTTP
python -m imgforensics.mcp_server --http --host 0.0.0.0 --port 8001

# CLI
python -m imgforensics.cli report photo.jpg --out out/
python -m imgforensics.cli batch dataset/ --csv results.csv   # dataset/real/*.jpg, dataset/ai/*.jpg ...

# Docker
docker build -t image-forensics-api . && docker run -p 8000:8000 -v $PWD/data:/data image-forensics-api

pytest
```

Add to Claude Code: `claude mcp add image-forensics -- python -m imgforensics.mcp_server`

### REST endpoints

```
POST /images                                  upload -> {image_id, width, height}
GET  /images/{id}/report                      all automatic tests
GET  /images/{id}/overview.png                ?max_side&grid_step
GET  /images/{id}/zoom.png                    ?x&y&w&h&scale&grid
GET  /images/{id}/enhance.png                 ?mode=highpass|contrast|channels|saturation|noise|recompress&x&y&w&h
GET  /images/{id}/consistency                 JSON;  /consistency.png heatmap
POST /images/{id}/compare_regions             {"boxes": [[x,y,w,h],...], "names": [...]}
GET  /images/{id}/vanishing_points | clipping | dynamic_range | aspect_ratio | spectral
POST /images/{id}/floor_test                  {"floor_polygons": [[[x,y],...]], "horizon_line": [a,b,c]?, "exclude_boxes": [...]}
GET  /health
```

## Recommended LLM workflow

The MCP server ships these instructions to the model:

1. `load_image` → `run_report`.
2. `overview`, then `consistency_map`; zoom into flagged regions and into hands, faces, teeth,
   eyes, text, jewellery, object boundaries, reflections and corners.
3. `enhance` suspicious crops; `compare_regions` against a same-material reference.
4. `floor_test` / `vanishing_points` where the scene allows.
5. Verdict: `camera | ai_generated | ai_edited | cannot_determine`, confidence, edited regions,
   and evidence citing a tool or crop for each point. Several independent findings must agree.

## Validation so far

- **Synthetic ground truth** (`tests/synthetic.py` renders a floor through a known pinhole camera):
  vanishing points recover the focal length exactly; `floor_test` recovers it within ~5-10%
  across pitch/yaw/focal combinations (slight low bias from rasterisation blur).
- **One real WhatsApp image** (an office photo with an AI edit on the person): vanishing points give
  23-25 mm-equiv with the principal point 1% from centre, and the floor test independently gives
  23 mm (21-26). Both match a phone main camera, which is correct: the scene is a real photo.
  The consistency map concentrates on the edited person, but also flags some unedited areas.
  This is exactly the case scene-level tests cannot settle on their own.

**Accuracy on real data is unknown until measured.** Use `cli batch` on a labelled, WhatsApp-laundered
set and check accuracy per tool and per verdict type (generated vs edited) before trusting thresholds.

## Limitations

- Scene-level physics passes for AI edits of real photos, by design.
- Cropping breaks principal-point and aspect-ratio checks.
- Thresholds are hand-set first guesses, not calibrated.
- Generators improve; physics-based checks age slower than texture statistics, but they do age.

## Roadmap

- Light-direction consistency across objects (shading + cast shadows)
- Corneal and eyeglass reflections/refraction
- Mains-flicker banding (50/100 Hz) under artificial light
- Patch-recurrence search for repeated synthetic micro-texture
- Automatic floor segmentation; reverse-image-search adapter for finding originals
