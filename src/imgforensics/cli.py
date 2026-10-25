"""Command line: python -m imgforensics.cli report IMAGE [--out DIR]
Batch:        python -m imgforensics.cli batch FOLDER --csv results.csv  (one row per image, for calibration)"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from .common import load
from .report import full_report

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _report(path: Path, out: Path | None) -> dict:
    rep, heat = full_report(load(str(path)))
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{path.stem}.report.json").write_text(json.dumps(rep, indent=2))
        cv2.imwrite(str(out / f"{path.stem}.consistency.png"), heat)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(prog="imgforensics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="full report for one image")
    r.add_argument("image")
    r.add_argument("--out", type=Path)
    b = sub.add_parser("batch", help="flat feature table for a folder (label = sub-folder name)")
    b.add_argument("folder", type=Path)
    b.add_argument("--csv", type=Path, required=True)
    a = ap.parse_args()

    if a.cmd == "report":
        print(json.dumps(_report(Path(a.image), a.out), indent=2))
        return

    rows = []
    for p in sorted(a.folder.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        try:
            rep = _report(p, None)
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        row = {"file": str(p), "label": p.parent.name}
        for res in rep["results"]:
            t = res["tool"]
            row[f"{t}.leaning"] = res["leaning"]
            row[f"{t}.reliability"] = res["reliability"]
        row.update({f"weight.{k}": v for k, v in rep["summary"]["evidence_weight"].items()})
        rows.append(row)
        print(f"{p.name}: {rep['summary']['evidence_weight']}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("file", "label"), k))
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.csv}")


if __name__ == "__main__":
    main()
