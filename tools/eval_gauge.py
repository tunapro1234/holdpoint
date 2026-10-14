"""Width bands against gauge readings: photos whose file names carry the value read before the photo was taken.

File name: <group>_<no>_<value-mm|na>_<distance-cm>_<angle-deg>.jpg, e.g. dogal_07_0.35_35_0.jpg
(groups: ref = printed line, real optics; sentil = feeler-gauge gap; dogal = natural crack).
For every photo the full pipeline runs (scale card, mask, width band, suggestion) and the report gives, per group
and per limit of the threshold table:
  coverage      share of measured photos whose reading lies inside the band
  gauge_rate    share sent to "measure with a gauge" (band crosses the limit or crack too thin to resolve)
  false_accept  photos suggested "accept" although the reading is above the limit (the error that matters)
  not_measured  photos without a band (no scale card found)

    python tools/eval_gauge.py photos/ --out results/gauge-eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdpoint.decide import ACCEPT, GAUGE, ThresholdTable, suggest  # noqa: E402

NAME = re.compile(r"^(?P<group>[a-z]+)_(?P<no>\d+)_(?P<value>na|\d+(?:\.\d+)?)_(?P<dist>\d+)_(?P<angle>\d+)$")


def parse_name(path: Path) -> dict | None:
    m = NAME.match(path.stem.lower())
    if not m:
        return None
    v = m["value"]
    return {"file": path.name, "group": m["group"], "no": int(m["no"]), "truth_mm": None if v == "na" else float(v),
            "distance_cm": int(m["dist"]), "angle_deg": int(m["angle"])}


def score(rows: list[dict], table: ThresholdTable) -> dict:
    """rows: parsed names plus band_mm (or None) and resolution_limited. Pure, no image work."""
    out = {}
    for group in sorted({r["group"] for r in rows}) + ["all"]:
        rs = [r for r in rows if r["truth_mm"] is not None and (group == "all" or r["group"] == group)]
        measured = [r for r in rs if r["band_mm"] is not None]
        cov = [r["band_mm"][0] <= r["truth_mm"] <= r["band_mm"][1] for r in measured]
        per_limit = {}
        for cls, limit in sorted(table.limits_mm.items(), key=lambda kv: kv[1]):
            states = [suggest(tuple(r["band_mm"]), cls, table, r["resolution_limited"]).state for r in measured]
            per_limit[f"{cls} ({limit} mm)"] = {
                "gauge_rate": round(sum(s == GAUGE for s in states) / len(states), 3) if states else None,
                "false_accept": sum(s == ACCEPT and r["truth_mm"] > limit for s, r in zip(states, measured)),
                "readings_above_limit": sum(r["truth_mm"] > limit for r in measured),
            }
        out[group] = {"n": len(rs), "not_measured": len(rs) - len(measured),
                      "coverage": round(sum(cov) / len(cov), 3) if cov else None, "per_limit": per_limit}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", type=Path)
    ap.add_argument("--table", type=Path, default=Path("config/thresholds.example.yaml"))
    ap.add_argument("--weights", type=Path, help="UNet checkpoint; without it the classical fallback runs")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    from holdpoint.backends.cpu import CpuBackend
    from holdpoint.pipeline import inspect

    table = ThresholdTable.load(a.table)
    backend = CpuBackend(weights=a.weights) if a.weights else CpuBackend()
    cls0 = next(iter(table.limits_mm))
    rows, skipped = [], []
    for p in sorted(a.photos.glob("*")):
        meta = parse_name(p) if p.suffix.lower() in (".jpg", ".jpeg", ".png") else None
        if meta is None:
            skipped.append(p.name)
            continue
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        w = inspect(img, backend, table, cls0).width
        rows.append({**meta, "band_mm": list(w.band_mm) if w.band_mm else None,
                     "resolution_limited": bool(w.resolution_limited)})
    res = {"table": table.label, "backend": type(backend).__name__,
           "weights": str(a.weights) if a.weights else None, "skipped_names": skipped,
           "summary": score(rows, table), "photos": rows,
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "command": "python tools/eval_gauge.py " + " ".join(argv if argv is not None else sys.argv[1:])}
    text = json.dumps(res, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(json.dumps(res["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
