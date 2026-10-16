"""What the app would have suggested at each ruler point of the VT test cracks, for each example limit.

Reads the per-point bands written by `tools/eval_vt.py` (default: the width-pass side in
results/vt-side-select.json) and runs them through `holdpoint.decide.suggest`, the function the app uses, with
the example limits 0.2 / 0.3 / 0.4 mm. Each suggestion is set against the ruler reading: "repair" is right when
the ruler is above the limit, "accept" when it is at or below it; "measure with a gauge" is neither right nor
wrong, it is the cost of not guessing. The resolution flag is not applied here (the points file has no pixel
widths), so a few thin points the app would send to the gauge may count as accept.

Also reports how wide the bands are, because a band that always holds the ruler value but spans 3 mm is no help.

    python tools/vt_decisions.py --out results/vt-decisions.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdpoint.decide import ACCEPT, GAUGE, REPAIR, ThresholdTable, suggest  # noqa: E402
from tools.eval_vt import LIMITS_MM  # noqa: E402


def tally(points: list[dict], limits=LIMITS_MM) -> dict:
    found = [p for p in points if p.get("band_mm")]
    widths = [p["band_mm"][1] - p["band_mm"][0] for p in found]
    out = {"points": len(points), "band_found": len(found),
           "band_width_mm": {"median": round(statistics.median(widths), 3) if widths else None,
                             "mean": round(statistics.mean(widths), 3) if widths else None},
           "by_limit": {}}
    for lim in limits:
        table = ThresholdTable("vt", "example values, not a standard", {"c": lim})
        c = {"repair_ruler_above": 0, "repair_ruler_at_or_below": 0, "accept_ruler_at_or_below": 0,
             "accept_ruler_above": 0, "gauge_ruler_above": 0, "gauge_ruler_at_or_below": 0}
        for p in found:
            state = suggest(tuple(p["band_mm"]), "c", table).state
            name = {REPAIR: "repair", ACCEPT: "accept", GAUGE: "gauge"}[state]
            c[f"{name}_ruler_{'above' if p['truth_mm'] > lim else 'at_or_below'}"] += 1
        c["ruler_above"] = sum(1 for p in found if p["truth_mm"] > lim)
        out["by_limit"][str(lim)] = c
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--side", type=int, help="default: chosen_side in vt-side-select.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    side = a.side or json.loads((a.results / "vt-side-select.json").read_text())["chosen_side"]
    src = a.results / f"vt-ruler-unet-{side}.json"
    d = json.loads(src.read_text())
    out = {"source": src.name, "side": side, "test_samples": d.get("test_samples"),
           "rule": "holdpoint.decide.suggest on each band; resolution flag not applied",
           "limits_note": "example values, not a standard", **tally(d["results"]["unet"]["points"])}
    if a.out:
        a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["by_limit"]["0.3"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
