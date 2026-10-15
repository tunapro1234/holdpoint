"""Choose the UNet input size for width measurement on the calibration cracks only.

Reads results/vt-ruler-unet-<side>.json written by `tools/eval_vt.py --report-calib` and applies the rule that was
written down before these runs: the highest band coverage on the calibration cracks (samples 1-5, where the crop
side S was fitted with the ground-truth masks); if two sides are within 0.02, the lower mean absolute error wins.
The test cracks (samples 6-15) play no part in the choice; their numbers are copied next to it for reporting.

Disclosure: test-crack results at sides 512 and 1024 had been seen before this rule was set (that is why it
exists: 512 had been chosen for segmentation IoU and measured widths badly).

    python tools/vt_side_select.py --out results/vt-side-select.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TIE = 0.02


def choose(calib: dict[int, dict]) -> int:
    best = max(c["coverage"] for c in calib.values())
    close = [s for s, c in calib.items() if best - c["coverage"] <= TIE]
    return min(close, key=lambda s: (calib[s]["mae_mm"], s))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    calib, test, mm = {}, {}, None
    for f in sorted(a.results.glob("vt-ruler-unet-*.json")):
        m = re.fullmatch(r"vt-ruler-unet-(\d+)\.json", f.name)
        d = json.loads(f.read_text())
        if not m or "calib_summary" not in d["results"]["unet"]:
            continue
        side = int(m[1])
        calib[side] = d["results"]["unet"]["calib_summary"]["all"]
        test[side] = d["results"]["unet"]["summary"]
        mm = d["scale"]["crop_side_mm_fitted"]
    side = choose(calib)
    out = {"rule": f"highest calibration-crack coverage; within {TIE} -> lower MAE", "calib_all": calib,
           "chosen_side": side, "crop_side_mm": mm, "mm_per_input_px": round(mm / side, 4),
           "test_at_each_side": test, "test_at_chosen_side": test[side],
           "note": "test-crack results at 512 and 1024 were seen before the rule was set; see the docstring"}
    if a.out:
        a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("chosen_side", "mm_per_input_px")} |
                     {"calib_coverage": {s: c["coverage"] for s, c in calib.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
