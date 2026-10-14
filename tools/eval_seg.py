"""Re-evaluate a segmentation checkpoint on a split, optionally without listed images (e.g. near duplicates).

Same view and metric as train_seg's evaluate (short side --size, centre crop, threshold 0.5). Prints both the
full split and the split minus --exclude, side by side.

    python tools/eval_seg.py --ckpt runs/seg-gpu-1/model.pt --exclude results/near-dupes-test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdpoint.vision import common  # noqa: E402
from holdpoint.vision.infer import load_checkpoint  # noqa: E402
from holdpoint.vision.train_seg import SegSet, evaluate, real_items  # noqa: E402


def main(argv=None) -> int:
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/raw/ozgenel_seg/extracted"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits/ozgenel_seg"))
    ap.add_argument("--part", default="test", choices=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--exclude", type=Path, help="near_dupes.py output; its near_duplicates are left out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    net, meta = load_checkpoint(a.ckpt, device)
    items = real_items(a.root, common.read_split_list(a.splits, a.part))
    drop = set(json.loads(a.exclude.read_text())["near_duplicates"]) if a.exclude else set()

    def run(its):
        return evaluate(net, DataLoader(SegSet(its, a.size, False), batch_size=8), device)

    out = {"checkpoint": str(a.ckpt), "part": a.part, "view": f"short side {a.size}, centre {a.size}x{a.size} crop",
           "full": run(items)}
    if drop:
        kept = [it for it in items if Path(it[0]).name not in drop]
        out["excluded"] = sorted(drop)
        out["without_excluded"] = run(kept)
    out["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["command"] = "python tools/eval_seg.py " + " ".join(argv if argv is not None else sys.argv[1:])
    text = json.dumps(out, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
