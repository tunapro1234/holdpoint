"""Near-duplicate check between the test split and train/val: is the same surface photographed twice across splits?

dHash and pHash (64 bit each) on the full image, grayscale. For each test image: the closest train/val image by
Hamming distance on both hashes. A pair counts as a near duplicate when both distances are <= --max-dist.
CPU only, seconds.

    python tools/near_dupes.py --out results/near-dupes-test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdpoint.vision import common  # noqa: E402
from holdpoint.vision.train_seg import real_items  # noqa: E402


def dhash(g: np.ndarray) -> np.ndarray:
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (s[:, 1:] > s[:, :-1]).ravel()


def phash(g: np.ndarray) -> np.ndarray:
    s = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(s)[:8, :8].ravel()
    return d > np.median(d[1:])


def hashes(items) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, dh, ph = [], [], []
    for img_p, _, _ in items:
        g = cv2.cvtColor(common.load_rgb(img_p), cv2.COLOR_RGB2GRAY)
        names.append(Path(img_p).name)
        dh.append(dhash(g))
        ph.append(phash(g))
    return names, np.array(dh), np.array(ph)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/ozgenel_seg/extracted"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits/ozgenel_seg"))
    ap.add_argument("--max-dist", type=int, default=10)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    def load(part):
        return hashes(real_items(a.root, common.read_split_list(a.splits, part)))

    tn, td, tp = load("test")
    rn, rd, rp = [], [], []
    for part in ("train", "val"):
        n, d, p = load(part)
        rn += [f"{part}/{x}" for x in n]
        rd.append(d)
        rp.append(p)
    rd, rp = np.concatenate(rd), np.concatenate(rp)

    rows = []
    for i, name in enumerate(tn):
        dd = (rd != td[i]).sum(1)
        pd = (rp != tp[i]).sum(1)
        j = int(np.argmin(dd + pd))
        rows.append({"test": name, "closest": rn[j], "dhash_dist": int(dd[j]), "phash_dist": int(pd[j]),
                     "near_duplicate": bool(dd[j] <= a.max_dist and pd[j] <= a.max_dist)})
    rows.sort(key=lambda r: r["dhash_dist"] + r["phash_dist"])
    dupes = [r["test"] for r in rows if r["near_duplicate"]]
    out = {"method": "dHash + pHash, 64 bit each, full image grayscale; both Hamming distances <= max_dist",
           "max_dist": a.max_dist, "n_test": len(tn), "n_reference": len(rn), "n_near_duplicates": len(dupes),
           "near_duplicates": dupes,
           "min_combined_dist": rows[0]["dhash_dist"] + rows[0]["phash_dist"] if rows else None,
           "closest_pairs": rows[:10],
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "command": "python tools/near_dupes.py " + " ".join(argv if argv is not None else sys.argv[1:])}
    text = json.dumps(out, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
