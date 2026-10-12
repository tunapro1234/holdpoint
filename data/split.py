"""Fixed-seed train/val/test splits, committed under data/splits/<dataset>/{train,val,test}.txt.

    python data/split.py ozgenel_seg      # reads data/raw/ozgenel_seg/extracted
    python data/split.py ozgenel_cls
    python data/split.py sdnet2018

Each line is a path relative to data/raw/<dataset>/extracted. The split is stratified by group (crack label,
and surface type for SDNET2018): items are sorted, shuffled with random.Random(seed) per group, then cut by
the fractions. Same file list + same seed = same split, on any machine.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
SEED = 20261012
FRACTIONS = (0.70, 0.15, 0.15)
NAMES = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def make_split(items: Iterable[str], seed: int = SEED, fractions: tuple[float, float, float] = FRACTIONS,
               group: Callable[[str], str] | None = None) -> dict[str, list[str]]:
    """Deterministic, disjoint split of unique item names. Order of the input does not matter."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    uniq = sorted(set(items))
    groups: dict[str, list[str]] = defaultdict(list)
    for it in uniq:
        groups[group(it) if group else ""].append(it)
    out: dict[str, list[str]] = {n: [] for n in NAMES}
    for key in sorted(groups):
        members = groups[key]
        random.Random(f"{seed}:{key}").shuffle(members)
        n_train = int(round(len(members) * fractions[0]))
        n_val = int(round(len(members) * fractions[1]))
        out["train"] += members[:n_train]
        out["val"] += members[n_train:n_train + n_val]
        out["test"] += members[n_train + n_val:]
    return {n: sorted(v) for n, v in out.items()}


def _rel_images(root: Path) -> list[str]:
    return [q.relative_to(root).as_posix() for q in root.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES]


def _is_mask(rel: str) -> bool:
    parts = [p.lower() for p in Path(rel).parts]
    return any(p in {"mask", "masks", "bw", "label", "labels", "gt", "groundtruth"} for p in parts[:-1])


def items_for(dataset: str, root: Path) -> tuple[list[str], Callable[[str], str] | None]:
    """Split units and their stratification group for each dataset layout."""
    rels = _rel_images(root)
    if dataset == "ozgenel_seg":
        # Split the photos; each mask follows its photo by file stem.
        return [r for r in rels if not _is_mask(r)], None
    if dataset == "ozgenel_cls":
        return rels, lambda r: Path(r).parts[-2].lower()  # Positive / Negative
    if dataset == "sdnet2018":
        def grp(r: str) -> str:
            folder = Path(r).parts[-2].upper()  # CD, UD, CP, UP, CW, UW
            return folder
        return rels, grp
    raise KeyError(dataset)


def write_split(dataset: str, split: dict[str, list[str]], meta: dict) -> Path:
    out = SPLITS / dataset
    out.mkdir(parents=True, exist_ok=True)
    for n in NAMES:
        (out / f"{n}.txt").write_text("".join(f"{x}\n" for x in split[n]))
    info = {**meta, "counts": {n: len(split[n]) for n in NAMES}}
    (out / "split.json").write_text(json.dumps(info, indent=2) + "\n")
    return out


def read_split(dataset: str, base: Path = SPLITS) -> dict[str, list[str]]:
    d = base / dataset
    return {n: (d / f"{n}.txt").read_text().splitlines() for n in NAMES}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["ozgenel_seg", "ozgenel_cls", "sdnet2018"])
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    root = RAW / a.dataset / "extracted"
    if not root.is_dir():
        raise SystemExit(f"{root} not found; run data/download.py {a.dataset} first")
    items, grp = items_for(a.dataset, root)
    if not items:
        raise SystemExit(f"no images under {root}")
    split = make_split(items, a.seed, FRACTIONS, grp)
    out = write_split(a.dataset, split, {"dataset": a.dataset, "seed": a.seed, "fractions": FRACTIONS,
                                         "root": f"data/raw/{a.dataset}/extracted"})
    print(f"{a.dataset}: " + ", ".join(f"{n} {len(split[n])}" for n in NAMES) + f" -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
