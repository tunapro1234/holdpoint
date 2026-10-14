"""Add real held-out photos to the app's recorded examples, segmented by the trained UNet on CPU.

The photos come from the Ozgenel crack segmentation set (CC BY 4.0, see ATTRIBUTION), test split only, so the model
never saw them in training or model selection. Near duplicates of training photos (results/near-dupes-test.json)
are skipped. The first --n remaining test photos in split order are used, not hand-picked ones.
Each photo is stored at long side --side; its result carries its own provenance and the IoU against the real mask.
The photos have no scale card, so the app reports widths in pixels only.

    python tools/make_ozgenel_examples.py --ckpt runs/seg-gpu-1/model.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdpoint.backends.base import image_sha256  # noqa: E402
from holdpoint.backends.cpu import CpuBackend  # noqa: E402
from holdpoint.backends.recorded import mask_to_rle  # noqa: E402
from holdpoint.vision import common  # noqa: E402
from holdpoint.vision.train_seg import real_items  # noqa: E402

PREFIX = "ozgenel-test-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/raw/ozgenel_seg/extracted"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits/ozgenel_seg"))
    ap.add_argument("--dupes", type=Path, default=Path("results/near-dupes-test.json"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--examples", type=Path, default=Path("examples"))
    a = ap.parse_args(argv)
    cmd = "python tools/make_ozgenel_examples.py " + " ".join(argv if argv is not None else sys.argv[1:])

    drop = set(json.loads(a.dupes.read_text())["near_duplicates"]) if a.dupes.exists() else set()
    items = [it for it in real_items(a.root, common.read_split_list(a.splits, "test")) if Path(it[0]).name not in drop]
    ckpt_sha = hashlib.sha256(a.ckpt.read_bytes()).hexdigest()
    be = CpuBackend(weights=a.ckpt, side=a.side)
    rec_p, idx_p = a.examples / "recorded.json", a.examples / "index.json"
    rec, index = json.loads(rec_p.read_text()), json.loads(idx_p.read_text())
    for old in a.examples.glob(PREFIX + "*.jpg"):
        old.unlink()
    rec["results"] = {k: v for k, v in rec["results"].items() if not str(v.get("file", "")).startswith(PREFIX)}
    index = [e for e in index if not e["file"].startswith(PREFIX)]

    for img_p, mask_p, _ in items[:a.n]:
        rgb = common.load_rgb(img_p, min_side=a.side // 2)
        s = a.side / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        path = a.examples / f"{PREFIX}{Path(img_p).stem}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)  # hash what the app will read
        sha = image_sha256(rgb)
        seg = be.segment(rgb)
        c = common.binary_counts(seg.mask, common.load_mask(mask_p, rgb.shape[:2]) > 0)
        iou = common.prf(*c)["iou"]
        when = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prov = {"recorded_backend": "cpu", "device": f"cpu ({platform.machine()})", "model": seg.provenance["model"],
                "checkpoint_sha256": ckpt_sha, "timestamp": when, "command": cmd, "synthetic": False,
                "note": "Real photo from the Ozgenel set (CC BY 4.0), held-out test split. UNet trained on "
                        "NVIDIA RTX 3060, run here on CPU. Not computed on AMD.",
                "label": f"computed on CPU, {when}, {cmd}"}
        rec["results"][sha] = {"mask": mask_to_rle(seg.mask), "file": path.name, "provenance": prov}
        index.append({"file": path.name, "image_sha256": sha, "synthetic": False, "source": "ozgenel_seg test split",
                      "original": Path(img_p).name, "license": "CC BY 4.0", "card": False,
                      "iou_vs_real_mask": round(iou, 4)})
        print(f"{path.name} iou {iou:.3f}", flush=True)
    rec_p.write_text(json.dumps(rec))
    idx_p.write_text(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
