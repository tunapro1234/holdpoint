"""Download the public datasets into data/raw/ (never committed) and verify them by SHA-256.

    python data/download.py ozgenel_seg               # Concrete Crack Segmentation Dataset (Mendeley, 746 MB)
    python data/download.py ozgenel_cls               # Concrete Crack Images for Classification (Mendeley, 241 MB)
    python data/download.py sdnet2018                 # SDNET2018 (Utah State University)
    python data/download.py sdnet2018 --from-file ~/Downloads/SDNET2018.zip
    python data/download.py all --no-extract

Mendeley files are checked against the SHA-256 published by the Mendeley public API. The SDNET2018 host serves
a browser challenge to scripted clients; when that happens the script stops and asks for the zip to be
downloaded in a browser and passed with --from-file. Its SHA-256 is then recorded, not checked.

Each dataset gets data/raw/<name>/ with the archive, SHA256SUMS and an extracted/ folder.
Downloads resume from a partial file, so the command can be rerun after an interruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHECKSUMS = ROOT / "data" / "checksums.json"
USER_AGENT = "holdpoint-data/0.1 (+https://github.com/tunapro1234/holdpoint)"
CHUNK = 1 << 20

DATASETS = {
    "ozgenel_seg": {
        "title": "Concrete Crack Segmentation Dataset",
        "author": "Özgenel, Ç. F. (2019)",
        "doi": "10.17632/jwsn7tfbrp.1",
        "license": "CC BY 4.0",
        "mendeley": ("jwsn7tfbrp", 1),
    },
    "ozgenel_cls": {
        "title": "Concrete Crack Images for Classification",
        "author": "Özgenel, Ç. F. (2019)",
        "doi": "10.17632/5y9wdsg2zt.2",
        "license": "CC BY 4.0",
        "mendeley": ("5y9wdsg2zt", 2),
    },
    "sdnet2018": {
        "title": "SDNET2018: A concrete crack image dataset for machine learning applications",
        "author": "Maguire, M., Dorafshan, S., Thomas, R. J. (2018)",
        "doi": "10.15142/T3TD19",
        "license": "CC BY 4.0",
        "url": "https://digitalcommons.usu.edu/context/all_datasets/article/1047/type/native/viewcontent",
        "filename": "SDNET2018.zip",
    },
}


class BrowserChallenge(RuntimeError):
    """The host answered with an HTML page instead of the file."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _get(url: str, headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def mendeley_files(dataset_id: str, version: int) -> list[dict]:
    url = (f"https://data.mendeley.com/public-api/datasets/{dataset_id}/files"
           f"?folder_id=root&version={version}")
    with _get(url) as r:
        items = json.load(r)
    return [{
        "filename": it["filename"],
        "size": it["content_details"]["size"],
        "sha256": it["content_details"]["sha256_hash"],
        "url": it["content_details"]["download_url"],
    } for it in items]


def fetch(url: str, dest: Path, size: int | None = None) -> None:
    """Stream url into dest, resuming from dest.part when it exists."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size is not None and have == size:
        part.replace(dest)
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    with _get(url, headers) as r:
        ctype = r.headers.get("Content-Type", "")
        if ctype.startswith("text/html"):
            raise BrowserChallenge(f"{url} returned an HTML page, not the file")
        if have and r.status != 206:
            have = 0  # server ignored the range request, start over
        mode = "ab" if have else "wb"
        done = have
        with part.open(mode) as f:
            for block in iter(lambda: r.read(CHUNK), b""):
                f.write(block)
                done += len(block)
                if size and done % (64 * CHUNK) < CHUNK:
                    print(f"  {done / 1e6:.0f} / {size / 1e6:.0f} MB", flush=True)
    part.replace(dest)


def extract(archive: Path, out: Path) -> int:
    """Extract zip/rar into out with libarchive (RAR needs it). Returns the number of files written."""
    marker = out / ".extracted"
    if marker.exists():
        return int(marker.read_text() or 0)
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            for n in names:
                target = (out / n).resolve()
                if out.resolve() not in target.parents:
                    raise ValueError(f"unsafe path in archive: {n}")
            z.extractall(out)
        count = len(names)
    else:
        import libarchive  # pip install libarchive-c (uses the system libarchive)
        count = 0
        base = out.resolve()
        with libarchive.file_reader(str(archive)) as entries:
            for entry in entries:
                if entry.isdir:
                    continue
                target = (base / entry.pathname.replace("\\", "/")).resolve()
                if base not in target.parents:
                    raise ValueError(f"unsafe path in archive: {entry.pathname}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as f:
                    for block in entry.get_blocks():
                        f.write(block)
                count += 1
    marker.write_text(str(count))
    return count


def record(name: str, filename: str, digest: str, source: str) -> None:
    sums = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    sums.setdefault(name, {})[filename] = {"sha256": digest, "source": source}
    CHECKSUMS.write_text(json.dumps(sums, indent=2, ensure_ascii=False) + "\n")


def download(name: str, from_file: Path | None = None, do_extract: bool = True) -> Path:
    meta = DATASETS[name]
    folder = RAW / name
    folder.mkdir(parents=True, exist_ok=True)
    archives = []

    if "mendeley" in meta:
        for f in mendeley_files(*meta["mendeley"]):
            dest = folder / f["filename"]
            if dest.exists() and sha256_file(dest) == f["sha256"]:
                print(f"{name}: {f['filename']} already verified")
            else:
                print(f"{name}: downloading {f['filename']} ({f['size'] / 1e6:.0f} MB)")
                fetch(f["url"], dest, f["size"])
                got = sha256_file(dest)
                if got != f["sha256"]:
                    dest.rename(dest.with_name(dest.name + ".bad"))
                    raise RuntimeError(f"{name}: SHA-256 mismatch for {f['filename']}: {got} != {f['sha256']}")
                print(f"{name}: SHA-256 matches the Mendeley record")
            record(name, f["filename"], f["sha256"], "mendeley public API, verified")
            archives.append(dest)
    else:
        dest = folder / meta["filename"]
        if from_file is not None:
            shutil.copyfile(from_file, dest)
        elif not dest.exists():
            try:
                fetch(meta["url"], dest)
            except BrowserChallenge:
                sys.exit(f"{name}: the host asks for a browser check. Download the zip in a browser from "
                         f"https://doi.org/{meta['doi']} and rerun with --from-file <path>.")
        digest = sha256_file(dest)
        record(name, dest.name, digest, "recorded at download (no published hash)")
        print(f"{name}: {dest.name} sha256 {digest}")
        archives.append(dest)

    (folder / "SHA256SUMS").write_text("".join(f"{sha256_file(a)}  {a.name}\n" for a in archives))
    if do_extract:
        for a in archives:
            n = extract(a, folder / "extracted")
            print(f"{name}: {n} files in {folder / 'extracted'}")
    return folder


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", choices=[*DATASETS, "all"])
    p.add_argument("--from-file", type=Path, help="use an archive downloaded by hand (SDNET2018)")
    p.add_argument("--no-extract", action="store_true")
    a = p.parse_args(argv)
    names = list(DATASETS) if a.name == "all" else [a.name]
    for n in names:
        download(n, a.from_file if n == a.name else None, not a.no_extract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
