"""Shared helpers for the Streamlit pages: paths, state files, provenance badge, glossary."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from holdpoint.decide import ThresholdTable  # noqa: E402
from holdpoint.defects import DefectStore  # noqa: E402
from holdpoint.ledger import Ledger  # noqa: E402

EXAMPLES = ROOT / "examples"
DOCS = ROOT / "docs"
# The released UNet (README "Trained model"). Fetched once for live uploads when no local weights are given,
# and used only if its sha256 matches. HOLDPOINT_UNET_URL="" turns the download off.
UNET_URL = "https://github.com/tunapro1234/holdpoint/releases/download/v0.1-unet/model.pt"
UNET_SHA256 = "1a992634324e9ef8b4fedb8e4495ea056c872eee0f8f09c46f774af60865b6a7"
LOCAL_UNET = ROOT / "runs" / "seg-gpu-1" / "model.pt"


# read on every call so tests and deployments can point them elsewhere
def recorded_path() -> Path:
    return Path(os.environ.get("HOLDPOINT_RECORDED", EXAMPLES / "recorded.json"))


def results_dir() -> Path:
    return Path(os.environ.get("HOLDPOINT_RESULTS", ROOT / "results"))


def state_dir() -> Path:
    return Path(os.environ.get("HOLDPOINT_STATE", ROOT / "results" / "tmp" / "app-state"))


def thresholds_path() -> Path:
    return Path(os.environ.get("HOLDPOINT_THRESHOLDS", ROOT / "config" / "thresholds.example.yaml"))

GLOSSARY = {
    "defect report": "The written record that a concrete element has a defect (in the trade: NCR, non-conformance report).",
    "pay application": "The monthly list a subcontractor sends to get paid for finished work, one line per element.",
    "hold": "A flag that says: do not pay for this element yet, it has an open defect.",
    "width band": "Crack width given as a range (low-high mm), never one number. Needs the printed scale card.",
    "scale card": "A printed 50 mm black-and-white marker placed next to the crack so pixels can be turned into mm.",
}

STATE_LABEL = {
    "accept_suggested": "Accept suggested",
    "repair_suggested": "Repair suggested",
    "measure_with_gauge": "Measure with a gauge",
    "width_not_measurable": "Width not measurable",
}


def ledger() -> Ledger:
    return Ledger(state_dir() / "decisions.jsonl")


def defects() -> DefectStore:
    return DefectStore(state_dir() / "defects.jsonl")


def queue_path() -> Path:
    return state_dir() / "queue.jsonl"


def read_queue() -> list[dict]:
    p = queue_path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def push_queue(item: dict) -> None:
    p = queue_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(item, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unet_weights() -> Path | None:
    """UNet weights for live CPU runs: HOLDPOINT_UNET_WEIGHTS, else a local training run, else the release file
    (downloaded once and checked against UNET_SHA256). None means the classical baseline runs, and says so."""
    if os.environ.get("HOLDPOINT_UNET_WEIGHTS"):
        return Path(os.environ["HOLDPOINT_UNET_WEIGHTS"])
    if LOCAL_UNET.is_file():
        return LOCAL_UNET
    url = os.environ.get("HOLDPOINT_UNET_URL", UNET_URL)
    try:
        import segmentation_models_pytorch  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return None  # no point fetching weights that cannot run
    if not url:
        return None
    cache = Path(os.environ.get("HOLDPOINT_CACHE", Path.home() / ".cache" / "holdpoint")) / "model.pt"
    if cache.is_file() and _sha256(cache) == UNET_SHA256:
        return cache
    import urllib.request

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except OSError:
        tmp.unlink(missing_ok=True)
        return None
    if _sha256(tmp) != UNET_SHA256:
        tmp.unlink()
        return None
    tmp.replace(cache)
    return cache


@st.cache_resource(show_spinner="Loading the crack model (first live photo only)...")
def live_cpu_backend():
    from holdpoint.backends import get_backend

    return get_backend("cpu", weights=unet_weights())


def amd_second_check():
    """Second check on the AMD vLLM endpoint when HOLDPOINT_VLLM_BASE_URL is set, else None (the backend's own
    check runs). Answers, "no answer" included, are kept per session and crop, so a rerun of the page (every
    click) does not wait on the endpoint again."""
    from holdpoint.backends import image_sha256
    from holdpoint.vlm import VlmClient

    client = VlmClient.from_env()
    if client is None:
        return None
    answers = st.session_state.setdefault("amd_vlm_answers", {})

    def check(crop):
        key = image_sha256(crop)
        if key not in answers:
            answers[key] = client.check_or_unclear(crop)
        return answers[key]

    return check


def report_writer():
    """LLM writer for defect report drafts when HOLDPOINT_LLM_BASE_URL is set, else None (template text)."""
    from holdpoint.ncr import LlmWriter

    return LlmWriter.from_env()


def threshold_table() -> ThresholdTable:
    return ThresholdTable.load(thresholds_path())


def read_rgb(path_or_bytes) -> np.ndarray:
    if isinstance(path_or_bytes, (str, Path)):
        bgr = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not read image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def badge_text(prov: dict) -> str:
    parts = [prov.get("label") or f"{prov.get('backend', '?')} on {prov.get('device', '?')}"]
    parts.append(f"model {prov.get('model', '?')}")
    if prov.get("synthetic"):
        parts.insert(0, "SYNTHETIC EXAMPLE")
    return " · ".join(parts)


def provenance_badge(prov: dict, title: str = "Source") -> None:
    """Every number on screen says where it came from."""
    color = "orange" if prov.get("synthetic") else ("blue" if prov.get("backend") == "recorded" else "green")
    st.markdown(f":{color}-badge[{title}] {badge_text(prov)}")
    with st.expander("Provenance details", expanded=False):
        st.json(prov)


def table_label(table: ThresholdTable) -> None:
    if table.label:
        st.warning(f"Threshold table '{table.name}': {table.label}. Replace it with your project's limits.")


def glossary(sidebar: bool = True) -> None:
    box = st.sidebar if sidebar else st
    with box.expander("Glossary", expanded=False):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}**: {v}")
