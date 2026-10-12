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
