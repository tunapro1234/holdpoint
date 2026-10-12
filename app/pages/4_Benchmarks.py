import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import glossary, provenance_badge, results_dir  # noqa: E402

st.set_page_config(page_title="Benchmarks · HoldPoint", layout="wide")
st.title("Benchmarks")
st.caption("Throughput, latency and cost per image. Each file shows where, when and with which command it was "
           "measured.")
glossary()

files = sorted(results_dir().glob("bench-*.json"))
if not files:
    st.info("No benchmark results yet. They are written by bench/bench.py to results/bench-*.json.")

for f in files:
    try:
        data = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        st.error(f"{f.name}: not valid JSON ({e})")
        continue
    with st.container(border=True):
        st.subheader(f.name)
        prov = data.get("provenance") or {}
        if prov:
            provenance_badge(prov, title="Measured")
            st.markdown(" · ".join(f"**{k}**: {prov[k]}" for k in ("device", "timestamp", "command") if k in prov))
        else:
            st.warning("This file has no provenance block; treat its numbers as unverified.")
        rows = data.get("rows") or data.get("results")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            st.dataframe(rows, width="stretch")
        rest = {k: v for k, v in data.items() if k not in ("provenance", "rows", "results")}
        if rest:
            st.json(rest, expanded=False)
