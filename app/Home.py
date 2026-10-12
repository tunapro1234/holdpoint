import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.common import GLOSSARY, glossary, recorded_path  # noqa: E402

st.set_page_config(page_title="HoldPoint", layout="wide")
st.title("HoldPoint")
st.markdown(
    "*Photos of concrete elements become crack-width ranges and a drafted defect report; "
    "the element stays on hold until an inspector decides.*"
)
st.markdown("**AI measured. Human decided.**")

st.markdown(
    """
1. **Inspect**: a photo with the printed scale card becomes a crack mask and a width band in mm.
2. **Review queue**: an inspector approves or overrides the suggestion. Every decision is logged.
3. **Defects**: open defects put their element on hold. A repair photo and an inspector close them.
4. **Benchmarks**: speed and cost numbers, each with where and how it was measured.
5. **Data & limits**: what we measured, what we did not.
"""
)
st.info(
    f"Recorded mode reads results from `{recorded_path().name}`. Uploaded photos are processed live on CPU and "
    "labelled so. Badges next to every result show where it was computed."
)
st.subheader("Words used here")
for k, v in GLOSSARY.items():
    st.markdown(f"- **{k}**: {v}")
glossary()
