import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import glossary  # noqa: E402

st.set_page_config(page_title="Data & limits · HoldPoint", layout="wide")
st.title("Data & limits")
glossary()

st.subheader("Training data")
st.markdown(
    """
- **SDNET2018** (CC BY 4.0): crack / no-crack patches of bridge decks, walls and pavements.
- **Concrete Crack Images for Classification** (Özgenel, CC BY 4.0): crack / no-crack patches of campus buildings.
- **None of this is precast concrete.** The models have not seen factory-made panels, form-oil stains on fresh
  precast or curing marks. Expect lower accuracy there until such photos are collected.
- The example images in this app are **drawn** (synthetic texture, drawn line, drawn card), not photographs.
"""
)
st.subheader("What we measure")
st.markdown(
    """
- Crack mask quality on held-out dataset images (precision, recall, IoU), reported in pixels.
- Width as a **range**, from the mask and the printed scale card. On drawn test images the range contains the
  true width; that checks the geometry, not the crack detector.
- Speed and cost per image on the hardware named in each benchmark file.
"""
)
st.subheader("What we do not measure")
st.markdown(
    """
- Width accuracy against a crack gauge on real cracks, unless a photo set with gauge readings is listed here.
- Crack depth, cause, or whether a crack is structurally significant. That is an engineer's call.
- Cracks thinner than a few pixels in the photo: the app says "measure with a gauge" instead of guessing.
- Without the scale card in view, there is no width at all: "width not measurable".
"""
)
st.subheader("Rules the software follows")
st.markdown(
    """
- Numbers come from code, never from the language model. Report text with a number not in the inputs is rejected.
- There is no built-in acceptance standard. Limits come from your own table; the demo table is example values.
- The model suggests. Opening, repairing and closing a defect each need a named person.
"""
)
