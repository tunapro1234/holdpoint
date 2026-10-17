import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import (  # noqa: E402
    DOCS,
    EXAMPLES,
    STATE_LABEL,
    amd_second_check,
    glossary,
    live_cpu_backend,
    provenance_badge,
    push_queue,
    read_rgb,
    recorded_path,
    table_label,
    threshold_table,
)
from holdpoint.backends import get_backend  # noqa: E402
from holdpoint.backends.cpu import BASELINE_MODEL  # noqa: E402
from holdpoint.backends.recorded import NotRecorded  # noqa: E402
from holdpoint.pipeline import inspect, overlay  # noqa: E402

DEFAULT_EXAMPLE = "example-4-card-hairline"

st.set_page_config(page_title="Inspect · HoldPoint", layout="wide")
st.title("Inspect")
glossary()

table = threshold_table()
table_label(table)

examples = sorted(EXAMPLES.glob("*.jpg")) + sorted(EXAMPLES.glob("*.png"))
c1, c2 = st.columns(2)
source = c1.radio("Photo", ["Recorded example", "Upload (live on CPU)"], horizontal=True)
element_class = c2.selectbox("Element class (from the threshold table)", list(table.limits_mm))
element_id = c2.text_input("Element id (from the QR label)", value="P-14")

img, backend = None, None
if source == "Recorded example":
    if not examples:
        st.info("No example images found.")
    else:
        # the hairline example opens first: its answer is "measure with a gauge", the case the app is built for
        first = next((i for i, p in enumerate(examples) if p.stem == DEFAULT_EXAMPLE), 0)
        pick = c1.selectbox("Example", examples, index=first, format_func=lambda p: p.stem)
        img = read_rgb(pick)
        backend = get_backend("recorded", path=recorded_path()) if recorded_path().exists() else live_cpu_backend()
else:
    up = c1.file_uploader("Photo of a concrete surface with the scale card in view", type=["jpg", "jpeg", "png"])
    d1, d2 = c1.columns(2)
    card_pdf, card_example = DOCS / "scale-card.pdf", EXAMPLES / "example-1-card-head-on.jpg"
    if card_pdf.is_file():
        d1.download_button("Scale card (PDF, print at 100%)", card_pdf.read_bytes(), file_name="holdpoint-scale-card.pdf",
                           mime="application/pdf")
    if card_example.is_file():
        d2.download_button("Test image with the card (drawn, not a photo)", card_example.read_bytes(),
                           file_name=card_example.name, mime="image/jpeg")
    if up is not None:
        img = read_rgb(up.getvalue())
        backend = live_cpu_backend()

if img is not None:
    second_check = amd_second_check()
    try:
        insp = inspect(img, backend, table, element_class, vlm_check=second_check)
    except NotRecorded:
        st.warning("This image has no recorded result; processing live on CPU instead.")
        insp = inspect(img, live_cpu_backend(), table, element_class, vlm_check=second_check)

    if insp.provenance["segment"].get("model") == BASELINE_MODEL:
        st.warning("The trained UNet is not loaded here, so this mask comes from the classical baseline "
                   "(black-hat filter + Otsu threshold). Expect a rougher mask than the model gives.")
    left, right = st.columns([3, 2])
    left.image(overlay(img, insp), caption="Crack mask in orange, scale card outline in cyan", width="stretch")
    with right:
        w = insp.width
        st.metric("Crack width", w.text())
        if w.reason:
            st.caption(w.reason)
        s = insp.suggestion
        st.metric("Suggestion", STATE_LABEL[s.state])
        st.caption(f"{s.reason}" + (f" · limit {s.limit_mm} mm ({s.table_label})" if s.limit_mm else ""))
        st.markdown(f"**Second check:** {insp.vlm_label.replace('_', ' ')}. {insp.vlm_sentence}")
        if insp.p_crack is not None:
            st.caption(f"Crack score {insp.p_crack:.2f}")
        for step, prov in insp.provenance.items():
            provenance_badge(prov, title=step)
        st.caption("The suggestion is not a decision. An inspector decides in the Review queue.")
        if st.button("Send to review queue", type="primary"):
            push_queue({
                "element_id": element_id,
                "element_class": element_class,
                "photo_sha256": insp.image_sha256,
                "band_mm": list(w.band_mm) if w.band_mm else None,
                "width_text": w.text(),
                "suggestion": s.state,
                "limit_mm": s.limit_mm,
                "vlm_label": insp.vlm_label,
                "vlm_sentence": insp.vlm_sentence,
                "model_version": insp.provenance["segment"].get("model", "?"),
                "provenance": insp.provenance,
                "queued_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            })
            st.success(f"Queued {element_id} for review.")
    with st.expander("Raw result (JSON)"):
        summary = insp.summary()
        st.code(json.dumps(summary, indent=1, default=str), language="json")
