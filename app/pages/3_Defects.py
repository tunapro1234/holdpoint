import hashlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import defects, glossary  # noqa: E402
from holdpoint.defects import ACTIVE_STATES, Decision, DefectError, short_id  # noqa: E402
from holdpoint.defects.export import COLUMNS, hold_rows  # noqa: E402

st.set_page_config(page_title="Defects · HoldPoint", layout="wide")
st.title("Defects")
st.caption("An element with an open defect gets one hold line. Only a repair photo of the same element and an "
           "inspector's decision close it.")
glossary()

store = defects()
rows = [{"defect": d.defect_id, "element": d.element_id, "QR id": short_id(d.element_id), "report": d.report_id,
         "state": d.state, "opened": d.opened_at, "by": d.opened_by, "closed": d.closed_at or ""}
        for d in store.defects()]
tab_open, tab_all, tab_hold = st.tabs(["Open", "All", "Hold list"])
with tab_open:
    active = [r for r in rows if r["state"] in ACTIVE_STATES]
    if active:
        st.dataframe(active, width="stretch")
    else:
        st.info("No open defects.")
with tab_all:
    if rows:
        st.dataframe(rows, width="stretch")
    else:
        st.info("No defects yet.")
with tab_hold:
    holds = hold_rows(store)
    st.caption("One line per element. No amounts: what a receiving system does with a hold is up to it.")
    if holds:
        st.dataframe(holds, width="stretch")
    else:
        st.info("No element is on hold.")
    buf = io.StringIO()
    buf.write(",".join(COLUMNS) + "\n")
    for h in holds:
        buf.write(",".join(h[c] for c in COLUMNS) + "\n")
    st.download_button("Download hold list (CSV)", buf.getvalue(), file_name="holds.csv", mime="text/csv")

active_ids = [d.defect_id for d in store.active()]
if active_ids:
    st.subheader("Repair and close")
    pick = st.selectbox("Defect", active_ids)
    d = store.get(pick)
    st.markdown(f"Element **{d.element_id}** · state **{d.state}**")
    who = st.text_input("Inspector (name or role tag)")
    if d.state == "open":
        photo_element = st.text_input("Element id on the repair photo (QR label)", value=d.element_id)
        up = st.file_uploader("Repair photo", type=["jpg", "jpeg", "png"])
        if st.button("Attach repair photo", disabled=up is None):
            try:
                store.add_repair_photo(pick, photo_element, hashlib.sha256(up.getvalue()).hexdigest(), Decision(who))
                st.rerun()
            except DefectError as e:
                st.error(str(e))
    else:
        c1, c2 = st.columns(2)
        if c1.button("Accept repair and close"):
            try:
                store.close(pick, Decision(who))
                st.rerun()
            except DefectError as e:
                st.error(str(e))
        if c2.button("Reject repair"):
            try:
                store.reject_repair(pick, Decision(who))
                st.rerun()
            except DefectError as e:
                st.error(str(e))

st.subheader("Element history")
q = st.text_input("Element id or QR id", value="")
if q:
    hist = store.history(q.strip())
    if hist:
        st.dataframe(hist, width="stretch")
    else:
        st.info("No events for this element.")
