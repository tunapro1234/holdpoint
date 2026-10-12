import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import STATE_LABEL, defects, glossary, ledger, provenance_badge, read_queue  # noqa: E402
from holdpoint.defects import Decision, DefectError  # noqa: E402
from holdpoint.ledger import OBJECTION_CODES, LedgerError  # noqa: E402
from holdpoint.ncr import DefectReport, draft_report  # noqa: E402

st.set_page_config(page_title="Review queue · HoldPoint", layout="wide")
st.title("Review queue")
st.caption("AI suggests, the inspector decides. Each decision is written to an append-only, hash-chained log.")
glossary()

led = ledger()
decided = {(e["element_id"], e["photo_sha256"]) for e in led.entries()}
pending = [q for q in read_queue() if (q["element_id"], q["photo_sha256"]) not in decided]

if not pending:
    st.info("Nothing waiting. Send a result from the Inspect page.")

for i, q in enumerate(pending):
    with st.container(border=True):
        st.subheader(f"{q['element_id']} · {q['width_text']}")
        st.markdown(f"Suggestion: **{STATE_LABEL[q['suggestion']]}** · second check: {q['vlm_label'].replace('_', ' ')}")
        provenance_badge(q["provenance"].get("segment", {}), title="segment")
        report = draft_report(DefectReport(
            element_id=q["element_id"], photo_sha256=q["photo_sha256"],
            band_mm=tuple(q["band_mm"]) if q["band_mm"] else None, suggestion=q["suggestion"],
            limit_mm=q["limit_mm"], vlm_label=q["vlm_label"], vlm_sentence=q["vlm_sentence"],
            model_version=q["model_version"]))
        st.text_area("Defect report draft", report.text, key=f"txt{i}", disabled=True)
        st.caption(f"Text source: {report.text_source}")
        with st.form(f"decide{i}"):
            who = st.text_input("Inspector (name or role tag)", key=f"who{i}")
            decision = st.radio("Decision", ["approve", "override"], horizontal=True, key=f"dec{i}")
            codes = st.multiselect("Objection codes (override only)", OBJECTION_CODES, key=f"codes{i}")
            open_defect = st.checkbox("Open a defect report for this element (puts it on hold)",
                                      value=q["suggestion"] == "repair_suggested", key=f"open{i}")
            note = st.text_input("Note", key=f"note{i}")
            if st.form_submit_button("Record decision"):
                try:
                    led.append(who=who, element_id=q["element_id"], photo_sha256=q["photo_sha256"],
                               model_version=q["model_version"], suggestion=q["suggestion"],
                               decision=decision, objection_codes=codes, note=note)
                    if open_defect:
                        n = len(defects().defects()) + 1
                        defects().open(q["element_id"], f"NCR-{n:04d}", q["photo_sha256"], Decision(who, note=note))
                    st.success("Decision recorded.")
                    st.rerun()
                except (LedgerError, DefectError) as e:
                    st.error(str(e))

st.subheader("Decision log")
entries = led.entries()
try:
    led.verify()
    st.caption(f"{len(entries)} entries · hash chain intact")
except LedgerError as e:
    st.error(f"Hash chain broken: {e}")
if entries:
    st.dataframe([{k: e[k] for k in ("seq", "ts", "who", "element_id", "suggestion", "decision", "objection_codes",
                                       "model_version")} for e in entries], width="stretch")
