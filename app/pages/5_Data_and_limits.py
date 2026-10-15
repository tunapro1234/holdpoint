import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import glossary, results_dir  # noqa: E402
from holdpoint.measure.width import MIN_RESOLVED_PX  # noqa: E402
from holdpoint.pipeline import WIDTH_MM_PER_INPUT_PX  # noqa: E402

# Ruler-check numbers are read from these files (written by tools/eval_vt.py and tools/vt_side_select.py),
# never typed into this page. Without the files the section says "not measured".
SIDE_SELECT = "vt-side-select.json"
RULER_GT = "vt-ruler-gt.json"
BANDS = [("all", "All points"), ("ruler <0.5 mm", "under 0.5 mm"), ("ruler 0.5-2 mm", "0.5-2 mm"),
         ("ruler >=2 mm", "2 mm and above")]


def load(name: str) -> dict | None:
    try:
        return json.loads((results_dir() / name).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def pct(x) -> str:
    return f"{x * 100:.0f}%" if isinstance(x, (int, float)) else "–"


def mm(x, signed: bool = False) -> str:
    if not isinstance(x, (int, float)):
        return "–"
    return f"{x:+.2f} mm" if signed else f"{x:.2f} mm"


def mm3(x) -> str:
    """0.1555 -> 0.156 (half up, as written in the README)."""
    if not isinstance(x, (int, float)):
        return "–"
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def under_limit(summary: dict) -> str:
    parts = [f"{bad} of {n} points at {lim} mm" for lim, (bad, n) in summary.get("band_under_limit_ruler_above",
                                                                                   {}).items()]
    return ", ".join(parts)


def ruler_check(sel: dict | None, gt: dict | None) -> None:
    unet = (sel or {}).get("test_at_chosen_side")
    if not unet:
        st.markdown(f"- **Width against a crack ruler on real cracks: not measured** "
                    f"(no `results/{SIDE_SELECT}`).")
        return
    human = ((gt or {}).get("results", {}).get("gt") or {}).get("summary", {})
    test = (gt or {}).get("test_samples") or []
    fit = (gt or {}).get("scale", {}).get("fitted_on_samples") or []
    allu = unet.get("all", {})
    st.markdown(
        f"Public field set: {(gt or {}).get('data', 'Virginia Tech Crack Quantification ROIs (CC BY 4.0)')}. "
        f"Cracks photographed from 3 distances, ruler readings at 5 points each. "
        + (f"The crop size was fitted on {len(fit)} cracks (calibration); every number below comes from the other "
           f"{len(test)} " if fit and test else "Numbers below come from the test cracks ")
        + f"({allu.get('n', '–')} ruler points, the band could be placed at {allu.get('found', '–')}). "
        f"The UNet measures width on its own pass at {mm3(sel.get('mm_per_input_px'))} mm per input pixel "
        f"(long side {sel.get('chosen_side', '–')} px), chosen on the calibration cracks only. "
        f"Files: `results/{SIDE_SELECT}`" + (f", `results/{RULER_GT}`." if gt else ".")
    )
    rows = ["| Ruler reading | Points | Band holds the ruler value: human-drawn masks | "
            "Band holds the ruler value: UNet masks | UNet mean absolute error | UNet bias |", "|---|---|---|---|---|---|"]
    for key, label in BANDS:
        u, h = unet.get(key, {}), human.get(key, {})
        rows.append(f"| {label} | {u.get('n', '–')} | {pct(h.get('coverage')) if human else 'not measured'} | "
                    f"{pct(u.get('coverage'))} | {mm(u.get('mae_mm'))} | {mm(u.get('bias_mm'), signed=True)} |")
    st.markdown("\n".join(rows))
    st.caption("\"Band holds\": of the points where a band could be placed, the share whose band contains the "
               "ruler value. Error and bias: band median minus ruler reading, absolute and signed means; negative bias "
               "reads too narrow. Human-drawn masks test the measurement code alone; UNet masks test the whole "
               "chain.")
    thin = unet.get("ruler <0.5 mm", {})
    st.warning(f"**Under 0.5 mm: not usable, check with a gauge.** Only {thin.get('n', '–')} "
               f"ruler points are that thin; the band held the ruler value at {pct(thin.get('coverage'))} of them "
               f"and read {mm(thin.get('bias_mm'), signed=True)} on average.")
    if allu.get("band_under_limit_ruler_above"):
        st.markdown(f"- *The error that matters*, the whole UNet band at or under an example limit while the ruler "
                    f"reads above it: {under_limit(allu)}.")
    if sel.get("note"):
        st.markdown(f"- Disclosure: {sel['note'].replace('see the docstring', 'see `tools/vt_side_select.py`')}.")
    st.markdown("- **The scale card was not part of this test.** The published photos have their scale markers "
                "cropped out, so the scale was fitted on the calibration cracks.")


st.set_page_config(page_title="Data & limits · HoldPoint", layout="wide")
st.title("Data & limits")
glossary()

st.subheader("Training data")
st.markdown(
    """
- **Concrete Crack Segmentation Dataset** (Özgenel, CC BY 4.0): 458 photos with real crack masks, split once with
  a fixed seed into 321 train / 69 validation / 68 test. The released UNet is trained on this set only.
- **SDNET2018** and **Concrete Crack Images for Classification** (Özgenel), both CC BY 4.0: crack / no-crack
  patches of bridge decks, walls, pavements and campus buildings. The download and training scripts support them;
  the released checkpoint does not use them.
- **None of this is precast concrete.** The models have not seen factory-made panels, form-oil stains on fresh
  precast or curing marks. Expect lower accuracy there until such photos are collected.
- Example images: `examples/example-*` are **drawn** (synthetic texture, drawn line, drawn card), not photographs.
  `examples/ozgenel-test-*` are real photos from the held-out test split; they have no scale card, so the app
  gives no width in mm for them.
"""
)
st.subheader("What we measure")
st.markdown(
    """
- Crack mask quality on held-out dataset photos (precision, recall, IoU), reported in pixels.
- Width as a **range**, from the mask and the printed scale card. On drawn test images the range contains the
  true width; that checks the geometry, not the crack detector.
- Speed and cost per image on the hardware named in each benchmark file.
"""
)
st.markdown("**Width against a crack ruler on real cracks**")
ruler_check(load(SIDE_SELECT), load(RULER_GT))

st.subheader("What we do not measure")
st.markdown(
    f"""
- Width with the printed scale card on real cracks. The ruler check above had no card in the photos, and the card
  has not yet been tested with real phone photos.
- Crack depth, cause, or whether a crack is structurally significant. That is an engineer's call.
- Cracks under {MIN_RESOLVED_PX:g} pixels wide: the app says "measure with a gauge" instead of guessing. The pixel
  counted is the model input pixel ({mm3(WIDTH_MM_PER_INPUT_PX)} mm on the width pass), not the phone's, because
  the mask cannot be finer than that.
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
