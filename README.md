# HoldPoint

*Photos of concrete elements become crack-width ranges and a drafted defect report; the element stays on hold until an inspector decides.*

**AI measured. Human decided.**

> Built during the AMD Developer Hackathon: ACT III. Work in progress; the [Limits](#limits) section says what is
> not done yet.

## What it does
1. A phone photo of a concrete element, with a printed ArUco scale card in view, goes in.
2. A crack mask is segmented. Width is measured from the mask and reported as a **range in mm**, never a single
   number. Without the scale card the result is "width not measurable".
3. The range is compared with the user's own threshold table and the app suggests one of: accept, repair, measure
   with a gauge, width not measurable. There is no built-in standard; the demo table is labelled
   "example values, not a standard".
4. A vision-language model gives a second opinion from a fixed vocabulary (crack, form oil stain, shadow, joint
   edge, air void, exposed rebar, unclear) and one sentence. It is not allowed to write numbers.
5. A defect report is drafted. Every number in its text must come from the measurement fields; a draft with an
   invented number is rejected and replaced by the template text.
6. An inspector approves or overrides (with an objection code). Each decision is appended to a hash-chained log:
   who, when, model version, photo SHA-256, suggestion, decision.
7. Elements with an open defect produce one `hold` line (CSV or webhook) for whatever system releases elements
   downstream. Closing a defect needs a repair photo and a human decision.

## Screenshot
<!-- Screenshot of the Inspect page goes here (docs/screenshot-inspect.png), taken from the running app. -->
_Screenshot pending: it will be taken from the running app, not mocked up._

## Architecture
```
 phone photo + scale card
          |
          v
 +------------------+     +--------------------------------------------+
 | holdpoint.measure|     | backends (one interface, provenance on all) |
 |  ArUco -> mm/px  |<----|  cpu       UNet weights or classical fallback|
 |  homography      |mask |  recorded  replays results from a JSON file  |
 |  skeleton + EDT  |     |  amd_vllm  OpenAI-compatible endpoint        |
 |  -> width band   |     +----------------------|----------------------+
 +--------|---------+                            |
          v                                      v
 +------------------+            +-------------------------------------+
 | decide.py        |            | AMD Instinct MI300X (ROCm)          |
 |  user threshold  |            |  UNet segmentation, batch inference |
 |  table -> state  |            |  vLLM + Qwen2.5-VL-7B second check  |
 +--------|---------+            |  bench: img/s, p50/p95, cost/1000   |
          v                      +-------------------------------------+
 +------------------+   +-------------------+   +----------------------+
 | ncr.py           |-->| ledger.py         |-->| defects/             |
 |  defect report   |   |  inspector decides|   |  lifecycle, QR id    |
 |  number guard    |   |  hash-chained log |   |  one-line hold CSV   |
 +------------------+   +-------------------+   +----------------------+
                         app/ (Streamlit): Inspect · Review queue · Defects · Benchmarks · Data & limits
```
Numbers come from deterministic code (`measure/`, `decide.py`). The language models read them and write words.

| Backend | Where it runs | Label shown in the app |
|---|---|---|
| `cpu` | local CPU; UNet weights if present, otherwise a classical image-processing fallback | "live on CPU" |
| `recorded` | reads results computed earlier from a JSON file | "computed on <device>, <date>, <command>" |
| `amd_vllm` | OpenAI-compatible vLLM server (`HOLDPOINT_VLLM_BASE_URL`) | "live on AMD" |

## How to run
Python 3.12. The light install is enough for the app and the tests; torch is optional.
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # app + tests, no torch
pip install -r requirements-ml.txt       # optional: torch + segmentation_models_pytorch for the UNet
python -m pytest -q

streamlit run app/Home.py                # http://localhost:8501
```
With Docker (light image, recorded mode, bound to 127.0.0.1 only):
```bash
docker compose up --build
```
Useful settings:

| Variable | Meaning |
|---|---|
| `HOLDPOINT_RECORDED` | results file the `recorded` backend replays (default: `examples/recorded.json`) |
| `HOLDPOINT_UNET_WEIGHTS` | UNet weights for the `cpu` backend; without it the classical fallback runs |
| `HOLDPOINT_VLLM_BASE_URL` | vLLM endpoint for the `amd_vllm` backend |
| `HOLDPOINT_HOLD_WEBHOOK` | where the hold list is posted (optional) |

Other tools:
```bash
python -m tools.make_card results/card.pdf      # A4 scale card; print at 100 %, check the 100 mm ruler
python -m tools.check_repo               # secrets / personal data / deny-list scan
python -m tools.results_table --write    # regenerate the table below from results/
```

## Measurements
The table is generated from `results/bench-*.json`. Each row names the device, time and command it came from.

<!-- results-table:start (generated by tools/results_table.py, do not edit) -->
_No benchmark results yet. `bench/bench.py` writes them to `results/bench-*.json`._
<!-- results-table:end -->

## Limits
- **Training data is not precast concrete.** SDNET2018 and Özgenel are bridge decks, walls, pavements and campus
  buildings. Accuracy on factory panels, form-oil stains and curing marks is unknown until such photos are tested.
- **Width has not been checked against a crack gauge on real cracks.** The geometry is tested on drawn images
  with known widths; that tests the measurement code, not the crack detector.
- The printed scale card has not yet been tested with real phone photos.
- Cracks a few pixels wide are not measured; the app says "measure with a gauge".
- The bundled `examples/` are **drawn** images (synthetic texture, drawn line, drawn card). Their results come
  from the classical CPU fallback and are labelled as synthetic in the app. Its width ranges are wide
  (example 1: 0.06-1.44 mm for a drawn 0.6-1.2 mm line).
- The decision log detects edits and reordering, but not lines cut off the end of the file.
- No AMD results are in this repository yet; the Benchmarks page and the table above stay empty until they are.
- The app does not judge structural significance, depth or cause. That is an engineer's call.

## Glossary
- **Defect report (NCR, non-conformance report):** the written record that an element does not meet its
  requirement, with what was found, where, and what happens next.
- **Hold point:** a step that the work cannot pass until someone with authority signs it off. Here: an element with
  an open defect stays on hold until an inspector closes it.
- **Inspector:** the person who decides. The app only suggests.
- **Scale card:** the printed ArUco marker that turns pixels into millimetres.
- **Width range (band):** the 10th-90th percentile of measured local widths, widened by pixel and calibration
  uncertainty.
- **Provenance:** where a result was computed: backend, device, model version, timestamp, command.

## Attribution
Datasets: SDNET2018 and Concrete Crack Images for Classification (Özgenel), both CC BY 4.0. Models and libraries:
segmentation_models_pytorch (MIT), Qwen2.5-VL-7B-Instruct (Apache-2.0), PyTorch, OpenCV, scikit-image, NumPy.
Full citations: [ATTRIBUTION.md](ATTRIBUTION.md).

## License
MIT, see [LICENSE](LICENSE).
