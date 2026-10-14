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
| `HOLDPOINT_UNET_SIDE` | long side a photo is resized to before the UNet (default 1024; the model is trained on 256 px crops, the probability map is resized back) |
| `HOLDPOINT_VLLM_BASE_URL` | vLLM endpoint for the `amd_vllm` backend |
| `HOLDPOINT_HOLD_WEBHOOK` | where the hold list is posted (optional) |

Other tools:
```bash
python -m tools.make_card results/card.pdf      # A4 scale card; print at 100 %, check the 100 mm ruler
python -m tools.check_repo               # secrets / personal data / deny-list scan
python -m tools.results_table --write    # regenerate the table below from results/
```

## Trained model
`model.pt` (55 MB) is attached to the GitHub release
[v0.1-unet](https://github.com/tunapro1234/holdpoint/releases/tag/v0.1-unet).
sha256 `1a992634324e9ef8b4fedb8e4495ea056c872eee0f8f09c46f774af60865b6a7`. Check it before use:
```bash
sha256sum model.pt
HOLDPOINT_UNET_WEIGHTS=model.pt streamlit run app/Home.py   # the cpu backend then runs the UNet
```
The code loads checkpoints with `torch.load(..., weights_only=True)` (`holdpoint/vision/common.py`), so a file
cannot run code when it is opened.

- **Model:** UNet, ResNet-18 encoder (segmentation_models_pytorch 0.5.0), 20 epochs, 256 px, trained on an
  NVIDIA RTX 3060. Not trained or measured on AMD hardware.
- **Data:** Özgenel crack segmentation set, 458 photos with real masks (CC BY 4.0, see Attribution), split once
  with a fixed seed into 321 train / 69 validation / 68 test (`data/splits/ozgenel_seg`). The best epoch is picked
  on validation; the test split is measured once, at the end.
- **Encoder pre-training:** ImageNet weights for ResNet-18 as distributed by segmentation_models_pytorch
  ([smp-hub/resnet18.imagenet](https://huggingface.co/smp-hub/resnet18.imagenet), license field "other"; the
  weights were first published by torchvision). The ImageNet images have their own terms of access
  (non-commercial research). Check both before any commercial use of this checkpoint.

Test split, 68 photos, short side resized to 256 and a 256 x 256 centre crop, threshold 0.5, pixel counts summed
over the split. Four training runs with different seeds (`results/seg-seeds-test.json`):

| Method | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| UNet, mean ± std over 4 seeds | 0.573 ± 0.006 | 0.729 ± 0.005 | 0.711 ± 0.014 | 0.747 ± 0.013 |
| Classical baseline (black-hat + Otsu), same photos and view | 0.268 | 0.423 | 0.656 | 0.312 |

Near duplicates: 3 of the 68 test photos are near copies of training photos (dHash and pHash distance 0-2 of 64
bits, `results/near-dupes-test.json`). Without them (65 photos): UNet IoU 0.568 (seed 20261012 run, the released
checkpoint) against 0.570 with them; classical 0.266.

## Measurements
The table is generated from `results/bench-*.json`. Each row names the device, time and command it came from.
The files are written by `bench/bench.py` (run on AMD from `notebooks/bench.ipynb`, with a CPU run for comparison):
```bash
python bench/bench.py --backend unet --checkpoint runs/seg/model.pt --images examples --device auto
# cost per 1000 images only with a price and its source; there is no default price
python bench/bench.py --backend classifier --checkpoint runs/cls/model.pt --images examples \
    --price-per-hour <USD per hour> --price-source "<where the price was read, date>"
```
Each file also keeps the device name, torch and ROCm versions, the raw `amd-smi` output ("not available" when the
tool is missing) and the git commit.

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
- The bundled `examples/example-*` are **drawn** images (synthetic texture, drawn line, drawn card). Their results
  come from the classical CPU fallback and are labelled as synthetic in the app. Its width ranges are wide
  (example 1: 0.06-1.44 mm for a drawn 0.6-1.2 mm line).
- `examples/ozgenel-test-*` are four real test photos (the first four in split order after removing near
  duplicates), segmented by the released UNet on CPU. They have no scale card, so the app gives no width in mm.
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
Datasets: Concrete Crack Segmentation Dataset (Özgenel), SDNET2018 and Concrete Crack Images for Classification
(Özgenel), all CC BY 4.0. Models and libraries:
segmentation_models_pytorch (MIT) with ImageNet ResNet-18 encoder weights (see Trained model), Qwen2.5-VL-7B-Instruct (Apache-2.0), PyTorch, OpenCV, scikit-image, NumPy.
Full citations: [ATTRIBUTION.md](ATTRIBUTION.md).

## License
MIT, see [LICENSE](LICENSE).
