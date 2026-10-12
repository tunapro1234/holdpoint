# HoldPoint

*Photos of concrete elements become crack-width ranges and a drafted defect report; the element stays on hold until an inspector decides.*

**AI measured. Human decided.**

> Draft. Work in progress during the AMD Developer Hackathon: ACT III.

## What it does
1. A phone photo of a concrete element (with a printed ArUco scale card) goes in.
2. A crack mask is segmented; width is measured from the mask and reported as a **range in mm**, never a single number.
   Without a scale card the result is "width not measurable".
3. A vision-language model gives a second opinion from a fixed vocabulary (crack, form oil stain, shadow, joint edge, ...).
4. An inspector accepts or rejects. Every decision is logged (who, when, model version, photo SHA-256).
5. Elements with an open defect produce a single `hold` line for downstream systems.

Numbers come from deterministic code. The language model only reads them and writes the explanation.

## Backends
Model calls sit behind one interface (`holdpoint.backends`):

| Backend | Where it runs | Label shown in the app |
|---|---|---|
| `cpu` | local CPU; UNet weights if present, otherwise a classical image-processing baseline | "live on CPU" |
| `recorded` | reads results computed earlier on AMD hardware from a JSON cache | "computed on AMD MI300X, <date>, <command>" |
| `amd_vllm` | OpenAI-compatible vLLM server (`HOLDPOINT_VLLM_BASE_URL`) | "live on AMD" |

Every result carries provenance: backend, device, model version, timestamp.

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # light deps, enough for tests
pip install -r requirements-ml.txt       # torch + segmentation_models_pytorch (CPU wheels)
python -m pytest -q
```

## App
```bash
streamlit run app/Home.py              # recorded mode + live CPU on upload, no torch needed
docker compose up --build              # same, in a container on 127.0.0.1:8501
```
Pages: Inspect · Review queue · Defects · Benchmarks · Data & limits. The bundled `examples/` are drawn images
(synthetic texture, drawn line, drawn scale card) with results from the classical CPU baseline, and are labelled
as such in the app. Set `HOLDPOINT_RECORDED` to a results file computed on AMD hardware to show those instead.

## License
MIT. Dataset and model attributions: see [ATTRIBUTION.md](ATTRIBUTION.md).
