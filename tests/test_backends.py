import json
from types import SimpleNamespace

import numpy as np
import pytest

from holdpoint.backends import VLM_LABELS, VlmResult, get_backend, image_sha256
from holdpoint.backends.amd_vllm import parse_vlm_reply
from holdpoint.backends.recorded import NotRecorded, mask_to_rle, rle_to_mask

PROV_KEYS = {"backend", "device", "model", "timestamp"}


def crack_image(h=64, w=96):
    """Light grey surface with one dark diagonal line, a stand-in for a crack in unit tests only."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    for x in range(10, w - 10):
        y = int(h * 0.2 + (x - 10) * 0.4)
        img[y : y + 3, x] = 40
    return img


# -- recorded -----------------------------------------------------------------

def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    m = rng.random((17, 23)) > 0.7
    assert np.array_equal(rle_to_mask(mask_to_rle(m)), m)


@pytest.fixture
def cache(tmp_path):
    img = crack_image()
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[20:24, 10:80] = True
    data = {
        "provenance": {
            "backend": "amd_notebook",
            "device": "AMD Instinct MI300X",
            "model": "unet-resnet18:test",
            "timestamp": "2026-10-13T10:00:00+00:00",
            "command": "python notebooks/batch_infer.py",
        },
        "results": {
            image_sha256(img): {
                "mask": mask_to_rle(mask),
                "p_crack": 0.91,
                "vlm": {"label": "crack", "sentence": "A thin continuous line follows the surface."},
            }
        },
    }
    path = tmp_path / "recorded.json"
    path.write_text(json.dumps(data))
    return path, img, mask


def test_recorded_returns_cached_results_with_provenance(cache):
    path, img, mask = cache
    be = get_backend("recorded", path=path)

    seg = be.segment(img)
    assert np.array_equal(seg.mask, mask)
    assert PROV_KEYS <= seg.provenance.keys()
    assert seg.provenance["backend"] == "recorded"
    assert seg.provenance["recorded_backend"] == "amd_notebook"
    assert seg.provenance["device"] == "AMD Instinct MI300X"
    assert seg.provenance["image_sha256"] == image_sha256(img)
    assert seg.provenance["label"].startswith("computed on AMD Instinct MI300X, 2026-10-13")

    assert be.classify(img).p_crack == pytest.approx(0.91)
    v = be.vlm_check(img)
    assert v.label == "crack" and v.provenance["model"] == "unet-resnet18:test"


def test_recorded_segment_at_replays_width_mask(cache):
    path, img, mask = cache
    data = json.loads(path.read_text())
    wide = np.zeros_like(mask)
    wide[18:26, 10:80] = True
    data["results"][image_sha256(img)]["width_masks"] = {"128": mask_to_rle(wide)}
    path.write_text(json.dumps(data))
    be = get_backend("recorded", path=path)
    seg = be.segment_at(img, 128)
    assert np.array_equal(seg.mask, wide) and seg.provenance["input_side"] == 128
    with pytest.raises(NotRecorded):
        be.segment_at(img, 256)  # another size was never recorded


def test_recorded_segment_at_without_width_masks_raises(cache):
    path, img, _ = cache
    with pytest.raises(NotRecorded):
        get_backend("recorded", path=path).segment_at(img, 128)


def test_recorded_unknown_image_raises(cache):
    path, img, _ = cache
    be = get_backend("recorded", path=path)
    with pytest.raises(NotRecorded):
        be.segment(np.zeros((8, 8, 3), dtype=np.uint8))


def test_recorded_needs_path(monkeypatch):
    monkeypatch.delenv("HOLDPOINT_RECORDED", raising=False)
    with pytest.raises(ValueError):
        get_backend("recorded")


# -- cpu ------------------------------------------------------------------------

def test_cpu_segment_returns_mask_without_torch(monkeypatch):
    monkeypatch.delenv("HOLDPOINT_UNET_WEIGHTS", raising=False)
    be = get_backend("cpu")
    img = crack_image()
    seg = be.segment(img)
    assert seg.mask.shape == img.shape[:2] and seg.mask.dtype == bool
    assert seg.mask.any()
    # the line pixels are found, the flat background mostly is not
    assert seg.mask[img[..., 0] == 40].mean() > 0.8
    assert seg.mask.mean() < 0.2
    assert PROV_KEYS <= seg.provenance.keys()
    assert seg.provenance["backend"] == "cpu" and seg.provenance["device"] == "cpu"
    assert seg.provenance["model"] == "classical-blackhat-otsu-v0"
    assert seg.provenance["label"].startswith("live on CPU, classical baseline")


def test_cpu_blank_image_empty_mask_and_low_p():
    be = get_backend("cpu")
    blank = np.full((40, 40, 3), 128, dtype=np.uint8)
    assert not be.segment(blank).mask.any()
    assert be.classify(blank).p_crack == 0.0
    assert be.classify(crack_image()).p_crack > 0.5


def test_cpu_vlm_is_unclear():
    v = get_backend("cpu").vlm_check(crack_image())
    assert v.label == "unclear" and v.provenance["backend"] == "cpu"


def test_cpu_unet_path_with_torch(tmp_path):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    net = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    w = tmp_path / "unet.pt"
    torch.save(net.state_dict(), w)
    be = get_backend("cpu", weights=w)
    seg = be.segment(crack_image(50, 70))
    assert seg.mask.shape == (50, 70)
    assert seg.provenance["model"] == "unet-resnet18:unet.pt"


def test_cpu_unet_resizes_to_side_and_back(tmp_path):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    net = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    w = tmp_path / "unet.pt"
    torch.save(net.state_dict(), w)
    seen = []
    be = get_backend("cpu", weights=w, side=64)
    be._unet().register_forward_hook(lambda m, i, o: seen.append(tuple(i[0].shape[-2:])))
    assert be.segment(crack_image(120, 200)).mask.shape == (120, 200)
    assert seen == [(64, 64)]  # long side 200 -> 64, short side 38 padded to 64



def test_cpu_segment_at_uses_given_side_and_reports_it(tmp_path):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    net = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    w = tmp_path / "unet.pt"
    torch.save(net.state_dict(), w)
    seen = []
    be = get_backend("cpu", weights=w, side=64)
    be._unet().register_forward_hook(lambda m, i, o: seen.append(tuple(i[0].shape[-2:])))
    seg = be.segment_at(crack_image(120, 200), 128)
    assert seg.mask.shape == (120, 200) and seen == [(96, 128)]
    assert seg.provenance["input_side"] == 128


def test_cpu_segment_at_without_unet_is_not_implemented(monkeypatch):
    monkeypatch.delenv("HOLDPOINT_UNET_WEIGHTS", raising=False)
    with pytest.raises(NotImplementedError):
        get_backend("cpu").segment_at(crack_image(50, 70), 64)


# -- pipeline: width pass ------------------------------------------------------

def test_width_input_side_targets_mm_per_px_and_never_upsamples():
    from holdpoint.pipeline import WIDTH_MM_PER_INPUT_PX, width_input_side

    assert width_input_side((3000, 4000, 3), None) is None
    assert width_input_side((3000, 4000, 3), SimpleNamespace(mm_per_px=0.1)) == round(4000 * 0.1 / WIDTH_MM_PER_INPUT_PX)
    assert width_input_side((300, 400, 3), SimpleNamespace(mm_per_px=1.0)) == 400


def _fake_pipeline(monkeypatch, backend):
    from holdpoint import pipeline

    seen = {}

    def fake_measure(mask, scale, calib_rel_err, source_mm_per_px):
        seen["mask"], seen["source_mm"] = mask, source_mm_per_px
        return SimpleNamespace(band_mm=None, resolution_limited=False)

    monkeypatch.setattr(pipeline, "detect_scale", lambda img: SimpleNamespace(mm_per_px=0.2, corners_px=None))
    monkeypatch.setattr(pipeline, "card_exclusion", lambda shape, scale: np.zeros(shape, bool))
    monkeypatch.setattr(pipeline, "measure_width", fake_measure)
    monkeypatch.setattr(pipeline, "suggest", lambda *a, **k: None)
    insp = pipeline.inspect(np.zeros((30, 40, 3), np.uint8), backend, None, "x")
    _fake_pipeline.source_mm = seen["source_mm"]
    return insp, seen["mask"]


def test_inspect_measures_width_on_the_width_pass(monkeypatch):
    from holdpoint.backends.base import SegmentResult

    display, wide = np.zeros((30, 40), bool), np.ones((30, 40), bool)
    be = SimpleNamespace(segment=lambda img: SegmentResult(mask=display, provenance={"step": "display"}),
                         segment_at=lambda img, side: SegmentResult(mask=wide, provenance={"input_side": side}),
                         classify=lambda img: (_ for _ in ()).throw(NotImplementedError()),
                         vlm_check=lambda crop: (_ for _ in ()).throw(NotImplementedError()))
    insp, measured = _fake_pipeline(monkeypatch, be)
    assert measured.all() and not insp.mask.any() and insp.width_mask.all()
    assert insp.provenance["width_segment"]["input_side"] == 40  # 40 * 0.2 / 0.1555 > 40, capped at the photo
    assert _fake_pipeline.source_mm == pytest.approx(0.2)  # no downscale: the photo's own pixel size


def test_inspect_falls_back_to_display_mask_without_segment_at(monkeypatch):
    from holdpoint.backends.base import SegmentResult

    display = np.ones((30, 40), bool)

    def no_side(img, side):
        raise NotImplementedError

    be = SimpleNamespace(segment=lambda img: SegmentResult(mask=display, provenance={}), segment_at=no_side,
                         classify=lambda img: (_ for _ in ()).throw(NotImplementedError()),
                         vlm_check=lambda crop: (_ for _ in ()).throw(NotImplementedError()))
    insp, measured = _fake_pipeline(monkeypatch, be)
    assert measured.all() and insp.width_mask is None and "width_segment" not in insp.provenance


# -- amd_vllm -------------------------------------------------------------------

def test_amd_vllm_needs_base_url(monkeypatch):
    monkeypatch.delenv("HOLDPOINT_VLLM_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        get_backend("amd_vllm")


def test_amd_vllm_reads_env_and_parses_reply(monkeypatch):
    monkeypatch.setenv("HOLDPOINT_VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    sent = {}

    class FakeCompletions:
        def create(self, **kw):
            sent.update(kw)
            msg = SimpleNamespace(content='{"label": "shadow", "sentence": "Soft edge, no depth."}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    be = get_backend("amd_vllm", client=client)
    assert be.base_url == "http://127.0.0.1:9/v1"
    v = be.vlm_check(crack_image())
    assert (v.label, v.sentence) == ("shadow", "Soft edge, no depth.")
    assert v.provenance["backend"] == "amd_vllm" and v.provenance["base_url"] == be.base_url
    assert sent["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    with pytest.raises(NotImplementedError):
        be.segment(crack_image())


@pytest.mark.parametrize(
    "reply,label",
    [
        ('{"label": "crack", "sentence": "x"}', "crack"),
        ('Sure! {"label": "Joint_Edge", "sentence": "x"}', "joint_edge"),
        ('{"label": "spalling", "sentence": "x"}', "unclear"),
        ("not json", "unclear"),
    ],
)
def test_parse_vlm_reply(reply, label):
    assert parse_vlm_reply(reply)[0] == label


def test_vlm_result_rejects_off_vocabulary():
    assert "crack" in VLM_LABELS
    with pytest.raises(ValueError):
        VlmResult(label="spalling", sentence="x")


def test_unknown_backend():
    with pytest.raises(ValueError):
        get_backend("cuda")


def test_checkpoint_loads_with_weights_only(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from holdpoint.vision.common import load_checkpoint_dict

    w = tmp_path / "ck.pt"
    torch.save({"state_dict": {"x": torch.zeros(1)}, "meta": {"torch": torch.__version__}}, w)  # old-style meta
    assert load_checkpoint_dict(w)["meta"]["torch"] == str(torch.__version__)

    class Evil:
        def __reduce__(self):
            return (print, ("ran",))

    torch.save({"meta": Evil()}, w)
    with pytest.raises(Exception):
        load_checkpoint_dict(w)
