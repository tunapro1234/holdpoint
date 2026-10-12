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
    assert seg.provenance["label"] == "live on CPU"


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
