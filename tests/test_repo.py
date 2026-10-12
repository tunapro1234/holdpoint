"""Repository packaging checks: README sections, repo scan, results table, compose file."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import check_repo, results_table

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

REQUIRED_HEADINGS = [
    "What it does",
    "Screenshot",
    "Architecture",
    "How to run",
    "Measurements",
    "Limits",
    "Glossary",
    "Attribution",
    "License",
]


def _headings(text: str) -> list[str]:
    return [line.lstrip("#").strip() for line in text.splitlines() if line.startswith("## ")]


def test_readme_has_required_headings():
    heads = _headings(README.read_text())
    missing = [h for h in REQUIRED_HEADINGS if h not in heads]
    assert not missing, f"README is missing sections: {missing}"


def test_readme_one_sentence_and_glossary():
    text = README.read_text()
    first_para = text.split("\n\n")[1].strip()
    assert first_para.startswith("*") and first_para.endswith("*") and first_para.count(". ") == 0
    glossary = text.split("## Glossary", 1)[1].split("\n## ", 1)[0].lower()
    for term in ("defect report", "ncr", "hold point"):
        assert term in glossary, term


def test_readme_architecture_has_amd_box():
    arch = README.read_text().split("## Architecture", 1)[1].split("\n## ", 1)[0]
    assert "```" in arch
    assert any("AMD" in line and "|" in line for line in arch.splitlines()), "no AMD box in the diagram"


def test_readme_points_to_attribution():
    text = README.read_text()
    assert "ATTRIBUTION.md" in text
    assert (ROOT / "ATTRIBUTION.md").exists()


# ---- results table ----------------------------------------------------------

def test_readme_results_table_matches_results_dir():
    """The table in the README is generated from results/; regenerate with tools/results_table.py."""
    block = results_table.readme_block(README.read_text())
    assert block is not None, "README has no generated results block"
    assert block.strip() == results_table.render(ROOT / "results").strip(), (
        "README results table is stale: run `python -m tools.results_table --write`"
    )


def test_results_table_empty_dir(tmp_path):
    out = results_table.render(tmp_path)
    assert "No benchmark results yet" in out


def test_results_table_renders_provenance_and_rows(tmp_path):
    fixture = {
        "fixture": "test fixture, not a measurement",
        "provenance": {"device": "TEST-DEVICE", "timestamp": "2000-01-01T00:00:00Z", "command": "pytest fixture"},
        "rows": [{"batch": 1, "img_per_s": 1.5}, {"batch": 8, "img_per_s": 9.25, "p95_ms": 12}],
    }
    (tmp_path / "bench-fixture.json").write_text(json.dumps(fixture))
    (tmp_path / "bench-noprov.json").write_text(json.dumps({"rows": [{"batch": 1}]}))
    out = results_table.render(tmp_path)
    lines = [ln for ln in out.splitlines() if ln.startswith("|")]
    assert lines[0].startswith("| file | device | measured | command | batch | img_per_s | p95_ms")
    assert any("TEST-DEVICE" in ln and "9.25" in ln for ln in lines)
    assert any("bench-noprov.json" in ln and "unverified" in ln for ln in lines)


def test_results_table_write_roundtrip(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n\n{results_table.START}\nold\n{results_table.END}\n\ntail\n")
    results_table.write_readme(readme, tmp_path)
    text = readme.read_text()
    assert "old" not in text and "No benchmark results yet" in text and text.endswith("tail\n")


# ---- repo scan --------------------------------------------------------------

def test_check_repo_clean():
    findings = check_repo.scan(ROOT)
    assert findings == [], "\n".join(map(str, findings))


@pytest.mark.parametrize(
    "content,rule",
    [
        ("key = 'sk-" + "a" * 40 + "'\n", "secret"),
        ("-----BEGIN RSA " + "PRIVATE KEY-----\n", "secret"),
        ("token: ghp_" + "A1" * 18 + "\n", "secret"),
        ('password = "' + "hunter2hunter2" + '"\n', "secret"),
        ("contact: jane.doe@" + "gmail.com\n", "personal"),
        ("call +" + "90 532 123 45 67\n", "personal"),
        ("built at " + "Pro" + "bot Studio\n", "forbidden"),
        ("see /s" + "rv/some/internal/path\n", "forbidden"),
    ],
)
def test_check_repo_flags(tmp_path, content, rule):
    f = tmp_path / "planted.txt"
    f.write_text(content)
    findings = check_repo.scan_files(tmp_path, [f])
    assert [x.rule for x in findings] == [rule], findings


def test_check_repo_allows_benign(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text(
        'client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))\n'
        "# Co-Authored-By: x <noreply@anthropic.com>\n"
        "mail = 'someone@example.com'\n"
        "ids = [12345678901, 2026]\n"
    )
    assert check_repo.scan_files(tmp_path, [f]) == []


def test_check_repo_extra_terms(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("Reviewed by Zeynep Q. Example.\n")
    assert check_repo.scan_files(tmp_path, [f]) == []
    hits = check_repo.scan_files(tmp_path, [f], extra_terms=["zeynep q. example"])
    assert [x.rule for x in hits] == ["personal"]
    assert "zeynep" not in str(hits[0]).lower(), "extra terms must not be echoed"


# ---- docker compose ---------------------------------------------------------

def test_compose_config_valid():
    compose = ROOT / "docker-compose.yml"
    assert compose.exists()
    if shutil.which("docker") is None:
        pytest.skip("docker not installed: compose config not validated")
    r = subprocess.run(
        ["docker", "compose", "-f", str(compose), "config", "--format", "json"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0 and "unknown" in r.stderr.lower() and "compose" in r.stderr.lower():
        pytest.skip("docker compose plugin not available")
    assert r.returncode == 0, r.stderr
    cfg = json.loads(r.stdout)
    app = cfg["services"]["app"]
    assert app["environment"]["HOLDPOINT_RECORDED"].endswith("examples/recorded.json")
    assert all(p.get("host_ip") == "127.0.0.1" for p in app["ports"]), "app port must bind to localhost only"
