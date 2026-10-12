"""Scan the repository for secrets, personal data and words that must not be published.

    python -m tools.check_repo                 # scan files git would publish (tracked + untracked, not ignored)
    python -m tools.check_repo --terms FILE    # also flag the names listed in FILE (one per line, kept outside the repo)

Exit code 0 when clean, 1 when something was found. Findings name the file, line and rule, never the matched
text of an extra term, so the output can be pasted into a log.

Internal names (build infrastructure, unrelated products) are stored as truncated SHA-256 hashes of lowercased
tokens, so the deny-list itself does not publish them.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SECRET_PATTERNS = [
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("hard-coded credential",
     re.compile(r"""(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*["'][^"'\s]{8,}["']""")),
]

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_OK = re.compile(r"(?i)(^noreply@|@users\.noreply\.github\.com$|@example\.(?:com|org|net)$)")
PHONE = re.compile(r"\+\d{1,3}[ .-]?\(?\d{2,4}\)?(?:[ .-]?\d{2,4}){2,4}\b")

FORBIDDEN_PATTERNS = [
    ("internal server path", re.compile(r"(?<![\w.])/(?:srv|root)/")),
]
FORBIDDEN_TOKEN_HASHES = {
    "952dab41064b6ae1", "ac044204248b67a9", "865e49116cf00221", "d2ba0f41115bdb49", "96b83171bd747a20",
    "969f3755c398b0da", "04355cddc3145de8", "b370de14e94142d4", "fcbba2b38b6096a8", "25daf5c311d4c710",
    "8ec46ec65d8c22b1", "70845345228ede3d", "b1ece0f3fb4f7be0", "ab68207b26d483b5", "83a84109e18d5a2e",
}
TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
MAX_JOIN = 4

BINARY_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".pdf", ".pt", ".pth", ".onnx", ".zip", ".gz", ".ico"}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str  # secret | personal | forbidden
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail}"


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def _forbidden_token(line: str) -> bool:
    for tok in TOKEN.findall(line.lower()):
        parts = re.split(r"[-_.]", tok)
        for i in range(len(parts)):
            for j in range(i + 1, min(len(parts), i + MAX_JOIN) + 1):
                if _h("-".join(parts[i:j])) in FORBIDDEN_TOKEN_HASHES:
                    return True
        if _h(tok) in FORBIDDEN_TOKEN_HASHES:
            return True
    return False


def scan_text(text: str, rel: str, extra_terms: list[str] = ()) -> list[Finding]:
    out: list[Finding] = []
    terms = [t.strip().lower() for t in extra_terms if t.strip()]
    for n, line in enumerate(text.splitlines(), 1):
        for name, pat in SECRET_PATTERNS:
            if pat.search(line):
                out.append(Finding(rel, n, "secret", name))
                break
        for m in EMAIL.finditer(line):
            if not EMAIL_OK.search(m.group(0)):
                out.append(Finding(rel, n, "personal", "e-mail address"))
                break
        if PHONE.search(line):
            out.append(Finding(rel, n, "personal", "phone number"))
        low = line.lower()
        if any(t in low for t in terms):
            out.append(Finding(rel, n, "personal", "listed name"))
        for name, pat in FORBIDDEN_PATTERNS:
            if pat.search(line):
                out.append(Finding(rel, n, "forbidden", name))
                break
        else:
            if _forbidden_token(line):
                out.append(Finding(rel, n, "forbidden", "deny-listed name"))
    return out


def publishable_files(root: Path) -> list[Path]:
    """Files git would publish: tracked plus untracked-but-not-ignored."""
    r = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root, capture_output=True, check=True,
    )
    names = sorted({n for n in r.stdout.decode().split("\0") if n})
    return [root / n for n in names if (root / n).is_file()]


def scan_files(root: Path, files: list[Path], extra_terms: list[str] = ()) -> list[Finding]:
    out: list[Finding] = []
    for f in files:
        rel = f.relative_to(root).as_posix()
        if f.suffix.lower() in BINARY_SUFFIXES:
            continue
        data = f.read_bytes()
        if b"\0" in data[:4096]:
            continue
        out.extend(scan_text(data.decode("utf-8", errors="replace"), rel, extra_terms))
    return out


def scan(root: Path, extra_terms: list[str] = ()) -> list[Finding]:
    root = Path(root)
    return scan_files(root, publishable_files(root), extra_terms)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    ap.add_argument("--terms", type=Path, help="file with extra names to flag, one per line (keep it out of the repo)")
    args = ap.parse_args(argv)
    terms = args.terms.read_text().splitlines() if args.terms else []
    findings = scan(args.root, terms)
    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s)" if findings else "clean")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
