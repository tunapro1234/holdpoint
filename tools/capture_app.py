"""Screenshots and short clips of the running app, for the README, slides and the demo video.

    <venv with playwright>/bin/python tools/capture_app.py --out results/tmp/capture

Starts Streamlit in recorded mode on 127.0.0.1 (a free local port, nothing exposed) with a throwaway state folder,
walks through the pages in a headless 1920 x 1080 Chromium, and stops the Streamlit process it started. Every
scene is saved as one or more PNGs (viewport only) plus one .webm clip, with pauses so the clip can be read.

Scenes: s1 Home · s2 Inspect the drawn hairline example with the scale card (the page's default, "measure with a
gauge"), send it to the review queue · s2-inspect-repair the drawn example 1 ("repair suggested") · s3 Inspect a real
Özgenel test photo (no card, so no width in mm) · s4 Review queue: approve, open a defect, record the decision ·
s5 Defects: Open and Hold list tabs · s6 Benchmarks · s7 Data & limits, scrolled to the crack-ruler table.

Playwright is not an app dependency: run this with any Python that has `playwright` installed. The browser is
given with --chrome (no download). Streamlit is started from the repository venv (--streamlit).
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROME = Path.home() / ".cache/ms-playwright/chromium-1248/chrome-linux64/chrome"
W, H = 1920, 1080


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_streamlit(streamlit: Path, port: int, state: Path, log: Path) -> subprocess.Popen:
    env = {**os.environ, "HOLDPOINT_STATE": str(state), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2"}
    env.pop("HOLDPOINT_UNET_WEIGHTS", None)  # recorded mode only
    cmd = [str(streamlit), "run", "app/Home.py", "--server.address", "127.0.0.1", "--server.port", str(port),
           "--server.headless", "true", "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal",
           "--server.runOnSave", "false", "--theme.base", "light"]
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)


def wait_http(url: str, proc: subprocess.Popen, timeout: float = 90) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"streamlit exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"streamlit did not answer at {url} within {timeout:.0f} s")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(5)


class Shooter:
    def __init__(self, browser, base: str, out: Path, pause: float):
        self.browser, self.base, self.out, self.pause = browser, base, out, pause
        self.pngs: list[Path] = []
        self.clips: list[Path] = []
        self.vid_tmp = out / "_video"

    def scene(self, name: str, path: str, steps) -> None:
        ctx = self.browser.new_context(viewport={"width": W, "height": H}, color_scheme="light",
                                       record_video_dir=str(self.vid_tmp), record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        try:
            page.goto(self.base + path)
            settle(page)
            steps(page, lambda tag="": self.shot(page, name, tag))
            page.wait_for_timeout(int(self.pause * 1000))
        finally:
            video = page.video
            ctx.close()
        if video:
            clip = self.out / f"{name}.webm"
            Path(video.path()).replace(clip)
            self.clips.append(clip)
        print(f"{name}: done", flush=True)

    def shot(self, page, name: str, tag: str = "") -> None:
        settle(page)
        page.mouse.move(W - 4, H - 4)  # no hover outlines or tooltips in the picture
        page.evaluate("() => document.activeElement && document.activeElement.blur()")  # no focus ring
        page.wait_for_timeout(int(self.pause * 1000))
        p = self.out / f"{name}{'-' + tag if tag else ''}.png"
        page.screenshot(path=str(p))
        self.pngs.append(p)


def settle(page, timeout: float = 120) -> None:
    """Wait until Streamlit has finished the current script run."""
    page.wait_for_selector('[data-testid="stApp"]', timeout=timeout * 1000)
    page.wait_for_selector('[data-testid="stMain"] h1', timeout=timeout * 1000)
    t0 = time.time()
    while time.time() - t0 < timeout:
        running = page.evaluate("""() => {
            const w = document.querySelector('[data-testid="stStatusWidget"]');
            const skel = document.querySelector('[data-testid="stSkeleton"]');
            return Boolean((w && w.innerText.match(/Running|Connecting/)) || skel);
        }""")
        if not running:
            page.wait_for_timeout(400)
            return
        page.wait_for_timeout(200)
    raise TimeoutError("Streamlit run did not finish")


def scroll_to(page, text: str, block: str = "start") -> None:
    page.get_by_text(text, exact=False).first.evaluate(
        f"el => el.scrollIntoView({{behavior: 'smooth', block: '{block}'}})")
    page.wait_for_timeout(1500)


def pick_example(page, name: str) -> None:
    page.get_by_test_id("stSelectbox").filter(has_text="Example").click()
    page.get_by_role("option", name=name).click()
    settle(page)


def run(a) -> int:
    from playwright.sync_api import sync_playwright

    out = a.out.resolve()
    if out.exists():
        for p in list(out.glob("s*.png")) + list(out.glob("s*.webm")):
            p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    state = Path(tempfile.mkdtemp(prefix="holdpoint-capture-state-"))
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = start_streamlit(a.streamlit, port, state, out / "streamlit.log")
    print(f"streamlit pid {proc.pid} on {base}, state {state}", flush=True)
    try:
        wait_http(base, proc)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=str(a.chrome), headless=True)
            s = Shooter(browser, base, out, a.pause)

            s.scene("s1-home", "/", lambda page, shot: shot())

            def inspect_card(page, shot):
                page.get_by_test_id("stMetricLabel").filter(has_text="Crack width").wait_for()
                shot()
                scroll_to(page, "Send to review queue", "center")
                page.get_by_role("button", name="Send to review queue").click()
                page.get_by_text("for review.").wait_for()
                shot("queued")
            s.scene("s2-inspect-card", "/Inspect", inspect_card)

            def inspect_repair(page, shot):
                pick_example(page, a.repair_example)
                page.get_by_test_id("stMetricLabel").filter(has_text="Crack width").wait_for()
                shot()
            s.scene("s2-inspect-repair", "/Inspect", inspect_repair)

            def inspect_real(page, shot):
                pick_example(page, a.real_example)
                page.get_by_test_id("stMetricLabel").filter(has_text="Crack width").wait_for()
                shot()
            s.scene("s3-inspect-real-photo", "/Inspect", inspect_real)

            def review(page, shot):
                page.get_by_label("Inspector (name or role tag)").first.fill(a.inspector)
                page.get_by_label("Inspector (name or role tag)").first.press("Tab")
                box = page.get_by_role("checkbox", name="Open a defect report for this element (puts it on hold)")
                if not box.is_checked():
                    page.get_by_text("Open a defect report for this element").first.click()
                shot("draft")
                scroll_to(page, "Record decision", "center")
                shot("form")
                page.get_by_role("button", name="Record decision").click()
                page.get_by_text("hash chain intact").wait_for()
                scroll_to(page, "Decision log", "start")
                shot("log")
            s.scene("s4-review", "/Review_queue", review)

            def defects(page, shot):
                page.get_by_role("tab", name="Open").click()
                shot("open")
                page.get_by_role("tab", name="Hold list").click()
                shot("hold-list")
            s.scene("s5-defects", "/Defects", defects)

            def benchmarks(page, shot):
                shot()
                scroll_to(page, "bench-unet-rtx3060.json", "start")
                shot("gpu")
            s.scene("s6-benchmarks", "/Benchmarks", benchmarks)

            def limits(page, shot):
                shot("top")
                scroll_to(page, "What we measure", "start")
                shot("ruler")
            s.scene("s7-data-limits", "/Data_and_limits", limits)
            browser.close()
    finally:
        stop(proc)
        shutil.rmtree(state, ignore_errors=True)
        shutil.rmtree(out / "_video", ignore_errors=True)
    print(f"streamlit pid {proc.pid} stopped (exit {proc.returncode})")
    print(f"{len(s.pngs)} png, {len(s.clips)} webm in {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tmp" / "capture")
    ap.add_argument("--chrome", type=Path, default=DEFAULT_CHROME, help="Chromium executable (no download)")
    ap.add_argument("--streamlit", type=Path, default=ROOT / ".venv" / "bin" / "streamlit")
    ap.add_argument("--pause", type=float, default=2.5, help="seconds held before each screenshot and scene end")
    ap.add_argument("--real-example", default="ozgenel-test-018")
    ap.add_argument("--repair-example", default="example-1-card-head-on")
    ap.add_argument("--inspector", default="site inspector")
    a = ap.parse_args(argv)
    if not a.chrome.is_file():
        ap.error(f"no Chromium at {a.chrome}")
    if not a.streamlit.is_file():
        ap.error(f"no streamlit at {a.streamlit}")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
