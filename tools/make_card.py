"""Printable A4 scale card: one 50 mm DICT_4X4_50 marker, a 100 mm ruler with 10 mm lines, print warning,
and printed reference lines of known nominal width (REF_LINES_MM) for checking the width measurement.

    python tools/make_card.py [out.pdf] [--id 0]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holdpoint.measure.aruco import ARUCO_DICT, MARKER_MM  # noqa: E402

REF_LINES_MM = (0.2, 0.3, 0.5, 1.0, 2.0)
WARNING = "PRINT AT 100% (actual size, no fit-to-page). The black square must measure 50 mm."


def marker_png(marker_id: int, px: int = 600) -> bytes:
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), marker_id, px)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def make_card(out: str | Path, marker_id: int = 0) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = A4
    c = canvas.Canvas(str(out), pagesize=A4)
    c.setTitle("HoldPoint scale card")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, page_h - 20 * mm, WARNING)
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, page_h - 26 * mm,
                 f"ArUco DICT_4X4_50, id {marker_id}. Place flat on the concrete, next to the crack, in the same plane.")

    # marker: 50 mm black square, 10 mm white quiet zone around it
    side = MARKER_MM * mm
    x0 = (page_w - side) / 2
    y0 = page_h - 50 * mm - side
    c.drawImage(ImageReader(io.BytesIO(marker_png(marker_id))), x0, y0, side, side)
    c.setLineWidth(0.3)
    c.setStrokeGray(0.6)
    c.rect(x0 - 10 * mm, y0 - 10 * mm, side + 20 * mm, side + 20 * mm)

    # ruler: 100 mm, 1 mm ticks, 10 mm long lines with labels
    rx, ry = (page_w - 100 * mm) / 2, y0 - 35 * mm
    c.setStrokeGray(0)
    c.setLineWidth(0.4)
    c.line(rx, ry, rx + 100 * mm, ry)
    c.setFont("Helvetica", 7)
    for i in range(101):
        x = rx + i * mm
        tick = 8 * mm if i % 10 == 0 else (4 * mm if i % 5 == 0 else 2 * mm)
        c.line(x, ry, x, ry + tick)
        if i % 10 == 0:
            c.drawCentredString(x, ry - 4 * mm, f"{i}")
    c.drawString(rx + 104 * mm, ry - 4 * mm, "mm")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, ry - 14 * mm, "Check after printing: ruler 0-100 must measure 100 mm. If not, reprint at 100%.")

    # reference lines: nominal widths, 40 mm long; the printed width depends on the printer
    c.drawString(20 * mm, ry - 26 * mm, "Reference lines (nominal width; printers spread ink, check with a crack gauge):")
    c.setFont("Helvetica", 8)
    for k, w in enumerate(REF_LINES_MM):
        ly = ry - (36 + 10 * k) * mm
        c.setLineWidth(w * mm)
        c.setLineCap(0)
        c.line(rx + 15 * mm, ly, rx + 55 * mm, ly)
        c.drawString(rx, ly - 1 * mm, f"{w:.1f} mm")
    c.showPage()
    c.save()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="results/scale_card.pdf")
    ap.add_argument("--id", type=int, default=0)
    a = ap.parse_args(argv)
    print(make_card(a.out, a.id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
