#!/usr/bin/env python3
"""Generate print-ready calibration targets (PDF, US letter, exact scale).

Outputs to vision/targets/:
  charuco_letter.pdf — ChArUco board, letter landscape, for lens intrinsics
  aruco_tags_letter.pdf — 4 individual ArUco tags, letter portrait, for
                          ambient-light startup frames / index anchors

Print at 100% / "Actual size" (never "fit to page"), ideally on a laser
printer. Rendered at exactly 20 px/mm (508 dpi).

These constants are the single source of truth for the physical target
dimensions — the intrinsics tool must use the same values.
"""

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

PX_PER_MM = 20          # 508 dpi exactly
DPI = PX_PER_MM * 25.4  # 508.0

# ── ChArUco board (intrinsics) ──────────────────────────────────────
CHARUCO_COLS = 11        # squares in x
CHARUCO_ROWS = 8         # squares in y
SQUARE_MM = 24.0
MARKER_MM = 18.0
CHARUCO_DICT = cv2.aruco.DICT_5X5_100

# ── Individual tags (startup / indexing) ────────────────────────────
TAG_DICT = cv2.aruco.DICT_4X4_50
TAG_IDS = [0, 1, 2, 3]
TAG_MM = 60.0

LETTER_MM = (279.4, 215.9)  # US letter landscape (w, h)

OUT_DIR = Path(__file__).resolve().parent.parent / "targets"


def mm(v):
    return int(round(v * PX_PER_MM))


def font(size_px):
    for path in ("/usr/share/fonts/TTF/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size_px)
        except OSError:
            continue
    return ImageFont.load_default()


def gen_charuco():
    d = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS), SQUARE_MM / 1000.0, MARKER_MM / 1000.0, d)
    bw, bh = mm(CHARUCO_COLS * SQUARE_MM), mm(CHARUCO_ROWS * SQUARE_MM)
    img = board.generateImage((bw, bh), marginSize=0, borderBits=1)

    page = Image.new("L", (mm(LETTER_MM[0]), mm(LETTER_MM[1])), 255)
    ox = (page.width - bw) // 2
    oy = (page.height - bh) // 2
    page.paste(Image.fromarray(img), (ox, oy))

    label = (f"ChArUco {CHARUCO_COLS}x{CHARUCO_ROWS}  square {SQUARE_MM:.0f}mm  "
             f"marker {MARKER_MM:.0f}mm  DICT_5X5_100  —  print at 100% scale, "
             f"verify square = {SQUARE_MM:.0f}mm with calipers")
    draw = ImageDraw.Draw(page)
    draw.text((ox, page.height - oy + mm(1.5)), label, fill=0, font=font(mm(3)))

    out = OUT_DIR / "charuco_letter.pdf"
    page.save(out, "PDF", resolution=DPI)
    print(f"wrote {out}  ({page.width}x{page.height}px, board {bw}x{bh}px)")


def gen_tags():
    d = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    page = Image.new("L", (mm(LETTER_MM[1]), mm(LETTER_MM[0])), 255)  # portrait
    draw = ImageDraw.Draw(page)
    f = font(mm(5))

    cell_w = page.width // 2
    cell_h = page.height // 2
    side = mm(TAG_MM)
    for i, tag_id in enumerate(TAG_IDS):
        tag = cv2.aruco.generateImageMarker(d, tag_id, side)
        cx = (i % 2) * cell_w + (cell_w - side) // 2
        cy = (i // 2) * cell_h + (cell_h - side) // 2
        page.paste(Image.fromarray(tag), (cx, cy))
        draw.text((cx, cy + side + mm(2)),
                  f"DICT_4X4_50 id={tag_id}  {TAG_MM:.0f}mm", fill=0, font=f)

    out = OUT_DIR / "aruco_tags_letter.pdf"
    page.save(out, "PDF", resolution=DPI)
    print(f"wrote {out}  ({page.width}x{page.height}px, tags {side}x{side}px)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen_charuco()
    gen_tags()


if __name__ == "__main__":
    main()
