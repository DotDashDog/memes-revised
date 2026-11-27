"""Contact sheet of gold-sample images, for side-by-side checks while labelling.

Usage: uv run python -m experiments.wp04.sheet OUT.png id1 id2 id3 ...
Tiles are labelled with their ids, up to 5 per row, 380 px each.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SRC = Path("/home/linus/memes-revised/experiments/WP-04/sample-001/label_images")
T, PER_ROW = 380, 5


def main() -> None:
    out, ids = sys.argv[1], sys.argv[2:]
    rows = (len(ids) + PER_ROW - 1) // PER_ROW
    canvas = Image.new("RGB", (min(len(ids), PER_ROW) * (T + 6), rows * (T + 22)), "white")
    d = ImageDraw.Draw(canvas)
    for k, pid in enumerate(ids):
        im = Image.open(SRC / f"{pid}.jpg")
        im.thumbnail((T, T))
        x, y = (k % PER_ROW) * (T + 6), (k // PER_ROW) * (T + 22)
        canvas.paste(im, (x, y + 16))
        d.text((x + 2, y + 2), pid, fill="black")
    canvas.save(out)
    print(out, len(ids), "tiles")


if __name__ == "__main__":
    main()
