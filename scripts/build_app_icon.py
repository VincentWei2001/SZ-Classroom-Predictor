"""Generate assets/app_icon.ico (run from repo root; uses Pillow)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "app_icon.ico"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w = 256
    img = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    margin = 18
    d.rounded_rectangle(
        [margin, margin, w - margin, w - margin],
        radius=36,
        fill=(30, 136, 229, 255),
    )
    d.rounded_rectangle(
        [margin + 28, margin + 28, w // 2 - 8, w - margin - 28],
        radius=8,
        fill=(187, 222, 251, 255),
    )
    d.rounded_rectangle(
        [w // 2 + 8, margin + 28, w - margin - 28, w - margin - 28],
        radius=8,
        fill=(187, 222, 251, 255),
    )
    d.rounded_rectangle(
        [margin + 24, w - margin - 52, w - margin - 24, w - margin - 28],
        radius=6,
        fill=(255, 255, 255, 220),
    )

    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    images = [img.resize(s, Image.Resampling.LANCZOS) for s in sizes]
    images[0].save(
        OUT,
        format="ICO",
        sizes=[(im.width, im.height) for im in images],
        append_images=images[1:],
    )
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
