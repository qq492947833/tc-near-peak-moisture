"""Place the native Figure 6e time-series panel into the v2.1.2 Figure 6 layout."""

import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageChops


BASE_SHA256 = "b0a58fe18fe33584a5b398012f04813fb1c5c2fc20cb33f42de45cbe08150bd2"
# Native coordinates of the third-row left panel in the released layout.
LEFT, TOP = 0, 3771
PANEL_SIZE = (3144, 1926)
CANVAS_SIZE = (6377, 7157)


def compose(base_png, rate_panel_png, output_dir):
    """Write a native-resolution PNG and a lossless TIFF of the same RGB pixels."""
    base_png = Path(base_png)
    rate_panel_png = Path(rate_panel_png)
    output_dir = Path(output_dir)
    if hashlib.sha256(base_png.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Expected the native v2.1.2 Figure_06.png as the layout source")
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "Figure_06.png"
    tiff = output_dir / "Figure_06.tiff"
    if any(src.resolve() == dst.resolve()
           for src in (base_png, rate_panel_png) for dst in (png, tiff)):
        raise ValueError("Choose an output directory separate from the source images")
    with Image.open(base_png) as canvas, Image.open(rate_panel_png) as panel:
        if canvas.size != CANVAS_SIZE or panel.size != PANEL_SIZE:
            raise ValueError("Input dimensions differ from the native canvas or panel")
        if canvas.mode != "RGB" or panel.mode != "RGB":
            raise ValueError("Both images must contain native RGB pixels")
        # The previous integral panel ends at row 5558. The additional 139 rows
        # are unused white space in the third row, above the wind-matched panel.
        margin = canvas.crop((LEFT, 5558, LEFT + PANEL_SIZE[0], TOP + PANEL_SIZE[1]))
        if margin.getextrema() != ((255, 255),) * 3:
            raise ValueError("Unexpected content in the lower margin of the panel slot")
        canvas.paste(panel, (LEFT, TOP))
        canvas.save(png, dpi=(900, 900))
        canvas.save(tiff, compression="tiff_lzw", dpi=(900, 900))
    with Image.open(png) as a, Image.open(tiff) as b:
        if ImageChops.difference(a, b).getbbox() is not None:
            raise ValueError("The decoded PNG and TIFF differ")
    return png, tiff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="v2.1.2 Figure_06.png")
    parser.add_argument("--panel", type=Path, required=True, help="native Figure 6e rate panel")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for result in compose(args.base, args.panel, args.out):
        print(result)
