"""Compare regenerated raster figures with the versioned figure asset bundle."""
from pathlib import Path
import argparse
import json


def compare_images(generated, reference):
    import numpy as np
    from PIL import Image
    with Image.open(generated) as x, Image.open(reference) as y:
        size_x, size_y = x.size, y.size
        if size_x != size_y:
            return {'same_dimensions': False, 'generated_dimensions': size_x,
                    'reference_dimensions': size_y, 'pixel_exact': False}
        x = x.convert('RGB'); y = y.convert('RGB')
        different = 0; maximum = 0
        for top in range(0, x.height, 256):
            box = (0, top, x.width, min(x.height, top + 256))
            a = np.asarray(x.crop(box), dtype=np.int16)
            b = np.asarray(y.crop(box), dtype=np.int16)
            difference = np.abs(a - b)
            maximum = max(maximum, int(difference.max(initial=0)))
            different += int(np.count_nonzero(np.any(difference != 0, axis=2)))
        return {'same_dimensions': True, 'dimensions_pixels': list(size_x),
                'different_pixels': different, 'maximum_channel_difference': maximum,
                'pixel_exact': different == 0}


def verify(generated_dir, reference_dir):
    generated_dir, reference_dir = Path(generated_dir), Path(reference_dir)
    results = []
    for stem, folder in [('Figure_06', 'formal_stage'), ('Figure_S29', 'formal_stage'),
                         ('MWT_vertical_three_panel', '')]:
        for extension in ('png', 'tiff'):
            name = f'{stem}.{extension}'
            outcome = compare_images(generated_dir / folder / name, reference_dir / 'figures' / name)
            results.append({'file': name, **outcome})
    tile = compare_images(generated_dir / 'Fig6b_new_tile.png', reference_dir / 'source_assets/Fig6b_new_tile.png')
    results.append({'file': 'Fig6b_new_tile.png', **tile})
    return {'status': 'PASS' if all(r['pixel_exact'] for r in results) else 'DIFFERENCES',
            'scope': 'Saved-estimate plotting and raster composition only; no scientific refit or field calculation',
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.generated, args.reference)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':raise SystemExit(1)


if __name__ == '__main__':
    main()
