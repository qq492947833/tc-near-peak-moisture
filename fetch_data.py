"""Fetch and verify the processed inputs for this fixed release."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, urllib.request, zipfile

ROOT = Path(__file__).absolute().parent

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def fetch(archive=None):
    manifest = json.loads((ROOT / 'data_manifest.json').read_text(encoding='utf8'))
    asset = manifest['asset']
    path = Path(archive) if archive else ROOT / asset['name']
    if archive is None and not path.exists():
        temporary = path.with_suffix('.part')
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'tc-near-peak-moisture'})
        with urllib.request.urlopen(request, timeout=120) as source, open(temporary, 'wb') as target:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                target.write(block)
        if digest(temporary) != asset['sha256']:
            raise ValueError('Download checksum mismatch')
        os.replace(temporary, path)
    assert path.stat().st_size == asset['bytes'], 'Archive size mismatch'
    assert digest(path) == asset['sha256'], 'Archive checksum mismatch'
    expected = {r['path']: r for r in manifest['files']}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and set(names) == set(expected), 'Archive membership mismatch'
        for name in names:
            parts = PurePosixPath(name).parts
            assert parts[0] == 'source_data' and '..' not in parts and '\\' not in name and ':' not in name
            data = z.read(name)
            row = expected[name]
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], name
        for name in names:
            target = ROOT.joinpath(*PurePosixPath(name).parts)
            if target.exists():
                assert digest(target) == expected[name]['sha256'], f'Existing input differs: {name}'
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
    print(f'Verified {len(expected)} processed input files.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive')
    fetch(parser.parse_args().archive)
