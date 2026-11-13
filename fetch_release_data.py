"""Fetch and verify the processed inputs and source assets for this release.

No raw ERA5 archive or manuscript document is downloaded. Existing matching
files are reused; a different local file is never silently overwritten.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil, tempfile, urllib.request, zipfile

ROOT = Path(__file__).resolve().parent

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def verify_and_extract(archive, spec, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    if sha256(archive) != spec['sha256']:
        raise ValueError('Archive checksum mismatch: ' + archive.name)
    members = {v['path']: v for v in spec['members']}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(members):
            raise ValueError('Archive members differ from manifest')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path: ' + name)
            item = members[name]
            h = hashlib.sha256()
            with z.open(name) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                    h.update(chunk)
            if z.getinfo(name).file_size != item['bytes'] or h.hexdigest() != item['sha256']:
                raise ValueError('Member checksum mismatch: ' + name)
            target = destination.joinpath(*p.parts)
            if destination not in target.resolve().parents:
                raise ValueError('Destination escapes root: ' + name)
            if target.exists() and sha256(target) != item['sha256']:
                raise FileExistsError('Different local input, refusing to overwrite: ' + str(target))
        for name in names:
            target = destination.joinpath(*PurePosixPath(name).parts)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, target.open('xb') as out:
                shutil.copyfileobj(source, out, 4 * 1024**2)
            if sha256(target) != members[name]['sha256']:
                raise ValueError('Extracted file checksum mismatch: ' + name)
    return len(names)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'release_data_manifest.json')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--archive', type=Path, action='append', help='Local versioned ZIP; repeat for both assets')
    parser.add_argument('--asset', action='append', help='Select a named asset from the manifest')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    assets = manifest['assets']
    if args.asset:
        assets = [x for x in assets if x['name'] in args.asset]
        if len(assets) != len(set(args.asset)):
            parser.error('Unknown or duplicate asset name')
    args.destination.mkdir(parents=True, exist_ok=True)
    local = {p.name: p for p in args.archive or []}
    for spec in assets:
        if args.archive and spec['name'] not in local:
            continue
        path = local.get(spec['name'])
        if path is None:
            folder = args.destination/'.downloads'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/spec['name']
            if not path.exists() or sha256(path) != spec['sha256']:
                part = path.with_suffix(path.suffix+'.part')
                with urllib.request.urlopen(spec['url'], timeout=120) as response, part.open('wb') as out:
                    shutil.copyfileobj(response, out, 4 * 1024**2)
                if sha256(part) != spec['sha256']:
                    raise ValueError('Downloaded archive checksum mismatch')
                part.replace(path)
        count = verify_and_extract(path, spec, args.destination)
        print(f'{spec["name"]}: verified and available ({count} files)')
    if args.archive and not set(local).issubset({x['name'] for x in manifest['assets']}):
        parser.error('Unrecognized archive filename')

if __name__ == '__main__':
    main()
