"""Resolve portable production paths; no scientific transformations."""
from pathlib import Path
import os,json
def release_path(key, suffix=""):
    root=next(p for p in Path(__file__).resolve().parents if (p/"paths.json").is_file())
    cfg=json.loads((root/"paths.json").read_text(encoding="utf-8"))
    value=os.environ.get("FLTCCF_"+key.upper(),cfg[key])
    path=Path(value).expanduser()
    if not path.is_absolute():path=root/path
    return path/suffix
