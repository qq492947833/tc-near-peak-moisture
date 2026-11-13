from pathlib import Path
import json,hashlib
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
