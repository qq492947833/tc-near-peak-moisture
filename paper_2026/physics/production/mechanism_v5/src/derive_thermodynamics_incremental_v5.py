"""Incremental wrapper retaining original accepted derivation identities.
Known 1500-to800 q700 cache mappings use the separately versioned repair; no source edits.
"""
from pathlib import Path
import sys,json,traceback
import derive_thermodynamics_v5 as d
import derive_thermodynamics_gap_repair_v5 as repair
original=d.derive_one
def main():
 argv=list(sys.argv);known=set()
 if '--known-failure-log' in sys.argv:
  i=sys.argv.index('--known-failure-log');p=Path(sys.argv[i+1]);del sys.argv[i:i+2]
  for line in p.read_text(encoding='utf-8').splitlines():
   try:z=json.loads(line)
   except ValueError:continue
   if z.get('status')=='FAILED' and 'q700.json' in z.get('traceback',''):known.add(z['source'])
 def dispatch(folder,*args,**kwargs):
  if str(folder) not in known:
   try:return original(folder,*args,**kwargs)
   except FileNotFoundError as exc:
    if not str(exc.filename).endswith('q700.json'):raise
    print(d.canon(dict(status='LOCAL_CACHE_MAPPING_REPAIR',source=str(folder),original_error=repr(exc),traceback=traceback.format_exc(),utc=d.stamp())),flush=True)
  oldfile=d.__file__;oldgap=d.gap_support
  try:
   d.__file__=repair.__file__;d.gap_support=repair.gap_support
   return original(folder,*args,**kwargs)
  finally:d.__file__=oldfile;d.gap_support=oldgap
 d.derive_one=dispatch
 print(d.canon(dict(status='CACHE_MAPPING_WRAPPER_ATTACHED',actual_argv=[sys.executable]+argv,wrapper_sha256=d.sha(__file__),base_deriver_sha256=d.sha(d.__file__),repair_sha256=d.sha(repair.__file__),known_failed_sources=len(known))),flush=True)
 return d.main()
if __name__=='__main__':sys.exit(main())
