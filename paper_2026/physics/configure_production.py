"""Bind a copied production workspace to local raw files without scientific runs.

The saved coordinate hashes, native-grid shapes, variables, units, sizes and
bounded scalar probes remain acceptance criteria. Host inode/mtime values are
recorded only after those checks, never copied from the authors' machine.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text(x):return x.decode() if isinstance(x,bytes) else str(x)
def verify_field(item,raw):
 p=raw/Path(item['path'].replace('${RAW_ERA5}/','')).name
 if not p.is_file():raise FileNotFoundError(p)
 if p.stat().st_size!=item['size_bytes']:raise ValueError('Frozen native file size differs: '+str(p))
 with h5py.File(p,'r') as f:
  v=f[item['variable']]
  if list(v.shape)!=item['shape']:raise ValueError('Native dimensions differ: '+str(p))
  if text(v.attrs.get('units',''))!=text(item['attributes'].get('units','')):raise ValueError('Native units differ: '+str(p))
  for name,ref in item['coordinates'].items():
   a=f[name][:]
   if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()!=ref['sha256_values']:raise ValueError('Native coordinate identity differs: '+str(p)+'/'+name)
  for ref in item.get('scalar_probes',[]):
   if not np.isclose(float(v[tuple(ref['index'])]),ref['value'],rtol=0,atol=1e-12,equal_nan=True):raise ValueError('Frozen scalar probe differs: '+str(p))
 return {'field':item['field'],'level_hpa':item['level_hpa'],'filename':p.name,'size':p.stat().st_size,'coordinates_and_scalar_probes':'PASS'}
def main(args):
 raw=Path(args.era5_root).resolve();templates=ROOT/'production';inventory=json.loads((templates/'mwt/reports/data_era5_inventory.json').read_text())
 selected=inventory[:args.limit] if args.limit else inventory
 checked=[verify_field(q,raw) for q in selected]
 receipt={'status':'CHECKED_SUBSET' if args.limit else 'CHECKED_COMPLETE_MWT_INVENTORY','checked':checked,'scope':'Read-only metadata/coordinate/bounded scalar-probe check; no fields calculated or full-file hash claimed'}
 if args.check_only:
  target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'status':receipt['status'],'files':len(checked)}));return
 if args.limit:raise ValueError('Partial verification cannot initialize a production workspace')
 out=Path(args.workspace).resolve()
 if out.exists():raise FileExistsError('Use a new workspace: '+str(out))
 shutil.copytree(templates,out)
 cfg=json.loads((out/'paths.json').read_text());cfg['raw_era5']=str(raw)
 if args.ibtracs:cfg['ibtracs']=str(Path(args.ibtracs).resolve())
 if args.ocean_shapefile:cfg['ocean_shapefile']=str(Path(args.ocean_shapefile).resolve())
 aliases={k:str((Path(v) if Path(v).is_absolute() else out/v).resolve()) for k,v in cfg.items()}
 def materialize(x):
  if isinstance(x,dict):return {k:materialize(v) for k,v in x.items()}
  if isinstance(x,list):return [materialize(v) for v in x]
  if isinstance(x,str):
   for k,v in aliases.items():x=x.replace('${'+k.upper()+'}',v.replace('\\','/'))
  return x
 oldhash={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in ['.py','.json']}
 for p in out.rglob('*.json'):
  q=materialize(json.loads(p.read_text(encoding='utf-8')));p.write_text(json.dumps(q,indent=2),encoding='utf-8')
 (out/'paths.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
 # PI's exact original coordinate identities and local binding metadata.
 u=out/'mechanism_v5/upper_profile_100_50_interpolation_v1';audit=u/'input_interpolation/four_source_input_audit.json';a=json.loads(audit.read_text());axes=np.load(u/'input_interpolation/source_coordinate_axes.npz')
 for rec in a['files']:
  p=Path(rec['identity']['path']);st=p.stat()
  if st.st_size!=rec['identity']['size']:raise ValueError('PI endpoint file size differs: '+str(p))
  with h5py.File(p,'r') as f:
   for axis in ['latitude','longitude','time']:
    if not np.array_equal(f[axis][:],axes[rec['key']+'__'+axis]):raise ValueError('PI endpoint coordinate mismatch')
  rec['identity'].update(mtime_ns=st.st_mtime_ns,file_id=st.st_ino)
 a['catalog_sha256']=sha(out/'mechanism_v5/config/available_variable_map_v5.json');audit.write_text(json.dumps(a,indent=2),encoding='utf-8')
 # Bind transported-code/configuration checks to the identical relocated files.
 repl={oldhash[str(p.relative_to(out))]:sha(p) for p in out.rglob('*') if p.is_file() and str(p.relative_to(out)) in oldhash and sha(p)!=oldhash[str(p.relative_to(out))]}
 for p in out.rglob('*.py'):
  s=p.read_text(encoding='utf-8')
  for old,new in repl.items():s=s.replace(old,new)
  p.write_text(s,encoding='utf-8')
 for sub in ['mwt/cache','mwt/logs','mwt/validation','mwt/statistics']:(out/sub).mkdir(parents=True,exist_ok=True)
 receipt.update(status='LOCAL_PRODUCTION_BINDING_READY',workspace=str(out),identity_fields='local inode/mtime recorded after coordinate/size/probe verification')
 (out/'LOCAL_INPUT_BINDING.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'status':receipt['status'],'workspace':str(out),'files':len(checked)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--era5-root',required=True);p.add_argument('--workspace');p.add_argument('--ibtracs');p.add_argument('--ocean-shapefile');p.add_argument('--check-only',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--report',default='raw_input_check.json');a=p.parse_args()
 if not a.check_only and not a.workspace:p.error('--workspace is required unless --check-only')
 main(a)
