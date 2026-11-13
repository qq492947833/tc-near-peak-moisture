"""Local repair for q700 already cached at1500km then exactly subset to800km.
The active original deriver is unchanged. Only previously failed sources are selected.
"""
from pathlib import Path
from datetime import datetime
import argparse,json,sys,os,traceback
import numpy as np
import derive_thermodynamics_v5 as d
EXPECTED_BASE_SHA='8f3116746eb03d834f97119532384b28807b4f24b1fc986719e09b7fdf9881ce'
BASE_PATH=Path(d.__file__)
if d.sha(BASE_PATH)!=EXPECTED_BASE_SHA:raise RuntimeError('Original deriver dependency changed')
ORIGINAL_GAP=d.gap_support
def exact_subset(source_lat,source_lon,target_lat,target_lon,values):
 li={int(x):i for i,x in enumerate(source_lat)};lj={int(x):i for i,x in enumerate(source_lon)}
 ii=[li[int(x)] for x in target_lat];jj=[lj[int(x)] for x in target_lon]
 return values[np.ix_(ii,jj)]
def gap_support(normal,lowest,expected_joint_area,root=d.ROOT):
 obj=dict(UTC=d.utc(normal['UTC']),latitude=float(normal['lat']),longitude=float(normal['lon'])%360,radius_km=800.,halo=1,reader_sha=d.sha(root/'src/native_fields_v5.py'))
 gid=d.digest(obj)[:20];small=root/'cache/native_fields'/gid
 if (small/f'q{int(lowest)}.npz').exists():return ORIGINAL_GAP(normal,lowest,expected_joint_area,root)
 if lowest!=700 or not normal['need_map']:raise FileNotFoundError('Only documented map-backed q700 is repairable without raw reads')
 largeobj=dict(obj,radius_km=1500.);lgid=d.digest(largeobj)[:20];large=root/'cache/native_fields'/lgid
 paths=[small/'sp.npz',small/'t700.npz',large/'sp.npz',large/'q700.npz'];identities=[]
 for p in paths:
  meta=d.read(p.with_name('geometry.json') if p.name=='sp.npz' else p.with_suffix('.json'));h=d.sha(p)
  if meta['npz_sha256']!=h:raise ValueError('Cache identity conflict '+str(p))
  identities.append(dict(path=str(p),sha256=h))
 with np.load(paths[0]) as z:
  if json.loads(str(z['binding_json'].item()))['geometry']!=obj:raise ValueError('Small geometry mismatch')
  sp=z['sp'];area=z['area'];dist=z['distance'];il=z['latitude_indices'];jl=z['longitude_indices']
 with np.load(paths[1]) as z:
  if json.loads(str(z['binding_json'].item()))['geometry']!=gid:raise ValueError('T geometry mismatch')
  t=z['values']
 with np.load(paths[2]) as z:
  if json.loads(str(z['binding_json'].item()))['geometry']!=largeobj:raise ValueError('Large geometry mismatch')
  ilbig=z['latitude_indices'];jlbig=z['longitude_indices']
 with np.load(paths[3]) as z:
  if json.loads(str(z['binding_json'].item()))['geometry']!=lgid:raise ValueError('q map geometry mismatch')
  q=exact_subset(ilbig,jlbig,il,jl,z['values'])
 mask=(dist>=200)&(dist<800)&np.isfinite(sp)&(sp>=lowest*100)&np.isfinite(t)&np.isfinite(q)
 weights=area[mask];gaps=sp[mask]/100-lowest;wa=float(weights.sum())
 if not np.isclose(wa,expected_joint_area,rtol=1e-10,atol=1.):raise ValueError('Repaired subset joint support differs from profile')
 if not len(gaps):raise ValueError('No supported lowest-layer gap')
 order=np.argsort(gaps);cdf=np.cumsum(weights[order])/wa
 return dict(cache_geometry=gid,lowest_valid_pressure_hpa=float(lowest),valid_cells=len(gaps),joint_area_m2=wa,
  gap_mean_hpa=float(np.average(gaps,weights=weights)),gap_min_hpa=float(gaps.min()),gap_max_hpa=float(gaps.max()),
  gap_p10_hpa=float(np.interp(.1,cdf,gaps[order])),gap_p50_hpa=float(np.interp(.5,cdf,gaps[order])),gap_p90_hpa=float(np.interp(.9,cdf,gaps[order])),
  quantile_rule='linear interpolation of cumulative area-weighted ordered gaps',cache_identities=identities,
  mapping='q700 from1500km cache, exact native integer-index subset to800km; no interpolation or raw source read',
  base_deriver_sha256=EXPECTED_BASE_SHA,repair_implementation_sha256=d.sha(__file__))
def repair_one(folder,launch,sem,cfg,ix):
 oldfile=d.__file__;oldgap=d.gap_support
 try:
  d.__file__=__file__;d.gap_support=gap_support
  return d.derive_one(folder,launch,sem,cfg,ix)
 finally:d.__file__=oldfile;d.gap_support=oldgap
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--failure-log',type=Path,required=True);ap.add_argument('--launch',type=Path,required=True);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
 sources={}
 for line in args.failure_log.read_text(encoding='utf-8').splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if r.get('status')=='FAILED' and 'q700.json' in r.get('traceback',''):sources[r['source']]=r
 launch=d.read(args.launch);cfg=d.read(d.ROOT/'config/thermodynamic_estimand_v5.json');sem=d.read(d.ROOT/'config/extraction_semantics_v5.json')
 rp=Path(launch['source_identity']['requests'])
 if d.sha(rp)!=launch['source_identity']['requests_sha256']:raise ValueError('Request identity conflict')
 ix=d.wind_index(d.rows(rp));started=d.stamp();run=d.ROOT/'audit'/('thermodynamic_gap_repair_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+str(os.getpid()));run.mkdir()
 events=[];results=[];errors=[]
 for src,prior in list(sources.items())[:args.limit or None]:
  try:
   row,ev=repair_one(Path(src),launch,sem,cfg,ix);results.append(row);events.append(dict(**ev,original_failure=prior))
  except Exception as exc:errors.append(dict(source=src,error=repr(exc),traceback=traceback.format_exc()))
 d.csvsave(run/'derived_requests_repaired.csv',results)
 rec=dict(started_utc=started,finished_utc=d.stamp(),command=[sys.executable]+sys.argv,status='PASS' if not errors else 'FAIL',n_repaired=len(results),errors=errors,events=events,raw_ERA5_reads=0,original_failure_log=str(args.failure_log),original_failure_log_snapshot_sha256=d.sha(args.failure_log),base_implementation_sha256=EXPECTED_BASE_SHA,repair_implementation_sha256=d.sha(__file__),exit_code=bool(errors))
 d.atomic(run/'execution.json',rec);print(str(run));print(d.canon({k:v for k,v in rec.items() if k not in ('events','errors')}));print(d.canon(errors))
 return bool(errors)
if __name__=='__main__':sys.exit(main())
