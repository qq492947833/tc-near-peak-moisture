"""Incremental PI/VI from COMPLETE extraction caches only; never opens raw ERA5."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,os,sys,time,traceback
import numpy as np
from thermodynamics_v5 import pi_environment_profile,assemble_ventilation
ROOT=Path(__file__).resolve().parents[1]
def stamp():return datetime.now(timezone.utc).isoformat()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,Path):return str(x)
 return x
def canon(x):return json.dumps(clean(x),sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def boolean(x):
 if isinstance(x,(bool,np.bool_)):return bool(x)
 if str(x).strip().lower() in ('true','1'):return True
 if str(x).strip().lower() in ('false','0'):return False
 raise ValueError('Boolean not explicit: '+repr(x))
def number(x):
 try:return float(x)
 except (TypeError,ValueError):return np.nan
def utc(x):
 d=datetime.fromisoformat(str(x).replace('Z','+00:00'))
 if d.tzinfo is None:d=d.replace(tzinfo=timezone.utc)
 return d.astimezone(timezone.utc).replace(tzinfo=None).isoformat()+'Z'
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvsave(p,rr):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with tmp.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([clean(r) for r in rr])
 os.replace(tmp,p)
def wind_index(requests):
 ix={}
 for r in requests:
  if r['module']!='P1' or r['center_kind']!='actual_track':continue
  if boolean(r['position_extrapolated']) or not np.isfinite(number(r.get('wind_kt'))):continue
  key=(r['sid'],utc(r['UTC']));value=number(r['wind_kt'])
  if key in ix and ix[key]['wind']!=value:raise ValueError('Conflicting P1 contemporaneous wind '+repr(key))
  ix[key]=dict(wind=value,request_id=r['request_id'],UTC=key[1])
 return ix
def wind_for_margin(r,ix):
 t=utc(r['UTC']);v=number(r.get('wind_kt'))
 if r['center_kind']=='actual_track' and not boolean(r['position_extrapolated']) and np.isfinite(v):
  return dict(wind_for_headroom_kt=v,wind_source_UTC=t,wind_source_request_id=r['request_id'],wind_source_policy='current_request_actual_track')
 z=ix.get((r['sid'],t))
 if z:return dict(wind_for_headroom_kt=z['wind'],wind_source_UTC=t,wind_source_request_id=z['request_id'],wind_source_policy='same_SID_UTC_P1_actual_track_join')
 return dict(wind_for_headroom_kt=np.nan,wind_source_UTC=t,wind_source_request_id='',wind_source_policy='NOT_AVAILABLE_NO_t0_WIND_FALLBACK')
def gap_support(normal,lowest,expected_joint_area,root=ROOT):
 """Read immutable native caches; neither instantiate reader nor open source nc."""
 reader_sha=sha(root/'src/native_fields_v5.py')
 obj=dict(UTC=utc(normal['UTC']),latitude=float(normal['lat']),longitude=float(normal['lon'])%360,radius_km=800.,halo=1,reader_sha=reader_sha)
 gid=digest(obj)[:20];d=root/'cache/native_fields'/gid
 paths=[d/'sp.npz',d/f't{int(lowest)}.npz',d/f'q{int(lowest)}.npz']
 identities=[]
 for p in paths:
  meta=read(p.with_name('geometry.json') if p.name=='sp.npz' else p.with_suffix('.json'))
  actual=sha(p)
  if meta.get('npz_sha256')!=actual:raise ValueError('Native cache identity conflict '+str(p))
  identities.append(dict(path=str(p),sha256=actual))
 with np.load(paths[0],allow_pickle=False) as z:
  if json.loads(str(z['binding_json'].item()))['geometry']!=obj:raise ValueError('SP geometry binding mismatch')
  sp=z['sp'].copy();area=z['area'].copy();dist=z['distance'].copy()
 with np.load(paths[1],allow_pickle=False) as z:
  if json.loads(str(z['binding_json'].item()))['geometry']!=gid:raise ValueError('T geometry mismatch')
  t=z['values'].copy()
 with np.load(paths[2],allow_pickle=False) as z:
  if json.loads(str(z['binding_json'].item()))['geometry']!=gid:raise ValueError('q geometry mismatch')
  q=z['values'].copy()
 mask=(dist>=200)&(dist<800)&np.isfinite(sp)&(sp>=lowest*100)&np.isfinite(t)&np.isfinite(q)
 weights=area[mask];gaps=sp[mask]/100-lowest;wa=float(weights.sum())
 if not np.isclose(wa,expected_joint_area,rtol=1e-10,atol=1.):raise ValueError('Lowest-layer cached joint area differs from extracted profile')
 if not len(gaps):raise ValueError('No supported lowest-layer gap')
 order=np.argsort(gaps);cdf=np.cumsum(weights[order])/wa
 return dict(cache_geometry=gid,lowest_valid_pressure_hpa=float(lowest),valid_cells=len(gaps),joint_area_m2=wa,
  gap_mean_hpa=float(np.average(gaps,weights=weights)),gap_min_hpa=float(gaps.min()),gap_max_hpa=float(gaps.max()),
  gap_p10_hpa=float(np.interp(.1,cdf,gaps[order])),gap_p50_hpa=float(np.interp(.5,cdf,gaps[order])),gap_p90_hpa=float(np.interp(.9,cdf,gaps[order])),
  quantile_rule='linear interpolation of cumulative area-weighted ordered gaps',cache_identities=identities)
def checked_inputs(folder,launch,semantics,thermo_sha):
 donep=folder/'COMPLETE.json';done=read(donep)
 if done.get('status')!='COMPLETE':raise ValueError('Extraction checkpoint not COMPLETE')
 bind=done['binding'];ids=launch['source_identity']
 if bind['driver_sha']!=ids['driver_sha256']:raise ValueError('Extraction driver version mismatch')
 if bind['semantics_sha']!=digest(semantics):raise ValueError('Canonical extraction semantics mismatch')
 needed=['request.json','scalars.csv']
 if bind['normalised']['need_profile']:needed+=['profile.npz','profile.csv','entropy600_metadata.json']
 listed={x['name']:x for x in done['outputs']}
 for name in needed:
  if name not in listed or sha(folder/name)!=listed[name]['sha256']:raise ValueError('Consumed source identity mismatch: '+name)
 if read(folder/'request.json')['binding']!=bind:raise ValueError('Request/checkpoint binding differs')
 if bind['normalised']['need_profile']:
  e=read(folder/'entropy600_metadata.json')
  if e['thermodynamics_sha256']!=thermo_sha:raise ValueError('Entropy function identity changed since extraction')
  if e['pressure_pa']!=60000 or e['core_radius_km']!=[0,100] or e['environment_radius_km']!=[100,300]:raise ValueError('Entropy spatial definition conflict')
 return done,dict(checkpoint_sha256=sha(donep),consumed_files=[listed[n] for n in needed])
def derive_one(folder,launch,semantics,cfg,ix,root=ROOT):
 thermo_sha=sha(root/'src/thermodynamics_v5.py')
 done,source=checked_inputs(folder,launch,semantics,thermo_sha);r=done['binding']['request'];normal=done['binding']['normalised']
 binding=dict(source=source,request_folder=str(folder),config_sha256=sha(root/'config/thermodynamic_estimand_v5.json'),
  thermodynamics_sha256=thermo_sha,deriver_sha256=sha(__file__),requests_sha256=launch['source_identity']['requests_sha256'])
 outdir=root/'data/thermodynamic_derived'/digest(binding)[:20];cp=outdir/'COMPLETE.json'
 if cp.exists():
  prev=read(cp)
  if prev['binding']!=binding:raise ValueError('Derived binding conflict')
  for a in prev['outputs']:
   if sha(outdir/a['name'])!=a['sha256']:raise ValueError('Derived output checksum conflict')
  return read(outdir/'derived.json'),dict(status='REUSED_VERIFIED_CHECKPOINT',folder=str(outdir),request_id=r['request_id'])
 base={k:r.get(k,'') for k in ('request_id','module','sid','event_id','event_class','UTC','lag_h','center_kind','stage')}
 base.update(request_folder=str(folder),extraction_request_hash=folder.name,original_request_wind_kt=number(r.get('wind_kt')),**wind_for_margin(r,ix))
 base.update(pi_vmax_ms=np.nan,pi_valid=False,pi_quality_status='NOT_REQUESTED_PROFILE',thermal_headroom_ms=np.nan,
  raw_signed_index=np.nan,chi600=np.nan,entropy_deficit_jkgk=np.nan,sea_air_entropy_difference_jkgk=np.nan,
  signed_estimable=False,ventilation_index=np.nan,standard_ventilation_domain_valid=False,ventilation_quality_status='NOT_REQUESTED_PROFILE')
 support={};fullpi={};vent={}
 if normal['need_profile']:
  sr=rows(folder/'scalars.csv');sc={x['variable']:x for x in sr}
  if len(sc)!=len(sr):raise ValueError('Duplicate scalar variable')
  need=['sst_K','msl','shear_200_850_vector_magnitude_ms','core600_saturated_entropy_J_kgK','environment600_entropy_J_kgK']
  if any(x not in sc for x in need):raise ValueError('Missing required scalar fields')
  for name,unit in [('sst_K','K'),('msl','Pa'),('shear_200_850_vector_magnitude_ms','m s-1'),('core600_saturated_entropy_J_kgK','J kg-1 K-1'),('environment600_entropy_J_kgK','J kg-1 K-1')]:
   if sc[name]['units']!=unit:raise ValueError('Unexpected scalar unit '+name)
  support['scalars']={k:sc[k] for k in need}
  def valid(name):
   a=sc[name];return boolean(a['support_pass']) and number(a['valid_fraction'])>=.9 and np.isfinite(number(a['value']))
  def val(name):return number(sc[name]['value']) if valid(name) else np.nan
  with np.load(folder/'profile.npz',allow_pickle=False) as z:
   p=z['pressure_hpa'].copy();t=z['temperature_K'].copy();q=z['specific_humidity_kgkg'].copy();frac=z['joint_Tq_valid_fraction'].copy()
  profile=rows(folder/'profile.csv')
  if len(profile)!=len(p):raise ValueError('Profile csv/npz lengths differ')
  for a,pp,tt,qq,ff in zip(profile,p,t,q,frac):
   np.testing.assert_allclose([number(a[x]) for x in ('pressure_hpa','temperature_K','specific_humidity_kgkg','joint_Tq_valid_fraction')],[pp,tt,qq,ff],rtol=1e-12,atol=1e-12,equal_nan=True)
   if a['request_id']!=r['request_id']:raise ValueError('Profile request mismatch')
  support['profile']=profile
  fullpi=pi_environment_profile(t,q,p,val('sst_K'),val('msl'),frac,sp_qc_confirmed=True,
    original_area_denominator_confirmed=True,surface_support_confirmed=bool(valid('sst_K') and valid('msl')),
    config=cfg['pi_parameters'],support_metadata=dict(extraction_source=source,scalar_support=support['scalars']))
  low=fullpi.get('metadata',{}).get('lowest_valid_pressure_hpa')
  if low is not None:
   area=next(number(z['joint_valid_area_m2']) for z in profile if number(z['pressure_hpa'])==low)
   support['lowest_layer_surface_gap']=gap_support(normal,low,area,root)
  base.update(pi_vmax_ms=fullpi['vmax_ms'],pi_valid=fullpi['pi_valid'],pi_quality_status=fullpi['quality_status'],
   pi_iflag=fullpi.get('iflag'),pi_pmin_hpa=fullpi['pmin_hpa'],lowest_valid_pressure_hpa=low,
   rmw_pressure_used_hpa=fullpi['rmw_pressure_used_hpa'],rmw_pressure_next_hpa=fullpi['rmw_pressure_next_hpa'],
   sea_air_entropy_difference_jkgk=fullpi['sea_air_entropy_difference_jkgk'])
  if base['pi_valid'] and np.isfinite(base['wind_for_headroom_kt']):
   base['thermal_headroom_ms']=base['pi_vmax_ms']-base['wind_for_headroom_kt']*1852/3600
  vent=assemble_ventilation(val(need[2]),val(need[3]),val(need[4]),fullpi,
    support_confirmed=bool(all(valid(x) for x in need[2:])),support_metadata=support['scalars'])
  base.update(raw_signed_index=vent['raw_signed_index'],chi600=vent['chi600'],entropy_deficit_jkgk=vent['entropy_deficit_jkgk'],
    signed_estimable=bool(np.isfinite(vent['raw_signed_index'])),ventilation_index=vent['ventilation_index'],
    standard_ventilation_domain_valid=bool(vent['valid']),ventilation_quality_status=vent['quality_status'])
  base.update(shear_200_850_ms=val(need[2]),core600_saturated_entropy_jkgk=val(need[3]),environment600_entropy_jkgk=val(need[4]))
 # Recheck completion before publication: tolerate no source mutation during read.
 if sha(folder/'COMPLETE.json')!=source['checkpoint_sha256']:raise ValueError('Source changed during derivation')
 outdir.mkdir(parents=True,exist_ok=True)
 atomic(outdir/'derived.json',base);atomic(outdir/'support.json',support)
 atomic(outdir/'thermodynamic_details.json',dict(pi=fullpi,ventilation=vent))
 outputs=[dict(name=n,sha256=sha(outdir/n)) for n in ('derived.json','support.json','thermodynamic_details.json')]
 atomic(cp,dict(status='COMPLETE',binding=binding,finished_utc=stamp(),outputs=outputs,raw_ERA5_reads=0))
 return clean(base),dict(status='COMPLETE',folder=str(outdir),request_id=r['request_id'])
def launch_records(launch):
 found={}
 for w in launch['workers']:
  p=Path(w['stdout'])
  if not p.exists():continue
  with p.open(encoding='utf-8-sig',errors='replace') as f:
   for line in f:
    try:r=json.loads(line)
    except json.JSONDecodeError:continue  # incomplete final line retried on next pass
    candidates=r.get('records',[]) if isinstance(r,dict) else []
    if isinstance(r,dict) and 'folder' in r:candidates=[r]+candidates
    for x in candidates:
     if x.get('status') in ('COMPLETE','REUSED_VERIFIED_CHECKPOINT') and x.get('folder'):found[x['folder']]=x
 return list(found)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--launch',type=Path,required=True);ap.add_argument('--watch-seconds',type=int,default=0);ap.add_argument('--poll-seconds',type=int,default=20);args=ap.parse_args()
 launch=read(args.launch);sem=read(ROOT/'config/extraction_semantics_v5.json');cfg=read(ROOT/'config/thermodynamic_estimand_v5.json')
 ids=launch['source_identity'];request_path=Path(ids['requests'])
 if sha(request_path)!=ids['requests_sha256']:raise ValueError('Request table differs from launched version')
 if sha(ROOT/'config/extraction_semantics_v5.json')!=ids['semantics_sha256']:raise ValueError('Extraction semantics raw-file identity differs')
 if sha(ROOT/'src/extract_environment_v5.py')!=ids['driver_sha256']:raise ValueError('Running extraction driver identity differs')
 if sha(ROOT/'src/thermodynamics_v5.py')!=cfg['provenance']['thermodynamics_source_sha256']:raise ValueError('Frozen thermodynamics source changed')
 ix=wind_index(rows(request_path));runid=datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+str(os.getpid());aud=ROOT/'audit'/('derive_thermodynamics_'+runid);aud.mkdir(parents=True)
 # OS-managed lock: no stale-PID deletion and no duplicate derived worker for a launch.
 lock_path=ROOT/'checkpoints'/('thermodynamic_'+sha(args.launch)[:16]+'.lock');lock_path.parent.mkdir(exist_ok=True)
 lock=lock_path.open('a+b');lock.seek(0)
 if os.name=='nt':
  import msvcrt
  if lock_path.stat().st_size==0:lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 started=stamp();end=time.monotonic()+args.watch_seconds;seen=set();table=[];events=[];failures=[]
 atomic(aud/'STARTED.json',dict(pid=os.getpid(),argv=[sys.executable]+sys.argv,started_utc=started,launch=str(args.launch)))
 try:
  while True:
   for source in launch_records(launch):
    if source in seen:continue
    try:
     row,ev=derive_one(Path(source),launch,sem,cfg,ix);table.append(row);events.append(ev);seen.add(source)
     print(canon(dict(utc=stamp(),**ev)),flush=True)
    except Exception as exc:
     error=dict(source=source,error=repr(exc),traceback=traceback.format_exc(),utc=stamp());failures.append(error);seen.add(source)
     print(canon(dict(status='FAILED',**error)),flush=True)
   csvsave(aud/'derived_requests.csv',table)
   atomic(aud/'progress.json',dict(updated_utc=stamp(),derived=len(table),failures=len(failures),raw_ERA5_reads=0))
   if time.monotonic()>=end:break
   time.sleep(min(args.poll_seconds,max(.1,end-time.monotonic())))
  result=dict(status='COMPLETE_INCREMENTAL_PASS' if not failures else 'COMPLETE_WITH_FAILURES',started_utc=started,finished_utc=stamp(),
   argv=[sys.executable]+sys.argv,launch=str(args.launch),derived=len(table),events=events,failures=failures,raw_ERA5_reads=0,exit_code=1 if failures else 0)
  atomic(aud/'execution.json',result);print(canon(result),flush=True)
 finally:lock.close()
 return 1 if failures else 0
if __name__=='__main__':sys.exit(main())
