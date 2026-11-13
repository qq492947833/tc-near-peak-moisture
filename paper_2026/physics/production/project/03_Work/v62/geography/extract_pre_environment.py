from _release_paths import release_path as _release_path
"""Exact P3 pre-window fields; old native caches read-only, missing fields local overlay.

The inherited SP, original-area, derivative and accepted-C PI functions are called
without changes. This driver omits unrelated surface flux/map requests.
"""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,traceback,shutil
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent;W6=O.parents[1]/'v6'
V=Path(str(_release_path('mechanism_v5','')));U=V/'upper_profile_100_50_interpolation_v1'
os.environ['NUMBA_CACHE_DIR']=str(O/'numba_cache')
sys.path.insert(0,str(V/'src'));sys.path.insert(0,str(U/'production_candidate_v1'));sys.path.insert(0,str(U/'input_interpolation'))
import numpy as np,pandas as pd
from dataclasses import replace
import extract_environment_v5 as ex
from native_fields_v5 import area_weighted_support,spherical_gradient,spherical_divergence,vertical_advection_700
from production_inputs import prepare13,add70
from production_thermodynamics import evaluate
from interpolation_core import joint70
from thermodynamics_v5 import entropy600_fields

def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],low_memory=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):return ex.clean(x)
def dump(p,x):ex.atomic_json(p,x)
LEVELS=np.array([1000,925,850,700,600,500,400,300,250,200,150,100,50])
KEYS=list(dict.fromkeys(['sst','u200','v200','u500','v500','u850','v850','r500','r700','vo850','u700','v700','q700','q600','q850','w700','d850','msl']+[x+str(l) for x in ['t','q'] for l in LEVELS]))

class Overlay(ex.ExtractionCache):
    roots=[V/'cache/native_fields',W6/'physics/native_cache']
    def reference(self,utc,lat,lon,radius):
        geo,_=self.geometry(utc,lat,lon,radius);new=self.root/geo
        if geo in self.refs:return self.refs[geo]
        if not (new/'sp.npz').exists():
            for root in self.roots:
                old=root/geo
                if (old/'sp.npz').exists() and (old/'geometry.json').exists():
                    m=json.loads((old/'geometry.json').read_text());assert sha(old/'sp.npz')==m['npz_sha256']
                    new.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/'sp.npz',new/'sp.npz');shutil.copyfile(old/'geometry.json',new/'geometry.json');break
        return super().reference(utc,lat,lon,radius)
    def get(self,key,utc,lat,lon,radius=800.):
        geo,_=self.geometry(utc,lat,lon,radius);self.verify_source_identity(key)
        for root in self.roots:
            old=root/geo;p=old/(key+'.npz');j=old/(key+'.json')
            if key!='sp' and p.exists() and j.exists():
                b=self.reference(utc,lat,lon,radius);m=json.loads(j.read_text());assert sha(p)==m['npz_sha256']
                with np.load(p,allow_pickle=False) as z:
                    assert json.loads(str(z['binding_json'].item()))==m['binding'];a=z['values'].copy();raw=z['raw'].copy();mask=z['support_mask'].copy()
                cfg=self.reader.map[key];expected=dict(key=key,source_identity=cfg.get('source_identity'),metadata_sha=cfg.get('metadata_sha256'),geometry=geo,mask='same native cell/time SP>=pressurePa' if cfg.get('pressure_level_hpa') is not None else 'no SP layer mask')
                assert m['binding']==clean(expected)
                if cfg.get('pressure_level_hpa') is not None:
                    desired=np.isfinite(raw)&np.isfinite(b.sp_pa)&(b.sp_pa>=cfg['pressure_level_hpa']*100)
                    assert np.array_equal(mask,desired) and np.array_equal(a,np.where(desired,raw,np.nan),equal_nan=True)
                self.cache_hits+=1
                return replace(b,key=key,variable=cfg['variable'],units=cfg['units'],pressure_level_pa=None if cfg.get('pressure_level_hpa') is None else cfg['pressure_level_hpa']*100,raw=raw,values=a,support_mask=mask,metadata=dict(cache=str(p),npz_sha256=m['npz_sha256'],**m['binding']))
        return super().get(key,utc,lat,lon,radius)

class Masks:
    def __init__(self,path):
        import shapefile
        from shapely.geometry import shape
        from shapely.ops import unary_union
        self.land=unary_union([shape(s.__geo_interface__) for s in shapefile.Reader(str(path)).shapes()]);self.cache={}
    def ocean(self,b):
        from shapely import contains_xy
        key=ex.digest([b.latitude_indices.tolist(),b.longitude_indices.tolist()])
        if key not in self.cache:
            la,lo=np.meshgrid(b.latitude,b.longitude,indexing='ij');self.cache[key]=~contains_xy(self.land,(lo+180)%360-180,la)
            if len(self.cache)>8:self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

def calculate(row,cache,masks,out):
    ident=ex.digest(row)[:20];folder=out/ident;folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'COMPLETE.json'
    if cp.exists():
        r=json.loads(cp.read_text());assert r['request']==clean(row) and sha(folder/'values.json')==r['values_sha256'];return json.loads((folder/'values.json').read_text())
    tick=time.time();nr=cache.new_raw_reads;nc=cache.cache_hits
    utc=row['UTC'];lat=float(row['lat']);lon=float(row['lon']);b=cache.reference(utc,lat,lon,800.);geo,_=cache.geometry(utc,lat,lon,800.)
    fields={k:cache.get(k,utc,lat,lon,800.) for k in KEYS};a={k:z.values for k,z in fields.items()}
    ring=(b.distance_km>=200)&(b.distance_km<800);inner=(b.distance_km>=100)&(b.distance_km<300);outer=(b.distance_km>=300)&(b.distance_km<800);ocean=masks.ocean(b)
    values={};support={};units={}
    def mean(key,z,unit,domain=ring):
        st=area_weighted_support(z,b.cell_area_m2,domain,min_fraction=.9);values[key]=st['mean'];support[key]={k:v for k,v in st.items() if k!='mean'};units[key]=unit;return st['mean']
    for name,hi,lo in [('shear200_850',200,850),('shear500_850',500,850),('shear200_500',200,500)]:
        us=a['u'+str(hi)]-a['u'+str(lo)];vs=a['v'+str(hi)]-a['v'+str(lo)];joint=np.isfinite(us)&np.isfinite(vs)
        x=area_weighted_support(np.where(joint,us,np.nan),b.cell_area_m2,ring,min_fraction=.9);y=area_weighted_support(np.where(joint,vs,np.nan),b.cell_area_m2,ring,min_fraction=.9)
        values[name]=float(np.hypot(x['mean'],y['mean']));units[name]='m s-1';support[name]={k:v for k,v in x.items() if k!='mean'}
    mean('midRH',(a['r500']+a['r700'])/2,'%');mean('cyclonic_vo850',np.sign(lat)*a['vo850']*1e5,'1e-5 s-1')
    mean('SST',a['sst']-273.15,'degC',ring&ocean)
    for name,domain in [('inner',inner),('outer',outer)]:
        mean('q700_'+name,a['q700']*1000,'g kg-1',domain)
    gx,gy=spherical_gradient(a['q700'],b.latitude,b.longitude)
    cu=float(row['motion_u_past6h_ms']);cv=float(row['motion_v_past6h_ms'])
    h=-(a['u700']-cu)*gx-(a['v700']-cv)*gy
    vert=vertical_advection_700(a['q600'],a['q700'],a['q850'],a['w700'])
    # Derivative diagnostics use original-domain support including both u/v and q gradients.
    joint=np.isfinite(a['q700'])&np.isfinite(a['u700'])&np.isfinite(a['v700'])&np.isfinite(gx)&np.isfinite(gy)
    h=np.where(joint,h,np.nan)
    mfc=-spherical_divergence(a['q850']*a['u850'],a['q850']*a['v850'],b.latitude,b.longitude)
    mfc=np.where(np.isfinite(a['q850'])&np.isfinite(a['u850'])&np.isfinite(a['v850']),mfc,np.nan)
    for name,domain in [('inner',inner),('outer',outer)]:
        mean('q700_hadv_'+name,h*1000*86400,'g kg-1 day-1',domain);mean('q700_vadv_'+name,vert*1000*86400,'g kg-1 day-1',domain);mean('q850_MFC_'+name,mfc*1000*86400,'g kg-1 day-1',domain)
    T=[];Q=[];tf=[];qf=[];jf=[]
    for l in LEVELS:
        t,q=a['t'+str(l)],a['q'+str(l)];joint=np.isfinite(t)&np.isfinite(q)
        ts=area_weighted_support(np.where(joint,t,np.nan),b.cell_area_m2,ring,min_fraction=.9);qs=area_weighted_support(np.where(joint,q,np.nan),b.cell_area_m2,ring,min_fraction=.9)
        T.append(ts['mean']);Q.append(qs['mean']);jf.append(ts['valid_fraction']);tf.append(area_weighted_support(t,b.cell_area_m2,ring,min_fraction=.9)['valid_fraction']);qf.append(area_weighted_support(q,b.cell_area_m2,ring,min_fraction=.9)['valid_fraction'])
    arrays=dict(pressure_hpa=LEVELS,temperature_K=np.array(T),specific_humidity_kgkg=np.array(Q),temperature_original_valid_fraction=np.array(tf),humidity_original_valid_fraction=np.array(qf),joint_Tq_valid_fraction=np.array(jf));prepared=prepare13(arrays)
    for method,label in [('logp','logp'),('linear_p','plinear')]:
        t,q,j=joint70(a['t100'],a['t50'],a['q100'],a['q50'],method);ts=area_weighted_support(t,b.cell_area_m2,ring,min_fraction=.9);qs=area_weighted_support(q,b.cell_area_m2,ring,min_fraction=.9);assert ts['valid_area_m2']==qs['valid_area_m2'];aug=add70(prepared,ts['mean'],qs['mean'],ts['valid_fraction']);arrays.update({'T_'+label+'_K':aug['temperature_K'],'q_'+label+'_kgkg':aug['specific_humidity_kgkg'],'pressure_aug_hpa':aug['pressure_hpa'],'joint_fraction_aug':aug['joint_Tq_valid_fraction']})
    ent=entropy600_fields(a['t600'],a['q600'],b.sp_pa)
    ss=area_weighted_support(a['sst'],b.cell_area_m2,ring&ocean,min_fraction=.9)['mean'];msl=area_weighted_support(a['msl'],b.cell_area_m2,ring,min_fraction=.9)['mean']
    core=area_weighted_support(ent['saturated_entropy_jkgk'],b.cell_area_m2,b.distance_km<100,min_fraction=.9)['mean'];env=area_weighted_support(ent['entropy_jkgk'],b.cell_area_m2,inner,min_fraction=.9)['mean']
    scalars=dict(SST_K=ss,msl_Pa=msl,shear_200_850_ms=values['shear200_850'],core600_saturated_entropy_jkgk=core,environment600_entropy_jkgk=env,ventilation_support_confirmed=bool(np.isfinite([values['shear200_850'],core,env]).all()))
    trace=evaluate(arrays,scalars,float(row['wind_kt']));pi=trace['C_14_logp_inclusive50']['authoritative']
    values['PI_headroom']=pi['thermal_headroom_ms'];values['PI']=pi['pi_vmax_ms'];units.update(PI_headroom='m s-1',PI='m s-1');support['PI_headroom']=dict(support_pass=pi['pi_valid'],valid_fraction=float(np.min(jf)))
    result=dict(request=row,geometry_id=geo,values=values,support=support,units=units,PI_status=trace['C_14_logp_inclusive50']['status'],pi_valid=pi['pi_valid'],sources={k:z.metadata for k,z in fields.items()},new_raw_reads=cache.new_raw_reads-nr,cache_hits=cache.cache_hits-nc,elapsed_s=time.time()-tick,folder=str(folder))
    dump(folder/'values.json',result);dump(folder/'PI_trace.json',trace);np.savez_compressed(folder/'profiles.npz',**arrays)
    dump(cp,dict(status='COMPLETE',request=row,values_sha256=sha(folder/'values.json'),PI_trace_sha256=sha(folder/'PI_trace.json'),profiles_sha256=sha(folder/'profiles.npz'),driver_sha256=sha(__file__)))
    return clean(result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);ap.add_argument('--validate-t0',action='store_true');ap.add_argument('--shard-count',type=int,default=1);ap.add_argument('--shard-index',type=int,default=0);args=ap.parse_args();assert 0<=args.shard_index<args.shard_count
    spec=json.loads((O.parent/'analysis_spec_v62.json').read_text());assert spec['module_C']['status']=='FROZEN_FOR_EXECUTION'
    e=read(W6/'causal/cohort_recovered.csv');r=read(V/'data/p3_environment_request_sources.csv');r=r[r.event_id.isin(e.event_id)&r.lag_h.isin([0] if args.validate_t0 else [-24,-18,-12,-6])].copy()
    r=r.sort_values(['UTC','sid','lag_h']);r=r.iloc[:args.limit or None]
    if args.shard_count>1:
        identities=read(O/'native_cache_presence.csv').set_index('request_id').geometry_id
        r=r[[int(identities.loc[k],16)%args.shard_count==args.shard_index for k in r.request_id]]
    tag='' if args.shard_count==1 else '_s'+str(args.shard_index)+'of'+str(args.shard_count)
    out=O/('validation_t0' if args.validate_t0 else 'pre_requests');out.mkdir(exist_ok=True)
    progress=O/('validation_extraction_progress.json' if args.validate_t0 else 'pre_extraction_progress'+tag+'.json');ledger=O/('validation_extraction_ledger.jsonl' if args.validate_t0 else 'pre_extraction_ledger'+tag+'.jsonl')
    lock=O/('validation_extraction.lock' if args.validate_t0 else 'pre_extraction'+tag+'.lock')
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),UTC=ex.stamp(),argv=sys.argv),f)
    sem=json.loads((V/'config/extraction_semantics_v5.json').read_text());masks=Masks(sem['sst']['shapefile']);started=time.time();results=[]
    try:
        with ex.NativeFieldReader(V/'config/available_variable_map_v5.json',max_open=48,chunk_cache_bytes=2*1024**2) as reader:
            cache=Overlay(reader,O/'native_cache',sem)
            for i,row in enumerate(r.to_dict('records')):
                z=calculate(clean(row),cache,masks,out);results.append(z)
                with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(request_id=row['request_id'],folder=z['folder'],status='COMPLETE'))+'\n')
                dump(progress,dict(status='RUNNING',pid=os.getpid(),done=i+1,total=len(r),elapsed_s=time.time()-started,raw_reads=cache.new_raw_reads,cache_hits=cache.cache_hits,last_request=row['request_id'],last_seconds=z['elapsed_s']))
                reader.read_log.clear()
                if (i+1)%10==0:print('extract',i+1,len(r),'raw',cache.new_raw_reads,'seconds',round(time.time()-started,1),flush=True)
            rows=[]
            for z in results:
                for k,v in z['values'].items():rows.append(dict(**z['request'],variable=k,value=v,units=z['units'][k],valid_fraction=z['support'].get(k,{}).get('valid_fraction'),support_pass=z['support'].get(k,{}).get('support_pass'),geometry_id=z['geometry_id'],source_folder=z['folder']))
            pd.DataFrame(rows).to_csv(O/('validation_t0_values.csv' if args.validate_t0 else 'pre_environment_sources'+tag+'.csv'),index=False)
            receipt=dict(status='COMPLETE',pid=os.getpid(),n_requests=len(results),n_rows=len(rows),raw_reads=cache.new_raw_reads,cache_hits=cache.cache_hits,elapsed_s=time.time()-started,science_dependencies={str(p):sha(p) for p in [V/'src/native_fields_v5.py',V/'src/extract_environment_v5.py',U/'production_candidate_v1/production_thermodynamics.py',U/'input_interpolation/interpolation_core.py']},driver_sha256=sha(__file__),spec_module_hash=spec['module_hashes']['C'] if 'C' in spec['module_hashes'] else spec['module_hashes'])
            dump(O/('validation_t0_receipt.json' if args.validate_t0 else 'pre_environment_receipt'+tag+'.json'),receipt);print(json.dumps(receipt),flush=True)
    except Exception:
        dump(O/'extraction_failure'+tag+'.json',dict(pid=os.getpid(),error=traceback.format_exc(),time=ex.stamp()));raise
    finally:
        if lock.exists():lock.rename(lock.with_name(lock.stem+'_released_'+str(os.getpid())+'.json'))
if __name__=='__main__':main()
