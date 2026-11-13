from _release_paths import release_path as _release_path
"""Bounded rise-window environment extraction; original sources stay read-only."""
import os,sys,json,hashlib,time,datetime,argparse,traceback,shutil
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
W=Path(__file__).absolute().parents[1]
os.environ['NUMBA_CACHE_DIR']=str(W/'cache/numba')
V=Path(str(_release_path('mechanism_v5','')))
BASE=_release_path('project')
G=BASE/'03_Work/v62/geography'
sys.path.insert(0,str(G))
import extract_pre_environment as old
# The historical module chooses a cache directory on import; isolate this run.
os.environ['NUMBA_CACHE_DIR']=str(W/'cache/numba')
import numpy as np,pandas as pd
from dataclasses import replace
from native_fields_v5 import spherical_gradient,vertical_advection_700,area_weighted_support,great_circle_distance_km,longitude_difference,EARTH_RADIUS_M
ex=old.ex
ROOTS=[V/'cache/native_fields',BASE/'03_Work/v6/physics/native_cache',G/'native_cache',Path(str(_release_path('legacy_project','v67_review_20261014/M5/native_cache')))]

class MissingNativeCache(RuntimeError):pass
class RiseOverlay(old.Overlay):
    roots=ROOTS
    def __init__(self,*args,cache_only=False,**kw):
        super().__init__(*args,**kw);self.cache_only=cache_only
    def reference(self,utc,lat,lon,radius):
        geo,_=self.geometry(utc,lat,lon,radius)
        if self.cache_only and not any((root/geo/'sp.npz').exists() and (root/geo/'geometry.json').exists() for root in [self.root]+self.roots):
            raise MissingNativeCache(f'{geo}/sp|{utc}|{lat}|{lon}')
        return super().reference(utc,lat,lon,radius)
    def get(self,key,utc,lat,lon,radius=800.):
        geo,_=self.geometry(utc,lat,lon,radius)
        if self.cache_only and not any((root/geo/(key+'.npz')).exists() and (root/geo/(key+'.json' if key!='sp' else 'geometry.json')).exists() for root in [self.root]+self.roots):
            raise MissingNativeCache(f'{geo}/{key}|{utc}|{lat}|{lon}')
        return super().get(key,utc,lat,lon,radius)

def atomic(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(ex.clean(obj),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    start=time.monotonic()
    while True:
        try:os.replace(tmp,p);return
        except PermissionError:
            if time.monotonic()-start>60:raise
            time.sleep(.1)

def budget(row,cache):
    utc=row['UTC'];lat=row['lat'];lon=row['lon'];b=cache.reference(utc,lat,lon,800.)
    fields={k:cache.get(k,utc,lat,lon) for k in ['q700','u700','v700','q600','q850','w700']}
    minus=cache.get('q700',pd.Timestamp(utc)-pd.Timedelta(hours=6),lat,lon);plus=cache.get('q700',pd.Timestamp(utc)+pd.Timedelta(hours=6),lat,lon)
    for x in [minus,plus]:
        if not(np.array_equal(x.latitude_indices,b.latitude_indices) and np.array_equal(x.longitude_indices,b.longitude_indices)):raise ValueError('Temporal derivative native coordinates differ')
    a={k:x.values for k,x in fields.items()};q=a['q700'];gx,gy=spherical_gradient(q,b.latitude,b.longitude)
    cu=row['motion_u_past6h_ms'];cv=row['motion_v_past6h_ms'];scale=86400000.
    dt=np.where(np.isfinite(q),(plus.values-minus.values)/43200.,np.nan)*scale
    motion=(cu*gx+cv*gy)*scale;h=(-(a['u700']-cu)*gx-(a['v700']-cv)*gy)*scale
    v=vertical_advection_700(a['q600'],q,a['q850'],a['w700'])*scale;m=dt+motion
    values=dict(fixed_tendency=dt,motion_term=motion,moving_tendency=m,horizontal_advection=h,horizontal_advection_fixed_frame=h-motion,vertical_advection=v,residual=m-h-v)
    joint=np.isfinite(np.stack(list(values.values()))).all(0);out=[]
    for radial,lo,hi in [('inner',100,300),('outer',300,800)]:
        domain=(b.distance_km>=lo)&(b.distance_km<hi)
        for metric,val in [('q700',q*1000)]+[(k,np.where(joint,vv,np.nan)) for k,vv in values.items()]:
            st=area_weighted_support(val,b.cell_area_m2,domain,min_fraction=.9)
            out.append(dict(radial=radial,metric=metric,value=st.pop('mean'),units='g kg-1' if metric=='q700' else 'g kg-1 day-1',**st))
    return ex.clean({'values':out,'geometry_id':cache.geometry(utc,lat,lon,800.)[0], 'sources':{k:x.metadata for k,x in fields.items()},'temporal_sources':{'minus6h':minus.metadata,'plus6h':plus.metadata}})

def reuse_current_environment(row,prior,cache,masks,folder,sem):
    """Reuse accepted C diagnostics only after source and sampling verification."""
    source=Path(prior['source_complete_json']);pi_source=Path(prior['PI_source_complete_json'])
    assert ex.filehash(source)==prior['source_complete_sha256']
    assert ex.filehash(pi_source)==prior['PI_source_complete_sha256']
    complete=json.loads(source.read_text());pc=json.loads(pi_source.read_text())
    assert complete['status']=='COMPLETE' and complete['binding']['semantics_sha']==ex.digest(sem)
    assert pc['binding']['source']['checkpoint_sha256']==prior['source_complete_sha256']
    assert prior['PI_source_version']=='C_14_logp_inclusive50'
    for name in ['configuration','route_receipt']:
        h='configuration_sha256' if name=='configuration' else 'route_sha256'
        assert ex.filehash(pc['binding']['implementation'][name])==pc['binding']['implementation'][h]
    for item in complete['outputs']:
        assert ex.filehash(source.parent/item['name'])==item['sha256']
    for name,h in pc['outputs'].items():assert ex.filehash(pi_source.parent/name)==h
    oldrow=complete['binding']['normalised'];request=complete['binding']['request']
    assert request['sid']==row['sid'] and pd.Timestamp(oldrow['UTC'])==pd.Timestamp(row['UTC'])
    assert abs(float(oldrow['lat'])-row['lat'])<5e-12
    assert abs((float(oldrow['lon'])-row['lon']+180)%360-180)<5e-12
    before=cache.reference(row['UTC'],float(oldrow['lat']),float(oldrow['lon']),800.)
    # Equal physical native cells can have different last-bit longitude wrapping.
    # Reconstruct selection from source coordinates without a second raw SP read.
    glat,glon=cache.reader._coordinates('sp');delta=800000./EARTH_RADIUS_M;deg=np.rad2deg(delta)
    il=np.flatnonzero((glat>=max(-90,row['lat']-deg))&(glat<=min(90,row['lat']+deg)))
    if not len(il):il=np.array([int(np.argmin(abs(glat-row['lat'])))])
    il=np.arange(max(0,int(il.min())-1),min(len(glat),int(il.max())+2))
    half=180. if abs(row['lat'])+deg>=90 else np.rad2deg(np.arcsin(np.clip(np.sin(delta)/np.cos(np.deg2rad(row['lat'])),0,1)))
    rel=longitude_difference(glon,row['lon']);order=np.argsort(rel);hit=np.flatnonzero(abs(rel[order])<=half)
    if not len(hit):hit=np.array([int(np.argmin(abs(rel[order])))])
    jl=order[max(0,int(hit.min())-1):min(len(glon),int(hit.max())+2)]
    assert np.array_equal(il,before.latitude_indices) and np.array_equal(jl,before.longitude_indices)
    distance=great_circle_distance_km(before.latitude[:,None],before.longitude[None,:],row['lat'],row['lon'])
    for lo,hi in [(0,100),(100,300),(200,800),(300,800)]:
        assert np.array_equal((before.distance_km>=lo)&(before.distance_km<hi),(distance>=lo)&(distance<hi))
    oldvalues=json.loads((source.parent/'request_values.json').read_text())
    ps=json.loads((pi_source.parent/'summary.json').read_text())
    c=ps['configurations']['C_14_logp_inclusive50'];pi=c['authoritative']
    values={};support={};units={}
    for name,key,unit,skey in [('SST','sst_C','degC','sst_C'),('midRH','mid_rh_pct','%','mid_rh_pct'),('shear200_850','shear_200_850_vector_magnitude_ms','m s-1','shear_200_850_u_ms')]:
        values[name]=oldvalues[key];units[name]=unit
        support[name]={'support_pass':oldvalues.get(skey+'__support_pass'),'valid_fraction':oldvalues.get(skey+'__valid_fraction'),'inherited_source':str(source)}
    values['PI']=pi['pi_vmax_ms'];values['PI_headroom']=None if pi['pi_vmax_ms'] is None else pi['pi_vmax_ms']-row['wind_kt']*1852./3600.
    for name in ['PI','PI_headroom']:
        units[name]='m s-1';support[name]={'support_pass':pi['pi_valid'],'inherited_source':str(pi_source),'physical_reasons':c['physical_reasons']}
    assert pi['pi_valid'] or values['PI'] is None
    folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(pi_source.parent/'profile_inputs.npz',folder/'profiles.npz')
    shutil.copyfile(pi_source.parent/'C_D_full_trace.json.gz',folder/'PI_trace.json.gz')
    atomic(folder/'PI_summary.json',ps)
    reuse={'environment_complete':str(source),'environment_sha256':ex.filehash(source),'PI_complete':str(pi_source),'PI_sha256':ex.filehash(pi_source),'PI_version':'C_14_logp_inclusive50','same_native_indices_weights_SP_and_annulus_masks':True,'old_coordinate':[oldrow['lat'],oldrow['lon']],'new_coordinate':[row['lat'],row['lon']],'headroom_wind_kt':row['wind_kt'],'sampling_coordinate_rule':'Original native cell indices, area weights and SP at identical UTC; at most 5e-12 degree coordinate serialization difference, with exact annulus-mask equality; no interpolation.'}
    atomic(folder/'reuse_sources.json',reuse)
    result=ex.clean({'request':row,'geometry_id':cache.geometry(row['UTC'],float(oldrow['lat']),float(oldrow['lon']),800.)[0],'values':values,'units':units,'support':support,'pi_valid':pi['pi_valid'],'PI_status':c['status'],'reuse':reuse})
    atomic(folder/'values.json',result)
    return result

def process_create_time():
    import ctypes
    from ctypes import wintypes
    created=wintypes.FILETIME();exited=wintypes.FILETIME();kernel=wintypes.FILETIME();user=wintypes.FILETIME()
    fn=ctypes.windll.kernel32.GetProcessTimes
    fn.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    if not fn(wintypes.HANDLE(-1),ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel),ctypes.byref(user)):raise ctypes.WinError()
    return ((created.dwHighDateTime<<32)+created.dwLowDateTime-116444736000000000)/10000000.

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-only',action='store_true');ap.add_argument('--limit',type=int);ap.add_argument('--request-id');ap.add_argument('--assemble-only',action='store_true');ap.add_argument('--reuse-only',action='store_true');ap.add_argument('--shard-count',type=int,default=1);ap.add_argument('--shard-index',type=int,default=0);ap.add_argument('--max-open',type=int,default=64);args=ap.parse_args()
    assert args.shard_count>=1 and 0<=args.shard_index<args.shard_count
    import ctypes
    from ctypes import wintypes
    setpriority=ctypes.windll.kernel32.SetPriorityClass;setpriority.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    if not setpriority(wintypes.HANDLE(-1),0x4000):raise ctypes.WinError()
    suffix='' if args.shard_count==1 else '_s'+str(args.shard_index)+'of'+str(args.shard_count)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];kernel.CreateMutexW.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.WaitForSingleObject.restype=ctypes.c_uint32
    handle=kernel.CreateMutexW(None,False,'Local\\FLTCCF_RISE_ENV_'+str(args.shard_index)+'_OF_'+str(args.shard_count))
    if not handle or kernel.WaitForSingleObject(handle,0) not in (0,0x80):raise RuntimeError('An environment worker already owns this fixed shard')
    source=W/'source_data/rise_unique_point_requests.csv';d=pd.read_csv(source,keep_default_na=False,na_values=['','nan','NaN']);d=d[d.position_available].sort_values(['UTC','sid'])
    if args.shard_count>1:d=d[[int(ex.digest(s),16)%args.shard_count==args.shard_index for s in d.sid]]
    prior=pd.read_csv(W/'source_data/rise_current_environment_reuse_index.csv',keep_default_na=False,na_values=['','nan','NaN']).set_index('request_id_new').to_dict('index')
    if args.reuse_only:d=d[d.request_id.isin(prior)]
    if args.request_id:d=d[d.request_id.eq(args.request_id)]
    if args.limit:d=d.head(args.limit)
    out=W/'cache/rise_environment';out.mkdir(parents=True,exist_ok=True);cachepath=W/'cache/rise_native';cachepath.mkdir(parents=True,exist_ok=True)
    ident=dict(pid=os.getpid(),create_time=process_create_time(),command=sys.argv,started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),cache_only=args.cache_only,selected=len(d),source_sha256=ex.filehash(source),driver_sha256=ex.filehash(__file__),priority='BELOW_NORMAL',shard_count=args.shard_count,shard_index=args.shard_index)
    atomic(W/'reports'/('rise_environment_worker_identity'+suffix+'.json'),ident)
    sem=json.loads((V/'config/extraction_semantics_v5.json').read_text());masks=old.Masks(Path(sem['sst']['shapefile']))
    records=[];allvalues=[];start=time.monotonic()
    with ex.NativeFieldReader(V/'config/available_variable_map_v5.json',max_open=args.max_open,chunk_cache_bytes=4*1024**2) as reader:
        cache=RiseOverlay(reader,cachepath,sem,cache_only=args.cache_only)
        for i,x in enumerate(d.to_dict('records')):
            row=dict(request_id=x['request_id'],sid=x['sid'],UTC=pd.Timestamp(x['UTC']).isoformat(),lat=float(x['lat']),lon=float(x['lon']),wind_kt=float(x['wind_kt']),motion_u_past6h_ms=float(x['motion_u_ms']),motion_v_past6h_ms=float(x['motion_v_ms']))
            folder=out/ex.digest(row)[:20];cp=folder/'RISE_COMPLETE.json'
            try:
                if cp.exists():
                    z=json.loads(cp.read_text());assert z['request']==ex.clean(row)
                    for n,h in z['outputs_sha256'].items():assert ex.filehash(folder/n)==h
                    v=json.loads((folder/'values.json').read_text());b=json.loads((folder/'budget.json').read_text())
                else:
                    if args.assemble_only:raise MissingNativeCache('requested result not yet completed')
                    if row['request_id'] in prior:
                        v=reuse_current_environment(row,prior[row['request_id']],cache,masks,folder,sem)
                        names=['values.json','PI_trace.json.gz','PI_summary.json','profiles.npz','reuse_sources.json','budget.json']
                    else:
                        v=old.calculate(row,cache,masks,out);names=['values.json','PI_trace.json','profiles.npz','budget.json']
                    sample_row=row
                    if 'reuse' in v:
                        sample_row=dict(row,lat=float(v['reuse']['old_coordinate'][0]),lon=float(v['reuse']['old_coordinate'][1]))
                    b=budget(sample_row,cache);atomic(folder/'budget.json',b)
                    atomic(cp,{'request':row,'status':'COMPLETE','outputs_sha256':{n:ex.filehash(folder/n) for n in names},'driver_sha256':ex.filehash(__file__),'historical_environment_driver_sha256':ex.filehash(old.__file__),'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()})
                for name,value in v['values'].items():
                    if name in ['SST','midRH','shear200_850','PI','PI_headroom'] or name.startswith('q700_'):
                        allvalues.append(dict(**row,metric=name,radial='200-800' if name in ['SST','midRH','shear200_850','PI','PI_headroom'] else name.rsplit('_',1)[-1],value=value,units=v['units'][name],support=v['support'].get(name,{}),source=str(folder)))
                for q in b['values']:allvalues.append(dict(**row,**q,source=str(folder)))
                records.append({'request_id':row['request_id'],'status':'COMPLETE','folder':str(folder)})
            except MissingNativeCache as exc:records.append({'request_id':row['request_id'],'status':'DEFERRED_RAW_FIELD_REQUIRED','detail':str(exc)})
            except Exception as exc:
                records.append({'request_id':row['request_id'],'status':'FAILED','detail':str(exc),'traceback':traceback.format_exc()})
            reader.read_log.clear()
            rec=dict(**ident,updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_points=i+1,counts=pd.Series([v['status'] for v in records]).value_counts().to_dict(),raw_reads=cache.new_raw_reads,cache_hits=cache.cache_hits,elapsed_s=time.monotonic()-start,last_request=row['request_id'])
            atomic(W/'reports'/('rise_environment_status'+suffix+'.json'),rec)
            if (i+1)%10==0 or i+1==len(d):print(json.dumps(rec),flush=True)
        if args.cache_only and cache.new_raw_reads:raise AssertionError('Cache-only mode read raw ERA5')
        tag=('cache_only' if args.cache_only else 'production')+('_reuse' if args.reuse_only else '')+('_'+str(args.limit) if args.limit else '')+('_request_'+ex.digest(args.request_id)[:12] if args.request_id else '')+suffix
        pd.DataFrame(records).to_csv(W/'source_data'/('rise_environment_ledger_'+tag+'.csv'),index=False)
        if allvalues:pd.DataFrame(allvalues).to_csv(W/'source_data'/('rise_environment_values_'+tag+'.csv'),index=False,encoding='utf-8-sig')
        atomic(W/'reports'/('rise_environment_receipt_'+tag+'.json'),dict(**rec,status='CACHE_PASS_COMPLETE' if args.cache_only else 'EXTRACTION_PASS_COMPLETE',output_rows=len(allvalues),complete_scope=len(records)==len(d),records=records))
if __name__=='__main__':main()
