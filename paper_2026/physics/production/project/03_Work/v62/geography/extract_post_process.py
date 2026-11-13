"""Subsequent along-track SCS 700-hPa budget and separate 850-hPa convergence.
Run only after the two pre-window extraction workers have released their locks.
"""
from extract_pre_environment_resume_atomic_v62 import *

POST_KEYS=['q700','u700','v700','q600','q850','w700','u850','v850','d850']

def post_one(row,cache):
    folder=O/'post_requests'/ex.digest(row)[:20];folder.mkdir(parents=True,exist_ok=True);cp=folder/'COMPLETE.json'
    if cp.exists():
        receipt=json.loads(cp.read_text());assert receipt['request']==clean(row) and sha(folder/'values.json')==receipt['values_sha256'];return json.loads((folder/'values.json').read_text())
    tick=time.time();nr=cache.new_raw_reads;nc=cache.cache_hits;utc=row['UTC'];lat=float(row['lat']);lon=float(row['lon'])
    b=cache.reference(utc,lat,lon,800.);fields={k:cache.get(k,utc,lat,lon,800.) for k in POST_KEYS}
    minus=cache.get('q700',pd.Timestamp(utc)-pd.Timedelta(hours=6),lat,lon,800.);plus=cache.get('q700',pd.Timestamp(utc)+pd.Timedelta(hours=6),lat,lon,800.)
    for z in [minus,plus]:assert np.array_equal(z.latitude_indices,b.latitude_indices) and np.array_equal(z.longitude_indices,b.longitude_indices)
    a={k:z.values for k,z in fields.items()};q=a['q700'];gx,gy=spherical_gradient(q,b.latitude,b.longitude)
    cu=float(row['motion_u_past6h_ms']);cv=float(row['motion_v_past6h_ms']);scale=86400000.
    dt=np.where(np.isfinite(q),(plus.values-minus.values)/43200.,np.nan)*scale
    motion=(cu*gx+cv*gy)*scale
    h=(-(a['u700']-cu)*gx-(a['v700']-cv)*gy)*scale
    vert=vertical_advection_700(a['q600'],q,a['q850'],a['w700'])*scale
    moving=dt+motion;res=moving-h-vert
    budget={'fixed_tendency':dt,'motion_term':motion,'moving_tendency':moving,'horizontal_advection':h,'horizontal_advection_fixed_frame':h-motion,'vertical_advection':vert,'residual':res}
    joint=np.isfinite(np.stack(list(budget.values()))).all(0)
    closure=float(np.max(abs((moving-h-vert-res)[joint]))) if joint.any() else np.nan
    mfc=-spherical_divergence(a['q850']*a['u850'],a['q850']*a['v850'],b.latitude,b.longitude)*scale
    joint850=np.isfinite(mfc)&np.isfinite(a['d850'])&np.isfinite(a['q850'])&np.isfinite(a['u850'])&np.isfinite(a['v850'])
    values=[]
    for radial,lo,hi in [('inner',100,300),('outer',300,800)]:
        domain=(b.distance_km>=lo)&(b.distance_km<hi)
        def put(k,v,units,support_family):
            st=area_weighted_support(v,b.cell_area_m2,domain,min_fraction=.9)
            values.append(dict(radial=radial,field=k,value=st['mean'],units=units,support_family=support_family,**{key:val for key,val in st.items() if key!='mean'}))
        put('q700',q*1000,'g kg-1','q700_only')
        for k,v in budget.items():put(k,np.where(joint,v,np.nan),'g kg-1 day-1','joint_700_budget')
        put('q850_absolute_MFC',np.where(joint850,mfc,np.nan),'g kg-1 day-1','joint_850')
        put('convergence850',np.where(joint850,-a['d850']*1e5,np.nan),'1e-5 s-1','joint_850')
    sources={k:z.metadata for k,z in fields.items()};sources.update(q700_minus6h=minus.metadata,q700_plus6h=plus.metadata)
    result=clean(dict(request=row,values=values,sources=sources,max_budget_identity_error=closure,new_raw_reads=cache.new_raw_reads-nr,cache_hits=cache.cache_hits-nc,elapsed_s=time.time()-tick,folder=str(folder)))
    dump(folder/'values.json',result);dump(cp,dict(status='COMPLETE',request=row,values_sha256=sha(folder/'values.json'),driver_sha256=sha(__file__),dependency_sha256=sha(O/'extract_pre_environment_resume_atomic_v62.py')))
    return result

def main():
    assert not (O/'pre_extraction_s0of2.lock').exists() and not (O/'pre_extraction_s1of2.lock').exists(),'pre workers still active; no overlapping extraction permitted'
    requests=read(O/'post_process_requests.csv');valid=requests[requests.observed_position.astype(str).str.lower().eq('true')].sort_values(['UTC','sid','lag_h'])
    lock=O/'post_extraction.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),UTC=ex.stamp(),argv=sys.argv),f)
    started=time.time();results=[];sem=json.loads((V/'config/extraction_semantics_v5.json').read_text())
    try:
        with ex.NativeFieldReader(V/'config/available_variable_map_v5.json',max_open=32,chunk_cache_bytes=2*1024**2) as reader:
            cache=Overlay(reader,O/'native_cache',sem)
            for i,row in enumerate(valid.to_dict('records')):
                result=post_one(clean(row),cache);results.append(result);reader.read_log.clear()
                with (O/'post_extraction_ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(request_id=row['request_id'],folder=result['folder'],status='COMPLETE'))+'\n')
                dump(O/'post_extraction_progress.json',dict(status='RUNNING',pid=os.getpid(),done=i+1,total=len(valid),elapsed_s=time.time()-started,raw_reads=cache.new_raw_reads,cache_hits=cache.cache_hits))
                if (i+1)%20==0:print('post_process',i+1,len(valid),'raw',cache.new_raw_reads,'seconds',round(time.time()-started,1),flush=True)
            rows=[dict(**z['request'],**v,source_folder=z['folder']) for z in results for v in z['values']]
            d=pd.DataFrame(rows);assert not d.duplicated(['request_id','radial','field']).any();d.to_csv(O/'post_process_native_sources.csv',index=False)
            rec=dict(status='COMPLETE',pid=os.getpid(),planned_requests=len(requests),actual_position_requests=len(valid),missing_position_requests=len(requests)-len(valid),rows=len(d),raw_reads=cache.new_raw_reads,cache_hits=cache.cache_hits,elapsed_s=time.time()-started,max_budget_identity_error=float(np.nanmax([z['max_budget_identity_error'] for z in results])),source_sha256={str(p):sha(p) for p in [Path(__file__),O/'extract_pre_environment_resume_atomic_v62.py',O/'post_process_requests.csv',O/'post_unique_native_requests.csv']},time_derivative='centered +/-6h at the same spatial grid; transformed with contemporaneous past6h storm motion',support='all 700 budget terms share finite native cells and >=0.9 original radial area; 850 terms separately share their own native cells')
            dump(O/'post_extraction_receipt.json',rec);print(json.dumps(rec),flush=True)
    except Exception:
        dump(O/'post_extraction_failure.json',dict(pid=os.getpid(),error=traceback.format_exc(),utc=ex.stamp()));raise
    finally:
        if lock.exists():lock.rename(lock.with_name('post_extraction_released_'+str(os.getpid())+'.json'))

if __name__=='__main__':main()
