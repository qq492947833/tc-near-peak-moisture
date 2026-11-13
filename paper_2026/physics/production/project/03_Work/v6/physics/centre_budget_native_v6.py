from process_v6 import *
from rh_native_v6 import geometry,loadz,checked,means
from types import SimpleNamespace

def main():
 x=pd.read_csv(O/'centre_native_process_sources.csv',low_memory=False);meta=x[['request_id','sid','event_id','event_class','lag_h','center_kind','source_folder']].drop_duplicates()
 req=pd.read_csv(V/'data/p1_environment_requests_v5.csv').set_index('request_id');along=req[req.center_kind.eq('actual_track')].set_index(['sid','event_id','lag_h'])
 tree=ast.parse((V/'src/native_fields_v5.py').read_text(encoding='utf-8'));names={'_as_float','_field_grid','spherical_gradient','storm_relative_advection','vertical_advection_700'};env={'np':np,'EARTH_RADIUS_M':6371008.8};exec(compile(ast.Module(body=[z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name in names],type_ignores=[]),'native_pure','exec'),env)
 rows=[];errors=[];missing=[];t=time.perf_counter()
 for i,r in enumerate(meta.itertuples()):
  if not isinstance(r.source_folder,str):continue
  rq=req.loc[r.request_id];ar=along.loc[(r.sid,r.event_id,int(r.lag_h))]
  if not np.isfinite(ar.motion_u_ms+ar.motion_v_ms):continue
  rd=SimpleNamespace(**rq.to_dict());geo=geometry(rd);ng=O/'native_cache'/geo.name;g=loadz((ng if (ng/'sp.npz').exists() else geo)/'sp.npz');mp=None
  def field(key,lag=0):
   nonlocal mp
   row=SimpleNamespace(**rq.to_dict());row.UTC=pd.Timestamp(row.UTC)+pd.Timedelta(hours=lag);gg=geometry(row);nn=O/'native_cache'/gg.name;p=(nn if (nn/(key+'.npz')).exists() else gg)/(key+'.npz')
   if p.exists():return checked(p)
   if lag!=0:raise FileNotFoundError(p)
   if mp is None:mp=loadz(Path(r.source_folder)/'native_map_1500km.npz')
   il={int(k):j for j,k in enumerate(mp['latitude_indices'])};jl={int(k):j for j,k in enumerate(mp['longitude_indices'])};ix=np.ix_([il[int(k)] for k in g['latitude_indices']],[jl[int(k)] for k in g['longitude_indices']]);return mp[key][ix]
  try:
   q=field('q700');gx,gy=env['spherical_gradient'](q,g['latitude'],g['longitude']);p=Path(r.source_folder)/'native_process_800km.npz'
   if p.exists():
    z=loadz(p);h0=z['q700_horizontal']*86400000.;dt=z['q700_fixed_tendency']*86400000.;vert=z['q700_vertical']*86400000.
   else:
    h0=env['storm_relative_advection'](q,field('u700'),field('v700'),rq.motion_u_ms,rq.motion_v_ms,g['latitude'],g['longitude'])*86400000.
    dt=np.where(np.isfinite(q),(field('q700',6)-field('q700',-6))/43200.,np.nan)*86400000.
    vert=env['vertical_advection_700'](field('q600'),q,field('q850'),field('w700'))*86400000.
   motion=(ar.motion_u_ms*gx+ar.motion_v_ms*gy)*86400000.;motion0=(rq.motion_u_ms*gx+rq.motion_v_ms*gy)*86400000.
   h=h0+motion-motion0;moving=dt+motion;res=moving-h-vert
   fields={'moving_tendency':moving,'fixed_tendency':dt,'motion_term':motion,'horizontal_advection':h,'horizontal_advection_c0':h0,'vertical_advection':vert,'residual':res,'horizontal_advection_fixed_frame':h-motion,'legacy_motion_term':motion0}
   joint=np.isfinite(np.stack(list(fields.values()))).all(0);ring=(g['distance']>=200)&(g['distance']<800)
   if joint.any():errors.append(float(np.nanmax(abs((moving-h-vert-res)[joint]))))
   for k,a in fields.items():
    val,frac=means(np.where(joint,a,np.nan),g['area'],ring)
    rows.append(dict(request_id=r.request_id,sid=r.sid,event_id=r.event_id,event_class=r.event_class,lag_h=r.lag_h,center_kind=r.center_kind,field=k,value=val,units='g kg-1 day-1',valid_fraction=frac,valid_area_m2=float(g['area'][ring&joint].sum()),original_area_m2=float(g['area'][ring].sum())))
  except (FileNotFoundError,KeyError) as e:missing.append(dict(request_id=r.request_id,missing=str(e)))
  if i%500==0:print('centre budget',i,len(meta),round(time.perf_counter()-t),flush=True)
 raw=pd.DataFrame(rows);raw.to_csv(O/'centre_budget_request_sources.csv',index=False);pd.DataFrame(missing).to_csv(O/'centre_budget_missing_cache.csv',index=False)
 pairs=[]
 for (sid,field),g in raw.groupby(['sid','field']):
  cols=pd.MultiIndex.from_product([['internal','final'],['actual_track','fixed_t0']]);a=g.pivot(index='lag_h',columns=['event_class','center_kind'],values='value').reindex(index=range(-24,25,6),columns=cols).to_numpy(copy=True).reshape(9,2,2);common=np.isfinite(a).all((1,2));a[~common]=np.nan
  for j,policy in enumerate(['all_along','all_fixed']):
   y=a[:,:,j]
   vals={'pre_mean':np.nanmean(y[:4],0) if common[:4].sum()>=3 else [np.nan,np.nan],'integral_0_24':np.trapezoid(y[4:],dx=.25,axis=0) if common[4:].all() else [np.nan,np.nan]}
   for name,v in vals.items():pairs.append(dict(sid=sid,field=field,policy=policy,estimand=name,internal=v[0],final=v[1],value=v[0]-v[1],units='g kg-1' if name.startswith('integral') else 'g kg-1 day-1'))
 pd.DataFrame(pairs).to_csv(O/'centre_budget_SID_pairs.csv',index=False);tab,piv,d,w=summary_table(pairs,['field','policy','estimand','units']);tab.to_csv(O/'centre_budget_estimates.csv',index=False);np.savez_compressed(O/'centre_budget_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)
 js(O/'centre_budget_receipt.json',{'status':'COMPLETE','request_rows':len(meta),'diagnostic_rows':len(raw),'missing_cache':missing,'raw_reads':0,'seconds':time.perf_counter()-t,'max_budget_identity_error':max(errors),'common_support':'all nine terms share finite native cells; both centres and events share SID-lag support; contemporaneous actual velocity standardizes storm-relative terms','integration':'five6h instantaneous samples, dx=.25day, rate integral g/kg'})
 print('centre budget complete',len(tab),len(missing),flush=True)

if __name__=='__main__':main()
