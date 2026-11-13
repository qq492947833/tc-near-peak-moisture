from process_v6 import *
from rh_native_v6 import geometry,loadz,checked,means
from types import SimpleNamespace

def main():
 x=pd.read_csv(O/'centre_native_process_sources.csv',low_memory=False);meta=x[['request_id','sid','event_id','event_class','lag_h','center_kind','source_folder']].drop_duplicates();req=pd.read_csv(V/'data/p1_environment_requests_v5.csv').set_index('request_id')
 tree=ast.parse((V/'src/native_fields_v5.py').read_text(encoding='utf-8'));names={'_as_float','_field_grid','spherical_divergence'};env={'np':np,'EARTH_RADIUS_M':6371008.8};exec(compile(ast.Module(body=[z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name in names],type_ignores=[]),'native_pure','exec'),env)
 rows=[];missing=[];tick=time.perf_counter()
 for i,r in enumerate(meta.itertuples()):
  if not isinstance(r.source_folder,str):continue
  rq=req.loc[r.request_id];geo=geometry(SimpleNamespace(**rq.to_dict()));ng=O/'native_cache'/geo.name;g=loadz((ng if (ng/'sp.npz').exists() else geo)/'sp.npz');a={};mp=None
  try:
   for key in ['q850','u850','v850','d850']:
    p=(ng if (ng/(key+'.npz')).exists() else geo)/(key+'.npz')
    if p.exists():a[key]=checked(p)
    else:
     if mp is None:mp=loadz(Path(r.source_folder)/'native_map_1500km.npz');il={int(k):j for j,k in enumerate(mp['latitude_indices'])};jl={int(k):j for j,k in enumerate(mp['longitude_indices'])};ix=np.ix_([il[int(k)] for k in g['latitude_indices']],[jl[int(k)] for k in g['longitude_indices']])
     a[key]=mp[key][ix]
   mfc=-env['spherical_divergence'](a['q850']*a['u850'],a['q850']*a['v850'],g['latitude'],g['longitude'])*86400000.;conv=-a['d850']*1e5;joint=np.isfinite(mfc)&np.isfinite(conv);ring=(g['distance']>=200)&(g['distance']<800)
   for field,arr,unit in [('q850_absolute_MFC',mfc,'g kg-1 day-1'),('convergence850',conv,'1e-5 s-1')]:
    v,frac=means(np.where(joint,arr,np.nan),g['area'],ring);rows.append(dict(request_id=r.request_id,sid=r.sid,event_id=r.event_id,event_class=r.event_class,lag_h=r.lag_h,center_kind=r.center_kind,field=field,value=v,units=unit,valid_fraction=frac,valid_area_m2=float(g['area'][ring&joint].sum()),original_area_m2=float(g['area'][ring].sum())))
  except (FileNotFoundError,KeyError) as e:missing.append(dict(request_id=r.request_id,missing=str(e)))
  if i%1000==0:print('850',i,len(meta),round(time.perf_counter()-tick),flush=True)
 raw=pd.DataFrame(rows);raw.to_csv(O/'centre_850_request_sources.csv',index=False);pairs=[]
 for (sid,field,unit),g in raw.groupby(['sid','field','units']):
  cols=pd.MultiIndex.from_product([['internal','final'],['actual_track','fixed_t0']]);a=g.pivot(index='lag_h',columns=['event_class','center_kind'],values='value').reindex(index=range(-24,25,6),columns=cols).to_numpy(copy=True).reshape(9,2,2);common=np.isfinite(a).all((1,2));a[~common]=np.nan
  for j,policy in enumerate(['all_along','all_fixed']):
   y=a[:,:,j];vals={'pre_mean':np.nanmean(y[:4],0) if common[:4].sum()>=3 else [np.nan,np.nan],'integral_0_24':np.trapezoid(y[4:],dx=.25,axis=0) if common[4:].all() else [np.nan,np.nan]}
   for name,v in vals.items():pairs.append(dict(sid=sid,field=field,policy=policy,estimand=name,units=unit if name=='pre_mean' else 'g kg-1' if field=='q850_absolute_MFC' else '1e-5 s-1 day',internal=v[0],final=v[1],value=v[0]-v[1]))
 pd.DataFrame(pairs).to_csv(O/'centre_850_SID_pairs.csv',index=False);tab,piv,d,w=summary_table(pairs,['field','policy','estimand','units']);tab.to_csv(O/'centre_850_estimates.csv',index=False);np.savez_compressed(O/'centre_850_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)
 js(O/'centre_850_receipt.json',{'status':'COMPLETE','rows':len(raw),'missing':missing,'raw_reads':0,'seconds':time.perf_counter()-tick,'MFC_definition':'minus spherical divergence of native q850 times absolute horizontal850 wind; includes advective and convergence terms','convergence_definition':'minus archived d850; both850 diagnostics share finite native support','layer_support':'independent of700 budget; never combined into700 closure','integration':'dx=.25day; rate integral g/kg'})
 print('850 complete',len(tab),len(missing),flush=True)

if __name__=='__main__':main()
