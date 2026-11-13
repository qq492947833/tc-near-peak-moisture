from process_v6 import *
from rh_native_v6 import geometry,loadz,checked,means
from centre_complete_v6 import FIELDS
from types import SimpleNamespace

def main():
 while not (O/'centre_complete_receipt.json').exists():time.sleep(5)
 x=pd.read_csv(O/'centre_complete_process_sources.csv',low_memory=False)
 req=pd.read_csv(V/'data/p1_environment_requests_v5.csv');rr=req.set_index('request_id');along=req[req.center_kind.eq('actual_track')].set_index(['sid','event_id','lag_h'])
 tree=ast.parse((V/'src/native_fields_v5.py').read_text(encoding='utf-8'));names={'_as_float','_field_grid','spherical_gradient','storm_relative_advection'};env={'np':np,'EARTH_RADIUS_M':6371008.8};exec(compile(ast.Module(body=[z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name in names],type_ignores=[]),'native_pure','exec'),env)
 audit=[];areas=[]
 for idx,r in x[x.center_kind.eq('fixed_t0')&x.variable.eq('q700_storm_relative_horizontal_advection')].iterrows():
  rq=rr.loc[r.request_id];ar=along.loc[(r.sid,r.event_id,int(r.lag_h))];geo=geometry(SimpleNamespace(**rq.to_dict()));ng=O/'native_cache'/geo.name
  g=loadz((ng if (ng/'sp.npz').exists() else geo)/'sp.npz');a={};mp=None
  for key in ['q700','u700','v700']:
   p=(ng if (ng/(key+'.npz')).exists() else geo)/(key+'.npz')
   if p.exists():a[key]=checked(p)
   else:
    if mp is None:mp=loadz(Path(r.source_folder)/'native_map_1500km.npz');il={int(k):j for j,k in enumerate(mp['latitude_indices'])};jl={int(k):j for j,k in enumerate(mp['longitude_indices'])};ix=np.ix_([il[int(k)] for k in g['latitude_indices']],[jl[int(k)] for k in g['longitude_indices']])
    a[key]=mp[key][ix]
  h=env['storm_relative_advection'](a['q700'],a['u700'],a['v700'],ar.motion_u_ms,ar.motion_v_ms,g['latitude'],g['longitude'])*86400000.
  ring=(g['distance']>=200)&(g['distance']<800);val,frac=means(h,g['area'],ring);audit.append(dict(sid=r.sid,event_id=r.event_id,lag_h=r.lag_h,scalar_corrected=r.value,native_corrected=val,difference=val-r.value,valid_fraction=frac,valid_area_m2=float(g['area'][ring&np.isfinite(h)].sum()),original_area_m2=float(g['area'][ring].sum())))
  x.loc[idx,'value']=val
 for rid,g in x.groupby('request_id'):
  r=g.iloc[0]
  if not isinstance(r.source_folder,str):continue
  sc=pd.read_csv(Path(r.source_folder)/'scalars.csv').set_index('variable')
  for var,(key,_,__) in FIELDS.items():
   if key not in sc.index:continue
   row=sc.loc[key];areas.append(dict(request_id=rid,sid=r.sid,event_id=r.event_id,event_class=r.event_class,lag_h=r.lag_h,center_kind=r.center_kind,variable=var,valid_area_m2=row.valid_area_m2,original_area_m2=row.original_area_m2,valid_fraction=row.valid_fraction,domain=row.domain,support_pass=row.support_pass))
 pd.DataFrame(audit).to_csv(O/'centre_native_motion_support_audit.csv',index=False)
 pd.DataFrame(areas).to_csv(O/'centre_native_field_areas.csv',index=False)
 x.to_csv(O/'centre_native_process_sources.csv',index=False)
 rows=[];support=[]
 for (sid,var),g in x.groupby(['sid','variable'],sort=True):
  cols=pd.MultiIndex.from_product([['internal','final'],['actual_track','fixed_t0']]);z=g.pivot(index='lag_h',columns=['event_class','center_kind'],values='value').reindex(index=range(-24,25,6),columns=cols);a=z.to_numpy(copy=True).reshape(9,2,2);common=np.isfinite(a).all((1,2));a[~common]=np.nan
  legacy=g.pivot(index='lag_h',columns=['event_class','center_kind'],values='legacy_value').reindex(index=range(-24,25,6),columns=cols).to_numpy(copy=True).reshape(9,2,2)
  for policy in ['all_along','all_fixed','historical_mixed']:
   y=a[:,:,0] if policy=='all_along' else a[:,:,1] if policy=='all_fixed' else np.concatenate([legacy[:4,:,1],a[4:,:,0]])
   y[~common]=np.nan
   vals={'pre_mean':np.nanmean(y[:4],0) if common[:4].sum()>=3 else np.array([np.nan,np.nan]),'change_0_24':y[-1]-y[4] if common[[4,8]].all() else np.array([np.nan,np.nan]),'integral_0_24':np.trapezoid(y[4:],dx=6,axis=0) if common[4:].all() else np.array([np.nan,np.nan])}
   if var in ['midRH','shear200_850','shear500_850','shear200_500','PI_headroom','Tang_Emanuel_ventilation']:
    direction=-1 if var in ['midRH','PI_headroom'] else 1;bad=direction*(y[4:]-vals['pre_mean'])>0;vals['adverse_hours_0_24']=np.sum(bad[:-1]&bad[1:],axis=0)*6. if common[4:].all() and np.isfinite(vals['pre_mean']).all() else np.array([np.nan,np.nan])
   for name,yy in vals.items():rows.append(dict(sid=sid,variable=var,policy=policy,estimand=name,units='h' if name.startswith('adverse') else FIELDS[var][2]+' h' if name.startswith('integral') else FIELDS[var][2],internal=yy[0],final=yy[1],difference=yy[0]-yy[1]))
  support.append(dict(sid=sid,variable=var,n_joint_lags=int(common.sum()),n_pre_joint=int(common[:4].sum()),complete_integral=bool(common[4:].all())))
 pd.DataFrame(rows).to_csv(O/'centre_native_SID_sources.csv',index=False);pd.DataFrame(support).to_csv(O/'centre_native_support.csv',index=False);tab,piv,d,w=summary_table(rows,['variable','policy','estimand','units'],value='difference');tab.to_csv(O/'centre_native_estimates.csv',index=False);np.savez_compressed(O/'centre_native_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)

 before=pd.read_csv(O/'centre_SID_sources.csv');after=pd.DataFrame(rows);keys=['sid','variable','policy','estimand'];before=before[before.policy.eq('all_along')&before.estimand.eq('pre_mean')];after=after[after.policy.eq('all_along')&after.estimand.eq('pre_mean')];check=before.merge(after,on=keys,suffixes=('_old','_new'))
 check['absolute_difference']=(check.difference_old-check.difference_new).abs();check['same_finite']=np.isfinite(check.difference_old)==np.isfinite(check.difference_new);check.to_csv(O/'centre_native_pre_source_reproduction.csv',index=False)
 assert check.same_finite.all() and check.absolute_difference.max()<1e-10
 aa=pd.DataFrame(audit)
 js(O/'centre_native_receipt.json',{'status':'COMPLETE','n_fixed_advection_requests':len(aa),'max_motion_support_correction_gkg_day':float(aa.difference.abs().max()),'raw_reads':0,'pre_allalong_identical':True,'pre_allalong_max_difference':float(check.absolute_difference.max()),'motion_policy':'actual lag cyclone velocity; gradient and advection are averaged on the identical finite native-cell support','effective_area_table':'centre_native_field_areas.csv; corrected horizontal advection area in centre_native_motion_support_audit.csv'})
 print('centre native complete',len(tab),flush=True)

if __name__=='__main__':main()
