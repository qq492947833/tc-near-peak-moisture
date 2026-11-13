from process_v6 import *
FIELDS={
 'shear200_850':('shear_200_850_vector_magnitude_ms',1.,'m s-1'),
 'shear500_850':('shear_500_850_vector_magnitude_ms',1.,'m s-1'),
 'shear200_500':('shear_200_500_vector_magnitude_ms',1.,'m s-1'),
 'midRH':('mid_rh_pct',1.,'pp'),
 'q700_storm_relative_horizontal_advection':('q700_horizontal_storm_relative_kgkg_s',86400000.,'g kg-1 day-1'),
 'q700_vertical_advection':('q700_vertical_advection_kgkg_s',86400000.,'g kg-1 day-1'),
 'SST':('sst_C',1.,'degC'),
 'PI_headroom':('thermal_headroom_ms',1.,'m s-1'),
 'Tang_Emanuel_ventilation':('raw_signed_index',1.,'dimensionless')}
def main():
 while not ((O/'fixed_full_receipt.json').exists() and (O/'fixed_PI_receipt.json').exists()):time.sleep(10)
 old=pd.read_csv(V/'upper_profile_100_50_interpolation_v1/p13_stats_v1/p1_C/process_sources.csv',low_memory=False);old=old[old.current117&old.lag_h.between(-24,24)&old.variable.isin(FIELDS)].copy();old['legacy_value']=old.value
 req=pd.read_csv(V/'data/p1_environment_requests_v5.csv');along=req[req.center_kind.eq('actual_track')].set_index(['sid','event_id','lag_h']);pi=pd.read_csv(O/'fixed_PI_values.csv').set_index('request_id');records=json.loads((O/'fixed_full_receipt.json').read_text())['records'];new=[];zero_checks=[]
 for rec in records:
  f=Path(rec['folder']);c=json.loads((f/'COMPLETE.json').read_text());r=c['binding']['request'];sc=pd.read_csv(f/'scalars.csv').set_index('variable');th=pi.loc[r['request_id']];ar=along.loc[(r['sid'],r['event_id'],int(r['lag_h']))]
  def value(k):
   v=sc.loc[k];return float(v.value) if str(v.support_pass).lower()=='true' and v.valid_fraction>=.9 else np.nan
  for var,(key,factor,unit) in FIELDS.items():
   val=float(th[key]) if var in ('PI_headroom','Tang_Emanuel_ventilation') else value(key)*factor
   # Constant t0 motion in old fixed-centre requests is standardized to actual lag motion.
   if var=='q700_storm_relative_horizontal_advection':val+=((float(ar.motion_u_ms)-float(r['motion_u_ms']))*value('q700_dq_dx')+(float(ar.motion_v_ms)-float(r['motion_v_ms']))*value('q700_dq_dy'))*86400000.
   new.append(dict(request_id=r['request_id'].removeprefix('v6|'),sid=r['sid'],event_id=r['event_id'],event_class=r['event_class'],lag_h=int(r['lag_h']),center_kind='fixed_t0',UTC=r['UTC'],current117=True,source_folder=str(f),variable=var,value=val,legacy_value=np.nan,units=unit,source_motion='contemporaneous_actual_track_for_centre_isolation'))
   if int(r['lag_h'])==0:
    a=old[(old.event_id==r['event_id'])&(old.lag_h==0)&(old.center_kind=='actual_track')&(old.variable==var)].value.iloc[0];zero_checks.append(dict(request_id=r['request_id'],variable=var,new=val,old=a,absolute_difference=abs(val-a),same_finite=bool(np.isfinite(val)==np.isfinite(a))))
 # Remove old SST-only positive fixed placeholders, retain original legacy pre values.
 x=old[~(old.center_kind.eq('fixed_t0')&old.lag_h.ge(0))].copy()
 for idx,r in x[(x.center_kind=='fixed_t0')&x.variable.eq('q700_storm_relative_horizontal_advection')].iterrows():
  sc=pd.read_csv(Path(r.source_folder)/'scalars.csv').set_index('variable');ar=along.loc[(r.sid,r.event_id,int(r.lag_h))];fr=req[req.request_id.eq(r.request_id)].iloc[0]
  gx=float(sc.loc['q700_dq_dx','value']);gy=float(sc.loc['q700_dq_dy','value']);x.loc[idx,'value']=r.value+((ar.motion_u_ms-fr.motion_u_ms)*gx+(ar.motion_v_ms-fr.motion_v_ms)*gy)*86400000.
 x=pd.concat([x,pd.DataFrame(new)],ignore_index=True);x.to_csv(O/'centre_complete_process_sources.csv',index=False);pd.DataFrame(zero_checks).to_csv(O/'centre_zero_lag_reproduction.csv',index=False)
 checks=pd.DataFrame(zero_checks);assert checks.same_finite.all();assert checks.absolute_difference.max()<1e-8
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
 pd.DataFrame(rows).to_csv(O/'centre_complete_SID_sources.csv',index=False);pd.DataFrame(support).to_csv(O/'centre_complete_support.csv',index=False);tab,piv,d,w=summary_table(rows,['variable','policy','estimand','units'],value='difference');tab.to_csv(O/'centre_complete_estimates.csv',index=False);np.savez_compressed(O/'centre_complete_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)
 js(O/'centre_complete_receipt.json',{'status':'COMPLETE','new_fixed_requests':len(records),'zero_lag_identical_support':bool(checks.same_finite.all()),'zero_lag_max_difference':float(checks.absolute_difference.max()),'raw_reads_in_combiner':0,'motion_policy':'all_along and all_fixed use actual lag cyclone translation for storm-relative diagnostics; historical_mixed preserves original fixed t0 translation on negative lags','same_support':'every reported policy uses intersection of both events and both centres per SID/lag; annuli remain in own geographic position'})
 print('centre complete FULL',len(tab),flush=True)
if __name__=='__main__':main()
