from process_v6 import *
b=V/'upper_profile_100_50_interpolation_v1';c=json.loads((b/'p13_stats_v1/p1_C/completion.json').read_text())
x=pd.read_csv(c['source']['processed_index']);x=x[x.configuration.eq('C_14_logp_inclusive50') & x.center_kind.eq('actual_track') & x.lag_h.isin([-24,-18,-12,-6])]
s=pd.read_csv(O/'centre_SID_sources.csv');s=s[(s.policy=='all_along')&(s.estimand=='pre_mean')&(s.variable=='PI_headroom')&s.difference.notna()]
orig=pd.read_csv(b/'p13_stats_v1/p1_C/process_sources.csv');orig=orig[orig.current117&orig.lag_h.isin([-24,-18,-12,-6])&orig.variable.eq('PI_headroom')];rows=[]
for sid in s.sid:
 g=orig[orig.sid==sid].pivot(index='lag_h',columns=['event_class','center_kind'],values='value');lags=g.index[np.isfinite(g.to_numpy()).all(1)];xx=x[(x.sid==sid)&x.lag_h.isin(lags)]
 for side in ['internal','final']:
  z=xx[xx.event_class==side];assert len(z)>=3
  for field,unit in [('pi_vmax_ms','m s-1'),('wind_for_headroom_kt','kt'),('thermal_headroom_ms','m s-1'),('raw_signed_index','1'),('entropy_deficit_jkgk','J kg-1 K-1'),('sea_air_entropy_difference_jkgk','J kg-1 K-1')]:
   rows.append(dict(sid=sid,event_class=side,component=field,value=z[field].mean(),unit=unit,n_common_pre=len(z),source_request_ids=';'.join(z.request_id)))
pd.DataFrame(rows).to_csv(O/'along_pre_components.csv',index=False)
print('written',len(rows),'rows',len(s),'SIDs')
