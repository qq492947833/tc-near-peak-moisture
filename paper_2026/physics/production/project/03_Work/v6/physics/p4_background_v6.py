from process_v6 import *
source=V/'upper_profile_100_50_interpolation_v1/p4field_src/P4_shared_climate_SID_source.csv'
x=pd.read_csv(source,low_memory=False);rows=[];errs=[]
for keys,g in x.groupby(['sid','boundary','field','raw_unit'],sort=True):
 z=g.set_index('lag_h')
 for lo,hi in [(-24,0),(0,24)]:
  if lo not in z.index or hi not in z.index:continue
  y=z.loc[[lo,hi],['raw_common','climate_common','anomaly_common']].to_numpy(float)
  valid=np.isfinite(y).all() and z.loc[[lo,hi],'raw_common__valid'].all() and z.loc[[lo,hi],'climate_common__valid'].all() and z.loc[[lo,hi],'anomaly_common__valid'].all()
  d=y[1]-y[0] if valid else [np.nan]*3
  if valid:errs.append(abs(d[0]-d[1]-d[2]))
  for field,value in zip(['raw_change','climate_change','anomaly_change'],d):rows.append(dict(zip(['sid','boundary','field','units'],keys),window=f'{lo}_{hi}',component=field,value=value))
pd.DataFrame(rows).to_csv(O/'P4_background_SID_sources.csv',index=False);t,piv,d,w=summary_table(rows,['boundary','field','units','window','component']);t.to_csv(O/'P4_background_estimates.csv',index=False);np.savez_compressed(O/'P4_background_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w);js(O/'P4_background_receipt.json',{'source':str(source),'max_reconstruction_error':max(errs),'support':'same SID and endpoints, native raw/climatology/anomaly common mask and original90percent area','new_raw_reads':0,'n_targets':len(t)});print('P4 background complete',len(t))
