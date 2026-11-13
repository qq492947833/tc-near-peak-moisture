from process_v6 import *
x=pd.read_csv(O/'P4_transport_SID_sources.csv');x=x[x.estimand.str.startswith('integral')];rows=[]
for keys,g in x.groupby(['sid','boundary','field','units']):
 z=g.set_index('estimand').value
 if {'integral_minus24_0','integral_0_24'}<=set(z.index):
  before=z['integral_minus24_0'];after=z['integral_0_24']
  if not np.isfinite(before+after):before=after=np.nan
  for target,val in [('before',before),('after',after),('after_minus_before',after-before)]:rows.append(dict(zip(['sid','boundary','field','units'],keys),target=target,value=val))
pd.DataFrame(rows).to_csv(O/'P4_transport_matched_windows_SID.csv',index=False);tab,piv,d,w=summary_table(rows,['boundary','field','units','target']);tab.to_csv(O/'P4_transport_matched_windows_estimates.csv',index=False);np.savez_compressed(O/'P4_transport_matched_windows_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w);print('matched complete',len(tab))
