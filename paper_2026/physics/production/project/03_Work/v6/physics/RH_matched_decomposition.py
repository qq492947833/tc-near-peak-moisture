from process_v6 import *
x=pd.read_csv(O/'RH_native_SID_sources.csv');fields=['q_contribution','T_contribution','p_contribution','local_RH_change','sampling_support_remainder','native_RH_change','archived_RH_change','rh_archive_difference_change'];good=np.isfinite(x[fields]).all(axis=1);x=x[good].copy();x.to_csv(O/'RH_matched_decomposition_SID.csv',index=False)
assert np.max(abs(x.native_RH_change-x.q_contribution-x.T_contribution-x.p_contribution-x.sampling_support_remainder))<1e-10
r=x.melt(id_vars=['sid','boundary','window','level_hpa','region'],value_vars=fields,var_name='component',value_name='value');t,piv,d,w=summary_table(r.to_dict('records'),['boundary','window','level_hpa','region','component']);t['units']='pp';t.to_csv(O/'RH_matched_decomposition_estimates.csv',index=False);np.savez_compressed(O/'RH_matched_decomposition_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)
check=[]
for keys,g in t.groupby(['boundary','window','level_hpa','region']):
 z=g.set_index('component');assert z.n_sid.nunique()==1;err=z.loc['native_RH_change','estimate']-z.loc[['q_contribution','T_contribution','p_contribution','sampling_support_remainder'],'estimate'].sum();assert abs(err)<1e-10;check.append(dict(target=str(keys),n=int(z.n_sid.iloc[0]),mean_identity_error=err))
js(O/'RH_matched_decomposition_receipt.json',{'status':'COMPLETE','support':'each plotted component shares all eight finite components for same SID level region window','rows':len(x),'identity_checks':check});print('RH matched complete',len(t))
