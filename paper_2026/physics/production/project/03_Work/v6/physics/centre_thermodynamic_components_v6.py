from process_v6 import *
b=V/'upper_profile_100_50_interpolation_v1';c=json.loads((b/'p13_stats_v1/p1_C/completion.json').read_text())
x=pd.read_csv(c['source']['processed_index']);x=x[x.configuration.eq('C_14_logp_inclusive50')&x.module.eq('P1')&x.lag_h.between(-24,24)].copy()
s=pd.read_csv(O/'centre_native_process_sources.csv');sids=set(s.sid);x=x[x.sid.isin(sids)&~(x.center_kind.eq('fixed_t0')&x.lag_h.ge(0))]
y=pd.read_csv(O/'fixed_PI_values.csv');y.request_id=y.request_id.str.removeprefix('v6|');y['configuration']='C_14_logp_inclusive50';z=pd.concat([x,y],ignore_index=True)
meta=s[['request_id','event_id','sid','event_class','lag_h','center_kind']].drop_duplicates()
z=z.drop(columns=['event_id','sid','event_class','lag_h','center_kind'],errors='ignore').merge(meta,on='request_id',how='inner',validate='one_to_one')
z.to_csv(O/'centre_thermodynamic_components.csv',index=False)
assert not z.request_id.duplicated().any()
js(O/'centre_thermodynamic_components_receipt.json',{'status':'COMPLETE','rows':len(z),'configuration':'C_14_logp_inclusive50','source':'accepted current production C negative fixed and all-along, new complete C positive fixed; unchanged physical/profile gates','signed':'raw_signed_index, never clip negative; chi600 is signed entropy deficit ratio','disk_annulus':'600 hPa 0–100 km saturated core and 100–300 km environmental entropy use same requested physical centre'})
print('components complete',len(z))
