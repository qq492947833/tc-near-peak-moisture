from process_v6 import *

def main():
    x=pd.read_csv(O/'regions_SID_pairs.csv')
    rows=[]
    for (sid,field,sector,estimand,units),g in x.groupby(['sid','field','sector','estimand','units']):
        z=g.set_index('region')
        if not {'100_300km','300_800km'}.issubset(z.index):continue
        a=z.loc['100_300km'];b=z.loc['300_800km']
        rows.append(dict(sid=sid,field=field,sector=sector,estimand=estimand,units=units,value=b.value-a.value,
                         internal=b.internal-a.internal,final=b.final-a.final))
    pd.DataFrame(rows).to_csv(O/'regions_outer_minus_inner_SID.csv',index=False)
    tab,piv,d,w=summary_table(rows,['field','sector','estimand','units'])
    tab.to_csv(O/'regions_outer_minus_inner_estimates.csv',index=False)
    np.savez_compressed(O/'regions_outer_minus_inner_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)
    # Native area components and budgets must retain identical support before interpretation.
    raw=pd.read_csv(O/'regions_request_sources.csv')
    ids=['sid','event_id','event_class','lag_h','region','sector']
    q=raw[raw.field.isin(['q700_raw_common','q700_climatology','q700_anomaly'])].pivot(index=ids,columns='field',values='value')
    b=raw[raw.field.isin(['moving_tendency','horizontal_advection','vertical_advection','residual'])].pivot(index=ids,columns='field',values='value')
    qerr=(q.q700_raw_common-q.q700_climatology-q.q700_anomaly).abs().max()
    berr=(b.moving_tendency-b.horizontal_advection-b.vertical_advection-b.residual).abs().max()
    assert qerr<1e-10 and berr<1e-10
    rh=pd.read_csv(O/'RH_native_SID_sources.csv')
    coverage=[]
    for k,g in rh.groupby(['boundary','window','level_hpa','region']):
        nfull=int(np.isfinite(g.native_RH_change).sum());nlocal=int(np.isfinite(g.local_RH_change).sum())
        coverage.append(dict(zip(['boundary','window','level_hpa','region'],k))|dict(n_full_along=nfull,n_local_decomposition=nlocal,n_excluded_local=nfull-nlocal))
    pd.DataFrame(coverage).to_csv(O/'RH_support_counts.csv',index=False)
    js(O/'region_finalize_receipt.json',{'status':'COMPLETE','q_climatology_identity_max':float(qerr),'700_budget_identity_max':float(berr),'raw_reads':0,'contrast':'(internal minus final at outer ring) minus (internal minus final at inner ring), paired within SID'})
    print(tab[(tab.sector=='whole')&tab.field.isin(['q700','q700_anomaly','RH700','vertical_advection','moving_tendency'])].to_csv(index=False),flush=True)

if __name__=='__main__':main()
