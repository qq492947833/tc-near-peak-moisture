import numpy as np
import pandas as pd
CORE=["X1_sst_c","X2_shear_ms","X3_mid_rh_pct","X4_d200_1e5s","X5_minus_omega500_pa_s","X6_signed_vo850_1e5s"]

def aggregate(d,keys,mode='descriptive',controls=None):
    controls=controls or []; outs=[]
    total=d.groupby(keys,dropna=False).interval_hours_v3.sum().rename('legal_h')
    for x in CORE:
        good=d[x+'_value'].notna()&d.interval_hours_v3.gt(0)
        if controls: good &= d[controls].notna().all(axis=1)
        q=d.loc[good].copy()
        if q.empty:continue
        # Frozen legal intervals are exactly six hours; thus equal interval means
        # are identical to the explicit duration-weighted ratio below.
        q['_weighted_value']=q[x+'_value']*q.interval_hours_v3
        gb=q.groupby(keys,dropna=False,sort=False)
        a=gb.agg(weighted_sum=('_weighted_value','sum'),median=(x+'_value','median'),covered_h=('interval_hours_v3','sum'),n_valid_intervals=('row_id','size'),source_row_ids=('row_id',lambda z:'|'.join(z)),basin_genesis=('basin_genesis','first'))
        a['estimate']=a.weighted_sum/a.covered_h;a=a.drop(columns='weighted_sum');a=a.join(total)
        for c in controls:
            q['_w_'+c]=q[c]*q.interval_hours_v3
            a[c]=q.groupby(keys,dropna=False,sort=False)['_w_'+c].sum()/a.covered_h
        a['variable']=x;a['coverage_fraction']=a.covered_h/a.legal_h;a['mode']=mode
        outs.append(a.reset_index())
    return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame()
