"""Verify imports, frozen identities and saved diagnostic values without refitting."""
from pathlib import Path
import argparse
import ast
import importlib
import json
import sys
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]


def verify(output):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    checks=[]
    def check(name,ok,**detail):
        checks.append({'check':name,'passed':bool(ok),**detail})
    for p in sorted((PACKAGE/'scripts').glob('*.py')):
        ast.parse(p.read_text(encoding='utf-8'));check('syntax_'+p.name,True)
    names=['read_era5_fixed_block','theory_MWT_Atmosphere_Direct','vertical_terms','_adjusted_legacy_helpers','raw_statistics','derive_sensitivity','adjusted_statistics','curate_diagnostics','extract_vertical','assemble_adjusted_requests','bind_cache','preflight_inputs']
    for name in names:importlib.import_module(name);check('import_'+name,True)
    from raw_statistics import load_frozen_templates, rise_values, utc_seconds
    from adjusted_statistics import load_frozen_background
    from test_vertical_terms import run_checks
    numeric=run_checks(output.parent/'NUMERICAL_CHECKS.json')
    check('fourteen_analytic_numerical_checks',numeric['status']=='PASS' and len(numeric['checks'])==14)
    templates=load_frozen_templates()
    check('frozen_event_identities',len(templates['events'])==133 and templates['events'].sid.nunique()==123 and templates['pre'].sid.nunique()==117)
    check('matched_identities',len(templates['pairs'])==165 and templates['pairs'].event_id.nunique()==115 and templates['pairs'].sid.nunique()==108)
    pairs,A,C,names=load_frozen_background()
    check('background_input_identities',len(pairs)==165 and pairs.sid.nunique()==108 and A.shape==C.shape and A.shape[1]==12)
    data=pd.read_parquet(PACKAGE/'inputs/example/accepted_request_values.parquet')
    saved=pd.read_parquet(PACKAGE/'inputs/example/accepted_event_integrals.parquet')
    data['UTC_s']=utc_seconds(data.UTC)
    errors=[]
    for metric,rows in data.groupby('metric'):
        target=saved[saved.metric.eq(metric)]
        if target.empty:continue
        values=rows.set_index(['sid','UTC_s']).value
        derived=rise_values(values,templates,integrate_rate=True)['rise_rate_integral']
        same=derived.merge(target[['sid','event_id','value']],on=['sid','event_id'],suffixes=('_readback','_saved'),validate='one_to_one')
        delta=np.abs(same.value_readback-same.value_saved)
        check('saved_event_integral_'+metric,np.allclose(same.value_readback,same.value_saved,rtol=2e-13,atol=2e-11,equal_nan=True),max_abs_error_Jkg=float(delta.max()))
        errors.append(float(delta.max()))
    summary=pd.read_csv(PACKAGE/'inputs/example/common_support_summary.csv',float_precision='round_trip')
    with np.load(PACKAGE/'inputs/example/common_support_saved_draws.npz') as z:
        for row in summary.itertuples(index=False):
            d=z[row.metric];limits=np.quantile(d,[.025,.975]);p=(1+np.count_nonzero(np.abs(d-row.estimate)>=abs(row.estimate)))/(len(d)+1)
            check('saved_population_interval_P_'+row.metric,len(d)==2000 and np.isfinite(d).all() and np.allclose(limits,[row.ci_lower,row.ci_upper],rtol=2e-14,atol=2e-13) and abs(p-row.p)<1e-15)
        for e in ['K','A']:check('saved_draw_additivity_'+e,np.allclose(z['Gamma_'+e+'_total'],z['Gamma_'+e+'_h']+z['Gamma_'+e+'_p'],rtol=1e-13,atol=1e-12))
    result={'status':'PASS' if all(v['passed'] for v in checks) else 'FAIL','checks':checks,'scope':'Syntax/imports, fourteen predefined analytic tests, frozen support identities, one saved SID integral readback and saved population interval/P/additivity checks. No ERA5 calculation, model fitting or new bootstrap draws.','maximum_event_integral_error_Jkg':max(errors),'python':sys.version.split()[0]}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    result=verify(args.output);print(json.dumps({'status':result['status'],'checks':len(result['checks'])}))
    if result['status']!='PASS':raise SystemExit(1)
