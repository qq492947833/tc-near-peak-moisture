"""Replay the frozen additive duration statistics without generating new samples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from duration_linear_targets import bh, contrasts, interval, solve_targets

HERE = Path(__file__).resolve().parent
ENV = ['X1_sst_c', 'X2_shear_ms', 'X3_mid_rh_pct']
FAMILY = {'full': 'C_primary_standardized_trends_16', 'recent': 'C_recent_standardization_16'}
ATOL = 1e-9  # comparison of re-evaluated stored contrasts only; not an estimability threshold

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def equal(actual, expected, what):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL, equal_nan=True, err_msg=what)

def recover_failed(design, bank, draws, failed):
    """Re-evaluate the prespecified linear functionals on the original draw IDs."""
    X, Y = design['X'], design['Y']
    names = design['names'].tolist()
    groups = {k: design[k] for k in ['geo', 'state', 'env']}
    scales = design['scales']
    cols = [names.index(c) for c in ENV]
    mapping = {s:i for i,s in enumerate(bank['sid'])}
    if len(mapping)!=len(bank['sid']) or len(set(design['sid']))!=len(X):
        raise ValueError('non-unique SID in frozen bank or design')
    ix = np.array([mapping[s] for s in design['sid']])
    years = design['year'] - design['year'].min()
    for b in failed:
        n = bank['year_multiplicity'][b]
        counts = bank['counts'][b, ix].astype(float)
        ss = np.bincount(years, weights=counts, minlength=len(n))
        ww = counts*np.divide(n[years], ss[years], out=np.zeros(len(X)), where=ss[years]>0)
        c, fs, _, _, _, _ = contrasts(X,ww,years,groups,len(n),design['reference'])
        ec = np.zeros((3,X.shape[1]))
        ep = np.zeros_like(ec)
        for j,k in enumerate(cols):
            ec[j,k]=1/scales[k]
            ep[j,k]=fs[k]
        solution = solve_targets(X,Y,ww,np.vstack([c,ec,ep]))
        values, ok = solution['values'], solution['ok']
        np.testing.assert_array_equal(ok[:8], draws['estimable'][b])
        equal(values[:8],draws['trends'][b],'recovered duration and coverage contrasts')
        aux = np.r_[values[8:11,0], fs[cols]*scales[cols], values[11:14,0]]
        np.testing.assert_array_equal(np.r_[ok[8:11],[True]*3,ok[11:14]],draws['auxiliary_estimable'][b])
        equal(aux,draws['auxiliary'][b],'recovered auxiliary targets')

def replay(inputs, output, check_recovery=False):
    """Summarize saved target values; optionally replay previously rank-deficient draws."""
    files=json.loads((inputs/'source_manifest.json').read_text(encoding='utf8'))['files']
    if inputs.resolve()==output.resolve():
        raise ValueError('choose a separate output directory')
    for name,record in files.items():
        if hashlib.sha256((inputs/name).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('input checksum mismatch: '+name)
    rows=[]; checks=[]
    for period in ('full','recent'):
        bank=load_npz(inputs/f'{period}_joint_year_sid_draws.npz')
        for stage in ('D','E','F','Dec'):
            prefix=f'{period}_{stage}'
            des=load_npz(inputs/f'{prefix}_design.npz')
            z=load_npz(inputs/f'{prefix}_target_draws.npz')
            old=load_npz(inputs/f'{prefix}_original_values.npz')
            success=np.isfinite(old['trends']).all(axis=(1,2))
            if len(success)!=len(bank['counts']) or len(success)!=2000: raise ValueError('draw count')
            np.testing.assert_array_equal(success,z['original_success'])
            np.testing.assert_array_equal(old['trends'][success],z['trends'][success,:4,:])
            if period=='full':
                np.testing.assert_array_equal(old['auxiliary'][success],z['auxiliary'][success])
            failed=np.flatnonzero(~success)
            if check_recovery:
                recover_failed(des, bank, z, failed)
            checks.append(dict(period=period,stage=stage,original_success=int(success.sum()),
                               rank_guard_failures=len(failed),replayed_failures=len(failed) if check_recovery else 0))
            def add(target, point, values, estimable, family, inference=True):
                np.testing.assert_array_equal(np.isfinite(values),estimable)
                summary=interval(values)
                rows.append(dict(period=period,stage=stage,target=target,n_sid=len(des['sid']),
                                  estimate=point,ci_low=summary['ci_low'],ci_high=summary['ci_high'],
                                  p_boot=summary['p_boot'] if inference else np.nan,q_BH=np.nan,family=family,
                                  valid_draws=summary['valid_draws'],attempted_draws=len(values),
                                  original_successes=int(success.sum()),recovered_targets=int(estimable[~success].sum()),
                                  still_nonestimable=int((~estimable).sum())))
            ds=z['trends'][:,:,0]; ok=z['estimable']; target=z['targets']
            for i in range(4):
                add(target[i],z['point'][i,0],ds[:,i],ok[:,i],FAMILY[period])
            if period=='recent': continue
            equal(ds[:,7],ds[:,5]+ds[:,6],'same-draw paired sums')
            for i in range(4,8):
                add(target[i],z['point'][i,0],ds[:,i],ok[:,i],
                    'C_standardization_contrasts_12' if i<7 else 'NA',i<7)
            for j,(kind,fam) in enumerate([('duration_association_coefficient','C_entry_environment_duration_associations_12'),
                       ('environment_history_trend','C_entry_environment_trends_12'),('environment_product','descriptive_no_new_family')]):
                for ix,k in enumerate(ENV):
                    i=3*j+ix
                    add(kind+':'+k,des['point_auxiliary'][i],z['auxiliary'][:,i],z['auxiliary_estimable'][:,i],fam)
    tab=pd.DataFrame(rows)
    for fam,n in {**dict.fromkeys(FAMILY.values(),16),'C_standardization_contrasts_12':12,
                  'C_entry_environment_trends_12':12,'C_entry_environment_duration_associations_12':12}.items():
        mask=tab.family.eq(fam)
        if mask.sum()!=n or tab.loc[mask,'p_boot'].isna().any():raise ValueError('incomplete family: '+fam)
        tab.loc[mask,'q_BH']=bh(tab.loc[mask,'p_boot'])
    by=['period','stage','target']; obs=tab.set_index(by).sort_index()
    expected=pd.read_csv(inputs/'frozen_statistics.csv',float_precision='round_trip',keep_default_na=False).set_index(by).sort_index()
    pd.testing.assert_index_equal(obs.index,expected.index)
    pd.testing.assert_series_equal(obs.family,expected.family)
    for col in obs.drop(columns='family'):
        v=pd.to_numeric(expected[col].replace('NA',np.nan),errors='raise')
        equal(obs[col],v,col)
    output.mkdir(parents=True,exist_ok=True)
    tab.to_csv(output/'replayed_statistics.csv',index=False,float_format='%.17g')
    (output/'replay_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    return tab

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=HERE/'duration_targets')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--check-recovery',action='store_true',help='evaluate the frozen contrasts on each original failed design')
    args=p.parse_args()
    result=replay(args.inputs,args.output,args.check_recovery)
    print(f'{len(result)} statistics matched the frozen targets; output: {args.output}')
if __name__=='__main__': main()
