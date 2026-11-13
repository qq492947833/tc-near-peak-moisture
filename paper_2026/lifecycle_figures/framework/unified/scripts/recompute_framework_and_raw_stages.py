"""Reuse accepted framework estimators and raw-stage summaries on frozen SID.

The original functions are read and executed unchanged in an isolated namespace.
Outputs stay under this unified work directory; source files stay read-only.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'
from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import hashlib
import time
import numpy as np
import pandas as pd
from scipy import stats
import psutil

HERE = Path(__file__).absolute().parents[1]
ADD = HERE.parent
V62 = HERE.parent / 'dependencies/v62'
FRAME = HERE / 'statistics/framework'
STAGE = HERE / 'statistics/stages'
VARS = ['X1_sst_c','X2_shear_ms','X3_mid_rh_pct','X4_d200_1e5s','X5_minus_omega500_pa_s','X6_signed_vo850_1e5s']
UNITS = ['degC','m s-1','percentage points','1e-5 s-1','Pa s-1','1e-5 s-1']
REP = {'A4-F':['D','E','F','Res'],'A3-F':['DE','F','Res'],'A4-R':['D','E','R','post_R_tail'],'A3-R':['DE','R','post_R_tail']}
METHODS = ['coarse','actual_DE','time_midpoint','continuous_progress','continuous_age']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def extract(path, names, scope):
    parsed = ast.parse(path.read_text('utf-8-sig'))
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def save(frame, folder, name):
    frame.to_parquet(folder / (name + '.parquet'), index=False)
    frame.to_csv(folder / (name + '.csv'), index=False)


def main():
    started = time.monotonic()
    if os.name == "nt": psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    receipt = HERE / 'reports/framework_raw_stages_receipt.json'
    assert not receipt.exists(), 'Completed isolated output exists'
    for folder in [FRAME, STAGE]:
        folder.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((HERE / 'reports/main_sid_freeze_receipt.json').read_text('utf-8'))
    assert sha(HERE/'source_data/main_sid.txt') == freeze['main_sid_sha256']
    master = pd.read_parquet(HERE / 'source_data/main_sid.parquet')
    ids = set(master.sid)
    original_rows = pd.read_parquet(ADD / 'source_data/de_existing_sources/analysis_rows.parquet')
    rows = original_rows[original_rows.sid.isin(ids)].reset_index(drop=True)
    assert set(rows.sid) == ids
    source_ledger = pd.read_csv(V62 / 'framework/A2_full_evaluable_range.csv', keep_default_na=False)
    ledger = source_ledger[source_ledger.sid.isin(ids)].set_index('sid').sort_index()
    assert set(ledger.index) == ids and ledger.main_complete.all()
    for label in ['genesis_time','escalation_time','maturity_time','resolution_time','termination_time','lmi_time','lmi_last_time']:
        expected = pd.to_datetime(master.set_index('sid')[label], utc=True)
        assert np.array_equal(rows[label], rows.sid.map(expected)), label
    rows.to_parquet(FRAME / 'analysis_rows.parquet', index=False)
    ledger.reset_index().to_csv(FRAME / 'A2_full_evaluable_range.csv', index=False)
    checks = []
    def check(name, value, detail=None):
        checks.append(dict(test=name, passed=bool(value), detail=detail))
        assert value, (name, detail)
    def dump(name, value):
        (FRAME / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    fw_path = V62 / 'framework/run_framework_v62.py'
    env = extract(fw_path, ['bools','fold','ridge','bootstrap_indices','a2','design_matrix','a3'],
                  dict(np=np,pd=pd,hashlib=hashlib,BASE=FRAME,VARS=VARS,UNITS=UNITS,REP=REP,METHODS=METHODS,check=check,dump=dump))
    s = master.set_index('sid').copy()
    s['A_signed'] = s.lifecycle_asymmetry_time
    print('FRAMEWORK_A2_START', flush=True)
    tab, contrasts = env['a2'](rows, ledger, s)
    chron = pd.read_csv(FRAME / 'A2_chronology_allocation.csv')
    chron[chron.representation.eq('A4-F')].groupby(['stage','chronology'], as_index=False).agg(
        mean_h=('chronology_overlap_h','mean'), n_sid=('sid','nunique')).to_csv(FRAME / 'stage_chronology_summary.csv', index=False)
    print('FRAMEWORK_A3_50_FIXED_FITS_START', flush=True)
    a3, losses, counts = env['a3'](rows, sorted(ids))
    # Fixed YT definitions have no changed boundary/input within each retained
    # SID. Reuse the exact per-SID values and recompute aggregate comparisons.
    for name in ['A1_YT_adapted_events.csv','A1_YT_evaluability.csv','A1_YT_interval_assignments.parquet',
                 'A1_YT_perSID_time_allocation.csv','A1_YT_environment_perSID.parquet']:
        path = V62 / 'framework' / name
        d = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, keep_default_na=False, na_values=['','nan','NaN'])
        key = 'originalSID' if 'originalSID' in d else 'sid'
        d = d[d[key].isin(ids)].copy()
        if path.suffix == '.parquet':
            d.to_parquet(FRAME / name, index=False)
        else:
            d.to_csv(FRAME / name, index=False)
    yt = pd.read_csv(FRAME / 'A1_YT_perSID_time_allocation.csv')
    yt_events = pd.read_csv(FRAME / 'A1_YT_adapted_events.csv')
    yt_flags = pd.read_csv(FRAME / 'A1_YT_evaluability.csv')
    yt_summary = []
    for label, g in yt.groupby('adaptation'):
        f = yt_flags[yt_flags.adaptation.eq(label)]
        yt_summary.append(dict(adaptation=label, original_sid_n=len(g),
                               adapted_event_n=int(yt_events.adaptation.eq(label).sum()),
                               SID_with_restart=int(f.event_n.gt(1).sum()),
                               whole_history_fully_observed_SID=int(f.fully_evaluable.sum()),
                               **{'mean_' + c: g[c].mean() for c in ['covered_h','developing_h','mature_h','decaying_h']}))
    pd.DataFrame(yt_summary).to_csv(FRAME / 'A1_YT_time_comparison.csv', index=False)
    yt_env = pd.read_parquet(FRAME / 'A1_YT_environment_perSID.parquet')
    yt_compare = []
    for (stage, variable), g in yt_env.groupby(['stage','variable']):
        p = g.pivot(index='sid', columns='adaptation', values='mean').dropna(subset=['YT80','YT_FLTthreshold'])
        yt_compare.append(dict(stage=stage,variable=variable,unit=g.unit.iloc[0],paired_sid_n=len(p),
                               YT80_mean=p.YT80.mean(),YT_FLTthreshold_mean=p.YT_FLTthreshold.mean(),
                               paired_difference_native_minus_FLT=(p.YT80-p.YT_FLTthreshold).mean(),
                               inference='Original descriptive paired SID mean; no new CI/test family'))
    pd.DataFrame(yt_compare).to_csv(FRAME / 'A1_YT_environment_paired_comparison.csv', index=False)
    print('RAW_STAGE_STATISTICS_START', flush=True)
    source_stage = ADD / 'source_data/de_existing_sources/raw_stage_sid.parquet'
    raw = pd.read_parquet(source_stage)
    raw = raw[raw.sid.isin(ids)].copy()
    save(raw, STAGE, 'raw_stage_sid')
    model_stage = pd.read_parquet(ADD / 'source_data/de_existing_sources/model_stage_sid.parquet')
    model_stage = model_stage[model_stage.sid.isin(ids)].copy()
    save(model_stage, STAGE, 'model_stage_sid')
    sid_order = sorted(ids)
    index = {sid:i for i,sid in enumerate(sid_order)}
    rng = np.random.Generator(np.random.PCG64(2026101102))
    weights = rng.multinomial(len(sid_order), np.repeat(1 / len(sid_order), len(sid_order)), size=2000).astype(np.uint16)
    np.save(STAGE / 'bootstrap_sid_multiplicities.npy', weights)
    pd.DataFrame(dict(sid=sid_order,index=np.arange(len(sid_order)))).to_csv(STAGE / 'bootstrap_sid_order.csv', index=False)
    spec = json.loads((V62 / 'analysis_spec_v62.json').read_text('utf-8-sig'))['module_B']
    refpath = V62 / spec['inputs']['reference_scale']['path']
    reference = pd.read_csv(refpath, keep_default_na=False).set_index('variable')
    original_b = V62 / 'stages/analysis_b.py'
    b = extract(original_b, ['make_pair','summarize_raw','bh_adjust'], dict(np=np,pd=pd,stats=stats))
    summaries, pairs, draws, adjacent = [], [], [], []
    for variable, unit in zip(VARS, UNITS):
        a = raw[raw.variable.eq(variable)].copy()
        for stage in ['D','E','F','Res']:
            aa = a[a.stage.eq(stage)]
            values = aa.estimate.to_numpy(float)
            rr = np.random.default_rng(20260920)
            estimates = np.array([rr.choice(values, len(values), replace=True).mean() for _ in range(2000)])
            low, high = np.percentile(estimates, [2.5,97.5])
            mu, sd = reference.loc[variable,['reference_mean','reference_sd']]
            summaries.append(dict(variable=variable,unit=unit,stage=stage,estimate=values.mean(),
                                   ci_low=low,ci_high=high,n_sid=len(values),covered_h=aa.covered_h.sum(),
                                   legal_h=aa.legal_h.sum(),mean_temporal_coverage=aa.coverage_fraction.mean(),
                                   min_temporal_coverage=aa.coverage_fraction.min(),
                                   estimate_standardized=(values.mean()-mu)/sd,
                                   ci_low_standardized=(low-mu)/sd,ci_high_standardized=(high-mu)/sd,
                                   ci_provenance='Original E1 SID percentile estimator and seed20260920; frozen main restriction',
                                   original_bootstrap_seed=20260920))
        for before, after in [('D','E'),('E','F'),('F','Res')]:
            pp = b['make_pair'](a,before,after,[])
            stat, draw = b['summarize_raw'](pp.delta.to_numpy(float),weights[:,[index[sid] for sid in pp.sid]])
            sd = reference.loc[variable,'reference_sd']
            stat.update(variable=variable,unit=unit,contrast=after+'-'+before,stage_from=before,stage_to=after,
                        before_matched_estimate=pp.estimate_before.mean(),after_matched_estimate=pp.estimate_after.mean(),
                        covered_h_before=pp.covered_h_before.sum(),covered_h_after=pp.covered_h_after.sum(),
                        reference_sd=sd,estimate_standardized=stat['estimate']/sd,
                        ci_low_standardized=stat['ci_low']/sd,ci_high_standardized=stat['ci_high']/sd,
                        status='ESTIMATED',ci_method='2000 SID percentile bootstrap',p_method='two-sided paired t; mean SID difference=0')
            adjacent.append(stat)
            pp['variable'],pp['contrast'],pp['unit']=variable,after+'-'+before,unit
            pairs.append(pp)
            draws.append(pd.DataFrame(dict(draw=np.arange(2000),variable=variable,contrast=after+'-'+before,estimate=draw)))
    adjacent = pd.DataFrame(adjacent)
    adjacent['p_BH18'] = b['bh_adjust'](adjacent.p_value)
    adjacent['BH_family_size'] = 18
    save(pd.DataFrame(summaries),STAGE,'raw_stage_summary')
    save(adjacent,STAGE,'raw_adjacent_contrasts')
    save(pd.concat(pairs,ignore_index=True),STAGE,'raw_adjacent_sid')
    save(pd.concat(draws,ignore_index=True),STAGE,'raw_adjacent_bootstrap')
    # Verify saved loss and finite support once more independently of estimators.
    predictions = pd.read_parquet(FRAME/'A3_OOF_predictions.parquet')
    assert set(predictions.sid) <= ids and np.allclose(predictions.groupby('sid').weight.sum(),1)
    for response in ['X2_shear_ms','X3_mid_rh_pct']:
        for method in METHODS:
            loss = (((predictions[response+'_observed']-predictions[response+'__'+method])**2)*predictions.weight).groupby(predictions.sid).sum().mean()
            target = losses[(losses.response==response)&(losses.method==method)].iloc[0]
            check('saved_OOF_'+response+'_'+method,np.isclose(loss,target.mse,rtol=1e-12,atol=1e-12))
    sources = [fw_path,original_b,refpath,ADD/'source_data/de_existing_sources/analysis_rows.parquet',source_stage,
               ADD/'source_data/de_existing_sources/model_stage_sid.parquet',HERE/'source_data/main_sid.parquet']
    record = dict(status='COMPLETE_FRAMEWORK_AND_RAW_STAGE_RECOMPUTATION',UTC=datetime.now(timezone.utc).isoformat(),
                  main_sid_n=len(ids),main_sid_sha256=freeze['main_sid_sha256'],framework_controlled_sid_n=int(ledger.controlled_eligible.sum()),
                  a3_supported_sid_n=predictions.sid.nunique(),a3_rows=len(predictions),existing_models_refit=50,
                  A2_bootstrap_seed=620122026,A3_bootstrap_seed=620112026,stage_bank_seed=2026101102,stage_mean_seed=20260920,
                  resampling='Original functions/seeds regenerated with frozen master or the originally defined diagnostic subset; old excluded SIDs never sampled',
                  source_data_reextracted=False,ERA5_reads=0,new_algorithms=False,checks=checks,
                  stage_adjusted_status='INPUTS_FILTERED_REQUIRES_ALL36_TARGET_REFITS',
                  F_minus_R_hours_reporting='A2 contrast is comparison bank; use original RUF method/seed20260920 recomputation for established reporting target when ready',
                  elapsed_s=time.monotonic()-started,
                  source_hashes=[dict(path=str(p),sha256=sha(p)) for p in sources],
                  outputs=[dict(path=str(p),sha256=sha(p)) for folder in [FRAME,STAGE] for p in folder.rglob('*') if p.is_file()])
    receipt.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps({k:record[k] for k in ['status','main_sid_n','framework_controlled_sid_n','a3_supported_sid_n','a3_rows','existing_models_refit','elapsed_s']}))


if __name__=='__main__':
    main()
