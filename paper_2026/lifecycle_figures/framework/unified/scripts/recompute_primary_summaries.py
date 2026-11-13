"""Reaggregate accepted lifecycle outputs on the single frozen master cohort.

No boundary is recomputed, and no ERA5 data are read. Original statistical
functions, parameter grids, follow-up labels and genesis-basin rules are reused.
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
import statsmodels.api as sm
from scipy.stats import spearmanr, t as student_t
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).absolute().parents[1]
ADD = HERE.parent
OUT = HERE / 'statistics/primary'
THR = HERE / 'statistics/threshold'
OLDTEMP = HERE.parent / 'dependencies/temporal'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_functions(path, names, extra=None):
    tree = ast.parse(Path(path).read_text('utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    scope = {'np': np, 'pd': pd, 'sm': sm, 'student_t': student_t,
             'spearmanr': spearmanr, **(extra or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def main():
    started = time.monotonic()
    receipt = HERE / 'reports/primary_summaries_receipt.json'
    assert not receipt.exists(), 'Completed output must not be overwritten'
    OUT.mkdir(parents=True, exist_ok=True)
    THR.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((HERE / 'reports/main_sid_freeze_receipt.json').read_text('utf-8'))
    assert sha(HERE/'source_data/main_sid.txt') == freeze['main_sid_sha256']
    master = pd.read_parquet(HERE / 'source_data/main_sid.parquet')
    ids = set(master.sid)
    durations = pd.read_parquet(HERE / 'source_data/main_durations.parquet')
    events = pd.read_parquet(HERE / 'source_data/all_return_events.parquet')
    outcomes = pd.read_parquet(HERE / 'source_data/historical_first_events_with_retained_outcomes.parquet')
    assert master.sid.is_unique and len(master) == freeze['main_sid_n']
    basin = []
    for basin_name in ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']:
        group = master[master.basin_genesis.eq(basin_name)]
        number = int(group.sid.isin(set(events.sid)).sum())
        basin.append(dict(mode='new', basin_genesis=basin_name, recurrent_SID=number,
                          eligible_SID=len(group), percentage=100 * number / len(group),
                          denominator='unique frozen main SID in genesis basin, 1982-2023'))
    basin = pd.DataFrame(basin)
    assert basin.recurrent_SID.sum() == events.sid.nunique()
    assert basin.eligible_SID.sum() == len(master)
    basin.to_csv(HERE / 'source_data/basin_recurrence_proportions.csv', index=False)
    events.to_csv(HERE / 'source_data/new_return_events.csv', index=False)
    # Same event records, alias for the existing standalone plotting interface.
    pd.testing.assert_frame_equal(events, pd.read_parquet(ADD / 'source_data/new_return_events.parquet'), check_exact=True)
    print('BASIN_COUNTS_READY', flush=True)
    summary = []
    for stage, col in [('D', 'development_duration_hours'), ('E', 'escalation_duration_hours'),
                       ('M', 'maturity_duration_hours'), ('Res', 'resolution_duration_hours')]:
        v = master[col].to_numpy(float)
        summary.append(dict(cohort='single_master', stage=stage, n_sid=len(v),
                            mean_h=v.mean(), median_h=np.median(v), q25_h=np.quantile(v, .25),
                            q75_h=np.quantile(v, .75), mean_fraction=np.mean(v / master.lifetime_hours)))
    pd.DataFrame(summary).to_csv(OUT / 'stage_duration_summary.csv', index=False)
    previous = pd.read_parquet(ADD / 'source_data/old_main_SID.parquet').set_index('sid').loc[master.sid]
    changed = pd.to_datetime(previous.escalation_time).to_numpy() != pd.to_datetime(master.escalation_time).to_numpy()
    boundary_change = master.loc[changed, ['sid', 'escalation_time']].copy()
    boundary_change['old_escalation_time'] = previous.loc[boundary_change.sid, 'escalation_time'].to_numpy()
    boundary_change['change_h'] = (pd.to_datetime(boundary_change.escalation_time) - pd.to_datetime(boundary_change.old_escalation_time)).dt.total_seconds() / 3600
    boundary_change.to_csv(OUT / 'changed_E_boundaries_master.csv', index=False)
    master.groupby('genesis_year').size().rename('main_sid_n').to_csv(OUT / 'annual_master_counts.csv')
    outcomes.groupby(['return48_status', 'followup48_complete']).size().rename('sid_n').to_csv(OUT / 'retained_followup_counts.csv')
    # Preserve the original primary seven-target family and the original
    # full-history complexity label; this is distinct from anchored recurrence.
    legacy = pd.read_csv(OLDTEMP / 'tables/temporal_per_SID.csv', keep_default_na=False,
                         na_values=['', 'nan', 'NaN']).set_index('sid').loc[master.sid].copy()
    legacy['D_h'] = master.development_duration_hours.to_numpy()
    legacy['E_h'] = master.escalation_duration_hours.to_numpy()
    for metric, col in [('F_h', 'maturity_duration_hours'), ('Resolution_h', 'resolution_duration_hours'),
                        ('A_signed', 'lifecycle_asymmetry_time')]:
        assert np.allclose(legacy[metric], master[col], atol=1e-12, rtol=0)
    legacy['A_absolute'] = legacy.A_signed.abs()
    metrics = pd.read_csv(OLDTEMP / 'tables/primary_trends.csv').metric.tolist()
    env = source_functions(OLDTEMP / 'temporal_analysis.py', ['fit_trend'],
                           {'PROPORTIONS': ['complex_fraction', 'multiple_fraction', 'single_reintensification_fraction']})
    trends, annual, fitted = [], [], []
    for metric in metrics:
        y = legacy.groupby('genesis_year')[metric].mean().reindex(range(1982, 2024))
        record, fit = env['fit_trend'](y.index, y.to_numpy(), metric)
        record['family'] = 'primary'
        record['main_sid_n'] = len(master)
        trends.append(record)
        fitted.append(fit)
        for year, value in y.items():
            g = legacy[legacy.genesis_year.eq(year)]
            annual.append(dict(metric=metric, year=int(year), mean=value, median=g[metric].median(),
                               denominator=len(g), year_basis='framework genesis UTC year'))
    trends = pd.DataFrame(trends)
    trends['q_value'] = multipletests(trends.p_value, method='fdr_bh')[1]
    trends['family_test_n'] = len(metrics)
    trends.to_csv(OUT / 'primary_trends.csv', index=False)
    pd.DataFrame(annual).to_csv(OUT / 'annual_metrics.csv', index=False)
    pd.concat(fitted, ignore_index=True).to_csv(OUT / 'annual_fitted_values.csv', index=False)
    # Fixed master membership for every already-computed original configuration.
    manifest = pd.read_csv(ADD / 'source_data/de_threshold/manifest.csv', keep_default_na=False)
    assert len(manifest) == 18
    compare = source_functions(ADD / 'scripts/de_threshold_sensitivity.py', ['compare'])['compare']
    tables, observations, grid_rows, support_rows = {}, {}, [], []
    hashes = []
    for r in manifest.to_dict('records'):
        name = r['experiment_id']
        folder = ADD / 'source_data/de_threshold' / name
        cp = json.loads((folder / 'complete.json').read_text('utf-8'))
        assert cp['status'] == 'COMPLETE'
        for file_name, expected in cp['outputs'].items():
            assert sha(folder / file_name) == expected
            hashes.append({'path': str(folder / file_name), 'sha256': expected})
        q = pd.read_parquet(folder / 'QC.parquet')
        s = pd.read_parquet(folder / 'summary.parquet')
        t = pd.read_parquet(folder / 'observations.parquet')
        q = q[q.sid.isin(ids)].copy()
        s = s[s.sid.isin(ids)].sort_values('sid').reset_index(drop=True)
        t = t[t.sid.isin(ids)].copy()
        assert set(q.sid) == ids and set(t.sid) == set(s.sid)
        destination = THR / name
        destination.mkdir(exist_ok=True)
        for label, table in [('QC', q), ('summary', s), ('observations', t)]:
            table.to_parquet(destination / (label + '.parquet'), index=False)
        tables[name], observations[name] = s, t
        for row in q[['sid', 'valid_lifecycle', 'qc_reason']].to_dict('records'):
            support_rows.append({'experiment_id': name, **row})
        rec = {k: r[k] for k in ['experiment_no', 'experiment_id', 'is_baseline', 'escalation_progress', 'maturity_progress', 'min_peak_gain']}
        rec.update(fixed_main_sid_n=len(master), n_valid=len(s), n_unavailable=len(master) - len(s),
                   valid_fraction=len(s) / len(master), n_obs=len(t),
                   qc_reason_counts_json=json.dumps(q.qc_reason.value_counts().to_dict()))
        for dest, col in [('lifetime_h', 'lifetime_hours'), ('development_h', 'development_duration_hours'),
                          ('escalation_h', 'escalation_duration_hours'), ('maturity_h', 'maturity_duration_hours'),
                          ('resolution_h', 'resolution_duration_hours'), ('lmi_kt', 'lmi'),
                          ('mir_kt24h', 'mir'), ('mwr_kt24h', 'mwr')]:
            rec['median_' + dest] = s[col].median()
        rec['complex_lifecycle_fraction'] = (s.near_lmi_episode_count.gt(1) | s.reintensification_event_count.gt(0)).mean()
        rec['recurrence_scope'] = 'existing full-lifecycle complexity diagnostic; not anchored-return event count'
        grid_rows.append(rec)
    base = manifest.loc[manifest.is_baseline.eq(1), 'experiment_id'].item()
    assert set(tables[base].sid) == ids
    comparisons = []
    for r in grid_rows:
        name = r['experiment_id']
        comparisons.append({**{k: r[k] for k in ['experiment_id', 'is_baseline', 'escalation_progress', 'maturity_progress', 'min_peak_gain']},
                             'fixed_main_sid_n': len(master),
                             **compare(tables[base], observations[base], tables[name], observations[name])})
    pd.DataFrame(grid_rows).to_csv(THR / 'manifest.csv', index=False)
    pd.DataFrame(comparisons).to_csv(THR / 'stability_vs_master_baseline.csv', index=False)
    pd.DataFrame(support_rows).to_parquet(THR / 'per_configuration_master_support.parquet', index=False)
    core = sorted(set.intersection(*(set(s.sid) for s in tables.values())))
    pd.DataFrame({'sid': core}).to_csv(THR / 'all_configurations_valid_subset.csv', index=False)
    by_sid = {name: s.set_index('sid') for name, s in tables.items()}
    spread = []
    for sid in core:
        row = {'sid': sid}
        for boundary in ['escalation', 'maturity', 'resolution']:
            tt = pd.to_datetime([s.loc[sid, boundary + '_time'] for s in by_sid.values()])
            row[boundary + '_range_h'] = (tt.max() - tt.min()).total_seconds() / 3600
        spread.append(row)
    pd.DataFrame(spread).to_csv(THR / 'boundary_spread_on_common_valid_subset.csv', index=False)
    result = dict(status='COMPLETE_PRIMARY_DESCRIPTIVE_AND_FIXED_MASTER_LIFECYCLE_SENSITIVITY',
                  UTC=datetime.now(timezone.utc).isoformat(), main_sid_n=len(master),
                  main_sid_sha256=freeze['main_sid_sha256'], changed_E_sid_n=len(boundary_change),
                  basin_counts=basin.to_dict('records'), primary_trend_targets=len(metrics),
                  sensitivity_configurations=18, all_configuration_valid_subset_n=len(core),
                  sensitivity_support_range=[min(r['n_valid'] for r in grid_rows), max(r['n_valid'] for r in grid_rows)],
                  no_new_base_cohort=True, ERA5_reads=0, new_boundary_computations=0,
                  source_hashes=hashes,
                  methods_reused={'trends': 'Original fit_trend; equal-year means, Bartlett HAC3, finite correction, t40 and original seven-target BH family',
                                  'sensitivity': 'Original compare function and unchanged 18-configuration outputs, each restricted to frozen master; unavailable values retained'},
                  remaining=['Anchored-return threshold sensitivity and environmental sensitivity must retain their original exact analyses; this receipt covers lifecycle configuration comparisons only',
                             'Downstream adjusted models, physical diagnostics, regional and recovery fits, manuscripts and publication are not covered'],
                  elapsed_s=time.monotonic() - started,
                  outputs=[{'path': str(p), 'sha256': sha(p)} for folder in [OUT, THR] for p in folder.rglob('*') if p.is_file()])
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'main_sid_n', 'changed_E_sid_n', 'sensitivity_support_range', 'all_configuration_valid_subset_n', 'elapsed_s']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
