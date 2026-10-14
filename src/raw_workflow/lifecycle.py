"""Run the original lifecycle calculation on explicit frozen IBTrACS identities."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from .ibtracs_prepare import prepare_records
from .lifecycle_core import identify_tc_crisis_lifecycle
from .representations import contiguous_window, decompose_track, reintensification_episodes

ROOT = Path(__file__).absolute().parents[2]
NA_VALUES = ['', ' ', 'NaN', '-9999', '-9999.0']
FIELDS = ['SID', 'SEASON', 'NUMBER', 'BASIN', 'SUBBASIN', 'NAME', 'ISO_TIME',
          'NATURE', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'TRACK_TYPE',
          'DIST2LAND', 'LANDFALL', 'USA_STATUS', 'USA_RECORD', 'USA_WIND', 'USA_PRES']


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_selected(csv_file, sids):
    """Retain all raw records for each selected SID, including cross-year records."""
    available = pd.read_csv(csv_file, nrows=0).columns.str.strip().tolist()
    required = {'SID', 'ISO_TIME', 'USA_WIND', 'LAT', 'LON'}
    if not required.issubset(available):
        raise ValueError('Missing required IBTrACS columns: ' + repr(sorted(required-set(available))))
    chunks = []
    for part in pd.read_csv(csv_file, usecols=[c for c in FIELDS if c in available],
                            dtype=str, chunksize=100000,
                            keep_default_na=False, na_values=NA_VALUES):
        part.columns = part.columns.str.strip()
        chunks.append(part[part.SID.astype(str).isin(sids)])
    return pd.concat(chunks, ignore_index=True)


def build_representation(track, summary):
    g = track.rename(columns={'time': 'UTC', 'wind': 'wind_kt', 'stage_code': 'F_stage'}).copy()
    g['sid'] = summary['sid']
    g['record_index'] = np.arange(len(g))
    r = contiguous_window(g.UTC, g.wind_kt, summary['lmi_time'],
                          summary['lmi'], summary['mature_threshold_wind'], relative=True)
    g['R90C_selected'] = r['selected']
    row = dict(summary, R90C_first_selected_time=g.UTC.iloc[r['first_i']])
    records, durations = decompose_track(g, row)
    events = reintensification_episodes(records, pd.Timestamp(summary['lmi_time']), 10.0)
    return records, durations, events


def compare_frozen(summary, records, expected, expected_records):
    """Compare scientific fields; record_index is local to this explicit run."""
    checks = []
    date_fields = ['genesis_time', 'escalation_time', 'maturity_time',
                   'resolution_time', 'termination_time', 'lmi_time', 'lmi_last_time']
    numeric_fields = ['genesis_wind', 'lmi', 'mature_threshold_wind',
                      'development_duration_hours', 'escalation_duration_hours',
                      'maturity_duration_hours', 'resolution_duration_hours',
                      'lifecycle_asymmetry_time']
    for row in summary.to_dict('records'):
        old = expected.loc[row['sid']]
        for field in date_fields:
            okay = pd.to_datetime(row[field],utc=True) == pd.to_datetime(old[field],utc=True)
            checks.append({'sid': row['sid'], 'field': field, 'matches': bool(okay)})
        for field in numeric_fields:
            if field in old.index:
                okay = np.isclose(float(row[field]), float(old[field]), rtol=0, atol=1e-12, equal_nan=True)
                checks.append({'sid': row['sid'], 'field': field, 'matches': bool(okay)})
    pairs = [('wind_kt','wind_kt'), ('F_stage','F_stage'), ('wind_filled_flag','wind_filled_flag'),
             ('R_selected','R_selected'), ('U_selected','U_selected'),
             ('F_selected','F_selected'), ('valid_6h_interval','valid_6h_interval')]
    expected_records = expected_records.copy()
    expected_records['UTC'] = pd.to_datetime(expected_records.UTC,utc=True)
    for sid,g in records.groupby('sid',sort=False):
        a = g.sort_values('UTC').reset_index(drop=True)
        b = expected_records[expected_records.sid.eq(sid)].sort_values('UTC').reset_index(drop=True)
        same_time = np.array_equal(pd.to_datetime(a.UTC,utc=True).to_numpy(),b.UTC.to_numpy())
        checks.append({'sid':sid,'field':'complete_lifecycle_UTC','matches':bool(same_time)})
        if same_time:
            for col,newcol in pairs:
                aa=a[col].to_numpy();bb=b[newcol].to_numpy()
                okay = np.array_equal(aa,bb,equal_nan=True) if aa.dtype.kind in 'fi' else np.array_equal(aa,bb)
                checks.append({'sid':sid,'field':col,'matches':bool(okay)})
    return pd.DataFrame(checks)


def run(csv_file, out, sids=None, all_frozen=False, config=None):
    cfg=json.loads(Path(config or ROOT/'config/raw_inputs.json').read_text(encoding='utf-8'))['ibtracs']
    actual_hash=file_hash(csv_file)
    if actual_hash!=cfg['sha256']:
        raise ValueError('IBTrACS snapshot differs from the frozen study input; no implicit substitution.')
    frozen=pd.read_csv(ROOT/cfg['frozen_sid_table'],keep_default_na=False)
    if not frozen.sid.is_unique:
        raise ValueError('Frozen SID table is not unique')
    if not all_frozen and not sids:
        raise ValueError('Supply explicit --sid values or --all-frozen')
    selected=set(frozen.sid) if all_frozen else set(sids)
    if not selected.issubset(set(frozen.sid)):
        raise ValueError('Requested SID is outside the frozen primary cohort')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    raw=read_selected(csv_file,selected)
    data=prepare_records(raw,*cfg['genesis_years'])
    summaries=[];qc=[];tracks=[];events=[]
    for sid,g in data.groupby('SID',sort=False):
        g=g.sort_values('ISO_TIME').reset_index(drop=True)
        names=g.NAME.dropna().astype(str);seasons=pd.to_numeric(g.SEASON,errors='coerce').dropna()
        result=identify_tc_crisis_lifecycle(time=g.ISO_TIME.to_numpy(),lat=g.LAT.to_numpy(),
            lon=g.LON_180.to_numpy(),wind=g.WIND.to_numpy(),pressure=g.PRES.to_numpy(),
            basin=g.BASIN.to_numpy(),is_tropical=g.IS_TROPICAL.to_numpy(),
            dist2land=g.DIST2LAND.to_numpy(),landfall=g.LANDFALL.to_numpy(),
            wind_filled_flag=g.WIND_WAS_FILLED.to_numpy(),sid=sid,
            name=names.iloc[0] if len(names) else '',season=float(seasons.iloc[0]) if len(seasons) else np.nan,
            **cfg['lifecycle_parameters'])
        q=result['summary'];qc.append(q)
        if not q['valid_lifecycle']:continue
        if not cfg['genesis_years'][0]<=pd.Timestamp(q['genesis_time']).year<=cfg['genesis_years'][1]:
            raise ValueError('Frozen SID has a changed framework genesis year')
        records,durations,episodes=build_representation(result['track'],q)
        summaries.append(dict(q,**durations));tracks.append(records);events.extend(episodes)
    pd.DataFrame(qc).to_csv(out/'lifecycle_qc.csv',index=False)
    summary=pd.DataFrame(summaries)
    if set(summary.sid)!=selected:
        raise ValueError('A frozen SID is absent or no longer satisfies the original lifecycle definition; inspect lifecycle_qc.csv')
    records=pd.concat(tracks,ignore_index=True)
    expected_records=pd.read_csv(ROOT/cfg['frozen_record_table'],keep_default_na=False,na_values=['','NaN','nan'])
    comparisons=compare_frozen(summary,records,frozen.set_index('sid'),expected_records)
    summary.to_csv(out/'lifecycle_summary.csv',index=False)
    records.to_csv(out/'lifecycle_records.csv.gz',index=False)
    pd.DataFrame(events).to_csv(out/'reintensification_episodes.csv',index=False)
    comparisons.to_csv(out/'frozen_field_comparison.csv',index=False)
    receipt={'input_name':Path(csv_file).name,'input_sha256':actual_hash,'selected_sid_n':len(selected),
             'retained_sid_n':len(summary),'lifecycle_record_n':len(records),
             'all_frozen_fields_match':bool(comparisons.matches.all()),
             'entire_frozen_cohort_run':bool(all_frozen),'new_fitting':False,
             'time_basis':'UTC; complete selected-SID tracks; genesis-year cohort'}
    (out/'calculation_summary.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    if not comparisons.matches.all():
        raise ValueError('Recomputed fields differ from frozen inputs; new outputs do not replace published inputs')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ibtracs',required=True);p.add_argument('--out',required=True)
    q=p.add_mutually_exclusive_group(required=True)
    q.add_argument('--sid',action='append');q.add_argument('--all-frozen',action='store_true')
    p.add_argument('--config')
    a=p.parse_args();print(json.dumps(run(a.ibtracs,a.out,a.sid,a.all_frozen,a.config),indent=2))
