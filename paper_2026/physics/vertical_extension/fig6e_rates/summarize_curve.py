"""Reproduce the frozen SID draws; summarize fixed-event and available-event rate curves."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


METRICS = ['Gamma_K_h', 'Gamma_K_p', 'Gamma_K_total']
KEYS = ['sid', 'event_id', 'relative_hour']
ESTIMATES = ['estimate', 'ci_lower', 'ci_upper']
LAGS = list(range(-24, 25, 6))
PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE/'data'
RTOL = 1e-12
ATOL = 1e-13


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8', float_format='%.17g', lineterminator='\r\n')


def dump_json(value, path):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


def get_weights(indices, n):
    return np.vstack([np.bincount(row, minlength=n) for row in indices])


def summarize(records, out, qa):
    diag = json.loads(qa.read_text(encoding='utf-8-sig'))
    if diag['status']!='PASS' or diag['csv_sha256']!=sha(records):
        raise ValueError('Event-hour scientific overlap gate has not passed')
    out = Path(out).resolve()
    if out == DATA or DATA in out.parents:
        raise ValueError('Choose a separate output directory for the replay')
    out.mkdir(exist_ok=True, parents=True)
    old = read_csv(DATA/'EVENT_HOUR_RATE_COVERAGE.csv')
    assert len(old)==1188 and not old.duplicated(KEYS).any()
    r = read_csv(records)
    assert len(r)==len(old) and not r.duplicated(KEYS).any()
    assert set(r.unit)=={'W kg-1'} and set(r.layer)=={'500-850 hPa'} and set(r.region)=={'0-300 km'}
    assert set(r.window)=={'2-16 d'} and set(r.variant)=={'primary'} and diag['event_hours']==len(r)
    r = r.sort_values(KEYS).set_index(KEYS,verify_integrity=True)
    old = old.sort_values(KEYS).set_index(KEYS,verify_integrity=True)
    assert r.index.equals(old.index)
    if 'sec' in r:
        np.testing.assert_array_equal(r.sec,old.sec)
    else:
        assert pd.to_datetime(r.UTC,utc=True).equals(pd.to_datetime(old.UTC,utc=True))
    r['original_status']=old.status
    if 'UTC' not in r:r['UTC']=old['UTC']
    r['sec']=old.sec
    valid = np.isfinite(r[METRICS].to_numpy())
    assert not ((valid.sum(axis=1)>0)&(valid.sum(axis=1)<3)).any()
    r['valid']=valid.all(axis=1)
    old_valid = old[METRICS].notna().all(axis=1)
    assert old_valid.sum()==810
    close(r.loc[old_valid,METRICS],old.loc[old_valid,METRICS])
    assert (~r.loc[old.status.isin(['no accepted center','cached but joint regional rates invalid']),'valid']).all()
    joint = r[r.valid]
    close(joint[METRICS[2]], joint[METRICS[0]]+joint[METRICS[1]])
    r['common_n_lags']=r.valid.groupby(level=KEYS[:2]).transform('sum')
    r['common_nine']=r.common_n_lags.eq(9)
    r = r.reset_index()
    group = r.groupby(KEYS[:2], sort=True)
    eventlist = group[['common_n_lags','common_nine']].first()
    eventlist['unavailable_hours'] = group.apply(lambda d:','.join(str(h) for h in d.loc[~d.valid,'relative_hour']))
    eventlist['reason_by_hour'] = group.apply(lambda d:' | '.join(f"{x.relative_hour}h {x.status}" for x in d.loc[~d.valid].itertuples()))
    dump(eventlist.reset_index(),out/'FIXED_NINE_EVENT_MEMBERSHIP.csv')
    dump(r[KEYS+['UTC','sec','valley_s','return_s','within_rise','original_status','status','valid','common_nine']+METRICS],out/'EVENT_HOUR_SUPPORT_AND_RATES.csv')
    membership = read_csv(DATA/'FIXED_NINE_EVENT_MEMBERSHIP.csv').set_index(KEYS[:2]).sort_index()
    assert eventlist.index.equals(membership.index)
    np.testing.assert_array_equal(eventlist.common_nine,membership.common_nine)
    np.testing.assert_array_equal(eventlist.common_n_lags,membership.common_n_lags)
    draw_path = DATA/'ORIGINAL_SID_DRAW_INDICES.npz'
    if sha(draw_path)!='0a8f303af255616df3dfce4aa32df08b15191261e7a72868b09f77f4664fd8d4':
        raise ValueError('Saved SID draw archive differs from the published identity')
    with np.load(draw_path,allow_pickle=False) as ref:
        ids=ref['sid_order'].tolist()
        ix=ref['indices']
        assert ids==sorted(old.reset_index().sid.unique()) and ix.shape==(2000,122)
        np.testing.assert_array_equal(ref['draw_ids'],np.arange(2000))
        assert np.min(ix)>=0 and np.max(ix)<len(ids)
    boot_weights = get_weights(ix,len(ids))
    stats, support, draw_store, sid_tables = [], [], {}, []
    masks = {'nine_time_common':r.common_nine, 'available_at_time':r.valid}
    max_resid={'event':float(abs(joint[METRICS[2]]-joint[METRICS[0]]-joint[METRICS[1]]).max()),
               'sid':0., 'mean':0., 'draw':0.}
    for setname, mask in masks.items():
        kept=r[mask]
        if set(kept.relative_hour)!=set(LAGS):raise ValueError(f'Insufficient event support in {setname}')
        for lag,g in kept.groupby('relative_hour'):
            sv=g.groupby('sid')[METRICS].mean().sort_index()
            present=[sid for sid in ids if sid in sv.index]
            values=sv.loc[present].to_numpy()
            weights=boot_weights[:,np.isin(ids,present)]
            num=weights.sum(axis=1)
            assert len(num)==2000 and num.min()>0
            point=values.mean(axis=0)
            draws=(weights@values)/num[:,None]
            lo,hi=np.percentile(draws,[2.5,97.5],axis=0)
            for name,arr in [('sid',values),('mean',point[None,:]),('draw',draws)]:
                close(arr[:,2],arr[:,0]+arr[:,1])
                max_resid[name]=max(max_resid[name],float(np.max(abs(arr[:,2]-arr[:,0]-arr[:,1]))))
            draw_store[f'{setname}_lag{lag}']=draws
            draw_store[f'{setname}_n_lag{lag}']=num
            sid_tab=sv.reset_index()
            sid_tab['n_events_in_sid']=sid_tab.sid.map(g.groupby('sid').size())
            sid_tab['relative_hour']=lag
            sid_tab['support']=setname
            sid_tables.append(sid_tab)
            common_fields=dict(support=setname,relative_hour=int(lag),n_sid=len(present),n_events=len(g))
            support.append(dict(**common_fields, min_sid_draw_slots=int(num.min()),max_sid_draw_slots=int(num.max()),
                                sid_list=';'.join(present), event_list=';'.join(sorted(g.event_id))))
            for j,m in enumerate(METRICS):
                stats.append(dict(**common_fields,metric=m,unit='W kg-1',estimate=point[j],ci_lower=lo[j],ci_upper=hi[j],
                                  n_draws=len(draws),interval='SID bootstrap percentile 95%',region='0-300 km',
                                  layer='500-850 hPa',window='2-16 d',variant='primary',source_sha256=sha(records)))
    tab=pd.DataFrame(stats)
    dump(tab,out/'FIG6E_RATE_ESTIMATES.csv')
    dump(pd.DataFrame(support),out/'NINE_TIME_COVERAGE.csv')
    dump(pd.concat(sid_tables),out/'SID_MEANS_BY_HOUR.csv')
    np.savez_compressed(out/'RATE_DRAWS_BY_HOUR.npz',metric=np.asarray(METRICS),draw_id=np.arange(2000),**draw_store)
    common=tab.query("support == 'nine_time_common'").set_index(['relative_hour','metric'])
    avail=tab.query("support == 'available_at_time'").set_index(['relative_hour','metric'])
    diff=avail[ESTIMATES+['n_sid','n_events']].join(common[ESTIMATES+['n_sid','n_events']],lsuffix='_available',rsuffix='_common')
    diff['common_minus_available_estimate']=diff.estimate_common-diff.estimate_available
    dump(diff.reset_index(),out/'SUPPORT_COMPOSITION_COMPARISON.csv')
    ref=read_csv(DATA/'COMPLETE_LAGS_ONLY_RATES.csv').set_index(['relative_hour','metric'])
    for lag in [0,6]:close(avail.loc[lag,ESTIMATES].sort_index(),ref.loc[lag,ESTIMATES].sort_index())
    # Re-read the committed products and verify the 2.5/97.5 quantiles for every line.
    check=read_csv(out/'FIG6E_RATE_ESTIMATES.csv')
    assert len(check)==54 and not check.duplicated(['support','relative_hour','metric']).any()
    with np.load(out/'RATE_DRAWS_BY_HOUR.npz',allow_pickle=False) as archive:
        for row in check.itertuples():
            b=archive[f'{row.support}_lag{row.relative_hour}'][:,METRICS.index(row.metric)]
            close([row.ci_lower,row.ci_upper],np.percentile(b,[2.5,97.5]))
    entry=dict(status='PASS',frozen_sid=122,frozen_events=132,n_slots=len(r),originally_covered=int(old_valid.sum()),
               n_valid_event_hours=int(r.valid.sum()),n_fixed_events=int(eventlist.common_nine.sum()),
               n_fixed_sid=int(common.n_sid.iloc[0]),same_lags=LAGS,
               event_hour_sha256=sha(records),source_qa_sha256=sha(qa),max_identity_abs_error_Wkg=max_resid,
               plot_data_sha256=sha(out/'FIG6E_RATE_ESTIMATES.csv'),saved_draw_sha256=sha(out/'RATE_DRAWS_BY_HOUR.npz'),
               source_measure='KE canonical-transfer rate (W kg-1)', bootstrap=2000, plot_support='nine_time_common')
    dump_json(entry,out/'CURVE_STATISTICS_QA.json')
    return entry


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rates', type=Path, default=DATA/'FINAL_EVENT_HOUR_RATES.csv')
    parser.add_argument('--qa', type=Path, default=DATA/'FINAL_EVENT_HOUR_QA.json')
    parser.add_argument('--out', type=Path, default=Path('fig6e_statistics'))
    args=parser.parse_args()
    summarize(args.rates,args.out,args.qa)
    keys=['support','relative_hour','metric']
    saved=read_csv(DATA/'FIG6E_RATE_ESTIMATES.csv').set_index(keys).sort_index()
    replayed=read_csv(args.out/'FIG6E_RATE_ESTIMATES.csv').set_index(keys).sort_index()
    assert saved.index.equals(replayed.index)
    for col in ['n_sid','n_events','n_draws']:
        np.testing.assert_array_equal(replayed[col],saved[col])
    close(replayed[ESTIMATES],saved[ESTIMATES])
    print('Verified 54 estimates and both interval limits for each against the released table.')
