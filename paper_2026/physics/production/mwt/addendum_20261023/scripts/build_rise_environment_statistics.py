"""Full fixed-scope rise/environment statistics after all requests are resolved."""
from pathlib import Path
import os,sys,json,hashlib,datetime
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from Rise_Event_Statistics import Rise_Event_Statistics

W=Path(__file__).absolute().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    s=W/'source_data';out=W/'results/rise_environment';out.mkdir(parents=True,exist_ok=True)
    requests=pd.read_csv(s/'rise_unique_point_requests.csv',keep_default_na=False,na_values=['','nan','NaN'])
    ledger=pd.read_csv(s/'rise_environment_ledger_production.csv')
    expected=set(requests.loc[requests.position_available,'request_id'])
    assert set(ledger.request_id)==expected and ledger.request_id.is_unique
    assert ledger.status.eq('COMPLETE').all(),'Unresolved actual-point requests remain'
    records=[];sources=[]
    for row in ledger.itertuples(index=False):
        p=Path(row.folder);c=json.loads((p/'RISE_COMPLETE.json').read_text())
        assert c['status']=='COMPLETE' and c['request']['request_id']==row.request_id
        for name,h in c['outputs_sha256'].items():assert sha(p/name)==h
        v=json.loads((p/'values.json').read_text());b=json.loads((p/'budget.json').read_text())
        for metric in ['SST','midRH','shear200_850','PI','PI_headroom']:
            records.append({'request_id':row.request_id,'metric':metric,'radial':'200-800',
                            'value':v['values'][metric],'units':v['units'][metric]})
        for a in b['values']:
            records.append({'request_id':row.request_id,'metric':a['metric'],'radial':a['radial'],
                            'value':a['value'],'units':a['units']})
        records.append({'request_id':row.request_id,'metric':'wind_PI_support','radial':'200-800',
                        'value':c['request']['wind_kt']*1852./3600. if v['values']['PI'] is not None and c['request']['wind_kt'] is not None else np.nan,
                        'units':'m s-1'})
        sources.append({'request_id':row.request_id,'complete':str(p/'RISE_COMPLETE.json'),'sha256':sha(p/'RISE_COMPLETE.json')})
    records.extend({'request_id':r.request_id,'metric':'wind','radial':'track','value':r.wind_kt,'units':'kt'} for r in requests.itertuples(index=False))
    values=pd.DataFrame(records)
    assert not values.duplicated(['request_id','metric','radial','units']).any()
    values.to_parquet(out/'point_values.parquet',index=False)
    pd.DataFrame(sources).to_csv(out/'source_identity.csv',index=False)
    roles=pd.read_csv(s/'rise_point_roles.csv',keep_default_na=False,na_values=['','nan','NaN'])
    pairs=pd.read_parquet(s/'rise_comparable_state_pairs.parquet');events=pd.read_parquet(s/'new_return_events.parquet');intervals=pd.read_parquet(s/'new_intervals.parquet')
    pieces={k:[] for k in ['summary','event_estimates','sid_estimates','coverage','matched_bins','time_summary','time_sid']}
    rate_metrics={'fixed_tendency','motion_term','moving_tendency','horizontal_advection','horizontal_advection_fixed_frame','vertical_advection','residual'}
    for (metric,radial,unit),group in values.groupby(['metric','radial','units'],sort=True):
        result=Rise_Event_Statistics(group,roles,pairs,events,intervals,unit=unit,integrate_rate=metric in rate_metrics)
        for name in pieces:
            frame=result[name].copy();frame['metric']=metric;frame['radial']=radial
            if 'units' not in frame:frame['units']=unit
            pieces[name].append(frame)
        np.savez_compressed(out/(metric+'_'+radial+'_bootstrap.npz'),**result['bootstrap_draws'])
        print('completed',metric,radial,flush=True)
    for name,frames in pieces.items():pd.concat(frames,ignore_index=True).to_parquet(out/(name+'.parquet'),index=False)
    # The headroom components use the same PI-valid point and paired-bin support.
    sid=pd.read_parquet(out/'sid_estimates.parquet');sub=sid[sid.metric.isin(['PI','PI_headroom','wind_PI_support'])]
    pivot=sub.pivot(index=['sid','estimand'],columns='metric',values='value')
    assert pivot.notna().nunique(axis=1).eq(1).all()
    difference=pivot.PI-pivot.wind_PI_support-pivot.PI_headroom
    tol=512*np.finfo(float).eps*np.maximum(1,pivot.abs().max(axis=1))
    assert (difference.abs()<=tol).all()
    report={'status':'COMPLETE_FIXED_RISE_ENVIRONMENT_STATISTICS','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'requests':len(requests),'actual_position_requests':len(ledger),'missing_position_requests':int((~requests.position_available).sum()),
            'independent_unit':'SID','equal_bins_then_equal_events_then_equal_SID':True,'bootstrap_B':2000,'seed':650117,
            'headroom_identity_max_absolute_difference_ms':float(difference.abs().max()),
            'source_request_sha256':sha(s/'rise_unique_point_requests.csv'),'contract_sha256':sha(W/'config/rise_analysis_contract.json'),
            'statistics_sha256':sha(W/'scripts/Rise_Event_Statistics.py'),'driver_sha256':sha(__file__),
            'scope':'Actual cached/extracted environment and 700-hPa local moisture diagnostics; energy results are a separate dependency'}
    (out/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
