"""Streaming, restartable statistics of the frozen actual rise-point requests."""
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_name]='1'
from pathlib import Path
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import shutil
import sys
import time
import h5py
import numpy as np
import pandas as pd
from Rise_Event_Statistics import Rise_Event_Statistics

ADD=Path(__file__).absolute().parents[1];WORK=ADD.parent
KEYS=['metric','window','level','region','scheme']
TABLES=['summary','event_estimates','sid_estimates','coverage','matched_bins','time_summary','time_sid']
TABLE_COLUMNS={
 'summary':['estimand','sample','units','n_events','estimate','lower','upper','n_sid'],
 'event_estimates':['sid','event_id','return_order','estimand','value','units','n_joint_bins'],
 'sid_estimates':['sid','estimand','value','units'],
 'coverage':['sid','event_id','n_pre_valid','rise_complete','rise_duration_h','rise_n_intervals','rise_n_valid_intervals','Resolution_complete','Resolution_duration_h'],
 'matched_bins':['sid','event_id','pair_id','wind_bin','value_rise','value_Resolution','value','joint_valid'],
 'time_summary':['relative_hour','units','n_events','n_events_within_rise','n_sid_within_rise','estimate','lower','upper','n_sid'],
 'time_sid':['sid','relative_hour','value']}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=True),encoding='utf8')
    deadline=time.monotonic()+60
    while True:
        try:os.replace(temp,path);return
        except OSError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or time.monotonic()>deadline:raise
            time.sleep(.2)


def schema(work=WORK,add=ADD):
    semantics=load(work/'reports/mwt_metric_semantics.json')['metrics']
    groups=[]
    for m in semantics:
        levels=['500-850'] if m['metric'].startswith('vertical_input_') else [str(int(p)) for p in m['pressure_hpa']]
        for level in levels:
            for window in range(3):
                for region in ['r0_300','r200_800','r0_500']:
                    for scheme in ['main','shift16d']:
                        groups.append(dict(group_id=f'g{len(groups):04d}',metric=m['metric'],level=level,window=window,
                            region=region,scheme=scheme,units=m['units_stored'],category=m['category'],
                            integrate_rate=m['units_stored'] in ('W kg-1','W m-2')))
    groups=pd.DataFrame(groups)
    if len(groups)!=3834:raise ValueError('The frozen 25-diagnostic pressure inventory changed')
    primary_metrics=['KE','APE','C_A_to_K','vertical_input_K_Wkg','vertical_input_A_Wkg','vertical_input_P_Wkg']
    primary=groups[groups.metric.isin(primary_metrics)&groups.level.isin(['700','500-850'])&
                   groups.window.eq(1)&groups.region.eq('r0_300')&groups.scheme.eq('main')].copy()
    if len(primary)!=6:raise ValueError('Primary diagnostic identity is ambiguous')
    jobs=[]
    def add_job(row,mode,source_ids,role='value',comparison=''):
        jobs.append({**row,'job_id':f'j{len(jobs):04d}','support_mode':mode,'source_group_ids':source_ids,
                     'comparison_role':role,'comparison_id':comparison})
    for row in groups.to_dict('records'):add_job(row,'native_diagnostic',[row['group_id']])
    common_ids=primary.group_id.tolist()
    for row in primary.to_dict('records'):add_job(row,'joint_primary_six',[row['group_id']]+[g for g in common_ids if g!=row['group_id']])
    for row in primary.to_dict('records'):
        for compare,scheme,region in [('input_shift16d_minus_main','shift16d','r0_300'),('radius500_minus300','main','r0_500')]:
            match=groups[groups.metric.eq(row['metric'])&groups.level.eq(row['level'])&groups.window.eq(1)&groups.scheme.eq(scheme)&groups.region.eq(region)]
            if len(match)!=1:raise ValueError('Sensitivity identity mismatch')
            alternative=match.iloc[0].to_dict();source=[row['group_id'],alternative['group_id']]
            for role in ['reference','alternative','difference']:
                add_job(alternative if role=='alternative' else row,'paired_configuration_common',source,role,compare)
    if len(jobs)!=3876:raise ValueError('Unexpected fixed statistics scope')
    return groups,jobs


def metadata(add=ADD):
    roles=pd.read_parquet(add/'source_data/rise_point_roles.parquet')
    pairs=pd.read_parquet(add/'source_data/rise_comparable_state_pairs.parquet')
    events=pd.read_parquet(add/'source_data/new_return_events.parquet')
    intervals=pd.read_parquet(add/'source_data/new_intervals.parquet')
    intervals=intervals[intervals.sid.isin(events.sid.unique())].copy()
    requests=pd.read_csv(add/'source_data/rise_unique_point_requests.csv',dtype={'sid':str,'request_id':str}).sort_values(['sid','UTC']).reset_index(drop=True)
    if len(events)!=133 or events.sid.nunique()!=123 or len(requests)!=2646 or requests.request_id.duplicated().any():
        raise ValueError('Frozen rise population changed')
    return dict(roles=roles,pairs=pairs,events=events,intervals=intervals,requests=requests)


def source_gate(add=ADD,work=WORK):
    execution=load(add/'config/increment_execution_contract.json')
    for relative,digest in execution['files_sha256'].items():
        if sha(work/relative)!=digest:raise ValueError('Increment execution source changed: '+relative)
    status=load(add/'reports/increment_queue_status.json')
    if status.get('stage')!='COMPLETE_123_DIRECT_POINT_CACHES' or status.get('running') or status.get('failed'):
        raise RuntimeError('Complete 123-SID validated increments are required before reading any energy values')
    meta=metadata(add);ids=sorted(meta['requests'].sid.unique())
    if sorted(status['completed'])!=ids:raise ValueError('Increment completion population differs')
    manifest=[]
    for sid in ids:
        folder=add/'cache'/sid;rp=folder/'completed.json';vp=add/'validation/increment_cache'/sid/'validated.json'
        done=load(rp);review=load(vp)
        if done['status']!='COMPLETE_EXACT_REQUEST_MWT_INCREMENT' or review['status']!='VALIDATED_INCREMENT_STATED_SCOPE':raise ValueError('SID incomplete: '+sid)
        if review['completed_receipt_sha256']!=sha(rp) or review['validator_sha256']!=sha(add/'scripts/validate_mwt_rise_increment.py'):
            raise ValueError('Independent SID validation identity changed')
        for name,info in review['output_identities'].items():
            st=(folder/name).stat()
            if st.st_size!=info['bytes'] or st.st_mtime_ns!=info['mtime_ns'] or done['outputs_sha256'][name]!=info['sha256']:
                raise ValueError('Input changed after independent validation')
        if sid=='1982005S10105' and (not review['actual_source_original_MWT_checked'] or review['genuinely_new_source_UTC_rows_checked']<1):
            raise ValueError('Increment pilot source/new-row gate absent')
        manifest.append({'sid':sid,'completion_sha256':sha(rp),'validation_sha256':sha(vp),
                         'regional_path':str(folder/'regional_metrics.parquet'),
                         'regional_sha256':done['outputs_sha256']['regional_metrics.parquet'],
                         'all_original_support_fields':'Retained unchanged in the referenced regional parquet and protected diagnostics sidecar'})
    return manifest,meta


def build_matrix(destination,groups,manifest,requests):
    target=destination/'request_values.h5';receipt=destination/'request_values_completed.json'
    identity={'manifest':manifest,'group_records':groups.to_dict('records'),'request_ids':requests.request_id.tolist()}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        prior=load(receipt)
        if prior['input_identity']==digest and sha(target)==prior['matrix_sha256']:return
        raise ValueError('Existing request matrix differs; no overwrite')
    temp=destination/'request_values.partial.h5'
    lookup=groups[KEYS+['group_id','units']].copy();lookup['row']=np.arange(len(groups))
    with h5py.File(temp,'w') as h:
        h.attrs['scope']='Actual stored regional values after original per-metric QC; request-level masks do not impose common grid-cell support'
        h.create_dataset('request_id',data=requests.request_id.to_numpy(object),dtype=h5py.string_dtype('utf8'))
        h.create_dataset('group_id',data=groups.group_id.to_numpy(object),dtype=h5py.string_dtype('utf8'))
        for field in ['value','valid_area_fraction']:
            h.create_dataset(field,shape=(len(groups),len(requests)),dtype='float64',fillvalue=np.nan,
                             chunks=(1,len(requests)),compression='gzip',compression_opts=1,shuffle=True)
        for item in manifest:
            path=Path(item['regional_path'])
            if sha(path)!=item['regional_sha256']:raise ValueError('Regional values differ from completed receipt')
            frame=pd.read_parquet(path)
            expected=requests[requests.sid.eq(item['sid'])]
            if len(frame)!=len(groups)*len(expected) or set(frame.request_id)!=set(expected.request_id):raise ValueError('Missing or extra diagnostic rows')
            if frame.duplicated(KEYS+['request_id']).any():raise ValueError('Repeated physical diagnostic')
            merged=frame.merge(lookup,on=KEYS,how='left',suffixes=('_source','_expected'),validate='many_to_one')
            if merged.row.isna().any() or not merged.units_source.equals(merged.units_expected):raise ValueError('Unknown identity or changed units')
            columns=expected.index.to_numpy();start=int(columns.min());end=int(columns.max())+1
            if not np.array_equal(columns,np.arange(start,end)):raise ValueError('SID request columns are not contiguous')
            for field in ['value','valid_area_fraction']:
                arr=merged.pivot(index='row',columns='request_id',values=field).reindex(index=np.arange(len(groups)),columns=expected.request_id).to_numpy(float)
                h[field][:,start:end]=arr
        h.attrs['input_identity']=digest
    os.replace(temp,target)
    save(receipt,{'input_identity':digest,'matrix_sha256':sha(target),'groups':len(groups),'requests':len(requests),
                  'all_other_support_columns':'Preserved at exact source parquet paths in input_manifest.json; not duplicated or dropped'})


def draw_membership(result):
    records=[];events=result['event_estimates']
    for estimand,g in events.groupby('estimand',sort=True):
        valid=g[np.isfinite(g.value)];sids=valid.groupby('sid').value.mean().sort_index()
        first=valid[valid.return_order.eq(1)].set_index('sid').value.sort_index();common=sids.index.intersection(first.index).sort_values()
        for suffix,ids in [('all_events',sids.index),('first_event',first.index),('all_common',common),('difference',common)]:
            key=estimand+'|'+suffix
            if key in result['bootstrap_draws']:
                records.extend({'draw_key':key,'sid_order':i,'sid':sid} for i,sid in enumerate(ids))
    for lag,g in result['time_sid'].groupby('relative_hour',sort=True) if len(result['time_sid']) else []:
        key='time|'+str(lag)
        if key not in result['bootstrap_draws']:
            # Pandas can preserve integer vs float lag scalar dtypes differently.
            candidates=[k for k in result['bootstrap_draws'] if k.startswith('time|') and float(k.split('|')[1])==float(lag)]
            if len(candidates)!=1:raise ValueError('Time draw membership key mismatch')
            key=candidates[0]
        records.extend({'draw_key':key,'sid_order':i,'sid':sid} for i,sid in enumerate(sorted(g.sid)))
    return pd.DataFrame(records,columns=['draw_key','sid_order','sid'])


_STATE={}
def initialize_worker(add,matrix):
    global _STATE
    add=Path(add);_STATE=metadata(add)
    _STATE['contract']=load(add/'config/rise_analysis_contract.json')
    _STATE['matrix']=h5py.File(matrix,'r');_STATE['all_nan_results']={}
    _STATE['row_map']={s:i for i,s in enumerate(_STATE['matrix']['group_id'].asstr()[:])}


def process_batch(batch,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    job_identity=hashlib.sha256(json.dumps(batch,sort_keys=True).encode()).hexdigest()
    completed=destination/'completed.json'
    if completed.exists():
        r=load(completed)
        if r['job_identity']==job_identity and all(sha(destination/n)==v for n,v in r['outputs_sha256'].items()):
            return {'batch':str(destination),'reused':True,'jobs':len(batch)}
        raise ValueError('Completed statistics batch changed')
    tables={name:[] for name in TABLES+['bootstrap_membership','request_support']}
    matrix=_STATE['matrix'];requests=_STATE['requests'];config=_STATE['contract'];clock=time.perf_counter()
    with h5py.File(destination/'bootstrap_draws.partial.h5','w') as draws:
        draws.attrs['bootstrap_B']=config['statistics']['bootstrap_B'];draws.attrs['seed']=config['statistics']['seed']
        draws.attrs['meaning']='SID bootstrap estimates, not cyclone-level physical observations'
        for job in batch:
            arrays=[matrix['value'][_STATE['row_map'][gid],:] for gid in job['source_group_ids']]
            joint=np.logical_and.reduce([np.isfinite(x) for x in arrays])
            if job['support_mode']=='paired_configuration_common':
                if job['comparison_role']=='reference':vector=arrays[0].copy()
                elif job['comparison_role']=='alternative':vector=arrays[1].copy()
                else:vector=arrays[1]-arrays[0]
            else:vector=arrays[0].copy()
            vector[~joint]=np.nan
            values=pd.DataFrame({'request_id':requests.request_id,'value':vector})
            cache_key=(job['units'],bool(job['integrate_rate']))
            if not np.isfinite(vector).any() and cache_key in _STATE['all_nan_results']:
                result=_STATE['all_nan_results'][cache_key]
            else:
                result=Rise_Event_Statistics(values,_STATE['roles'],_STATE['pairs'],_STATE['events'],_STATE['intervals'],
                    bootstrap_B=config['statistics']['bootstrap_B'],seed=config['statistics']['seed'],
                    pre_required_count=config['time_composite']['pre_required_count'],unit=job['units'],integrate_rate=bool(job['integrate_rate']))
                if not np.isfinite(vector).any():_STATE['all_nan_results'][cache_key]=result
            info={k:job[k] for k in ['job_id','group_id','metric','level','window','region','scheme','support_mode','comparison_role','comparison_id']}
            for table in TABLES:
                table_value=result[table].reindex(columns=TABLE_COLUMNS[table]).copy()
                for key,value in info.items():table_value[key]=value
                tables[table].append(table_value)
            membership=draw_membership(result);membership['job_id']=job['job_id'];tables['bootstrap_membership'].append(membership)
            support=pd.DataFrame({'job_id':job['job_id'],'request_id':requests.request_id,'joint_valid':joint,
                                  'first_source_valid':np.isfinite(arrays[0]),'source_count':len(arrays)})
            tables['request_support'].append(support)
            group=draws.create_group(job['job_id']);group.attrs['job_json']=json.dumps(job)
            for key,estimate in result['bootstrap_draws'].items():
                group.create_dataset(key,data=estimate,compression='gzip',compression_opts=1,shuffle=True)
    os.replace(destination/'bootstrap_draws.partial.h5',destination/'bootstrap_draws.h5')
    for name,parts in tables.items():pd.concat(parts,ignore_index=True).to_parquet(destination/(name+'.parquet'),index=False)
    output_names=['bootstrap_draws.h5']+[name+'.parquet' for name in tables]
    output_hashes={name:sha(destination/name) for name in output_names}
    save(completed,{'status':'COMPLETE_RISE_STATISTICS_BATCH','job_identity':job_identity,'job_ids':[j['job_id'] for j in batch],
                    'outputs_sha256':output_hashes,'seconds':time.perf_counter()-clock,
                    'output_identities':{name:{'sha256':output_hashes[name],'bytes':(destination/name).stat().st_size,
                        'mtime_ns':(destination/name).stat().st_mtime_ns} for name in output_names},
                    'bootstrap_indices':'Regenerate from saved seed/B and sorted SID membership; identical fixed function bank rule',
                    'no_P_or_q_created':True})
    return {'batch':str(destination),'reused':False,'jobs':len(batch),'seconds':time.perf_counter()-clock}


def run(workers=2,batch_size=16):
    contract_path=ADD/'config/mwt_rise_statistics_execution.json';contract=load(contract_path)
    for name,digest in contract['files_sha256'].items():
        if sha(WORK/name)!=digest:raise ValueError('Frozen rise statistics code/input changed: '+name)
    manifest,meta=source_gate();groups,jobs=schema()
    input_identity={'execution_contract_sha256':sha(contract_path),'SID_manifest':manifest}
    run_id=hashlib.sha256(json.dumps(input_identity,sort_keys=True).encode()).hexdigest()[:20]
    destination=ADD/'mwt_rise_statistics'/('run_'+run_id);destination.mkdir(parents=True,exist_ok=True)
    if (destination/'input_identity.json').exists() and load(destination/'input_identity.json')!=input_identity:
        raise ValueError('Statistics restart input mismatch')
    save(destination/'input_identity.json',input_identity);save(destination/'input_manifest.json',manifest)
    groups.to_parquet(destination/'diagnostic_groups.parquet',index=False);save(destination/'job_manifest.json',jobs)
    meta['requests'].to_parquet(destination/'request_identity.parquet',index=False)
    build_matrix(destination,groups,manifest,meta['requests'])
    batches=[jobs[i:i+batch_size] for i in range(0,len(jobs),batch_size)];parts=destination/'parts';parts.mkdir(exist_ok=True)
    outcomes=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,
            initargs=(str(ADD),str(destination/'request_values.h5'))) as pool:
        futures={pool.submit(process_batch,b,parts/f'{i:04d}'):i for i,b in enumerate(batches)}
        for future in concurrent.futures.as_completed(futures):
            result=future.result();outcomes.append(result)
            save(destination/'progress.json',{'completed_batches':len(outcomes),'total_batches':len(batches),'last':result,
                 'workers':workers,'BLAS_threads_per_worker':1,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    ordered=[parts/f'{i:04d}' for i in range(len(batches))]
    for table in ['summary','time_summary']:
        pd.concat([pd.read_parquet(p/(table+'.parquet')) for p in ordered],ignore_index=True).to_parquet(destination/(table+'.parquet'),index=False)
    receipts=[]
    for p,batch in zip(ordered,batches):
        receipt=load(p/'completed.json')
        if receipt['job_ids']!=[j['job_id'] for j in batch]:raise ValueError('Batch jobs differ from fixed order')
        for name,item in receipt['output_identities'].items():
            st=(p/name).stat()
            if st.st_size!=item['bytes'] or st.st_mtime_ns!=item['mtime_ns'] or receipt['outputs_sha256'][name]!=item['sha256']:
                raise ValueError('Batch output changed after its write/hash verification')
        receipts.append({'path':str(p),'completed_sha256':sha(p/'completed.json'),
            'hash_verification':'Worker hashed complete output; finalization rechecks receipt identity, byte size and mtime'})
    for name,digest in contract['files_sha256'].items():
        if sha(WORK/name)!=digest:raise ValueError('Source changed while statistics ran')
    final_outputs=['summary.parquet','time_summary.parquet','diagnostic_groups.parquet','job_manifest.json',
                   'request_identity.parquet','input_manifest.json','input_identity.json','request_values_completed.json']
    done={'status':'COMPLETE_FROZEN_MWT_RISE_STATISTICS','candidate_SID':123,'candidate_events':133,'requests':2646,
          'native_groups':3834,'total_jobs':len(jobs),'batch_receipts':receipts,
          'input_identity_sha256':sha(destination/'input_identity.json'),'execution_contract_sha256':sha(contract_path),
          'primary_support':'Six predeclared quantities at 700 hPa/middle window/0-300 km, boundary inputs at 500-850 hPa',
          'common_mask_scope':'Regional-QC-passed request intersection; not a new grid-cell common-support recomputation',
          'same_SID_differences':'First-event and configuration direct differences calculated before group averaging',
          'time_integrals':'Only W kg-1 and W m-2 rates; KE/APE stores retain mean and observed endpoint difference',
          'outputs_sha256':{name:sha(destination/name) for name in final_outputs},
          'new_P_q':False,'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(destination/'completed.json',done)
    save(ADD/'mwt_rise_statistics/latest_completed.json',{'path':str(destination),'completed_receipt_sha256':sha(destination/'completed.json')})
    print(json.dumps({'status':done['status'],'path':str(destination),'jobs':len(jobs)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--schema-only',action='store_true');parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.schema_only:
        g,j=schema();print(json.dumps({'native_groups':len(g),'jobs':len(j),'effects_read':False}))
    elif args.check_only:
        source_gate();print(json.dumps({'source_gate':'complete_and_validated','statistics_started':False}))
    else:run()
