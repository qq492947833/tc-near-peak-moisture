"""Replay a fixed subset from processed values with the original functions."""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
from pathlib import Path
import argparse,datetime,hashlib,json,platform,sys,time
import h5py
import numpy as np
import pandas as pd
import scipy
import pyarrow

ROOT=Path(__file__).absolute().parent
sys.path.insert(0,str(ROOT/'src'))
from mwt_paired_statistics import mwt_paired_statistics
from mwt_motion_intensity_sensitivity import mwt_motion_intensity_sensitivity
from Rise_Event_Statistics import Rise_Event_Statistics
from run_mwt_statistics import coverage_tables,bootstrap_identity_map,save_bootstrap
from mwt_rise_statistics_driver import TABLES,TABLE_COLUMNS,draw_membership


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=True)+'\n',encoding='utf8')


def compare_array(actual,expected,label,rtol,atol,exact=False):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape:raise ValueError('Shape changed: '+label)
    if not np.array_equal(pd.isna(a),pd.isna(b)):raise ValueError('Missing support changed: '+label)
    mask=~pd.isna(b)
    a=a[mask];b=b[mask]
    if exact:
        if not np.array_equal(a,b):raise ValueError('Exact identity/count changed: '+label)
        return 0.
    a=a.astype(float);b=b.astype(float)
    if not np.array_equal(np.isfinite(a),np.isfinite(b)):raise ValueError('Finite support changed: '+label)
    good=np.isfinite(a)
    if not np.array_equal(a[~good],b[~good]):raise ValueError('Infinity sign changed: '+label)
    delta=np.abs(a[good]-b[good]);maximum=float(delta.max()) if delta.size else 0.
    if not np.all(delta<=atol+rtol*np.abs(b[good])):raise ValueError('Numeric difference outside fixed precision: '+label)
    return maximum


def compare_frame(actual,expected,label,config):
    if set(actual.columns)!=set(expected.columns):raise ValueError('Column identity changed: '+label)
    actual=actual.reindex(columns=expected.columns)
    if actual.shape!=expected.shape:raise ValueError('Row/column count changed: '+label)
    order=[k for k in ['job_id','scheme','window','level','region','metric','summary','sensitivity',
            'estimand','sample','sid','event_id','pair_id','wind_bin','lag_h','relative_hour','draw','draw_key','sid_order','request_id'] if k in expected]
    if order:
        actual=actual.sort_values(order,kind='stable',na_position='last').reset_index(drop=True)
        expected=expected.sort_values(order,kind='stable',na_position='last').reset_index(drop=True)
    else:
        actual=actual.reset_index(drop=True);expected=expected.reset_index(drop=True)
    checks={}
    exact_names=set(order)|{'n_sid','n_events','n_offsets','n_pre_valid','n_joint_bins','successful','failed','attempts',
        'n_events_within_rise','n_sid_within_rise','source_count','n_common_offsets','candidate_sid','paired_valid_sid',
        'temporary_valid_sid','final_valid_sid','eligible_paired_sid','excluded_sid','n_temporary_offsets','n_final_offsets'}
    for col in expected:
        a=actual[col];b=expected[col]
        numeric=pd.api.types.is_numeric_dtype(a.dtype) and pd.api.types.is_numeric_dtype(b.dtype)
        exact=col in exact_names or pd.api.types.is_bool_dtype(b.dtype) or not numeric
        checks[col]=compare_array(a.to_numpy(),b.to_numpy(),label+'/'+col,
            config['comparison']['rtol'],config['comparison']['atol'],exact)
    return {'table':label,'rows':len(actual),'columns':len(actual.columns),'maximum_absolute_difference':max(checks.values(),default=0.),'column_maxima':checks,'identities_counts_units_missing_support':'exact'}


def verify_package(root=ROOT):
    manifest=load(root/'manifest.json')
    if manifest.get('status')!='READY_PROCESSED_STATISTICAL_REPLAY':raise RuntimeError('Complete production statistics and assembled inputs are required')
    for item in manifest['files']:
        p=root/item['relative_path']
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Package file identity changed: '+item['relative_path'])
    return manifest


def rise_vector(arrays,job):
    joint=np.logical_and.reduce([np.isfinite(x) for x in arrays])
    if job['support_mode']=='paired_configuration_common':
        role=job['comparison_role']
        if role=='reference':vector=arrays[0].copy()
        elif role=='alternative':vector=arrays[1].copy()
        elif role=='difference':vector=arrays[1]-arrays[0]
        else:raise ValueError('Unknown configuration comparison')
    else:vector=arrays[0].copy()
    vector[~joint]=np.nan
    return vector,joint


def run_departure(target,config):
    p=ROOT/'inputs/departure';ref=ROOT/'reference_results/departure'
    rows=pd.read_parquet(p/'regional_values.parquet');events=pd.read_csv(p/'events.csv');centres=pd.read_csv(p/'centres.csv')
    if events.sid.nunique()!=117 or len(events)!=234 or set(rows.sid)!=set(events.sid):raise ValueError('Departure candidate identity mismatch')
    kw=config['departure'];cfg={'minimum_valid_area_fraction':kw['min_area'],'primary_event_window_hours':kw['pre_hours'],
        'bootstrap':{'replicates':kw['replicates'],'seed':kw['seed']}}
    result=mwt_paired_statistics(rows,events,centres,**kw)
    coverage=coverage_tables(rows,events,cfg);checks=[]
    for name,frame in result.items():
        if isinstance(frame,pd.DataFrame):
            frame.to_parquet(target/(name+'.parquet'),index=False)
            checks.append(compare_frame(frame,pd.read_parquet(ref/(name+'.parquet')),'departure/'+name,config))
    identities=bootstrap_identity_map(result,coverage,events)
    save_bootstrap(target/'paired_bootstrap.h5',result,identities,cfg)
    with h5py.File(ref/'paired_bootstrap.h5','r') as h:
        for kind,key in [('draws','bootstrap_draws'),('index_banks','bootstrap_index_banks')]:
            if set(h[kind])!=set(result[key]):raise ValueError('Bootstrap vector identity changed: '+kind)
            for name,arr in result[key].items():
                if kind=='index_banks' and json.loads(h[kind][name].attrs['sorted_SID_json'])!=identities[name]:raise ValueError('Bootstrap SID membership changed')
                maximum=compare_array(arr,h[kind][name][:],'departure/'+kind+'/'+name,config['comparison']['rtol'],config['comparison']['atol'],kind=='index_banks')
                checks.append({'array':'departure/'+kind+'/'+name,'maximum_absolute_difference':maximum})
    adjusted=mwt_motion_intensity_sensitivity(rows,centres,minimum_SID=30,**kw)
    for name,frame in adjusted.items():
        frame.to_parquet(target/('motion_intensity_'+name+'.parquet'),index=False)
        checks.append(compare_frame(frame,pd.read_parquet(ref/('motion_intensity_'+name+'.parquet')),'departure/motion_intensity_'+name,config))
    for name in ['coverage_n_by_time','coverage_pre_counts','SID_pre_support']:
        coverage[name].to_parquet(target/(name+'.parquet'),index=False)
        checks.append(compare_frame(coverage[name],pd.read_parquet(ref/(name+'.parquet')),'departure/'+name,config))
    return checks


def run_rise(target,config):
    p=ROOT/'inputs/rise';ref=ROOT/'reference_results/rise'
    roles=pd.read_parquet(p/'roles.parquet');pairs=pd.read_parquet(p/'pairs.parquet')
    events=pd.read_parquet(p/'events.parquet');intervals=pd.read_parquet(p/'intervals.parquet')
    requests=pd.read_parquet(p/'requests.parquet');jobs=load(p/'jobs.json')
    if events.sid.nunique()!=123 or len(events)!=133 or len(requests)!=2646 or len(jobs)!=48:raise ValueError('Rise scientific selection changed')
    tables={name:[] for name in TABLES+['bootstrap_membership','request_support']};checks=[]
    with h5py.File(p/'request_values.h5','r') as matrix,h5py.File(ref/'bootstrap_draws.h5','r') as expected,h5py.File(target/'bootstrap_draws.h5','w') as output:
        if list(matrix['request_id'].asstr()[:])!=requests.request_id.tolist():raise ValueError('Request order changed')
        row_map={s:i for i,s in enumerate(matrix['group_id'].asstr()[:])}
        if set(expected)!=set(j['job_id'] for j in jobs):raise ValueError('Reference job set changed')
        for job in jobs:
            arrays=[matrix['value'][row_map[g],:] for g in job['source_group_ids']]
            vector,joint=rise_vector(arrays,job)
            values=pd.DataFrame({'request_id':requests.request_id,'value':vector})
            result=Rise_Event_Statistics(values,roles,pairs,events,intervals,unit=job['units'],integrate_rate=bool(job['integrate_rate']),**config['rise'])
            info={k:job[k] for k in ['job_id','group_id','metric','level','window','region','scheme','support_mode','comparison_role','comparison_id']}
            for name in TABLES:
                frame=result[name].reindex(columns=TABLE_COLUMNS[name]).copy()
                for k,v in info.items():frame[k]=v
                tables[name].append(frame)
            membership=draw_membership(result);membership['job_id']=job['job_id'];tables['bootstrap_membership'].append(membership)
            tables['request_support'].append(pd.DataFrame({'job_id':job['job_id'],'request_id':requests.request_id,'joint_valid':joint,
                'first_source_valid':np.isfinite(arrays[0]),'source_count':len(arrays)}))
            group=output.create_group(job['job_id']);group.attrs['job_json']=json.dumps(job)
            source=expected[job['job_id']]
            if set(source)!=set(result['bootstrap_draws']):raise ValueError('Rise draw key changed')
            for key,arr in result['bootstrap_draws'].items():
                group.create_dataset(key,data=arr,compression='gzip',compression_opts=1,shuffle=True)
                maximum=compare_array(arr,source[key][:],'rise/'+job['job_id']+'/'+key,config['comparison']['rtol'],config['comparison']['atol'])
                checks.append({'array':'rise/'+job['job_id']+'/'+key,'maximum_absolute_difference':maximum})
            print(json.dumps({'completed_rise_job':job['job_id']}),flush=True)
    for name,parts in tables.items():
        frame=pd.concat(parts,ignore_index=True);frame.to_parquet(target/(name+'.parquet'),index=False)
        checks.append(compare_frame(frame,pd.read_parquet(ref/(name+'.parquet')),'rise/'+name,config))
    return checks


def main(output,check_only=False):
    manifest=verify_package()
    if check_only:
        print(json.dumps({'package_inputs_verified':True,'files':len(manifest['files']),'statistics_executed':False}));return
    target=Path(output)
    if not target.is_absolute():target=ROOT/target
    if target.exists():raise ValueError('Use a new output directory; prior results are preserved')
    target.mkdir(parents=True);(target/'departure').mkdir();(target/'rise').mkdir()
    cfg=load(ROOT/'config/replay.json');clock=time.perf_counter()
    checks=run_departure(target/'departure',cfg)+run_rise(target/'rise',cfg)
    verify_package()
    receipt={'status':'COMPLETE_SELECTED_PROCESSED_MWT_STATISTICAL_REPLAY','scope':manifest['scope'],
        'manifest_sha256':sha(ROOT/'manifest.json'),'elapsed_seconds':time.perf_counter()-clock,
        'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,
        'maximum_absolute_difference':max(c['maximum_absolute_difference'] for c in checks),
        'versions':{'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'h5py':h5py.__version__,'pyarrow':pyarrow.__version__}}
    save(target/'completed.json',receipt)
    print(json.dumps({'status':receipt['status'],'path':str(target),'max_difference':receipt['maximum_absolute_difference']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/check');parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();main(args.output,args.check_only)
