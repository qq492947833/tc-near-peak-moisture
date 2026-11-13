"""Reproduce the recorded environment analysis from the supplied derived inputs."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse,datetime,hashlib,json,platform,sys,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).absolute().parent
sys.path.insert(0,str(ROOT/'src'))
from Rise_Event_Statistics import Rise_Event_Statistics

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(output):
    manifest=json.loads((ROOT/'environment_manifest.json').read_text())
    for item in manifest['files']:
        if sha(ROOT/item['relative_path'])!=item['sha256']:
            raise ValueError('Changed package input: '+item['relative_path'])
    target=Path(output)
    if not target.is_absolute():target=ROOT/target
    if target.exists() and any(target.iterdir()):raise ValueError('Use an empty output directory; existing results remain protected')
    target.mkdir(parents=True,exist_ok=True)
    point=pd.read_parquet(ROOT/'inputs/rise_environment_points.parquet')
    roles=pd.read_parquet(ROOT/'inputs/rise_point_roles.parquet')
    pairs=pd.read_parquet(ROOT/'inputs/rise_comparable_state_pairs.parquet')
    events=pd.read_parquet(ROOT/'inputs/return_events.parquet')
    intervals=pd.read_parquet(ROOT/'inputs/legal_intervals.parquet')
    names=['summary','event_estimates','sid_estimates','coverage','matched_bins','time_summary','time_sid']
    parts={name:[] for name in names}
    rates={'fixed_tendency','motion_term','moving_tendency','horizontal_advection',
           'horizontal_advection_fixed_frame','vertical_advection','residual'}
    clock=time.perf_counter();groups=[]
    for (metric,radial,unit),data in point.groupby(['metric','radial','units'],sort=True):
        result=Rise_Event_Statistics(data,roles,pairs,events,intervals,unit=unit,integrate_rate=metric in rates,
                                    bootstrap_B=manifest['bootstrap_B'],seed=manifest['seed'])
        for name in names:
            frame=result[name].copy();frame['metric']=metric;frame['radial']=radial
            if 'units' not in frame:frame['units']=unit
            parts[name].append(frame)
        groups.append({'metric':metric,'radial':radial,'units':unit})
        print(json.dumps({'completed_group':len(groups),'metric':metric,'radial':radial}),flush=True)
    checks=[]
    for name,frames in parts.items():
        actual=pd.concat(frames,ignore_index=True)
        actual.to_parquet(target/(name+'.parquet'),index=False)
        expected=pd.read_parquet(ROOT/'reference_results/environment'/(name+'.parquet'))
        if list(actual.columns)!=list(expected.columns) or actual.shape!=expected.shape:
            raise ValueError('Table shape/columns changed: '+name)
        maximum=0.
        for col in actual:
            a=actual[col];b=expected[col]
            if not np.array_equal(a.isna(),b.isna()):raise ValueError('NaN support changed: '+name+'/'+col)
            valid=~a.isna()
            if pd.api.types.is_numeric_dtype(a.dtype) and pd.api.types.is_numeric_dtype(b.dtype):
                delta=np.abs(a[valid].to_numpy(float)-b[valid].to_numpy(float))
                diff=float(delta.max()) if len(delta) else 0.
                tolerance=512*np.finfo(float).eps*np.maximum(1,np.abs(b[valid].to_numpy(float)))
                if not np.all(delta<=tolerance):raise ValueError('Numeric difference outside precision: '+name+'/'+col)
                maximum=max(maximum,diff)
            elif not np.array_equal(a[valid].astype(str),b[valid].astype(str)):
                raise ValueError('Identity/label mismatch: '+name+'/'+col)
        checks.append({'table':name,'rows':len(actual),'max_absolute_difference':maximum,
                       'NaN_and_identity_support':'exact','sha256':sha(target/(name+'.parquet'))})
    if len(groups)!=23:raise ValueError('Diagnostic count changed')
    receipt={'status':'COMPLETE_DERIVED_ENVIRONMENT_REPRODUCTION','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'elapsed_seconds':time.perf_counter()-clock,'scope':'23 processed-input environment groups, 2000 original-seed SID draws, seven result tables. No ERA5 or energy recomputation.',
        'groups':groups,'tables':checks,'manifest_sha256':sha(ROOT/'environment_manifest.json'),
        'versions':{'Python':platform.python_version(),'NumPy':np.__version__,'pandas':pd.__version__},
        'max_absolute_difference':max(c['max_absolute_difference'] for c in checks)}
    (target/'completed.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/environment')
    args=parser.parse_args();main(args.output)
