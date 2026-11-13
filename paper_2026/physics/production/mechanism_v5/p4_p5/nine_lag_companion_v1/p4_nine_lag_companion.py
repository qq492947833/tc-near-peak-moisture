"""Fixed P4 nine-lag descriptive companion; no models, P/q or raw ERA5 reads.
Production requires complete per-process P4 input. PI clearance is process-specific.
"""
from pathlib import Path
import argparse,csv,json,hashlib,datetime,sys,os,collections,math
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from v5_statistics import bootstrap_weights
NAMESPACE='P4_nine_lag_descriptive_v1'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def hashobj(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def readjson(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def readrows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(z) for z in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def writejson(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def writerows(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r));tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
 with tmp.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(tmp,p)
def explicit(x):
 if isinstance(x,(bool,np.bool_)):return bool(x)
 s=str(x).strip().lower()
 if s in ('true','1'):return True
 if s in ('false','0'):return False
 raise ValueError('unknown Boolean '+repr(x))
def number(x):
 try:v=float(x);return v if np.isfinite(v) else np.nan
 except (ValueError,TypeError):return np.nan
def utc(x):return datetime.datetime.fromisoformat(str(x).replace('Z','+00:00')).astimezone(datetime.timezone.utc).isoformat()
def process_gate(planned,available,pi_cleared):
 allowed=[];pending={}
 for process in planned:
  if process not in available:pending[process]='PENDING_COMPLETE_VERIFIED_INPUT'
  elif process=='PI_headroom' and not pi_cleared:pending[process]='PENDING_PHYSICAL_CLEARANCE'
  else:allowed.append(process)
 return dict(planned=list(planned),allowed=allowed,pending=pending)
def prepare_points(source,environment,cfg):
 """Pure table adapter; absent input distinct from an explicit valid QC exclusion."""
 env={};seen=set();points=[]
 for e in environment:
  if not e['request_id'].startswith('P4|'):continue
  if e['request_id'] in env:raise ValueError('duplicate environment request')
  env[e['request_id']]=e
 for s in source:
  key=(s['sid'],s['boundary'],int(s['lag_h']))
  if key in seen:raise ValueError('duplicate SID/boundary/lag')
  seen.add(key);rid='P4|'+s['request_id'];e=env.get(rid)
  exists=explicit(s['actual_track_point']);stage=str(s.get('frozen_stage','')) if exists else 'MISSING'
  if e is not None:
   if e['sid']!=s['sid'] or utc(e['UTC'])!=utc(s['UTC']):raise ValueError('SID/UTC binding conflict')
   if exists and str(e.get('stage',''))!=stage:raise ValueError('frozen stage binding conflict')
  for process,meta in cfg['processes'].items():
   col=meta['column'];missing=e is None or any(k not in e for k in (col,col+'__support_pass',col+'__reason'))
   passed=False if missing else explicit(e[col+'__support_pass']);value=np.nan if missing else number(e[col])
   if not exists and (passed or np.isfinite(value)):raise ValueError('missing actual location cannot have process support/value')
   valid=bool(exists and passed and np.isfinite(value))
   reason='MISSING_PROCESS_INPUT' if missing else str(e[col+'__reason'])
   if passed and not np.isfinite(value):reason='QC_TRUE_BUT_NONFINITE_VALUE'
   points.append(dict(request_id=rid,sid=s['sid'],boundary=s['boundary'],lag_h=int(s['lag_h']),UTC=utc(s['UTC']),process=process,column=col,unit=meta['mean_unit'],value=value if valid else np.nan,source_reported_value=value,process_valid=valid,input_missing=missing,actual_track_point=exists,frozen_stage=stage,legal_forward6h=explicit(s['legal_forward6h']),source_support_reason=s.get('source_support_reason',''),QC_reason=reason,source_QC_pass=passed,valid_fraction=np.nan if e is None else number(e.get(col+'__valid_fraction')),source_complete_json='' if e is None else e.get('source_complete_json',''),source_complete_sha256='' if e is None else e.get('source_complete_sha256',''),PI_source_complete_json='' if e is None else e.get('PI_source_complete_json',''),PI_source_complete_sha256='' if e is None else e.get('PI_source_complete_sha256','')))
 return points

def build_draw_plan(sid_pool,draws=2000,seed=510082026,bank=None):
 """One fixed original SID pool shared across every boundary, lag and process."""
 sid=np.array(sorted(set(str(x) for x in sid_pool)),str)
 if len(sid)!=len(sid_pool) or not len(sid):raise ValueError('nonempty unique original SID pool required')
 binding=dict(namespace=NAMESPACE,policy='full_original_SID_trajectory_pool_v1',seed=int(seed),draws=int(draws),ordered_sid=sid.tolist(),statistics_sha256=sha(ROOT/'src/v5_statistics.py'))
 key=hashobj(binding);path=None if bank is None else Path(bank)/('SIDdraw_'+key[:20]+'.npz')
 if path is not None and path.exists():
  info=readjson(path.with_suffix('.json'))
  if info['binding']!=binding or sha(path)!=info['sha256']:raise ValueError('draw bank identity conflict')
  with np.load(path,allow_pickle=False) as z:indices=z['sampled_sid_indices'].copy()
  if indices.shape!=(draws,len(sid)) or np.any(indices<0) or np.any(indices>=len(sid)):raise ValueError('invalid shared draw index')
  weights=np.zeros((draws,len(sid)),dtype=np.int32);np.add.at(weights,(np.arange(draws)[:,None],indices),1);action='REUSED_COMPATIBLE'
 else:
  indices,weights=bootstrap_weights(len(sid),draws,seed);indices=indices.astype(np.int32);action='NEW_IN_NAMESPACE'
  if path is not None:
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('wb') as f:np.savez_compressed(f,sampled_sid_indices=indices,ordered_sid=sid)
   writejson(path.with_suffix('.json'),dict(binding=binding,sha256=sha(path),created_utc=stamp(),origin='new single full-SID trajectory index; not recovered historical or endpoint draws'))
 return dict(sid=sid,indices=indices,weights=weights,binding=binding,binding_sha256=key,bank_path='' if path is None else str(path),action=action)

def summarize_group(points,draws=2000,seed=510082026,bank=None,draw_plan=None):
 """Per-cell mean on own valid support, using the single original SID trajectory draw."""
 if len({p['sid'] for p in points})!=len(points):raise ValueError('duplicate SID within mean group')
 plan=build_draw_plan([p['sid'] for p in points],draws,seed,bank) if draw_plan is None else draw_plan
 sid=plan['sid'];indices=plan['indices'];weights=plan['weights'];draws=len(indices);seed=plan['binding']['seed'];lookup={str(s):i for i,s in enumerate(sid)}
 values=np.full(len(sid),np.nan);valid=[]
 for p in points:
  if p['sid'] not in lookup:raise ValueError('SID outside original draw pool')
  if p['process_valid'] and np.isfinite(p['value']):values[lookup[p['sid']]]=p['value'];valid.append(p)
 finite=np.isfinite(values);n=int(finite.sum());count=weights@finite.astype(np.int32);distinct=(weights>0).astype(np.int32)@finite.astype(np.int32)
 numerator=weights@np.where(finite,values,0.);d=np.divide(numerator,count,out=np.full(draws,np.nan),where=count>0)
 # Zero is used solely for masked summation, never substituted into values or scientific source rows.
 ok=np.isfinite(d);ci=np.quantile(d[ok],[.025,.975],method='linear') if n>=2 and ok.any() else [np.nan,np.nan]
 summary=dict(estimate=float(values[finite].mean()) if n else np.nan,ci_low=float(ci[0]),ci_high=float(ci[1]),n_sid=n,n_original_SID_pool=len(sid),n_requested_sid=len(points),n_actual_position_sid=sum(p['actual_track_point'] for p in points),requested_draws=int(draws),attempted_draws=int(draws),valid_draws=int(ok.sum()),failed_draws=int((~ok).sum()),ci_method='pointwise descriptive percentile from shared full-original-SID trajectory draws; not simultaneous; no P/q',draw_namespace=NAMESPACE,seed=int(seed),draw_index_action=plan['action'],draw_bank=plan['bank_path'],draw_binding_sha256=plan['binding_sha256'],status='DESCRIPTIVE_MEAN_CI_COMPUTED' if n>=2 else 'MEAN_ONLY_LT2_SID' if n else 'NO_VALID_SID',empty_draw_reason='no process-valid sampled SID; no top-up')
 return dict(summary=summary,sid=sid,valid_sid=sid[finite],values=values,sampled_sid_indices=indices,draws=d,per_draw_valid_n=count,per_draw_distinct_valid_n=distinct,stage_counts=dict(collections.Counter(p['frozen_stage'] for p in valid)))

def load_verified_input(root,ready_folder,clearance=None):
 root=Path(root);folder=Path(ready_folder);cfg=readjson(HERE/'frozen_config.json');status_path=folder/'coverage_status.json'
 report=dict(created_utc=stamp(),planned_processes=list(cfg['processes']),input_status='WAITING',ready_table=None,physical_clearance=False,formal_statistics='NOT_RUN',process_gate=process_gate(list(cfg['processes']),set(),False))
 if not status_path.exists():return report,None,None
 status=readjson(status_path);report['coverage_status']=str(status_path);report['coverage_status_sha256']=sha(status_path)
 if status.get('status') not in ('ASSEMBLED_INPUTS_PENDING_SCIENCE','P4_PROCESS_INPUTS_VERIFIED'):
  report['reason']='ASSEMBLED_TABLE_NOT_AVAILABLE';return report,None,None
 table=Path(status['ready_table']);expected_requests=root/'data/environment_requests_v5.csv';source=root/'p4_p5/P4_boundary_requests.csv'
 if sha(table)!=status['ready_table_sha256'] or sha(expected_requests)!=status['source_request_table_sha256']:raise ValueError('source input hash conflict')
 if status.get('analysis_module_sha256') and sha(root/'src/p4_p5_analysis_v5.py')!=status['analysis_module_sha256']:raise ValueError('input assembly implementation changed')
 sources=readrows(source);environment=readrows(table);expected={'P4|'+x['request_id'] for x in sources};have=[x['request_id'] for x in environment if x['request_id'].startswith('P4|')]
 if len(sources)!=42624 or len({x['sid'] for x in sources})!=2368:raise ValueError('frozen P4 source population conflict')
 if len(have)!=len(expected) or set(have)!=expected:
  report.update(reason='NOT_ALL_P4_REQUESTS_TERMINAL',missing_requests=len(expected-set(have)));return report,None,None
 for r in environment:
  if r['request_id'] not in expected:continue
  if r.get('extraction_terminal_status') not in ('COMPLETE','REUSED_VERIFIED_CHECKPOINT','EXCLUDED_MISSING_ACTUAL_POSITION'):raise ValueError('nonterminal row inside purported assembled P4 input')
  if r['extraction_terminal_status']!='EXCLUDED_MISSING_ACTUAL_POSITION' and not (r.get('source_complete_json') and r.get('source_complete_sha256')):raise ValueError('missing scientific-source binding')
 points=prepare_points(sources,environment,cfg);available=[]
 for p in cfg['processes']:
  z=[r for r in points if r['process']==p]
  if not any(r['input_missing'] for r in z):available.append(p)
 pi_clear=False
 if clearance:
  c=readjson(clearance)
  # Clearance must explicitly bind this table; a raw legacy pi_valid flag is not clearance.
  pi_clear=(c.get('status')=='PI_PHYSICAL_INPUTS_VERIFIED' and c.get('ready_table_sha256')==sha(table) and c.get('source_request_table_sha256')==sha(expected_requests) and bool(c.get('verification_evidence')))
  report['physical_clearance_receipt']=str(clearance);report['physical_clearance_receipt_sha256']=sha(clearance)
 gate=process_gate(list(cfg['processes']),set(available),pi_clear)
 report.update(input_status='COMPLETE_TABLE_PER_PROCESS_GATE',ready_table=str(table),ready_table_sha256=sha(table),source_table=str(source),source_table_sha256=sha(source),source_spec_sha256=sha(root/'p4_p5/source_spec.json'),physical_clearance=pi_clear,process_gate=gate)
 return report,points,cfg

def run_descriptive(report,points,cfg,out,selected_processes=None,draw_bank=None):
 out=Path(out)
 if out.exists() and any(out.iterdir()):raise ValueError('use a new output directory; existing artifacts are never replaced')
 out.mkdir(parents=True,exist_ok=True);started=stamp();selected=list(cfg['processes']) if selected_processes is None else selected_processes
 if not set(selected)<=set(cfg['processes']):raise ValueError('unknown process selection')
 allowed=[p for p in report['process_gate']['allowed'] if p in selected];pending=dict(report['process_gate']['pending'])
 for p in cfg['processes']:
  if p not in selected and p not in pending:pending[p]='NOT_EXECUTED_THIS_RUN'
 bank=out/'draw_indices' if draw_bank is None else Path(draw_bank);pool=sorted({p['sid'] for p in points});plan=build_draw_plan(pool,cfg['requested_draws'],cfg['seed'],bank)
 if len(pool)!=2368:raise ValueError('production draw pool must retain all frozen2368SID')
 summaries=[];stages=[];sidrows=[];draw_table={}
 for boundary in cfg['boundaries']:
  for lag in cfg['lags_h']:
   for process in cfg['processes']:
    z=[r for r in points if r['boundary']==boundary and r['lag_h']==lag and r['process']==process]
    base=dict(boundary=boundary,lag_h=lag,process=process,unit=cfg['processes'][process]['mean_unit'])
    if process not in allowed:
     summaries.append(dict(base,status=pending[process],estimate=np.nan,ci_low=np.nan,ci_high=np.nan,n_sid=None,n_requested_sid=len(z),requested_draws=cfg['requested_draws'],attempted_draws=0,no_P_or_q=True));continue
    a=summarize_group(z,draw_plan=plan)
    summaries.append(dict(base,**a['summary'],no_P_or_q=True));sidrows.extend(z)
    drawkey=boundary+'__lag'+str(lag)+'__'+process;draw_table[drawkey]=a['draws'];draw_table[drawkey+'__valid_multiplicity']=a['per_draw_valid_n'];draw_table[drawkey+'__distinct_valid_SID']=a['per_draw_distinct_valid_n']
    for stage in sorted(set(r['frozen_stage'] for r in z)):
     zz=[r for r in z if r['frozen_stage']==stage];stages.append(dict(base,frozen_stage=stage,n_requested=len(zz),n_actual_position=sum(r['actual_track_point'] for r in zz),n_process_valid=sum(r['process_valid'] for r in zz),n_legal_forward6h=sum(r['legal_forward6h'] for r in zz)))
 writerows(out/'P4_nine_lag_mean_CI_n.csv',summaries);writerows(out/'P4_nine_lag_SID_source.csv',sidrows);writerows(out/'P4_nine_lag_actual_stage_composition.csv',stages)
 with (out/'pointwise_mean_draws.npz').open('wb') as f:np.savez_compressed(f,**draw_table)
 writejson(out/'input_gate_receipt.json',report);writejson(out/'execution_receipt.json',dict(execution='DESCRIPTIVE_SUMMARIES_COMPUTED_WITH_DISCLOSED_PENDING' if pending else 'DESCRIPTIVE_SUMMARIES_COMPUTED',verification='PENDING_INDEPENDENT_REAL_RESULT_AND_FIGURE_REVIEW',scientific_conclusion='NOT_REVIEWED',started_utc=started,finished_utc=stamp(),process_gate=report['process_gate'],executed_processes=allowed,pending_this_run=pending,shared_draw_binding=plan['binding'],shared_draw_bank=plan['bank_path'],scope='No model fit; new pointwise descriptive SID resampling only; original endpoint statistics and P/q unchanged',config_sha256=sha(HERE/'frozen_config.json'),code_sha256=sha(__file__),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
 return summaries

def main():
 p=argparse.ArgumentParser();p.add_argument('--ready-folder',required=True);p.add_argument('--output',required=True);p.add_argument('--physical-clearance');p.add_argument('--process',action='append');p.add_argument('--draw-bank');p.add_argument('--run',action='store_true');args=p.parse_args()
 report,points,cfg=load_verified_input(ROOT,args.ready_folder,args.physical_clearance)
 if not args.run:
  writejson(args.output,report);print(json.dumps(clean(report),ensure_ascii=False));return 0
 if points is None or not report.get('process_gate',{}).get('allowed'):
  writejson(Path(args.output)/'BLOCKED_INPUT_GATE.json',report);print(json.dumps(clean(report)));return 42
 run_descriptive(report,points,cfg,args.output,args.process,args.draw_bank);return 0
if __name__=='__main__':sys.exit(main())
