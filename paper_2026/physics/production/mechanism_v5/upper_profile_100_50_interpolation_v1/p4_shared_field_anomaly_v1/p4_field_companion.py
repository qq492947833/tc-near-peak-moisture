"""P4 fixed linear-field shared-climate companion. Default checks inputs only.
No raw NC, new draws, P/q, nonlinear climate processes, PI or original-model edits.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,hashlib,sys,os,collections,traceback,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;U=HERE.parent;ROOT=U.parent
sys.path.insert(0,str(HERE))
import input_adapter as adapter
KEYS=adapter.KEYS;KINDS=('raw_common','climate_common','anomaly_common')
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def digest(x):return hashlib.sha256(json.dumps(clean(x),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.write_'+str(os.getpid())+'.tmp');t.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def csvsave(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.write_'+str(os.getpid())+'.tmp');fields=list(dict.fromkeys(k for r in z for k in r))
 with t.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clean(z))
 os.replace(t,p)
def boolean(x):
 if x is True or str(x).lower() in ('true','1'):return True
 if x is False or str(x).lower() in ('false','0'):return False
 raise ValueError('Unknown explicit Boolean')
def number(x):
 try:return float(x)
 except (TypeError,ValueError):return np.nan
def dependencies():
 cfg=read(HERE/'frozen_config.json')
 for p,h in cfg['bound_files'].items():
  if sha(p)!=h:raise ValueError('Frozen companion dependency changed '+p)
 return cfg
def populations():
 sources=rows(ROOT/'p4_p5/P4_boundary_requests.csv');requests=rows(ROOT/'data/environment_requests_v5.csv');requests={r['request_id']:r for r in requests if r['module']=='P4'}
 if len(sources)!=42624 or len({r['sid'] for r in sources})!=2368 or len(requests)!=42624 or {'P4|'+s['request_id'] for s in sources}!=set(requests):raise ValueError('Frozen full P4 request population conflict')
 if len({(s['sid'],s['boundary'],int(s['lag_h'])) for s in sources})!=42624:raise ValueError('Repeated SID/boundary/lag')
 return sources,requests
def immutable_sources(receipts):
 result={};bindings=[]
 for p in receipts:
  r=read(p);table=Path(r['table'])
  if r.get('status')!='VERIFIED_COMPLETED_SUBSET_NOT_FINAL_COHORT' or sha(table)!=r['table_sha256']:raise ValueError('Require verified immutable source subset')
  bindings.append(dict(receipt=str(p),sha256=sha(p),table=str(table),table_sha256=r['table_sha256']))
  for z in rows(table):
   if not z['request_id'].startswith('P4|'):continue
   if z['request_id'] in result and result[z['request_id']]['derived_complete_sha256']!=z['derived_complete_sha256']:raise ValueError('Conflicting immutable source request')
   result[z['request_id']]=z
 return result,bindings
def native_identity(row,request):
 cp=Path(row['derived_complete_path'])
 if sha(cp)!=row['derived_complete_sha256']:raise ValueError('Immutable derived checkpoint changed')
 c=read(cp)
 if c.get('status')!='COMPLETE':raise ValueError('Non-complete pointer')
 ds=[x for x in c['outputs'] if x['name']=='derived.json']
 if len(ds)!=1 or sha(cp.parent/'derived.json')!=ds[0]['sha256']:raise ValueError('Native source pointer changed')
 d=read(cp.parent/'derived.json');native=Path(d['request_folder'])/'COMPLETE.json';h=c['binding']['source']['checkpoint_sha256']
 if sha(native)!=h or read(native)['binding']['request']!=request:raise ValueError('Original native request changed')
 return native,h
def input_gate(receipts):
 cfg=dependencies();sources,requests=populations();index,binding=immutable_sources(receipts);pending_native=[]
 for s in sources:
  if boolean(s['actual_track_point']) and 'P4|'+s['request_id'] not in index:pending_native.append('P4|'+s['request_id'])
 pending_climate=[]
 for tile in cfg['required_tiles']:
  cp=ROOT/'cache/climatology_1991_2020_v5_1'/tile/'complete.json'
  if not cp.exists() or read(cp).get('status')!='COMPLETE':pending_climate.append(tile)
 result=dict(status='READY_COMPLETE_INPUTS_PENDING_VALUE_VERIFICATION' if not pending_native and not pending_climate else 'WAITING_COMPLETE_NATIVE_AND_CLIMATE',original_requests=42624,positioned_requests=sum(boolean(s['actual_track_point']) for s in sources),no_actual_position=sum(not boolean(s['actual_track_point']) for s in sources),required_climate_tiles=len(cfg['required_tiles']),pending_native_count=len(pending_native),pending_climate_tiles=pending_climate,immutable_index_bindings=binding,config_sha256=sha(HERE/'frozen_config.json'),formal_statistics_run=False,new_draws=0)
 return result,sources,requests,index
def lock(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);f=(out/'companion.lock').open('a+b')
 try:
  if os.name=='nt':
   import msvcrt
   if f.seek(0,2)==0:f.write(b'0');f.flush()
   f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  return f
 except Exception:f.close();raise
def safeout(path):
 out=Path(path).resolve()
 if not out.is_relative_to(U.resolve()):raise ValueError('New output must stay in upper-profile isolated directory')
 return out
def verify_lineage(result,reader):
 if sha(result['source_complete'])!=result['source_complete_sha256']:raise ValueError('Consumed native source changed')
 folder=ROOT/'cache/native_fields'/result['geometry_id'];gm=read(folder/'geometry.json')
 if sha(folder/'geometry.json')!=result['geometry_metadata_sha256'] or gm['npz_sha256']!=result['sp_sha256'] or sha(folder/'sp.npz')!=result['sp_sha256']:raise ValueError('Consumed SP changed')
 for k,f in result['native_fields'].items():
  if sha(f['npz'])!=f['npz_sha256'] or sha(Path(f['npz']).with_suffix('.json'))!=f['metadata_sha256']:raise ValueError('Consumed actual field changed')
  for identity in result['climate_tiles'][k]:
   _,tail=identity['tile'].split('_y');i,j=map(int,tail.split('_x'));_,current=reader.load(k,i,j)
   if any(current[key]!=identity[key] for key in ('npz_sha256','complete_sha256','signature')):raise ValueError('Consumed climate tile changed')
def one(source,request,index,reader,out,cfgsha):
 rid=request['request_id'];dest=out/'r'/digest(rid)[:16];cp=dest/'COMPLETE.json';actual=boolean(source['actual_track_point']);native,h=native_identity(index[rid],request) if actual else ('','')
 binding=dict(request=request,original_boundary_source=source,configuration_sha256=cfgsha,source_complete=str(native),source_complete_sha256=h)
 if cp.exists():
  c=read(cp)
  if c.get('status')!='COMPLETE_LINEAR_FIELD_SOURCE' or c['binding']!=binding or sha(dest/'source.json')!=c['source_sha256']:raise ValueError('Completed companion source conflict')
  r=read(dest/'source.json')
  if actual:verify_lineage(r['field_source'],reader)
  return dict(r,companion_complete=str(cp),companion_complete_sha256=sha(cp)),True
 if dest.exists() and any(dest.iterdir()):
  hist=out/'incomplete_history'/(dest.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
  shutil.copytree(dest,hist);write(hist/'preservation.json',dict(reason='No COMPLETE; retain incomplete attempt before checkpoint resume',files={str(p.relative_to(hist)):sha(p) for p in hist.rglob('*') if p.is_file()}))
 base=dict(request_id=rid,sid=source['sid'],boundary=source['boundary'],lag_h=int(source['lag_h']),UTC=source['UTC'],frozen_stage=source['frozen_stage'] if actual else 'MISSING',actual_track_point=actual,legal_forward6h=boolean(source['legal_forward6h']),no_trajectory_extrapolation=True)
 if actual:
  result,_=adapter.source_anomaly(request,native,h,reader);r=dict(base,status='COMPLETE_COMMON_SUPPORT_SOURCE',field_source=result)
 else:r=dict(base,status='EXCLUDED_NO_ACTUAL_POSITION',field_source=None)
 write(dest/'source.json',r);write(cp,dict(status='COMPLETE_LINEAR_FIELD_SOURCE',binding=binding,source_sha256=sha(dest/'source.json')));return dict(r,companion_complete=str(cp),companion_complete_sha256=sha(cp)),False
def flatten(r):
 out=[]
 for field in KEYS:
  common=r['field_source']['statistics'][field] if r['field_source'] else {}
  base={k:r[k] for k in ('request_id','sid','boundary','lag_h','UTC','frozen_stage','actual_track_point','legal_forward6h','companion_complete','companion_complete_sha256')}
  z=dict(base,field=field,raw_unit='%' if field=='r700' else adapter.UNITS[field],anomaly_unit=adapter.UNITS[field],source_status=r['status'])
  for kind in ('raw_own_support',)+KINDS:
   v=common.get(kind,{})
   z[kind]=v.get('mean',np.nan);z[kind+'__valid']=v.get('support_pass',False);z[kind+'__fraction']=v.get('valid_fraction',np.nan)
  z.update(valid_area_m2=common.get('raw_common',{}).get('valid_area_m2',np.nan),original_area_m2=common.get('raw_common',{}).get('original_area_m2',np.nan),source_complete_json='' if not r['field_source'] else r['field_source']['source_complete'],source_complete_sha256='' if not r['field_source'] else r['field_source']['source_complete_sha256'])
  out.append(z)
 return out
def prepare(receipts,output,max_new=None):
 out=safeout(output);gate,sources,requests,index=input_gate(receipts);start=stamp()
 if gate['status']!='READY_COMPLETE_INPUTS_PENDING_VALUE_VERIFICATION':write(out/('WAIT_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),gate);return 42
 handle=lock(out);new=0;reused=0;reader=adapter.BoundedClimate();allrows=[]
 try:
  for s in sources:
   rid='P4|'+s['request_id'];dest=out/'r'/digest(rid)[:16]/'COMPLETE.json'
   if max_new is not None and new>=max_new and not dest.exists():break
   r,old=one(s,requests[rid],index,reader,out,gate['config_sha256']);new+=not old;reused+=old;allrows.extend(flatten(r))
   if (new+reused)%100==0:write(out/'progress.json',dict(started_utc=start,updated_utc=stamp(),new_records=new,reused_records=reused,original_requests=42624))
  complete=len(allrows)==42624*4
  receipt=dict(gate,status='COMPLETE_COMMON_SUPPORT_SOURCES_PENDING_SID_SUMMARY' if complete else 'BOUNDED_SOURCE_PASS_NOT_COMPLETE_MODULE',started_utc=start,finished_utc=stamp(),new_records=new,reused_records=reused,max_new=max_new,remaining_requests=42624-new-reused,new_draws=0,formal_statistics_run=False,exit_code=0)
  if complete:
   csvsave(out/'P4_shared_climate_SID_source.csv',allrows);receipt.update(source_table=str(out/'P4_shared_climate_SID_source.csv'),source_table_sha256=sha(out/'P4_shared_climate_SID_source.csv'))
  write(out/'source_receipt.json',receipt);return 0
 except Exception as e:
  write(out/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(started_utc=start,finished_utc=stamp(),new_records=new,reused_records=reused,error=repr(e),traceback=traceback.format_exc(),exit_code=1,dispatch_stopped=True));raise
 finally:handle.close()
def existing_plan(bank,prior_receipt,pool):
 p=Path(bank)
 if not p.exists():raise adapter.PendingInput('Original shared SID bank has not yet been executed')
 info=read(p.with_suffix('.json'));b=info['binding'];prior=read(prior_receipt)
 if sha(p)!=info['sha256'] or b.get('namespace')!='P4_nine_lag_descriptive_v1' or b.get('policy')!='full_original_SID_trajectory_pool_v1' or b.get('ordered_sid')!=pool or len(pool)!=2368 or b.get('draws')!=2000 or b.get('seed')!=510082026 or b.get('statistics_sha256')!=sha(ROOT/'src/v5_statistics.py'):raise ValueError('Original shared SID bank binding conflict')
 if Path(prior['shared_draw_bank']).resolve()!=p.resolve() or prior['shared_draw_binding']!=b or prior.get('code_sha256')!=sha(ROOT/'p4_p5/nine_lag_companion_v1/p4_nine_lag_companion.py') or prior.get('config_sha256')!=sha(ROOT/'p4_p5/nine_lag_companion_v1/frozen_config.json'):raise ValueError('Not original actually executed nine-lag bank')
 with np.load(p,allow_pickle=False) as f:idx=f['sampled_sid_indices'].copy();sid=f['ordered_sid'].copy()
 if idx.shape!=(2000,2368) or not np.array_equal(sid,np.array(pool)) or np.any(idx<0) or np.any(idx>=2368):raise ValueError('Bank payload mismatch')
 weights=np.zeros(idx.shape,dtype=np.int32);np.add.at(weights,(np.arange(len(idx))[:,None],idx),1)
 return weights,dict(path=str(p),sha256=sha(p),binding=b,prior_execution=str(prior_receipt),prior_execution_sha256=sha(prior_receipt))
def summarize_cell(records,pool,weights):
 if len({r['sid'] for r in records})!=len(records) or any(r['sid'] not in set(pool) for r in records):raise ValueError('Duplicate or out-of-pool SID')
 loc={s:i for i,s in enumerate(pool)};x=np.full((len(pool),3),np.nan)
 for r in records:
  valid=[boolean(r[k+'__valid']) and np.isfinite(number(r[k])) for k in KINDS]
  if len(set(valid))!=1:raise ValueError('Raw/climate/anomaly do not share sample support')
  if valid[0]:x[loc[r['sid']]]=[number(r[k]) for k in KINDS]
 good=np.isfinite(x).all(1)
 if good.any() and not np.allclose(x[good,2],x[good,0]-x[good,1],rtol=0,atol=1e-11):raise ValueError('PerSID raw-climate identity conflict')
 n=weights@good.astype(np.int32);distinct=(weights>0).astype(np.int32)@good.astype(np.int32);d=np.divide(weights@np.where(good[:,None],x,0),n[:,None],out=np.full((len(weights),3),np.nan),where=n[:,None]>0)
 if np.isfinite(d).any() and not np.allclose(d[:,2],d[:,0]-d[:,1],equal_nan=True,rtol=0,atol=1e-11):raise ValueError('Shared-draw raw-climate identity conflict')
 means=x[good].mean(0) if good.any() else np.full(3,np.nan);ci=np.quantile(d[n>0],[.025,.975],axis=0,method='linear') if good.sum()>=2 and (n>0).any() else np.full((2,3),np.nan)
 return dict(n=int(good.sum()),mean=means,ci=ci,draws=d,valid_multiplicity=n,distinct_valid_sid=distinct,values=x,finite=good)
def verify_prepared_records(table,requests,sources):
 reader=adapter.BoundedClimate();lookup={'P4|'+s['request_id']:s for s in sources};seen=set();byrequest=collections.defaultdict(list)
 for row in table:byrequest[row['request_id']].append(row)
 for row in table:
  rid=row['request_id']
  if rid in seen:continue
  seen.add(rid);cp=Path(row['companion_complete'])
  if sha(cp)!=row['companion_complete_sha256']:raise ValueError('Prepared companion checkpoint changed')
  c=read(cp);b=c['binding']
  if b['request']!=requests[rid] or b['original_boundary_source']!=lookup[rid] or b['configuration_sha256']!=sha(HERE/'frozen_config.json') or sha(cp.parent/'source.json')!=c['source_sha256']:raise ValueError('Prepared source record lineage changed')
  record=read(cp.parent/'source.json')
  if record['field_source']:verify_lineage(record['field_source'],reader)
  compare=flatten(dict(record,companion_complete=str(cp),companion_complete_sha256=sha(cp)))
  byfield={x['field']:x for x in compare}
  for current in byrequest[rid]:
   original=byfield[current['field']]
   for kind in ('raw_own_support',)+KINDS:
    if boolean(current[kind+'__valid'])!=original[kind+'__valid'] or not np.isclose(number(current[kind]),number(original[kind]),rtol=0,atol=0,equal_nan=True):raise ValueError('Summary table values differ from verified source')
 return len(seen)

def summarize(source_folder,bank,prior_receipt,output):
 dependencies();out=safeout(output)
 if out.exists() and any(out.iterdir()):raise ValueError('Use new summary version; preserve prior outputs')
 r=read(Path(source_folder)/'source_receipt.json')
 if r.get('status')!='COMPLETE_COMMON_SUPPORT_SOURCES_PENDING_SID_SUMMARY':raise adapter.PendingInput('All original source rows required')
 if sha(r['source_table'])!=r['source_table_sha256'] or r['config_sha256']!=sha(HERE/'frozen_config.json'):raise ValueError('Completed source identity conflict')
 sources,requests=populations();pool=sorted({s['sid'] for s in sources});weights,identity=existing_plan(bank,prior_receipt,pool);table=rows(r['source_table'])
 expected={(s['sid'],s['boundary'],int(s['lag_h']),k) for s in sources for k in KEYS}
 if len(table)!=len(expected) or {(z['sid'],z['boundary'],int(z['lag_h']),z['field']) for z in table}!=expected:raise ValueError('Full fourfield support table changed')
 verified_records=verify_prepared_records(table,requests,sources)
 grouped=collections.defaultdict(list)
 for z in table:grouped[(z['boundary'],int(z['lag_h']),z['field'])].append(z)
 summaries=[];stages=[];draws={}
 for (boundary,lag,field),group in sorted(grouped.items()):
  a=summarize_cell(group,pool,weights);key=boundary+'__'+str(lag)+'__'+field
  for i,kind in enumerate(KINDS):
   summaries.append(dict(boundary=boundary,lag_h=lag,field=field,quantity=kind,unit=('%' if field=='r700' and kind!='anomaly_common' else adapter.UNITS[field]),estimate=a['mean'][i],ci_low=a['ci'][0,i],ci_high=a['ci'][1,i],n_sid=a['n'],n_original_sid_pool=len(pool),n_actual_position=sum(boolean(z['actual_track_point']) for z in group),n_raw_own_valid=sum(boolean(z['raw_own_support__valid']) for z in group),raw_own_mean=float(np.mean([number(z['raw_own_support']) for z in group if boolean(z['raw_own_support__valid'])])) if any(boolean(z['raw_own_support__valid']) for z in group) else np.nan,n_draws=2000,valid_draws=int((a['valid_multiplicity']>0).sum()),failed_draws=int((a['valid_multiplicity']==0).sum()),ci_method='pointwise percentile reused shared original2368SID trajectories; not simultaneous',P_q='NOT_TESTED_NO_NEW_FAMILY'))
   draws[key+'__'+kind]=a['draws'][:,i]
  draws[key+'__valid_multiplicity']=a['valid_multiplicity'];draws[key+'__distinct_valid_SID']=a['distinct_valid_sid']
  for stage in sorted({z['frozen_stage'] for z in group}):
   zz=[z for z in group if z['frozen_stage']==stage];stages.append(dict(boundary=boundary,lag_h=lag,field=field,frozen_stage=stage,n_requested=len(zz),n_common_valid=sum(boolean(z['raw_common__valid']) for z in zz),n_actual_position=sum(boolean(z['actual_track_point']) for z in zz),n_legal_forward6h=sum(boolean(z['legal_forward6h']) for z in zz)))
 csvsave(out/'P4_raw_climate_anomaly_CI_n.csv',summaries);csvsave(out/'P4_actual_stage_composition.csv',stages)
 with (out/'pointwise_draws.npz').open('wb') as f:np.savez_compressed(f,**draws)
 write(out/'execution_receipt.json',dict(status='DESCRIPTIVE_COMMON_SUPPORT_SUMMARY_PENDING_INDEPENDENT_REVIEW',source_receipt=str(Path(source_folder)/'source_receipt.json'),source_receipt_sha256=sha(Path(source_folder)/'source_receipt.json'),source_table=r['source_table'],source_table_sha256=r['source_table_sha256'],bank=identity,source_records_reverified=verified_records,new_draws=0,models=0,P_q_families_added=0,raw_original_fiveprocess_results_unchanged=True,config_sha256=sha(HERE/'frozen_config.json'),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},finished_utc=stamp()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-receipt',action='append',default=[]);p.add_argument('--output',required=True);p.add_argument('--run',action='store_true');p.add_argument('--max-new',type=int);p.add_argument('--summarize-source');p.add_argument('--existing-bank');p.add_argument('--prior-nine-execution');a=p.parse_args()
 if a.summarize_source:
  if not a.existing_bank or not a.prior_nine_execution:raise ValueError('Existing actual original draw bank and execution required')
  summarize(a.summarize_source,a.existing_bank,a.prior_nine_execution,a.output);return 0
 if a.run:return prepare(a.source_receipt,a.output,a.max_new)
 gate,_,_,_=input_gate(a.source_receipt);write(safeout(a.output),gate);print(json.dumps(gate));return 0
if __name__=='__main__':sys.exit(main())

