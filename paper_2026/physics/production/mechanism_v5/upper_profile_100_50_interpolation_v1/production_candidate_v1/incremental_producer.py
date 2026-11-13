"""Single-worker immutable-cache C/D producer. No fits, raw nc or old writes.
Default audits a supplied immutable source receipt; --run also requires root's
SHA-bound execution acceptance. A new checkpoint scan is a new explicit call.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,gzip,hashlib,json,os,sys,time,traceback,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;U=HERE.parent;ROOT=U.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'))
from production_thermodynamics import evaluate
from pi_endpoint_trace import clean
from derive_thermodynamics_v5 import wind_index,wind_for_margin

REQUESTS=ROOT/'data/environment_requests_v5.csv'
REQUESTS_SHA='a305a2282e71369d3ccbb765992c3bca2521896ff3ecbc95dc3e9fe2b856425d'
ROUTE=U/'route_acceptance_v2.json'
def stamp():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x):return json.dumps(clean(x),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canon(x).encode('utf-8')).hexdigest()
def boolean(x):
 if x is True or x in ('True','true','1'):return True
 if x is False or x in ('False','false','0'):return False
 raise ValueError('Explicit boolean required: '+repr(x))
def number(x):
 try:return float(x)
 except (ValueError,TypeError):return np.nan
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.write_{os.getpid()}.tmp');t.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def csvsave(p,table):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.write_{os.getpid()}.tmp');fields=list(dict.fromkeys(k for r in table for k in r))
 with t.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clean(table))
 os.replace(t,p)
def lock():
 f=(HERE/'producer.lock').open('a+b')
 try:
  if os.name=='nt':
   import msvcrt
   if f.seek(0,2)==0:f.write(b'0');f.flush()
   f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  else:
   import fcntl
   fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  return f
 except Exception:f.close();raise

def immutable_sources(receipts):
 result={};bindings=[]
 for path in receipts:
  rec=read(path)
  if rec.get('status')!='VERIFIED_COMPLETED_SUBSET_NOT_FINAL_COHORT':raise ValueError('Require original immutable completed-subset index receipt')
  table=Path(rec['table']);expected=rec['table_sha256']
  if sha(table)!=expected:raise ValueError('Immutable index SHA conflict')
  bindings.append(dict(receipt=str(path),receipt_sha256=sha(path),index=str(table),index_sha256=expected))
  for row in rows(table):
   rid=row['request_id']
   if rid in result and result[rid]['derived_complete_sha256']!=row['derived_complete_sha256']:raise ValueError('Conflicting immutable source versions for '+rid)
   result[rid]=row
 return result,bindings

def verified_source(row,request):
 cp=Path(row['derived_complete_path'])
 if sha(cp)!=row['derived_complete_sha256']:raise ValueError('Old derived COMPLETE identity mismatch')
 done=read(cp)
 if done.get('status')!='COMPLETE':raise ValueError('Source is not completed')
 for f in done['outputs']:
  if sha(cp.parent/f['name'])!=f['sha256']:raise ValueError('Old derived output changed')
 dr=read(cp.parent/'derived.json');folder=Path(dr['request_folder']);native=folder/'COMPLETE.json';expected=done['binding']['source']['checkpoint_sha256']
 if sha(native)!=expected:raise ValueError('Native source COMPLETE identity mismatch')
 d=read(native)
 if d.get('status')!='COMPLETE':raise ValueError('Native source incomplete')
 original=d['binding']['request']
 if original['request_id']!=request['request_id'] or dr['request_id']!=request['request_id']:raise ValueError('Request binding conflict')
 # Require the exact frozen request, not an ID-only join.
 if canon(original)!=canon(request):raise ValueError('Native request differs from frozen requested fields')
 return folder,dict(request_id=request['request_id'],checkpoint_sha256=expected,old_derived_complete=str(cp),old_derived_complete_sha256=sha(cp),consumed_old_PI_as_input=False)

def dependencies(check_requests=True):
 p=HERE/'production_configuration.json';cfg=read(p)
 if cfg['route']!='C_14_logp_inclusive50' or cfg['sensitivity']!='D_14_plinear_inclusive50':raise ValueError('Frozen C/D route differs')
 for x,h in cfg['bound_files'].items():
  if sha(x)!=h:raise ValueError('Frozen dependency changed: '+x)
 if (check_requests and sha(REQUESTS)!=REQUESTS_SHA) or cfg['requests_sha256']!=REQUESTS_SHA:raise ValueError('Frozen request list identity differs')
 if read(ROUTE).get('status')!='ACCEPTED_DERIVED_ROUTE_FOR_INCREMENTAL_PER_REQUEST_VALIDATION':raise ValueError('Scientific route is not accepted')
 return cfg,dict(configuration=str(p),configuration_sha256=sha(p),route_receipt=str(ROUTE),route_sha256=sha(ROUTE),requests_sha256=REQUESTS_SHA)

def acceptance(path,cfgbinding):
 a=read(path)
 if a.get('status')!='ACCEPTED_FOR_ISOLATED_INCREMENTAL_C_D_EXECUTION':raise ValueError('Explicit root execution gate required')
 if a.get('configuration_sha256')!=cfgbinding['configuration_sha256'] or a.get('route_sha256')!=cfgbinding['route_sha256']:raise ValueError('Execution acceptance binding differs')
 if a.get('automatic_model_release') is not False or a.get('raw_nc_reads')!=0:raise ValueError('Acceptance scope differs')
 return dict(path=str(path),sha256=sha(path))

def arrays_and_scalars(data):
 o=data['original_profile'];c=data['C_profile'];d=data['D_profile']
 arrays=dict(pressure_hpa=o['pressure_hpa'],temperature_K=o['temperature_K'],specific_humidity_kgkg=o['specific_humidity_kgkg'],joint_Tq_valid_fraction=o['joint_Tq_valid_fraction'],pressure_aug_hpa=c['pressure_hpa'],T_logp_K=c['temperature_K'],q_logp_kgkg=c['specific_humidity_kgkg'],T_plinear_K=d['temperature_K'],q_plinear_kgkg=d['specific_humidity_kgkg'],joint_fraction_aug=c['joint_Tq_valid_fraction'])
 if not np.array_equal(c['pressure_hpa'],d['pressure_hpa']) or not np.array_equal(c['joint_Tq_valid_fraction'],d['joint_Tq_valid_fraction'],equal_nan=True):raise ValueError('C/D support differs')
 return arrays,dict(data['surface'],**data['named_scalars'])

def preserve_incomplete(dest,output):
 dest=Path(dest).resolve();output=Path(output).resolve()
 if not dest.is_relative_to(output/'records'):raise ValueError('incomplete record outside isolated output')
 if not dest.exists() or not any(dest.iterdir()):return None
 if (dest/'COMPLETE.json').exists():raise ValueError('Never archive/overwrite a complete checkpoint')
 h=output/'incomplete_history'/digest(dict(path=str(dest),utc=stamp(),pid=os.getpid()))[:16]
 shutil.copytree(dest,h)
 files={str(x.relative_to(dest)):sha(x) for x in dest.rglob('*') if x.is_file()}
 if any(sha(h/name)!=value for name,value in files.items()):raise ValueError('Incomplete history copy verification failed')
 record=dict(source=str(dest),history=str(h),preserved_utc=stamp(),reason='existing incomplete per-request output retained before explicit resume',files=files,original_files_deleted=False)
 atomic(h/'preservation.json',record);return record

def process_one(request,row,output,cfgbinding,wind):
 from production_inputs import build_profile
 folder,source=verified_source(row,request);data=build_profile(folder,source)
 # Re-read and verify every real endpoint/cache binding even on checkpoint reuse.
 # Source support excludes run timestamps; digest binds arrays and complete lineage.
 binding=dict(source=source,implementation=cfgbinding,actual_input_support_sha256=digest(data))
 dest=Path(output)/'records'/digest(request['request_id'])[:20];cp=dest/'COMPLETE.json'
 if cp.exists():
  old=read(cp)
  if old.get('status')!='COMPLETE_REQUEST_DIAGNOSTIC_NOT_MODEL_RELEASE':raise ValueError('Existing request checkpoint not completed')
  if old['binding']!=binding:raise ValueError('Existing record belongs to different source/config; use new output version')
  for p,h in old['outputs'].items():
   if sha(dest/p)!=h:raise ValueError('Existing record checksum mismatch')
  return read(dest/'summary.json'),'REUSED_VERIFIED_CHECKPOINT'
 history=preserve_incomplete(dest,output)
 a,s=arrays_and_scalars(data);w=wind_for_margin(request,wind)
 values=evaluate(a,s,w['wind_for_headroom_kt'])
 # Source was immutable for the complete actual input read.
 if sha(folder/'COMPLETE.json')!=source['checkpoint_sha256']:raise ValueError('Native input changed during read')
 summary=dict(request_id=request['request_id'],module=request['module'],sid=request['sid'],UTC=request['UTC'],stage=request.get('stage',''),event_class=request.get('event_class',''),lag_h=request.get('lag_h',''),center_kind=request.get('center_kind',''),request_folder=str(folder),input_quality_status=data['input_quality_status'],**w)
 summary['incomplete_history_preserved']=history
 summary['configurations']={k:dict(status=v['status'],authoritative=v['authoritative'],failure_reason=v.get('failure_reason'),physical_reasons=v.get('physical_gate',{}).get('reasons',[])) for k,v in values.items()}
 dest.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest/'profile_inputs.npz',**a)
 with (dest/'C_D_full_trace.json.gz').open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as g:g.write(canon(values).encode('utf-8'))
 atomic(dest/'source_support.json',data);atomic(dest/'summary.json',summary)
 outputs={p.name:sha(p) for p in (dest/'profile_inputs.npz',dest/'C_D_full_trace.json.gz',dest/'source_support.json',dest/'summary.json')}
 atomic(cp,dict(status='COMPLETE_REQUEST_DIAGNOSTIC_NOT_MODEL_RELEASE',binding=binding,outputs=outputs,finished_utc=stamp(),raw_nc_reads=0,model_runs=0))
 return clean(summary),'COMPLETE_NEW_RECORD'

def inventory(requests,sources):
 table=[]
 for r in requests:
  need=boolean(r['need_profile']);position=np.isfinite(number(r['lat'])) and np.isfinite(number(r['lon']))
  status='NOT_REQUESTED_PROFILE' if not need else 'EXCLUDED_MISSING_ACTUAL_POSITION' if not position else 'IMMUTABLE_SOURCE_AVAILABLE' if r['request_id'] in sources else 'PENDING_IMMUTABLE_SOURCE'
  table.append(dict(request_id=r['request_id'],module=r['module'],sid=r['sid'],stage=r.get('stage',''),event_class=r.get('event_class',''),need_profile=need,valid_position=position,status=status))
 return table

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-receipt',type=Path,action='append',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run',action='store_true');ap.add_argument('--acceptance',type=Path);ap.add_argument('--max-new',type=int);ap.add_argument('--module',action='append',choices=['P1','P3','P4','P5']);args=ap.parse_args()
 output=args.output.resolve()
 if not output.is_relative_to(U.resolve()) or output==HERE.resolve():raise ValueError('All new output must be isolated under upper-profile directory')
 lk=lock();start=stamp();runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid());audit=output/'executions'/runid;audit.mkdir(parents=True,exist_ok=False)
 execution=dict(status='STARTED',started_utc=start,argv=[sys.executable]+sys.argv,pid=os.getpid(),models_run=0,raw_nc_reads=0,new_records=0,reused_records=0,errors=[])
 atomic(audit/'STARTED.json',execution)
 try:
  cfg,cb=dependencies();sources,sb=immutable_sources(args.source_receipt);requests=rows(REQUESTS);inv=inventory(requests,sources);csvsave(audit/'full_request_scope.csv',inv)
  execution.update(implementation=cb,immutable_source_checkpoints=sb,all_requests=len(requests),scope_counts={s:sum(r['status']==s for r in inv) for s in sorted(set(r['status'] for r in inv))})
  if not args.run:execution['status']='PREFLIGHT_ONLY_NO_SCIENCE';returncode=0
  else:
   if args.acceptance is None:raise ValueError('--run requires explicit SHA-bound --acceptance')
   execution['execution_acceptance']=acceptance(args.acceptance,cb);wind=wind_index(requests);allsummary=[]
   execution.update(selected_modules=args.module or ['P1','P3','P4','P5'],maximum_new_records=args.max_new,selected_available_records=sum(i['status']=='IMMUTABLE_SOURCE_AVAILABLE' and (not args.module or r['module'] in args.module) for r,i in zip(requests,inv)),whole_module_complete=False)
   for r,i in zip(requests,inv):
    if i['status']!='IMMUTABLE_SOURCE_AVAILABLE' or (args.module and r['module'] not in args.module):continue
    if args.max_new is not None and execution['new_records']>=args.max_new:break
    try:
     _,nowbinding=dependencies(check_requests=False)
     if nowbinding!=cb:raise ValueError('Dependencies changed during producer')
     summary,status=process_one(r,sources[r['request_id']],output,cb,wind);allsummary.append(summary)
     execution['new_records' if status=='COMPLETE_NEW_RECORD' else 'reused_records']+=1
     print(canon(dict(time=stamp(),request_id=r['request_id'],status=status)),flush=True)
    except Exception as e:
     err=dict(request_id=r['request_id'],error=repr(e),traceback=traceback.format_exc());execution['errors'].append(err);print(canon(dict(status='FAILED_RECORD_STOP_DISPATCH',**err)),flush=True);break
    if (execution['new_records']+execution['reused_records'])%100==0:atomic(audit/'progress.json',execution)
   execution['selected_available_unprocessed']=execution['selected_available_records']-execution['new_records']-execution['reused_records']-len(execution['errors'])
   flattened=[]
   for s in allsummary:
    for k,v in s['configurations'].items():flattened.append(dict({x:y for x,y in s.items() if x!='configurations'},configuration=k,status=v['status'],failure_reason=v['failure_reason'],physical_reasons=';'.join(v['physical_reasons']),**v['authoritative']))
   dependencies(check_requests=True)
   csvsave(audit/'current_processed_source_table.csv',flattened)
   execution.update(status='COMPLETED_BOUNDED_PASS_NOT_MODEL_RELEASE' if not execution['errors'] else 'COMPLETED_WITH_RECORDED_ERRORS',processed_index=str(audit/'current_processed_source_table.csv'),processed_index_sha256=sha(audit/'current_processed_source_table.csv'))
   returncode=1 if execution['errors'] else 0
 except Exception as e:
  execution.update(status='FAILED_NO_AUTOMATIC_RETRY',error=repr(e),traceback=traceback.format_exc());returncode=1
 finally:
  execution.update(finished_utc=stamp(),exit_code=locals().get('returncode',1));atomic(audit/'execution.json',execution);lk.close()
 print(canon(execution));return returncode
if __name__=='__main__':sys.exit(main())





