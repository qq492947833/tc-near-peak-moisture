"""Resume-safe bounded direct-MWT queue. Only starts this task's SID workers."""
import os,json,time,subprocess,sys,hashlib,datetime,shutil
from pathlib import Path
import psutil,pandas as pd
W=Path(__file__).resolve().parents[1]
runner=W/'scripts/run_mwt_sid.py'
def atomic_status_write(path, value, timeout=60.0):
 # Windows readers may briefly deny replacement; retain the last complete receipt.
 path=Path(path)
 tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
 with tmp.open('w',encoding='utf8') as f:
  json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 deadline=time.monotonic()+timeout
 while True:
  try:
   os.replace(tmp,path);return
  except OSError as exc:
   if getattr(exc,'winerror',None) not in (5,32,33) or time.monotonic()>=deadline:
    raise
   time.sleep(.2)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
expected='c70e9f077510d4d2661a0d166ee26eab41470b94f4dededf2ec0c52129fc34d7'
assert sha(runner)==expected
cfgsha=sha(W/'config/analysis_contract.json')
ids=sorted(pd.read_csv(W/'source_data/events_frozen.csv').sid.unique())
runroot=W/'logs';runroot.mkdir(exist_ok=True)
statuspath=W/'reports/production_status.json'
running={};done=[];failed=[];pending=[];reused=[]
def check_completed(sid,full=False):
 p=W/'cache'/sid/'completed.json'
 if not p.exists():return False
 r=json.loads(p.read_text(encoding='utf8'))
 if r['config_sha256']!=cfgsha:raise RuntimeError('Config changed for '+sid)
 code=r['code_sha256']['run_mwt_sid.py']
 if code!=expected:
  assert sid=='1982005S10105' and code=='27fb7422b3efff8f484a968b97ce7a867a07afd143be88e04e262430ee2f204e'
  guard=json.loads((W/'validation/runner_anchor_guard_adoption.json').read_text(encoding='utf8'))
 if full:
  for name,h in r['outputs_sha256'].items():
   if sha(p.parent/name)!=h:raise RuntimeError('Output hash changed '+sid+'/'+name)
 return True
for sid in ids:
 if check_completed(sid,True):done.append(sid);reused.append(sid)
 else:
  lock=W/'cache'/sid/'running.json'
  active=False
  if lock.exists():
   old=json.loads(lock.read_text(encoding='utf8'))
   try:
    proc=psutil.Process(old['pid'])
    active=proc.is_running() and abs(proc.create_time()-old['create_time'])<.001
    if active:
     running[sid]={'p':proc,'external':True,'log':None,'created':old['create_time'],'started':'resumed existing exact-identity worker'}
   except psutil.NoSuchProcess:pass
  if not active:pending.append(sid)
identity_path=W/'reports/production_queue_identity.json'
if identity_path.exists():
 old=json.loads(identity_path.read_text(encoding='utf8'))
 try:
  proc=psutil.Process(old['pid'])
  if proc.is_running() and abs(proc.create_time()-old['create_time'])<.001:
   raise RuntimeError('An exact-identity production queue is already alive; do not start another')
 except psutil.NoSuchProcess:pass
me=psutil.Process()
queue_id={'pid':os.getpid(),'create_time':me.create_time(),'command':sys.argv,'config_sha256':cfgsha,'runner_sha256':expected}
(W/'reports/production_queue_identity.json').write_text(json.dumps(queue_id,indent=2),encoding='utf8')
def save():
 state=dict(updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='CALCULATING' if pending or running else 'CALCULATIONS_COMPLETE' if not failed else 'CALCULATIONS_INCOMPLETE',
            total=len(ids),completed=len(done),reused=reused,failed=failed,pending=pending,
            running={s:{'pid':a['p'].pid,'create_time':a['created'],'started_UTC':a['started']} for s,a in running.items()},
            completed_SID=done,worker_limit=4,priority='BELOW_NORMAL',gpu_used=False,
            publication_complete=False)
 atomic_status_write(statuspath,state)
save()
while pending or running:
 for sid,a in list(running.items()):
  if a.get('external'):
   try:alive=a['p'].is_running() and abs(a['p'].create_time()-a['created'])<.001
   except psutil.NoSuchProcess:alive=False
   ret=None if alive else 0
  else:ret=a['p'].poll()
  if ret is None:continue
  if a['log'] is not None:a['log'].close()
  if ret==0 and check_completed(sid,True):done.append(sid)
  else:failed.append({'sid':sid,'exit_code':ret,'log':str(runroot/(sid+'.log'))})
  del running[sid];save()
 cpu=psutil.cpu_percent(interval=1);mem=psutil.virtual_memory().available
 free=shutil.disk_usage(W).free
 # Existing higher-priority jobs retain scheduling preference; no GPU and no process intervention.
 can_start=(mem>120*1024**3 and free>250*1024**3 and (cpu<90 or not running))
 if pending and len(running)<4 and can_start:
  sid=pending.pop(0);log=open(runroot/(sid+'.log'),'a',encoding='utf8')
  cmd=[sys.executable,'-X','utf8','-u',str(runner),'--work',str(W),'--sid',sid]
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS,
                     env={**os.environ,'PYTHONUTF8':'1'},cwd=str(W/'scripts'))
  running[sid]={'p':p,'log':log,'created':psutil.Process(p.pid).create_time(),'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  save()
 if mem<=120*1024**3 or free<=250*1024**3:
  with open(runroot/'resource_wait.jsonl','a') as f:f.write(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'available_memory':mem,'free_H':free})+'\n')
 time.sleep(10)
save()
print(json.dumps({'completed':len(done),'failed':failed,'publication_complete':False}),flush=True)

