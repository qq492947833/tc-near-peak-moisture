from _release_paths import release_path as _release_path
import os,sys
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,time,csv,hashlib,shutil,argparse,ctypes,traceback
from dataclasses import replace
import numpy as np,pandas as pd
V=Path(str(_release_path('mechanism_v5','')));O=Path(__file__).resolve().parent
sys.path.insert(0,str(V/'src'))
import extract_environment_v5 as ex
class SharedReadonlyCache(ex.ExtractionCache):
 def reference(self,utc,lat,lon,radius):
  geo,obj=self.geometry(utc,lat,lon,radius);old=V/'cache/native_fields'/geo;new=self.root/geo
  if not(new/'sp.npz').exists() and (old/'sp.npz').exists() and(old/'geometry.json').exists():
   meta=json.loads((old/'geometry.json').read_text());assert ex.filehash(old/'sp.npz')==meta['npz_sha256'];new.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(old/'sp.npz',new/'sp.npz');shutil.copyfile(old/'geometry.json',new/'geometry.json')
  return super().reference(utc,lat,lon,radius)
 def get(self,key,utc,lat,lon,radius):
  geo,obj=self.geometry(utc,lat,lon,radius);old=V/'cache/native_fields'/geo;p=old/(key+'.npz');meta=old/(key+'.json')
  if key!='sp' and p.exists() and meta.exists():
   b=self.reference(utc,lat,lon,radius);m=json.loads(meta.read_text());assert m['npz_sha256']==ex.filehash(p)
   with np.load(p,allow_pickle=False) as z:
    assert json.loads(str(z['binding_json'].item()))==m['binding'];raw=z['raw'].copy();a=z['values'].copy();mask=z['support_mask'].copy()
   cfg=self.reader.map[key];assert m['binding']['geometry']==geo;self.cache_hits+=1
   return replace(b,key=key,variable=cfg['variable'],units=cfg['units'],pressure_level_pa=None if cfg.get('pressure_level_hpa') is None else cfg['pressure_level_hpa']*100,raw=raw,values=a,support_mask=mask,metadata=dict(cache=str(p),**m['binding']))
  return super().get(key,utc,lat,lon,radius)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();out=O/'fixed_extraction';out.mkdir(exist_ok=True);cache_dir=O/'native_cache';cache_dir.mkdir(exist_ok=True)
 requests=pd.read_csv(V/'data/p1_environment_requests_v5.csv',keep_default_na=False);requests=requests[(requests.current117.astype(str).str.lower()=='true')&(requests.center_kind=='fixed_t0')&requests.lag_h.astype(int).between(0,24)].copy()
 requests['variable_set']='full_process';requests['need_profile']='True';requests['need_map']='False';requests['request_id']=requests.request_id.map(lambda x:'v6|'+x)
 # use current event wind at this UTC from along requests, never the fixed t0 wind.
 along=pd.read_csv(V/'data/p1_environment_requests_v5.csv',keep_default_na=False);along=along[along.center_kind=='actual_track'].set_index(['sid','event_id','lag_h'])
 requests['wind_kt']=[along.loc[(r.sid,r.event_id,r.lag_h),'wind_kt'] for r in requests.itertuples()]
 requests.to_csv(O/'fixed_extraction_requests.csv',index=False)
 requests=requests.sort_values(['UTC','sid','event_class']);requests=requests.iloc[:a.limit or None]
 sem=json.loads((V/'config/extraction_semantics_v5.json').read_text());masks=ex.OceanMasks(sem['sst']['shapefile']);lock=O/'fixed_extraction_active.lock'
 with lock.open('x') as f:json.dump({'pid':os.getpid(),'started':ex.stamp(),'limit':a.limit},f)
 started=time.perf_counter();records=[]
 try:
  with ex.NativeFieldReader(V/'config/available_variable_map_v5.json',max_open=8,chunk_cache_bytes=2*1024**2) as reader:
   cache=SharedReadonlyCache(reader,cache_dir,sem)
   for i,r in enumerate(requests.to_dict('records')):
    r={k:(str(v) if not isinstance(v,str) else v) for k,v in r.items()};tick=time.perf_counter()
    try:rec=ex.extract_one(r,cache,masks,out,sem)
    except Exception:
     rec={'request_id':r['request_id'],'status':'FAILED','error':traceback.format_exc()};ex.atomic_json(O/'fixed_extraction_error.json',rec);raise
    records.append(rec);ex.atomic_json(O/'fixed_extraction_progress.json',{'pid':os.getpid(),'done':i+1,'total':len(requests),'new_raw_reads':cache.new_raw_reads,'cache_hits':cache.cache_hits,'last_seconds':time.perf_counter()-tick,'elapsed_seconds':time.perf_counter()-started,'last':rec})
    with (O/'fixed_extraction_ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
    reader.read_log.clear()
    if i%10==0:print(i+1,len(requests),round(time.perf_counter()-tick,2),cache.new_raw_reads,flush=True)
   ex.atomic_json(O/('fixed_pilot_receipt.json' if a.limit else 'fixed_full_receipt.json'),{'status':'COMPLETE','n_requests':len(records),'raw_read_operations':cache.new_raw_reads,'cache_hits':cache.cache_hits,'seconds':time.perf_counter()-started,'records':records})
 finally:
  if lock.exists():lock.rename(O/('fixed_lock_released_'+str(os.getpid())+'.json'))
if __name__=='__main__':main()
