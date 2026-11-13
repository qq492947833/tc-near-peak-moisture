from _release_paths import release_path as _release_path
import os,sys
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,time,hashlib,argparse,csv
import numpy as np,pandas as pd
V=Path(str(_release_path('mechanism_v5','')));O=Path(__file__).resolve().parent;U=V/'upper_profile_100_50_interpolation_v1';os.environ['NUMBA_CACHE_DIR']=str(O/'numba_cache')
sys.path.insert(0,str(V/'src'));sys.path.insert(0,str(U/'production_candidate_v1'));sys.path.insert(0,str(U/'input_interpolation'))
from production_inputs import prepare13,add70
from production_thermodynamics import evaluate
from interpolation_core import joint70
from native_fields_v5 import area_weighted_support
from extract_environment_v5 import clean,atomic_json,digest,filehash
from rh_native_v6 import geometry,loadz,checked
def calc(folder):
 c=json.loads((folder/'COMPLETE.json').read_text());r=c['binding']['request'];dest=O/'fixed_PI'/folder.name;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return json.loads((dest/'derived.json').read_text())
 with np.load(folder/'profile.npz') as z:a={k:z[k].copy() for k in z.files}
 prepared=prepare13(a);sc=pd.read_csv(folder/'scalars.csv').set_index('variable')
 def val(k):
  rr=sc.loc[k];return float(rr.value) if str(rr.support_pass).lower()=='true' and rr.valid_fraction>=.9 else np.nan
 scalars=dict(SST_K=val('sst_K'),msl_Pa=val('msl'),shear_200_850_ms=val('shear_200_850_vector_magnitude_ms'),core600_saturated_entropy_jkgk=val('core600_saturated_entropy_J_kgK'),environment600_entropy_jkgk=val('environment600_entropy_J_kgK'))
 scalars['ventilation_support_confirmed']=all(np.isfinite(scalars[k]) for k in ['shear_200_850_ms','core600_saturated_entropy_jkgk','environment600_entropy_jkgk'])
 row=pd.Series(r);geo=geometry(row).name;new=O/'native_cache'/geo;old=V/'cache/native_fields'/geo;g=loadz(new/'sp.npz');native={}
 for key in ['t100','t50','q100','q50']:native[key]=checked((new if (new/(key+'.npz')).exists() else old)/(key+'.npz'))
 # Both endpoints were read on the same native reference coordinates by the bound reader.
 ring=(g['distance']>=200)&(g['distance']<800);aug={}
 for method,label in [('logp','logp'),('linear_p','plinear')]:
  t,q,j=joint70(native['t100'],native['t50'],native['q100'],native['q50'],method);ts=area_weighted_support(t,g['area'],ring,min_fraction=.9);qs=area_weighted_support(q,g['area'],ring,min_fraction=.9);assert ts['valid_area_m2']==qs['valid_area_m2'];pp=add70(prepared,ts['mean'],qs['mean'],ts['valid_fraction']);aug['T_'+label+'_K']=pp['temperature_K'];aug['q_'+label+'_kgkg']=pp['specific_humidity_kgkg'];aug['pressure_aug_hpa']=pp['pressure_hpa'];aug['joint_fraction_aug']=pp['joint_Tq_valid_fraction']
 arrays={**a,**aug};wind=float(r['wind_kt']) if r['wind_kt'] else np.nan;result=evaluate(arrays,scalars,wind)
 rec={k:r[k] for k in ['request_id','sid','event_id','event_class','UTC','lag_h','lat','lon','center_kind']};rec.update(result['C_14_logp_inclusive50']['authoritative']);rec.update(wind_for_headroom_kt=wind,source_complete=str(folder/'COMPLETE.json'),source_complete_sha256=filehash(folder/'COMPLETE.json'),original_profile_sha256=filehash(folder/'profile.npz'),geometry_id=geo,scalars=scalars,status=result['C_14_logp_inclusive50']['status'])
 atomic_json(dest/'derived.json',rec);atomic_json(dest/'physical_trace.json',result);np.savez_compressed(dest/'augmented_profiles.npz',**arrays);atomic_json(dest/'COMPLETE.json',{'status':'COMPLETE','source_complete_sha256':rec['source_complete_sha256'],'derived_sha256':filehash(dest/'derived.json'),'trace_sha256':filehash(dest/'physical_trace.json'),'profile_sha256':filehash(dest/'augmented_profiles.npz'),'adapter':'unmodified production_thermodynamics.evaluate, logp70 including real50, original13 strict joint support'})
 return clean(rec)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args();records={};tick=time.perf_counter()
 while True:
  text=(O/'fixed_extraction_ledger.jsonl').read_text();lines=text.splitlines();candidates={}
  for line in lines:
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if r['status'] in ['COMPLETE','REUSED_VERIFIED_CHECKPOINT']:candidates[r['request_id']]=r
  for rid,r in candidates.items():
   if rid in records:continue
   records[rid]=calc(Path(r['folder']))
   if len(records)%20==0:
    pd.DataFrame(list(records.values())).to_csv(O/'fixed_PI_values.csv',index=False);atomic_json(O/'fixed_PI_progress.json',{'complete':len(records),'elapsed_seconds':time.perf_counter()-tick});print('PI',len(records),round(time.perf_counter()-tick),flush=True)
   if args.limit and len(records)>=args.limit:break
  if (args.limit and len(records)>=args.limit) or ((O/'fixed_full_receipt.json').exists() and len(records)==len(candidates)):break
  time.sleep(10)
 pd.DataFrame(list(records.values())).to_csv(O/('fixed_PI_pilot.csv' if args.limit else 'fixed_PI_values.csv'),index=False);atomic_json(O/('fixed_PI_pilot_receipt.json' if args.limit else 'fixed_PI_receipt.json'),{'status':'COMPLETE','n':len(records),'seconds':time.perf_counter()-tick,'raw_reads':0,'full_trace_saved':True});print('PI done',len(records),flush=True)
if __name__=='__main__':main()
