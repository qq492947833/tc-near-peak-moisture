"""Explicit index of currently verified thermodynamic checkpoints; no globbing result folders."""
from pathlib import Path
import argparse,json,sys
from datetime import datetime
import derive_thermodynamics_v5 as d
def index(main_log,repairs,output):
 events=[]
 for line in Path(main_log).read_text(encoding='utf-8').splitlines():
  try:x=json.loads(line)
  except ValueError:continue
  if x.get('status') in ('COMPLETE','REUSED_VERIFIED_CHECKPOINT') and x.get('folder'):events.append(dict(x,origin=str(main_log)))
 for p in repairs:
  x=d.read(p)
  if x.get('status')!='PASS':raise ValueError('Unaccepted repair audit')
  events.extend(dict(e,origin=str(p)) for e in x['events'])
 picked={};evidence=[]
 for e in events:
  folder=Path(e['folder']);cp=folder/'COMPLETE.json';c=d.read(cp)
  if c.get('status')!='COMPLETE':raise ValueError('Incomplete derived output')
  for x in c['outputs']:
   if d.sha(folder/x['name'])!=x['sha256']:raise ValueError('Derived file integrity conflict')
  row=d.read(folder/'derived.json');rid=row['request_id'];source=Path(row['request_folder'])/'COMPLETE.json'
  if d.sha(source)!=c['binding']['source']['checkpoint_sha256']:raise ValueError('Source checkpoint changed')
  if rid!=e['request_id']:raise ValueError('Request mapping conflict')
  if rid in picked:
   if picked[rid]['derived_complete_sha256']!=d.sha(cp):raise ValueError('Two differing accepted derived outputs for same request '+rid)
   continue
  row.update(derived_folder=str(folder),derived_complete_path=str(cp),derived_complete_sha256=d.sha(cp),derivation_evidence_path=e['origin'])
  picked[rid]=row;evidence.append(dict(request_id=rid,complete=str(cp),sha256=d.sha(cp)))
 d.csvsave(output,list(picked.values()))
 receipt=dict(status='VERIFIED_COMPLETED_SUBSET_NOT_FINAL_COHORT',updated_utc=d.stamp(),command=[sys.executable]+sys.argv,n_requests=len(picked),table=str(output),table_sha256=d.sha(output),primary_log=str(main_log),repair_execution_files=[str(p) for p in repairs],source_binding='each request follows actual COMPLETE record; no result-directory name/time inference',raw_ERA5_reads=0)
 d.atomic(output.with_suffix('.json'),receipt);print(d.canon(receipt));return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--main-log',type=Path,required=True);p.add_argument('--repair-execution',action='append',type=Path,default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();index(a.main_log,a.repair_execution,a.output)
