"""Refit all 36 accepted adjacent-stage adjusted models on the frozen master SID."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import ast,json,hashlib,time,datetime,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).absolute().parents[1]
ADD=HERE.parent
BASE=ADD/'dependencies/v62/stages'
SRC=HERE/'statistics/stages';OUT=HERE/'statistics/stage_adjusted';OUT.mkdir(parents=True,exist_ok=True)
GEO=['lat','lon_sin','lon_cos','month_sin','month_cos'];STATE=GEO+['wind_kt','age_h','dv24_kt','center_land','past24_land_contact']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def js(p,z):
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(clean(z),indent=2)+'\n',encoding='utf-8');start=time.monotonic()
    while True:
        try:os.replace(t,p);return
        except OSError as e:
            if getattr(e,'winerror',None) not in [5,32,33] or time.monotonic()-start>60:raise
            time.sleep(.15)
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip')
def main():
    import psutil
    proc=psutil.Process()
    if os.name == "nt": proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    tree=ast.parse((BASE/'analysis_b.py').read_text(encoding='utf-8')); ns=dict(np=np,pd=pd)
    names={'make_pair','fit_pair'};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'accepted_analysis_b_functions','exec'),ns)
    new=pd.read_parquet(SRC/'model_stage_sid.parquet');old=pd.read_parquet(ADD/'source_data/de_existing_sources/model_stage_sid.parquet');orig=read(ADD/'source_data/de_stage_adjusted/adjusted_adjacent_summary.csv')
    weights=np.load(SRC/'bootstrap_sid_multiplicities.npy');sidlist=read(SRC/'bootstrap_sid_order.csv').sid.tolist();ix={s:i for i,s in enumerate(sidlist)}
    assert weights.shape==(2000,len(sidlist))
    bind=dict(source=sha(SRC/'model_stage_sid.parquet'),original_functions=sha(BASE/'analysis_b.py'),master_bank=sha(SRC/'bootstrap_sid_multiplicities.npy'),master_sid=sha(HERE/'source_data/main_sid.txt'),driver=sha(__file__))
    identity=dict(pid=os.getpid(),create_time=proc.create_time(),command_line=proc.cmdline(),argv=sys.argv,started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),bindings=bind)
    js(HERE/'reports/stage_adjusted_identity.json',identity)
    assert len(sidlist)==2340 and set(new.sid)<=set(sidlist)
    out=[];all_members=[];all_ranges=[];all_details=[];all_draws=[];checks=[];tick=time.monotonic()
    for vv in new.variable.unique():
        for scheme,ctrl in [('geography',GEO),('state',STATE)]:
            a=new[new.variable.eq(vv)&new.scheme.eq(scheme)];a0=old[old.variable.eq(vv)&old.scheme.eq(scheme)];retain=[] if scheme=='geography' else ['age_h']
            for frm,to in [('D','E'),('E','F'),('F','Res')]:
                contrast=f'{to}-{frm}';key=scheme+'_'+vv+'_'+contrast
                identity=dict(variable=vv,scheme=scheme,contrast=contrast,stage_from=frm,stage_to=to,unit=str(a.unit.iloc[0]))
                p=ns['make_pair'](a,frm,to,ctrl);p0=ns['make_pair'](a0,frm,to,ctrl)
                folder=OUT/key;folder.mkdir(exist_ok=True);cp=folder/'complete.json'
                if cp.exists():
                    done=json.loads(cp.read_text());assert done['bindings']==bind
                    for name,h in done['output_hashes'].items():assert sha(folder/name)==h
                    r=read(folder/'summary.csv').iloc[0].to_dict();member=read(folder/'members.csv');ranges=read(folder/'ranges.csv');details=json.loads((folder/'details.json').read_text());draws=pd.read_parquet(folder/'draws.parquet')
                else:
                    point0=ns['fit_pair'](p0,ctrl,retain=retain,details=True);rr=orig[orig.variable.eq(vv)&orig.scheme.eq(scheme)&orig.contrast.eq(contrast)].iloc[0]
                    assert point0['status']==rr.status
                    if np.isfinite(rr.estimate):assert abs(point0['estimate']-rr.estimate)<1e-10
                    checks.append(dict(**identity,old_point_reproduced=True,old_status=point0['status']))
                    point=ns['fit_pair'](p,ctrl,retain=retain,details=True)
                    member=p.copy();member['selected']=point.get('selected_mask',np.zeros(len(p),bool));member['exclusion_reasons']=point.get('exclusion_reasons',['NO_RESAMPLED_PAIRS']*len(p))
                    for k,v in identity.items():member[k]=v
                    ranges=pd.DataFrame([{**identity,**v} for v in point['ranges']]);details={**identity,**{k:v for k,v in point.items() if k not in ['selected_mask','exclusion_reasons']}}
                    ii=np.array([ix[s] for s in p.sid]);dr=[]
                    for b in range(2000):
                        fit=ns['fit_pair'](p,ctrl,weights=weights[b,ii],retain=retain)
                        dr.append({**identity,'draw':b,**{k:fit.get(k,np.nan) for k in ['estimate','raw_matched_estimate','n_pairs','n_unique','status','condition_number','rank','residual_df','target_identifiability_error']}})
                    draws=pd.DataFrame(dr);good=np.isfinite(draws.estimate)
                    r={**identity,**{k:point.get(k,np.nan) for k in ['estimate','raw_matched_estimate','before_matched_estimate','after_matched_estimate','n_original_pairs','n_pairs','n_unique','retained_fraction','status','rank','residual_df','condition_number','residual_sd','max_leverage','target_identifiability_error']},'ci_low':draws.loc[good,'estimate'].quantile(.025) if good.any() else np.nan,'ci_high':draws.loc[good,'estimate'].quantile(.975) if good.any() else np.nan,'bootstrap_attempted':2000,'bootstrap_valid':int(good.sum()),'bootstrap_failed':int((~good).sum()),'failure_reasons':json.dumps(draws.loc[~good,'status'].value_counts().to_dict()),'raw_on_support_ci_low':draws.raw_matched_estimate.quantile(.025),'raw_on_support_ci_high':draws.raw_matched_estimate.quantile(.975),'reference_sd':rr.reference_sd,'target_age':'observed delta_age retained' if scheme=='state' else 'not adjusted'}
                    if not np.isfinite(r['estimate']):r['ci_low']=r['ci_high']=np.nan
                    r['interval_scope']='conditional on successful resamples if any fixed attempts fail; no unconditional coverage claim'
                    pd.DataFrame([r]).to_csv(folder/'summary.csv',index=False);member.to_csv(folder/'members.csv',index=False);ranges.to_csv(folder/'ranges.csv',index=False);js(folder/'details.json',details);draws.to_parquet(folder/'draws.parquet',index=False)
                    js(cp,dict(status='COMPLETE',bindings=bind,output_hashes={n:sha(folder/n) for n in ['summary.csv','members.csv','ranges.csv','details.json','draws.parquet']}))
                out.append(r);all_members.append(member);all_ranges.append(ranges);all_details.append(details);all_draws.append(draws)
                js(HERE/'reports/stage_adjusted_status.json',dict(status='RUNNING',completed_targets=len(all_details),planned_targets=36,latest=identity,elapsed_s=time.monotonic()-tick,pid=os.getpid(),create_time=proc.create_time()))
                print(key,'complete',len(all_details),'/36',round(time.monotonic()-tick,1),flush=True)
    pd.DataFrame(out).to_csv(OUT/'adjusted_adjacent_summary.csv',index=False);pd.concat(all_draws,ignore_index=True).to_parquet(OUT/'adjustment_bootstrap_all.parquet',index=False)
    for name,frames in [('adjustment_pair_membership.csv',all_members),('adjustment_support_ranges.csv',all_ranges)]:
        pd.concat(frames,ignore_index=True).to_csv(OUT/name,index=False)
    js(OUT/'adjustment_model_details.json',all_details)
    receipt=dict(status='COMPLETE_ALL_36_ADJUSTED_MODELS_ON_FROZEN_MASTER',changed_targets=36,unchanged_targets_reused=0,main_sid_n=len(sidlist),bootstrap_attempts_per_target=2000,no_redraw=True,bindings=bind,baseline_point_checks=checks,raw_ERA5_reads=0,elapsed_s=time.monotonic()-tick,output_hashes={n:sha(OUT/n) for n in ['adjusted_adjacent_summary.csv','adjustment_bootstrap_all.parquet','adjustment_pair_membership.csv','adjustment_support_ranges.csv','adjustment_model_details.json']})
    js(HERE/'reports/stage_adjusted_receipt.json',receipt);js(HERE/'reports/stage_adjusted_status.json',receipt)
if __name__=='__main__':main()
