"""Regional observed pre-window distributions and fixed-SD descriptive contrasts."""
from pathlib import Path
import os,json,datetime
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
from geography_tables import O,read,regions,sha
from fit_environment_families import ENV,boot

def main():
    e=read(O/'environment_model_all_risk_SID_source.csv').sort_values('sid').reset_index(drop=True)
    long=read(O/'pre_environment_SID_means_long.csv');variables=sorted(long.variable.unique())
    indices=np.random.default_rng(62112026).integers(0,len(e),size=(2000,len(e)),dtype=np.int32);w=np.array([np.bincount(ix,minlength=len(e)) for ix in indices],np.int16)
    np.savez_compressed(O/'pre_environment_SID_bootstrap_indices.npz',sid=e.sid.to_numpy(str),indices=indices)
    rows=[];contrasts=[];draws=[];keys=[];ref=[]
    outcome=e.recovery48;groups={'all':np.ones(len(e),bool),'known_return':outcome.eq(1).to_numpy(),'complete_nonreturn':outcome.eq(0).to_numpy(),'unknown':outcome.isna().to_numpy()}
    common=np.isfinite(e[ENV].to_numpy(float)).all(1)
    for var in variables:
        values=e[var].to_numpy(float);referenceSD=float(np.nanstd(values[common],ddof=1));referenceMean=float(np.nanmean(values[common]));ref.append(dict(variable=var,reference='all shared primary-environment SID',n_sid=int(np.isfinite(values[common]).sum()),mean=referenceMean,sd=referenceSD))
        for reg,rmask in regions(e):
            store={}
            for group,gmask in groups.items():
                mask=rmask&gmask;est,ci,draw=boot(values,mask,w);n=int((mask&np.isfinite(values)).sum())
                rows.append(dict(region=reg,variable=var,outcome_group=group,n_sid=n,eligible_sid=int(mask.sum()),mean=est,ci_low=ci[0],ci_high=ci[1],B_valid=int(np.isfinite(draw).sum()),support='variable requires >=3 of four actual pre-times; individual SID means equally weighted'))
                store[group]=(n,est,draw);keys.append(var+'|'+reg+'|'+group);draws.append(draw)
            for group in ['known_return','unknown']:
                n,est,draw=store[group];nb,baseline,bd=store['complete_nonreturn'];contrast=est-baseline;cd=draw-bd;ci=np.nanquantile(cd,[.025,.975]) if min(n,nb)>=3 else [np.nan,np.nan]
                contrasts.append(dict(region=reg,variable=var,comparison=group+' minus complete_nonreturn',n_group=n,n_nonreturn=nb,estimate=contrast,ci_low=ci[0],ci_high=ci[1],B_valid=int(np.isfinite(cd).sum()),reference_sd=referenceSD,standardized_difference=contrast/referenceSD if referenceSD>0 else np.nan,standardized_ci_low=ci[0]/referenceSD if referenceSD>0 else np.nan,standardized_ci_high=ci[1]/referenceSD if referenceSD>0 else np.nan))
                keys.append(var+'|'+reg+'|'+group+'minus_nonreturn');draws.append(cd)
    pd.DataFrame(rows).to_csv(O/'pre_environment_regional_distributions.csv',index=False);pd.DataFrame(contrasts).to_csv(O/'pre_environment_group_contrasts.csv',index=False);pd.DataFrame(ref).to_csv(O/'pre_environment_standardization_reference.csv',index=False)
    np.savez_compressed(O/'pre_environment_bootstrap_draws.npz',keys=np.array(keys),draws=np.array(draws))
    receipt=dict(status='COMPLETE_PENDING_REVIEW',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),n_risk=len(e),variables=variables,bootstrap_draws=2000,bootstrap_unit='SID',event_groups='known 48h return, complete nonreturn, and unknown; never code unknown as negative',standardized_display='Differences in measured environmental means divided by fixed SD of the shared primary-environment SID set; original units and CI retained',source_sha256={str(p):sha(p) for p in [Path(__file__),O/'environment_model_all_risk_SID_source.csv',O/'pre_environment_SID_means_long.csv']})
    (O/'pre_environment_summary_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))

if __name__=='__main__':main()
