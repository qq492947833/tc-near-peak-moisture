from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.special import expit,logit
import fit_policies_v6 as f
HERE=Path(__file__).resolve().parent
class Unpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='__main__' and hasattr(f,name):return getattr(f,name)
        return super().find_class(module,name)
def load(p):
    with open(p,'rb') as stream:return Unpickler(stream).load()
def main():
    e=f.read(HERE/'cohort_recovered.csv');e['_rowid']=np.arange(len(e));e.complete48=e.complete48.astype(str).str.lower().eq('true')
    p=f.read(HERE/'point_predictions.csv')
    for c in ['covered','R48','R24']:p[c]=p[c].astype(str).str.lower().eq('true')
    cal=[];influence=[];sens=[];bench=[];replay=[]
    for policy,g in p.groupby('policy'):
        for out in ['48','24']:
            good=g['R'+out].to_numpy();y=g['Y'+out].to_numpy();pred=g['m'+out].to_numpy();w=g['ipcw'+out].to_numpy()
            qs=pd.qcut(pd.Series(pred),10,labels=False,duplicates='drop').to_numpy()
            for q in np.unique(qs):
                ss=(qs==q)&good
                cal.append(dict(policy=policy,outcome=out,bin=int(q),n=int(ss.sum()),predicted=np.average(pred[ss],weights=w[ss]),observed=np.average(y[ss],weights=w[ss])))
            residual=np.nan_to_num(y-pred);correction=(g.density_ratio.to_numpy()-1)*w*residual
            ix=np.argsort(np.abs(correction))[::-1]
            for k in ix[:20]:influence.append(dict(policy=policy,outcome=out,sid=g.iloc[k].sid,event_id=g.iloc[k].event_id,correction=correction[k],weight=w[k],ratio=g.iloc[k].density_ratio,outcome_value=y[k],prediction=pred[k]))
            for k in [0,1,5,10,20]:
                keep=np.ones(len(g),bool);keep[ix[:k]]=False
                # Diagnostic only: original average denominator retained; no alternative estimate selected.
                sens.append(dict(policy=policy,outcome=out,largest_corrections_set_to_zero=k,gcomp=g['gcomp'+out].mean(),remaining_average_correction=correction[keep].sum()/len(g),original_onestep=g['onestep'+out].mean()))
        for fold in range(5):
            model=load(HERE/'models'/f'{policy}_fold{fold}.pkl');tr=e[e.fold.ne(fold)];te=e[e.fold.eq(fold)];pp=g.set_index('event_id').loc[te.event_id]
            T=model['transform'];w=T.transform(te);X=f.design(w,model['basis'],te[policy].to_numpy())
            replay.append(dict(policy=policy,fold=fold,max_prediction_error48=float(np.max(np.abs(model['hazard'].predict(X)-pp.m48.to_numpy()))),max_prediction_error24=float(np.max(np.abs(model['wind'].predict(X)-pp.m24.to_numpy()))),train_indices_exact=model['train_rows']==tr._rowid.tolist(),test_indices_exact=model['test_rows']==te._rowid.tolist(),optimizer=model['hazard'].optimization))
            # Residualized observed pret0 wind change supplies an actual-variable benchmark.
            wt=T.transform(tr);j=T.names.index('delta_v24_backward_kt');keep=np.ones(wt.shape[1],bool);keep[j]=False;keep[j+len(T.names)]=False
            A=tr[policy].to_numpy();U=tr.delta_v24_backward_kt.fillna(tr.delta_v24_backward_kt.median()).to_numpy();wa=wt[:,keep]
            ar=A-Ridge(alpha=1).fit(wa,A).predict(wa);ur=U-Ridge(alpha=1).fit(wa,U).predict(wa)
            rho=float(np.corrcoef(ar,ur)[0,1]);sdA=float(ar.std());sdU=float(ur.std())
            altered=te.copy();altered.delta_v24_backward_kt=altered.delta_v24_backward_kt+sdU;Xu=f.design(T.transform(altered),model['basis'],te[policy].to_numpy())
            br=(model['hazard'].predict(Xu)-model['hazard'].predict(X));bv=model['wind'].predict(Xu)-model['wind'].predict(X)
            covered=pp.covered.to_numpy();delta=f.POLICIES[policy]
            for out,beta,scale in [('48',br,100),('24',bv,1)]:
                bias=delta*np.mean(covered*beta)*rho/sdA*scale
                bench.append(dict(policy=policy,fold=fold,outcome=out,n=len(te),benchmark='pre_t0_24h_wind_change',residual_U_SD_kt=sdU,partial_correlation_exposure_U=rho,residual_exposure_SD=sdA,outcome_change_per_SD=float(np.mean(beta)*scale),approximate_policy_bias_one_benchmark=bias))
    pd.DataFrame(cal).to_csv(HERE/'calibration_deciles.csv',index=False);pd.DataFrame(influence).to_csv(HERE/'largest_correction_contributions.csv',index=False);pd.DataFrame(sens).to_csv(HERE/'correction_influence_diagnostics.csv',index=False);pd.DataFrame(bench).to_csv(HERE/'observed_confounder_benchmark.csv',index=False)
    assert all(x['max_prediction_error48']<1e-10 and x['max_prediction_error24']<1e-10 and x['train_indices_exact'] and x['test_indices_exact'] for x in replay)
    (HERE/'saved_model_replay.json').write_text(json.dumps(replay,indent=2))
    # Benchmark-calibrated two-sided omitted-confounder scenario; local approximation, not an E-value.
    b=pd.DataFrame(bench);ss=[]
    for (policy,out),z in b.groupby(['policy','outcome']):
        bias=np.average(z.approximate_policy_bias_one_benchmark,weights=z.n);estimate=float(p.loc[p.policy.eq(policy),'gcomp'+out].mean())*(100 if out=='48' else 1)
        for ka in [.5,1,2,4]:
            for ky in [.5,1,2,4]:
                for sign in [-1,1]:ss.append(dict(policy=policy,outcome=out,exposure_association_multiple=ka,outcome_association_multiple=ky,bias_direction=sign,one_benchmark_bias=abs(bias),bias=sign*abs(bias)*ka*ky,corrected_estimate=estimate-sign*abs(bias)*ka*ky))
    pd.DataFrame(ss).to_csv(HERE/'unmeasured_confounder_benchmark_sensitivity.csv',index=False)
    # Expanded observed-recovery sensitivity: only USA O/V non-synoptic reports, never P/I interpolations.
    raw=f.read(HERE/'raw_cohort_all_timestamps.csv');raw.ISO_TIME=pd.to_datetime(raw.ISO_TIME,utc=True);e.t0=pd.to_datetime(e.t0,utc=True)
    changes=[];extra=e.copy()
    for z in e.itertuples():
        valid=raw[(raw.SID==z.sid)&(raw.ISO_TIME>z.t0)&(raw.ISO_TIME<=z.t0+pd.Timedelta(hours=48))&raw.wind.ge(z.Vref_kt)&raw.IFLAG.fillna('').str[0].isin(['O','V'])]
        if not len(valid):continue
        lag=float((valid.ISO_TIME.min()-z.t0)/pd.Timedelta(hours=1));bin_upper=6*np.ceil(lag/6)
        if lag%6==0:continue
        old_upper=z.U_h
        if not np.isfinite(old_upper) or bin_upper<old_upper:
            L=min(z.L_h,6*np.floor(lag/6));extra.loc[z.Index,'L_h']=L;extra.loc[z.Index,'U_h']=bin_upper;extra.loc[z.Index,'recovery48']=1
            changes.append(dict(event_id=z.event_id,sid=z.sid,original_recovery=z.recovery48,original_U=old_upper,new_L=L,new_U=bin_upper,observed_hit_h=lag,record_UTC=str(valid.ISO_TIME.min()),rule='first-return6h bin bounded by genuine O/V report; never claim exact onset'))
    pd.DataFrame(changes).to_csv(HERE/'nonsynoptic_recovery_changes.csv',index=False)
    edir=HERE/'nonsynoptic_sensitivity';edir.mkdir(exist_ok=True);ep=f.crossfit(extra,edir);ep.to_csv(edir/'predictions.csv',index=False);f.summarize(ep).to_csv(edir/'point_estimates.csv',index=False)
    (HERE/'diagnostics_receipt.json').write_text(json.dumps({'models_replayed':len(replay),'postfit_model_parameters_changed':False,'expanded_report_changes':len(changes),'source_documentation':'https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf','sensitivity_status':'point estimates only; no confidence interval claimed','confounding_approximation':'local linear omitted-variable bias; observed past-wind benchmark, no claim that this is all possible hidden confounding'},indent=2))
    print(json.dumps({'models_replayed':len(replay),'expanded_report_changes':len(changes),'nonsynoptic_estimates':f.summarize(ep).to_dict('records')}),flush=True)
if __name__=='__main__':main()
