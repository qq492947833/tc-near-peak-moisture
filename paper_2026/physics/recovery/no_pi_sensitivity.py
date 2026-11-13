from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import pickle,json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import fit_policies_v6 as f
from diagnostics_v6 import load
P=Path(__file__).resolve().parent
def main():
    e=f.read(P/'cohort_recovered.csv');e['_rowid']=np.arange(len(e));e.complete48=e.complete48.astype(str).str.lower().eq('true');p=f.read(P/'point_predictions.csv')
    for b in ['covered','R48','R24']:p[b]=p[b].astype(str).str.lower().eq('true')
    output=[];dest=P/'no_PI_outcome_models';dest.mkdir(exist_ok=True)
    for policy in f.POLICIES:
        for fold in range(5):
            z=load(P/'models'/f'{policy}_fold{fold}.pkl');tr=e[e.fold.ne(fold)];te=e[e.fold.eq(fold)];pp=p[p.policy.eq(policy)].set_index('event_id').loc[te.event_id].copy();T=z['transform'];B=z['basis'];wt=T.transform(tr);w=T.transform(te)
            Xt=f.design(wt,B,tr[policy].to_numpy());X0=f.design(w,B,pp.A.to_numpy());Xd=f.design(w,B,pp.A_shift.to_numpy())
            j=T.names.index('t0_PIheadroom');keep=np.ones(Xt.shape[1],bool);keep[4+j]=False;keep[4+j+len(T.names)]=False
            h=f.IntervalHazard().fit(Xt[:,keep],tr.L_h.to_numpy()/6,tr.U_h.to_numpy()/6);R=tr.delta24.notna().to_numpy();G=f.proba(z['observation24'],Xt[:,1:]);v=Ridge(alpha=1,fit_intercept=False).fit(Xt[R][:,keep],tr.loc[R,'delta24'],sample_weight=np.minimum(1/G[R],20))
            pp['m48']=h.predict(X0[:,keep]);pp['m48_shift']=h.predict(Xd[:,keep]);pp['m24']=v.predict(X0[:,keep]);pp['m24_shift']=v.predict(Xd[:,keep])
            for out in ['48','24']:
                diff=pp['m'+out+'_shift']-pp['m'+out];res=pp['Y'+out].fillna(0)-pp['m'+out];r=pp.density_ratio
                pp['gcomp'+out]=diff;pp['onestep'+out]=diff+(r-1)*pp['ipcw'+out]*res;pp['onestep'+out+'_uncapped']=diff+(r-1)*pp['ipcw'+out+'_uncapped']*res
            output.append(pp.reset_index());
            with (dest/f'{policy}_fold{fold}.pkl').open('wb') as stream:pickle.dump(dict(original_model=str(P/'models'/f'{policy}_fold{fold}.pkl'),column_mask=keep,hazard=h,wind=v,train_rows=tr._rowid.to_list(),test_rows=te._rowid.to_list()),stream)
    pred=pd.concat(output,ignore_index=True);pred.to_csv(P/'no_PI_fixed_policy_predictions.csv',index=False);f.summarize(pred).to_csv(P/'no_PI_fixed_policy_estimates.csv',index=False)
    # Separate design sensitivity; same target cohort but learned policies can differ.
    f.BASE=[x for x in f.BASE if x!='t0_PIheadroom'];dest2=P/'no_PI_relearned_policy_models';dest2.mkdir(exist_ok=True);new=f.crossfit(e,dest2);new.to_csv(P/'no_PI_relearned_policy_predictions.csv',index=False);f.summarize(new).to_csv(P/'no_PI_relearned_policy_estimates.csv',index=False)
    note={'status':'POINT_SENSITIVITY_COMPLETE','same_policy':'remove concurrent PI only from outcome regressions; original support,density and observation models retained','relearned_policy':'remove PI throughout and re-fit all nuisances; same cohort, different learned shift eligibility; do not call exactly same intervention','physical_dependency':'PI reads same-time temperature and specific-humidity pressure profile, including midtroposphere. RH also depends on that moisture/temperature state. These are concurrent pre-hypothetical-operation measurements, not independently timed causes.','causal_warning':'Sensitivity assesses numerical dependence on adjustment, not proof of no mediator or cross-world ambiguity.','confidence_intervals':'not refit for these point-only specification sensitivities'}
    (P/'no_PI_sensitivity_readme.json').write_text(json.dumps(note,indent=2));print(f.summarize(pred).to_string(index=False));print(f.summarize(new).to_string(index=False))
if __name__=='__main__':main()
