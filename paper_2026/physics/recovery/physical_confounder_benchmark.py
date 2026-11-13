import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import fit_policies_v6 as f
from diagnostics_v6 import load
P=Path(__file__).resolve().parent
def main():
    e=f.read(P/'cohort_recovered.csv');p=f.read(P/'point_predictions.csv');p.covered=p.covered.astype(str).str.lower().eq('true');records=[];trainweights=[]
    for policy in f.POLICIES:
        prior='past_shear' if policy=='shear200_850' else 'past_RH'
        for fold in range(5):
            model=load(P/'models'/f'{policy}_fold{fold}.pkl');tr=e[e.fold.ne(fold)];te=e[e.fold.eq(fold)];pp=p[p.policy.eq(policy)].set_index('event_id').loc[te.event_id]
            T=model['transform'];wt=T.transform(tr);w=T.transform(te);X=f.design(w,model['basis'],te[policy].to_numpy());Xt=f.design(wt,model['basis'],tr[policy].to_numpy())
            for outcome,obs in [('24',model['observation24']),('48',model['observation48'])]:
                G=f.proba(obs,Xt[:,1:]);R=tr.delta24.notna().to_numpy() if outcome=='24' else tr.complete48.astype(str).str.lower().eq('true').to_numpy()
                for i,(_,z) in enumerate(tr.iterrows()):trainweights.append(dict(policy=policy,fold=fold,event_id=z.event_id,sid=z.sid,outcome=outcome,observed=bool(R[i]),G=G[i],uncapped=1/G[i],capped=min(1/G[i],20),used_in_outcome_fit=bool(R[i]) if outcome=='24' else False))
            for benchmark in ['delta_v24_backward_kt',prior,'current_land_contact']:
                j=T.names.index(benchmark);keep=np.ones(wt.shape[1],bool);keep[j]=False;keep[j+len(T.names)]=False
                A=tr[policy].to_numpy();U=tr[benchmark].fillna(tr[benchmark].median()).to_numpy();W=wt[:,keep]
                ar=A-Ridge(alpha=1).fit(W,A).predict(W);ur=U-Ridge(alpha=1).fit(W,U).predict(W)
                rho=float(np.corrcoef(ar,ur)[0,1]);sA=float(ar.std());sU=float(ur.std())
                if benchmark=='current_land_contact':
                    v1=te.copy();v0=te.copy();v1[benchmark]=1;v0[benchmark]=0
                    X1=f.design(T.transform(v1),model['basis'],te[policy].to_numpy());X0=f.design(T.transform(v0),model['basis'],te[policy].to_numpy());dR=(model['hazard'].predict(X1)-model['hazard'].predict(X0))*sU;dV=(model['wind'].predict(X1)-model['wind'].predict(X0))*sU
                else:
                    alt=te.copy();alt[benchmark]=alt[benchmark]+sU;X1=f.design(T.transform(alt),model['basis'],te[policy].to_numpy());dR=model['hazard'].predict(X1)-model['hazard'].predict(X);dV=model['wind'].predict(X1)-model['wind'].predict(X)
                for out,der,scale in [('48',dR,100),('24',dV,1)]:
                    bias=f.POLICIES[policy]*np.mean(pp.covered.to_numpy()*der)*rho/sA*scale
                    records.append(dict(policy=policy,fold=fold,outcome=out,benchmark=benchmark,n=len(te),residual_U_SD=sU,partial_correlation_A_U=rho,residual_A_SD=sA,outcome_association_per_residual_SD=float(np.mean(der)*scale),signed_bias_one_benchmark=float(bias)))
    b=pd.DataFrame(records);b.to_csv(P/'physical_confounder_benchmarks.csv',index=False);pd.DataFrame(trainweights).to_csv(P/'training_observation_weights.csv',index=False)
    totals=[];scenarios=[]
    for (policy,out,benchmark),g in b.groupby(['policy','outcome','benchmark']):
        bias=float(np.average(g.signed_bias_one_benchmark,weights=g.n));est=float(p.loc[p.policy.eq(policy),'gcomp'+out].mean())*(100 if out=='48' else 1);strength=abs(est/bias) if bias else np.inf
        totals.append(dict(policy=policy,outcome=out,benchmark=benchmark,mean_signed_bias_one_benchmark=bias,point_effect=est,product_strength_to_null=strength,balanced_strength_each_to_null=np.sqrt(strength)))
        for strengthA in [.5,1,2,4]:
            for strengthY in [.5,1,2,4]:
                for sign in [-1,1]:scenarios.append(dict(policy=policy,outcome=out,benchmark=benchmark,exposure_strength=strengthA,outcome_strength=strengthY,sign=sign,bias=sign*abs(bias)*strengthA*strengthY,adjusted_effect=est-sign*abs(bias)*strengthA*strengthY))
    pd.DataFrame(totals).to_csv(P/'physical_confounder_summary.csv',index=False);pd.DataFrame(scenarios).to_csv(P/'physical_confounder_sensitivity.csv',index=False)
    print(pd.DataFrame(totals).to_string(index=False))
if __name__=='__main__':main()
