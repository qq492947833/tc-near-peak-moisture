from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import brentq
import fit_policies_v6 as f
P=Path(__file__).resolve().parent
def main():
    p=f.read(P/'point_predictions.csv');p.covered=p.covered.astype(str).str.lower().eq('true');e=f.read(P/'cohort_recovered.csv').set_index('event_id');rows=[];tips=[];balance=[];algebra=[]
    for policy,g in p.groupby('policy'):
        a=e.loc[g.event_id];r=g.density_ratio.to_numpy();c=g.covered.to_numpy()
        for name in f.BASE+list(f.POLICIES):
            if name==policy:continue
            z=a[name].to_numpy(float);valid=np.isfinite(z);sd=np.std(z[valid])
            balance.append(dict(policy=policy,baseline=name,n=int(valid.sum()),SMD_density_vs_original=(np.average(z[valid],weights=r[valid])-np.mean(z[valid]))/sd if sd>0 else 0.))
        for out in ['48','24']:
            m=g['m'+out].to_numpy();md=g['m'+out+'_shift'].to_numpy();q=g['g'+out].to_numpy();qd=g['g'+out+'_shift'].to_numpy();scale=100 if out=='48' else 1
            def diff(b):
                # Identity units have exactly identical potential outcomes in this sensitivity.
                if out=='48':changed=md+c*(1-qd)*(expit(logit(np.clip(md,1e-8,1-1e-8))+b)-md)
                else:changed=md+c*(1-qd)*b
                return float(np.mean(changed-m)*scale)
            domain=(-8,8) if out=='48' else (-100,100)
            for b in ([-2,-1,-.5,0,.5,1,2] if out=='48' else [-10,-5,0,5,10]):
                rows.append(dict(policy=policy,outcome=out,offset=b,units='log_odds' if out=='48' else 'kt',scenario='policy_specific_missing_outcome_offset_among_shifted',estimate=diff(b)))
            tip=brentq(diff,*domain) if diff(domain[0])*diff(domain[1])<0 else np.nan
            tips.append(dict(policy=policy,outcome=out,tipping_offset=tip,units='log_odds' if out=='48' else 'kt',tipping_odds_ratio=float(np.exp(tip)) if out=='48' and np.isfinite(tip) else np.nan,search_min=domain[0],search_max=domain[1],shifted_missing_mass=float(np.mean(c*(1-qd)))))
            # Independently expand both means to audit natural-policy cancellation.
            R=g['R'+out].astype(str).str.lower().eq('true').to_numpy();y=g['Y'+out].fillna(0).to_numpy();w=R*np.minimum(1/q,20)
            natural=m+w*(y-m);intervened=md+r*w*(y-m)
            direct=intervened-natural;stored=g['onestep'+out].to_numpy()
            err=float(np.max(np.abs(direct-stored)));assert err<1e-12
            algebra.append(dict(policy=policy,outcome=out,max_natural_intervention_expansion_error=err,natural_mean=float(np.mean(natural)*scale),intervention_mean=float(np.mean(intervened)*scale)))
    pd.DataFrame(rows).to_csv(P/'MNAR_policy_response_sensitivity.csv',index=False);pd.DataFrame(tips).to_csv(P/'MNAR_policy_response_tipping.csv',index=False);pd.DataFrame(balance).to_csv(P/'density_baseline_balance.csv',index=False);pd.DataFrame(algebra).to_csv(P/'onestep_independent_algebra_check.csv',index=False)
    print(pd.DataFrame(tips).to_string(index=False));print(pd.DataFrame(balance).groupby('policy').SMD_density_vs_original.apply(lambda x:x.abs().max()).to_string())
if __name__=='__main__':main()
