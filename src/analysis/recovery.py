"""Reproduce recovery contrasts from saved cross-fitted predictions and fixed refits."""
import numpy as np
import pandas as pd
from .common import repository,read,boolean,close,interval,output,save_table

def _scores(g,h):
    y=g["Y"+h].fillna(0).to_numpy(float);R=boolean(g["R"+h])
    p=g["g"+h].to_numpy(float);m=g["m"+h].to_numpy(float)
    md=g["m"+h+"_shift"].to_numpy(float);ratio=g.density_ratio.to_numpy(float)
    capped=R*np.minimum(1/p,20);uncapped=R/p
    return {"gcomp":(m,md),"onestep":(m+capped*(y-m),md+ratio*capped*(y-m)),
            "ipw":(capped*y,ratio*capped*y),
            "onestep_uncapped":(m+uncapped*(y-m),md+ratio*uncapped*(y-m))}

def run(out=None):
    s=repository()/"source_data/recovery";out=output("recovery",out)
    pred=read(s/"primary/point_predictions.csv.gz")
    expected=read(s/"primary/effects_with_CI.csv.gz")
    draws=read(s/"primary/bootstrap_draws.csv.gz")
    cohort=read(s/"primary/cohort_recovered.csv.gz")
    identities=np.load(s/"primary/bootstrap_indices.npz")["indices"]
    assert identities.shape==(1000,len(cohort)) and len(cohort)==2371 and cohort.sid.is_unique
    rows=[]
    for policy,g in pred.groupby("policy"):
        assert len(g)==2371 and g.sid.is_unique and set(g.sid)==set(cohort.sid)
        for h in ["48","24"]:
            outcome="recovery48_pp" if h=="48" else "delta24_kt";scale=100 if h=="48" else 1
            for method,(natural,shifted) in _scores(g,h).items():
                contrast=shifted-natural;col=method+h if method!="onestep_uncapped" else "onestep"+h+"_uncapped"
                close(contrast,g[col]);z=expected[(expected.policy==policy)&(expected.method==method)&(expected.outcome==outcome)].iloc[0]
                d=draws[(draws.policy==policy)&(draws.method==method)&(draws.outcome==outcome)].sort_values("draw")
                assert len(d)==1000 and np.array_equal(d.draw.to_numpy(),np.arange(1000))
                lo,hi=interval(d.estimate);close([contrast.mean()*scale,lo,hi],[z.estimate,z.ci_low,z.ci_high])
                rows.append(dict(scenario="primary",policy=policy,outcome=outcome,method=method,n=len(g),
                  natural_mean=natural.mean()*scale,policy_mean=shifted.mean()*scale,estimate=contrast.mean()*scale,
                  ci_low=lo,ci_high=hi,attempted=len(d),accepted=int(d.estimate.notna().sum()),
                  natural_policy_interval="not saved for primary; contrast interval from original refits"))
    baseline=pd.DataFrame(rows);save_table(baseline,out/"primary_all_estimators.csv")
    target=read(s/"scenarios/scenario_estimates_with_intervals.csv.gz")
    d=read(s/"scenarios/scenario_bootstrap_draws.csv.gz");rows=[]
    for (scenario,policy),group in target.groupby(["scenario","policy"]):
        g=read(s/"scenarios"/(scenario+"_"+policy+"_predictions.csv.gz"))
        assert len(g)==2371 and g.sid.is_unique and set(g.sid)==set(cohort.sid)
        y=g.Y.fillna(0).to_numpy();R=boolean(g.R);prob=g.g.to_numpy();m=g.m.to_numpy();md=g.md.to_numpy();r=g.ratio.to_numpy()
        cap=R*np.minimum(1/prob,20);uncap=R/prob
        scores={"gcomp":(m,md),"onestep":(m+cap*(y-m),md+r*cap*(y-m)),
                "ipw":(cap*y,r*cap*y),"onestep_uncapped":(m+uncap*(y-m),md+r*uncap*(y-m))}
        for z in group.itertuples():
            a,b=scores[z.method];close(a,g["natural_"+z.method]);close(b,g["policy_"+z.method])
            means=[a.mean()*100,b.mean()*100,(b-a).mean()*100]
            close(means,[z.natural_risk_pct,z.policy_risk_pct,z.contrast_pp])
            selected=d[(d.scenario==scenario)&(d.policy==policy)&(d.method==z.method)].sort_values("draw")
            assert len(selected)==1000 and selected.status.eq("OK").all() and np.array_equal(selected.draw,np.arange(1000))
            row=dict(scenario=scenario,policy=policy,method=z.method,n=len(g),attempted=1000,accepted=1000,failed=0)
            for field,value in zip(["natural_risk_pct","policy_risk_pct","contrast_pp"],means):
                lo,hi=interval(selected[field]);close([lo,hi],[getattr(z,field+"_ci_low"),getattr(z,field+"_ci_high")])
                row[field]=value;row[field+"_ci_low"]=lo;row[field+"_ci_high"]=hi
            rows.append(row)
    scenarios=pd.DataFrame(rows);save_table(scenarios,out/"all_observation_scenarios.csv")
    return baseline,scenarios

