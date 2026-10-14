"""Stage environment on the original physical support and original SID draws."""
import numpy as np
import pandas as pd
from .common import repository,read,close,interval,output,save_table
from .stage_functions import make_pair

def run(out=None):
    s=repository()/"source_data/environment";out=output("environment",out)
    stage=read(s/"raw_stage_sid.csv.gz");reference=read(s/"raw_stage_summary.csv.gz")
    records=[]
    for z in reference.itertuples():
        a=stage[(stage.variable==z.variable)&(stage.stage==z.stage)]
        close(a.estimate.mean(),z.estimate);assert len(a)==z.n_sid
        records.append(dict(variable=z.variable,stage=z.stage,unit=z.unit,estimate=a.estimate.mean(),n_sid=len(a),
                            covered_h=a.covered_h.sum(),ci_low=z.ci_low,ci_high=z.ci_high,
                            interval_source="saved original stage bootstrap; point reaggregated"))
    save_table(pd.DataFrame(records),out/"stage_summary.csv")
    expected=read(s/"raw_adjacent_contrasts.csv.gz")
    oldpairs=read(s/"raw_adjacent_sid.csv.gz")
    order=read(s/"bootstrap_sid_order.csv.gz").sid.tolist()
    weights=np.load(s/"bootstrap_sid_multiplicities.npy")
    order_index={sid:i for i,sid in enumerate(order)}
    original_draws=read(s/"raw_adjacent_bootstrap.csv.gz")
    rows=[]
    for z in expected.itertuples():
        frm,to=z.contrast.split("-")[1],z.contrast.split("-")[0]
        pair=make_pair(stage[stage.variable==z.variable],frm,to,[])
        original=oldpairs[(oldpairs.variable==z.variable)&(oldpairs.contrast==z.contrast)].set_index("sid").loc[pair.sid]
        close(pair.delta,original.delta)
        sub=weights[:,[order_index[x] for x in pair.sid]].astype(float)
        draws=(sub@pair.delta.to_numpy())/sub.sum(1)
        olddraw=original_draws[(original_draws.variable==z.variable)&(original_draws.contrast==z.contrast)].sort_values("draw")
        close(draws,olddraw.estimate)
        lo,hi=interval(draws);close([pair.delta.mean(),lo,hi],[z.estimate,z.ci_low,z.ci_high])
        rows.append(dict(variable=z.variable,contrast=z.contrast,unit=z.unit,estimate=pair.delta.mean(),ci_low=lo,ci_high=hi,n_sid=len(pair),draws=len(draws)))
    raw=pd.DataFrame(rows);save_table(raw,out/"adjacent_stage_contrasts.csv")
    draws=read(s/"adjustment_bootstrap_all.csv.gz")
    target=read(s/"adjusted_adjacent_summary.csv.gz"); adjusted=[]
    for z in target.itertuples():
        g=draws[(draws.variable==z.variable)&(draws.scheme==z.scheme)&(draws.contrast==z.contrast)]
        lo,hi=interval(g.estimate);close([lo,hi],[z.ci_low,z.ci_high])
        valid=int(np.isfinite(g.estimate).sum())
        assert len(g)==z.bootstrap_attempted and valid==z.bootstrap_valid and len(g)-valid==z.bootstrap_failed
        adjusted.append(dict(variable=z.variable,scheme=z.scheme,contrast=z.contrast,estimate=z.estimate,
          ci_low=lo,ci_high=hi,n_sid=z.n_pairs,attempted=len(g),accepted=valid,failed=len(g)-valid,
          interval_scope="conditional on accepted fixed draws" if valid<len(g) else "original fixed SID percentile draws",
          point_source="saved model estimate; no new adjustment fit"))
    save_table(pd.DataFrame(adjusted),out/"adjusted_stage_contrasts.csv")
    return raw

