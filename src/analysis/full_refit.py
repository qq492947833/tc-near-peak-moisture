"""Optional full cross-fitted model calculation; never called by quick notebooks."""
from pathlib import Path
import json
import numpy as np
from .common import repository,read,boolean

def recovery_refit(destination,*,confirm_full=False,scenarios=False):
    """Refit on the saved cohort and saved SID identities; may take many hours."""
    if not confirm_full:
        raise ValueError("Set confirm_full=True explicitly. Quick reproduction uses saved predictions and refits.")
    out=Path(destination).resolve()
    repo=repository()
    if out.exists(): raise FileExistsError("Choose a new output directory.")
    if str(out).lower().startswith(str(repo/"source_data").replace("\\\\?\\","").lower()):
        raise ValueError("Output cannot overwrite source_data.")
    out.mkdir(parents=True)
    from . import recovery_frozen as f
    e=read(repo/"source_data/recovery/primary/cohort_recovered.csv.gz")
    e["_rowid"]=np.arange(len(e));e.complete48=boolean(e.complete48);e.fold=e.fold.astype(int)
    ids=np.load(repo/"source_data/recovery/primary/bootstrap_indices.npz")["indices"]
    assert ids.shape==(1000,2371)
    if not scenarios:
        (out/"models").mkdir()
        pred=f.crossfit(e,out/"models");pred.to_csv(out/"point_predictions.csv",index=False)
        f.summarize(pred).to_csv(out/"point_estimates.csv",index=False)
        with (out/"fixed_draws.jsonl").open("w",encoding="utf-8") as stream:
            for draw,index in enumerate(ids):
                try:
                    p=f.crossfit(e.iloc[index].copy())
                    row=dict(draw=draw,status="OK",rows=f.summarize(p).to_dict("records"))
                except Exception as error:
                    row=dict(draw=draw,status="FAILED",error=type(error).__name__+": "+str(error))
                stream.write(json.dumps(row)+"\n");stream.flush()
    else:
        from . import recovery_scenario_functions as m
        m.HERE=out
        sets={}
        for name in m.SCENARIOS:
            q=read(repo/"source_data/recovery/scenarios"/(name+"_cohort.csv.gz"))
            q["R_scenario"]=boolean(q.R_scenario);q.complete48=boolean(q.complete48)
            sets[name]=q
        point,optimizers=m.estimate_set(e,sets,point=True)
        (out/"point_estimates.json").write_text(json.dumps(point),encoding="utf-8")
        with (out/"fixed_draws.jsonl").open("w",encoding="utf-8") as stream:
            for draw,index in enumerate(ids):
                try:
                    estimates,details=m.estimate_set(e,sets,indices=index)
                    row=dict(draw=draw,estimates=estimates,optimizers=details)
                except Exception as error:
                    row=dict(draw=draw,status="FAILED",error=type(error).__name__+": "+str(error))
                stream.write(json.dumps(row)+"\n");stream.flush()
    return out

