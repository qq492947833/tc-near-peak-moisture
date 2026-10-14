"""Equal-SID environmental path contrasts and conditional saved-draw ranges."""
import json
import numpy as np
import pandas as pd
from .common import repository,read,boolean,close,interval,mean_draws,output,save_table
from .path_functions import stats,slopefit

PROCESSES=["midRH","q700_storm_relative_horizontal_advection","q850_moisture_flux_convergence","shear200_500","PI_headroom"]

def run(out=None):
    s=repository()/"source_data/environmental_paths";out=output("environmental_paths",out)
    expected=read(s/"sensitivity_results.csv.gz");results=[]
    for process in PROCESSES:
        p=s/process;f=read(p/"per_pair_support_and_values.csv.gz")
        original=json.loads((p/"original/summary.json").read_text("utf-8-sig"))
        good=boolean(f.original_process_valid);strict=boolean(f.strict_process_valid)
        all_sid=f[good].groupby("sid").difference_after_minus_before.mean().sort_index()
        strict_sid=f[strict].groupby("sid").difference_after_minus_before.mean().sort_index()
        common=all_sid.reindex(strict_sid.index)
        assert (f.loc[strict,"wind_kt_pre"]==f.loc[strict,"wind_kt_post"]).all()
        with np.load(p/"original/raw_draws.npz") as z:
            assert all_sid.index.tolist()==z["sid"].astype(str).tolist()
            close(all_sid,z["per_sid"])
            raw=mean_draws(all_sid,z["sampled_sid_indices"]);close(raw,z["draw"])
            lo,hi=interval(raw);close([all_sid.mean(),lo,hi],[original["raw"][k] for k in ["estimate","lo","hi"]])
        with np.load(p/"raw_common_draws.npz") as z:
            assert strict_sid.index.tolist()==z["sid"].astype(str).tolist()
            vals={"original_commonSID":common.to_numpy(),"strict_equal_wind":strict_sid.to_numpy(),"direct_strict_minus_common":(strict_sid-common).to_numpy()}
            keys={"original_commonSID":"original_common","strict_equal_wind":"strict","direct_strict_minus_common":"direct_strict_minus_common"}
            for representation,y in vals.items():
                draws=mean_draws(y,z["sampled_sid_indices"]);close(draws,z[keys[representation]])
                target=expected[(expected.process==process)&(expected.analysis=="mean")&(expected.representation==representation)].iloc[0]
                lo,hi=interval(draws);close([y.mean(),lo,hi],[target.estimate,target.ci_low,target.ci_high])
                results.append(dict(process=process,analysis="mean",representation=representation,metric=target.metric,estimate=y.mean(),ci_low=lo,ci_high=hi,n_sid=len(y),units=target.units,attempted=len(draws),failed=0))
        with np.load(p/"original/slope_design_and_draws.npz") as z:
            X=z["X"];y=z["y"];sid=z["sid"].astype(str);valid=z["input_valid"]
            assert np.array_equal(valid,boolean(f.original_slope_valid))
            original_st=stats(sid[valid],X[valid],y[valid]);po,_,_=slopefit(original_st)
            close(po[0],original["continuous_slope"]["estimate"],1e-9,1e-8)
            # Original saved slope draws are per unit A; the published contrast is per 0.1 A.
            close(interval(z["draw"]*.1),[original["continuous_slope"]["lo"],original["continuous_slope"]["hi"]])
            fits={}
            for label,col in [("original_commonSID","original_slope_commonSID"),("strict_equal_wind","strict_slope_valid")]:
                mask=boolean(f[col]);st=stats(sid[mask],X[mask],y[mask]);point,_,_=slopefit(st);fits[label]=point
            fits["direct_strict_minus_common"]=np.asarray(fits["strict_equal_wind"][:3])-np.asarray(fits["original_commonSID"][:3])
        with np.load(p/"slope_common_draws.npz") as z:
            keys={"original_commonSID":"original_common","strict_equal_wind":"strict","direct_strict_minus_common":"direct"}
            close(z["strict"]-z["original_common"],z["direct"])
            for rep,point in fits.items():
                for j,metric in enumerate(["A_slope","incremental_R2","partial_R2"]):
                    target=expected[(expected.process==process)&(expected.analysis=="slope")&(expected.representation==rep)&(expected.metric==metric)].iloc[0]
                    draws=z[keys[rep]][:,j];lo,hi=interval(draws);close([point[j],lo,hi],[target.estimate,target.ci_low,target.ci_high],1e-8,1e-8)
                    results.append(dict(process=process,analysis="slope",representation=rep,metric=metric,estimate=point[j],ci_low=lo,ci_high=hi,n_sid=len(z["sid"]),units=target.units,attempted=len(draws),failed=int((~np.isfinite(draws)).sum())))
        # Original complete contrasts remain side-by-side with the strict restriction.
        old=expected[(expected.process==process)&expected.representation.isin(["original_all_reused","original_all_point"])]
        results+=old.to_dict("records")
    result=pd.DataFrame(results);save_table(result,out/"all_path_contrasts.csv")
    compact=read(s/"support/background_target_support.csv.gz");rows=[]
    for path in sorted((s/"support").glob("*/summary.json")):
        summary=json.loads(path.read_text());d=read(path.parent/"draw_diagnostics.csv.gz")
        assert len(d)==2000 and d.draw.nunique()==2000
        counts=d.classification.value_counts().to_dict();assert counts==summary["classification_counts"]
        accepted=int(d.classification.eq("success_joint_support_and_estimable").sum())
        for job in summary["jobs"]:
            target=compact[compact.job==job["job"]].iloc[0]
            assert accepted==target.accepted and len(d)-accepted==target.failed_joint_support
            rows.append(dict(job=job["job"],process=job["process"],attempted=len(d),accepted=accepted,
                             failed_joint_support=len(d)-accepted,
                             rank_deficient_but_target_estimable=int(((d.outcome_design_rank<d.outcome_design_columns)&boolean(d.phase_target_estimable)).sum()),
                             adjusted_point=target.adjusted_point,conditional_p025=target.conditional_p025,
                             conditional_p975=target.conditional_p975,interval_status=target.interval_status))
    save_table(pd.DataFrame(rows),out/"background_conditional_support.csv")
    return result
