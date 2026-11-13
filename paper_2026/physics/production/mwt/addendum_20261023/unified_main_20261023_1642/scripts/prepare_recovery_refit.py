from _release_paths import release_path as _release_path
"""Prepare exact saved-input subset for the unchanged v62 recovery estimator."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,shutil
import numpy as np
import pandas as pd
U=Path(str(_release_path('mwt','addendum_20261023/unified_main_20261023_1642')))
OLD=Path(str(_release_path('project','03_Work/v62/causal')))
OUT=U/"recovery_refit"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,keep_default_na=False,na_values=["","nan","NaN"])
def main():
    freeze=json.loads((U/"reports/main_sid_freeze_receipt.json").read_text(encoding="utf-8"))
    main_path=U/"source_data/main_sid.txt"
    assert sha(main_path)==freeze["main_sid_sha256"] and freeze["main_sid_n"]==2340
    main_sid=set(main_path.read_text().split())
    source=OLD/"cohort_recovered.csv"; code=OLD/"fit_policies_v6.py"
    old=read(source); new=old[old.sid.isin(main_sid)].copy()
    assert old.sid.is_unique and new.sid.is_unique and len(new)==2231
    assert set(new.fold.astype(int))==set(range(5))
    events_path=U/"source_data/historical_first_events_with_retained_outcomes.parquet"
    events=pd.read_parquet(events_path).set_index("sid").loc[new.sid]
    assert events.index.tolist()==new.sid.tolist()
    assert np.array_equal(events.event_id.to_numpy(),new.event_id.to_numpy())
    assert np.array_equal(pd.to_datetime(events.t0,utc=True).array.asi8,pd.to_datetime(new.t0,utc=True).array.asi8)
    for col in ["Vref_kt","wind_t0_kt","Vmax_past_kt","threshold_gap_kt"]:
        assert np.allclose(events[col].to_numpy(float),new[col].to_numpy(float),rtol=0,atol=1e-12,equal_nan=True),col
    OUT.mkdir(exist_ok=True)
    for name in ["cohort_recovered.csv","fit_policies_v6.py"]:
        if (OUT/name).exists():raise RuntimeError("Refuse to overwrite prepared input "+name)
    with source.open("r",encoding="utf-8-sig",newline="") as f,(OUT/"cohort_recovered.csv").open("w",encoding="utf-8",newline="") as g:
        reader=csv.reader(f); writer=csv.writer(g); header=next(reader);writer.writerow(header);ix=header.index("sid")
        for row in reader:
            if row[ix] in main_sid:writer.writerow(row)
    shutil.copyfile(code,OUT/"fit_policies_v6.py")
    assert sha(code)==sha(OUT/"fit_policies_v6.py")
    loaded=read(OUT/"cohort_recovered.csv")
    pd.testing.assert_frame_equal(new.reset_index(drop=True),loaded,check_exact=True)
    assert np.array_equal(loaded.complete48.astype(str),new.complete48.astype(str))
    loaded[["sid","event_id","fold"]].to_csv(OUT/"point_fold_membership.csv",index=False)
    pd.DataFrame({"sid":sorted(main_sid-set(new.sid)),"reason":"Not present in the accepted measured-environment recovery model input; retained in main cohort"}).to_csv(OUT/"main_sid_without_accepted_model_inputs.csv",index=False)
    complete=loaded.complete48.astype(str).str.lower().eq("true")
    def ycounts(y):
        x=pd.to_numeric(y,errors="coerce")
        return {"observed_one":int(x.eq(1).sum()),"observed_zero":int(x.eq(0).sum()),"unknown":int(x.isna().sum())}
    receipt={
        "status":"READY_EXACT_SAVED_INPUT_SUBSET",
        "UTC":datetime.now(timezone.utc).isoformat(),
        "main_sid_n":len(main_sid),"main_sid_sha256":sha(main_path),
        "main_sid_path":str(main_path),"original_model_sid_n":len(old),"model_sid_n":len(loaded),
        "excluded_original_model_sid_n":int((~old.sid.isin(main_sid)).sum()),
        "main_sid_without_model_input_n":len(main_sid-set(loaded.sid)),
        "model_input_filter":"sid membership only; no follow-up survival/completeness filtering",
        "event_id_t0_historical_threshold_identity_verified":True,
        "all_accepted_input_values_exactly_retained":True,
        "recovery48":ycounts(loaded.recovery48),"legacy_return96":ycounts(loaded.return96),
        "delta24_observed":int(loaded.delta24.notna().sum()),"delta24_unknown":int(loaded.delta24.isna().sum()),
        "complete48":int(complete.sum()),"incomplete48_retained":int((~complete).sum()),
        "observed_return_with_incomplete48_retained":int((loaded.recovery48.eq(1)&~complete).sum()),
        "interval_events":int(np.isfinite(loaded.U_h.to_numpy(float)).sum()),
        "interval_censored_events":int((np.isfinite(loaded.U_h.to_numpy(float))&((loaded.U_h-loaded.L_h)>6)).sum()),
        "right_censored":int((~np.isfinite(loaded.U_h.to_numpy(float))).sum()),
        "zero_information_right_censored":int((~np.isfinite(loaded.U_h.to_numpy(float))&loaded.L_h.eq(0)).sum()),
        "missing_reason_counts":loaded.missing_reasons.fillna("none").value_counts().to_dict(),
        "fold_test_sid_n":loaded.groupby("fold").sid.nunique().to_dict(),
        "fold_train_sid_n":{str(k):int(loaded.fold.ne(k).sum()) for k in range(5)},
        "seed":610102026,"fixed_bootstrap_draws":1000,"policy_branches":2,"folds":5,
        "method":"unchanged v62 48-hour interval-hazard g-formula, one-step, density diagnostics and delta24",
        "code_byte_identical_to_accepted_v62":True,
        "source_inputs":{str(p):sha(p) for p in [code,source,events_path,main_path,U/"reports/main_sid_freeze_receipt.json"]},
        "prepared_inputs":{name:sha(OUT/name) for name in ["fit_policies_v6.py","cohort_recovered.csv","point_fold_membership.csv"]},
        "bootstrap_note":"same original seed and number on the new SID population; old-population bootstrap arrays and fitted predictions cannot be reused",
        "execution":"CPU only; thread pools limited to one; below-normal process priority; no raw ERA5 reads"
    }
    (OUT/"input_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ["source_inputs","prepared_inputs","missing_reason_counts"]},ensure_ascii=False))
if __name__=="__main__":main()

