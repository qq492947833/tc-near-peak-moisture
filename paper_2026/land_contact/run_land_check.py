"""Reproduce the retrospective land-contact filter and both sets of six contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from classify_track import TrackAudit
from estimators import bootstrap_summary, raw_calc

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "inputs"
REF = ROOT / "reference"
WATER = ["gross_outward_difference", "outward_humidity_contribution",
         "outward_relative_motion_contribution", "net_inward_difference"]
ENERGY = {"APE": "J kg-1", "C_A_to_K": "W kg-1"}


def read(path):
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")


def accepted(left, right):
    return left.eq("no_contact") & right.eq("no_contact")


def classify():
    windows = read(DATA / "event_windows.csv")
    archive = read(REF / "event_states.csv").set_index(["sid", "event_id"])
    members = read(DATA / "water_members.csv")
    tracks = TrackAudit(windows.sid.unique())
    rows = []
    delta = pd.Timedelta(hours=24)
    for row in windows.itertuples(index=False):
        t0, trough, returned, final = pd.to_datetime(
            [row.temporary_departure_utc, row.trough_utc, row.return_utc, row.final_departure_utc], utc=True)
        intervals = {"temp_window": (t0-delta, t0+delta), "final_window": (final-delta, final+delta),
                     "full_recovery": (t0, returned)}
        if row.energy_original_member:
            intervals.update(energy_rise=(trough, returned), energy_decay=(final, pd.to_datetime(row.energy_decay_end_UTC, utc=True)))
        record = {"sid": row.sid, "event_id": row.event_id}
        for label, (start, stop) in intervals.items():
            state = tracks.window(row.sid, start, stop)
            record[label+"_status"] = state["status"]
            if label == "full_recovery":
                record["full_recovery_land_to_sea"] = state["land_to_sea"]
        rows.append(record)
    events = windows.merge(pd.DataFrame(rows), on=["sid", "event_id"], validate="one_to_one")
    events["water_retained"] = events.water_original_member & accepted(events.temp_window_status, events.final_window_status)
    events["energy_retained"] = events.energy_original_member & accepted(events.energy_rise_status, events.energy_decay_status)
    actual = events.set_index(["sid", "event_id"]).loc[archive.index]
    for key in ["temp_window_status", "final_window_status", "full_recovery_status", "full_recovery_land_to_sea",
                "energy_rise_status", "energy_decay_status"]:
        np.testing.assert_array_equal(actual[key].fillna(""), archive[key])
    for domain in ("water", "energy"):
        mask=actual[domain+"_original_member"]
        np.testing.assert_array_equal(actual.loc[mask, domain+"_retained"].astype(str), archive.loc[mask, domain+"_retained"].astype(str))
    member_rows=[]
    for member in members.itertuples(index=False):
        start = pd.to_datetime(member.t0, utc=True)
        five = pd.date_range(start-delta, start, freq="6h")
        distances = tracks.tracks[member.sid].point.reindex(five)
        past24 = float((distances==0).any()) if distances.notna().all() else np.nan
        member_rows.append({"sid":member.sid, "event_id":member.event_id, "historical_from_five_fixes":past24,
                            "status":tracks.window(member.sid, start-delta, start+delta)["status"]})
    members = members.merge(pd.DataFrame(member_rows),on=["sid","event_id"],validate="one_to_one")
    np.testing.assert_array_equal(members.historical_land_contact, members.historical_from_five_fixes)
    s3 = members[members.s3_adjusted_member]
    assert len(s3)==108 and s3.sid.nunique()==54 and s3.historical_from_five_fixes.eq(0).all()
    water_events = events[events.water_original_member].set_index("sid")
    water_states = members.pivot(index="sid",columns="event_class",values="status").loc[water_events.index]
    np.testing.assert_array_equal(water_states.internal, water_events.temp_window_status)
    np.testing.assert_array_equal(water_states.final, water_events.final_window_status)
    return tracks,events,members


def water_results(events):
    values = read(DATA / "water_values_kg.csv").set_index("sid").sort_index()
    draw_rule = json.loads((DATA / "resampling.json").read_text(encoding="utf8"))
    water = events[events.water_original_member]
    assert water.sid.is_unique and set(values.index)==set(water.sid)
    results = []
    for sample,ids in [("original",water.sid),("restricted",water.loc[water.water_retained,"sid"])]:
        v = values.loc[sorted(ids)]
        rng = np.random.default_rng()
        rng.bit_generator.state = draw_rule["water_original_bank_initial_state"]
        ix = rng.integers(0, len(v), (draw_rule["draws"],len(v)))
        if sample=="original":
            assert hashlib.sha256(ix.astype("<i8").tobytes()).hexdigest()==draw_rule["water_original_bank_little_endian_int64_sha256"]
        for col in WATER:
            s = bootstrap_summary(v[col].to_numpy()/1e12, ix)
            assert s["status"]=="COMPUTED"
            results.append(dict(metric=col, unit="10^12 kg", sample=sample, estimate=s["estimate"], ci_low=s["low"],
                                ci_high=s["high"], n_sid=len(v), n_return_events=len(v), n_units=len(v), valid_draws=len(ix)))
    return results


def energy_results(events):
    pair = read(DATA / "energy_wind_bin_pairs.csv")
    assert pair.pair_id.is_unique
    stages = events[events.energy_original_member][["sid","event_id","energy_retained"]]
    p = pair.merge(stages,on=["sid","event_id"],validate="many_to_one",how="left")
    assert p.energy_retained.notna().all() and p.event_id.nunique()==len(stages)
    results=[]
    for sample, grp in [("original",p),("restricted",p[p.energy_retained])]:
        for col,unit in ENERGY.items():
            s,_ = raw_calc(grp.sid.to_numpy(),grp.event_id.to_numpy(),grp["decay_"+col].to_numpy(),grp["rise_"+col].to_numpy())
            results.append(dict(metric=col,unit=unit,sample=sample,estimate=s["estimate"],ci_low=s["lower"],
                                ci_high=s["upper"],n_sid=s["n_sid"],n_return_events=s["n_events"],
                                n_units=s["n_bin_pairs"],valid_draws=s["valid_bootstrap"]))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory for replayed outputs")
    out = parser.parse_args().out.resolve()
    inputs = json.loads((ROOT/"file_hashes.json").read_text(encoding="utf8"))
    for name, h in inputs.items():
        with (ROOT/name).open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest()==h,name
    if out in (ROOT,DATA,REF):
        raise ValueError("Choose a separate output directory.")
    out.mkdir(parents=True,exist_ok=True)
    track,events,members = classify()
    stats = pd.DataFrame(water_results(events)+energy_results(events))
    result=stats.pivot(index=["metric","unit"],columns="sample")
    result.columns=[f"{sample}_{col}" for col,sample in result.columns]
    reference=read(REF/"core_estimates.csv").set_index(["metric","unit"])
    result=result.loc[reference.index,reference.columns]
    # Verify unrounded decimal-to-binary round trips for all 36 estimates and
    # interval endpoints, both samples' support and the number of valid draws.
    np.testing.assert_array_equal(result.to_numpy(),reference.to_numpy())
    both=events.loc[events.energy_original_member]
    w=events.loc[events.water_original_member]
    assert len(events)==133 and events.sid.nunique()==123
    assert (len(w),int(w.water_retained.sum()))==(80,75)
    assert (len(both),int(both.energy_retained.sum()))==(115,74)
    result.to_csv(out/"core_estimates.csv")
    events.to_csv(out/"event_windows.csv",index=False)
    members.to_csv(out/"water_members.csv",index=False)
    precision_cols=[c for c in result if c.endswith(("estimate","ci_low","ci_high"))]
    record={"release":"v2.1.6", "base_science":"v2.1.5","python":sys.version.split()[0],
            "numpy":np.__version__, "pandas":pd.__version__, "native_reports":sum(map(len,track.tracks.values())),
            "return_events":len(events),"return_SIDs":events.sid.nunique(), "water_pairs":[len(w),int(w.water_retained.sum())],
            "energy_SIDs":[int(both.sid.nunique()),int(both[both.energy_retained].sid.nunique())],
            "energy_events":[len(both),int(both.energy_retained.sum())],
            "energy_wind_bin_pairs":[int(result.loc[("APE","J kg-1"),col]) for col in ("original_n_units","restricted_n_units")],
            "past24_covariates_reproduced":len(members),"table_S3_adjusted_water_pairs":int(members[members.s3_adjusted_member].sid.nunique()),
            "bootstrap_draws_per_target_sample":2000,"exact_estimate_and_interval_cells":result[precision_cols].size,
            "max_abs_difference":float((result[precision_cols]-reference[precision_cols]).abs().max().max())}
    (out/"run.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf8")
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()

