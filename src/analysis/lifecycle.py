"""Lifecycle summaries from fixed processed track records; no new storm selection."""
import numpy as np
import pandas as pd
from .common import repository,read,boolean,close,output,save_table

def _episodes(mask,utc):
    ii=np.flatnonzero(mask)
    return int(len(ii)>0)+int(np.sum((np.diff(ii)!=1)|(np.diff(utc.asi8[ii])!=21600*10**9))) if len(ii) else 0

def run(out=None):
    source=repository()/"source_data/lifecycle"
    sid=read(source/"analysis_per_SID_main_window.csv.gz")
    track=read(source/"analysis_records.csv.gz")
    track=track[track.sid.isin(sid.sid)].copy()
    timecols=["genesis_time","escalation_time","maturity_time","resolution_time","termination_time","lmi_time","lmi_last_time"]
    for c in timecols:sid[c]=pd.to_datetime(sid[c],utc=True)
    track["UTC"]=pd.to_datetime(track.UTC,utc=True)
    sid=sid.set_index("sid"); rows=[]
    assert sid.index.is_unique and len(sid)==2368 and not track.duplicated(["sid","UTC"]).any()
    for name,g in track.sort_values(["sid","UTC"]).groupby("sid",sort=True):
        z=sid.loc[name]; t=pd.DatetimeIndex(g.UTC).as_unit("ns");w=pd.to_numeric(g.wind_kt).to_numpy(float)
        gap=np.r_[np.diff(t.asi8)/3.6e12,np.nan];finite=np.isfinite(w)
        legal=(gap==6)&finite&np.r_[finite[1:],False]&(t>=z.genesis_time)&(t+pd.Timedelta(hours=6)<=z.termination_time)
        assert np.array_equal(legal,boolean(g.valid_6h_interval))
        hot=finite&(w>=z.mature_threshold_wind)
        original=legal&hot&(t>=z.maturity_time);alllife=legal&hot;early=alllife&(t<z.maturity_time)
        assert np.array_equal(original,boolean(g.U_selected)&legal)
        pre=(z.lmi_time-z.genesis_time).total_seconds()/3600
        post=(z.termination_time-z.lmi_last_time).total_seconds()/3600
        A=(pre-post)/(pre+post);close(A,z.A_signed,1e-12,1e-12)
        stages=np.select([t<z.escalation_time,t<z.maturity_time,t<z.resolution_time],["D","E","F"],default="Res")
        node_episodes=_episodes(hot&(t>=z.maturity_time),t)
        assert node_episodes==int(z.near_lmi_episode_count)
        row=dict(sid=name,basin_genesis=z.basin_genesis,A=A,original_node_episode_n=node_episodes,
                 original_legal_episode_n=_episodes(original,t),all_life_legal_episode_n=_episodes(alllife,t),
                 early_legal_episode_n=_episodes(early,t),early_h=int(early.sum()*6),
                 original_qualifying_h=int(original.sum()*6),all_life_qualifying_h=int(alllife.sum()*6))
        for stage in ["D","E","F","Res"]:row[stage+"_legal_h"]=int(np.sum(legal&(stages==stage))*6)
        rows.append(row)
    per_sid=pd.DataFrame(rows);target=read(source/"accepted/per_SID_comparison.csv.gz").set_index("sid")
    q=per_sid.set_index("sid").loc[target.index]
    close(q.early_h,target.early_qualifying_h if "early_qualifying_h" in target else target.early_U_h)
    summary=pd.DataFrame([dict(n_sid=len(q),original_recurrent_n=int((q.original_node_episode_n>1).sum()),
         all_life_legal_recurrent_n=int((q.all_life_legal_episode_n>1).sum()),
         early_sid_n=int((q.early_h>0).sum()),early_episode_n=int(q.early_legal_episode_n.sum()),
         early_covered_h=int(q.early_h.sum()),A_mean=float(q.A.mean()),
         original_and_all_life_intersection_n=int(((q.original_node_episode_n>1)&(q.all_life_legal_episode_n>1)).sum()))])
    stage=pd.DataFrame([dict(stage=s,n_sid=len(q),mean_legal_h=float(q[s+"_legal_h"].mean()),total_legal_h=int(q[s+"_legal_h"].sum())) for s in ["D","E","F","Res"]])
    out=output("lifecycle",out)
    for name,d in [("per_sid",per_sid),("recurrence_summary",summary),("stage_covered_duration",stage)]:save_table(d,out/(name+".csv"))
    return summary

