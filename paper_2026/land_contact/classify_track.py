"""Classify track-centre contact on all native reports inside a UTC interval."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "inputs"
RAW = DATA / "ibtracs_v04r01_tracks.csv.gz"
RETURNS = DATA / "event_windows.csv"
COLS = ["SID","ISO_TIME","TRACK_TYPE","LAT","LON","DIST2LAND","LANDFALL"]
HOUR = pd.Timedelta(hours=1)

def status_empty(reason):
    return dict(status="unknown",land_to_sea="unknown",reason=reason,
                track_coverage_complete=False, native_fixes=0, native_segments=0,
                point_zero_n=0, segment_zero_n=0, sea_endpoint_crossing_n=0,
                max_gap_h=None, first_contact_utc="", first_contact_basis="",
                first_subsequent_sea_utc="", missing_point_n=0, missing_segment_n=0)


class TrackAudit:
    """window(sid, start, end): inclusive fixes / only forward segments ending within window."""
    def __init__(self,sids=None):
        ids = set(pd.read_csv(RETURNS,usecols=["sid"]).sid if sids is None else map(str,sids))
        columns = COLS
        pieces = []
        for chunk in pd.read_csv(RAW,usecols=columns,dtype=str,chunksize=40000,keep_default_na=False):
            take = chunk[chunk.SID.isin(ids) & chunk.TRACK_TYPE.str.strip().str.upper().isin(["MAIN",""])]
            if len(take):
                pieces.append(take.copy())
        self.tracks = {}
        self.duplicates = 0
        self.cadences = Counter()
        if not pieces:
            return
        frame = pd.concat(pieces,ignore_index=True)
        frame["UTC"] = pd.to_datetime(frame.ISO_TIME,utc=True,format="%Y-%m-%d %H:%M:%S",errors="raise")
        for col in ["LAT","LON","DIST2LAND","LANDFALL"]:
            frame[col] = pd.to_numeric(frame[col],errors="coerce")
        for sid,g in frame.groupby("SID"):
            g = g.sort_values("UTC",kind="stable")
            duplicated = g.duplicated("UTC",keep=False)
            self.duplicates += int(duplicated.sum())
            conflicted = g.loc[duplicated].groupby("UTC")[["LAT","LON","DIST2LAND","LANDFALL"]].nunique(dropna=False).gt(1).any(axis=1)
            g = g.drop_duplicates("UTC",keep="first").set_index("UTC").copy()
            g["conflict"] = g.index.isin(conflicted[conflicted].index)
            sane = g.LAT.between(-90,90) & g.LON.between(-360,360) & ~g.conflict
            g["point"] = g.DIST2LAND.where(sane & g.DIST2LAND.ge(0))
            g["seg"] = g.LANDFALL.where(sane & g.LANDFALL.ge(0))
            self.tracks[sid] = g
            self.cadences.update((g.index.to_series().diff().dropna()/HOUR).tolist())

    @lru_cache(maxsize=8192)
    def window(self,sid,start,end):
        if pd.isna(start) or pd.isna(end):
            return status_empty("missing_frozen_window_anchor")
        start,end=pd.to_datetime(start,utc=True),pd.to_datetime(end,utc=True)
        if end<start or sid not in self.tracks:
            return status_empty("invalid_window_or_absent_track")
        track=self.tracks[sid]
        g=track.loc[start:end]
        if g.empty:
            return status_empty("no_native_fixes_in_window")
        ts=g.index
        gaps=(ts[1:]-ts[:-1])/HOUR
        p=g.point.to_numpy(dtype=float)
        # Segments touching either conflicting duplicate have unresolved identity.
        s=g.seg.to_numpy(dtype=float)[:-1].copy()
        s[g.conflict.to_numpy()[1:]]=np.nan
        # All accepted windows use existing UTC observations. LANDFALL covers the
        # following report, so never include the segment beginning at the final fix.
        point=np.flatnonzero(p==0)
        seg=np.flatnonzero(s==0)
        pass_through=[j for j in seg if p[j]>0 and p[j+1]>0]
        issues=[]
        if ts[0]!=start: issues.append("missing_start")
        if ts[-1]!=end: issues.append("missing_end")
        if np.any(gaps>3): issues.append("native_gap_over_3h")
        if np.any(~np.isfinite(p)): issues.append("missing_or_conflicting_point")
        if np.any(~np.isfinite(s)): issues.append("missing_or_conflicting_segment")
        complete=not issues
        land=bool(len(point) or len(seg))
        evidence=sorted([(ts[j],0,"point_zero") for j in point]+[(ts[j],1,"forward_interval_zero") for j in seg])
        # A segment with an on-land minimum followed by a verified off-land endpoint
        # establishes a transition inside that segment even when both endpoints are at sea.
        first_sea=""
        if land:
            first_utc,_,basis=evidence[0]
            later=ts[(ts>first_utc)&(p>0)]
            if len(later): first_sea=later[0].isoformat()
        state="contact" if land else ("no_contact" if complete else "unknown")
        transition="yes" if first_sea else ("no" if complete else "unknown")
        return dict(status=state,land_to_sea=transition,reason="|".join(issues) or "complete",
                    track_coverage_complete=complete, native_fixes=len(ts),native_segments=len(s),
                    point_zero_n=len(point),segment_zero_n=len(seg),sea_endpoint_crossing_n=len(pass_through),
                    max_gap_h=float(max(gaps)) if len(gaps) else 0,
                    first_contact_utc=evidence[0][0].isoformat() if land else "",
                    first_contact_basis=evidence[0][2] if land else "",
                    first_subsequent_sea_utc=first_sea,
                    missing_point_n=int(np.sum(~np.isfinite(p))),missing_segment_n=int(np.sum(~np.isfinite(s))))
