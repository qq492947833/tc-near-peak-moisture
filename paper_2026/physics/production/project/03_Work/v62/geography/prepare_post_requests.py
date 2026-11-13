"""Exact subsequent SCS process requests; no ERA5 reads and no field statistics."""
from pathlib import Path
import json,hashlib,os,datetime
import numpy as np,pandas as pd
from geography_tables import O,W6,V,read,bools,state,sha

def geom(utc,lat,lon,reader_sha):
    utc=pd.Timestamp(utc).tz_convert('UTC').tz_localize(None).isoformat()+'Z'
    obj=dict(UTC=utc,latitude=float(lat),longitude=float(lon)%360,radius_km=800.,halo=1,reader_sha=reader_sha)
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()[:20]

def main():
    risk=read(O/'G3_departure_risk_SID_source.csv');counts=read(O/'G3_region_followup_counts.csv')
    selected=[x for x in ['SCS_box','Japan_Sea_box'] if bools(counts.loc[counts.region.eq(x),'process_support_qualified']).any()]
    e=risk[risk.region.isin(selected)].copy()
    r=read(W6/'causal/raw_cohort_all_timestamps.csv');r=r[r.SID.isin(e.sid)&r.TRACK_TYPE.fillna('').str.strip().str.upper().isin(['','MAIN'])].copy()
    r['UTC']=pd.to_datetime(r.ISO_TIME,utc=True);r=r[r.UTC.dt.minute.eq(0)&r.UTC.dt.hour.mod(6).eq(0)]
    r=r.sort_values(['SID','UTC','source_line']).drop_duplicates(['SID','UTC'])
    for col in ['LAT','LON','USA_WIND']:r[col]=pd.to_numeric(r[col],errors='coerce')
    raw=r.set_index(['SID','UTC']);rows=[]
    for z in e.itertuples():
        for lag in [0,6,12,18,24]:
            utc=pd.Timestamp(z.t0)+pd.Timedelta(hours=lag);key=(z.sid,utc);pk=(z.sid,utc-pd.Timedelta(hours=6))
            a=raw.loc[key] if key in raw.index else None;b=raw.loc[pk] if pk in raw.index else None
            loc=a is not None and np.isfinite([a.LAT,a.LON]).all();motion=loc and b is not None and np.isfinite([b.LAT,b.LON]).all()
            cu=cv=np.nan
            if motion:
                dl=(a.LON-b.LON+180)%360-180;cu=6371008.8*np.cos(np.deg2rad((a.LAT+b.LAT)/2))*np.deg2rad(dl)/21600;cv=6371008.8*np.deg2rad(a.LAT-b.LAT)/21600
            rows.append(dict(request_id=z.event_id+f'|Cpost|lag{lag:+03d}',event_id=z.event_id,sid=z.sid,t0=z.t0,lag_h=lag,UTC=utc,lat=a.LAT if loc else np.nan,lon=a.LON if loc else np.nan,observed_position=loc,position_support='observed_raw_MAIN_synoptic' if loc else 'missing_actual_position',source_line_number=a.source_line if a is not None else np.nan,wind_kt=a.USA_WIND if a is not None else np.nan,state_class=state(a.USA_STATUS,a.NATURE) if a is not None else 'C',USA_STATUS=a.USA_STATUS if a is not None else '',NATURE=a.NATURE if a is not None else '',motion_u_past6h_ms=cu,motion_v_past6h_ms=cv,motion_speed_past6h_ms=np.hypot(cu,cv),motion_valid=motion,motion_previous_UTC=pk[1] if motion else pd.NaT,motion_max_source_UTC=utc if motion else pd.NaT,position_max_source_UTC=utc if loc else pd.NaT,Vref_kt=z.Vref_kt,region_at_t0=z.region,known_return=bool(z.known_return),known_nonreturn=bool(z.known_nonreturn),unknown_outcome=bool(z.unknown_outcome),outcome_group='known_return' if z.known_return else 'complete_nonreturn' if z.known_nonreturn else 'unknown',center_policy='real_along_track_current_lag',interpolated=False))
    q=pd.DataFrame(rows);assert q.request_id.is_unique and len(q)==5*len(e)
    assert (q[['known_return','known_nonreturn','unknown_outcome']].sum(1)==1).all()
    # The lag-zero coordinates and motion reproduce the already accepted requests.
    old=read(V/'data/p3_environment_request_sources.csv');old=old[old.lag_h.eq(0)&old.event_id.isin(e.event_id)].set_index('event_id');z=q[q.lag_h.eq(0)].set_index('event_id').loc[old.index]
    for k in ['lat','lon','motion_u_past6h_ms','motion_v_past6h_ms']:assert np.allclose(z[k],old[k],rtol=0,atol=1e-12,equal_nan=True),k
    q.to_csv(O/'post_process_requests.csv',index=False)
    reader_sha=sha(V/'src/native_fields_v5.py');native=[];roots=[O/'native_cache',V/'cache/native_fields',W6/'physics/native_cache'];mainkeys=['sp','q700','u700','v700','q600','q850','w700','u850','v850','d850']
    for z in q.itertuples():
        if not z.observed_position:continue
        for lag,keys in [(0,mainkeys),(-6,['sp','q700']),(6,['sp','q700'])]:
            utc=pd.Timestamp(z.UTC)+pd.Timedelta(hours=lag);geo=geom(utc,z.lat,z.lon,reader_sha);present=[]
            for root in roots:
                try: names={x.name for x in os.scandir(root/geo)}
                except FileNotFoundError:names=set()
                present.append((root/geo,names))
            for key in keys:
                src=next((str(folder) for folder,names in present if key+'.npz' in names and ('geometry.json' if key=='sp' else key+'.json') in names),'')
                native.append(dict(request_id=z.request_id,event_id=z.event_id,sid=z.sid,request_lag_h=z.lag_h,field_lag_h=lag,UTC=utc,lat=z.lat,lon=z.lon,geometry_id=geo,variable=key,existing_cache_folder=src,presence_verified_only=True))
    nn=pd.DataFrame(native);nn.to_csv(O/'post_exact_native_requests.csv',index=False)
    unique=nn.drop_duplicates(['geometry_id','variable']);unique.to_csv(O/'post_unique_native_requests.csv',index=False)
    receipt=dict(status='EXACT_REQUEST_PLAN_AND_CACHE_PRESENCE_COMPLETE_NO_EXTRACTION_STARTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_boxes=selected,n_sid=len(e),outcome_groups=e.groupby('region')[['known_return','known_nonreturn','unknown_outcome']].sum().reset_index().to_dict('records'),requests=len(q),actual_positions=int(q.observed_position.sum()),valid_motion=int(q.motion_valid.sum()),missing_positions=q.loc[~q.observed_position,['sid','lag_h']].to_dict('records'),native_request_rows=len(nn),unique_geometry_variable_requests=len(unique),present=int(unique.existing_cache_folder.ne('').sum()),missing=int(unique.existing_cache_folder.eq('').sum()),no_future_covariates='All outputs are subsequent process descriptions; no post-time fields enter pre-event models.',groups='Known return and complete nonreturn compared within SCS; unknown outcomes remain a separate descriptive group.',common_support='Budget components share each native finite-cell mask; >=0.9 original radial area. Full 0-24 h integral requires all five valid requested times, integrated in days. 850 MFC has independent native support and is not summed with the 700 budget.',source_sha256={str(p):sha(p) for p in [O/'G3_departure_risk_SID_source.csv',W6/'causal/raw_cohort_all_timestamps.csv',V/'src/native_fields_v5.py',Path(__file__)]},runtime='Plan requires one of the two bounded extraction slots to become free; no third worker is launched.')
    (O/'post_request_plan_receipt.json').write_text(json.dumps(receipt,indent=2,default=str),encoding='utf-8');print(json.dumps(receipt,default=str))

if __name__=='__main__':main()
