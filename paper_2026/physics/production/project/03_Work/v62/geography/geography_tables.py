from _release_paths import release_path as _release_path
"""Frozen C geography: keep retrospective recurrence and prospective risk distinct."""
from pathlib import Path
import os,json,hashlib,datetime
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
from scipy.stats import norm
import shapefile
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import contains_xy
O=Path(__file__).resolve().parent;W6=O.parents[1]/'v6';V=_release_path('mechanism_v5');RUN=V.parent
SEED=62112026;B=2000
def read(p,**kw):return pd.read_csv(p,keep_default_na=False,na_values=['',' ','nan','NaN','-9999','-9999.0'],low_memory=False,**kw)
def bools(x):return x.astype(str).str.lower().isin(['true','1','1.0'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
spec=json.loads((O.parent/'analysis_spec_v62.json').read_text());C=spec['module_C'];assert C['status']=='FROZEN_FOR_EXECUTION'
sea=unary_union([shape(s.__geo_interface__) for s in shapefile.Reader(str(_release_path('ocean_shapefile'))).shapes()])
def region(basin,lat,lon):
    lat=np.asarray(lat,float);lon=(np.asarray(lon,float)+180)%360-180;basin=np.asarray(basin,str)
    reg=np.where(basin=='WP','WP_remainder',basin).astype(object);wet=contains_xy(sea,lon,lat)&np.isfinite(lon+lat)
    for z in C['regions']['subregions']:
        mask=(basin==z['parent'])&wet&(lat>=z['lat_min'])&(lat<z['lat_max'])&(lon>=z['lon_min'])&(lon<z['lon_max']);reg[mask]=z['id']
    return reg,wet
def regions(d,basin='basin',box='region'):
    yield 'GLOBAL',np.ones(len(d),bool)
    for b in ['NA','EP','WP','NI','SI','SP','SA','UNKNOWN']:
        yield b,d[basin].eq(b).to_numpy()
    for b in ['SCS_box','Japan_Sea_box','WP_remainder']:
        if box in d:yield b,d[box].eq(b).to_numpy()
def wilson(k,n):
    if n==0:return np.nan,np.nan
    z=norm.ppf(.975);p=k/n;a=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return max(0,a-h),min(1,a+h)
def weights(n,label):
    ix=np.random.default_rng(SEED).integers(0,n,size=(B,n),dtype=np.int32)
    np.savez_compressed(O/(label+'_SID_bootstrap_indices.npz'),indices=ix)
    w=np.array([np.bincount(x,minlength=n) for x in ix],np.int16);return w
def bootmean(v,mask,w):
    v=np.asarray(v,float);mask=np.asarray(mask,bool)&np.isfinite(v);den=w[:,mask].sum(1)
    out=np.full(B,np.nan);np.divide(w[:,mask]@v[mask],den,out=out,where=den>0)
    return float(np.mean(v[mask])) if mask.any() else np.nan,np.nanquantile(out,[.025,.975]) if mask.any() else [np.nan,np.nan],out
def state(u,n):
    u=str(u).strip().upper() if pd.notna(u) else '';n=str(n).strip().upper() if pd.notna(n) else ''
    if u in {'TD','TS','TY','ST','TC','HU','HR'}:return 'A'
    if u in {'DB','SD','SS','EX','PT','IN','DS','LO','WV','MD'}:return 'B'
    if u:return 'C'
    return 'A' if n=='TS' else 'B' if n in {'DS','ET','SS'} else 'C'
def raw_tracks(wanted):
    verified=O/'geography_raw_MAIN_source.csv'
    r=read(verified if verified.exists() else W6/'causal/raw_cohort_all_timestamps.csv');missing=set(wanted)-set(r.SID)
    if missing:
        chunks=[];total=0;cols=['SID','ISO_TIME','BASIN','LAT','LON','TRACK_TYPE','USA_WIND','USA_STATUS','NATURE']
        for q in read(Path(str(_release_path('ibtracs'))),usecols=cols,dtype=str,chunksize=100000):
            q['source_line']=np.arange(total+2,total+2+len(q));total+=len(q);chunks.append(q[q.SID.isin(missing)])
        r=pd.concat([r]+chunks,ignore_index=True)
    r=r[r.SID.isin(wanted)&r.TRACK_TYPE.fillna('').str.strip().str.upper().isin(['','MAIN'])].copy();r['UTC']=pd.to_datetime(r.ISO_TIME,utc=True,errors='coerce');r['wind']=pd.to_numeric(r.USA_WIND,errors='coerce');r['state']=[state(u,n) for u,n in zip(r.USA_STATUS,r.NATURE)]
    for k in ['LAT','LON']:r[k]=pd.to_numeric(r[k],errors='coerce')
    r=r.sort_values(['SID','UTC','source_line']).drop_duplicates(['SID','UTC']);return r

def g1():
    tables=[]
    for config in ['main','beta085','beta095']:
        p=RUN/'representation_value_v4/source_data'/('durations.csv' if config=='main' else 'durations_'+config+'.csv');d=read(p);d=d[bools(d.main_complete)].reset_index(drop=True);d['basin']=d.basin_genesis.fillna('UNKNOWN');w=weights(len(d),'G1_'+config)
        if config=='main':assert len(d)==2366 and int(d.near_lmi_episode_count.ge(2).sum())==123
        for sensitivity,dd in [('all',np.ones(len(d),bool)),('exclude_single6h_later',~d.later_class.eq('single_6h').to_numpy())]:
            for reg,mask in regions(d):
                mask=mask&dd;n=int(mask.sum())
                for count_kind,y in [('any_recurrence',d.near_lmi_episode_count.ge(2)),('exactly_two',d.near_lmi_episode_count.eq(2)),('three_or_more',d.near_lmi_episode_count.ge(3))]:
                    k=int(y[mask].sum());est,ci,draw=bootmean(y,mask,w);wi=wilson(k,n);sparse=n<30 or k<5 or k==n
                    tables.append(dict(config=config,sensitivity=sensitivity,region=reg,episode_category=count_kind,n_sid=n,n_recurrent=k,fraction=est,ci_low=wi[0] if sparse else ci[0],ci_high=wi[1] if sparse else ci[1],CI_method='Wilson binomial' if sparse else '2000 SID bootstrap',bootstrap_low=ci[0],bootstrap_high=ci[1],wilson_low=wi[0],wilson_high=wi[1]))
        d.to_csv(O/('G1_'+config+'_SID_source.csv'),index=False)
    pd.DataFrame(tables).to_csv(O/'G1_genesis_basin_recurrence.csv',index=False)

def g2():
    d=read(RUN/'representation_value_v4/source_data/durations.csv');d=d[bools(d.main_complete)&d.near_lmi_episode_count.ge(2)].set_index('sid');ledger=read(RUN/'representation_value_v4/source_data/C2_later_episode_ledger.csv');ledger=ledger[ledger.sid.isin(d.index)]
    raw=raw_tracks(d.index);raw.to_csv(O/'G2_raw_MAIN_records.csv',index=False)
    raw=raw[raw.UTC.dt.minute.eq(0)&raw.UTC.dt.hour.mod(6).eq(0)].copy();raw.to_csv(O/'G2_raw_MAIN_synoptic_records.csv',index=False)
    groups={sid:g.set_index('UTC') for sid,g in raw.groupby('SID')};rows=[]
    for z in ledger.itertuples():
        dd=d.loc[z.sid];g=groups[z.sid];start=pd.Timestamp(z.start_UTC);end=pd.Timestamp(z.end_UTC);tau=dd.threshold90_kt;mt=pd.Timestamp(dd.maturity_time)
        high=g[(g.index>=start)&(g.index<end)&g.wind.ge(tau)];ret=high.index.min() if len(high) else pd.NaT
        prior=g[(g.index>=mt)&(g.index<(ret if pd.notna(ret) else start))]
        low=prior[prior.wind.lt(tau)];reason='';dep=pd.NaT
        if pd.isna(ret):reason='frozen_later_piece_without_observed_original_wind_at_or_above_threshold'
        elif not len(low):reason='no_observed_subthreshold_record_between_near_peak_segments'
        else:
            # Last contiguous observed low spell before the observed return.
            li=low.index[-1];dep=li
            while dep-pd.Timedelta(hours=6) in low.index:dep-=pd.Timedelta(hours=6)
        known=pd.notna(ret) and pd.notna(dep)
        q=g.loc[ret] if pd.notna(ret) else None;p=g.loc[dep] if pd.notna(dep) else None
        rr=dict(sid=z.sid,genesis_basin=dd.basin_genesis,episode_count=dd.near_lmi_episode_count,episode_id=z.later_episode_id,episode_start_UTC=start,episode_end_UTC=end,threshold_kt=tau,departure_UTC=dep,return_UTC=ret,known_return=known,gap_reason=reason,valid_6h_count=z.valid_interval_n,covered_h=z.covered_h,later_class=dd.later_class)
        for tag,x in [('departure',p),('return',q)]:
            rr.update({tag+'_lat':x.LAT if x is not None else np.nan,tag+'_lon':x.LON if x is not None else np.nan,tag+'_basin':x.BASIN if x is not None and pd.notna(x.BASIN) else 'UNKNOWN',tag+'_state':x.state if x is not None else 'C',tag+'_source_line':x.source_line if x is not None else np.nan})
        rr['USA_STATUS']=q.USA_STATUS if q is not None else '';rr['NATURE']=q.NATURE if q is not None else '';rr['state_class']=q.state if q is not None else 'C'
        rr['return_interval_gap_h']=(ret-low.index[-1])/pd.Timedelta(hours=1) if known else np.nan;rr['return_time_exact_six_hour_bin']=known and rr['return_interval_gap_h']==6
        rr['agency_conflict']=bool(q is not None and str(q.NATURE).strip()=='MX')
        rows.append(rr)
    out=pd.DataFrame(rows)
    for tag in ['departure','return']:out[tag+'_box'],out[tag+'_ocean']=region(out[tag+'_basin'],out[tag+'_lat'],out[tag+'_lon'])
    out['land_contact']=~out.return_ocean;out['strict_tropical']=out.known_return&out.state_class.eq('A')&~out.agency_conflict
    out.to_csv(O/'G2_all_episode_ledger.csv',index=False)
    confirm=out[out.known_return].sort_values(['sid','return_UTC','episode_id']);first=confirm.drop_duplicates('sid');first.to_csv(O/'G2_first_confirmed_return.csv',index=False);confirm.to_csv(O/'G2_all_confirmed_returns.csv',index=False)
    summary=[]
    for selection,z in [('first_per_SID',first),('all_returns_SID_clustered',confirm),('strict_tropical_first',confirm[confirm.strict_tropical].drop_duplicates('sid'))]:
        for reg,mask in regions(z.rename(columns={'return_basin':'basin','return_box':'region'})):
            summary.append(dict(selection=selection,region=reg,events=int(mask.sum()),unique_sid=z.loc[mask,'sid'].nunique(),estimand='actual observed return locations/counts; no risk denominator'))
    pd.DataFrame(summary).to_csv(O/'G2_region_event_counts.csv',index=False)
    return out,first,confirm

def g3():
    e=read(W6/'causal/cohort_recovered.csv').reset_index(drop=True);e['basin']=e.basin_t0.fillna('UNKNOWN');e['region'],e['t0_ocean']=region(e.basin,e.lat,e.lon);e['known_return']=e.recovery48.eq(1);e['known_nonreturn']=e.recovery48.eq(0);e['unknown_outcome']=e.recovery48.isna();assert (e[['known_return','known_nonreturn','unknown_outcome']].sum(1)==1).all()
    e['month']=pd.to_datetime(e.t0,utc=True).dt.month
    raw=raw_tracks(e.sid);groups={sid:g.set_index('UTC') for sid,g in raw.groupby('SID')};pos=[]
    for z in e.itertuples():
        g=groups[z.sid];t0=pd.Timestamp(z.t0);tt=t0+pd.Timedelta(hours=z.first_observed_return_h) if np.isfinite(z.first_observed_return_h) else pd.NaT;q=g.loc[tt] if pd.notna(tt) and tt in g.index else None
        past=g[(g.index>=t0-pd.Timedelta(hours=24))&(g.index<=t0)&(g.index.hour%6==0)&(g.index.minute==0)];pbox,wet=region(past.BASIN.fillna('UNKNOWN'),past.LAT,past.LON)
        pos.append(dict(sid=z.sid,return_UTC=tt,return_lat=q.LAT if q is not None else np.nan,return_lon=q.LON if q is not None else np.nan,return_basin=q.BASIN if q is not None and pd.notna(q.BASIN) else 'UNKNOWN',return_USA_STATUS=q.USA_STATUS if q is not None else '',return_NATURE=q.NATURE if q is not None else '',return_state=q.state if q is not None else 'C',pre24_ocean_points=int(wet.sum()),pre24_land_points=int((~wet&np.isfinite(past.LAT+past.LON)).sum()),pre24_observed_points=len(past),sea_reentry_within24=bool(len(wet)>=2 and wet[-1] and (~wet[:-1]).any())))
    p=pd.DataFrame(pos);p['return_region'],p['return_ocean']=region(p.return_basin,p.return_lat,p.return_lon);e=e.merge(p,on='sid',validate='1:1');e['known_tropical_return']=e.known_return&e.return_state.eq('A');e['known_other_return']=e.known_return&e.return_state.eq('B');e['known_unknown_state_return']=e.known_return&e.return_state.eq('C');e.to_csv(O/'G3_departure_risk_SID_source.csv',index=False)
    w=weights(len(e),'G3');pred=read(W6/'causal/point_predictions.csv');joined=pred.merge(e[['sid','basin','region','known_return','known_nonreturn','unknown_outcome','return_state']],on='sid',suffixes=('_genesis',''),validate='m:1');joined.to_csv(O/'G3_regional_policy_OOF_scores.csv',index=False)
    tables=[];risks=[];effects=[];drawcols={};eidx=e.set_index('sid');counts=[]
    for reg,mask in regions(e):
        q=e[mask];n=len(q);k=int(q.known_return.sum());unknown=int(q.unknown_outcome.sum())
        counts.append(dict(region=reg,n_risk=n,known_return=k,complete_nonreturn=int(q.known_nonreturn.sum()),unknown_outcome=unknown,early_return_incomplete=int((q.known_return&~bools(q.complete48)).sum()),known_tropical_return=int(q.known_tropical_return.sum()),known_other_return=int(q.known_other_return.sum()),known_unknown_state_return=int(q.known_unknown_state_return.sum()),lower_risk_bound=k/n if n else np.nan,upper_risk_bound=(k+unknown)/n if n else np.nan,median_t0_wind=q.wind_t0_kt.median(),median_age_h=q.age_h.median(),past_land_contact_n=int(bools(q.past_land_contact).sum()),current_land_contact_n=int(bools(q.current_land_contact).sum()),sea_reentry24_n=int(q.sea_reentry_within24.sum()),process_support_qualified=bool(n>=50 and k>=10 and q.known_nonreturn.sum()>=20)))
        for policy in ['shear200_850','midRH']:
            p=pred[pred.policy.eq(policy)].set_index('sid').loc[e.sid];assert np.array_equal(p.index,e.sid)
            m=p.m48.to_numpy();Y=p.Y48.fillna(0).to_numpy();R=bools(p.R48).to_numpy();gw=R*np.minimum(1/p.g48.to_numpy(),20)
            for method,val in [('natural_gformula',m),('natural_augmented',m+gw*(Y-m))]:
                est,ci,draw=bootmean(val*100,mask,w);risks.append(dict(region=reg,branch=policy,method=method,n_sid=n,risk_percent=est,ci_low=ci[0],ci_high=ci[1],inference='2000 SID bootstrap conditional on fixed OOF fits',primary=policy=='shear200_850' and method=='natural_gformula'));drawcols[f'{reg}|{policy}|{method}']=draw
            ratio=p.density_ratio.to_numpy();cover=bools(p.covered).to_numpy()
            for outcome in ['48','24']:
                obs=bools(p['R'+outcome]).to_numpy();ow=obs*np.minimum(1/p['g'+outcome].to_numpy(),20);cw=ow*ratio
                def ess(v):return float(v.sum()**2/np.square(v).sum()) if np.square(v).sum() else 0.
                tables.append(dict(region=reg,policy=policy,outcome=outcome,n_sid=n,shifted_n=int(cover[mask].sum()),coverage=float(cover[mask].mean()) if n else np.nan,ratio_ESS=ess(ratio[mask]),observed_IPCW_ESS=ess(ow[mask]),policy_observed_ESS=ess(cw[mask]),weight_max=float(cw[mask].max()) if n else np.nan,observed_outcome_n=int(obs[mask].sum())))
                for method in ['gcomp','onestep','ipw','onestep_uncapped']:
                    col=method+outcome if method!='onestep_uncapped' else 'onestep'+outcome+'_uncapped';scale=100 if outcome=='48' else 1
                    est,ci,draw=bootmean(p[col].to_numpy()*scale,mask,w);effects.append(dict(region=reg,policy=policy,outcome='recovery48_pp' if outcome=='48' else 'delta24_kt',method=method,n_sid=n,estimate=est,ci_low=ci[0],ci_high=ci[1],inference='2000 SID bootstrap conditional on fixed OOF fits',headline_support=bool(n>=50 and k>=10)));drawcols[f'{reg}|{policy}|{outcome}|{method}']=draw
    for name,rows in [('G3_region_followup_counts',counts),('G3_region_risk_estimates',risks),('G3_region_policy_estimates',effects),('G3_region_policy_support',tables)]:pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
    pd.DataFrame(drawcols).to_csv(O/'G3_fixed_OOF_bootstrap_draws.csv',index_label='draw')
    e[e.known_return].groupby(['region','return_region'],dropna=False).agg(n_sid=('sid','size'),tropical_returns=('known_tropical_return','sum')).reset_index().to_csv(O/'G3_origin_return_region_migration.csv',index=False)
    e.groupby(['region','month']).agg(n_risk=('sid','size'),known_return=('known_return','sum'),unknown=('unknown_outcome','sum')).reset_index().to_csv(O/'G3_region_season.csv',index=False)
    horizon=[]
    for reg,mask in regions(e):
        for h,y in [('48',e.recovery48),('96',pd.to_numeric(e.return96,errors='coerce'))]:horizon.append(dict(region=reg,horizon_h=h,n_risk=int(mask.sum()),known_return=int(y[mask].eq(1).sum()),known_nonreturn=int(y[mask].eq(0).sum()),unknown=int(y[mask].isna().sum()),scope='observed-status counts; unknown retained, no new 96h hazard fit'))
    pd.DataFrame(horizon).to_csv(O/'G3_horizon_sensitivity_counts.csv',index=False)
    return e

def grids(first,allreturns,risk):
    rows=[]
    for res in [5,10]:
        for name,g,lat,lon in [('G2_first',first,'return_lat','return_lon'),('G2_all',allreturns,'return_lat','return_lon'),('G3_risk',risk,'lat','lon')]:
            g=g[np.isfinite(g[lat]+g[lon])].copy();g['lon_bin']=np.floor((((g[lon]+180)%360)-180)/res)*res;g['lat_bin']=np.floor((g[lat]+90)/res)*res-90
            for (x,y),q in g.groupby(['lon_bin','lat_bin']):
                k=int(q.known_return.sum()) if name=='G3_risk' else len(q);n=len(q);rows.append(dict(grid_degrees=res,quantity=name,lon_left=x,lat_bottom=y,n_records=n,n_sid=q.sid.nunique(),known_return=k,unknown=int(q.unknown_outcome.sum()) if name=='G3_risk' else 0,risk_display_support=bool(name=='G3_risk' and n>=20 and k>=5),estimand='risk-start count' if name=='G3_risk' else 'observed return-location count'))
    pd.DataFrame(rows).to_csv(O/'geographic_grid_counts_5_and_10deg.csv',index=False)

def tests():
    checks=[]
    def ck(name,z):checks.append(dict(test=name,passed=bool(z)));assert z,name
    ck('NA_is_literal_basin',read(W6/'causal/cohort_recovered.csv').basin_genesis.eq('NA').any())
    ck('zero_events_positive_Wilson_upper',wilson(0,10)[1]>0)
    ck('dateline_wrap_180_equals_minus180',((180+180)%360-180)==-180)
    for winds,expected in [([100,90,100],True),([100,np.nan,100],False),([100,100,100],False)]:ck('observed_low_required_'+str(winds),bool(np.any(np.asarray(winds,float)<95))==expected)
    ck('ET_status_not_tropical',state('EX','ET')=='B');ck('unknown_not_tropical',state('','')=='C')
    ck('disjoint_boxes',C['regions']['subregions'][0]['lat_max']<=C['regions']['subregions'][1]['lat_min'])
    pd.DataFrame(checks).to_csv(O/'geography_basic_tests.csv',index=False)

def main():
    tests();g1();ledger,first,allreturns=g2();risk=g3();grids(first,allreturns,risk)
    receipt=dict(status='GEOGRAPHY_TABLES_COMPUTED_PENDING_REVIEW',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),G1_n=2366,G1_frozen_recurrence=123,G2_ledger_rows=len(ledger),G2_confirmed_first_SID=len(first),G2_confirmed_all_events=len(allreturns),G3_risk_SID=len(risk),global_bootstrap_not_refit_here=True,regional_bootstrap=B,regional_uncertainty='fixed OOF conditional SID bootstrap',driver_sha256=sha(__file__),module_hash=spec['module_hashes'].get('C'),source_manifest_sha256=sha(O/'input_sources.json'))
    dump(O/'geography_tables_receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
