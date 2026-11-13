"""P3 time-zero cohort. No environmental extraction or effect-model fitting.
Reads the frozen raw IBTrACS snapshot; writes only mechanism_revision_v5.
The accepted lifecycle state function is executed as an isolated AST function,
never importing or running the legacy analysis module.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, sys, traceback, time
import numpy as np
import pandas as pd

V5=Path(__file__).resolve().parents[1]
R=V5.parent
PROJECT=R.parents[1]
RAW=_release_path('ibtracs')
RAW_SHA='85d421e69e6e963ae4e6025d3e270d8c3e5edf81b981fa06bc7b753aa62fc5a3'
STATE=Path(__file__).resolve().parent/'accepted_state_dictionary.py'
HOUR=pd.Timedelta(hours=1)
TESTS=[]
INPUTS=[]

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def out_json(rel,obj):
    p=V5/rel;assert V5 in p.resolve().parents
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def csv(d,rel):
    p=V5/rel;assert V5 in p.resolve().parents
    d.to_csv(p,index=False,encoding='utf-8-sig')
def register(p,role,expected=None):
    h=sha(p)
    if expected is not None:assert h==expected,(p,h,expected)
    INPUTS.append(dict(path=str(p),role=role,sha256=h,bytes=p.stat().st_size,expected_sha256=expected))
    return p
def check(name,value,detail=None):
    TESTS.append(dict(test=name,passed=bool(value),detail=detail))
    assert value,(name,detail)
def readcsv(p,**kw):
    return pd.read_csv(p,keep_default_na=False,na_values=['',' ','NaN','nan','-9999','-9999.0'],**kw)

def freeze():
    spec={
      'schema':'P3_pre_t0_cohort_v1',
      'frozen_utc':now(),
      'scientific_status':'COHORT_DEFINITION_FROZEN_BEFORE_RAW_OUTCOME_READ; no effect fit',
      'root_authorization':'Parent approval of prefix-only quality rules and fixed96h refractory at 2026-10-08; original user v5 section7.2',
      'raw_source':str(RAW),'raw_expected_sha256':RAW_SHA,
      'state_dictionary':str(STATE),
      'state_priority':'USA_STATUS A={TD,TS,TY,ST,TC,HU,HR}; B={DB,SD,SS,EX,PT,IN,DS,LO,WV,MD}; other nonempty USA=C. Only missing USA uses NATURE TS=A; DS/ET/SS=B; else C. NATURE MX recorded independently as agency conflict.',
      'time_and_track':'00/06/12/18 UTC exactly; MAIN or missing track_type; stable first physical source row if exact duplicate SID UTC; all eligible candidates retained with source line',
      'wind':'raw USA_WIND in kt; no WMO substitution; no interpolation, including isolated missing nodes; recorded observational bounds reflect at-time eligibility rather than full-lifecycle admission',
      'genesis':'first raw finite USA_WIND>=35kt with state A; genesis latitude finite and |latitude|<=40; assign cohort by genesis calendar year1982–2023; keep subsequent history beyond calendar endpoints',
      'prefix_quality':'evaluated only genesis through candidate t0 inclusive: >=5 observed records; finite raw wind fraction>=0.80; largest gap between finite raw winds<=12h. These inherit original notebook QC constants but not future lifecycle support.',
      't0':'raw finite adjacent exact6h pair; leftwind>=Vref and currentwind<Vref; current state A; finite valid current latitude/longitude; historical raw observed max through t0-6h minus genesis wind>=15kt',
      'Vref':'Vg+0.90*(max raw observed wind from genesis through t0−6h−Vg), fixed for every future outcome of this event',
      'main':'first eligible candidate per SID; no future LMI, allstage duration, finaldeparture, future return or full-lifecycle completeness admission',
      'all_nonoverlap_sensitivity':'greedily retain earliest eligible t0 then next eligible t0 at least96h after previous retainedt0; shared boundary at96h permitted, no overlapping positive-duration followup. Uses only already observed departure times.',
      'binary_return':'any observed finite raw wind>=fixedVref in (t0,t0+horizon], regardless state; additionally save tropical-confirmed return for endpoint sensitivity. Observed post-gap return is known1 but return timing is interval-censored.',
      'no_return':'0 only when all scheduled6h winds within horizon observed finite and none reachVref; otherwise unknown, no termination-as0',
      'longitudinal':'event or censor at first missing scheduled6h wind/timestamp; other/unknown state itself does not imply permanent death. Save later raw evidence separately.',
      'outcomes':{'primary_horizon_h':48,'continuous_horizon_h':24,'secondary_horizon_h':96},
      'continuous24':'wind at exact t0+24 minus t0 raw wind; missing if endpoint absent/nonfinite; intermediate gaps recorded; no interpolation',
      'candidate_exclusions':'prefix lowquality/state unknown/other/current location invalid retained in exhaustive crossing candidate table; SID withoutgenesis/eligibledeparture retained in SID audit',
      'baseline_controls_timing':'only raw fields and derivedhistory with max source UTC<=t0; thresholdhistoricalmax source<=t0−6h; no field prefixed outcome or return permitted as baseline',
      'effect_models':'NOT_RUN; requires shared root analysis_spec_v5'
    }
    p=V5/'config/p3_cohort_spec_v1.json'
    if p.exists():
        old=json.loads(p.read_text('utf-8'))
        old.pop('frozen_utc',None);compare=dict(spec);compare.pop('frozen_utc',None)
        assert old==compare,'Existing P3 specification differs; do not overwrite'
    else:out_json('config/p3_cohort_spec_v1.json',spec)
    out_json('audit/p3_spec_identity.json',{'path':str(p),'sha256':sha(p),'read_before_outcomes_utc':now()})
    return p

def dictionary():
    register(STATE,'accepted state function')
    fn=next(n for n in ast.parse(STATE.read_text('utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='state')
    ns={'pd':pd};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(STATE),'exec'),ns)
    return ns['state']

def load_raw(state):
    register(RAW,'same frozen raw IBTrACS snapshot',RAW_SHA)
    use=['SID','NAME','SEASON','ISO_TIME','LAT','LON','USA_WIND','USA_STATUS','NATURE','TRACK_TYPE','BASIN','DIST2LAND','LANDFALL','USA_PRES']
    raw=readcsv(RAW,usecols=use,low_memory=False,dtype=str)
    raw['source_line_number']=np.arange(len(raw),dtype='int64')+2
    raw['UTC']=pd.to_datetime(raw.ISO_TIME,errors='coerce',utc=True,format='%Y-%m-%d %H:%M:%S')
    raw=raw[raw.UTC.notna()].copy()
    intersect=raw.UTC.dt.year.between(1982,2023)
    candidates=set(raw.loc[intersect,'SID'])
    raw=raw[raw.SID.isin(candidates)].copy()
    raw=raw[raw.UTC.dt.hour.isin([0,6,12,18])&raw.UTC.dt.minute.eq(0)&raw.UTC.dt.second.eq(0)].copy()
    raw['TRACK_TYPE']=raw.TRACK_TYPE.fillna('').str.strip().str.upper()
    raw=raw[raw.TRACK_TYPE.isin(['','MAIN'])].copy()
    raw=raw.sort_values(['SID','UTC','source_line_number'],kind='stable')
    duplicated=raw[raw.duplicated(['SID','UTC'],keep=False)].copy()
    csv(duplicated,'audit/p3_duplicate_sid_utc_rows.csv')
    raw=raw.drop_duplicates(['SID','UTC'],keep='first').copy()
    for c in ['LAT','LON','USA_WIND','DIST2LAND','LANDFALL','USA_PRES']:
        raw[c]=pd.to_numeric(raw[c],errors='coerce')
    st=[state(u,n,True) for u,n in zip(raw.USA_STATUS,raw.NATURE)]
    raw['state_class']=[z[0] for z in st];raw['state_source']=[z[1] for z in st]
    raw['state_label']=raw.state_class.map({'A':'tropical','B':'other','C':'unknown'})
    raw['agency_conflict_nature_mx']=raw.NATURE.fillna('').str.strip().str.upper().eq('MX')
    raw['lon']=((raw.LON+180)%360)-180
    raw=raw.rename(columns={'SID':'sid','LAT':'lat','USA_WIND':'wind_kt','BASIN':'basin','DIST2LAND':'dist2land_km'})
    # Compare accepted state labels only; cohort is NOT restricted to crosswalk SIDs.
    cross=register(R/'review_completion_v3_2/audit/state_dependency/state_sid_utc_crosswalk_v3_2.csv','accepted state crosswalk comparator')
    c=readcsv(cross,usecols=['sid','UTC','state_class','state_source'])
    c.UTC=pd.to_datetime(c.UTC,utc=True)
    m=raw[['sid','UTC','state_class','state_source']].merge(c,on=['sid','UTC'],suffixes=('_new','_accepted'),validate='one_to_one')
    check('accepted_state_dictionary_overlap_exact',((m.state_class_new==m.state_class_accepted)&(m.state_source_new==m.state_source_accepted)).all(),len(m))
    check('NA_basin_preserved',raw.basin.eq('NA').any())
    return raw

def history_candidates(g):
    g=g.sort_values('UTC').reset_index(drop=True)
    genesis=np.flatnonzero(g.state_class.eq('A')&np.isfinite(g.wind_kt)&g.wind_kt.ge(35))
    sid=str(g.sid.iloc[0]);audit={'sid':sid,'n_synoptic_raw_records':len(g),'eligible_departures':0}
    if not len(genesis):
        audit['sid_exclusion_reason']='no_observed_tropical_35kt_genesis'
        return [],audit
    gi=int(genesis[0]);gr=g.iloc[gi];tg=gr.UTC;vg=float(gr.wind_kt)
    audit.update(genesis_UTC=tg,genesis_wind_kt=vg,genesis_lat=gr.lat,genesis_year=tg.year)
    if not 1982<=tg.year<=2023:
        audit['sid_exclusion_reason']='genesis_year_outside_1982_2023';return [],audit
    if not np.isfinite(gr.lat) or abs(gr.lat)>40:
        audit['sid_exclusion_reason']='genesis_latitude_missing_or_abs_over40';return [],audit
    a=g.iloc[gi:].reset_index(drop=True)
    times=a.UTC.to_numpy();winds=a.wind_kt.to_numpy(float)
    rows=[]
    for j in range(1,len(a)):
        curr=a.iloc[j];prev=a.iloc[j-1]
        past=winds[:j];finite=np.isfinite(past)
        if not finite.any() or not np.isfinite(winds[j]):continue
        vmax=float(past[finite].max());vref=vg+.90*(vmax-vg)
        if not(np.isfinite(winds[j-1]) and winds[j-1]>=vref and winds[j]<vref):continue
        prefix=a.iloc[:j+1];valid=prefix.wind_kt.notna()
        ft=prefix.loc[valid,'UTC']
        maxgap=float(ft.diff().dt.total_seconds().div(3600).max()) if len(ft)>1 else np.inf
        fraction=float(valid.mean());gain=vmax-vg
        gap=(curr.UTC-prev.UTC)/HOUR
        why=[]
        if gap!=6:why.append('crossing_not_adjacent_exact6h')
        if gain<15:why.append('past_gain_under15kt')
        if curr.state_class!='A':why.append('t0_state_'+curr.state_label)
        if not(np.isfinite(curr.lat) and np.isfinite(curr.lon) and abs(curr.lat)<=90):why.append('t0_location_invalid')
        if len(prefix)<5:why.append('prefix_under5_records')
        if fraction<.80:why.append('prefix_wind_fraction_under0.80')
        if maxgap>12:why.append('prefix_finite_wind_gap_over12h')
        look=prefix.set_index('UTC')
        def prior_value(h,col):
            t=curr.UTC-pd.Timedelta(hours=h)
            return look.at[t,col] if t in look.index else np.nan
        prevlat=prev.lat;prevlon=prev.lon
        dl=((curr.lon-prevlon+180)%360)-180
        cu=6371008.8*np.cos(np.deg2rad((curr.lat+prevlat)/2))*np.deg2rad(dl)/(6*3600) if gap==6 and np.isfinite([curr.lat,prevlat,curr.lon,prevlon]).all() else np.nan
        cv=6371008.8*np.deg2rad(curr.lat-prevlat)/(6*3600) if gap==6 and np.isfinite([curr.lat,prevlat]).all() else np.nan
        event_id=sid+'|P3|'+curr.UTC.strftime('%Y%m%dT%H%MZ')
        observed_land=prefix.dist2land_km
        current_land=float(curr.dist2land_km==0) if np.isfinite(curr.dist2land_km) else np.nan
        history_land=float(observed_land.eq(0).any()) if observed_land.notna().any() else np.nan
        peak_at=a.iloc[np.flatnonzero(np.isfinite(past)&(past==vmax))[0]].UTC
        row=dict(event_id=event_id,sid=sid,t0=curr.UTC,source_line_number=int(curr.source_line_number),
          eligible=not why,exclusion_reason=';'.join(why) if why else 'eligible',
          genesis_UTC=tg,genesis_source_line_number=int(gr.source_line_number),genesis_year=tg.year,
          genesis_wind_kt=vg,genesis_lat=float(gr.lat),genesis_lon=float(gr.lon),basin_genesis=gr.basin,
          basin_t0=curr.basin,wind_t0_kt=float(curr.wind_kt),wind_prev_kt=float(prev.wind_kt),
          previous_UTC=prev.UTC,previous_source_line_number=int(prev.source_line_number),
          Vmax_past_kt=vmax,Vmax_past_first_UTC=peak_at,Vmax_past_latest_source_UTC=prev.UTC,
          Vref_kt=vref,threshold_gap_kt=float(curr.wind_kt-vref),historical_gain_kt=gain,
          lat=float(curr.lat),lon=float(curr.lon),age_h=float((curr.UTC-tg)/HOUR),
          delta_v6_backward_kt=float(curr.wind_kt-prev.wind_kt) if gap==6 else np.nan,
          delta_v24_backward_kt=float(curr.wind_kt-prior_value(24,'wind_kt')),
          delta_v12_backward_kt=float(curr.wind_kt-prior_value(12,'wind_kt')),
          prefix_n_records=len(prefix),prefix_wind_valid_fraction=fraction,prefix_max_finite_wind_gap_h=maxgap,
          t0_state_class=curr.state_class,t0_state_source=curr.state_source,
          t0_USA_STATUS=curr.USA_STATUS,t0_NATURE=curr.NATURE,agency_conflict_nature_mx=bool(curr.agency_conflict_nature_mx),
          current_land_contact=current_land,past_land_contact=history_land,dist2land_km=curr.dist2land_km,
          current_pressure_hpa=curr.USA_PRES,motion_u_prev_ms=cu,motion_v_prev_ms=cv,
          motion_speed_prev_ms=float(np.hypot(cu,cv)),month_sin=float(np.sin(2*np.pi*(curr.UTC.month-1)/12)),
          month_cos=float(np.cos(2*np.pi*(curr.UTC.month-1)/12)),lon_sin=float(np.sin(np.deg2rad(curr.lon))),
          lon_cos=float(np.cos(np.deg2rad(curr.lon))),baseline_max_source_UTC=curr.UTC,
          raw_wind_no_interpolation=True)
        rows.append(row)
    audit['eligible_departures']=sum(x['eligible'] for x in rows)
    audit['all_observed_downcrossings']=len(rows)
    audit['sid_exclusion_reason']='included' if audit['eligible_departures'] else 'no_eligible_past_threshold_departure'
    return rows,audit

def followup(g,event):
    t0=pd.Timestamp(event['t0']);vref=event['Vref_kt'];lookup=g.set_index('UTC')
    rows=[];first_censor=np.nan;first_event=np.nan;lastbelow=0.
    for lag in range(6,97,6):
        t=t0+pd.Timedelta(hours=lag)
        obs=lookup.loc[t] if t in lookup.index else None
        valid=obs is not None and np.isfinite(obs.wind_kt)
        at_risk=not np.isfinite(first_censor) and not np.isfinite(first_event)
        hit=bool(valid and obs.wind_kt>=vref)
        reason='' if valid else ('missing_raw_wind' if obs is not None else ('archive_track_ended' if t>g.UTC.max() else 'missing_synoptic_record'))
        if at_risk:
            if not valid:first_censor=float(lag)
            elif hit:first_event=float(lag)
        row=dict(event_id=event['event_id'],sid=event['sid'],t0=t0,lag_h=lag,UTC=t,
          source_line_number=int(obs.source_line_number) if obs is not None else np.nan,
          observed_record=obs is not None,finite_wind=valid,wind_kt=float(obs.wind_kt) if valid else np.nan,
          state_class=obs.state_class if obs is not None else 'C',state_source=obs.state_source if obs is not None else 'missing record',
          USA_STATUS=obs.USA_STATUS if obs is not None else '',NATURE=obs.NATURE if obs is not None else '',
          agency_conflict_nature_mx=bool(obs.agency_conflict_nature_mx) if obs is not None else False,
          Vref_kt=vref,observed_threshold_return=hit,tropical_confirmed_return=hit and obs.state_class=='A',
          censor_reason=reason,at_risk_contiguous=at_risk,event_observed_contiguous=at_risk and hit,
          censored_contiguous=at_risk and not valid)
        rows.append(row)
    d=pd.DataFrame(rows);out={}
    for h in [48,96]:
        sub=d[d.lag_h<=h]
        hits=sub[sub.observed_threshold_return]
        thits=sub[sub.tropical_confirmed_return]
        complete=bool(sub.finite_wind.all())
        binary=1. if len(hits) else (0. if complete else np.nan)
        first=float(hits.lag_h.iloc[0]) if len(hits) else np.nan
        before=sub[sub.lag_h<first] if len(hits) else sub
        # A gap could conceal an earlier transient return even if later observed winds are below.
        # Therefore the first-return interval starts at the last contiguous below-threshold point.
        gaps=before.loc[~before.finite_wind,'lag_h']
        lower=float(gaps.iloc[0]-6) if len(gaps) else (float(before.lag_h.max()) if len(before) else 0.)
        unknownstate=bool((sub.loc[sub.finite_wind,'state_class']=='C').any())
        tropicalbinary=1. if len(thits) else (0. if complete and not unknownstate else np.nan)
        out.update({f'return{h}':binary,f'return{h}_status':'observed_return' if len(hits) else ('complete_followup_no_return' if complete else 'unknown_incomplete_followup'),
          f'followup{h}_complete':complete,f'followup{h}_finite_n':int(sub.finite_wind.sum()),
          f'return{h}_first_observed_lag_h':first,
          f'return{h}_first_interval_lower_h':lower if len(hits) else np.nan,
          f'return{h}_first_interval_upper_h':first,
          f'return{h}_after_first_gap':bool(len(hits) and (~before.finite_wind).any()),
          f'return{h}_tropical_confirmed':tropicalbinary,
          f'followup{h}_other_state_n':int(sub.state_class.eq('B').sum()),
          f'followup{h}_unknown_state_n':int(sub.state_class.eq('C').sum()),
          f'censor{h}_first_lag_h':first_censor if np.isfinite(first_censor) and first_censor<=h else np.nan,
          f'censor{h}_reason':d.loc[d.lag_h.eq(first_censor),'censor_reason'].iloc[0] if np.isfinite(first_censor) and first_censor<=h else '',
          f'return{h}_contiguous_event_lag_h':first_event if np.isfinite(first_event) and first_event<=h else np.nan})
    at24=d[d.lag_h.eq(24)].iloc[0]
    out['delta_v24_forward_kt']=float(at24.wind_kt-event['wind_t0_kt']) if at24.finite_wind else np.nan
    out['delta_v24_status']='observed_exact_endpoint' if at24.finite_wind else at24.censor_reason
    out['delta_v24_intermediate_gaps']=int((~d[d.lag_h.le(24)].finite_wind).sum())
    return out,d

def synthetic(state):
    def track(winds,states=None,hours=None,sid='SYNTH'):
        hs=np.arange(len(winds))*6 if hours is None else hours
        d=pd.DataFrame({'sid':sid,'UTC':[pd.Timestamp('2000-01-01',tz='UTC')+pd.Timedelta(hours=int(h)) for h in hs],
          'wind_kt':winds,'lat':10.,'lon':179.,'basin':'NA','dist2land_km':100.,'USA_PRES':990.,
          'USA_STATUS':states or ['TS']*len(winds),'NATURE':'TS','source_line_number':np.arange(len(winds))+3})
        st=[state(s,'TS',True) for s in d.USA_STATUS];d['state_class']=[x[0] for x in st];d['state_source']=[x[1] for x in st]
        d['state_label']=d.state_class.map({'A':'tropical','B':'other','C':'unknown'});d['agency_conflict_nature_mx']=False
        return d
    g=track([35,45,50,60,50,55,80,40,40,40,40,40,40])
    rows,_=history_candidates(g);e=next(x for x in rows if x['eligible'])
    check('synthetic_fixed_past_threshold',e['Vref_kt']==57.5)
    check('synthetic_future_peak_not_threshold',e['Vref_kt']!=35+.9*(80-35))
    p,_=history_candidates(g[g.UTC<=e['t0']]);check('synthetic_prefix_invariance',p[0]==e)
    changed=g.copy();changed.loc[changed.UTC>e['t0'],'wind_kt']=200
    z,_=history_candidates(changed);check('synthetic_future_changes_leave_entry_unchanged',z[0]==e)
    f,d=followup(g,e);check('synthetic_fixed_threshold_return',f['return48']==1 and f['return48_first_observed_lag_h']==12)
    g2=track([35,45,50,60,50,40]);e2=next(x for x in history_candidates(g2)[0] if x['eligible'])
    check('synthetic_track_end_censored_not_zero',np.isnan(followup(g2,e2)[0]['return48']))
    g3=g.copy();g3.loc[5,'wind_kt']=np.nan
    f,d=followup(g3,e);check('synthetic_gap_then_confirmed_return',f['return48']==1 and f['return48_after_first_gap'] and f['censor48_first_lag_h']==6)
    g4=g.copy();g4.loc[5,'state_class']='B';g4.loc[6,'state_class']='A'
    check('synthetic_transient_nontropical_not_death',followup(g4,e)[0]['return48']==1)
    g4b=g.copy();g4b.loc[5,'wind_kt']=np.nan;g4b.loc[6,'wind_kt']=40;g4b.loc[7,'wind_kt']=80
    f4b,_=followup(g4b,e)
    check('synthetic_first_return_lowerbound_before_first_gap',f4b['return48_first_interval_lower_h']==0 and f4b['return48_first_interval_upper_h']==18)
    g5=track([35,45,50,60,50,40,40,40,40,40,40,40,40])
    e5=next(x for x in history_candidates(g5)[0] if x['eligible'])
    check('synthetic_full_followup_no_return_zero',followup(g5,e5)[0]['return48']==0)
    gap=track([35,45,50,60,50],hours=[0,6,12,18,30])
    check('synthetic_illegal_crossing_gap_rejected',not any(x['eligible'] for x in history_candidates(gap)[0]))
    missing=track([35,45,50,60,np.nan,50])
    check('synthetic_no_future_interpolation',not any(x['eligible'] for x in history_candidates(missing)[0]))
    unknown=track([35,45,50,60,50],states=['TS','TS','TS','TS','??'])
    check('synthetic_unknown_current_state_excluded',not history_candidates(unknown)[0][0]['eligible'])
    dl=(((-179)-179+180)%360)-180
    check('synthetic_dateline_motion_short_arc',dl==2)
    check('state_fallback_priority',state('EX','TS',True)[0]=='B' and state('','TS',True)[0]=='A' and state('??','TS',True)[0]=='C')

def crossmap(events):
    candidate=register(R/'review_completion_v3_2/audit/state_dependency/H1_candidate123_event_state_join.csv','frozen123 candidate pairs')
    c=readcsv(candidate,usecols=['sid','t0_internal','t0_final']).drop_duplicates('sid')
    sample=register(R/'review_completion_v3_2/e2_e3/H1/samples.csv','SP-corrected H1 actual sample memberships')
    s=readcsv(sample)
    counts=s.groupby(['comparison','variable']).sid.nunique().reset_index(name='n_sid')
    csv(counts,'audit/p3_h1_membership_source_counts.csv')
    new=s[s.comparison.eq('new_available')]
    assert len(new),'H1 final new_available comparison missing'
    sets=[set(z.sid) for _,z in new.groupby('variable')]
    common=set.intersection(*sets)
    check('H1_final_actual_common117',len(common)==117,counts.to_dict('records'))
    multi=register(PROJECT/'IBTrACS_interdecadal_v19/tables/analysis_per_SID_main_window.csv','frozen main SID and123 recurrent group')
    m=readcsv(multi,usecols=['sid','near_lmi_episode_count','complete_common_window'])
    bool_complete=m.complete_common_window.astype(str).str.strip().str.lower().isin(['true','1','1.0'])
    recurrent=set(m.loc[bool_complete & m.near_lmi_episode_count.gt(1),'sid'])
    check('frozen_recurrent123',len(recurrent)==123)
    p3=set(events.loc[events.main_first_per_sid,'sid'])
    ids=sorted(p3|common|set(c.sid)|recurrent)
    cross=pd.DataFrame({'sid':ids})
    cross['P3_main']=cross.sid.isin(p3);cross['H1_current117']=cross.sid.isin(common)
    cross['H1_candidate123']=cross.sid.isin(c.sid);cross['frozen_recurrent123']=cross.sid.isin(recurrent)
    cross=cross.merge(events.loc[events.main_first_per_sid,['sid','event_id','t0','Vref_kt']],on='sid',how='left')
    cross=cross.merge(c,on='sid',how='left')
    cross['P3_t0_matches_H1_internal']=pd.to_datetime(cross.t0,utc=True)==pd.to_datetime(cross.t0_internal,utc=True)
    cross['P3_t0_matches_H1_final']=pd.to_datetime(cross.t0,utc=True)==pd.to_datetime(cross.t0_final,utc=True)
    csv(cross,'tables/p3_old_new_cohort_crossmap.csv')
    summary=[]
    for column in ['H1_current117','H1_candidate123','frozen_recurrent123']:
        summary.append(dict(cohort=column,old_sid_n=int(cross[column].sum()),overlap_P3_main=int((cross[column]&cross.P3_main).sum()),
          old_only=int((cross[column]&~cross.P3_main).sum()),P3_only=int((~cross[column]&cross.P3_main).sum())))
    csv(pd.DataFrame(summary),'tables/p3_old_new_cohort_crossmap_summary.csv')
    return summary

def main():
    start=now();spec=freeze();state=dictionary();synthetic(state)
    raw=load_raw(state)
    all_candidates=[];sid_audit=[]
    groups={sid:g.reset_index(drop=True) for sid,g in raw.groupby('sid',sort=False)}
    for sid,g in groups.items():
        rows,a=history_candidates(g);all_candidates.extend(rows);sid_audit.append(a)
    candidates=pd.DataFrame(all_candidates).sort_values(['sid','t0']).reset_index(drop=True)
    events=candidates[candidates.eligible].copy().reset_index(drop=True)
    events['main_first_per_sid']=~events.sid.duplicated()
    events['all_nonoverlap96h']=False
    for sid,g in events.groupby('sid',sort=False):
        last=None
        for i,t in zip(g.index,g.t0):
            if last is None or (t-last)>=pd.Timedelta(hours=96):
                events.loc[i,'all_nonoverlap96h']=True;last=t
    # All at-time enrollment decisions are completed before follow-up is read.
    csv(candidates,'data/p3_departure_candidates.csv')
    csv(pd.DataFrame(sid_audit),'audit/p3_sid_eligibility.csv')
    csv(events,'data/p3_event_baseline_before_outcomes.csv')
    follow=[];long=[]
    for e in events.to_dict('records'):
        f,d=followup(groups[e['sid']],e);follow.append(f);long.append(d)
    final=pd.concat([events,pd.DataFrame(follow)],axis=1)
    longitudinal=pd.concat(long,ignore_index=True)
    csv(final,'data/p3_all_eligible_events.csv');csv(final[final.main_first_per_sid],'data/p3_main_first_events.csv')
    csv(final[final.all_nonoverlap96h],'data/p3_all_nonoverlap96h_events.csv')
    csv(longitudinal,'data/p3_longitudinal_followup.csv')
    main_events=final[final.main_first_per_sid]
    baselinefields=[x for x in events.columns if x not in ['event_id','sid']]
    timing=[]
    for f in baselinefields:
        timing.append(dict(variable=f,role='cohort selection flag' if f in ['eligible','exclusion_reason','main_first_per_sid','all_nonoverlap96h'] else 'baseline or identity',
          max_source_UTC_rule='t0-6h' if f.startswith('Vmax_past') or f in ['Vref_kt','historical_gain_kt'] else 't0',
          future_information=False,source='raw USA_WIND or accepted state/position metadata; frozen deterministic transform'))
    csv(pd.DataFrame(timing),'audit/p3_variable_timing_dictionary.csv')
    check('actual_unique_event_id',final.event_id.is_unique)
    check('actual_unique_main_sid',main_events.sid.is_unique)
    check('actual_threshold_past_source',pd.to_datetime(final.Vmax_past_latest_source_UTC,utc=True).le(pd.to_datetime(final.t0,utc=True)-pd.Timedelta(hours=6)).all())
    check('actual_baseline_source_no_future',pd.to_datetime(final.baseline_max_source_UTC,utc=True).le(pd.to_datetime(final.t0,utc=True)).all())
    check('actual_threshold_exact',np.allclose(final.Vref_kt,final.genesis_wind_kt+.9*(final.Vmax_past_kt-final.genesis_wind_kt)))
    check('actual_all_current_tropical',final.t0_state_class.eq('A').all())
    check('actual_historical_gain_atleast15',final.historical_gain_kt.ge(15).all())
    check('actual_exact6h_crossings',(pd.to_datetime(final.t0,utc=True)-pd.to_datetime(final.previous_UTC,utc=True)).eq(pd.Timedelta(hours=6)).all())
    check('actual_all_crossing_directions',(final.wind_prev_kt.ge(final.Vref_kt)&final.wind_t0_kt.lt(final.Vref_kt)).all())
    check('actual_missing_not_zero',final.loc[final.return48_status.eq('unknown_incomplete_followup'),'return48'].isna().all())
    check('actual_all_full_followup_zero',(final.loc[final.return48.eq(0),'followup48_complete']).all())
    spacing=final[final.all_nonoverlap96h].groupby('sid').t0.diff().dropna()
    check('actual_nonoverlap96h',spacing.ge(pd.Timedelta(hours=96)).all())
    # Actual-data prefix invariance sample, chosen by fixed evenly spaced row index.
    for k in np.unique(np.linspace(0,len(main_events)-1,min(40,len(main_events)),dtype=int)):
        e=main_events.iloc[k];g=groups[e.sid];prefix=g[g.UTC<=e.t0]
        rr,_=history_candidates(prefix);match=[x for x in rr if x['event_id']==e.event_id]
        check('actual_prefix_invariance_'+str(k),len(match)==1 and match[0]['eligible'] and match[0]['Vref_kt']==e.Vref_kt)
    mapping=crossmap(final)
    summaries=[]
    for label,sel in [('main_first',final.main_first_per_sid),('all_nonoverlap96h',final.all_nonoverlap96h),('all_eligible_candidates',pd.Series(True,index=final.index))]:
        d=final[sel]
        for h in [48,96]:
            summaries.append(dict(cohort=label,event_n=len(d),sid_n=d.sid.nunique(),horizon_h=h,
               observed_return_n=int(d[f'return{h}'].eq(1).sum()),complete_no_return_n=int(d[f'return{h}'].eq(0).sum()),
               unknown_n=int(d[f'return{h}'].isna().sum()),confirmed_return_after_gap_n=int(d[f'return{h}_after_first_gap'].sum()),
               tropical_confirmed_return_n=int(d[f'return{h}_tropical_confirmed'].eq(1).sum()),
               exact24h_delta_n=int(d.delta_v24_forward_kt.notna().sum())))
    csv(pd.DataFrame(summaries),'tables/p3_cohort_outcome_support.csv')
    csv(main_events.groupby(['basin_genesis','return48_status'],dropna=False).size().reset_index(name='n'),'tables/p3_main_basin_followup.csv')
    csv(main_events.groupby(['censor48_reason'],dropna=False).size().reset_index(name='n'),'tables/p3_main_censor_reasons.csv')
    csv(candidates.groupby(['eligible','exclusion_reason'],dropna=False).size().reset_index(name='n'),'tables/p3_candidate_exclusion_counts.csv')
    out_json('audit/p3_source_identity.json',{'inputs':INPUTS,'spec_sha256':sha(spec),'code_sha256':sha(__file__)})
    out_json('audit/p3_cohort_tests.json',{'status':'PASS','tests_n':len(TESTS),'tests':TESTS,'scope':'time-zero enrollment and fixed-threshold followup only; no ERA5 extraction or effect estimation'})
    summary={'status':'COHORT_CONSTRUCTED_AND_TESTED_EFFECT_ESTIMATION_NOT_RUN','start_utc':start,'end_utc':now(),
      'python':sys.executable,'raw_source':str(RAW),'raw_sha256':RAW_SHA,'candidate_raw_sid_n':raw.sid.nunique(),'candidate_raw_synoptic_rows':len(raw),
      'crossing_candidates_n':len(candidates),'eligible_events_n':len(final),'main_sid_n':len(main_events),
      'nonoverlap96h_events_n':int(final.all_nonoverlap96h.sum()),'tests_n':len(TESTS),
      'outcome_support':summaries,'crossmap':mapping,'spec_sha256':sha(spec),
      'effects':'NOT_RUN; requires complete environmental inputs and frozen shared effect specification'}
    out_json('reports/p3_cohort_status.json',summary)
    report='# P3 当时信息队列构建记录\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n使用同快照原始 USA_WIND，不调用冻结生命周期的未来 LMI 或完整四阶段准入。所有原始资料和既有结果只读。main 与 all_nonoverlap96h 是两个独立标签，不替换原117配对、123候选或旧48小时模型。原始观测的有限精度、无内核结构资料及无未来质量准入均留在后续识别讨论。缺口后观察到返回可以确认二元结局，但不冒充无缺口的事件时刻。\n'
    (V5/'reports/p3_cohort_report_zh.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except BaseException:
        out_json('audit/p3_failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json',
          {'UTC':now(),'traceback':traceback.format_exc(),'tests':TESTS})
        raise

