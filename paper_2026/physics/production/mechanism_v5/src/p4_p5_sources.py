"""P4/P5 frozen source/time geometry and matched-intensity source construction.
No raw ERA5 reads, environmental effects, regression or bootstrap.
All output is scoped to mechanism_revision_v5/p4_p5.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,traceback,sys
import numpy as np
import pandas as pd

V5=Path(__file__).resolve().parents[1];R=V5.parent;B=R/'review_completion_v3_2';P=R.parents[1]
O=V5/'p4_p5'
SID=P/'IBTrACS_interdecadal_v19/tables/analysis_per_SID_main_window.csv'
TRACK=B/'sources/normalized_track_sp_qc_v3_2.parquet'
X=['X1_sst_c','X2_shear_ms','X3_mid_rh_pct','X4_d200_1e5s','X5_minus_omega500_pa_s','X6_signed_vo850_1e5s']
TIMES=['genesis_time','escalation_time','maturity_time','resolution_time','termination_time','lmi_time','lmi_last_time']
STAGE=['D','E','F','Res'];SEGMENT=['pre','middle','post']
CHECKS=[];INPUTS=[]

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(2**20),b''):h.update(z)
 return h.hexdigest()
def write_json(name,value):
 p=O/name;assert O.resolve() in p.resolve().parents
 p.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def write_csv(name,d):
 p=O/name;assert O.resolve() in p.resolve().parents
 d.to_csv(p,index=False,encoding='utf-8-sig')
def check(name,value,detail=None):
 CHECKS.append({'test':name,'passed':bool(value),'detail':detail})
 assert value,(name,detail)
def register(p,role,expected=None):
 h=sha(p)
 if expected:assert h==expected,(p,h,expected)
 INPUTS.append({'path':str(p),'role':role,'sha256':h,'bytes':p.stat().st_size,'expected':expected})
 return p
def readcsv(p,**kw):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],**kw)
def boolcol(s):
 if pd.api.types.is_bool_dtype(s):return s.fillna(False)
 v=s.astype(str).str.strip().str.lower()
 assert v.isin(['true','false','1','0','1.0','0.0','nan','none','']).all()
 return v.isin(['true','1','1.0'])
def hours(a,b):return float((b-a)/pd.Timedelta(hours=1))
def overlap(a,b,c,d):
 return max(0.,hours(max(a,c),min(b,d)))

def freeze():
 O.mkdir(exist_ok=True)
 spec={'schema':'P4_P5_frozen_source_geometry_v1','frozen_utc':now(),
  'parent_approval':'Parent approved10kt bins, within-SID shared-bin weighting and no environmental-outcome selection',
  'source_spec':'config/user_amendment_v5_1_record.md (faithful execution excerpt; not verbatim user original)',
  'population':'all2368 frozen v19 main-windowSID1982–2023; no complete_common filter silently added',
  'sources':[str(SID),str(TRACK)],
  'P4':{'boundaries':{'D_to_E':'escalation_time','E_to_F':'maturity_time'},'lags_h':list(range(-24,25,6)),
        'position':'exact observed normalized_track SID UTC location; no interpolation/extrapolation; absent key remains request with missing support',
        'temporal_support':'store point support separately from forward legal6h interval; no require full±24h survival',
        'environment':'read-only copy of accepted SP-corrected *_value and *_qc where source row exists; never older *_raw/short cache fallback',
        'stage':'frozen current stage and exact elapsed interval boundaries; lag may cross more than one stage',
        'D_fraction':'D/(D+E), continuous metadata; no new environment-dependent grouping'},
  'P5':{'Tpre':'elapsed genesis to first LMI','Tpost':'elapsed last LMI to frozen termination',
        'A':'(Tpre−Tpost)/(Tpre+Tpost);positive=longer pre; compute abs individually',
        'segments':{'pre':'[genesis,firstLMI)','middle':'[firstLMI,lastLMI)','post':'[lastLMI,termination)'},
        'stages':'[genesis,escalation),[escalation,maturity),[maturity,resolution),[resolution,termination)',
        'coverage':'sum actual overlaps of frozen legal forward6h intervals with each stage×segment intersection; elapsed distinct; no endpoint+6',
        'matched_intensity':{'bins_kt':'[35+10j,45+10j),integer j>=0,observedrangeonly',
          'eligible_record':'left endpoint of frozen legal6h interval with positive pre/post segment intersection; finite frozen wind, filled flag retained',
          'representative':'closest to bin midpoint; ties pre earliestUTC,post latestUTC; never environment-based',
          'repeated_crossings':'save original interval IDs and bin-visit episode counts; no joining gaps or out-of-bin segments',
          'shared_support':'withinSID bothsides have eligible samebin; source matching before environment QC; paired valid/QC flags recorded pervariable',
          'weight':'eachSID sum of shared-bin pair weights=1; eachpairweight=1/kSID; subsequent pervariable QC needs explicit renormalization/target statement',
          'difference_convention':'post minus pre; retain actual wind difference and require residualwind control for background-standardized analyses',
          'interpretation':'coarse same-intensity-bin pairedpath; not exactequalwind or causal hysteresis'}
        },
  'forbidden':'no raw NC read, no new effect estimate, no regression, no bootstrap, no H2 acceptance or legacy writes'}
 p=O/'source_spec.json'
 if p.exists():
  existing=json.loads(p.read_text('utf-8'));new=dict(spec);existing.pop('frozen_utc',None);new.pop('frozen_utc',None)
  assert existing==new,'Frozen P4/P5 spec changed'
 else:write_json('source_spec.json',spec)
 return p

def load():
 catalog=readcsv(register(B/'authoritative_data_catalog_v3_2.csv','authoritative v3.2 catalog',
 '214eb1e2426f35d69a3c96238b1558d892df70e2c9ef504b851ab3433db704b3')).set_index('relative_path')
 register(SID,'frozen v19 main SID source','6c419311f23cfdde39d5b753303ce7d032bf5830821ec16ce1fa8544efe2dd6a')
 register(TRACK,'accepted SP normalized track',catalog.loc['sources/normalized_track_sp_qc_v3_2.parquet','sha256'])
 s=readcsv(SID)
 for c in TIMES:s[c]=pd.to_datetime(s[c],utc=True)
 cols=['sid','UTC','next_UTC','stage_v3','F_stage','legal_interval_v3','interval_hours_v3','terminal_point_v3',
       'wind_kt','wind_filled_flag','lat','lon','dist2land','record_index','maturity_component','component_v3']+[c+suffix for c in X for suffix in ['_value','_qc']]
 t=pd.read_parquet(TRACK,columns=cols).sort_values(['sid','UTC']).reset_index(drop=True)
 t.UTC=pd.to_datetime(t.UTC,utc=True);t.next_UTC=pd.to_datetime(t.next_UTC,utc=True)
 for c in ['legal_interval_v3','terminal_point_v3']+[c+'_qc' for c in X]:t[c]=boolcol(t[c])
 t['within_sid_source_index']=t.groupby('sid').cumcount()
 t['frozen_interval_id']=t.sid+'|'+t.UTC.dt.strftime('%Y%m%dT%H%MZ')
 check('frozen_main2368',len(s)==2368 and s.sid.is_unique)
 check('normalized64210_sameSID',len(t)==64210 and set(t.sid)==set(s.sid))
 check('no_duplicateSID_UTC',not t.duplicated(['sid','UTC']).any())
 check('NA_basin_preserved',s.basin_genesis.eq('NA').any())
 nextw=t.groupby('sid').wind_kt.shift(-1)
 expected=(t.next_UTC-t.UTC).dt.total_seconds().eq(21600)&t.wind_kt.notna()&nextw.notna()
 check('legal6h_reproduces_frozen',expected.eq(t.legal_interval_v3).all())
 check('terminal_noextra6h',(~t.loc[t.terminal_point_v3,'legal_interval_v3']).all())
 check('source_coordinate_valid',t[['lat','lon']].notna().all(axis=1).all())
 return s,t,catalog

def p4(s,t):
 lookup=t.set_index(['sid','UTC']);out=[]
 for x in s.itertuples():
  for label,col in [('D_to_E','escalation_time'),('E_to_F','maturity_time')]:
   boundary=getattr(x,col)
   for lag in range(-24,25,6):
    utc=boundary+pd.Timedelta(hours=lag);key=(x.sid,utc);exists=key in lookup.index
    z=lookup.loc[key] if exists else None
    reason='observed_exactUTC' if exists else ('before_frozen_genesis' if utc<x.genesis_time else 'after_frozen_termination' if utc>x.termination_time else 'missing_frozen_track_UTC')
    rec=dict(request_id=f'{x.sid}|{label}|{lag:+03d}',sid=x.sid,boundary=label,t0=boundary,lag_h=lag,UTC=utc,
      basin_genesis=x.basin_genesis,genesis_year=x.genesis_year,D_h=x.development_duration_hours,E_h=x.escalation_duration_hours,
      D_over_DplusE=x.development_duration_hours/(x.development_duration_hours+x.escalation_duration_hours),
      actual_track_point=exists,source_support_reason=reason,lat=float(z.lat) if exists else np.nan,lon=float(z.lon) if exists else np.nan,
      wind_kt=float(z.wind_kt) if exists else np.nan,wind_filled_flag=float(z.wind_filled_flag) if exists else np.nan,
      frozen_stage=z.stage_v3 if exists else '',frozen_component=z.component_v3 if exists else '',
      frozen_record_index=float(z.record_index) if exists else np.nan,source_interval_id=z.frozen_interval_id if exists else '',
      legal_forward6h=bool(z.legal_interval_v3) if exists else False,
      legal_covered_h=float(z.interval_hours_v3) if exists and z.legal_interval_v3 else 0.,
      point_wind_finite=bool(np.isfinite(z.wind_kt)) if exists else False,
      requested_center_type='actual_track_at_lag',position_interpolated_or_extrapolated=False)
    valid=[]
    for var in X:
     ok=bool(exists and z[var+'_qc'] and np.isfinite(z[var+'_value']))
     rec[var+'_source_qc']=bool(z[var+'_qc']) if exists else False
     rec[var+'_actual_valid']=ok
     rec[var+'_value']=float(z[var+'_value']) if ok else np.nan
     valid.append(ok)
    rec['six_factor_joint_valid']=all(valid)
    out.append(rec)
 d=pd.DataFrame(out)
 check('P4_total_requested42624',len(d)==2368*2*9)
 check('P4_unique_request',d.request_id.is_unique)
 check('P4_missingposition_neverfilled',d.loc[~d.actual_track_point,['lat','lon']].isna().all().all())
 check('P4_nofullwindow_pruning',d.groupby(['sid','boundary']).size().eq(9).all())
 check('P4_allboundary0points_present',d.loc[d.lag_h.eq(0),'actual_track_point'].all())
 write_csv('P4_boundary_requests.csv',d)
 counts=[]
 for (bound,lag),g in d.groupby(['boundary','lag_h']):
  rec=dict(boundary=bound,lag_h=lag,requested_sid_n=len(g),actual_point_sid_n=int(g.actual_track_point.sum()),
    legal_interval_sid_n=int(g.legal_forward6h.sum()),finite_wind_sid_n=int(g.point_wind_finite.sum()),six_factor_joint_sid_n=int(g.six_factor_joint_valid.sum()))
  for x in X:rec[x+'_valid_sid_n']=int(g[x+'_actual_valid'].sum())
  counts.append(rec)
 write_csv('P4_lag_support.csv',pd.DataFrame(counts))
 write_csv('P4_support_by_stage.csv',d.groupby(['boundary','lag_h','frozen_stage','source_support_reason'],dropna=False).size().reset_index(name='n_sid'))
 return d

def geometries(s,t):
 sidrows=[];intersections=[];source_intervals=[]
 for row in s.itertuples():
  a=t[t.sid.eq(row.sid)]
  bounds=[row.genesis_time,row.escalation_time,row.maturity_time,row.resolution_time,row.termination_time]
  segs=[(row.genesis_time,row.lmi_time),(row.lmi_time,row.lmi_last_time),(row.lmi_last_time,row.termination_time)]
  pre=hours(row.genesis_time,row.lmi_time);post=hours(row.lmi_last_time,row.termination_time)
  A=(pre-post)/(pre+post)
  totlegal=0.;coveredseg={name:0. for name in SEGMENT}
  intervals=[]
  for z in a.itertuples():
   legal=bool(z.legal_interval_v3)
   end=z.next_UTC if legal else z.UTC
   rec=dict(sid=row.sid,UTC=z.UTC,next_UTC=z.next_UTC,frozen_interval_id=z.frozen_interval_id,
    frozen_record_index=z.record_index,source_index=z.within_sid_source_index,stage=z.stage_v3,
    lat=z.lat,lon=z.lon,wind_kt=z.wind_kt,wind_filled_flag=z.wind_filled_flag,
    legal6h=legal,legal_covered_h=6. if legal else 0.)
   for name,(c,d) in zip(SEGMENT,segs):
    h=overlap(z.UTC,end,c,d) if legal else 0.;rec[name+'_covered_h']=h;coveredseg[name]+=h
   for x in X:rec[x+'_value']=getattr(z,x+'_value');rec[x+'_qc']=bool(getattr(z,x+'_qc'))
   source_intervals.append(rec);intervals.append((z.UTC,end,legal))
   totlegal+=6 if legal else 0
  fields=dict(sid=row.sid,genesis_year=row.genesis_year,basin_genesis=row.basin_genesis,
    genesis_UTC=row.genesis_time,first_LMI_UTC=row.lmi_time,last_LMI_UTC=row.lmi_last_time,termination_UTC=row.termination_time,
    Tpre_elapsed_h=pre,Tpost_elapsed_h=post,middle_elapsed_h=hours(row.lmi_time,row.lmi_last_time),
    A_signed_from_frozen_times=A,A_absolute_from_frozen_times=abs(A),
    source_Tpre_h=row.pre_lmi_duration_hours,source_Tpost_h=row.post_lmi_duration_hours,source_A_signed=row.A_signed,source_A_absolute=row.A_absolute,
    Tpre_covered_h=coveredseg['pre'],Tpost_covered_h=coveredseg['post'],middle_covered_h=coveredseg['middle'],
    whole_lifecycle_elapsed_h=hours(row.genesis_time,row.termination_time),whole_legal6h_covered_h=totlegal,
    D_plus_E_h=row.development_duration_hours+row.escalation_duration_hours,Res_h=row.resolution_duration_hours,
    Tpre_minus_DplusE_h=pre-row.development_duration_hours-row.escalation_duration_hours,
    Tpost_minus_Res_h=post-row.resolution_duration_hours,
    complete_common_window=bool(str(row.complete_common_window).strip().lower() in ['true','1','1.0']))
  sidrows.append(fields)
  for stage,(a0,b0) in zip(STAGE,zip(bounds[:-1],bounds[1:])):
   for seg,(c0,d0) in zip(SEGMENT,segs):
    left=max(a0,c0);right=min(b0,d0);elapsed=max(0.,hours(left,right))
    covered=sum(overlap(x,y,left,right) for x,y,lawful in intervals if lawful) if elapsed>0 else 0.
    intersections.append(dict(sid=row.sid,stage=stage,segment=seg,stage_start_UTC=a0,stage_end_UTC=b0,segment_start_UTC=c0,segment_end_UTC=d0,
      intersection_start_UTC=left if elapsed>0 else pd.NaT,intersection_end_UTC=right if elapsed>0 else pd.NaT,
      elapsed_h=elapsed,legal6h_covered_h=covered,uncovered_h=elapsed-covered,
      boundary_convention='left-closed/right-open; no extra terminal interval'))
 d=pd.DataFrame(sidrows);i=pd.DataFrame(intersections);ledger=pd.DataFrame(source_intervals)
 check('P5_exact_Tpre',(d.Tpre_elapsed_h==d.source_Tpre_h).all())
 check('P5_exact_Tpost',(d.Tpost_elapsed_h==d.source_Tpost_h).all())
 check('P5_A_matches_tolerance',np.allclose(d.A_signed_from_frozen_times,d.source_A_signed,atol=1e-14,rtol=0))
 check('P5_absA_individual_matches',np.allclose(d.A_absolute_from_frozen_times,d.source_A_absolute,atol=1e-14,rtol=0))
 check('P5_intersection_elapsed_not_negative',i.elapsed_h.ge(0).all())
 check('P5_covered_between0andelapsed',(i.legal6h_covered_h.ge(0)&i.legal6h_covered_h.le(i.elapsed_h)).all())
 g=i.groupby('sid')[['elapsed_h','legal6h_covered_h']].sum().reindex(d.sid)
 check('P5_intersection_elapsed_partition',np.allclose(g.elapsed_h,d.whole_lifecycle_elapsed_h))
 check('P5_intersection_coverage_partition',np.allclose(g.legal6h_covered_h,d.whole_legal6h_covered_h))
 check('P5_interval_segments_partition',np.allclose(ledger[['pre_covered_h','middle_covered_h','post_covered_h']].sum(axis=1),ledger.legal_covered_h))
 check('P5_no_terminal_6h',ledger.loc[ledger.next_UTC.isna(),'legal_covered_h'].eq(0).all())
 write_csv('P5_per_SID_time_audit.csv',d);write_csv('P5_stage_segment_intersections.csv',i);write_csv('P5_legal_interval_ledger.csv',ledger)
 return d,i,ledger

def match_intensity(s,ledger):
 candidates=[]
 for side in ['pre','post']:
  q=ledger[ledger[side+'_covered_h'].gt(0)&ledger.legal6h&ledger.wind_kt.ge(35)&ledger.wind_kt.notna()].copy()
  q['side']=side;q['bin_index']=np.floor((q.wind_kt-35)/10).astype(int)
  q['bin_lower_kt']=35+q.bin_index*10;q['bin_upper_kt']=q.bin_lower_kt+10;q['bin_center_kt']=q.bin_lower_kt+5
  q['distance_to_center_kt']=(q.wind_kt-q.bin_center_kt).abs()
  q=q.sort_values(['sid','bin_index','UTC']).copy()
  previndex=q.groupby(['sid','bin_index']).source_index.shift()
  prevtime=q.groupby(['sid','bin_index']).UTC.shift()
  q['new_bin_visit']=q.source_index.sub(previndex).ne(1)|(q.UTC-prevtime).ne(pd.Timedelta(hours=6))
  q['bin_visit_number']=q.groupby(['sid','bin_index']).new_bin_visit.cumsum().astype(int)
  q['bin_visit_count']=q.groupby(['sid','bin_index']).new_bin_visit.transform('sum')
  q['candidate_record_count']=q.groupby(['sid','bin_index']).sid.transform('size')
  q['representative']=False
  # First sort criterion always raw intensity closeness, then prespecified time direction.
  selected=q.sort_values(['sid','bin_index','distance_to_center_kt','UTC'],ascending=[True,True,True,side=='pre'],kind='stable').drop_duplicates(['sid','bin_index'])
  q.loc[selected.index,'representative']=True
  candidates.append(q)
 cand=pd.concat(candidates,ignore_index=True)
 pre=cand[cand.side.eq('pre')&cand.representative].copy()
 post=cand[cand.side.eq('post')&cand.representative].copy()
 pairs=pre.merge(post,on=['sid','bin_index','bin_lower_kt','bin_upper_kt','bin_center_kt'],suffixes=('_pre','_post'),validate='one_to_one')
 pairs['shared_bin_n_sid']=pairs.groupby('sid').sid.transform('size')
 pairs['pair_weight_within_sid']=1/pairs.shared_bin_n_sid
 pairs['wind_post_minus_pre_kt']=pairs.wind_kt_post-pairs.wind_kt_pre
 pairs['elapsed_between_representatives_h']=(pairs.UTC_post-pairs.UTC_pre).dt.total_seconds()/3600
 pairs['pair_id']=pairs.sid+'|Vbin'+pairs.bin_index.astype(str)
 for x in X:
  pairs[x+'_paired_valid']=pairs[x+'_qc_pre']&pairs[x+'_qc_post']&pairs[x+'_value_pre'].notna()&pairs[x+'_value_post'].notna()
 pairs['six_factor_joint_paired_valid']=pairs[[x+'_paired_valid' for x in X]].all(axis=1)
 pairs=pairs.merge(s[['sid','A_signed','A_absolute','pre_lmi_duration_hours','post_lmi_duration_hours','genesis_year','basin_genesis']],on='sid',validate='many_to_one')
 check('P5_matching_uniqueSIDbin',not pairs.duplicated(['sid','bin_index']).any())
 check('P5_matching_temporal_order',pairs.UTC_pre.lt(pairs.UTC_post).all())
 check('P5_matching_legal_intervals',pairs.legal6h_pre.all() and pairs.legal6h_post.all())
 check('P5_matching_positive_prepost_coverage',pairs.pre_covered_h_pre.gt(0).all() and pairs.post_covered_h_post.gt(0).all())
 check('P5_matching_weights_eachSID1',np.allclose(pairs.groupby('sid').pair_weight_within_sid.sum(),1))
 check('P5_matching_binnedactualwind',(pairs.wind_kt_pre.ge(pairs.bin_lower_kt)&pairs.wind_kt_pre.lt(pairs.bin_upper_kt)&pairs.wind_kt_post.ge(pairs.bin_lower_kt)&pairs.wind_kt_post.lt(pairs.bin_upper_kt)).all())
 check('P5_matching_winddifference_under10',pairs.wind_post_minus_pre_kt.abs().lt(10).all())
 # Independently recompute a fixed subset of representative choices from candidates.
 for side,ascending in [('pre',True),('post',False)]:
  for (sid,binid),g in cand[cand.side.eq(side)].groupby(['sid','bin_index'],sort=True):
   chosen=g[g.representative]
   assert len(chosen)==1
   best=g[g.distance_to_center_kt.eq(g.distance_to_center_kt.min())].UTC
   expected=best.min() if ascending else best.max()
   assert chosen.UTC.iloc[0]==expected
 check('P5_all_representative_ties_verified',True)
 support=[]
 for sid in s.sid:
  a=cand[cand.sid.eq(sid)];g=pairs[pairs.sid.eq(sid)]
  support.append(dict(sid=sid,pre_candidate_records=int(a.side.eq('pre').sum()),post_candidate_records=int(a.side.eq('post').sum()),
    pre_bin_n=a.loc[a.side.eq('pre'),'bin_index'].nunique(),post_bin_n=a.loc[a.side.eq('post'),'bin_index'].nunique(),
    shared_bin_n=len(g),paired_support=len(g)>0,source_reason='matched_raw_intensity_bins' if len(g) else 'no_common_legal_pre_post_bin',
    six_factor_joint_valid_bin_n=int(g.six_factor_joint_paired_valid.sum()) if len(g) else 0))
 write_csv('P5_intensity_path_candidates.csv',cand);write_csv('P5_matched_intensity_pairs.csv',pairs);write_csv('P5_matching_support_per_SID.csv',pd.DataFrame(support))
 return pairs,cand

def reuse_index(catalog):
 paths=['h3_h4/H3/new_available/effects.csv','h3_h4/H3_exposures_new_available.csv',
 'h3_h4/H3_exact_hour_source_ledger.csv','h3_h4/H3_window_eligibility_four_scopes.csv',
 'h3_h4/H4/common_controls/new_available/effects.csv','h3_h4/H4/common_controls/new_available/design.csv',
 'h3_h4/H4/common_controls/new_available/samples.csv','h3_h4/H4/common_controls/new_available/draws.npz',
 'h3_h4/H4/common_controls/new_available/draw_status.csv','h3_h4/H4/common_controls/new_available/audit.json',
 'h3_h4/H4/dv24_complete_case/new_available/effects.csv','h3_h4/E5_exact_DE_allocation_all_scopes.csv',
 'h3_h4/run_audit.json']
 out=[]
 for rel in paths:
  p=B/rel;expected=catalog.loc[rel,'sha256'] if rel in catalog.index else None
  register(p,'accepted result reuse source '+rel,expected)
  obj=dict(relative_path=rel,path=str(p),sha256=sha(p),role='H3 duration asymmetry association' if '/H3' in rel else 'H4 stage duration and D−E' if '/H4/' in rel else 'D/E allocation and lineage',
   scientific_status='readonly source available; unchanged accepted result',refit=False)
  if 'dv24_complete_case' in rel:obj['scientific_status']='NOT_IDENTIFIABLE_NO_DV24_D_SUPPORT retained; cannot substitute as identified H4 result'
  if p.suffix=='.csv':
   d=readcsv(p);obj['rows']=len(d);obj['columns']='|'.join(d.columns)
  out.append(obj)
 write_csv('accepted_H3_H4_reuse_index.csv',pd.DataFrame(out))
 return out

def synthetic_tests():
 t=pd.Timestamp('2000-01-01',tz='UTC')
 check('synthetic_interval_partial_overlap',overlap(t,t+pd.Timedelta(hours=6),t+pd.Timedelta(hours=3),t+pd.Timedelta(hours=9))==3)
 check('synthetic_touching_boundary_no_extra',overlap(t,t+pd.Timedelta(hours=6),t+pd.Timedelta(hours=6),t+pd.Timedelta(hours=12))==0)
 check('synthetic_empty_middle_zero',overlap(t,t+pd.Timedelta(hours=6),t,t)==0)
 check('synthetic_absmean_not_meanabs',abs(np.mean([.5,-.5]))!=np.mean(np.abs([.5,-.5])))
 check('synthetic_bool_false_not_true',boolcol(pd.Series(['False','True','0','1'])).tolist()==[False,True,False,True])
 check('synthetic_halfopen_intensity_bins',list(np.floor((np.array([35,44.9,45,55])-35)/10).astype(int))==[0,0,1,2])

def main():
 start=now();spec=freeze();synthetic_tests()
 s,t,catalog=load()
 request=p4(s,t);timeaudit,intersection,ledger=geometries(s,t)
 pairs,candidates=match_intensity(s,ledger);reuse=reuse_index(catalog)
 summary={'status':'SOURCE_AND_TIME_GEOMETRY_COMPLETE_NO_NEW_ENVIRONMENT_EFFECT_FIT','start_utc':start,'end_utc':now(),
 'python':sys.executable,'source_sid_n':len(s),'source_observation_n':len(t),
 'P4_requests':len(request),'P4_actual_track_points':int(request.actual_track_point.sum()),
 'P4_missing_track_requests':int((~request.actual_track_point).sum()),
 'P4_unique_actual_sid_utc':request[request.actual_track_point][['sid','UTC']].drop_duplicates().shape[0],
 'P5_time_intersections':len(intersection),'P5_matched_pairs':len(pairs),'P5_matched_sid_n':pairs.sid.nunique(),
 'P5_no_sharedbin_sid_n':len(s)-pairs.sid.nunique(),'P5_source6factor_joint_pairs':int(pairs.six_factor_joint_paired_valid.sum()),
 'Tpre_not_equal_DplusE_sid_n':int(timeaudit.Tpre_minus_DplusE_h.ne(0).sum()),
 'Tpost_not_equal_Res_sid_n':int(timeaudit.Tpost_minus_Res_h.ne(0).sum()),
 'max_abs_A_numeric_error':float((timeaudit.A_signed_from_frozen_times-timeaudit.source_A_signed).abs().max()),
 'matched_wind_diff_abs_max_kt':float(pairs.wind_post_minus_pre_kt.abs().max()),
 'tests_n':len(CHECKS),'source_spec_sha256':sha(spec),'code_sha256':sha(__file__),
 'execution_status':'COMPLETE','verification_status':'PASS source/time/selection tests','scientific_conclusion_status':'NOT_ESTIMATED; no new environmental effect',
 'source_read_scope':'v19 small CSV and v3.2 approved cached parquet/tables only; no raw ERA5 NC',
 'limitations':['P4 outside frozen lifecycle lacks normalized-track position; no extrapolation or complete-window conditioning.',
 'Intensity matching is coarse10kt-bin matching; actual wind differences saved, not claimed exactly equal.',
 'Matched set and each-factor environment-supported set are separate; subsequent effects must report its own SID and bin support.',
 'P5 elapsed partitions and algebra reproduce definitions; this is verification, not a new physical result.']}
 write_json('source_identity.json',{'inputs':INPUTS,'source_spec_sha256':sha(spec),'code_sha256':sha(__file__)})
 write_json('verification.json',{'status':'PASS','tests':CHECKS,'tests_n':len(CHECKS)})
 write_json('execution_status.json',summary)
 lines=['# P4/P5 冻结来源和时间对应核对','',f"实际完成：{summary['end_utc']}；执行与时间/来源核验完成，尚未估计新环境效应。",'',
 f"- 冻结2368 SID、64210时次；P4保留全部{len(request)}个边界/lag请求，其中{summary['P4_actual_track_points']}个真实位置，{summary['P4_missing_track_requests']}个缺轨迹请求保留缺项原因。",
 f"- P5逐SID核对Tpre、Tpost及有符号/绝对A；最大A浮点误差{summary['max_abs_A_numeric_error']:.3g}。",
 f"- 阶段×pre/middle/post共{len(intersection)}行，elapsed与合法6h覆盖分别；终止点后不增加6h。",
 f"- D+E不等于Tpre的SID有{summary['Tpre_not_equal_DplusE_sid_n']}个；Res不等于Tpost的SID有{summary['Tpost_not_equal_Res_sid_n']}个。",
 f"- 冻结10kt箱、源风速邻近中心/固定时间平局规则后，{pairs.sid.nunique()}SID有{len(pairs)}对共同箱；每SID总权重1，实际风速差另存。", '',
 'H3/H4正式复用路径及指纹见accepted_H3_H4_reuse_index.csv。原H4 dv24分支未识别状态原样记录，不改名当作通过。', '',
 '所有环境数值为SP后权威缓存逐行复制，没有读取原始nc、拟合环境模型或bootstrap。后续分析仍需独立完成其自己的支持、统计与科学核验。']
 (O/'source_time_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
 try:main()
 except BaseException:
  O.mkdir(exist_ok=True)
  write_json('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json',{'UTC':now(),'traceback':traceback.format_exc(),'checks':CHECKS})
  raise

