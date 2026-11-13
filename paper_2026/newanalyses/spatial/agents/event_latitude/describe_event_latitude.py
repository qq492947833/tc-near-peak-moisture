# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
"""Bounded frozen-event latitude descriptor; reuse audited denominators, no fit."""
from pathlib import Path
import runpy,json,hashlib,shutil,datetime
O=Path(__file__).resolve().parent
bootstrap(O/'temp')
import numpy as np
import pandas as pd
R1=PACKAGE/'history'
U=ADDENDUM
paths={'frozen_events':U/'unified_main_20261023_1642/source_data/all_return_events.csv','frozen_tracks':U/'source_data/new_lifecycle_tracks.csv','event_reconciliation':R1/'agents/A/event_reconciliation.csv','departure_episodes':R1/'agents/A/departure_episodes.csv','sid_denominators':R1/'agents/A/sid_denominators.csv','risk_intervals':R1/'agents/A/risk_intervals_6h.csv','basin_estimates':R1/'agents/A/basin_estimates_ci.csv','annual_counts':R1/'agents/B/annual_basin_numerators_denominators.csv','annual_trends':R1/'agents/B/annual_trends_and_basin_interactions.csv','A_source_manifest':R1/'agents/A/SOURCE_MANIFEST.json','design':O/'DESIGN_BEFORE_DESCRIPTORS.json'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p,**kw):return pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip',**kw)
def save(df,name):df.to_csv(O/name,index=False,float_format='%.17g')
def dump(name,data):(O/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
checks=[]
def check(name,value):
 checks.append({'check':name,'pass':bool(value)})
 assert value,name
design=json.loads(paths['design'].read_text(encoding='utf8'));edges=np.array(design['latitude_bins_degrees'],float)
manifest={k:{'path':str(p),'resolved_path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size} for k,p in paths.items()}
dump('SOURCE_MANIFEST.json',manifest)
old=json.loads(paths['A_source_manifest'].read_text());check('frozen event hash matches audited A source',manifest['frozen_events']['sha256']==old['events']['sha256']);check('frozen track hash matches audited A source',manifest['frozen_tracks']['sha256']==old['track']['sha256'])
ev=read(paths['frozen_events']);rec=read(paths['event_reconciliation']);dep=read(paths['departure_episodes']);sid=read(paths['sid_denominators'])
check('133 frozen unique event IDs / 123 SIDs',len(ev)==133 and ev.event_id.is_unique and ev.sid.nunique()==123)
check('2340 unique eligible SIDs / 123 binary returns / 133 events',len(sid)==2340 and sid.sid.is_unique and sid.ever_return.sum()==123 and sid.n_return.sum()==133)
check('NA preserved as literal basin',('NA' in set(ev.basin_genesis)) and ('NA' in set(ev.return_basin)))
for frame,col in [(ev,'temporary_departure_UTC'),(ev,'return_UTC'),(rec,'return_UTC'),(dep,'t0'),(dep,'end_UTC')]:frame[col]=pd.to_datetime(frame[col],utc=True)
join=ev.merge(rec,on=['sid','event_id','return_UTC','return_basin'],validate='1:1')
check('all frozen events match audited event reconciliation',len(join)==133)
join=join.merge(dep[['sid','departure_order','t0','end_UTC','end_reason','lat','lon','position_basin','observable_hours']],on=['sid','departure_order'],validate='1:1')
check('canonical departure UTC equals matched risk-episode t0',(join.temporary_departure_UTC==join.t0).all())
check('canonical return UTC equals risk-episode end',(join.return_UTC==join.end_UTC).all() and join.end_reason.eq('qualifying_return').all())
check('canonical departure latitude equals risk table',np.allclose(join.temporary_departure_lat,join.lat,atol=1e-12,rtol=0))
check('canonical departure basin equals risk table',(join.temporary_departure_basin==join.position_basin).all())
needed=set(zip(ev.sid,ev.temporary_departure_UTC))|set(zip(ev.sid,ev.return_UTC));wanted_sids=set(ev.sid)
kept=[];stream_rows=0
cols=['sid','time','lat','lon','wind','basin','stage_code','is_tropical']
for chunk in read(paths['frozen_tracks'],usecols=cols,chunksize=20000):
 stream_rows+=len(chunk);chunk=chunk[chunk.sid.isin(wanted_sids)].copy();chunk['time']=pd.to_datetime(chunk.time,utc=True)
 keep=[(a,b) in needed for a,b in zip(chunk.sid,chunk.time)]
 kept.append(chunk.loc[keep])
tr=pd.concat(kept,ignore_index=True)
check('only exact required event anchors retained',len(tr)==len(needed) and not tr.duplicated(['sid','time']).any() and set(zip(tr.sid,tr.time))==needed)
save(tr.sort_values(['sid','time']),'matched_track_anchors.csv')
ledger=ev.merge(sid[['sid','genesis_year','basin_genesis']],on=['sid','basin_genesis'],validate='m:1')
ledger=ledger.merge(rec[['event_id','departure_order']],on='event_id',validate='1:1')
for anchor,utc,prefix in [('departure','temporary_departure_UTC','temporary_departure'),('return','return_UTC','return')]:
 t=tr.rename(columns={'time':utc,**{c:f'{anchor}_track_{c}' for c in ['lat','lon','wind','basin','stage_code','is_tropical']}})
 ledger=ledger.merge(t,on=['sid',utc],validate='m:1')
 for field in ['lat','lon','wind']:
  check(anchor+' canonical '+field+' matches exact track',np.allclose(ledger[f'{prefix}_{field}'],ledger[f'{anchor}_track_{field}'],atol=1e-10,rtol=0))
 check(anchor+' basin matches exact track',(ledger[f'{prefix}_basin']==ledger[f'{anchor}_track_basin']).all())
 check(anchor+' tropical flag matches exact track',(ledger[f'{prefix}_is_tropical']==ledger[f'{anchor}_track_is_tropical']).all())
 check(anchor+' all latitude positions finite and within Earth limits',np.isfinite(ledger[f'{anchor}_track_lat']).all() and ledger[f'{anchor}_track_lat'].between(-90,90).all())
 ledger[anchor+'_hemisphere']=np.select([ledger[f'{anchor}_track_lat']<0,ledger[f'{anchor}_track_lat']>0],['SH','NH'],default='EQ')
 labels=[f'[{int(a)},{int(b)}'+(']' if b==90 else ')') for a,b in zip(edges[:-1],edges[1:])]
 idx=np.minimum(np.searchsorted(edges,ledger[f'{anchor}_track_lat'].to_numpy(),side='right')-1,len(edges)-2)
 ledger[anchor+'_signed_10deg_bin']=[labels[i] for i in idx]
ledger['signed_delta_latitude_deg']=ledger.return_track_lat-ledger.departure_track_lat
ledger['poleward_magnitude_change_deg']=ledger.return_track_lat.abs()-ledger.departure_track_lat.abs()
ledger['elapsed_departure_to_return_h']=(ledger.return_UTC-ledger.temporary_departure_UTC).dt.total_seconds()/3600
ledger['same_hemisphere']=ledger.departure_hemisphere==ledger.return_hemisphere
check('bridge duration equals exact anchor difference',np.allclose(ledger.elapsed_departure_to_return_h,ledger.bridge_duration_h,atol=1e-12,rtol=0))
check('all actual returns strictly after departure',(ledger.elapsed_departure_to_return_h>0).all())
ledger=ledger.sort_values(['sid','return_order']).reset_index(drop=True)
save(ledger,'event_latitude_ledger.csv')
dump('CHECKPOINT_event_ledger.json',{'status':'COMPLETE_VALIDATED','events':len(ledger),'unique_sid':ledger.sid.nunique(),'retained_track_anchor_rows':len(tr),'streamed_track_rows':stream_rows,'all_rows_eligible':bool(ledger.ledger_eligible.all()),'canonical_departures_match_A_risk_episodes':True,'no_interpolation':True,'no_latitude_truncation':True})

# Fixed-bin summaries are event distributions. Distinct-SID counts are not additive across bins.
basins=['NA','EP','WP','NI','SI','SP','SA'];sums=[];bybasin=[];hemis=[];basintotal=[]
for anchor in ['departure','return']:
 for label in labels:
  g=ledger[ledger[anchor+'_signed_10deg_bin'].eq(label)]
  sums.append({'anchor':anchor,'signed_latitude_bin':label,'n_events':len(g),'n_unique_sid':g.sid.nunique(),'event_percent_of_133':100*len(g)/133})
 for basis,col in [('genesis','basin_genesis'),('departure_location','temporary_departure_basin'),('return_location','return_basin')]:
  for basin in basins:
   g=ledger[ledger[col].eq(basin)]
   for label in labels:
    z=g[g[anchor+'_signed_10deg_bin'].eq(label)];bybasin.append({'anchor':anchor,'basin_basis':basis,'basin':basin,'signed_latitude_bin':label,'n_events':len(z),'n_unique_sid':z.sid.nunique()})
   for hemi in ['SH','EQ','NH']:
    z=g[g[anchor+'_hemisphere'].eq(hemi)];hemis.append({'anchor':anchor,'basin_basis':basis,'basin':basin,'hemisphere':hemi,'n_events':len(z),'n_unique_sid':z.sid.nunique()})
for basis,col in [('genesis','basin_genesis'),('departure_location','temporary_departure_basin'),('return_location','return_basin')]:
 for basin in basins:
  z=ledger[ledger[col].eq(basin)];basintotal.append({'basin_basis':basis,'basin':basin,'n_events':len(z),'n_unique_sid':z.sid.nunique()})
save(pd.DataFrame(sums),'support_by_signed_bin.csv');save(pd.DataFrame(bybasin),'support_by_signed_bin_basin.csv');save(pd.DataFrame(hemis),'support_by_hemisphere_basin.csv');save(pd.DataFrame(basintotal),'event_support_by_basin.csv')
matrix=[]
for a in labels:
 for b in labels:
  z=ledger[ledger.departure_signed_10deg_bin.eq(a)&ledger.return_signed_10deg_bin.eq(b)];matrix.append({'departure_bin':a,'return_bin':b,'n_events':len(z),'n_unique_sid':z.sid.nunique()})
save(pd.DataFrame(matrix),'departure_return_bin_matrix.csv')
summary=[]
for hemi in ['ALL','NH','SH','EQ']:
 z=ledger if hemi=='ALL' else ledger[ledger.return_hemisphere.eq(hemi)]
 for field in ['departure_track_lat','return_track_lat','signed_delta_latitude_deg','poleward_magnitude_change_deg','elapsed_departure_to_return_h']:
  v=z[field];summary.append({'return_hemisphere':hemi,'quantity':field,'n_events':len(z),'n_unique_sid':z.sid.nunique(),'mean':v.mean(),'min':v.min(),'p25':v.quantile(.25),'median':v.median(),'p75':v.quantile(.75),'max':v.max()})
save(pd.DataFrame(summary),'descriptive_event_summary.csv')

# Reuse and audit the previously computed true opportunity and annual denominators; no model.
risk=read(paths['risk_intervals']);risk_start=pd.to_datetime(risk.interval_start,utc=True);risk_end=pd.to_datetime(risk.interval_end,utc=True)
check('true risk intervals all six observed hours',risk.hours.eq(6).all() and ((risk_end-risk_start).dt.total_seconds()/3600).eq(6).all())
check('true opportunity denominator 25824 intervals /154944h/2338SID',len(risk)==25824 and risk.hours.sum()==154944 and risk.sid.nunique()==2338)
check('true opportunity events reconcile 133/123',risk.return_event.sum()==133 and risk.loc[risk.return_event.eq(1),'sid'].nunique()==123)
per_sid=risk.groupby('sid').agg(observable_hours=('hours','sum'),n_intervals=('hours','size'),n_risk_return_events=('return_event','sum')).reset_index()
per_sid=sid[['sid','genesis_year','basin_genesis','ever_return','n_return']].merge(per_sid,on='sid',how='left',validate='1:1');per_sid[['observable_hours','n_intervals','n_risk_return_events']]=per_sid[['observable_hours','n_intervals','n_risk_return_events']].fillna(0)
check('2215 observed nonreturners retained in risk',((per_sid.ever_return==0)&(per_sid.observable_hours>0)).sum()==2215)
save(per_sid,'reused_opportunity_sid_denominators.csv')
shutil.copyfile(paths['basin_estimates'],O/'reused_A_basin_estimates_ci.csv')
annual=read(paths['annual_counts']);global_counts=annual[annual.basin.eq('GLOBAL')].copy().sort_values('year')
derived=sid.groupby('genesis_year').agg(n_sid=('sid','size'),n_return_sid=('ever_return','sum'),n_return_events=('n_return','sum')).reset_index().rename(columns={'genesis_year':'year'})
check('annual 42 genesis years complete',global_counts.year.tolist()==list(range(1982,2024)))
for field in ['n_sid','n_return_sid','n_return_events']:check('reused annual '+field+' matches uniqueSID genesis-year audit',np.array_equal(global_counts[field].to_numpy(),derived.sort_values('year')[field].to_numpy()))
check('annual totals 2340/123/133',global_counts.n_sid.sum()==2340 and global_counts.n_return_sid.sum()==123 and global_counts.n_return_events.sum()==133)
save(global_counts,'reused_global_annual_counts.csv')
slopes=read(paths['annual_trends']);global_slope=slopes[slopes.kind.eq('global slope')].copy();check('exact previous global model estimate and interval',len(global_slope)==1 and global_slope.iloc[0].estimate==.5284431632845179 and global_slope.iloc[0].lo==-.49922822434316405 and global_slope.iloc[0].hi==1.5561145509121999)
save(global_slope,'reused_global_annual_trend.csv')
check('all bins retain all133events each anchor',all(pd.DataFrame(sums).groupby('anchor').n_events.sum().eq(133)))
check('joint departure-return bins retain all133events',pd.DataFrame(matrix).n_events.sum()==133)
check('return location counts reproduce frozen figure1',all(pd.DataFrame(basintotal).query('basin_basis == "return_location"').set_index('basin').n_events.reindex(basins).to_numpy()==read(paths['basin_estimates']).set_index('basin').event_count.reindex(basins).to_numpy()))
roundtrip=read(O/'event_latitude_ledger.csv');check('ledger full precision readback',np.array_equal(roundtrip.return_track_lat,ledger.return_track_lat) and np.array_equal(roundtrip.departure_track_lat,ledger.departure_track_lat) and roundtrip.event_id.tolist()==ledger.event_id.tolist())
for k,entry in manifest.items():check('source unchanged '+k,sha(entry['path'])==entry['sha256'])
dump('VALIDATION_RECEIPT.json',{'status':'PASS','checks_passed':len(checks),'checks':checks,'models_fitted':0,'bootstrap_draws_generated':0,'old_sources_unchanged':True,'output_only':str(O),'known_48h_source_distinct':'historical_first_events_with_retained_outcomes.csv uses past-only threshold; never used here'})
result={'status':'DESCRIPTORS_AND_DENOMINATOR_REUSE_COMPLETE','events':len(ledger),'returning_sid':ledger.sid.nunique(),'return_hemisphere_counts':ledger.return_hemisphere.value_counts().to_dict(),'departure_latitude_range':[float(ledger.departure_track_lat.min()),float(ledger.departure_track_lat.max())],'return_latitude_range':[float(ledger.return_track_lat.min()),float(ledger.return_track_lat.max())],'abs_return_lat_over40_events':int(ledger.return_track_lat.abs().gt(40).sum()),'abs_return_lat_over40_sid':int(ledger.loc[ledger.return_track_lat.abs().gt(40),'sid'].nunique()),'cross_equator_events':int((~ledger.same_hemisphere).sum()),'poleward_events':int(ledger.poleward_magnitude_change_deg.gt(0).sum()),'equatorward_events':int(ledger.poleward_magnitude_change_deg.lt(0).sum()),'zero_delta_abs_events':int(ledger.poleward_magnitude_change_deg.eq(0).sum()),'risk_hours_reused':154944,'annual_global_estimate_reused':float(global_slope.iloc[0].estimate),'annual_global_ci_reused':[float(global_slope.iloc[0].lo),float(global_slope.iloc[0].hi)],'checks_passed':len(checks),'models_fitted':0}
dump('CHECKPOINT.json',result);dump('REQUIREMENT_STATUS.json',result)
print(json.dumps(result,indent=2))
