# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
"""Audit and reaggregate saved C results only. No model fit or new experiment."""
from pathlib import Path
import runpy,json,hashlib,shutil,os,sys,math
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
bootstrap(OUT/'temp')
import numpy as np,pandas as pd
R1=PACKAGE/'history';C=R1/'agents/C'
STAGES=['D','E','F','Res'];TARGETS=['actual_composition','fixed_basin_season','fixed_basin_season_state','fixed_basin_season_state_environment'];ENV=['X1_sst_c','X2_shear_ms','X3_mid_rh_pct']
SOURCES={};CHECKS=[];CACHE={};T=(np.arange(1982,2024)-2002.5)/10
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def source(p,role=None):
 p=Path(p);s=str(p)
 if s not in SOURCES:SOURCES[s]=dict(path=s,resolved=str(p.resolve()),size_bytes=p.stat().st_size,sha256=sha(p),role=role or p.name,access='read-only')
 return p
def read(rel):
 p=source(R1/rel)
 if str(p)not in CACHE:CACHE[str(p)]=pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip',low_memory=False)
 return CACHE[str(p)].copy()
def ctab(name):return read('agents/C/'+name)
def check(name,passed,detail=None):
 CHECKS.append(dict(check=name,passed=bool(passed),detail=detail));assert passed,(name,detail)
def save(d,name):d.to_csv(OUT/name,index=False,float_format='%.17g')
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v)for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float)and not math.isfinite(x):return None
 return x
def dump(name,obj):(OUT/name).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def checkpoint(unit):dump('CHECKPOINT.json',dict(status='IN_PROGRESS',completed_unit=unit,utc=datetime.now(timezone.utc).isoformat(),models_refitted=False,new_experiments=False))
def ci(v):
 v=np.asarray(v);v=v[np.isfinite(v)];return np.percentile(v,[2.5,97.5])
def signp(v):
 v=np.asarray(v);v=v[np.isfinite(v)];return min(1.,2*min((1+(v<=0).sum())/(len(v)+1),(1+(v>=0).sum())/(len(v)+1)))
def bh(p):
 p=np.asarray(p);o=np.argsort(p);q=np.empty(len(p));q[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)
def audit_interval(row,v,pcol='p_boot'):
 z=ci(v);check('saved_interval_'+str(row.get('stage',''))+'_'+str(row.get('target',row.get('variable',row.get('removed_distribution','')))),np.max(np.abs(z-np.array([row.ci_low,row.ci_high])))<1e-11)
 if pcol in row:check('saved_sign_tail',abs(signp(v)-row[pcol])<1e-12)
 check('saved_success_count',int(np.isfinite(v).sum())==int(row.valid_draws))
def slope(y):return float(T@np.asarray(y)/(T@T))

csv_names=['S1c_exact_reproduction.csv','standardized_duration_trends.csv','standardization_trend_contributions.csv','entry_environment_trends.csv','entry_environment_duration_associations.csv','entry_state_trends_associations.csv','individual_environment_trend_products.csv','positive_duration_model_sensitivity.csv','F_components_original_N1_replayed.csv','F_components_trends_HAC.csv','F_components_sid.csv','F_components_annual.csv','Dec_observation_trends_HAC.csv','Dec_observation_annual.csv','duration_sensitivities.csv','sensitivity_intervals.csv','support_sid_stage.csv','common_support_basin_period.csv','common_support_cutoffs.csv','entry_coverage_period.csv','track_coverage_period.csv','stage_annual_full_cohort.csv','standardized_annual_predictions.csv','bootstrap_draw_status.csv','positive_duration_draw_status.csv','stage_duration_coverage_sid.csv']
(OUT/'source_tables').mkdir(exist_ok=True)
for n in csv_names:
 p=source(C/n);shutil.copyfile(p,OUT/'source_tables'/n);check('byte_identical_copy_'+n,sha(p)==sha(OUT/'source_tables'/n))
# Input identities are recorded by this release's SOURCE_MANIFEST.json.
primary=read('manuscript/supplement_S1c_source.csv');save(primary,'primary_full_cohort_full_precision.csv')
spec=json.loads((C/'MODEL_SPEC_AND_AUDIT.json').read_text('utf8'));support=ctab('support_sid_stage.csv');entries=ctab('stage_entry_source_frozen.csv');coverage=ctab('stage_duration_coverage_sid.csv');trends=ctab('standardized_duration_trends.csv');contrib=ctab('standardization_trend_contributions.csv');envhist=ctab('entry_environment_trends.csv');assoc=ctab('entry_environment_duration_associations.csv');ann=ctab('standardized_annual_predictions.csv');statuses=ctab('bootstrap_draw_status.csv');gamma=ctab('positive_duration_model_sensitivity.csv');gstatus=ctab('positive_duration_draw_status.csv')
check('one_row_per_sid_stage',not entries.duplicated(['sid','stage']).any()and len(entries)==9360)
six=ENV+['X4_d200_1e5s','X5_minus_omega500_pa_s','X6_signed_vo850_1e5s']
check('saved_exposure_qc_requires_at_least3_and_six_finite',bool(((entries.n_valid>=3)&entries[six].notna().all(axis=1))[entries.exposure_qc].all()))
for d in [trends,contrib,envhist,assoc]:check('BH_family_'+str(d.family.iloc[0]),np.max(abs(bh(d.p_boot)-d.q_BH_new_family))<1e-12)
checkpoint('source_hashes_and_original_precision_tables')

bank=np.load(source(C/'joint_year_sid_draws.npz'));counts=bank['counts'];ym=bank['year_multiplicity'];sid=bank['sid'];sidix={s:i for i,s in enumerate(sid)};sidsha=hashlib.sha256('\n'.join(sid).encode()).hexdigest();sig=[]
for b in range(len(counts)):sig.append(hashlib.sha256(counts[b].tobytes()+ym[b].tobytes()+sidsha.encode()).hexdigest())
save(pd.DataFrame(dict(draw=np.arange(len(sig)),joint_bank_draw_sha256=sig,sid_order_sha256=sidsha)),'joint_draw_identity.csv')
summaries=[];pairrows=[];drawrows=[];decrows=[];banks={}
for st in STAGES:
 g=ctab('common_support_'+st+'.csv');d=ctab('design_'+st+'.csv');npz=np.load(source(C/f'model_bootstrap_{st}.npz'));banks[st]=npz;arr=npz['trends'];names=npz['names'].tolist();yc=g.genesis_year.to_numpy(int)-1982;ix=np.array([sidix[s]for s in g.sid]);nper=np.bincount(yc,minlength=42);weights=d.weight.to_numpy();ss=support[support.stage.eq(st)];status=statuses[statuses.stage.eq(st)].set_index('draw').reindex(range(2000));finite=np.all(np.isfinite(arr[:,:,0]),axis=1)
 check('targets_order_'+st,npz['targets'].tolist()==TARGETS)
 check('SID_order_design_support_'+st,g.sid.tolist()==d.sid.tolist()and set(g.sid)==set(ss.loc[ss.common_support,'sid']))
 check('equal_year_base_weights_'+st,np.max(abs(weights-1/nper[yc]))<1e-14 and np.all(nper>0))
 check('successful_status_matches_finite_bank_'+st,np.array_equal(finite,status.status.eq('ok').to_numpy()))
 cc=counts[:,ix].astype(float);wsum=np.column_stack([cc[:,yc==y].sum(axis=1)for y in range(42)]);unexpected=(ym>0)&(wsum==0);check('sampled_years_have_stage_support_'+st,not unexpected.any())
 check('all_targets_same_success_mask_'+st,all(np.array_equal(np.isfinite(arr[:,j,0]),finite)for j in range(4)))
 for j,target in enumerate(TARGETS):
  row=trends[(trends.stage==st)&(trends.target==target)].iloc[0];audit_interval(row,arr[:,j,0]);a=ann[(ann.stage==st)&(ann.target==target)].sort_values('genesis_year');check('saved_hour_predictions_slope_'+st+'_'+target,abs(slope(a.mean_h)-row.slope)<1e-11)
 for j,group in enumerate(['basin_season','entry_state','entry_environment']):audit_interval(contrib[(contrib.stage==st)&(contrib.removed_distribution==group)].iloc[0],arr[:,j,0]-arr[:,j+1,0])
 for ei,v in enumerate(ENV):
  audit_interval(envhist[(envhist.stage==st)&(envhist.variable==v)].iloc[0],npz['environment_trends'][:,ei]);k=names.index(v);audit_interval(assoc[(assoc.stage==st)&(assoc.variable==v)].iloc[0],npz['coefficients'][:,k,0]/g[v].std(ddof=0))
 grow=gamma[gamma.stage.eq(st)].iloc[0];gd=np.load(source(C/f'gamma_duration_draws_{st}.npy'));audit_interval(grow,gd);gs=gstatus[gstatus.stage.eq(st)].set_index('draw').reindex(range(2000));check('Gamma_status_exact_'+st,np.array_equal(np.isfinite(gd),gs.status.eq('ok').to_numpy()))
 summaries.append(dict(stage='Dec'if st=='Res'else st,source_stage=st,full_cohort_n=2340,complete_case_n=int(ss.complete.sum()),common_support_n=len(g),n_years=42,year_min_n=int(nper.min()),year_max_n=int(nper.max()),weight_sum=float(weights.sum()),attempted_draws=2000,successful_draws=int(finite.sum()),failed_draws=int((~finite).sum()),Gamma_successful_draws=int(np.isfinite(gd).sum()),Gamma_failed_draws=int((~np.isfinite(gd)).sum()),reference='pooled same-support empirical distribution with each year total weight one',six_field_entry_qc_retained=True))
 for a,b,label in [(1,3,'fixed_basin_season_minus_fully_fixed'),(2,3,'fixed_basin_season_state_minus_fully_fixed_environment_only')]:
  aa=trends[(trends.stage==st)&(trends.target==TARGETS[a])].iloc[0];bb=trends[(trends.stage==st)&(trends.target==TARGETS[b])].iloc[0];delta=arr[:,a,0]-arr[:,b,0];lo,hi=ci(delta);old=contrib[(contrib.stage==st)&(contrib.removed_distribution=='entry_environment')].iloc[0]if a==2 else None
  pairrows.append(dict(stage='Dec'if st=='Res'else st,source_stage=st,comparison=label,n_sid=len(g),unit='h/decade',basic_target=TARGETS[a],fully_fixed_target=TARGETS[b],basic_slope=aa.slope,basic_ci_low=aa.ci_low,basic_ci_high=aa.ci_high,fully_fixed_slope=bb.slope,fully_fixed_ci_low=bb.ci_low,fully_fixed_ci_high=bb.ci_high,paired_difference=aa.slope-bb.slope,paired_ci_low=lo,paired_ci_high=hi,attempted=2000,valid_draws=int(finite.sum()),p_boot_existing=np.nan if old is None else old.p_boot,q_BH_existing=np.nan if old is None else old.q_BH_new_family,family='descriptive saved-draw aggregation; no new p/q'if old is None else old.family,derivation='paired within original draw identity; no refit'))
 for b in range(2000):
  drawrows.append(dict(stage='Dec'if st=='Res'else st,draw=b,joint_bank_draw_sha256=sig[b],successful=bool(finite[b]),source_status=status.status.iloc[b],same_SID_count_vector=True,same_year_weight_vector=True,basic_composition_slope=arr[b,1,0],state_fixed_slope=arr[b,2,0],fully_fixed_slope=arr[b,3,0],basic_minus_fully_fixed=arr[b,1,0]-arr[b,3,0],state_fixed_minus_fully_fixed=arr[b,2,0]-arr[b,3,0]))
 if st=='Res':
  ds=ctab('duration_sensitivities.csv');ds=ds[(ds.stage==st)&(ds.scope=='effective_6h_wind_coverage_same_support')]
  for j,target in enumerate(TARGETS):
   er=trends[(trends.stage==st)&(trends.target==target)].iloc[0];cr=ds[ds.target==target].iloc[0];audit_interval(cr,arr[:,j,1]);lo,hi=ci(arr[:,j,0]-arr[:,j,1]);decrows.append(dict(stage='Dec',target=target,n_sid=len(g),elapsed_slope=er.slope,covered_slope=cr.slope,elapsed_minus_covered=er.slope-cr.slope,paired_ci_low=lo,paired_ci_high=hi,valid_draws=int(finite.sum()),unit='h/decade',family='descriptive saved-draw reaggregation; no new p/q'))
save(pd.DataFrame(summaries),'stage_support_and_attempts.csv');save(pd.DataFrame(pairrows),'same_support_paired_trend_comparisons.csv');save(pd.DataFrame(drawrows),'paired_draw_identity_audit.csv');save(pd.DataFrame(decrows),'Dec_common_support_elapsed_minus_covered.csv');checkpoint('saved_draw_identity_intervals_and_pairing')

fs=ctab('F_components_sid.csv');fa=ctab('F_components_annual.csv');fc=ctab('F_components_original_N1_replayed.csv');ff=read('agents/methods/F_forward_completeness.csv');first=read('agents/A/first_departure_reference.csv');ba=read('agents/methods/frozen_sid_boundary_audit.csv');missing=read('agents/methods/missing_wind_records.csv')
check('F_2340_unique_sid',len(fs)==fs.sid.nunique()==2340)
check('F_identity_exact_every_SID',np.array_equal(fs.T_R_h+fs.T_L_h+fs.T_B_h,fs.T_F_h))
check('F_valid_coverage_exact_elapsed',fs.F_uncovered_h.eq(0).all()and np.array_equal(fs.T_F_h,fs.maturity_duration_hours))
check('2217_zero_L_and_B',int((fs.T_L_h.eq(0)&fs.T_B_h.eq(0)).sum())==2217 and int(fs.T_L_h.gt(0).sum())==123)
ep=fs.merge(ff,on='sid',validate='1:1').merge(first[['sid','first_departure_UTC','ever_return']],on='sid',validate='1:1')
for k in ['F_onset','F_exit','last_valid_nearpeak_time','first_departure_UTC']:ep[k]=pd.to_datetime(ep[k],utc=True)
ep['R_from_endpoints_h']=(ep.first_departure_UTC-ep.F_onset).dt.total_seconds()/3600;ep['F_from_endpoints_h']=(ep.F_exit-ep.F_onset).dt.total_seconds()/3600
check('R_duration_endpoint_exact',np.array_equal(ep.R_from_endpoints_h,ep.T_R_h))
check('F_duration_endpoint_exact',np.array_equal(ep.F_from_endpoints_h,ep.T_F_h))
check('F_exit_next6h_after_last_nearpeak',(ep.F_exit-ep.last_valid_nearpeak_time).eq(pd.Timedelta(hours=6)).all())
check('B_actual_observed_records_times6',np.array_equal(ep.observed_subthreshold_bridge_records*6,ep.T_B_h))
check('F_original_boundary_audit_unchanged',ba.frozen_F_matches.all()and ba.strict6h_F_matches.all()and ba.stage_code_matches.all())
check('F_no_unknown_wind_or_time_gap',ep.missing_wind_through_F_exit.eq(0).all()and ep.non6h_gaps_through_F_exit.eq(0).all())
for v in ['T_R_h','T_L_h','T_B_h','T_F_h']:
 means=fs.groupby('genesis_year')[v].mean().reindex(range(1982,2024)).to_numpy();check('F_equal_year_annual_'+v,np.max(abs(means-fa[v]))<1e-12);row=fc[fc.component.eq(v[2])].iloc[0];check('F_saved_OLS_slope_'+v,abs(slope(means)-row.estimate)<1e-12)
save(ep[['sid','genesis_year','basin_genesis','F_onset','first_departure_UTC','last_valid_nearpeak_time','F_exit','T_R_h','T_L_h','T_B_h','T_F_h','R_from_endpoints_h','F_from_endpoints_h','F_uncovered_h','observed_subthreshold_bridge_records','ever_return']],'F_sid_identity_endpoints_audit.csv')
dec=coverage[coverage.stage.eq('Res')];check('Dec_394_missing_tail_sid',int(dec.uncovered_h.gt(0).sum())==394);check('Dec_963_missing_records',len(missing)==963 and missing.sid.nunique()==394 and not missing.after_valid.any());mg=missing.groupby('sid').size().reindex(dec.sid,fill_value=0);check('Dec_tail_missing_records_times6',np.array_equal(mg.to_numpy()*6,dec.uncovered_h.to_numpy()));check('Dec_exclusion_n1946',int(dec.uncovered_h.eq(0).sum())==1946)
ob=ctab('Dec_observation_trends_HAC.csv').set_index('metric');check('Dec_trend_elapsed_equals_covered_plus_missing',abs(ob.loc['elapsed_h','slope']-ob.loc['covered_h','slope']-ob.loc['tail_missing_wind_h','slope'])<1e-12)
check('D_E_F_no_missing_coverage',coverage.loc[coverage.stage!='Res','uncovered_h'].eq(0).all())
checkpoint('F_endpoint_and_terminal_missing_checks')

payload={'units':'hours per decade unless a source table explicitly gives environmental units or fraction/decade','display_stage_mapping':{'Res':'Dec'},'primary_full_cohort':primary.to_dict('records'),'standardized_trends':trends.to_dict('records'),'sequential_contrasts':contrib.to_dict('records'),'paired_same_support':pairrows,'environment_history':envhist.to_dict('records'),'environment_duration_associations':assoc.to_dict('records'),'support_and_attempts':summaries,'Gamma_positive_sensitivity':gamma.to_dict('records'),'F_original_N1':fc.to_dict('records'),'Dec_observation':ob.reset_index().to_dict('records'),'Dec_same_support_coverage_pairing':decrows,'support_sensitivities':ctab('sensitivity_intervals.csv').to_dict('records'),'family_policy':spec['family_policy'],'no_new_fit':True,'new_aggregation_only':'basic-composition versus fully fixed paired CI; elapsed versus valid coverage paired CI, all from original indexed draw bank','F_dominance_inference':'No saved paired dominance/share confidence interval or prespecified dominance test; use R itself shortened. L and B have positive point slopes with marginal intervals spanning zero.'}
dump('C_FULL_PRECISION_AUDIT_DATA.json',payload)
for p,item in SOURCES.items():check('source_unchanged_after_audit_'+Path(p).name,sha(p)==item['sha256'])
dump('SOURCE_MANIFEST.json',dict(input_root=str(R1),output_root=str(OUT),sources=list(SOURCES.values()),source_files=len(SOURCES),source_write_operations=0,model_fit_operations=0,bootstrap_attempts_created=0,existing_draws_used=2000,environment=dict(python=sys.version,temp=str(OUT/'temp'))))
dump('VALIDATION.json',dict(status='PASS_SAVED_EVIDENCE_REUSE',checks=CHECKS,n_checks=len(CHECKS),models_refitted=False,new_experiments=False))
dump('CHECKPOINT.json',dict(status='NUMERIC_AUDIT_COMPLETE_REPORT_PENDING',utc=datetime.now(timezone.utc).isoformat(),n_checks=len(CHECKS),source_files=len(SOURCES),models_refitted=False,new_experiments=False))
print(json.dumps({'checks':len(CHECKS),'source_files':len(SOURCES),'support':summaries,'paired_comparisons':pairrows,'F_sid_endpoints':len(ep),'output':str(OUT)},ensure_ascii=False,indent=2,default=str))
