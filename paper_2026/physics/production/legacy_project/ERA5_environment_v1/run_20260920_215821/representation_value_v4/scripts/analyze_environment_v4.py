"""v4 B1 and prespecified environmental C. No ERA5 extraction or old model fitting."""
from pathlib import Path
import argparse,ast,hashlib,json,time
import numpy as np
import pandas as pd
V=Path(__file__).resolve().parents[1]; R=V.parent; BASE=R/'review_completion_v3_2'; OUT=V/'results/B1'
CORE=['X1_sst_c','X2_shear_ms','X3_mid_rh_pct','X4_d200_1e5s','X5_minus_omega500_pa_s','X6_signed_vo850_1e5s']
UNITS=['degC','m s-1','percent','1e-5 s-1','Pa s-1','1e-5 s-1']; COMPS=['R','U','F','L','B']; CONTRASTS=['F_minus_U','U_minus_R']; B=2000; SEED=20260920
PERIODS={'P1':(1982,1992),'P2':(1993,2003),'P3':(2004,2014),'P4':(2015,2023)}
SHORT_LABEL='Excluding cyclones with exactly one 6-h later near-peak episode'
def read(p):return pd.read_parquet(p) if Path(p).suffix=='.parquet' else pd.read_csv(p,keep_default_na=False,na_values=[''],float_precision='round_trip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def save(d,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.parquet':d.to_parquet(p,index=False)
 else:d.to_csv(p,index=False)
def plan():
 p=V/'configs/active_analysis_plan.json'; a=json.loads(p.read_text(encoding='utf8'));dest=V/a['relative_path']
 assert sha(dest)==a['sha256'],'Active plan digest mismatch'
 z=json.loads(dest.read_text(encoding='utf8'));assert z['C']['short_filter']['rule'].startswith('excludeG1only')
 return z,a

def pure_aggregate():
 p=BASE/'frozen_code_dependencies/review_fix_v3/e1_completion/complete_e1.py';tree=ast.parse(p.read_text(encoding='utf8'))
 funcs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='aggregate'],type_ignores=[])
 ns={'pd':pd,'np':np,'CORE':CORE};exec(compile(funcs,str(p)+':aggregate_only','exec'),ns);return ns['aggregate']

def interval_sources(d,inter,track,config):
 """One factor mask is applied to all sets; duration support is independent of ERA5 QC."""
 d=d.sort_values('sid').reset_index(drop=True);q=inter[inter.sid.isin(d.sid)].copy();q.UTC=pd.to_datetime(q.UTC,utc=True)
 tr=track[['sid','UTC','row_id','interval_hours_v3','legal_interval_v3',*[x+s for x in CORE for s in ['_value','_qc','_valid_fraction']]]].copy();tr.UTC=pd.to_datetime(tr.UTC,utc=True)
 assert not q.duplicated(['sid','UTC']).any() and not tr.duplicated(['sid','UTC']).any()
 if config=='C5':
  for c in ['R','U','F']:q['mask_'+c]=q['mask_'+c+'5']
  q['mask_L']=q.mask_U&~q.mask_R;q['mask_B']=q.mask_F&~q.mask_U
 q=q[['sid','UTC','interval_end','valid_6h','valid_h',*['mask_'+c for c in COMPS]]].merge(tr,on=['sid','UTC'],how='left',validate='1:1',indicator=True)
 if not q._merge.eq('both').all():raise ValueError('Missing authoritative track observations for requested SID-UTC')
 # Existing canonical legal geometry must agree before any averaging.
 if not np.array_equal(q.valid_h.to_numpy(float),q.interval_hours_v3.to_numpy(float)):raise AssertionError('Physical interval support differs from accepted v3.2')
 if not (q.valid_6h.to_numpy(bool)==q.legal_interval_v3.to_numpy(bool)).all():raise AssertionError('Legal interval flag mismatch')
 q=q.drop(columns='_merge');q['config']=config
 assert ((q.mask_R.astype(int)+q.mask_L.astype(int)+q.mask_B.astype(int))==q.mask_F.astype(int)).all()
 assert ((q.mask_R.astype(int)+q.mask_L.astype(int))==q.mask_U.astype(int)).all()
 rows=[];checks=[];ids=d.sid.to_numpy();meta=d.set_index('sid')
 for x in CORE:
  usable=q[x+'_qc'].fillna(False).astype(bool)&q[x+'_value'].notna()&q.valid_h.gt(0)
  if not q.loc[usable,x+'_valid_fraction'].ge(.9).all():raise AssertionError('Inherited spatial90 fails')
  for c in COMPS:
   take=q['mask_'+c].astype(bool)&q.valid_h.gt(0);z=q[take].copy();z['ok']=usable[take];z['weighted_value']=z[x+'_value'].where(z.ok,0)*z.valid_h
   stats=z.groupby('sid').agg(physical_h=('valid_h','sum'),weighted_sum=('weighted_value','sum'))
   good=z[z.ok].groupby('sid').agg(valid_h=('valid_h','sum'),n_valid_intervals=('sid','size'))
   for sid in ids:
    ph=float(stats.loc[sid,'physical_h']) if sid in stats.index else 0.;vh=float(good.loc[sid,'valid_h']) if sid in good.index else 0.;n=int(good.loc[sid,'n_valid_intervals']) if sid in good.index else 0
    value=float(stats.loc[sid,'weighted_sum']/vh) if vh else np.nan
    expected=float(meta.loc[sid,'T_'+(c+'5' if config=='C5' and c in ['R','U','F'] else c)+'_h']) if not(config=='C5' and c in ['L','B']) else float(meta.loc[sid,'T_'+('U5' if c=='L' else 'F5')+'_h']-meta.loc[sid,'T_'+('R5' if c=='L' else 'U5')+'_h'])
    if not np.isclose(ph,expected,rtol=0,atol=0):raise AssertionError('Physical duration mismatch '+str((config,sid,c,ph,expected)))
    rows.append(dict(config=config,sid=sid,variable=x,component=c,physical_h=ph,valid_h=vh,n_valid_intervals=n,estimate=value,coverage_fraction=vh/ph if ph else np.nan,structural_empty=ph==0,usable=ph>0 and vh>=6,exclusion_reason='structural_empty' if ph==0 else ('nonempty_no_valid_environment_interval' if vh==0 else ''),genesis_year=int(meta.loc[sid,'genesis_year']),period=meta.loc[sid,'period'],basin_genesis=meta.loc[sid,'basin_genesis'],multi=bool(meta.loc[sid,'multi'])))
 # Validate the aggregation itself against the inherited pure exec function, no old analysis run.
 comp=pd.DataFrame(rows);ag=pure_aggregate()
 for c in COMPS:
  sub=q[q['mask_'+c]&q.valid_h.gt(0)].copy();sub['basin_genesis']=sub.sid.map(meta.basin_genesis);sub['interval_hours_v3']=sub.valid_h
  old=ag(sub,['sid']);now=comp[comp.component.eq(c)&comp.valid_h.gt(0)]
  if old.empty:
   assert now.empty,'Empty inherited aggregation has nonempty new counterpart'
   checks.append(dict(config=config,component=c,rows=0,max_abs_difference=0.,inherited_function='complete_e1.aggregate AST/exec only; empty preserved'));continue
  merged=old.merge(now,on=['sid','variable'],suffixes=('_legacy','_v4'),validate='1:1')
  if len(merged)!=len(now):raise AssertionError('Inherited aggregation membership mismatch')
  error=float(np.max(np.abs(merged.estimate_legacy-merged.estimate_v4))) if len(merged) else 0.
  if not np.allclose(merged.estimate_legacy,merged.estimate_v4,rtol=1e-12,atol=1e-11):raise AssertionError('Pure aggregate numerical disagreement')
  checks.append(dict(config=config,component=c,rows=len(merged),max_abs_difference=error,inherited_function='complete_e1.aggregate AST/exec only'))
 return comp,q,pd.DataFrame(checks)

def contrasts_from_components(comp):
 rows=[]
 for (config,sid,x),z in comp.groupby(['config','sid','variable'],sort=False):
  t=z.set_index('component');base={k:z.iloc[0][k] for k in ['config','sid','variable','genesis_year','period','basin_genesis','multi']}
  for a,b,part,nm in [('F','U','B','F_minus_U'),('U','R','L','U_minus_R')]:
   reasons=[]
   if t.loc[b,'valid_h']<6:reasons.append('required_'+b+'_has_no_valid_6h')
   structural=bool(t.loc[part,'structural_empty'])
   if not structural and t.loc[part,'valid_h']<6:reasons.append('nonempty_'+part+'_has_no_valid_6h')
   if t.loc[a,'valid_h']<6:reasons.append('required_'+a+'_has_no_valid_6h')
   eligible=not reasons
   value=float(t.loc[a,'estimate']-t.loc[b,'estimate']) if eligible else np.nan
   identity=0. if structural and eligible else (float(t.loc[part,'valid_h']/t.loc[a,'valid_h']*(t.loc[part,'estimate']-t.loc[b,'estimate'])) if eligible else np.nan)
   if eligible and not np.isclose(value,identity,rtol=1e-10,atol=2e-12):raise AssertionError('PerSID weighted identity fails')
   rows.append(dict(**base,contrast=nm,estimate=value,included=eligible,exclusion_reason=';'.join(reasons),part_structural_empty=structural,part_physical_h=t.loc[part,'physical_h'],part_valid_h=t.loc[part,'valid_h'],a_physical_h=t.loc[a,'physical_h'],a_valid_h=t.loc[a,'valid_h'],b_physical_h=t.loc[b,'physical_h'],b_valid_h=t.loc[b,'valid_h'],identity_rhs=identity,identity_error=value-identity if eligible else np.nan))
 return pd.DataFrame(rows)

def matrix_for(d,con):
 ids=d.sid.to_numpy(str);cols=[(x,c) for x in CORE for c in CONTRASTS]
 piv=con.pivot(index='sid',columns=['variable','contrast'],values='estimate').reindex(index=ids,columns=pd.MultiIndex.from_tuples(cols));return piv.to_numpy(float),cols

def weighted_boot(matrix,counts):
 ok=np.isfinite(matrix);den=counts@ok.astype(float);num=counts@np.where(ok,matrix,0.)
 val=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);return val,den.astype(int)

def summary_scope(d,con,label,ref,year_draws,plan_sha):
 dest=OUT/'scopes'/label;dest.mkdir(parents=True,exist_ok=True);d=d.sort_values('sid').reset_index(drop=True);m,cols=matrix_for(d,con);n=len(d)
 signature=hashlib.sha256(pd.util.hash_pandas_object(d[['sid','genesis_year','period','basin_genesis','multi']],index=False).to_numpy().tobytes()+np.nan_to_num(m,nan=9.876e199).tobytes()+plan_sha.encode()+sha(Path(__file__)).encode()).hexdigest()
 cp=dest/'complete.json'
 if cp.exists():
  old=json.loads(cp.read_text(encoding='utf8'));assert old['signature']==signature,'Existing B1 checkpoint differs; do not overwrite'
  for f,h in old['files'].items():assert sha(dest/f)==h
  print('Reuse verified B1 scope',label,flush=True);return read(dest/'summary.csv'),read(dest/'period_change.csv'),read(dest/'period_means.csv'),read(dest/'membership.csv')
 if n<1:raise ValueError('Empty requested B1 scope '+label)
 idx=np.random.default_rng(SEED).integers(0,n,size=(B,n),dtype=np.int32);counts=np.stack([np.bincount(r,minlength=n) for r in idx]).astype(np.uint16)
 assert np.all(counts.sum(1)==n)
 val,dn=weighted_boot(m,counts);point=np.nanmean(m,axis=0);rows=[]
 for j,(x,c) in enumerate(cols):
  ok=np.isfinite(val[:,j]);ci=np.quantile(val[ok,j],[.025,.975]) if ok.any() else [np.nan,np.nan];sd=float(ref.loc[x,'reference_sd'])
  rows.append(dict(scope=label,variable=x,contrast=c,estimate=point[j],ci_low=ci[0],ci_high=ci[1],estimate_sd=point[j]/sd,ci_low_sd=ci[0]/sd,ci_high_sd=ci[1]/sd,reference_sd=sd,n_sid=int(np.isfinite(m[:,j]).sum()),n_sampling_frame=n,B_planned=B,B_valid=int(ok.sum()),B_failed=int((~ok).sum()),min_draw_valid_n=int(dn[:,j].min()),max_draw_valid_n=int(dn[:,j].max()),unit=UNITS[CORE.index(x)],P=np.nan,q=np.nan,test_family='none; descriptive interval',seed=SEED,estimand='SID-equal paired difference',selection_label=SHORT_LABEL if label.startswith('exclude_G1') else label))
 np.savez_compressed(dest/'sid_bootstrap.npz',sid_multiplicity=counts,sids=d.sid.to_numpy(str),values=val,valid_n=dn,variables=np.array([x for x,c in cols]),contrasts=np.array([c for x,c in cols]))
 member=con[con.sid.isin(d.sid)].copy();member['scope']=label;member=member.merge(d[['sid','later_group']],on='sid',how='left',validate='m:1')
 if label.startswith('exclude_G1'):
  assert not member.later_group.eq('G1').any()
  # G2 is retained by the selection, although an independent factor QC may still fail.
  assert set(d.loc[d.later_group.eq('G2'),'sid'])==set(member.loc[member.later_group.eq('G2'),'sid'])
 stats=pd.DataFrame(rows);save(stats,dest/'summary.csv');save(member,dest/'membership.csv')
 pd.DataFrame(m,index=d.sid,columns=[x+'|'+c for x,c in cols]).rename_axis('sid').to_csv(dest/'paired_input.csv')
 perpoints=[];pb={};nb={};percount={}
 for per,(lo,hi) in PERIODS.items():
  sel=d.genesis_year.between(lo,hi).to_numpy();pm=m[sel];years=d.loc[sel,'genesis_year'].to_numpy(int);valid=np.isfinite(pm);summ=np.stack([np.where(valid[years==y],pm[years==y],0).sum(0) for y in range(lo,hi+1)]);nvalid=np.stack([valid[years==y].sum(0) for y in range(lo,hi+1)])
  drawidx=year_draws[per]-lo;num=summ[drawidx].sum(1);den=nvalid[drawidx].sum(1);pb[per]=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);nb[per]=den
  pp=np.nanmean(pm,axis=0) if len(pm) else np.full(len(cols),np.nan);percount[per]=valid.sum(0)
  for j,(x,c) in enumerate(cols):perpoints.append(dict(scope=label,period=per,variable=x,contrast=c,estimate=pp[j],estimate_sd=pp[j]/float(ref.loc[x,'reference_sd']),n_sid=int(valid[:,j].sum()),n_sampling_frame=int(sel.sum()),unit=UNITS[CORE.index(x)]))
 dif=pb['P4']-pb['P1'];pp=pd.DataFrame(perpoints);changes=[]
 for j,(x,c) in enumerate(cols):
  q=pp[pp.variable.eq(x)&pp.contrast.eq(c)].set_index('period');value=q.loc['P4','estimate']-q.loc['P1','estimate'];ok=np.isfinite(dif[:,j]);ci=np.quantile(dif[ok,j],[.025,.975]) if ok.any() else [np.nan,np.nan];sd=float(ref.loc[x,'reference_sd'])
  changes.append(dict(scope=label,variable=x,contrast=c,period_contrast='P4 minus P1',estimate=value,ci_low=ci[0],ci_high=ci[1],estimate_sd=value/sd,ci_low_sd=ci[0]/sd,ci_high_sd=ci[1]/sd,P1_n_sid=int(percount['P1'][j]),P4_n_sid=int(percount['P4'][j]),B_planned=B,B_valid=int(ok.sum()),B_failed=int((~ok).sum()),unit=UNITS[CORE.index(x)],P=np.nan,q=np.nan,test_family='none; descriptive interval',seed=SEED,estimand='SID-equal period mean difference; year repeats preserve all SID multiplicities'))
 change=pd.DataFrame(changes);save(change,dest/'period_change.csv');save(pp,dest/'period_means.csv');np.savez_compressed(dest/'period_bootstrap.npz',**pb,**{k+'_valid_n':v for k,v in nb.items()},difference=dif,variables=np.array([x for x,c in cols]),contrasts=np.array([c for x,c in cols]))
 dump(cp,dict(signature=signature,active_plan_sha256=plan_sha,source='same matrix + actual stored draws',files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'}))
 print('B1 scope complete',label,'sampling frame',n,'validn',stats.n_sid.min(),stats.n_sid.max(),flush=True)
 return stats,change,pp,member

def state_audit(d,inter):
 s=read(BASE/'baseline_authority_metadata/audit/h1_existing_state_crosswalk_v3_1.csv');s.UTC=pd.to_datetime(s.UTC,utc=True)
 q=inter[inter.mask_B&inter.valid_h.gt(0)].copy();q=q[['sid','UTC','interval_end','valid_h','bridge_state']];q.UTC=pd.to_datetime(q.UTC,utc=True);q.interval_end=pd.to_datetime(q.interval_end,utc=True)
 fields=['sid','UTC','state_class','USA_STATUS','NATURE'];q=q.merge(s[fields],on=['sid','UTC'],how='left',validate='m:1');end=s[fields].rename(columns={'UTC':'interval_end','state_class':'state_end','USA_STATUS':'USA_STATUS_end','NATURE':'NATURE_end'});q=q.merge(end,on=['sid','interval_end'],how='left',validate='m:1')
 conflict=q.NATURE.eq('MX')|q.NATURE_end.eq('MX');unknown=q.state_class.isna()|q.state_end.isna()|q.state_class.eq('C')|q.state_end.eq('C');tropical=q.state_class.eq('A')&q.state_end.eq('A')
 q['independent_state']=np.select([conflict,unknown,tropical],['agency_conflict','unknown','both_endpoints_tropical'],default='other');assert q.independent_state.eq(q.bridge_state).all()
 rows=[]
 for sid in d.sid:
  z=q[q.sid.eq(sid)];r={'sid':sid,'strict_independent':len(z)==0 or z.independent_state.eq('both_endpoints_tropical').all()}
  for k in ['both_endpoints_tropical','other','unknown','agency_conflict']:r['B_'+k+'_h']=z.loc[z.independent_state.eq(k),'valid_h'].sum()
  rows.append(r)
 a=pd.DataFrame(rows);v=a.merge(d[['sid','strict_tropical_bridge',*[c for c in a if c.startswith('B_')]]],on='sid',validate='1:1',suffixes=('_independent','_source'))
 assert v.strict_independent.eq(v.strict_tropical_bridge).all()
 for c in a:
  if c.startswith('B_'):assert np.array_equal(v[c+'_independent'],v[c+'_source'])
 save(q,OUT/'bridge_state_intervals_independent.csv');save(a,OUT/'bridge_state_SID_independent.csv');save(v,OUT/'bridge_state_crosscheck.csv')
 return {'bridge_intervals':len(q),'bridge_sid':q.sid.nunique(),'all_endpoint_source_rows_found':bool(q.state_class.notna().all()&q.state_end.notna().all()),'independent_state_agrees':True,'strict_agrees':True}

def loo(d,con,ref):
 m,cols=matrix_for(d,con);ok=np.isfinite(m);s=np.where(ok,m,0.).sum(0);n=ok.sum(0);base=s/n;rows=[]
 for k,r in enumerate(d.itertuples()):
  nn=n-ok[k];mm=(s-np.where(ok[k],m[k],0))/nn
  for j,(x,c) in enumerate(cols):rows.append(dict(deleted_sid=r.sid,deleted_name=r.name,deleted_basin=r.basin_genesis,variable=x,contrast=c,estimate=mm[j],change_from_main=mm[j]-base[j],estimate_sd=mm[j]/float(ref.loc[x,'reference_sd']),change_from_main_sd=(mm[j]-base[j])/float(ref.loc[x,'reference_sd']),n_remaining=int(nn[j]),deleted_was_environment_eligible=bool(ok[k,j]),unit=UNITS[CORE.index(x)]))
 save(pd.DataFrame(rows),OUT/'leave_one_sid_influence.csv')

def group_contributions(d,con,ref):
 a=con.merge(d[['sid','later_group','later_interval_episode_n','later_max_episode_h','T_L_h']],on='sid',validate='m:1');rows=[]
 for (x,c),z in a.groupby(['variable','contrast']):
  den=int(z.included.sum())
  for group,g in z.groupby('later_group'):
   valid=g[g.included];rows.append(dict(variable=x,contrast=c,later_group=group,n_sampling_frame=len(g),n_environment_valid=len(valid),environment_valid_denominator=den,mean_difference=valid.estimate.mean(),mean_difference_sd=valid.estimate.mean()/float(ref.loc[x,'reference_sd']),additive_contribution_to_all_SID_mean=valid.estimate.sum()/den,additive_contribution_sd=valid.estimate.sum()/den/float(ref.loc[x,'reference_sd']),total_later_h=g.T_L_h.sum(),longest_later_episode_h=g.later_max_episode_h.max(),unit=UNITS[CORE.index(x)],P=np.nan,q=np.nan))
 save(pd.DataFrame(rows),OUT/'C2_group_contributions.csv');save(a,OUT/'C2_group_environment_membership.csv')

def main():
 _,active=plan();OUT.mkdir(parents=True,exist_ok=True);started=time.time()
 dep=[V/'source_data/durations.csv',V/'source_data/intervals.parquet',V/'inputs/year_block_draws.npz',BASE/'sources/track_sp_qc_v3_2.parquet',V/'audit/environment_sources/B1_frozen_reference_scale.csv',V/'analysis_plan_v4_v2.json']
 for beta in ['085','095']:dep += [V/'source_data'/('durations_beta'+beta+'.csv'),V/'source_data'/('intervals_beta'+beta+'.parquet')]
 for p in dep:
  if not p.exists():raise FileNotFoundError('B1 dependency not ready: '+str(p))
 before={str(p):sha(p) for p in dep};ref=read(dep[4]).set_index('variable');track=read(dep[3]);d=read(dep[0]);inter=read(dep[1]);d=d[d.main_complete].sort_values('sid').reset_index(drop=True)
 assert len(d)==2366 and len(d[d.multi])==123;assert 'NA' in set(d.basin_genesis)
 year=np.load(dep[2]);year_draws={p:year[p] for p in PERIODS}
 for p,(lo,hi) in PERIODS.items():assert year_draws[p].shape==(B,hi-lo+1) and ((year_draws[p]>=lo)&(year_draws[p]<=hi)).all()
 configs={'main':(d,inter)}
 for beta in ['085','095']:
  q=read(V/'source_data'/('durations_beta'+beta+'.csv'));q=q[q.main_complete].copy();q['within_authoritative_ERA5_SID']=q.sid.isin(track.sid)
  save(q[['sid','within_authoritative_ERA5_SID']],OUT/('beta'+beta+'_source_eligibility.csv'));q=q[q.within_authoritative_ERA5_SID].copy();g=read(V/'source_data'/('intervals_beta'+beta+'.parquet'));configs['beta'+beta]=(q,g)
 configs['C5']=(d[d.six_complete].copy(),inter)
 comp_all=[];con_all=[];aggchecks=[];lookup={}
 for label,(dd,ii) in configs.items():
  print('B1 aggregate',label,len(dd),flush=True);comp,q,checks=interval_sources(dd,ii,track,label);con=contrasts_from_components(comp);comp_all.append(comp);con_all.append(con);aggchecks.append(checks);lookup[label]=con
  save(q,OUT/('interval_environment_'+label+'.parquet'))
 save(pd.concat(comp_all,ignore_index=True),OUT/'per_SID_component_environment.parquet');save(pd.concat(comp_all,ignore_index=True),OUT/'per_SID_component_environment.csv');save(pd.concat(con_all,ignore_index=True),OUT/'per_SID_environment_contrasts.csv');save(pd.concat(aggchecks,ignore_index=True),OUT/'inherited_aggregate_equivalence.csv')
 sa=state_audit(read(dep[0]),inter);loo(d,lookup['main'],ref);group_contributions(d,lookup['main'],ref)
 scopes=[('main_all','main',d),('main_multi','main',d[d.multi]),('main_nonmulti','main',d[~d.multi]),('exclude_G1_all','main',d[d.later_group.ne('G1')]),('exclude_G1_multi','main',d[d.multi&d.later_group.ne('G1')]),('strict_bridge_all','main',d[d.strict_tropical_bridge]),('strict_bridge_multi','main',d[d.multi&d.strict_tropical_bridge]),('six_common_main_all','main',d[d.six_complete]),('six_common_main_multi','main',d[d.six_complete&d.multi]),('six_common_C5_all','C5',d[d.six_complete]),('six_common_C5_multi','C5',d[d.six_complete&d.multi])]
 for label,year in [('Halola',2015),('Freddy',2023)]:
  hit=d[d.name.str.upper().eq(label.upper())&d.genesis_year.eq(year)];assert len(hit)==1,'Preset case identity ambiguous';sid=hit.sid.iloc[0]
  scopes += [('without_'+label+'_all','main',d[d.sid.ne(sid)]),('without_'+label+'_multi','main',d[d.multi&d.sid.ne(sid)])]
 for basin in ['WP','EP','SI','NA','SP','NI']:scopes += [('without_basin_'+basin+'_all','main',d[d.basin_genesis.ne(basin)]),('without_basin_'+basin+'_multi','main',d[d.multi&d.basin_genesis.ne(basin)])]
 for beta in ['085','095']:
  label='beta'+beta;q=configs[label][0];common=set(d.sid)&set(q.sid)
  scopes.extend([(label+'_available',label,q),(label+'_common',label,q[q.sid.isin(common)]),('main_for_'+label,'main',d[d.sid.isin(common)])])
 common3=set.intersection(*[set(configs[k][0].sid) for k in ['main','beta085','beta095']])
 for label in ['main','beta085','beta095']:scopes.append((label+'_three_beta_common',label,configs[label][0][configs[label][0].sid.isin(common3)]))
 ss=[];pc=[];pm=[];mm=[];scopeaudit=[]
 for label,config,dd in scopes:
  if not len(dd):continue
  a,b,c,m=summary_scope(dd,lookup[config],label,ref,year_draws,active['sha256']);ss.append(a);pc.append(b);pm.append(c);mm.append(m)
  scopeaudit.append(dict(scope=label,config=config,n_sampling_frame=len(dd),multi_n=int(dd.multi.sum()),G1_in_frame=int(dd.later_group.eq('G1').sum()),G2_in_frame=int(dd.later_group.eq('G2').sum()),n_environment_valid_min=int(a.n_sid.min()),n_environment_valid_max=int(a.n_sid.max()),selection_label=SHORT_LABEL if label.startswith('exclude_G1') else label))
 save(pd.concat(ss,ignore_index=True),OUT/'paired_difference_summary.csv');save(pd.concat(pc,ignore_index=True),OUT/'period_difference_changes.csv');save(pd.concat(pm,ignore_index=True),OUT/'period_difference_means.csv');save(pd.concat(mm,ignore_index=True),OUT/'membership_all_scopes.parquet');save(pd.DataFrame(scopeaudit),OUT/'scope_selection_summary.csv')
 for p,h in before.items():assert sha(p)==h,'Read-only dependency changed: '+p
 audit=dict(status='COMPLETE_B1_AND_ENVIRONMENTAL_C',active_plan=active,source_sha256=before,source_unchanged=True,main_sid=len(d),main_multi_sid=int(d.multi.sum()),scopes=len(scopes),planned_bootstrap=B,seed=SEED,state_independent_validation=sa,temporal_QC='at least one legal valid6h per nonempty required component; physical/valid coverage retained',spatial_QC='accepted90percent; no raw or legacy fallback',shared_draws='One full population SID multiplicity vector per replicate for all factors and both contrasts; valid paired denominator may differ by factor; missing response remains missing, never zero.',period_draws='Reuse exact input year arrays from A, preserve duplicated real years with every eligible SID',formal_new_BH_family=False,H2_or_E1_E6_recomputed=False,raw_NC_read=False,elapsed_s=time.time()-started)
 dump(OUT/'run_audit.json',audit);dump(V/'checkpoints/B1_complete.json',audit);print(json.dumps({'status':audit['status'],'scopes':len(scopes),'elapsed_s':audit['elapsed_s']}),flush=True)
if __name__=='__main__':main()
