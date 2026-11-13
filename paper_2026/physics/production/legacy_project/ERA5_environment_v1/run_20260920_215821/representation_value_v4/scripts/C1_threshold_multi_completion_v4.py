"""Complete only the prespecified C1 recurrence-subset threshold estimates.
Existing A/B1 remain unchanged. No raw ERA5 access, H2 fitting, or new test family.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
import pandas as pd
import analyze_durations_v4 as A
import analyze_environment_v4 as E
V=Path(__file__).resolve().parents[1]; O=V/'results/C1_threshold_multi'; B=2000; SEED=20260920

def boolean(s):
    mapped=s.map(lambda x: str(x).strip().lower()).map({'true':True,'false':False,'1':True,'0':False})
    if mapped.isna().any(): raise ValueError('Unrecognized boolean '+str(s.name))
    return mapped.astype(bool)
def read(p):
    d=pd.read_csv(p,keep_default_na=False,na_values=[''],float_precision='round_trip')
    for c in ['main_complete','six_complete','multi','included','exclude_correct_G1']:
        if c in d:d[c]=boolean(d[c])
    return d

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def save(d,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.parquet':d.to_parquet(p,index=False)
    else:d.to_csv(p,index=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def counts_for(sids):
    # Prefer the existing exact ordered SID draw; resample only new native populations.
    for p in [V/'results/A/sid_draws_main_multi.npz',V/'results/B1/scopes/main_multi/sid_bootstrap.npz']:
        z=np.load(p)
        if np.array_equal(z['sids'],np.asarray(sids,dtype=str)):
            return z['sid_multiplicity'],str(p.relative_to(V)),'REUSED_EXISTING_DRAW'
    n=len(sids);ix=np.random.default_rng(SEED).integers(0,n,size=(B,n),dtype=np.int32)
    return np.stack([np.bincount(r,minlength=n) for r in ix]).astype(np.uint16),'seed20260920; new native cohort','NEW_FOR_MISSING_SCOPE'

def main():
    _,active=E.plan();O.mkdir(parents=True,exist_ok=True)
    if (O/'complete.json').exists():raise RuntimeError('C1 addon already complete; verify rather than repeat')
    deps=[V/'source_data'/f for f in ['durations.csv','durations_beta085.csv','durations_beta095.csv','C2_four_groups_and_selection.csv']]+[V/'results/B1/per_SID_environment_contrasts.csv',V/'inputs/year_block_draws.npz',V/'results/A/sid_draws_main_multi.npz',V/'results/B1/scopes/main_multi/sid_bootstrap.npz',V/'audit/environment_sources/B1_frozen_reference_scale.csv']
    before={str(p.relative_to(V)):sha(p) for p in deps}
    raw=read(deps[0]); main=raw[raw.main_complete].sort_values('sid').reset_index(drop=True);multi=main[main.multi].copy()
    c2=read(deps[3]);assert len(c2)==2368 and len(c2[c2.main_complete&~c2.exclude_correct_G1])==2326
    assert len(main)==2366 and len(multi)==123 and '2002297N09206' in set(multi.sid)
    alt={k:read(V/'source_data'/('durations_'+k+'.csv')) for k in ['beta085','beta095']}
    alt={k:d[d.main_complete].sort_values('sid').reset_index(drop=True) for k,d in alt.items()}
    con=read(V/'results/B1/per_SID_environment_contrasts.csv');ref=read(V/'audit/environment_sources/B1_frozen_reference_scale.csv').set_index('variable')
    yd=np.load(V/'inputs/year_block_draws.npz');year={p:yd[p] for p in A.PERIODS}
    save(main,O/'inputs/main_complete.csv');save(multi,O/'inputs/main90_multi.csv')
    for k,d in alt.items():save(d,O/('inputs/'+k+'_complete.csv'))
    # All A key metrics use native populations separately from fixed-main90 multi common SID.
    A.D=O/'A';A.D.mkdir(exist_ok=True);summ=[];slopes=[];annual=[];periods=[];periodmeans=[];membership=[];flow=[];audit=[]
    for k,d in alt.items():
        native=d[d.multi].sort_values('sid').reset_index(drop=True);common=sorted(set(multi.sid)&set(d.sid));base=multi.set_index('sid').loc[common].reset_index();other=d.set_index('sid').loc[common].reset_index()
        delta=base.copy()
        for c in ['T_R_h','T_U_h','T_F_h','T_L_h','T_B_h','T_R_long_h']:delta[c]=other[c].to_numpy()-base[c].to_numpy()
        scopes=[(k+'_native_multi',native,'threshold-native recurrence; not a pure threshold comparison'),(k+'_fixed90_multi_main',base,'fixed main90 recurrence; common physical SID'),(k+'_fixed90_multi_alternate',other,'fixed main90 recurrence; common physical SID'),(k+'_fixed90_multi_difference',delta,'alternate minus main90 on same SID; direct paired contrast')]
        for label,d0,meaning in scopes:
            d0=d0.sort_values('sid').reset_index(drop=True);m,names=A.metric_matrix(d0);counts,source,disposition=counts_for(d0.sid.to_numpy(str));draw=counts@m/len(d0)
            np.savez_compressed(A.D/(label+'_mean_draws.npz'),sids=d0.sid.to_numpy(str),sid_multiplicity=counts,input_values=m,values=draw,metrics=np.array(names),draw_source=np.array(source))
            save(d0,A.D/(label+'_input.csv'))
            for j,name in enumerate(names):
                ci=np.quantile(draw[:,j],[.025,.975]);summ.append(dict(scope=label,metric=name,statistic='mean',estimate=m[:,j].mean(),ci_low=ci[0],ci_high=ci[1],n_sid=len(d0),unit='h',B_planned=B,B_valid=B,B_failed=0,P=np.nan,q=np.nan,test_family='none; prespecified C1 sensitivity',estimand=meaning))
            sl,an=A.annual(d0,label);pe,pp=A.period(d0,label,year);slopes+=sl;annual+=an;periods+=pe;periodmeans+=pp
            membership.extend([dict(scope=label,sid=s,selection_basis=meaning) for s in d0.sid]);flow.append(dict(module='A',scope=label,n_sid=len(d0),threshold_native_multi_n=int(d0.multi.sum()),selection_basis=meaning))
            audit.append(dict(scope=label,n_sid=len(d0),draw_source=source,disposition=disposition))
    for name,rows in [('duration_summaries',summ),('annual_slopes',slopes),('annual_means',annual),('period_changes',periods),('period_means',periodmeans),('membership',membership)]:save(pd.DataFrame(rows),A.D/(name+'.csv'))
    # C5 A on the fixed 123-SID recurrence subset has already been calculated and verified.
    reused=[]
    for name in ['duration_summaries','annual_slopes','annual_means','period_changes','period_means','membership']:
        p=V/'results/A'/(name+'.csv');z=read(p);z=z[z.scope.eq('six_common_multi')];assert len(z)>0;save(z,A.D/('C5_REUSED_'+name+'.csv'));reused.append(dict(source=str(p.relative_to(V)),source_sha256=sha(p),selected_scope='six_common_multi',rows=len(z),disposition='REUSED_EXISTING_ESTIMATES_NOT_RECALCULATED'))
    # B1 native and common groups; same physical SID and both-threshold factor/contrast QC.
    E.OUT=O/'B1';E.OUT.mkdir(exist_ok=True);allstats=[];allperiod=[];allpoints=[];members=[];pairchecks=[]
    cols=[(x,c) for x in E.CORE for c in E.CONTRASTS]
    def matrix(config,sids):
        z=con[con.config.eq(config)].pivot(index='sid',columns=['variable','contrast'],values='estimate')
        return z.reindex(index=sids,columns=pd.MultiIndex.from_tuples(cols)).to_numpy(float)
    def compute(d0,m,label,meaning,config):
        d0=d0.sort_values('sid').reset_index(drop=True);counts,source,disposition=counts_for(d0.sid.to_numpy(str));draw,valid_n=E.weighted_boot(m,counts);n=len(d0)
        dest=E.OUT/label;dest.mkdir(exist_ok=True);save(d0,dest/'SID_input.csv')
        np.savez_compressed(dest/'draws.npz',sids=d0.sid.to_numpy(str),sid_multiplicity=counts,input_values=m,values=draw,valid_n=valid_n,variables=np.array([x for x,c in cols]),contrasts=np.array([c for x,c in cols]),draw_source=np.array(source))
        pb={};pn={};pt={}
        for per,(lo,hi) in E.PERIODS.items():
            sel=d0.genesis_year.between(lo,hi).to_numpy();pm=m[sel];years=d0.loc[sel,'genesis_year'].to_numpy(int);ok=np.isfinite(pm)
            sums=np.stack([np.where(ok[years==y],pm[years==y],0).sum(0) for y in range(lo,hi+1)]);ns=np.stack([ok[years==y].sum(0) for y in range(lo,hi+1)]);ix=year[per]-lo;num=sums[ix].sum(1);den=ns[ix].sum(1);pb[per]=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);pn[per]=ok.sum(0);pt[per]=np.nanmean(pm,axis=0)
            for j,(x,c) in enumerate(cols):allpoints.append(dict(scope=label,period=per,variable=x,contrast=c,estimate=pt[per][j],n_sid=int(pn[per][j]),unit=E.UNITS[E.CORE.index(x)],selection_basis=meaning))
        pdif=pb['P4']-pb['P1'];np.savez_compressed(dest/'period_draws.npz',**pb,difference=pdif,variables=np.array([x for x,c in cols]),contrasts=np.array([c for x,c in cols]))
        for j,(x,c) in enumerate(cols):
            sd=float(ref.loc[x,'reference_sd']);ok=np.isfinite(draw[:,j]);ci=np.quantile(draw[ok,j],[.025,.975]);point=np.nanmean(m[:,j]);common=dict(scope=label,config=config,variable=x,contrast=c,n_sid=int(np.isfinite(m[:,j]).sum()),n_sampling_frame=n,reference_sd=sd,unit=E.UNITS[E.CORE.index(x)],B_planned=B,P=np.nan,q=np.nan,test_family='none; prespecified C1 sensitivity',selection_basis=meaning)
            allstats.append(dict(**common,estimate=point,ci_low=ci[0],ci_high=ci[1],estimate_sd=point/sd,ci_low_sd=ci[0]/sd,ci_high_sd=ci[1]/sd,B_valid=int(ok.sum()),B_failed=int((~ok).sum())))
            good=np.isfinite(pdif[:,j]);pci=np.quantile(pdif[good,j],[.025,.975]);pp=pt['P4'][j]-pt['P1'][j]
            allperiod.append(dict(**common,estimate=pp,ci_low=pci[0],ci_high=pci[1],estimate_sd=pp/sd,ci_low_sd=pci[0]/sd,ci_high_sd=pci[1]/sd,B_valid=int(good.sum()),B_failed=int((~good).sum()),P1_n_sid=int(pn['P1'][j]),P4_n_sid=int(pn['P4'][j]),period_contrast='P4 minus P1'))
            for i,sid in enumerate(d0.sid):members.append(dict(scope=label,sid=sid,variable=x,contrast=c,included=bool(np.isfinite(m[i,j])),exclusion_reason='' if np.isfinite(m[i,j]) else 'one_or_both_thresholds_environment_QC_incomplete',selection_basis=meaning))
        flow.append(dict(module='B1',scope=label,n_sid=n,n_environment_valid_min=int(np.isfinite(m).sum(0).min()),n_environment_valid_max=int(np.isfinite(m).sum(0).max()),selection_basis=meaning));audit.append(dict(scope=label,n_sid=n,draw_source=source,disposition=disposition));return draw,pdif
    for k,d in alt.items():
        ids=set(con.loc[con.config.eq(k),'sid']);native=d[d.multi&d.sid.isin(ids)].sort_values('sid').reset_index(drop=True);compute(native,matrix(k,native.sid),k+'_native_multi','threshold-native recurrence with available authoritative environment SID',k)
    for k in ['beta085','beta095','C5']:
        physical=alt[k] if k!='C5' else main[main.six_complete]
        ids=sorted(set(multi.sid)&set(physical.sid)&set(con.loc[con.config.eq(k),'sid']));base=multi.set_index('sid').loc[ids].reset_index();a=matrix('main',ids);b=matrix(k,ids);ok=np.isfinite(a)&np.isfinite(b);aa=np.where(ok,a,np.nan);bb=np.where(ok,b,np.nan)
        out=[]
        for suffix,m,cfg in [('main',aa,'main'),('alternate',bb,k),('difference',bb-aa,k+' minus main')]:out.append(compute(base,m,k+'_fixed90_multi_'+suffix,'fixed main90 recurrence; common physical SID and both-threshold factor/contrast QC',cfg))
        for j in [0,1]:np.testing.assert_allclose(out[2][j],out[1][j]-out[0][j],rtol=1e-10,atol=2e-12,equal_nan=True)
        pairchecks.append(dict(alternate=k,n_physical_common_sid=len(ids),n_factor_common_min=int(ok.sum(0).min()),n_factor_common_max=int(ok.sum(0).max()),direct_draw_difference_identity=True))
    for name,rows in [('paired_difference_summary',allstats),('period_difference_changes',allperiod),('period_difference_means',allpoints)]:save(pd.DataFrame(rows),E.OUT/(name+'.csv'))
    save(pd.DataFrame(members),E.OUT/'membership.parquet');save(pd.DataFrame(flow),O/'sample_flow.csv');save(pd.DataFrame(audit),O/'draw_reuse.csv');save(pd.DataFrame(reused),O/'C5_A_existing_results_binding.csv')
    for p,h in before.items():assert sha(V/p)==h,'Protected input changed '+p
    dump(O/'complete.json',dict(status='COMPUTED_PENDING_INDEPENDENT_VERIFICATION',active_plan=active,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=before,pair_checks=pairchecks,C2_selection_n=2326,main_source_n=2366,main_multi_n=123,no_main_results_changed=True,no_new_BH_family=True,no_ERA5_or_H2_work=True,script_sha256=sha(__file__)))
    print(json.dumps({'status':'COMPUTED','A_scopes':len(set(r['scope'] for r in summ)),'B1_scopes':len(set(r['scope'] for r in allstats)),'pair_checks':pairchecks}),flush=True)
if __name__=='__main__':main()
