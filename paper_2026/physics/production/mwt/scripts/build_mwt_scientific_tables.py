"""Build review and editable-table data only from a completed full MWT statistics run."""
from pathlib import Path
import argparse,datetime,hashlib,json,os
import numpy as np,pandas as pd

FILES={
 'effects':'effects.csv','primary':'primary_family.csv','sensitivity':'sensitivities.csv',
 'motion':'motion_intensity_summary.csv','motion_draws':'motion_intensity_draws.csv',
 'coverage_time':'coverage_n_by_time.csv','coverage_pre':'coverage_pre_counts.csv',
 'loss':'event_loss_reasons.csv','primary_support':'primary_candidate_SID_support.csv',
 'reference':'reference13_support.csv','reference_summary':'reference13_population_summary.csv',
 'hydro':'hydrostatic_saved.csv','units':'metric_units.csv'}
PRIMARY=['KE','APE','C_A_to_K','vertical_input_K_Wkg','vertical_input_A_Wkg','vertical_input_P_Wkg']
WORD_METRICS=PRIMARY+['dKE_dt','dAPE_dt','transport_K_h','transport_A_h','Gamma_K_h','Gamma_A_h','W_PG']
KEY=['scheme','window','level','region','metric']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def jsread(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def finite_number(x):return isinstance(x,(int,float,np.integer,np.floating)) and np.isfinite(x)


def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if value is pd.NA:return None
    return value


def jswrite(path,value):Path(path).write_text(json.dumps(clean_json(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')


def completed_inputs(work):
    work=Path(work).absolute();latest_path=work/'statistics/latest_completed.json'
    if not latest_path.exists():raise RuntimeError('No completed full-queue statistics receipt; no scientific summary created')
    latest=jsread(latest_path);run=Path(latest['path']).absolute()
    if (work/'statistics').resolve() not in run.resolve().parents:raise ValueError('Statistics source is outside the intended workspace')
    rp=run/'completed.json'
    if sha(rp)!=latest['completed_receipt_sha256']:raise ValueError('Latest statistics receipt identity mismatch')
    receipt=jsread(rp);cfg=jsread(run/'analysis_contract.json')
    if receipt['status']!='COMPLETE_FROZEN_DIRECT_MWT_STATISTICS' or receipt['candidate_SID']!=cfg['candidate_SID'] or receipt['candidate_events']!=2*cfg['candidate_SID']:raise ValueError('Not complete frozen paired statistics')
    if sha(run/'analysis_contract.json')!=receipt['input_identity']['config_sha256']:raise ValueError('Statistics configuration identity mismatch')
    frames={};identities={}
    for name,file in FILES.items():
        p=run/file;expected=receipt['outputs'][file]
        if p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']:raise ValueError('Completed statistics output changed: '+file)
        try:frames[name]=pd.read_csv(p,float_precision="round_trip")
        except pd.errors.EmptyDataError:frames[name]=pd.DataFrame()
        if 'level' in frames[name]:frames[name]['level']=frames[name].level.astype(str)
        identities[file]=expected
    sempath=work/'reports/mwt_metric_semantics.json';sem=jsread(sempath)
    identity={'statistics_run':str(run),'statistics_completed_sha256':sha(rp),'latest_pointer_sha256':sha(latest_path),'used_files':identities,'metric_semantics_sha256':sha(sempath),'summary_script_sha256':sha(__file__)}
    return frames,cfg,sem,identity


def annotate(frame,sem,config):
    f=frame.copy();lookup={r['metric']:r for r in sem['metrics']}
    if 'metric' in f:
        if not set(f.metric.dropna())<=set(lookup):raise ValueError('Unknown saved metric')
        if 'units' in f:
            for r in f[['metric','units']].drop_duplicates().itertuples(index=False):
                if r.units!=lookup[r.metric]['units_stored']:raise ValueError('Metric physical units mismatch: '+r.metric)
        f['quantity']=f.metric.map({k:v['short_name_en'] for k,v in lookup.items()})
        f['physical_category']=f.metric.map({k:v['category'] for k,v in lookup.items()})
        f['units_display']=f.metric.map({k:v['units_display'] for k,v in lookup.items()})
    if 'window' in f:f['window_description']=f.window.map(dict(enumerate(config['MWT']['window_labels'])))
    return f


def review_tables(frames,config,sem):
    """No new fits, resampling, hypotheses or significance-based selection."""
    fx=annotate(frames['effects'],sem,config);pr=annotate(frames['primary'],sem,config)
    if len(pr)!=6 or set(pr.metric)!=set(PRIMARY):raise ValueError('Primary family is not the declared six outcomes')
    if pr.duplicated('metric').any() or not pr.scheme.eq('main').all() or not pr.region.eq('r0_300').all() or not pr.window.eq(1).all():raise ValueError('Primary estimand identity mismatch')
    if set(pr.loc[pr.metric.isin(PRIMARY[:3]),'level'])!={'700'} or set(pr.loc[pr.metric.isin(PRIMARY[3:]),'level'])!={'500-850'}:raise ValueError('Primary pressure identity mismatch')
    ss=annotate(frames['sensitivity'],sem,config)
    full=fx[fx.summary.eq('pre')].copy();timecourse=fx[fx.summary.eq('timecourse')].copy()
    for r in sem['metrics']:
        expected={'500-850'} if r['metric'].startswith('vertical_input_') else set(map(str,r['pressure_hpa']))
        for _,g in full[full.metric.eq(r['metric'])].groupby(['scheme','window','region']):
            if set(g.level)!=expected:raise ValueError('Incomplete pressure inventory: '+r['metric'])
    expected_keys={(scheme,window,str(level),region,r['metric']) for scheme in ['main','shift16d'] for window in range(3) for region in ['r0_300','r200_800','r0_500'] for r in sem['metrics'] for level in (['500-850'] if r['metric'].startswith('vertical_input_') else r['pressure_hpa'])}
    actual=set(map(tuple,full[KEY].itertuples(index=False,name=None)))
    if actual!=expected_keys or full.duplicated(KEY).any():raise ValueError('Complete mean-contrast inventory is missing or duplicated')
    expected_time={key+(lag,) for key in expected_keys for lag in config['event_lags_hours']}
    if set(map(tuple,timecourse[KEY+['lag_h']].itertuples(index=False,name=None)))!=expected_time or timecourse.duplicated(KEY+['lag_h']).any():raise ValueError('Time-course key/lag inventory mismatch')
    base_primary=pr.merge(full,on=KEY,suffixes=('_primary','_full'),validate='one_to_one')
    for column in ['n_sid','temporary_mean','final_mean','estimate','lower','upper','p_centered']:
        if not np.array_equal(base_primary[column+'_primary'].to_numpy(),base_primary[column+'_full'].to_numpy(),equal_nan=True):raise ValueError('Primary/full estimate mismatch: '+column)
    for subset,coverage_keys,coverage_name,count in [(timecourse,KEY+['lag_h'],'coverage_time','paired_valid_sid'),(full,KEY,'coverage_pre','eligible_paired_sid')]:
        coverage=frames[coverage_name]
        merged=subset.merge(coverage,on=coverage_keys,validate='one_to_one')
        if len(merged)!=len(subset) or not merged.n_sid.eq(merged[count]).all() or not merged.candidate_sid.eq(config['candidate_SID']).all():raise ValueError('Independent coverage/estimate count mismatch: '+coverage_name)
    # Validate uncertainty reporting without changing or replacing any value.
    finite=full['lower'].notna()&full['upper'].notna()
    if (full.loc[finite,'lower']>full.loc[finite,'upper']).any():raise ValueError('Reversed confidence limits')
    if ((full.n_sid<2)&(full.lower.notna()|full.upper.notna())).any():raise ValueError('Inferential interval with fewer than two SID')
    paired=ss[ss.sensitivity.isin(['input_shift','radius_500'])].rename(columns={'temporary_mean':'alternative_contrast_same_support','final_mean':'main_contrast_same_support','estimate':'alternative_minus_main','lower':'direct_difference_lower','upper':'direct_difference_upper'}).copy()
    paired['comparison']=paired.sensitivity.map({'input_shift':'Input start shifted16d minus main input start','radius_500':'0–500km minus0–300km'})
    paired['target']='Mean within-SID difference between paired contrasts, using identical SID and offsets in both analyses'
    ocean=ss[ss.sensitivity.eq('both_events_ocean')].copy()
    base_main=full[full.scheme.eq('main')&full.region.eq('r0_300')]
    ocean_keys=set(map(tuple,ocean[['window','level','metric']].itertuples(index=False,name=None)))
    unavailable=base_main[[tuple(x) not in ocean_keys for x in base_main[['window','level','metric']].itertuples(index=False,name=None)]].copy()
    if len(unavailable) and not unavailable.n_sid.eq(0).all():raise ValueError('Ocean result missing despite nonempty main paired support')
    unavailable['status']='No stored ocean estimate because the main paired diagnostic has no eligible SID; not a zero effect'
    ocean['selection']='Both frozen events have no recorded land contact during the preceding24h; no future-survival selection'
    motion=annotate(frames['motion'],sem,config)
    if set(motion.metric)!=set(PRIMARY) or motion.duplicated('metric').any():raise ValueError('Incomplete motion/intensity family')
    draws=frames['motion_draws'];statuses=[]
    for row in motion.itertuples(index=False):
        g=draws[draws.metric.eq(row.metric)] if 'metric' in draws else pd.DataFrame()
        if int(row.attempts)==0:
            if len(g):raise ValueError('Draws exist for an unestimated conditional target')
            statuses.append({'metric':row.metric,'status':row.status,'n_attempts':0})
        else:
            if len(g)!=config['bootstrap']['replicates'] or len(g)!=row.attempts or g.duplicated('draw').any() or set(g.draw)!=set(range(int(row.attempts))):raise ValueError('Bootstrap attempt ledger incomplete')
            ok=g.status.eq('ok')
            if int(ok.sum())!=row.successful or int((~ok).sum())!=row.failed:raise ValueError('Bootstrap failure count mismatch')
            if row.failed>0 and row.interval_scope!='conditional on successful target-supported refits':raise ValueError('Failed refits lack conditional interval scope')
            if g.loc[~ok,['adjusted','adjustment']].notna().any().any():raise ValueError('Failed draw carries an invented estimate')
            for status,n in g.groupby('status').size().items():statuses.append({'metric':row.metric,'status':status,'n_attempts':int(n)})
    motion['estimand']='Linear paired contrast at zero pre-window translation-speed and wind-intensity differences'
    motion['raw_mean_scope']='Same energy/covariate-complete SID and offsets as this conditional model'
    motion['adjustment_scope']='Conditional estimate minus raw mean on that same sample; each interval uses the same draw'
    hydro=frames['hydro'].copy();hp=[]
    if set(hydro.units)!={'m2 s-2 Pa-1'} or hydro.duplicated(['sid','event_class','pressure_hpa']).any():raise ValueError('Hydrostatic identity/units mismatch')
    for pressure,g in hydro.groupby('pressure_hpa',sort=True):
        vv=g.pivot(index='sid',columns='event_class',values='mean_absolute_hydrostatic_residual').replace([np.inf,-np.inf],np.nan).dropna(subset=['internal','final'])
        ff=g.pivot(index='sid',columns='event_class',values='finite_fraction').reindex(vv.index)
        d=vv.internal-vv.final
        hp.append(dict(pressure_hpa=pressure,n_paired_sid=len(vv),temporary_mean_absolute_mismatch=vv.internal.mean(),final_mean_absolute_mismatch=vv.final.mean(),mean_paired_difference=d.mean(),median_paired_difference=d.median(),minimum_paired_difference=d.min(),maximum_paired_difference=d.max(),temporary_mean_finite_fraction=ff.internal.mean(),final_mean_finite_fraction=ff.final.mean(),units='m2 s-2 Pa-1',scope='Descriptive paired difference of saved fixed-domain,11-coefficient-time mean absolute dry-hydrostatic mismatch. No new interval or test; not an event-neighborhood energy budget.'))
    ref=frames['reference'].copy();refsum=frames['reference_summary'].copy();extra=[]
    if ref.duplicated(['sid','pressure_hpa']).any() or not ref.groupby('pressure_hpa').sid.nunique().eq(config['candidate_SID']).all() or set(ref.pressure_hpa)!=set(config['pressure_hpa']):raise ValueError('Incomplete thirteen-level reference population')
    if not hydro.groupby(['pressure_hpa','event_class']).sid.nunique().eq(config['candidate_SID']).all():raise ValueError('Incomplete hydrostatic candidate population')
    for pressure,g in ref.groupby('pressure_hpa',sort=True):
        cc=g.c.where(np.isfinite(g.c)&(g.c>0));dc=g.dc_dp.where(np.isfinite(g.dc_dp))
        extra.append(dict(pressure_hpa=pressure,minimum_finite_positive_c=cc.min(),median_finite_positive_c=cc.median(),maximum_finite_positive_c=cc.max(),finite_dc_dp_sid=int(dc.notna().sum()),c_valid_but_dc_unavailable_sid=int((cc.notna()&dc.isna()).sum())))
    refsum=refsum.merge(pd.DataFrame(extra),on='pressure_hpa',validate='one_to_one');refsum['coverage_scope']='Joint above-ground and finite-field coverage; reference uses main1024 times, fixed mask uses union1088 times'
    p_support=frames['primary_support'].copy();sp=p_support.groupby(['metric','level','window','region','scheme','support_status','n_common_offsets'],dropna=False).sid.nunique().rename('n_sid').reset_index()
    semantics=pd.DataFrame(sem['metrics']);dictionary=semantics[['metric','short_name_en','units_stored','units_display','category','pressure_scope','diagnostic_form','positive_sign_meaning','negative_sign_meaning','formula_ascii','source_reference','verified_equation_source','do_not_add_or_overinterpret']].copy()
    tables={'S10a_primary_six':pr,'full_pre_all_schemes_regions':full,'full_timecourse_all_schemes_regions':timecourse,'S10c_common_support_direct_sensitivities':paired,'S10d_past24h_land_free':ocean,'S10d_unavailable_group_inventory':unavailable,'S10e_motion_intensity_targets':motion,'S10e_bootstrap_attempt_status':pd.DataFrame(statuses),'S10f_timepoint_n':frames['coverage_time'].copy(),'S10f_pre_coverage':frames['coverage_pre'].copy(),'S10f_primary_inclusion_reasons':sp,'S10f_event_loss_reasons':frames['loss'].copy(),'S10g_reference13':refsum,'S10g_reference13_SID_values':ref,'S10h_hydrostatic_paired_descriptive':pd.DataFrame(hp),'S10h_hydrostatic_SID_values':hydro,'S10i_metric_dictionary':dictionary}
    commonlevels=full.level.isin(['500','700','850','500-850'])
    tables['S10b_word_candidate_direct_diagnostics']=full[full.scheme.eq('main')&full.region.eq('r0_300')&commonlevels&full.metric.isin(WORD_METRICS)].copy()
    notes=[]
    for row in pr.itertuples(index=False):
        note={'metric':row.metric,'n_sid':int(row.n_sid),'units':row.units,'temporary_mean':row.temporary_mean,'final_mean':row.final_mean,'temporary_minus_final':row.estimate,'CI_lower':row.lower,'CI_upper':row.upper,'P':row.p_centered,'BH_q_six_family':row.q_BH,'interval_contains_zero':bool(row.lower<=0<=row.upper) if finite_number(row.lower) and finite_number(row.upper) else None}
        note['interval_scope']='pointwise percentile95%; does not quantify MWT diagnostic system error'
        notes.append(note)
    review={'candidate_SID':config['candidate_SID'],'primary_outcomes_without_significance_selection':notes,'all_pre_rows':len(full),'timecourse_rows':len(timecourse),'word_candidate_rows':len(tables['S10b_word_candidate_direct_diagnostics']),'scientific_reading_rules':[sem['no_closure_or_donor_claim'],sem['pressure_work_category_note'],sem['boundary_normalization_relation'],'Window0 is slower than16d; window1 is2–16d; window2 is shorter than2d at6h sampling. Input sensitivity shifts the same-length input by16d; it is not a window-length or cutoff sensitivity.','Review raw, common-support input/radius and same-sample conditional estimates together. An interval containing zero is not equivalence evidence.','No sign, P or q is used to choose a table row or diagnostic. These tables do not select manuscript claims automatically.','Hydrostatic mismatch summaries are descriptive of the saved fixed grid and11-time derivative segment; no new resampling or physical-source attribution is performed.']}
    return tables,review


def number(x):
    if x is None or x is pd.NA or (isinstance(x,(float,np.floating)) and not np.isfinite(x)):return 'NA'
    if isinstance(x,(int,np.integer)):return str(int(x))
    if isinstance(x,(float,np.floating)):return format(float(x),'.6g').replace('-','−')
    return str(x)


def word_data(tables):
    """Native-table cell strings plus unrounded machine sources; no raster tables."""
    output=[]
    def add(code,title,frame,columns,headers,note):
        rows=[[number(r.get(c,np.nan)) for c in columns] for _,r in frame.iterrows()]
        output.append({'id':code,'title':title,'headers':headers,'rows':rows,'source_columns':columns,'n_rows':len(rows),'note':note,'display_precision':'Six significant digits for readable cells; complete CSV sources retain original floating values.'})
    common='Temporary minus final; SID means use at least3 identical offsets among−24,−18,−12,−6h. Intervals are pointwise95% percentiles from2000 paired-SID draws.'
    add('S10a','Primary paired energy diagnostics',tables['S10a_primary_six'],['quantity','units_display','n_sid','temporary_mean','final_mean','estimate','lower','upper','p_centered','q_BH'],['Quantity','Unit','SID','Temporary','Final','Difference','Lower95%','Upper95%','P','BH q'],common+' One BH family contains these six outcomes.')
    b=tables['S10b_word_candidate_direct_diagnostics']
    for (window,category),g in b.groupby(['window','physical_category'],sort=True):
        add(f'S10b_w{window}_{category.replace(" ","_")}',f'Direct diagnostics, {g.window_description.iloc[0]}: {category}',g,['quantity','level','units_display','n_sid','temporary_mean','final_mean','estimate','lower','upper'],['Quantity','hPa/layer','Unit','SID','Temporary','Final','Difference','Lower95%','Upper95%'],common+' 0–300km; these are distinct physical quantities, not an additive closed budget. The full13-level, all-region inventory is retained in CSV.')
    for kind in ['input_shift','radius_500']:
        s=tables['S10c_common_support_direct_sensitivities'];s=s[s.sensitivity.eq(kind)&s.window.eq(1)&(((s.level=='700')&s.metric.isin(PRIMARY[:3]))|((s.level=='500-850')&s.metric.isin(PRIMARY[3:])))]
        add('S10c_'+kind,'Matched-support sensitivity: '+kind,s,['quantity','units_display','n_sid','main_contrast_same_support','alternative_contrast_same_support','alternative_minus_main','direct_difference_lower','direct_difference_upper'],['Quantity','Unit','SID','Main contrast','Alternative contrast','Direct change','Lower95%','Upper95%'],'Both contrasts and their direct difference use the same SID and relative times. An alternate-minus-main interval, not overlap of two separate intervals, quantifies sensitivity. Full metric/pressure/window results remain in CSV.')
    o=tables['S10d_past24h_land_free'];o=o[o.window.eq(1)&(((o.level=='700')&o.metric.isin(PRIMARY[:3]))|((o.level=='500-850')&o.metric.isin(PRIMARY[3:])))]
    add('S10d','Pre-event land-contact sensitivity',o,['quantity','units_display','n_sid','temporary_mean','final_mean','estimate','lower','upper'],['Quantity','Unit','SID','Temporary','Final','Difference','Lower95%','Upper95%'],common+' Both events have no recorded land contact in their preceding24h; no future-survival rule.')
    m=tables['S10e_motion_intensity_targets']
    add('S10e1','Conditional zero-speed/zero-intensity-difference target',m,['quantity','units_display','n_sid','raw_mean','estimate','lower','upper','status'],['Quantity','Unit','SID','Same-sample raw','Conditional estimate','Lower95%','Upper95%','Status'],'The conditional target is a regression contrast at zero paired translation-speed and wind-intensity differences. The raw mean is from the same energy/covariate-complete sample.')
    add('S10e2','Adjustment and fixed-refit status',m,['quantity','adjustment','adjustment_lower','adjustment_upper','attempts','successful','failed','interval_scope'],['Quantity','Adjustment','Lower95%','Upper95%','Attempts','Successful','Failed','Interval scope'],'Adjustment subtracts the same-sample raw contrast within each draw. Failed attempts are retained; conditional intervals cannot be read as verified unconditional95%coverage.')
    n=tables['S10f_timepoint_n'];n=n[n.scheme.eq('main')&n.window.eq(1)&n.region.eq('r0_300')&(((n.level=='700')&n.metric.isin(PRIMARY[:3]))|((n.level=='500-850')&n.metric.isin(PRIMARY[3:])))]
    nw=n.pivot(index=['metric','level'],columns='lag_h',values='paired_valid_sid').reset_index();lagcols=[c for c in nw if isinstance(c,(int,float,np.integer,np.floating))]
    add('S10f1','Paired SID counts by departure-relative time',nw,['metric','level']+lagcols,['Quantity','hPa/layer']+[str(int(x))+'h' for x in lagcols],'Each count uses paired valid events at that time. No requirement for later track survival; full counts for all scales, pressure levels and regions are retained in CSV.')
    add('S10f2','Primary pre-window inclusion',tables['S10f_primary_inclusion_reasons'],['metric','level','support_status','n_common_offsets','n_sid'],['Quantity','hPa/layer','Coverage status','Common offsets','SID'],'Counts partition each fixed primary outcome. They must not be added across outcomes to claim more independent cyclones.')
    add('S10g','Shared reference and pressure-level coverage',tables['S10g_reference13'],['pressure_hpa','candidate_sid','finite_positive_c_sid','minimum_finite_positive_c','median_finite_positive_c','maximum_finite_positive_c','finite_dc_dp_sid','median_reference_coverage','median_fixed_joint_area'],['hPa','Candidate SID','Valid c SID','Minimum c','Median c','Maximum c','Finite dc/dp SID','Reference coverage','Fixed-mask coverage'],'c: J kg−1 K−2; dc/dp: J kg−1 K−2 Pa−1. Reference coverage is instantaneous valid coverage over the main1024 samples; fixed-mask coverage combines above-ground and finite-field support over1088 times.')
    add('S10h','Dry-hydrostatic mismatch magnitude',tables['S10h_hydrostatic_paired_descriptive'],['pressure_hpa','n_paired_sid','temporary_mean_absolute_mismatch','final_mean_absolute_mismatch','mean_paired_difference','temporary_mean_finite_fraction','final_mean_finite_fraction'],['hPa','Paired SID','Temporary mean |R|','Final mean |R|','Mean paired difference','Temporary finite fraction','Final finite fraction'],'R=dPhi/dp+Rd T/p, in m² s−2 Pa−1. Descriptive summaries of saved fixed-domain/11-time mean absolute mismatches; no new interval or test and no closure claim.')
    d=tables['S10i_metric_dictionary'];d=d[d.metric.isin(WORD_METRICS)]
    add('S10i','Physical quantities and units',d,['short_name_en','units_display','category','pressure_scope'],['Quantity','Unit','Physical object','Pressure support'],'Storage, local tendencies, space transport, pressure work, same-window conversion and horizontal canonical contributions are separate diagnostics. W_PG already combines pressure transport and conversion under the relevant theoretical identities.')
    return {'purpose':'Editable Word table data for author selection of Table S10 subparts; no DOCX written and no row chosen by significance.','tables':output,'all_machine_tables_preserved':True}


def save_package(dest,tables,review,identity,synthetic=False):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    for name,frame in tables.items():frame.to_csv(dest/(name+'.csv'),index=False,float_format='%.17g')
    wd=word_data(tables);wd['synthetic_schema_test']=synthetic
    jswrite(dest/'Table_S10_editable_data.json',wd);jswrite(dest/'scientific_readout.json',review);jswrite(dest/'input_identity.json',identity)
    inv={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in dest.iterdir() if p.is_file()}
    report={'status':'SYNTHETIC_SCHEMA_VALIDATED' if synthetic else 'COMPLETE_STATISTICAL_SUMMARY_NO_NEW_INFERENCE','outputs':inv,'table_rows':{k:len(v) for k,v in tables.items()},'editable_table_count':len(wd['tables']),'synthetic_schema_test':synthetic,'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()};jswrite(dest/'completed.json',report);return report


def run(work):
    work=Path(work).absolute();frames,cfg,sem,identity=completed_inputs(work);tables,review=review_tables(frames,cfg,sem)
    folder=work/'statistics_review'/('run_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    report=save_package(folder,tables,review,identity)
    jswrite(work/'statistics_review/latest_completed.json',{'path':str(folder),'completed_receipt_sha256':sha(folder/'completed.json')})
    print(json.dumps({'output':str(folder),'editable_tables':report['editable_table_count'],'primary_rows':6,'scope':'Complete saved statistics summarized; no new fit or resampling'}));return folder


def schema_test(work):
    work=Path(work).absolute();cfg=jsread(work/'config/analysis_contract.json');sem=jsread(work/'reports/mwt_metric_semantics.json');eff=[]
    for scheme in ['main','shift16d']:
      for window in range(3):
       for region in ['r0_300','r200_800','r0_500']:
        for met in sem['metrics']:
         levels=['500-850'] if met['metric'].startswith('vertical_input_') else list(map(str,met['pressure_hpa']))
         for level in levels:
          for summary,lag in [('pre',np.nan)]+[('timecourse',lag) for lag in cfg['event_lags_hours']]:
           eff.append(dict(scheme=scheme,window=window,region=region,metric=met['metric'],level=level,summary=summary,lag_h=lag,units=met['units_stored'],n_sid=4,temporary_mean=10.,final_mean=9.,estimate=1.,lower=-.25,upper=2.25,p_centered=.4,draw_key='SYNTHETIC_NOT_REAL'))
    effects=pd.DataFrame(eff);primary=effects[(effects.summary=='pre')&(effects.scheme=='main')&(effects.window==1)&(effects.region=='r0_300')&(((effects.level=='700')&effects.metric.isin(PRIMARY[:3]))|((effects.level=='500-850')&effects.metric.isin(PRIMARY[3:])))].copy();primary['q_BH']=.6
    sens=[]
    for row in effects[(effects.summary=='pre')&(effects.scheme=='main')&(effects.region=='r0_300')].to_dict('records'):
     for name in ['input_shift','radius_500','both_events_ocean']:sens.append({**row,'sensitivity':name})
    motion=[];draws=[]
    for metric in PRIMARY:
     unit='J kg-1' if metric in ['KE','APE'] else 'W kg-1'
     motion.append(dict(metric=metric,n_sid=40,raw_mean=1.,units=unit,status='estimated',attempts=2000,estimate=.9,lower=-.5,upper=2.,successful=1900,failed=100,interval_scope='conditional on successful target-supported refits',adjustment=-.1,adjustment_lower=-.3,adjustment_upper=.2,failure_reasons="{'target_outside_hull':100}"))
     for draw in range(2000):draws.append(dict(metric=metric,draw=draw,status='ok' if draw<1900 else 'target_outside_hull',adjusted=.9 if draw<1900 else np.nan,raw=1.,adjustment=-.1 if draw<1900 else np.nan))
    cov=effects[effects.summary=='timecourse'][KEY+['lag_h']].copy();cov['candidate_sid']=117;cov['temporary_valid_sid']=4;cov['final_valid_sid']=4;cov['paired_valid_sid']=4
    cp=effects[effects.summary=='pre'][KEY].copy();cp['candidate_sid']=117;cp['eligible_paired_sid']=4;cp['excluded_sid']=113
    ps=[]
    for row in primary.to_dict('records'):
     for i in range(117):ps.append({**{k:row[k] for k in KEY},'sid':f'SYN{i}','support_status':'eligible' if i<4 else 'both_events_no_valid_offsets','n_common_offsets':4 if i<4 else 0})
    ref=[];hydro=[]
    for p in cfg['pressure_hpa']:
     for i in range(cfg['candidate_SID']):
      ref.append(dict(sid=f'SYN{i}',pressure_hpa=p,c=2+i,dc_dp=.001,c_finite_positive=True,reference_weight_fraction=.99,fixed_joint_aboveground_finite_area_fraction=.95))
      for event in ['internal','final']:hydro.append(dict(sid=f'SYN{i}',event_class=event,pressure_hpa=p,mean_absolute_hydrostatic_residual=.1+.01*(event=='internal'),finite_fraction=.95,units='m2 s-2 Pa-1'))
    rs=pd.DataFrame([dict(pressure_hpa=p,candidate_sid=cfg['candidate_SID'],finite_positive_c_sid=cfg['candidate_SID'],minimum_reference_coverage=.99,median_reference_coverage=.99,minimum_fixed_joint_area=.95,median_fixed_joint_area=.95) for p in cfg['pressure_hpa']])
    loss=cov.copy();loss['event_class']='internal';loss['availability_reason']='valid';loss['n_sid']=4
    frames={'effects':effects,'primary':primary,'sensitivity':pd.DataFrame(sens),'motion':pd.DataFrame(motion),'motion_draws':pd.DataFrame(draws),'coverage_time':cov,'coverage_pre':cp,'loss':loss,'primary_support':pd.DataFrame(ps),'reference':pd.DataFrame(ref),'reference_summary':rs,'hydro':pd.DataFrame(hydro),'units':effects[['metric','units']].drop_duplicates()}
    tables,review=review_tables(frames,cfg,sem);assert len(tables['full_pre_all_schemes_regions'])==3834 and len(tables['S10b_word_candidate_direct_diagnostics'])==99
    assert len(tables['S10h_hydrostatic_paired_descriptive'])==13 and np.allclose(tables['S10h_hydrostatic_paired_descriptive'].mean_paired_difference,.01)
    # Verify no significance selection: every synthetic interval includes zero.
    assert tables['S10a_primary_six'].q_BH.eq(.6).all() and len(tables['S10a_primary_six'])==6
    bad={**frames,'effects':effects.iloc[1:].copy()};rejected=False
    try:review_tables(bad,cfg,sem)
    except ValueError:rejected=True
    assert rejected
    dest=work/'validation'/('statistics_summary_schema_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));report=save_package(dest,tables,review,{'synthetic':True,'script_sha256':sha(__file__)},synthetic=True)
    for name,frame in tables.items():
     saved=pd.read_csv(dest/(name+'.csv'),float_precision='round_trip')
     assert len(saved)==len(frame)
    wd=jsread(dest/'Table_S10_editable_data.json');assert all(len(row)==len(t['headers']) for t in wd['tables'] for row in t['rows'])
    print(json.dumps({'synthetic_schema_test':True,'output':str(dest),'all_pre_rows':len(tables['full_pre_all_schemes_regions']),'word_candidate_rows':99,'editable_tables':len(wd['tables']),'no_real_energy_effect_read':True,'missing_inventory_rejected':True}));return dest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',default=str(Path(__file__).absolute().parents[1]));ap.add_argument('--schema-test',action='store_true');args=ap.parse_args()
    if args.schema_test:schema_test(args.work)
    else:run(args.work)
