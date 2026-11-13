"""Frozen full-queue MWT statistics; technical I/O mode never estimates effects."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=True),encoding='utf8');os.replace(tmp,p)


def full_queue_gate(work):
    """Bind a complete scientific cache audit to the exact files being read."""
    import pandas as pd
    work=Path(work);cfg=read_json(work/'config/analysis_contract.json')
    events=pd.read_csv(work/'source_data/events_frozen.csv');sids=sorted(events.sid.astype(str).unique())
    if len(sids)!=cfg['candidate_SID'] or len(events)!=2*len(sids) or events.duplicated(['sid','event_class']).any():raise ValueError('Frozen candidate/event population mismatch')
    if not events.groupby('sid').event_class.apply(lambda x:set(x)=={'internal','final'}).all():raise ValueError('Frozen pairing is not one event of each class')
    path=work/'validation/all_cache_audit/summary.json'
    if not path.exists():raise RuntimeError('Full-cache validation summary is not yet available')
    audit=read_json(path);n=len(sids)
    if not(audit.get('complete_population_check') is True and audit.get('requested_SID')==n and audit.get('frozen_candidate_SID')==n and audit.get('validated')==n and audit.get('failed')==0):raise RuntimeError('Full candidate cache validation has not passed: '+str({k:audit.get(k) for k in ['requested_SID','validated','failed','complete_population_check']}))
    if audit['config_sha256']!=file_sha256(work/'config/analysis_contract.json'):raise ValueError('Analysis contract differs from full-cache audit')
    if audit['script_sha256']!=file_sha256(work/'scripts/validate_mwt_all_cache.py'):raise ValueError('Cache audit implementation differs from its completed report')
    status=pd.read_csv(work/'validation/all_cache_audit/SID_validation_summary.csv')
    if sorted(status.sid.astype(str))!=sids or not status.status.eq('VALIDATED_STATED_SCOPE').all():raise ValueError('Per-SID validation summary mismatch')
    sources={p.name:file_sha256(p) for p in (work/'source_data').glob('*.csv')}
    manifest=[]
    for sid in sids:
        folder=work/'cache'/sid;rp=folder/'completed.json';receipt=read_json(rp);perpath=work/'validation/all_cache_audit'/f'{sid}.json';per=read_json(perpath)
        if per['status']!='VALIDATED_STATED_SCOPE' or any(not x['pass'] for x in per['checks']):raise ValueError('Per-SID cache check failed: '+sid)
        if per['receipt_sha256']!=file_sha256(rp):raise ValueError('Completion receipt changed after validation: '+sid)
        if receipt['status']!='COMPLETE_DIRECT_DIAGNOSTICS' or receipt['sid']!=sid:raise ValueError('Incomplete SID: '+sid)
        if receipt['config_sha256']!=audit['config_sha256'] or receipt['small_frozen_sources_sha256']!=sources:raise ValueError('Frozen configuration/source identity mismatch: '+sid)
        for name,identity in per['output_identities'].items():
            fp=folder/name;st=fp.stat()
            if st.st_size!=identity['bytes'] or st.st_mtime_ns!=identity['mtime_ns']:raise ValueError('Validated cache output changed: '+str(fp))
        parquet=folder/'regional_metrics.parquet';actual=file_sha256(parquet)
        if actual!=receipt['outputs_sha256']['regional_metrics.parquet']:raise ValueError('Regional source hash mismatch: '+sid)
        st=parquet.stat()
        manifest.append(dict(sid=sid,regional_path=str(parquet.absolute()),regional_bytes=st.st_size,regional_mtime_ns=st.st_mtime_ns,regional_sha256=actual,receipt_sha256=file_sha256(rp),cache_audit_sha256=file_sha256(perpath),diagnostics_sha256_from_validated_receipt=receipt['outputs_sha256']['diagnostics.h5'],diagnostics_hash_scope='Full-cache audit bound by unchanged receipt, file size and mtime; not rehashed by statistics entry'))
    return cfg,events,sids,manifest,audit


def reference_support(work,sids):
    """Read small reference profiles and joint full-sequence masks only."""
    import h5py,numpy as np,pandas as pd
    rows=[];hydro=[]
    for sid in sids:
        folder=Path(work)/'cache'/sid
        with h5py.File(folder/'diagnostics.h5','r') as h:
            p=h['coordinates/pressure_Pa'][:];op=h['coordinates/omega_pressure_Pa'][:];area=h['basic_state/area_weights'][:]
            t=h['reference/temperature_mean'][:];c=h['basic_state/c'][:];dc=h['basic_state/dc_dp'][:] if 'dc_dp' in h['basic_state'] else np.full(len(p),np.nan)
            fixed=h['support/fixed_1088_pressure_grid'][:];omega=h['support/fixed_1088_omega_grid'][:]
            rw=h['reference/reference_weight_fraction'][:]
            for i,pp in enumerate(p):
                oi=np.flatnonzero(op==pp);wm=omega[oi[0]] if len(oi) else None
                rows.append(dict(sid=sid,pressure_hpa=pp/100,reference_T_K=t[i],c=c[i],c_units='J kg-1 K-2',dc_dp=dc[i],dc_dp_units='J kg-1 K-2 Pa-1',c_finite_positive=bool(np.isfinite(c[i]) and c[i]>0),reference_weight_fraction=rw[i],fixed_joint_aboveground_finite_area_fraction=float(np.sum(area*fixed[i])/area.sum()),fixed_omega_joint_area_fraction=float(np.sum(area*wm)/area.sum()) if wm is not None else np.nan,mask_definition='Above ground and finite required fields at every one of1088 raw times; not a pure ground-only mask'))
        hydro.append(pd.read_csv(folder/'hydrostatic_availability.csv'))
    return pd.DataFrame(rows),pd.concat(hydro,ignore_index=True)


def coverage_tables(rows,events,config):
    """Track sample loss without assigning an unsupported physical cause to NaNs."""
    import numpy as np,pandas as pd
    key=['scheme','window','level','region','metric'];r=rows.copy()
    positive=np.isfinite(r.valid_area_fraction)&r.valid_area_fraction.ge(config['minimum_valid_area_fraction'])
    r['valid']=positive&np.isfinite(r.value)
    if 'position_available' not in r:raise ValueError('Missing saved position-availability flag')
    r['availability_reason']=np.select([~r.position_available,~np.isfinite(r.valid_area_fraction),np.isfinite(r.valid_area_fraction)&~positive,~np.isfinite(r.value)],['no_actual_position','coverage_unavailable','below_fixed_area_threshold','diagnostic_nonfinite'],default='valid')
    if (r['valid']&~r.position_available).any():raise ValueError('Finite diagnostic reported without a real cyclone centre')
    event_cols=['sid','event_id','event_class','scheme','window','level','region','lag_h','metric','units','position_available','valid_area_fraction','valid','availability_reason']
    event=r[event_cols].copy()
    pair=r.pivot(index=['sid']+key+['lag_h'],columns='event_class',values='valid').fillna(False)
    if not {'internal','final'}<=set(pair):raise ValueError('Missing event class in coverage')
    pair['paired_valid']=pair.internal&pair.final
    pair=pair.reset_index();pre=pair[pair.lag_h.isin(config['primary_event_window_hours'])]
    pre=pre.groupby(['sid']+key,sort=True,observed=True).agg(n_common_offsets=('paired_valid','sum'),n_temporary_offsets=('internal','sum'),n_final_offsets=('final','sum')).reset_index()
    pre['eligible']=pre.n_common_offsets.ge(3)
    pre['support_status']=np.select([pre.eligible,pre.n_temporary_offsets.eq(0)&pre.n_final_offsets.eq(0),pre.n_temporary_offsets.lt(3),pre.n_final_offsets.lt(3)],['eligible','both_events_no_valid_offsets','temporary_fewer_than3_offsets','final_fewer_than3_offsets'],default='fewer_than3_identical_offsets')
    ntime=pair.groupby(key+['lag_h'],sort=True,observed=True).agg(candidate_sid=('sid','nunique'),temporary_valid_sid=('internal','sum'),final_valid_sid=('final','sum'),paired_valid_sid=('paired_valid','sum')).reset_index()
    npre=pre.groupby(key,sort=True,observed=True).agg(candidate_sid=('sid','nunique'),eligible_paired_sid=('eligible','sum')).reset_index();npre['excluded_sid']=npre.candidate_sid-npre.eligible_paired_sid
    reasons=event.groupby(key+['event_class','lag_h','availability_reason'],sort=True,observed=True).sid.nunique().rename('n_sid').reset_index()
    return {'event_availability':event,'SID_lag_pair_support':pair,'SID_pre_support':pre,'coverage_n_by_time':ntime,'coverage_pre_counts':npre,'event_loss_reasons':reasons}


def bootstrap_identity_map(result,coverage,events):
    """Recover the sorted SID order represented by each saved bootstrap index bank."""
    import pandas as pd
    identities={};key=['scheme','window','level','region','metric']
    def add(sids):
        ss=sorted(map(str,sids));identity='|'.join(ss);bank=hashlib.sha256(identity.encode()).hexdigest()
        if bank in result['bootstrap_index_banks']:identities[bank]=ss
    p=coverage['SID_lag_pair_support'];p=p[p.paired_valid]
    for _,g in p.groupby(key+['lag_h'],sort=True,observed=True):add(g.sid)
    p=result['sid_pre_pairs']
    for _,g in p.groupby(key,sort=True,observed=True):add(g.sid)
    p=result['sensitivity_pairs']
    if len(p):
        for _,g in p.groupby(['sensitivity','window','level','metric'],sort=True,observed=True):add(g.sid)
    sea=events.pivot(index='sid',columns='event_class',values='land_contact');seaids=set(sea.index[(sea.internal==0)&(sea.final==0)])
    p=result['sid_pre_pairs'];p=p[(p.scheme=='main')&(p.region=='r0_300')&p.sid.isin(seaids)]
    for _,g in p.groupby(['window','level','metric'],sort=True,observed=True):add(g.sid)
    for _,g in result['baseline_pairs'].groupby('metric',sort=True,observed=True):add(g.sid)
    if set(identities)!=set(result['bootstrap_index_banks']):raise ValueError('Cannot reconstruct exact SID order of every index bank')
    return identities


def save_bootstrap(path,result,identities,config):
    import h5py,numpy as np
    with h5py.File(path,'w') as h:
        h.attrs['bootstrap_unit']='SID';h.attrs['replicates']=config['bootstrap']['replicates'];h.attrs['seed']=config['bootstrap']['seed'];h.attrs['interval']='pointwise percentile95%; fixed attempts'
        g=h.create_group('draws')
        for key,values in result['bootstrap_draws'].items():g.create_dataset(key,data=np.asarray(values,dtype=np.float64),compression='gzip',compression_opts=1,shuffle=True)
        g=h.create_group('index_banks')
        for key,values in result['bootstrap_index_banks'].items():
            ds=g.create_dataset(key,data=values,compression='gzip',compression_opts=1,shuffle=True);ds.attrs['sorted_SID_json']=json.dumps(identities[key])


def io_test(work,sid):
    import h5py,numpy as np,pandas as pd
    work=Path(work);events=pd.read_csv(work/'source_data/events_frozen.csv');config=read_json(work/'config/analysis_contract.json')
    if sid not in set(events.sid):raise ValueError('Technical SID is not in frozen events')
    folder=work/'cache'/sid;receipt=read_json(folder/'completed.json');p=folder/'regional_metrics.parquet'
    if receipt['status']!='COMPLETE_DIRECT_DIAGNOSTICS' or file_sha256(p)!=receipt['outputs_sha256'][p.name]:raise ValueError('Technical input is not a verified complete regional table')
    rows=pd.read_parquet(p);cv=coverage_tables(rows,events[events.sid.eq(sid)],config);ref,hydro=reference_support(work,[sid])
    dest=work/'validation'/f'statistics_entry_io_{sid}';dest.mkdir(parents=True,exist_ok=True)
    sizes={}
    for key,frame in cv.items():
        frame.to_parquet(dest/(key+'.parquet'),index=False);back=pd.read_parquet(dest/(key+'.parquet'))
        if not frame.equals(back):raise ValueError('Coverage parquet roundtrip failed: '+key)
        sizes[key]=len(frame)
    ref.to_csv(dest/'reference13_support.csv',index=False);hydro.to_csv(dest/'hydrostatic_saved.csv',index=False)
    report={'scope':'Technical read/write and support structure only; neither statistics function called; no event-group energy effects or scientific conclusions generated.','sid':sid,'regional_rows':len(rows),'unique_metrics':int(rows.metric.nunique()),'metadata_columns':rows.columns.tolist(),'coverage_table_rows':sizes,'reference_rows':len(ref),'hydrostatic_rows':len(hydro),'source_parquet_sha256':file_sha256(p),'statistics_entry_sha256':file_sha256(__file__),'all_roundtrips_exact':True}
    write_json(dest/'receipt.json',report);return report


def run_statistics(work,output=None):
    import h5py,numpy as np,pandas as pd,scipy,pyarrow,platform,psutil
    work=Path(work).absolute();start=time.perf_counter()
    cfg,events,sids,manifest,audit=full_queue_gate(work)
    accepted=read_json(work/'validation/data_statistics_independent.json')
    if not accepted.get('all_pass'):raise ValueError('Independent statistics validation is not accepted')
    for name,digest in accepted['code_sha256'].items():
        if file_sha256(work/'scripts'/name)!=digest:raise ValueError('Statistics function differs from independently validated code: '+name)
    from mwt_paired_statistics import mwt_paired_statistics
    from mwt_motion_intensity_sensitivity import mwt_motion_intensity_sensitivity
    now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    parent=work/'statistics';parent.mkdir(exist_ok=True)
    target=Path(output).absolute() if output else parent/f'run_{now}'
    if parent not in target.parents or target.exists():raise ValueError('Output must be a new path below WORK/statistics')
    stage=target.with_name(target.name+'.working');stage.mkdir(parents=True,exist_ok=False)
    def progress(step,**extra):
        record={'step':step,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,'rss_bytes':psutil.Process().memory_info().rss,**extra}
        write_json(stage/'progress.json',record);print(json.dumps(record),flush=True)
    progress('all_117_validated_inputs_bound')
    codepaths=[Path(__file__),work/'scripts/mwt_paired_statistics.py',work/'scripts/mwt_motion_intensity_sensitivity.py']
    identity={'config_sha256':file_sha256(work/'config/analysis_contract.json'),'statistics_addendum_sha256':file_sha256(work/'config/statistics_addendum.json'),'events_sha256':file_sha256(work/'source_data/events_frozen.csv'),'centres_sha256':file_sha256(work/'source_data/centres_frozen.csv'),'cache_audit_summary_sha256':file_sha256(work/'validation/all_cache_audit/summary.json'),'code_sha256':{p.name:file_sha256(p) for p in codepaths}}
    write_json(stage/'input_identity.json',identity);pd.DataFrame(manifest).to_csv(stage/'input_cache_manifest.csv',index=False)
    for name in ['analysis_contract.json','statistics_addendum.json']:(stage/name).write_bytes((work/'config'/name).read_bytes())
    centres=pd.read_csv(work/'source_data/centres_frozen.csv')
    columns=['sid','event_id','event_class','t0','UTC','scheme','window','level','region','lag_h','metric','value','valid_area_fraction','units','position_available']
    parts=[pd.read_parquet(item['regional_path'],columns=columns) for item in manifest]
    rows=pd.concat(parts,ignore_index=True);del parts
    if set(rows.sid)!=set(sids):raise ValueError('Combined inputs do not contain exact frozen SID population')
    progress('regional_rows_loaded',rows=len(rows))
    coverage=coverage_tables(rows,events,cfg)
    for name,frame in coverage.items():
        frame.to_parquet(stage/(name+'.parquet'),index=False)
        if name in ['coverage_n_by_time','coverage_pre_counts','event_loss_reasons']:frame.to_csv(stage/(name+'.csv'),index=False)
    ref,hydro=reference_support(work,sids);ref.to_csv(stage/'reference13_support.csv',index=False);hydro.to_csv(stage/'hydrostatic_saved.csv',index=False)
    primary=cfg['primary_estimands'];pre=coverage['SID_pre_support'];prim=pre[(pre.scheme=='main')&(pre.window==1)&(pre.region=='r0_300')&(((pre.level=='700')&pre.metric.isin(primary[:3]))|((pre.level=='500-850')&pre.metric.isin(primary[3:])))]
    prim.to_csv(stage/'primary_candidate_SID_support.csv',index=False)
    ref.groupby('pressure_hpa',sort=True).agg(candidate_sid=('sid','nunique'),finite_positive_c_sid=('c_finite_positive','sum'),minimum_reference_coverage=('reference_weight_fraction','min'),median_reference_coverage=('reference_weight_fraction','median'),minimum_fixed_joint_area=('fixed_joint_aboveground_finite_area_fraction','min'),median_fixed_joint_area=('fixed_joint_aboveground_finite_area_fraction','median')).reset_index().to_csv(stage/'reference13_population_summary.csv',index=False)
    progress('coverage_reference_saved')
    kw=dict(replicates=cfg['bootstrap']['replicates'],seed=cfg['bootstrap']['seed'],pre_hours=tuple(cfg['primary_event_window_hours']),min_common=3,min_area=cfg['minimum_valid_area_fraction'])
    result=mwt_paired_statistics(rows,events,centres,**kw)
    for name,frame in result.items():
        if isinstance(frame,pd.DataFrame):
            frame.to_parquet(stage/(name+'.parquet'),index=False);frame.to_csv(stage/(name+'.csv'),index=False)
    # Link all actual draws and their exact sorted SID index banks to the tabulated draw keys.
    banks=bootstrap_identity_map(result,coverage,events);save_bootstrap(stage/'paired_bootstrap.h5',result,banks,cfg)
    effects=result['effects'];check=effects[effects.summary.eq('timecourse')].merge(coverage['coverage_n_by_time'],on=['scheme','window','level','region','metric','lag_h'],validate='one_to_one')
    if len(check)!=int(effects.summary.eq('timecourse').sum()) or not check.n_sid.eq(check.paired_valid_sid).all():raise ValueError('Time-course n/key set disagrees with independent support ledger')
    check=result['effects'][result['effects'].summary.eq('pre')].merge(coverage['coverage_pre_counts'],on=['scheme','window','level','region','metric'],validate='one_to_one')
    if len(check)!=int(result['effects'].summary.eq('pre').sum()) or not check.n_sid.eq(check.eligible_paired_sid).all():raise ValueError('Pre-window n/key set disagrees with independent support ledger')
    progress('paired_statistics_saved',effect_rows=len(result['effects']),bootstrap_draw_vectors=len(result['bootstrap_draws']),bootstrap_index_banks=len(result['bootstrap_index_banks']))
    del result,coverage
    adjusted=mwt_motion_intensity_sensitivity(rows,centres,minimum_SID=30,**kw)
    for name,frame in adjusted.items():frame.to_parquet(stage/('motion_intensity_'+name+'.parquet'),index=False);frame.to_csv(stage/('motion_intensity_'+name+'.csv'),index=False)
    progress('motion_intensity_sensitivity_saved',attempted_draw_rows=len(adjusted['draws']))
    units=rows[['metric','units']].drop_duplicates().sort_values('metric');units.to_csv(stage/'metric_units.csv',index=False)
    doc='''Statistics use the frozen one-temporary/one-final event pair within SID. The main contrast averages temporary-minus-final differences over at least three identical offsets among -24,-18,-12,-6 h. Time-course intervals use paired available SID at each lag, without a post-departure survival condition. Six predeclared outcomes at the middle scale and the main region form one BH family; intervals are pointwise percentile intervals. Other diagnostics retain estimates and pointwise intervals without selected significance claims. Input-window and radius differences are recomputed on identical SID and offsets. The frozen land_contact==0 subset requires both events to have no recorded land contact in the preceding24h. Motion/intensity sensitivity estimates the paired linear contrast at zero paired pre-window speed/intensity differences, only with at least30SID, full rank and joint convex-hull support. All2000attempts are retained; no failed refit is replaced. If failures occur, successful-refit percentile intervals are conditional and do not establish unconditional95%coverage. The adjusted raw mean and adjustment use the same covariate-complete SID sample. Reference/fixed-mask coverage combines above-ground status and finite required fields; it is not a ground-only mask. Cache-level diagnostics and hydrostatic residuals do not establish complete physical energy-budget closure. This computation reads saved direct diagnostics; it does not recompute ERA5, MWT or existing recovery models.\n'''
    (stage/'statistical_scope.txt').write_text(doc,encoding='utf8')
    runtime={'Python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'h5py':h5py.__version__,'pyarrow':pyarrow.__version__};write_json(stage/'runtime.json',runtime)
    # Reject source or analysis-function changes during a long statistics call.
    if identity['code_sha256']!={p.name:file_sha256(p) for p in codepaths}:raise ValueError('Statistics code changed during execution')
    for key,relative in [('config_sha256','config/analysis_contract.json'),('statistics_addendum_sha256','config/statistics_addendum.json'),('events_sha256','source_data/events_frozen.csv'),('centres_sha256','source_data/centres_frozen.csv')]:
        if identity[key]!=file_sha256(work/relative):raise ValueError('Frozen input changed during execution: '+relative)
    outputs={p.name:{'bytes':p.stat().st_size,'sha256':file_sha256(p)} for p in stage.iterdir() if p.is_file() and p.name!='progress.json'}
    complete={'status':'COMPLETE_FROZEN_DIRECT_MWT_STATISTICS','candidate_SID':len(sids),'candidate_events':len(events),'input_rows':len(rows),'input_identity':identity,'outputs':outputs,'elapsed_seconds':time.perf_counter()-start,'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Paired retrospective direct diagnostics and prespecified support sensitivities; no full energy-budget or causal identification claim.'}
    write_json(stage/'completed.json',complete);os.replace(stage,target);write_json(parent/'latest_completed.json',{'path':str(target),'completed_receipt_sha256':file_sha256(target/'completed.json'),'completed_UTC':complete['completed_UTC']})
    print(json.dumps({'completed_path':str(target),'candidate_SID':len(sids),'input_rows':len(rows),'elapsed_seconds':complete['elapsed_seconds']}),flush=True)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',default=str(Path(__file__).absolute().parents[1]));parser.add_argument('--output');m=parser.add_mutually_exclusive_group();m.add_argument('--check-ready',action='store_true');m.add_argument('--io-test-sid')
    args=parser.parse_args()
    if args.io_test_sid:print(json.dumps(io_test(args.work,args.io_test_sid),indent=2))
    elif args.check_ready:
        try:
            cfg,events,sids,manifest,audit=full_queue_gate(args.work);report={'ready':True,'candidate_SID':len(sids),'scope':'Input readiness only; no statistics calculated'}
        except (OSError,ValueError,RuntimeError,KeyError) as error:report={'ready':False,'reason':str(error),'scope':'Input readiness only; no statistics calculated'}
        write_json(Path(args.work)/'validation/statistics_entry_readiness.json',report);print(json.dumps(report,indent=2));raise SystemExit(0 if report['ready'] else 2)
    else:run_statistics(args.work,args.output)

