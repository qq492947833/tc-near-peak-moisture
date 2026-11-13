"""Read-only, bounded-I/O validation of frozen direct-MWT SID caches.

No event-group effects are estimated or used to choose nodes, lags or tolerances.
Default SHA checking streams every output. Optional queue-hash reuse is explicitly
recorded and requires the unchanged audited queue and its completed terminal state.
"""


def validate_mwt_all_cache(work_dir, sids=None, reuse_queue_hashes=False, output_dir=None):
    import datetime
    import hashlib
    import json
    from pathlib import Path
    import h5py
    import numpy as np
    import pandas as pd

    def check_cached_longitude(cached_longitude, source_longitude, domain):
        """Check exact source-column identity and continuous native quarter-degree grid.

        The source-recentering expression and boundary selection are identical to
        the frozen reader. A float64 operation bound is used only for membership
        in the physical quarter-degree grid, not to relax source identity.
        """
        import numpy as np
        cached = np.asarray(cached_longitude)
        native = np.asarray(source_longitude)
        reference = float(domain['lon_reference'])
        bounds = np.asarray(domain['lon_bounds'], dtype=float)
        if cached.dtype != np.float64 or native.dtype != np.float64:
            raise ValueError('Expected the actual float64 cached and source coordinates')
        if cached.ndim != 1 or native.ndim != 1 or min(len(cached), len(native)) < 3:
            raise ValueError('Invalid longitude dimensions')
        if not np.isfinite(cached).all() or not np.isfinite(native).all() or not np.isfinite(reference):
            raise ValueError('Nonfinite longitude coordinate')
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError('Invalid original longitude bounds')
        native_index = np.rint(native * 4).astype(np.int64)
        if not np.array_equal(native, native_index / 4.) or not np.all(np.diff(native_index) == 1):
            raise ValueError('Source is not the contiguous supplied quarter-degree grid')
        recentered = reference + (native - reference + 180) % 360 - 180
        selected = np.flatnonzero((recentered >= bounds[0]) & (recentered <= bounds[1]))
        selected = selected[np.argsort(recentered[selected])]
        expected = recentered[selected]
        if not np.array_equal(cached, expected):
            raise ValueError('Cached longitude does not match exact original source indices')
        index = np.rint(cached * 4).astype(np.int64)
        scale = max(360., abs(reference), float(np.max(np.abs(native))), float(np.max(np.abs(cached))))
        tolerance = 16 * np.finfo(np.float64).eps * scale
        error = np.abs(cached - index / 4.)
        if np.any(error > tolerance):
            raise ValueError('Cached longitude is outside the float64 quarter-grid roundoff bound')
        if not np.all(np.diff(index) == 1):
            raise ValueError('Cached columns are missing, duplicated, reversed or discontinuous')
        modulo = np.mod(index, 1440)
        if not np.array_equal(modulo, np.mod(native_index[selected], 1440)):
            raise ValueError('Native-column modulo identity differs')
        planned = np.arange(int(np.ceil(bounds[0]*4)), int(np.floor(bounds[1]*4))+1)
        missing = sorted(set(planned.tolist()) - set(index.tolist()))
        return {
            'source_column_indices': selected.tolist(),
            'quarter_degree_indices': index.tolist(),
            'source_reconstruction_exact': True,
            'native_modulo_identity_exact': True,
            'integer_spacing_exact': True,
            'coordinate_roundoff_bound_degrees': float(tolerance),
            'maximum_coordinate_roundoff_degrees': float(error.max()),
            'maximum_floating_spacing_difference_degrees': float(np.max(np.abs(np.diff(cached)-.25))),
            'planned_boundary_quarter_indices_not_selected_by_original_reader': missing,
        }

    work=Path(work_dir)
    output=Path(output_dir) if output_dir is not None else work/'validation/all_cache_audit'
    output.mkdir(parents=True,exist_ok=True)
    def sha(path):
        digest=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
        return digest.hexdigest()
    def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
    contract_path=work/'config/analysis_contract.json'
    config=load(contract_path);config_hash=sha(contract_path)
    audit_contract=load(work/'validation/all_cache_validation_contract.json')
    events=pd.read_csv(work/'source_data/events_frozen.csv')
    centres=pd.read_csv(work/'source_data/centres_frozen.csv')
    assert events.sid.nunique()==config['candidate_SID']
    assert len(events)==2*config['candidate_SID']
    assert not events.duplicated(['sid','event_class']).any()
    assert not centres.duplicated(['event_id','lag_h']).any()
    frozen_sids=sorted(events.sid.unique())
    requested=frozen_sids if sids is None else sorted(set(sids))
    assert set(requested)<=set(frozen_sids)
    source_hashes={p.name:sha(p) for p in (work/'source_data').glob('*.csv')}
    operator=work/'cache/mwt_operator_N1024_j5_8_symmetric_physical.npy'
    operator_hash=sha(operator)
    queue=None;queue_timestamp_ns=None
    if reuse_queue_hashes:
        assert sha(work/'scripts/run_mwt_queue.py')==audit_contract['queue_sha256'], 'Audited queue source changed'
        queue=load(work/'reports/production_status.json')
        assert queue['status']=='CALCULATIONS_COMPLETE' and not queue['running'] and not queue['pending'] and not queue['failed']
        assert sorted(queue['completed_SID'])==frozen_sids
        queue_identity=load(work/'reports/production_queue_identity.json')
        assert queue_identity['config_sha256']==config_hash
        assert queue_identity['runner_sha256']==audit_contract['production_runner_sha256']
        queue_timestamp_ns=pd.Timestamp(queue['updated_UTC']).value
    pressure=np.asarray(config['pressure_hpa'])*100.
    omega_pressure=np.asarray(config['omega_pressure_hpa'])*100.
    omega_indices=np.array([np.flatnonzero(pressure==pp)[0] for pp in omega_pressure])
    pressure700=int(np.flatnonzero(pressure==70000)[0])
    expected_units={
        'KE':'J kg-1','APE':'J kg-1',
        **{k:'W kg-1' for k in ['dKE_dt','dAPE_dt','transport_K_h','transport_A_h','Gamma_K_h','Gamma_A_h','W_PG','C_A_to_K','vertical_input_K_Wkg','vertical_input_A_Wkg','vertical_input_P_Wkg']},
        **{f'Q_{k}_p':'Pa m2 s-3' for k in 'KAP'},
        **{f'Q_{k}_{direction}':'m3 s-3' for k in 'KAP' for direction in ['lon','lat']},
        **{f'vertical_input_{k}_Wm2':'W m-2' for k in 'KAP'}}
    checked_raw={};all_summary=[];all_coverage=[];all_reference=[];all_hydrostatic=[]
    for sid in requested:
        record={'sid':sid,'status':'CHECKING','checks':[],'nodes':[], 'output_identities':{},'input_identities':[]}
        checks=record['checks']
        def exact(name,condition):
            checks.append({'name':name,'pass':bool(condition)})
            if not condition:raise AssertionError(name)
        def close(name,actual,expected,atol=1e-11,rtol=1e-10):
            actual=np.asarray(actual);expected=np.asarray(expected)
            same=actual.shape==expected.shape and np.array_equal(np.isfinite(actual),np.isfinite(expected))
            delta=np.abs(actual.astype(float)-expected.astype(float)) if actual.shape==expected.shape else np.array([np.inf])
            maximum=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.
            passed=same and np.allclose(actual,expected,atol=atol,rtol=rtol,equal_nan=True)
            checks.append({'name':name,'pass':bool(passed),'max_abs_difference':maximum,'same_finite_support':bool(same),'atol':atol,'rtol':rtol})
            if not passed:raise AssertionError(name)
        try:
            folder=work/'cache'/sid;receipt_path=folder/'completed.json'
            receipt=load(receipt_path)
            record['receipt_sha256']=sha(receipt_path)
            exact('receipt_completed',receipt['status']=='COMPLETE_DIRECT_DIAGNOSTICS' and receipt['sid']==sid)
            exact('config_identity',receipt['config_sha256']==config_hash)
            exact('operator_identity',receipt['operator_sha256']==operator_hash)
            exact('frozen_source_identity',receipt['small_frozen_sources_sha256']==source_hashes)
            exact('scale_settings',(receipt['N'],receipt['j0'],receipt['j1'],receipt['shift_steps'])==(1024,5,8,64))
            exact('no_continuity_reconstruction',not receipt['continuity_estimate_enabled'] and not config['continuity_constrained_Gamma'] and not config['vertical_omega_reconstruction'])
            exact('shared_reference_contract',receipt['main_and_shift_reference_identical'] and receipt['fixed_mask_uses_union1088'])
            for name,digest in receipt['code_sha256'].items():
                path=work/('source_package' if name=='MWT.py' else 'scripts')/name
                if name=='run_mwt_sid.py' and sid==audit_contract['pilot_sid']:
                    path=work/audit_contract['pilot_runner_archive']
                exact('code_identity_'+name,sha(path)==digest)
            ev=events[events.sid.eq(sid)].sort_values('event_class')
            expected_events=[{'event_class':r.event_class,'event_id':r.event_id,'t0':r.t0} for r in ev.itertuples(index=False)]
            exact('frozen_event_pair',receipt['events']==expected_events and set(ev.event_class)=={'internal','final'})
            for name,digest in receipt['outputs_sha256'].items():
                path=folder/name;stat=path.stat()
                identity={'sha256_expected':digest,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
                if reuse_queue_hashes:
                    exact('queue_checked_'+name,sid in queue['completed_SID'] and stat.st_mtime_ns<=queue_timestamp_ns and receipt_path.stat().st_mtime_ns<=queue_timestamp_ns)
                    identity['hash_check']='reused audited terminal queue validation; not rehashed in this pass'
                else:
                    actual_hash=sha(path);exact('output_hash_'+name,actual_hash==digest)
                    identity.update(sha256_actual=actual_hash,hash_check='independent streamed SHA-256')
                record['output_identities'][name]=identity
            expected_inputs={(field,int(p)) for field in ['u','v','t','z'] for p in config['pressure_hpa']}|{('omega',int(p)) for p in config['omega_pressure_hpa']}|{('sp',None)}
            exact('all_expected_raw_inputs',{(r['field'],r['pressure_hpa']) for r in receipt['raw_inputs']}==expected_inputs)
            for raw in receipt['raw_inputs']:
                path=Path(raw['path']);stat=path.stat()
                exact('raw_stat_'+raw['field']+'_'+str(raw['pressure_hpa']),stat.st_size==raw['bytes'] and stat.st_mtime_ns==raw['mtime_ns'])
                if str(path) not in checked_raw:
                    with h5py.File(path,'r') as source:
                        unit=source[raw['variable']].attrs['units']
                        unit=unit.decode() if isinstance(unit,bytes) else str(unit)
                        tn='time' if 'time' in source else 'valid_time'
                        checked_raw[str(path)]={'units':unit,'time_units':source[tn].attrs['units'],'time':source[tn][:], 'longitude':source['longitude' if 'longitude' in source else 'lon'][:]}
                exact('raw_units_'+raw['field']+'_'+str(raw['pressure_hpa']),checked_raw[str(path)]['units']==raw['units'])
                record['input_identities'].append({k:raw[k] for k in ['field','pressure_hpa','path','bytes','mtime_ns','units','time_start_index']})
            with h5py.File(folder/'diagnostics.h5','r') as h:
                exact('HDF_identity',h.attrs['sid']==sid and h.attrs['config_sha256']==config_hash and h.attrs['MWT_operator_sha256']==operator_hash)
                exact('HDF_recorded_code',json.loads(h.attrs['code_sha256_json'])==receipt['code_sha256'])
                exact('HDF_full_config',json.loads(h.attrs['config_json'])==config)
                exact('HDF_pressure_time_units',h.attrs['pressure_unit']=='Pa' and h.attrs['dt_seconds']==21600)
                close('pressure_levels',h['coordinates/pressure_Pa'][:],pressure,0,0)
                close('omega_levels',h['coordinates/omega_pressure_Pa'][:],omega_pressure,0,0)
                lat=h['coordinates/latitude'][:];lon=h['coordinates/longitude'][:]
                close('latitude_spacing',np.diff(lat),np.full(len(lat)-1,.25),0,0)
                raw_coordinate_source=receipt['raw_inputs'][0]['path']
                native_longitude=checked_raw[raw_coordinate_source]['longitude']
                exact('all_raw_longitudes_identical',all(np.array_equal(checked_raw[r['path']]['longitude'],native_longitude) for r in receipt['raw_inputs']))
                longitude_check=check_cached_longitude(lon,native_longitude,receipt['domain'])
                record['longitude_grid_identity']=longitude_check
                close('longitude_spacing',np.diff(longitude_check['quarter_degree_indices']),np.ones(len(lon)-1,dtype=int),0,0)
                exact('longitude_source_identity',longitude_check['source_reconstruction_exact'] and longitude_check['native_modulo_identity_exact'])
                utc=h['coordinates/raw_UTC_seconds'][:];selected=h['coordinates/selected_union_indices'][:]
                times=pd.to_datetime(ev.t0,utc=True);mid=(times.min()+(times.max()-times.min())/2).floor('6h')
                expected_utc=mid.value//10**9-512*21600+np.arange(1088)*21600
                close('raw_UTC_identity',utc,expected_utc,0,0)
                close('selected_UTC_identity',h['coordinates/selected_UTC_seconds'][:],utc[selected],0,0)
                event_lags=np.arange(-30,31,6);close('event_derivative_lags',h['coordinates/event_lags_hours'][:],event_lags,0,0)
                selected_expected=[]
                for row in ev.itertuples(index=False):
                    seconds=pd.Timestamp(row.t0).value//10**9+event_lags*3600
                    idx=np.searchsorted(utc,seconds);close('exact_event_UTC_'+row.event_class,utc[idx],seconds,0,0);selected_expected.extend(idx)
                close('selected_time_union',selected,np.unique(selected_expected),0,0)
                for raw in receipt['raw_inputs']:
                    info=checked_raw[raw['path']];unit=info['time_units'];unit=unit.decode() if isinstance(unit,bytes) else str(unit)
                    measure,origin=unit.split(' since ');factor={'seconds':1,'hours':3600,'days':86400}[measure]
                    tstart=raw['time_start_index'];source_seconds=np.datetime64(origin.replace(' ','T'),'s').astype(np.int64)+info['time'][tstart:tstart+1088]*factor
                    close('source_UTC_'+raw['field']+'_'+str(raw['pressure_hpa']),source_seconds,utc,0,0)
                exact('two_fixed_schemes',set(h['coefficients'])=={'main','shift16d'} and set(h['diagnostics'])=={'main','shift16d'})
                for scheme in ['main','shift16d']:
                    for key,ds in h['coefficients/'+scheme].items():
                        levels=3 if key.startswith('omega') else 13
                        exact('coefficient_dimensions_'+scheme+'_'+key,ds.shape==(3,len(selected),levels,len(lat),len(lon)))
                    for event in ['internal','final']:
                        group=h['diagnostics/'+scheme+'/'+event]
                        exact('complete_diagnostics_'+scheme+'_'+event,set(group)==set(expected_units))
                        for key,ds in group.items():
                            shape=(3,11,len(lat),len(lon)) if key.startswith('vertical_input_') else (3,11,3 if key in ['C_A_to_K','Q_K_p','Q_A_p','Q_P_p'] else 13,len(lat),len(lon))
                            exact('diagnostic_dimensions_'+scheme+'_'+event+'_'+key,ds.shape==shape)
                tb=h['reference/temperature_mean'][:];phi_ref=h['reference/geopotential_reference'][:]
                close('shared_reference_temperature',h['basic_state/temperature_mean'][:],tb,0,0)
                close('shared_reference_geopotential',h['basic_state/geopotential_reference'][:],phi_ref,0,0)
                c=h['basic_state/c'][:];reference_coverage=h['reference/reference_weight_fraction'][:]
                lapse=9.8*pressure/(287.058*tb)*np.gradient(tb,pressure,edge_order=2)
                stability=9.8/1005.-lapse
                expected_c=np.where(np.isfinite(tb)&(tb>0)&np.isfinite(stability)&(stability>0),9.8/(tb*stability),np.nan)
                close('13_level_APE_coefficient',c,expected_c,1e-11,1e-10)
                for i,p in enumerate(pressure):
                    all_reference.append(dict(sid=sid,pressure_hpa=p/100,reference_T_K=tb[i],reference_coverage=reference_coverage[i],c=c[i],stable_finite=bool(np.isfinite(c[i]) and c[i]>0)))
                fixed=h['support/fixed_1088_pressure_grid'][:];wf=h['support/fixed_1088_omega_grid'][:]
                exact('support_omega_subset',np.all(~wf|fixed[omega_indices]))
                close('basic_state_fixed_mask',h['basic_state/valid_mask'][:],fixed,0,0)
                close('basic_state_omega_mask',h['basic_state/omega_valid_mask'][:],wf,0,0)
                # Nodes depend only on geometry/support, never energy values.
                ny,nx=len(lat),len(lon)
                nodes={('geometric_quarter',ny//4,nx//4),('geometric_centre',ny//2,nx//2),('geometric_three_quarter',3*ny//4,3*nx//4)}
                good=np.all(wf,axis=0);where=np.argwhere(good)
                if len(where):
                    nearest=where[np.argmin(np.sum((where-np.array([ny//2,nx//2]))**2,axis=1))]
                    nodes.add(('nearest_all_three_level_valid',int(nearest[0]),int(nearest[1])))
                # Add a fixed invalid column per level, when one exists, to detect zero filling.
                for i in range(len(pressure)):
                    bad=np.argwhere(~fixed[i])
                    if len(bad):
                        y,x=bad[0];nodes.add(('first_invalid_level_'+str(int(pressure[i]/100)),int(y),int(x)))
                sample_selected=np.unique([0,len(selected)//2,len(selected)-1])
                for label,y,x in sorted(nodes):
                    record['nodes'].append(dict(label=label,y=y,x=x,latitude=float(lat[y]),longitude=float(lon[x]),valid_pressure=fixed[:,y,x].tolist(),valid_omega=wf[:,y,x].tolist()))
                    for scheme in ['main','shift16d']:
                        cs=h['coefficients/'+scheme]
                        for key in cs:
                            values=cs[key][:,sample_selected,:,y,x]
                            if key.startswith('omega'):
                                mask=wf[:,y,x].copy()
                                if key=='omega_t':mask &= np.isfinite(tb[omega_indices])
                            else:
                                mask=fixed[:,y,x].copy()
                                if key in ['t','u_t','v_t']:mask &= np.isfinite(tb)
                                if key=='phi':mask &= np.isfinite(phi_ref)
                            exact('coefficient_mask_'+scheme+'_'+label+'_'+key,np.array_equal(np.isfinite(values),np.broadcast_to(mask,values.shape)))
                        for row in ev.itertuples(index=False):
                            dg=h['diagnostics/'+scheme+'/'+row.event_class]
                            exact('event_identity_'+scheme+'_'+row.event_class,dg.attrs['event_id']==row.event_id and dg.attrs['t0']==row.t0)
                            for lag in [-18,0]:
                                ti=int(np.flatnonzero(event_lags==lag)[0]);sec=pd.Timestamp(row.t0).value//10**9+lag*3600
                                ci=int(np.flatnonzero(utc[selected]==sec)[0])
                                co={key:cs[key][:,ci,:,y,x] for key in ['u','v','t','phi','omega','omega_u','omega_v','omega_t']}
                                C=-287.058/omega_pressure[None,:]*co['t'][:,omega_indices]*co['omega']
                                qk=.5*(co['u'][:,omega_indices]*co['omega_u']+co['v'][:,omega_indices]*co['omega_v'])
                                qa=.5*c[omega_indices][None,:]*co['t'][:,omega_indices]*co['omega_t']
                                qp=co['omega']*co['phi'][:,omega_indices]
                                boundary_node=y in [0,ny-1] or x in [0,nx-1]
                                if boundary_node:
                                    for a in [C,qk,qa,qp]:a[:]=np.nan
                                prefix=scheme+'_'+row.event_class+'_'+label+'_'+str(lag)
                                close('C_'+prefix,dg['C_A_to_K'][:,ti,:,y,x],C,1e-14,1e-10)
                                for key,a in [('K',qk),('A',qa),('P',qp)]:
                                    close('Q'+key+'_'+prefix,dg['Q_'+key+'_p'][:,ti,:,y,x],a,1e-11,1e-10)
                                    close('boundary_Wm2_'+key+'_'+prefix,dg['vertical_input_'+key+'_Wm2'][:,ti,y,x],(a[:,0]-a[:,2])/9.8,1e-11,1e-10)
                                    close('boundary_Wkg_'+key+'_'+prefix,dg['vertical_input_'+key+'_Wkg'][:,ti,y,x],(a[:,0]-a[:,2])/35000.,1e-14,1e-10)
                frame=pd.read_parquet(folder/'regional_metrics.parquet')
                exact('regional_unique_rows',not frame.duplicated(['scheme','sid','event_class','metric','window','level','region','lag_h']).any())
                exact('regional_SID',set(frame.sid)=={sid})
                exact('regional_scheme',set(frame.scheme)=={'main','shift16d'})
                exact('regional_units',set(frame.metric)==set(expected_units) and all(set(g.units)=={expected_units[k]} for k,g in frame.groupby('metric')))
                exact('regional_label_grid',set(frame.window)=={0,1,2} and set(frame.lag_h)==set(config['event_lags_hours']))
                exact('regional_area_range',frame.valid_area_fraction.dropna().between(0,1+1e-12).all())
                for metric,g in frame.groupby('metric'):
                    expected_levels={'500-850'} if metric.startswith('vertical_input_') else set(str(int(p/100)) for p in (omega_pressure if metric in ['C_A_to_K','Q_K_p','Q_A_p','Q_P_p'] else pressure))
                    exact('regional_level_identity_'+metric,set(g.level.astype(str))==expected_levels)
                    exact('regional_complete_grid_'+metric,len(g)==len(expected_levels)*3*9*3*2*2)
                exact('no_unsupported_finite_mean',not ((~np.isfinite(frame.valid_area_fraction)|(frame.valid_area_fraction<.8))&np.isfinite(frame.value)).any())
                # Report coverage only: no means or paired energy effects are inspected here.
                columns=['scheme','event_class','metric','window','level','region']
                for keys,g in frame.groupby(columns,sort=True):
                    pre=g[g.lag_h.isin(config['primary_event_window_hours'])]
                    all_coverage.append(dict(sid=sid,**dict(zip(columns,keys)),n_pre_valid=int(np.isfinite(pre.value).sum()),pre_valid_lags=','.join(str(int(x)) for x in pre.loc[np.isfinite(pre.value),'lag_h']),n_all_valid=int(np.isfinite(g.value).sum()),minimum_area_fraction=float(g.valid_area_fraction.min()),maximum_area_fraction=float(g.valid_area_fraction.max())))
                br=np.radians(lat);lr=np.radians(lon)
                be=np.r_[br[0]-(br[1]-br[0])/2,(br[:-1]+br[1:])/2,br[-1]+(br[-1]-br[-2])/2]
                le=np.r_[lr[0]-(lr[1]-lr[0])/2,(lr[:-1]+lr[1:])/2,lr[-1]+(lr[-1]-lr[-2])/2]
                area=6371000.**2*np.abs(np.diff(np.sin(be)))[:,None]*np.abs(np.diff(le))[None,:]
                for row in ev.itertuples(index=False):
                    cc=centres[(centres.event_id==row.event_id)&(centres.lag_h==-18)]
                    exact('fixed_lag_centre_identity_'+row.event_class,len(cc)==1)
                    centre=cc.iloc[0]
                    available=bool(centre.actual_track_point and not centre.position_extrapolated and np.isfinite([centre.lat,centre.lon]).all())
                    if available:
                        phi0,lam0=np.radians([centre.lat,centre.lon])
                        a=np.sin((br[:,None]-phi0)/2)**2+np.cos(phi0)*np.cos(br[:,None])*np.sin((lr[None,:]-lam0)/2)**2
                        distance=2*6371000*np.arctan2(np.sqrt(np.clip(a,0,1)),np.sqrt(np.clip(1-a,0,1)))
                        region=distance<=300000
                    else:region=np.zeros(area.shape,bool)
                    for scheme in ['main','shift16d']:
                        for metric in config['primary_estimands']:
                            ds=h['diagnostics/'+scheme+'/'+row.event_class+'/'+metric]
                            ti=2 # -18 h in derivative segment [-30,...,+30].
                            layer='500-850' if ds.ndim==4 else '700'
                            z=ds[1,ti] if ds.ndim==4 else ds[1,ti,1 if ds.shape[2]==3 else pressure700]
                            valid=region&np.isfinite(z);va=np.sum(area[valid]);ga=np.sum(area[region]);support=va/ga if ga else np.nan
                            value=np.sum(z[valid]*area[valid])/va if va and support>=.8 else np.nan
                            saved=frame[(frame.scheme==scheme)&(frame.event_class==row.event_class)&(frame.metric==metric)&(frame.window==1)&(frame.level.astype(str)==layer)&(frame.region=='r0_300')&(frame.lag_h==-18)]
                            exact('unique_selected_summary_'+scheme+'_'+row.event_class+'_'+metric,len(saved)==1)
                            saved=saved.iloc[0];prefix=scheme+'_'+row.event_class+'_'+metric
                            close('regional_mean_'+prefix,saved.value,value,1e-10,1e-10)
                            close('regional_area_fraction_'+prefix,saved.valid_area_fraction,support,1e-13,1e-12)
                            close('regional_valid_cells_'+prefix,saved.n_valid_cells,int(valid.sum()),0,0)
                hydro=pd.read_csv(folder/'hydrostatic_availability.csv')
                exact('hydrostatic_coverage_identity',len(hydro)==26 and set(hydro.sid)=={sid} and set(hydro.event_class)=={'internal','final'} and set(hydro.pressure_hpa)==set(config['pressure_hpa']))
                exact('hydrostatic_units',set(hydro.units)=={'m2 s-2 Pa-1'})
                exact('hydrostatic_values_plausible',hydro.finite_fraction.between(0,1).all() and (hydro.mean_absolute_hydrostatic_residual.dropna()>=0).all())
                all_hydrostatic.extend(hydro.to_dict(orient='records'))
            record['status']='VALIDATED_STATED_SCOPE'
        except Exception as error:
            record['status']='FAILED';record['error']=repr(error)
        record['checked_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        record['scope']='Cache identity, fixed supported numerical expressions, NaN masks and selected area summaries; not independent physical closure or validation of event-group mechanisms.'
        (output/(sid+'.json')).write_text(json.dumps(record,indent=2,allow_nan=True),encoding='utf-8')
        summary={'sid':sid,'status':record['status'],'checks':len(checks),'failed_checks':sum(not c['pass'] for c in checks),'max_absolute_formula_difference':max([c.get('max_abs_difference',0.) for c in checks],default=0.),'error':record.get('error','')}
        all_summary.append(summary)
        print(json.dumps(summary),flush=True)
    pd.DataFrame(all_summary).to_csv(output/'SID_validation_summary.csv',index=False)
    coverage=pd.DataFrame(all_coverage)
    coverage.to_csv(output/'coverage_by_SID_metric.csv',index=False)
    paired_coverage=[]
    if len(coverage):
        for keys,g in coverage.groupby(['scheme','metric','window','level','region'],sort=True):
            count=0;hist={str(i):0 for i in range(5)}
            for sid,ss in g.groupby('sid'):
                sets=[set(str(v).split(','))-{''} for v in ss.pre_valid_lags]
                common=sets[0]&sets[1] if len(sets)==2 else set()
                hist[str(len(common))]+=1;count+=len(common)>=3
            paired_coverage.append(dict(zip(['scheme','metric','window','level','region'],keys))|{'eligible_paired_SID':count,'common_lag_count_histogram':json.dumps(hist)})
    pd.DataFrame(paired_coverage).to_csv(output/'paired_coverage_counts.csv',index=False)
    pd.DataFrame(all_reference).to_csv(output/'reference13_level_coverage.csv',index=False)
    pd.DataFrame(all_hydrostatic).to_csv(output/'hydrostatic_saved_diagnostics.csv',index=False)
    final={'requested_SID':len(requested),'frozen_candidate_SID':len(frozen_sids),
           'complete_population_check':requested==frozen_sids,
           'validated':sum(x['status']=='VALIDATED_STATED_SCOPE' for x in all_summary),
           'failed':sum(x['status']=='FAILED' for x in all_summary),
           'output_hash_mode':'reuse audited final queue checks' if reuse_queue_hashes else 'independent streamed SHA-256',
           'config_sha256':config_hash,'operator_sha256':operator_hash,
           'script_sha256':sha(Path(__file__)),
           'scope':'No group energy effects or significance used. This is not a physical-budget closure assessment.'}
    (output/'summary.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    return final


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--sid',action='append',help='Technical subset only; omitting checks the full frozen 117.')
    parser.add_argument('--reuse-queue-hashes',action='store_true')
    parser.add_argument('--output-dir', help='Separate audit output directory; original default remains unchanged')
    args=parser.parse_args()
    report=validate_mwt_all_cache(args.work,args.sid,args.reuse_queue_hashes,args.output_dir)
    print(json.dumps(report,indent=2))
    raise SystemExit(1 if report['failed'] else 0)
