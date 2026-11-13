"""Run one frozen SID's direct MWT diagnostics, checkpointed independently.

This orchestrator imports explicit independent calculation functions. It does not
choose samples or scales from energetic outcomes, and never opens source files writable.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='4'
import argparse,datetime,gc,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,h5py,psutil
from read_era5_fixed_block import read_era5_fixed_block
from theory_MWT_Prepare_Direct import MWT_Prepare_Direct
from theory_MWT_Apply_Operator import MWT_Apply_Operator
from theory_MWT_Atmosphere_Direct import MWT_Atmosphere_Direct


def shared_reference_profiles(temperature, geopotential, surface_pressure, pressure_pa,
                              latitude, longitude, primary_samples=1024,
                              stripe_rows=16, gas_constant=287.058):
    """Area/time reference on valid instantaneous above-ground observations.

    Inputs (time,pressure,latitude,longitude), sp(time,latitude,longitude).
    Return 13-level profiles and actual reference support; no project/global inputs.
    Stable MWT column eligibility is deliberately not used for this reference mean.
    """
    import numpy as np
    p=np.asarray(pressure_pa,dtype=float);br=np.radians(latitude);lr=np.unwrap(np.radians(longitude))
    def edges(x):return np.r_[x[0]-(x[1]-x[0])/2,(x[1:]+x[:-1])/2,x[-1]+(x[-1]-x[-2])/2]
    area=np.abs(np.diff(np.sin(np.clip(edges(br),-np.pi/2,np.pi/2))))[:,None]*np.abs(np.diff(edges(lr)))[None,:]
    numerator=np.zeros((2,len(p)));denominator=np.zeros_like(numerator);count=np.zeros(len(p),np.int64)
    for j in range(0,len(br),int(stripe_rows)):
        stop=min(len(br),j+int(stripe_rows));a=area[j:stop]
        sp=np.asarray(surface_pressure[:primary_samples,j:stop,:])
        above=np.isfinite(sp[:,None])&(p[None,:,None,None]<sp[:,None])
        for k,x in enumerate((temperature,geopotential)):
            xx=np.asarray(x[:primary_samples,:,j:stop,:]);valid=above&np.isfinite(xx)
            weight=a[None,None]*valid
            numerator[k]+=np.sum(np.where(valid,xx,0.)*weight,axis=(0,2,3))
            denominator[k]+=np.sum(weight,axis=(0,2,3))
            if k==0:count+=valid.sum(axis=(0,2,3))
    prof=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    tbar,phi_mean=prof;valid=np.isfinite(tbar)&np.isfinite(phi_mean)
    if not valid.any():raise ValueError('No valid common reference anchor')
    anchor=int(np.flatnonzero(valid)[np.argmax(p[valid])]);phi_ref=np.full(len(p),np.nan);phi_ref[anchor]=phi_mean[anchor]
    for step in (-1,1):
        k=anchor+step
        while 0<=k<len(p):
            previous=k-step
            if not valid[k] or not np.isfinite(phi_ref[previous]):break
            phi_ref[k]=phi_ref[previous]-gas_constant*.5*(tbar[k]+tbar[previous])*np.log(p[k]/p[previous])
            k+=step
    return {'temperature_mean':tbar,'geopotential_reference':phi_ref,
            'reference_weight_fraction':denominator[0]/(primary_samples*area.sum()),
            'reference_valid_sample_count':count,'area_weights':area}


def summarize_neighborhoods(diagnostics, pressure_hpa, omega_pressure_hpa, latitude,
                             longitude, centres, event_lags, regions_km,
                             minimum_valid_area_fraction=.8, earth_radius_m=6371000.):
    """Actual-grid area means with explicit support, no fitting or resampling.

    diagnostics axes: (scale,time,pressure,y,x) or (scale,time,y,x).
    centres maps requested lag to (latitude,longitude), absent/invalid centres to None.
    Each output row retains n/grid area; below-threshold values stay NaN.
    """
    import numpy as np
    lat=np.asarray(latitude);lon=np.asarray(longitude);br=np.radians(lat);lr=np.unwrap(np.radians(lon))
    def edges(x):return np.r_[x[0]-(x[1]-x[0])/2,(x[1:]+x[:-1])/2,x[-1]+(x[-1]-x[-2])/2]
    area=earth_radius_m**2*np.abs(np.diff(np.sin(np.clip(edges(br),-np.pi/2,np.pi/2))))[:,None]*np.abs(np.diff(edges(lr)))[None,:]
    fields={};units={}
    for metric,x in diagnostics.items():
        if x.ndim==4:
            xx=x[:,:,None];levels=['500-850'];levnum=[np.nan]
        else:
            xx=x;lp=omega_pressure_hpa if x.shape[2]==len(omega_pressure_hpa) else pressure_hpa
            if x.shape[2]!=len(lp):raise ValueError('Unidentified pressure axis')
            levels=[str(int(p)) for p in lp];levnum=list(lp)
        if metric in ('KE','APE'):unit='J kg-1'
        elif metric.endswith('_Wm2'):unit='W m-2'
        elif metric in ('Q_K_p','Q_A_p','Q_P_p'):unit='Pa m2 s-3'
        elif metric.endswith(('_lon','_lat')):unit='m3 s-3'
        else:unit='W kg-1'
        fields[metric]=(xx,levels,levnum,unit)
    rows=[]
    for ti,lag in enumerate(event_lags):
        centre=centres.get(int(lag))
        if centre is None or not np.isfinite(centre).all():
            distance=None
        else:
            pc,lc=np.radians(centre);dp=br[:,None]-pc;dl=(lr[None,:]-lc+np.pi)%(2*np.pi)-np.pi
            distance=2*earth_radius_m*np.arcsin(np.sqrt(np.clip(np.sin(dp/2)**2+np.cos(pc)*np.cos(br[:,None])*np.sin(dl/2)**2,0,1)))
        for zone,(inner,outer) in regions_km.items():
            mask=np.zeros(area.shape,bool) if distance is None else ((distance>=float(inner)*1000)&(distance<=float(outer)*1000))
            weights=np.where(mask,area,0.);geometric=float(weights.sum())
            for metric,(x,levels,levnum,unit) in fields.items():
                xx=x[:,ti];valid=np.isfinite(xx)&mask[None,None]
                valid_area=np.sum(valid*weights,axis=(-2,-1));support=valid_area/geometric if geometric>0 else np.full(valid_area.shape,np.nan)
                num=np.sum(np.where(valid,xx,0.)*weights,axis=(-2,-1))
                value=np.divide(num,valid_area,out=np.full_like(num,np.nan),where=(valid_area>0)&(support>=minimum_valid_area_fraction))
                n=valid.sum(axis=(-2,-1))
                for wi in range(3):
                    for pi,lev in enumerate(levels):
                        rows.append({'metric':metric,'window':wi,'level':lev,'pressure_hpa':float(levnum[pi]),'zone':zone,'inner_radius_km':inner,'outer_radius_km':outer,'lag_h':int(lag),'value':float(value[wi,pi]),'valid_area_fraction':float(support[wi,pi]),'valid_area_m2':float(valid_area[wi,pi]),'geometric_area_m2':geometric,'n_valid_cells':int(n[wi,pi]),'n_geometric_cells':int(mask.sum()),'units':unit,'position_available':centre is not None})
    return rows


def sid_fixed_domain(centres, radius_km=800., native_step=.25, halo_cells=2):
    """One rectangle enclosing observed centres and true spherical radius circles."""
    import numpy as np
    lat=np.asarray(centres['lat'],dtype=float);lon=np.asarray(centres['lon'],dtype=float)
    if not len(lat) or not np.isfinite(lat).all() or not np.isfinite(lon).all():raise ValueError('Invalid domain centres')
    reference=float(lon[0]);unwrapped=reference+(lon-reference+180)%360-180
    angular=radius_km*1000/6371000.;padding=halo_cells*native_step
    dl=np.degrees(np.arcsin(np.clip(np.sin(angular)/np.cos(np.radians(lat)),-1,1)))
    bounds_lat=(max(-89.75,np.floor((lat.min()-np.degrees(angular))/native_step)*native_step-padding),min(89.75,np.ceil((lat.max()+np.degrees(angular))/native_step)*native_step+padding))
    bounds_lon=(np.floor(np.min(unwrapped-dl)/native_step)*native_step-padding,np.ceil(np.max(unwrapped+dl)/native_step)*native_step+padding)
    return {'lat_bounds':bounds_lat,'lon_bounds':bounds_lon,'lon_reference':reference}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work',default=str(Path(__file__).resolve().parents[1]));parser.add_argument('--sid');parser.add_argument('--stripe-rows',type=int,default=16);parser.add_argument('--technical-pilot',action='store_true');args=parser.parse_args()
    W=Path(args.work);cfgpath=W/'config/analysis_contract.json';contract=json.loads(cfgpath.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cfgsha=sha(cfgpath);sources={p.name:sha(p) for p in (W/'source_data').glob('*.csv')}
    scripts=[Path(__file__),W/'scripts/read_era5_fixed_block.py',W/'scripts/theory_MWT_Prepare_Direct.py',W/'scripts/theory_MWT_Apply_Operator.py',W/'scripts/theory_MWT_Atmosphere_Direct.py',W/'source_package/MWT.py']
    codehash={p.name:sha(p) for p in scripts}
    e=pd.read_csv(W/'source_data/events_frozen.csv');sid=args.sid or sorted(e.sid.unique())[0];ev=e[e.sid.eq(sid)].sort_values('event_class')
    if len(ev)!=2 or set(ev.event_class)!={'internal','final'}:raise ValueError('Frozen pair identity invalid')
    c=pd.read_csv(W/'source_data/centres_frozen.csv');c=c[c.sid.eq(sid)];valid=np.isfinite(c.lat)&np.isfinite(c.lon)&c.actual_track_point&~c.position_extrapolated
    dm=c[valid&c.lag_h.between(-24,24)];domain=sid_fixed_domain(dm)
    folder=W/'cache'/sid;folder.mkdir(parents=True,exist_ok=True);completed=folder/'completed.json'
    if completed.exists():
        receipt=json.loads(completed.read_text(encoding='utf-8'))
        if receipt['config_sha256']==cfgsha and receipt['code_sha256']==codehash:
            if all(sha(folder/k)==v for k,v in receipt['outputs_sha256'].items()):print(json.dumps({'status':'REUSE_COMPLETE','sid':sid}));return
        raise RuntimeError('Existing completed SID differs from current code/config; do not overwrite')
    process=psutil.Process();lock=folder/'running.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf-8'))
        try:
            p=psutil.Process(old['pid'])
            if p.create_time()==old['create_time']:raise RuntimeError('SID worker already running')
        except psutil.NoSuchProcess:pass
    lock.write_text(json.dumps({'pid':os.getpid(),'create_time':process.create_time(),'sid':sid,'config_sha256':cfgsha}),encoding='utf8')
    log=folder/'run.jsonl';start=time.perf_counter();peak=0.
    def status(step,**kw):
        nonlocal peak
        rss=process.memory_info().rss;peak=max(peak,rss);rec={'sid':sid,'step':step,'seconds':time.perf_counter()-start,'rss_bytes':rss,'peak_sampled_rss_bytes':peak,**kw}
        with log.open('a',encoding='utf8') as f:f.write(json.dumps(rec,default=str)+'\n')
        print(json.dumps(rec,default=str),flush=True)
    def put(group,key,value):
        x=np.asarray(value)
        if x.ndim and x.size>32:return group.create_dataset(key,data=x,compression='gzip',compression_opts=1,shuffle=True)
        return group.create_dataset(key,data=x)
    try:
        N=int(contract['MWT']['primary_N']);shift=64;total=N+shift;dt=int(contract['dt_seconds']);pp=np.asarray(contract['pressure_hpa'],float)*100;op=np.asarray(contract['omega_pressure_hpa'],float)*100
        assert N==1024 and contract['MWT']['j0']==5 and contract['MWT']['j1']==8 and not contract['continuity_constrained_Gamma']
        times=pd.to_datetime(ev.t0,utc=True);mid=times.min()+(times.max()-times.min())/2;mid=mid.floor('6h');begin=mid-pd.Timedelta(seconds=N//2*dt)
        inventory=json.loads((W/'reports/data_era5_inventory.json').read_text(encoding='utf8'));filemap={(r['field'],r['level_hpa']):r for r in inventory}
        raw={};common=None;identities=[]
        for field in ('sp','u','v','t','z','omega'):
            levels=[None] if field=='sp' else contract['omega_pressure_hpa'] if field=='omega' else contract['pressure_hpa']
            target=None
            for pi,lev in enumerate(levels):
                meta=filemap[(field,lev)];path=Path(meta['path']);st=path.stat()
                if st.st_size!=meta['size_bytes'] or st.st_mtime_ns!=meta['mtime_ns']:raise RuntimeError('Raw input identity changed '+str(path))
                r=read_era5_fixed_block(path,meta['variable'],begin.isoformat(),total,**domain)
                if common is None:common={k:r[k] for k in ('UTC','latitude','longitude')}
                else:
                    for k in common:
                        if not np.array_equal(common[k],r[k]):raise ValueError('Native coordinates/time differ')
                data=r['field']
                if field=='sp':target=data
                else:
                    if target is None:target=np.empty((total,len(levels),*data.shape[-2:]),np.float64)
                    target[:,pi]=data
                identities.append({'field':field,'pressure_hpa':lev,'path':str(path),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'variable':meta['variable'],'units':r['units'],'time_start_index':r['native_time_start_index']})
                del data,r
                status('read',field=field,pressure_hpa=lev)
            raw[field]=target;del target
        lat=common['latitude'];lon=common['longitude'];ny,nx=len(lat),len(lon)
        fixed=np.ones((len(pp),ny,nx),bool)
        for x in ('u','v','t','z'):fixed&=np.isfinite(raw[x]).all(axis=0)
        for pi,p in enumerate(pp):fixed[pi]&=(np.isfinite(raw['sp'])&(raw['sp']>p)).all(axis=0)
        oi=np.array([int(np.flatnonzero(pp==p)[0]) for p in op]);wf=fixed[oi]&np.isfinite(raw['omega']).all(axis=0)
        ref=shared_reference_profiles(raw['t'],raw['z'],raw['sp'],pp,lat,lon,N,args.stripe_rows)
        if not np.isfinite(ref['temperature_mean']).all():status('reference_missing_levels',pressure_hpa=(pp[~np.isfinite(ref['temperature_mean'])]/100).tolist())
        event_info=[];all_ix=[]
        event_lags=np.arange(-30,31,6);used_lags=np.asarray(contract['event_lags_hours'],int)
        for row in ev.itertuples(index=False):
            t0=pd.Timestamp(row.t0);elapsed_seconds=(t0-begin).total_seconds()
            if not np.isfinite(elapsed_seconds) or elapsed_seconds % dt != 0:
                raise ValueError('Frozen departure is not exactly on the native six-hour UTC grid')
            index=int(elapsed_seconds/dt);ix=index+event_lags//6
            if ix.min()<shift or ix.max()>=N:raise ValueError('Event coefficient interval falls outside one input window')
            exact_t0=t0.tz_convert('UTC').tz_localize(None).to_datetime64().astype('datetime64[s]')
            if common['UTC'][index] != exact_t0:
                raise ValueError('Frozen departure/native UTC identity mismatch')
            if not np.array_equal(common['UTC'][ix],exact_t0+event_lags.astype('timedelta64[h]')):
                raise ValueError('Event diagnostic segment is not an exact continuous six-hour sequence')
            event_info.append({'event_class':row.event_class,'event_id':row.event_id,'t0':row.t0,'indices_union':ix});all_ix.extend(ix.tolist())
        selected=np.array(sorted(set(all_ix)),int);operator_path=W/'cache/fmwt_operator_nothing'
        operator_path=W/'cache'/f'mwt_operator_N{N}_j5_8_symmetric_physical.npy';operator=np.load(operator_path)
        if operator.shape!=(3,N,N):raise ValueError('Operator shape')
        partial=folder/'diagnostics.partial.h5';rows=[];gradient_report=[];basic_global=None
        with h5py.File(partial,'w') as h:
            h.attrs['sid']=sid;h.attrs['config_sha256']=cfgsha;h.attrs['MWT_operator_sha256']=sha(operator_path);h.attrs['source_mode']='actual native ERA5 fixed geographic grid; no omega reconstruction';h.attrs['pressure_unit']='Pa';h.attrs['dt_seconds']=dt
            h.attrs['config_json']=json.dumps(contract);h.attrs['input_identities_json']=json.dumps(identities);h.attrs['code_sha256_json']=json.dumps(codehash)
            coord=h.create_group('coordinates');put(coord,'latitude',lat);put(coord,'longitude',lon);put(coord,'pressure_Pa',pp);put(coord,'omega_pressure_Pa',op);put(coord,'raw_UTC_seconds',common['UTC'].astype('datetime64[s]').astype(np.int64));put(coord,'selected_union_indices',selected);put(coord,'selected_UTC_seconds',common['UTC'][selected].astype('datetime64[s]').astype(np.int64));put(coord,'event_lags_hours',event_lags)
            support=h.create_group('support');put(support,'fixed_1088_pressure_grid',fixed);put(support,'fixed_1088_omega_grid',wf)
            refg=h.create_group('reference')
            for k,v in ref.items():put(refg,k,v)
            gr=h.create_group('raw_event_vertical_gradients')
            for info in event_info:
                g=gr.create_group(info['event_class']);idx=info['indices_union'];put(g,'raw_UTC_seconds',common['UTC'][idx].astype('datetime64[s]').astype(np.int64))
                for field,name in [('u','du_dp'),('v','dv_dp'),('t','dtprime_dp'),('z','hydrostatic_residual')]:
                    x=raw[field][idx].copy()
                    if field=='t':x-=ref['temperature_mean'][None,:,None,None]
                    x=np.where(fixed[None],x,np.nan);deriv=np.gradient(x,pp,axis=1,edge_order=2)
                    if field=='z':deriv+=287.058*raw['t'][idx]/pp[None,:,None,None]
                    ds=put(g,name,deriv);ds.attrs['units']='m2 s-2 Pa-1' if field=='z' else 'K Pa-1' if field=='t' else 'm s-1 Pa-1'
                    if field=='z':
                        for li,p in enumerate(pp):
                            a=deriv[:,li];validd=np.isfinite(a);gradient_report.append({'sid':sid,'event_class':info['event_class'],'pressure_hpa':p/100,'finite_fraction':float(validd.mean()),'mean_absolute_hydrostatic_residual':float(np.mean(np.abs(a[validd]))) if validd.any() else None,'units':'m2 s-2 Pa-1'})
                    del x,deriv
            status('reference_and_masks',shape=[total,len(pp),ny,nx],selected_times=len(selected))
            for scheme,offset in [('main',0),('shift16d',shift)]:
                cs=h.create_group('coefficients/'+scheme);ix=selected-offset;coef={};basic_global=None
                for y0 in range(0,ny,args.stripe_rows):
                    y1=min(ny,y0+args.stripe_rows)
                    if y1-y0<3:
                        # A repeated overlap is harmless: same fixed temporal transform,
                        # overwrite the last up-to-two rows with bit-identical coefficients.
                        y0=max(0,ny-3);y1=ny
                    sl=slice(offset,offset+N);ys=slice(y0,y1)
                    data,basic=MWT_Prepare_Direct(raw['u'][sl,:,ys,:],raw['v'][sl,:,ys,:],raw['t'][sl,:,ys,:],lon,lat[ys],pp,
                        omega=raw['omega'][sl,:,ys,:],omega_pressure=op,geopotential=raw['z'][sl,:,ys,:],surface_pressure=raw['sp'][sl,ys,:],
                        temperature_mean=ref['temperature_mean'],geopotential_reference=ref['geopotential_reference'],valid_mask=fixed[:,ys,:],include_vertical_gradients=False)
                    # Omega support also uses the complete union, not separate N windows.
                    for key in ('omega','omega_u','omega_v','omega_t'):data[key][:,~wf[:,ys,:]]=np.nan
                    if basic_global is None:
                        basic_global={k:np.array(v,copy=True) for k,v in basic.items() if k not in ('valid_mask','omega_valid_mask','area_weights','data_shape')}
                        basic_global.update(lon=lon,lat=lat,valid_mask=fixed,omega_valid_mask=wf,area_weights=ref['area_weights'],data_shape=np.array([N,len(pp),ny,nx]),reference_weight_fraction=ref['reference_weight_fraction'],reference_valid_sample_count=ref['reference_valid_sample_count'])
                    for key in list(data):
                        x=MWT_Apply_Operator(data.pop(key),operator,time_indices=ix,block_size=1024)
                        if key not in coef:coef[key]=np.empty((3,len(selected),x.shape[2],ny,nx),np.float64)
                        coef[key][...,ys,:]=x;del x
                    del data,basic;gc.collect();status('MWT_stripe',scheme=scheme,y0=y0,y1=y1)
                for key,x in coef.items():put(cs,key,x)
                if scheme=='main':
                    bg=h.create_group('basic_state')
                    for key,x in basic_global.items():put(bg,key,x)
                for info in event_info:
                    event_idx=np.searchsorted(selected,info['indices_union']);assert np.array_equal(selected[event_idx],info['indices_union'])
                    event_coeff={k:v[:,event_idx] for k,v in coef.items()}
                    dg=MWT_Atmosphere_Direct(event_coeff,basic_global,dt=dt,gamma_form='conservative',mask_spatial_edges=True,return_flux_components=True)
                    gg=h.create_group('diagnostics/'+scheme+'/'+info['event_class']);gg.attrs['event_id']=info['event_id'];gg.attrs['t0']=info['t0']
                    for metric,x in dg.items():put(gg,metric,x)
                    event_centres={}
                    cc=c[c.event_id.eq(info['event_id'])]
                    for row in cc.itertuples(index=False):
                        good=np.isfinite(row.lat) and np.isfinite(row.lon) and row.actual_track_point and not row.position_extrapolated
                        event_centres[int(row.lag_h)]=(float(row.lat),float(row.lon)) if good else None
                    sel=np.array([int(np.flatnonzero(event_lags==lag)[0]) for lag in used_lags]);dd={k:v[:,sel] for k,v in dg.items()}
                    regional=summarize_neighborhoods(dd,pp/100,op/100,lat,lon,event_centres,used_lags,contract['regions_km'],contract['minimum_valid_area_fraction'])
                    for row in regional:
                        row.update(sid=sid,event_class=info['event_class'],event_id=info['event_id'],t0=info['t0'],UTC=(pd.Timestamp(info['t0'])+pd.Timedelta(hours=row['lag_h'])).isoformat(),scheme=scheme,region={'primary':'r0_300','environment':'r200_800','radius_sensitivity':'r0_500'}[row['zone']],window_label=contract['MWT']['window_labels'][row['window']])
                    rows.extend(regional);del regional,dd,dg,event_coeff;gc.collect();status('event_diagnostics',scheme=scheme,event_class=info['event_class'])
                del coef,basic_global;gc.collect();h.flush()
            h.attrs['technical_pilot']=args.technical_pilot
        os.replace(partial,folder/'diagnostics.h5')
        frame=pd.DataFrame(rows);tmp=folder/'regional_metrics.partial.parquet';frame.to_parquet(tmp,index=False);os.replace(tmp,folder/'regional_metrics.parquet')
        frame.to_csv(folder/'regional_metrics.csv',index=False)
        pd.DataFrame(gradient_report).to_csv(folder/'hydrostatic_availability.csv',index=False)
        del raw,rows,frame;gc.collect();status('data_saved')
        outputs={name:sha(folder/name) for name in ['diagnostics.h5','regional_metrics.parquet','regional_metrics.csv','hydrostatic_availability.csv']}
        receipt={'status':'COMPLETE_DIRECT_DIAGNOSTICS','sid':sid,'events':[{k:v for k,v in i.items() if k!='indices_union'} for i in event_info],'domain':domain,'shape_raw':[total,len(pp),ny,nx],'config_sha256':cfgsha,'code_sha256':codehash,'small_frozen_sources_sha256':sources,'raw_inputs':identities,'operator_sha256':sha(operator_path),'outputs_sha256':outputs,'elapsed_seconds':time.perf_counter()-start,'peak_sampled_rss_bytes':peak,'N':N,'j0':5,'j1':8,'shift_steps':shift,'raw_UTC_begin':str(common['UTC'][0]),'raw_UTC_end':str(common['UTC'][-1]),'main_and_shift_reference_identical':True,'fixed_mask_uses_union1088':True,'continuity_estimate_enabled':False,'energy_group_effects_reviewed':False,'technical_pilot':args.technical_pilot,'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        tmp=completed.with_suffix('.partial.json');tmp.write_text(json.dumps(receipt,indent=2,default=str),encoding='utf8');os.replace(tmp,completed);status('COMPLETE',output_bytes=sum((folder/k).stat().st_size for k in outputs))
    except Exception as exc:
        (folder/'failure.json').write_text(json.dumps({'sid':sid,'error':repr(exc),'traceback':traceback.format_exc(),'config_sha256':cfgsha,'code_sha256':codehash},indent=2),encoding='utf8');status('FAILED',error=repr(exc));raise
    finally:
        if lock.exists():
            owner=json.loads(lock.read_text(encoding='utf8'))
            if owner['pid']==os.getpid() and owner['create_time']==process.create_time():lock.unlink()

if __name__=='__main__':main()

