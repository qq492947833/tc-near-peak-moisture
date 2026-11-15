"""Bounded, read-only extraction from accepted MWT coefficients and direct ERA5 omega.
All outputs are confined to the configured run directory.
"""
from pathlib import Path
import os
from configuration import OUTPUT_ROOT as ROOT, MWT_ROOT as W, TEMPLATE_ROOT, RISE_INPUT_ROOT, era5_path, initialize_output
for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME'):os.environ[k]=str(ROOT/'tmp')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,traceback,argparse,gc
import numpy as np,pandas as pd,h5py
from concurrent.futures import ProcessPoolExecutor,as_completed
A=W/'addendum_20261023'

from read_era5_fixed_block import read_era5_fixed_block
from theory_MWT_Atmosphere_Direct import MWT_Atmosphere_Direct
from vertical_terms import vertical_terms
P=np.array([400,500,600,700,850,925])*100.
IL=np.array([1,2,3,4]); REGIONS={'r0_300':(0,300),'r0_500':(0,500),'r200_800':(200,800)}
def save(p,x):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str,allow_nan=True),encoding='utf8');os.replace(q,p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def prepare_inputs():
    pre=pd.read_csv(TEMPLATE_ROOT/'source_data/centres_frozen.csv');pre=pre[pre.lag_h.isin([-24,-18,-12,-6])&pre.current117&pre.actual_track_point&~pre.position_extrapolated].copy()
    assert len(pre)==936 and pre.sid.nunique()==117
    repo=RISE_INPUT_ROOT
    role=pd.read_parquet(repo/'roles.parquet');rq=pd.read_parquet(repo/'requests.parquet')
    role=role[role.role.isin(['actual_rise','matched_rise','matched_Resolution'])].copy()
    rq=rq[rq.request_id.isin(role.request_id)].copy()
    pre.UTC=pd.to_datetime(pre.UTC,utc=True);rq.UTC=pd.to_datetime(rq.UTC,utc=True)
    rr=pd.concat([pre[['sid','UTC','lat','lon']],rq[['sid','UTC','lat','lon']]]).sort_values(['sid','UTC'])
    for key,g in rr.groupby(['sid','UTC']):
        assert g.lat.max()==g.lat.min() and np.ptp((g.lon.to_numpy()+180)%360)<1e-8,key
    rr=rr.drop_duplicates(['sid','UTC']);assert rr.sid.nunique()==123
    rr.to_parquet(ROOT/'inputs_manifest/requests.parquet',index=False)
    pre.to_parquet(ROOT/'inputs_manifest/pre_roles.parquet',index=False)
    role.to_parquet(ROOT/'inputs_manifest/rise_roles.parquet',index=False)
    rq.to_parquet(ROOT/'inputs_manifest/rise_requests.parquet',index=False)
    save(ROOT/'inputs_manifest/EXTRACTION_INPUTS.json',dict(n_sid=rr.sid.nunique(),n_requests=len(rr),pre_requests=len(pre),rise_requests=len(rq),selected_roles=role.role.value_counts().to_dict(),operator=str(W/'cache/mwt_operator_N1024_j5_8_symmetric_physical.npy'),operator_sha256=sha(W/'cache/mwt_operator_N1024_j5_8_symmetric_physical.npy')))
    return sorted(rr.sid.unique())
def get_coeff(h,name,sec,ps,lat,lon):
    """Read only exact cached time/grid/pressure intersections, retaining NaNs elsewhere."""
    tt=h['coordinates/selected_UTC_seconds'][:];hp=h['coordinates/omega_pressure_Pa' if name.startswith('omega') else 'coordinates/pressure_Pa'][:]
    hy=h['coordinates/latitude'][:];hx=h['coordinates/longitude'][:]
    out=np.full((3,len(sec),len(ps),len(lat),len(lon)),np.nan)
    yo=np.flatnonzero(np.isin(lat,hy));xo=np.flatnonzero(np.isin(lon,hx))
    if not len(yo) or not len(xo):return out
    iy=np.searchsorted(hy,lat[yo]);ix=np.searchsorted(hx,lon[xo]);assert np.all(np.diff(iy)==1) and np.all(np.diff(ix)==1)
    ds=h['coefficients/main/'+name]
    target_pi=[int(np.flatnonzero(hp==p)[0]) for p in ps if p in hp]
    if not target_pi:return out
    pmin,pmax=min(target_pi),max(target_pi)
    block=ds[:,:,pmin:pmax+1,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]
    for j,t in enumerate(sec):
        hit=np.flatnonzero(tt==t)
        if not len(hit):continue
        for k,p in enumerate(ps):
            pi=np.flatnonzero(hp==p)
            if len(pi):out[:,j,k,yo[0]:yo[-1]+1,xo[0]:xo[-1]+1]=block[:,int(hit[0]),int(pi[0])-pmin]
    return out
def transformed(raw,op):
    a=raw.reshape(raw.shape[0],-1);valid=np.isfinite(a).all(axis=0)
    out=np.full((op.shape[0],a.shape[1]),np.nan)
    out[:,valid]=op@a[:,valid]
    return out
def run_sid(sid,pilot=False,resummarize=False):
    t0=time.time();dest=ROOT/'diagnostics'/sid;dest.mkdir(exist_ok=True)
    prior=json.loads((dest/'COMPLETE.json').read_text()) if (dest/'COMPLETE.json').exists() else None
    if prior and not resummarize:return prior
    if resummarize and prior and prior.get('derivation_version')==2 and prior.get('mask_schema_version')==3:return prior
    if resummarize:
        assert (ROOT/'inputs_manifest/TASK_CACHE_BEFORE_DERIVATION_V2.json').is_file(),'Final derivation requires the completed cache-binding manifest'
        assert all((dest/f'omega_products_{p}.npz').exists() for p in (40000,60000,92500))
        if prior:
            assert prior['status']=='COMPLETE'
            if not (dest/'COMPLETE_EXTRACTION_V1.json').exists():save(dest/'COMPLETE_EXTRACTION_V1.json',prior)
    rq=pd.read_parquet(ROOT/'inputs_manifest/requests.parquet');rq=rq[rq.sid.eq(sid)].sort_values('UTC')
    sec=pd.to_datetime(rq.UTC,utc=True).astype('int64').to_numpy()//10**9
    source=A/'cache'/sid/'increment.h5';legacy=W/'cache'/sid/'diagnostics.h5'
    with h5py.File(source,'r') as h:
        plan=json.loads(h.attrs['plan_json']);lat=h['coordinates/latitude'][:];lon=h['coordinates/longitude'][:]
        # Keep the accepted geographic rectangle; pressure diagnostics use only their required levels.
        pp=h['coordinates/pressure_Pa'][:];pi=[int(np.flatnonzero(pp==p)[0]) for p in P]
        c=h['basic_state/c'][:][pi];tb=h['reference/temperature_mean'][:][pi];dc=h['basic_state/dc_dp'][:][pi]
        mask=h['support/fixed_1088_pressure_grid'][:][pi]
        co={n:get_coeff(h,n,sec,P,lat,lon) for n in ('u','v','t','phi','uu','uv','vv','u_t','v_t','omega','omega_u','omega_v','omega_t')}
        cached_times=h['coordinates/selected_UTC_seconds'][:]
    if resummarize:
        binding=next(x for x in json.loads((ROOT/'inputs_manifest/TASK_CACHE_BEFORE_DERIVATION_V2.json').read_text())['sids'] if x['sid']==sid)
        actual_identity=dict(operator_sha256=sha(W/'cache/mwt_operator_N1024_j5_8_symmetric_physical.npy'),reference_sha256=hashlib.sha256(np.r_[c,tb,dc].tobytes()).hexdigest(),latitude_sha256=hashlib.sha256(lat.tobytes()).hexdigest(),longitude_sha256=hashlib.sha256(lon.tobytes()).hexdigest(),seconds_sha256=hashlib.sha256(sec.tobytes()).hexdigest())
        assert actual_identity==binding['original_identity'],'Pre-derivation accepted-cache identity mismatch'
    missing=~np.isin(sec,cached_times)
    if missing.any():
        assert legacy.exists()
        with h5py.File(legacy,'r') as h:
            rawsec=h['coordinates/raw_UTC_seconds'][:]
            assert rawsec[0]==pd.Timestamp(plan['raw_start_UTC']).value//10**9
            cfg=json.loads(h.attrs['config_json']);assert cfg['MWT']['primary_N']==1024 and cfg['MWT']['j0']==5 and cfg['MWT']['j1']==8
            for key,ref in [('c',c),('dc_dp',dc)]:assert np.allclose(h['basic_state/'+key][:][pi],ref,equal_nan=True,rtol=0,atol=0)
            assert np.array_equal(h['reference/temperature_mean'][:][pi],tb,equal_nan=True)
            for n in co:co[n][:,missing]=get_coeff(h,n,sec[missing],P,lat,lon)
    begin=pd.Timestamp(plan['raw_start_UTC']).value//10**9;idx=(sec-begin)//21600
    assert np.all(sec==begin+idx*21600) and np.all((idx>=0)&(idx<1024))
    opall=np.load(W/'cache/mwt_operator_N1024_j5_8_symmetric_physical.npy',mmap_mode='r');op=np.asarray(opall[:,idx,:]).reshape(3*len(sec),1024)
    inventory=json.loads((W/'reports/data_era5_inventory.json').read_text());metas={(m['field'],m['level_hpa']):m for m in inventory}
    ident=[];qa=[]
    for k,p in enumerate(P):
        if p in (50000,70000,85000) and not pilot:continue
        omega_cache=dest/f'omega_products_{int(p)}.npz'
        if p not in (50000,70000,85000) and omega_cache.exists():
            manifest=ROOT/'inputs_manifest/TASK_CACHE_BEFORE_DERIVATION_V2.json'
            if resummarize:
                assert manifest.is_file()
                bound=next(x for x in json.loads(manifest.read_text())['sids'] if x['sid']==sid)
                expected=next(x for x in bound['cache'] if Path(x['path'])==omega_cache)
                assert sha(omega_cache)==expected['sha256'] and omega_cache.stat().st_size==expected['size']
            with np.load(omega_cache) as z:
                assert np.array_equal(z['seconds'],sec)
                for n in ('omega','omega_u','omega_v','omega_t'):co[n][:,:,k]=z[n]
            continue
        assert not resummarize,'Resummarization must not re-read raw ERA5'
        raws={}
        for f,var in [('u','u'),('v','v'),('t','t'),('omega','w')]:
            if f=='omega' and int(p/100) not in (500,700,850):path=era5_path(level=int(p/100))
            else:path=era5_path(metas[(f,int(p/100))]['path'])
            block=read_era5_fixed_block(str(path),var,plan['raw_start_UTC'],1088,lat_bounds=(lat[0],lat[-1]),lon_bounds=(lon[0],lon[-1]),lon_reference=plan['domain']['lon_reference'])
            assert np.array_equal(block['latitude'],lat) and np.array_equal(block['longitude'],lon)
            if f=='omega':assert block['units'] in ('Pa s**-1','Pa s-1','Pa s^-1'),block['units']
            st=path.stat();ident.append(dict(field=f,level=p,path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns,units=block['units']))
            raws[f]=block['field'];del block
        wm=mask[k]&np.isfinite(raws['omega']).all(axis=0)
        for f in ('u','v','t'):assert np.isfinite(raws[f][:,mask[k]]).all(),(sid,p,f)
        raws['t']=raws['t']-tb[k]
        w=np.where(wm[None],raws['omega'][:1024],np.nan)
        for n,f in [('omega',None),('omega_u','u'),('omega_v','v'),('omega_t','t')]:
            raw=w if f is None else w*raws[f][:1024]
            val=transformed(raw,op).reshape(3,len(sec),len(lat),len(lon))
            if p in (50000,70000,85000):
                old=co[n][:,:,k];ok=np.isfinite(old)&np.isfinite(val)
                err=float(np.max(np.abs(old[ok]-val[ok]))) if ok.any() else np.nan
                qa.append(dict(sid=sid,field=n,pressure_Pa=p,max_abs_error=err,n=int(ok.sum()),same_mask=bool(np.array_equal(np.isfinite(old),np.isfinite(val)))))
                assert np.allclose(old[ok],val[ok],rtol=2e-10,atol=3e-9),(sid,n,p,err)
            else:co[n][:,:,k]=val
        if p not in (50000,70000,85000):np.savez_compressed(omega_cache,seconds=sec,**{n:co[n][:,:,k] for n in ('omega','omega_u','omega_v','omega_t')})
        del raws,w,raw,val;gc.collect()
        save(dest/'progress.json',dict(sid=sid,completed_pressure=p,seconds=time.time()-t0))
    bs=dict(pressure=P,lon=lon,lat=lat,c=c,omega_pressure=P,omega_indices=np.arange(len(P)))
    orig=MWT_Atmosphere_Direct(co,bs,mask_spatial_edges=True,return_flux_components=False)
    vert=vertical_terms(co['u'],co['v'],co['t'],co['omega_u'],co['omega_v'],co['omega_t'],P,c,tb,dc_dp=dc)
    altvert=vertical_terms(co['u'],co['v'],co['t'],co['omega_u'],co['omega_v'],co['omega_t'],P,c,tb,dc_dp=None,
        templates={50000:[40000,50000,70000],60000:[40000,60000,85000],70000:[50000,70000,85000],85000:[60000,85000,92500]})
    frozenalt=vertical_terms(co['u'],co['v'],co['t'],co['omega_u'],co['omega_v'],co['omega_t'],P,c,tb,dc_dp=dc,
        templates={50000:[40000,50000,70000],60000:[40000,60000,85000],70000:[50000,70000,85000],85000:[60000,85000,92500]})
    # Canonical horizontal and vertical contributions share exact pointwise support.
    fields={n:orig[n] for n in ('KE','APE','C_A_to_K','Gamma_K_h','Gamma_A_h')}
    for kind in ('K','A'):
        h=orig['Gamma_'+kind+'_h'];v=vert['Gamma_'+kind+'_p'];valid=np.isfinite(h)&np.isfinite(v)
        fields['Gamma_'+kind+'_h']=np.where(valid,h,np.nan)
        fields['Gamma_'+kind+'_p']=np.where(valid,v,np.nan)
        fields['Gamma_'+kind+'_total']=np.where(valid,h+v,np.nan)
    fields['APE_ref_correction']=vert['S_A_cprime'];fields['APE_compressibility']=vert['S_A_thermodynamic']
    for v in fields.values():v[...,0,:]=np.nan;v[...,-1,:]=np.nan;v[...,:,0]=np.nan;v[...,:,-1]=np.nan
    def layer(v,inds=IL):return np.trapz(v[:,:,inds],P[inds],axis=2)/35000.
    arrays={}
    for n,v in fields.items():
        arrays[('primary','700',n)]=v[:,:,3]
        arrays[('primary','500-850',n)]=layer(v)
        common=np.isfinite(v[:,:,IL]).all(axis=2)
        arrays[('common_layer','700',n)]=np.where(common,v[:,:,3],np.nan)
        arrays[('sparse_layer','500-850',n)]=np.where(common,layer(v,np.array([1,3,4])),np.nan)
    for kind in ('K','A'):
        n='Gamma_'+kind+'_p';vb=vert[n+'_bracket'];valid=np.isfinite(fields[n])&np.isfinite(vb)
        alt=np.where(valid,vb,np.nan)
        for lev,calc in [('700',lambda v:v[:,:,3]),('500-850',layer)]:
            arrays[('bracket_derivative',lev,n)]=calc(alt)
            arrays[('bracket_derivative',lev,'Gamma_'+kind+'_total')]=calc(fields['Gamma_'+kind+'_h']+alt)
            arrays[('same_support_primary_bracket',lev,n)]=calc(np.where(valid,fields[n],np.nan))
            arrays[('same_support_primary_bracket',lev,'Gamma_'+kind+'_total')]=calc(np.where(valid,fields['Gamma_'+kind+'_total'],np.nan))
        va=altvert[n];valid=np.isfinite(fields[n])&np.isfinite(va)
        for lev,calc in [('700',lambda v:v[:,:,3]),('500-850',layer)]:
            arrays[('combined_derivative_cprime',lev,n)]=calc(np.where(valid,va,np.nan))
            arrays[('combined_derivative_cprime',lev,'Gamma_'+kind+'_total')]=calc(np.where(valid,fields['Gamma_'+kind+'_h']+va,np.nan))
            arrays[('alternate_derivative',lev,n)]=calc(np.where(valid,frozenalt[n],np.nan))
            arrays[('alternate_derivative',lev,'Gamma_'+kind+'_total')]=calc(np.where(valid,fields['Gamma_'+kind+'_h']+frozenalt[n],np.nan))
            arrays[('same_support_primary_alternate',lev,n)]=calc(np.where(valid,fields[n],np.nan))
            arrays[('same_support_primary_alternate',lev,'Gamma_'+kind+'_total')]=calc(np.where(valid,fields['Gamma_'+kind+'_total'],np.nan))
    # Old metrics on their own accepted masks; layer boundary flux remains the two-boundary quantity.
    for n in ('KE','APE','C_A_to_K','Gamma_K_h','Gamma_A_h'):arrays[('old_support','700',n)]=orig[n][:,:,3]
    common_rate=np.isfinite(co['omega'][:,:,IL]).all(axis=2)&np.isfinite(co['t'][:,:,IL]).all(axis=2)
    for kind in ('K','A','P'):
        n='vertical_input_'+kind+'_Wkg';v=orig[n]
        arrays[('old_support','500-850',n)]=v
        arrays[('primary','500-850',n)]=np.where(common_rate,v,np.nan)
    # Joint process comparisons use the identical four-level geographic column.
    # The stricter transfer comparison additionally requires pressure-derivative support.
    joint=np.logical_and.reduce([np.isfinite(fields[n][:,:,IL]).all(axis=2) for n in ('KE','APE','C_A_to_K')])
    joint &= np.logical_and.reduce([np.isfinite(orig['vertical_input_'+k+'_Wkg']) for k in ('K','A','P')])
    joint_trans=joint & np.logical_and.reduce([np.isfinite(fields[n][:,:,IL]).all(axis=2) for n in ('Gamma_K_h','Gamma_K_p','Gamma_A_h','Gamma_A_p')])
    for variant,jmask in [('joint_layer_processes',joint),('joint_with_transfer',joint_trans)]:
        for n,v in fields.items():
            if variant=='joint_layer_processes' and n.startswith('Gamma_'):continue
            arrays[(variant,'700',n)]=np.where(jmask,v[:,:,3],np.nan)
            arrays[(variant,'500-850',n)]=np.where(jmask,layer(v),np.nan)
        for k in ('K','A','P'):
            n='vertical_input_'+k+'_Wkg';arrays[(variant,'500-850',n)]=np.where(jmask,orig[n],np.nan)
    br=np.radians(lat);lr=np.unwrap(np.radians(lon))
    def edges(x):return np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
    area=6371000.**2*np.abs(np.diff(np.sin(np.clip(edges(br),-np.pi/2,np.pi/2))))[:,None]*np.abs(np.diff(edges(lr)))[None,:]
    rows=[]
    for ri,r in enumerate(rq.itertuples(index=False)):
        pc,lc=np.radians([r.lat,r.lon]);dp=br[:,None]-pc;dl=(lr[None,:]-lc+np.pi)%(2*np.pi)-np.pi
        dist=2*6371000.*np.arcsin(np.sqrt(np.clip(np.sin(dp/2)**2+np.cos(pc)*np.cos(br[:,None])*np.sin(dl/2)**2,0,1)))/1000.
        for region,(lo,hi) in REGIONS.items():
            m=(dist>=lo)&(dist<=hi);w=np.where(m,area,0);geom=w.sum()
            for (variant,level,metric),data in arrays.items():
                xx=data[:,ri];valid=np.isfinite(xx)&m[None];va=(valid*w).sum(axis=(-2,-1));coverage=va/geom
                vals=np.divide((np.where(valid,xx,0)*w).sum(axis=(-2,-1)),va,out=np.full(3,np.nan),where=(va>0)&(coverage>=.8))
                for win in range(3):rows.append(dict(sid=sid,UTC=str(r.UTC),window=win,region=region,level=level,variant=variant,metric=metric,value=vals[win],valid_area_fraction=coverage[win],n_valid_cells=int(valid[win].sum()),n_geometric_cells=int(m.sum()),valid_mask_sha256=hashlib.sha256(np.packbits(valid[win]).tobytes()).hexdigest(),geometric_mask_sha256=hashlib.sha256(np.packbits(m).tobytes()).hexdigest(),units='J kg-1' if metric in ('KE','APE') else 'W kg-1'))
    df=pd.DataFrame(rows);df.to_parquet(dest/'request_values.parquet',index=False)
    pd.DataFrame(qa).to_csv(dest/'overlap_coefficients_QA.csv',index=False)
    wide=df[df.variant.eq('primary')].pivot(index=['UTC','window','region','level'],columns='metric',values='value')
    totalerr=max(float((wide['Gamma_'+q+'_total']-wide['Gamma_'+q+'_h']-wide['Gamma_'+q+'_p']).abs().max()) for q in ('K','A'))
    assert totalerr<1e-10,(sid,totalerr)
    if resummarize:
        if prior:ident=prior['raw_inputs']
        else:
            meta=json.loads((ROOT/'inputs_manifest/NC_METADATA.json').read_text())
            needed=[(f,int(p/100)) for p in (40000,60000,92500) for f in ('u','v','t','omega')]
            ident=[]
            for f,p in needed:
                path=era5_path(level=p) if f=='omega' else era5_path(metas[(f,p)]['path'])
                entry=next(x for x in meta if Path(x['path'])==path)
                st=path.stat();assert st.st_size==entry['size'] and st.st_mtime_ns==entry['mtime_ns']
                ident.append(dict(field=f,level=p*100,path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns,metadata_source='preflight NC_METADATA.json; cached products completed before unused temporal-derivative failure'))
    cache_identity=dict(operator_sha256=sha(W/'cache/mwt_operator_N1024_j5_8_symmetric_physical.npy'),reference_sha256=hashlib.sha256(np.r_[c,tb,dc].tobytes()).hexdigest(),seconds_sha256=hashlib.sha256(sec.tobytes()).hexdigest(),latitude_sha256=hashlib.sha256(lat.tobytes()).hexdigest(),longitude_sha256=hashlib.sha256(lon.tobytes()).hexdigest(),omega_products={p.name:sha(p) for p in dest.glob('omega_products_*.npz')},raw_inputs=ident)
    save(dest/'CACHE_IDENTITY.json',cache_identity)
    rec=dict(sid=sid,status='COMPLETE',n_requests=len(rq),rows=len(df),seconds=time.time()-t0,source_cache=str(source),legacy_pre_times=int(missing.sum()),raw_inputs=ident,coefficient_qa=qa,total_identity_max_abs_error_Wkg=totalerr,output_sha256=sha(dest/'request_values.parquet'),derivation_version=2,mask_schema_version=3,code_sha256=sha(Path(__file__)),resummarized_from_task_cache=resummarize)
    save(dest/'COMPLETE.json',rec);return rec
def main():
    initialize_output()
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--pilot',action='store_true');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--resummarize',action='store_true');ar=ap.parse_args()
    sids=sorted(pd.read_parquet(ROOT/'inputs_manifest/requests.parquet').sid.unique()) if ar.resummarize else prepare_inputs()
    if ar.limit:sids=sids[:ar.limit]
    failures=[]
    with ProcessPoolExecutor(max_workers=ar.workers) as pool:
        todo={pool.submit(run_sid,s,ar.pilot,ar.resummarize):s for s in sids}
        for f in as_completed(todo):
            sid=todo[f]
            try:r=f.result();print(json.dumps({k:r[k] for k in ('sid','status','seconds','rows')}),flush=True)
            except Exception as e:
                r=dict(sid=sid,error=repr(e),traceback=traceback.format_exc());save(ROOT/'diagnostics'/sid/'FAILURE.json',r);failures.append(r);print(json.dumps(r),flush=True)
            if ar.resummarize:
                complete=0
                for s in sids:
                    receipt=ROOT/'diagnostics'/s/'COMPLETE.json'
                    if receipt.exists():
                        state=json.loads(receipt.read_text())
                        complete+=state.get('derivation_version')==2 and state.get('mask_schema_version')==3
            else:complete=sum((ROOT/'diagnostics'/s/'COMPLETE.json').exists() for s in sids)
            save(ROOT/('checkpoints/RESUMMARIZATION.json' if ar.resummarize else 'checkpoints/EXTRACTION.json'),dict(total=len(sids),complete=complete,failed=len(failures),last_sid=sid,failures=failures))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
