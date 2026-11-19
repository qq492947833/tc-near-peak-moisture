"""Recover the frozen primary 2–16-day omega*u and omega*v coefficients for the missing Figure 6e event requests."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import sys,time,json,hashlib,argparse,traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import h5py
PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE.parent/'scripts'))
from configuration import OUTPUT_ROOT, MWT_ROOT, ERA5_ROOT
R=Path(os.environ.get('FLTCCF_FIG6E_OUTPUT',OUTPUT_ROOT/'fig6e_rates')).resolve()
V=OUTPUT_ROOT
T=PACKAGE/'data'
W=MWT_ROOT
from read_era5_fixed_block import read_era5_fixed_block
pressures=np.array([40000,60000,92500])
keys=('omega_u','omega_v')
rtol,atol=2e-10,3e-9
def sha(data):
    return hashlib.sha256(data).hexdigest()
def save(p,d):
    q=p.with_suffix('.working');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');os.replace(q,p)
def readjson(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def disk(lat,lon,lat0,lon0):
    br,lr=np.deg2rad(lat)[:,None],np.unwrap(np.deg2rad(lon))[None,:]
    pc,lc=np.deg2rad([lat0,lon0])
    dp,dl=br-pc,(lr-lc+np.pi)%(2*np.pi)-np.pi
    dist=2*6371000*np.arcsin(np.sqrt(np.clip(np.sin(dp/2)**2+np.cos(pc)*np.cos(br)*np.sin(dl/2)**2,0,1)))/1000.
    return dist<=300
def widen(lo,hi,n):
    if hi-lo>=3:return lo,hi
    start=max(0,min(lo-1,n-3))
    return start,start+3
def process(sid):
    t0=time.monotonic()
    dest=R/'coefficients'/sid;dest.mkdir(exist_ok=True)
    file=dest/'rate_products.npz'; receipt=dest/'receipt.json'
    if receipt.exists():
        prev=readjson(receipt);assert prev['status']=='PASS' and sha(file.read_bytes())==prev['sha256']
        return prev
    m=pd.read_csv(T/'MISSING_NATIVE_PRODUCT_REQUESTS.csv',float_precision='round_trip')
    m=m[m.sid==sid].drop_duplicates(['sid','sec','lat','lon']).sort_values('sec')
    assert not m.sec.duplicated().any()
    old=pd.read_csv(T/'EVENT_HOUR_RATE_COVERAGE.csv',float_precision='round_trip')
    old=old[(old.sid==sid)&(old.status=='valid frozen joint rate')].drop_duplicates('sec')
    # One overlap per affected SID; nearest in time to any missing request, then earliest UTC.
    delay=np.abs(old.sec.to_numpy()[:,None]-m.sec.to_numpy()[None,:]).min(axis=1)
    witness=old.assign(delay=delay).sort_values(['delay','sec']).iloc[0]
    req=pd.concat([m[['sec','lat','lon']], witness.to_frame().T[['sec','lat','lon']]],ignore_index=True).sort_values('sec').reset_index(drop=True)
    sec=req.sec.to_numpy(dtype='int64');wt=int(witness.sec);iw=int(np.flatnonzero(sec==wt)[0])
    catalog=readjson(T/'native_input_identity.json')
    identity=catalog['sid_identity'][sid]
    operator=W/'cache/mwt_operator_N1024_j5_8_symmetric_physical.npy'
    assert catalog['operator_sha256']==sha(operator.read_bytes())
    with h5py.File(W/'addendum_20261023/cache'/sid/'increment.h5','r') as h:
        plan=json.loads(h.attrs['plan_json'])
        assert (plan['N'],plan['raw_N'],plan['j0'],plan['j1'],plan['dt_seconds'])==(1024,1088,5,8,21600)
        lat=h['coordinates/latitude'][:];lon=h['coordinates/longitude'][:]
        pp=h['coordinates/pressure_Pa'][:];sel=[int(np.flatnonzero(pp==q)[0]) for q in [40000,50000,60000,70000,85000,92500]]
        c=h['basic_state/c'][:][sel];tb=h['reference/temperature_mean'][:][sel];dc=h['basic_state/dc_dp'][:][sel]
        assert sha(np.r_[c,tb,dc].tobytes())==identity['reference_sha256']
        assert sha(lat.tobytes())==identity['latitude_sha256'] and sha(lon.tobytes())==identity['longitude_sha256']
        mask=h['support/fixed_1088_pressure_grid'][:][[int(np.flatnonzero(pp==p)[0]) for p in pressures]]
        assert mask.dtype==np.bool_
    assert len(np.unique(sec))==len(sec)
    start=int(pd.Timestamp(plan['raw_start_UTC']).value//10**9)
    idx=(sec-start)//21600
    assert np.array_equal(start+idx*21600,sec) and np.all((idx>=0)&(idx<1024))
    op=np.asarray(np.load(operator,mmap_mode='r')[1,idx,:],dtype=np.float64)
    geom=np.stack([disk(lat,lon,float(row.lat),float(row.lon)) for row in req.itertuples()])
    yy,xx=np.where(geom.any(axis=0));ymin,ymax=yy.min(),yy.max()+1;xmin,xmax=xx.min(),xx.max()+1
    geomsub=geom[:,ymin:ymax,xmin:xmax]
    vals={key:np.full((len(sec),3,ymax-ymin,xmax-xmin),np.nan) for key in keys}
    from_records={(v['field'],v['level']):v for v in catalog['raw_inputs']}
    source_ids=[];tiles_seen=[]
    for pi,p in enumerate(pressures):
        sources={f:ERA5_ROOT/from_records[(f,int(p))]['path'] for f in ('u','v','omega')}
        for f,path in sources.items():
            ref=from_records[f,int(p)]; st=path.stat()
            assert st.st_size==ref['size'] and st.st_mtime_ns==ref['mtime_ns'],(sid,f,p,'NC version changed')
            source_ids.append(dict(field=f,pressure_pa=int(p),filename=path.name,bytes=st.st_size,mtime_ns=st.st_mtime_ns))
        # Group disk cells by native HDF horizontal chunk. Only their row/column
        # bounding box is requested, not the complete cyclone rectangle.
        with h5py.File(sources['omega'],'r') as h:
            glat=h['latitude'][:];glon=h['longitude'][:]
            yyfile=np.asarray([np.flatnonzero(glat==v)[0] for v in lat])
            unwrap=plan['domain']['lon_reference']+(glon-plan['domain']['lon_reference']+180)%360-180
            xxfile=np.asarray([np.flatnonzero(unwrap==v)[0] for v in lon])
            ct,cy,cx=h['w'].chunks
        tilekey=np.column_stack([yyfile[yy]//cy,xxfile[xx]//cx])
        for tile in np.unique(tilekey,axis=0):
            target=(tilekey==tile).all(axis=1)
            iy,ix=yy[target],xx[target]
            y0,y1=widen(int(iy.min()),int(iy.max())+1,len(lat))
            x0,x1=widen(int(ix.min()),int(ix.max())+1,len(lon))
            target_local=(iy-y0,ix-x0)
            def read(f,variable):
                b=read_era5_fixed_block(str(sources[f]),variable,plan['raw_start_UTC'],1088,
                        lat_bounds=(lat[y0],lat[y1-1]),lon_bounds=(lon[x0],lon[x1-1]),lon_reference=plan['domain']['lon_reference'])
                assert np.array_equal(b['latitude'],lat[y0:y1]) and np.array_equal(b['longitude'],lon[x0:x1])
                assert b['units']==from_records[f,int(p)]['units']
                return b['field'][:,target_local[0],target_local[1]]
            raww=read('omega','w')
            mask_roi=mask[pi,iy,ix];keep=mask_roi & np.isfinite(raww).all(axis=0)
            spatial=np.flatnonzero(keep)
            for label,f in [('omega_u','u'),('omega_v','v')]:
                rawuv=read(f,f)
                assert np.isfinite(rawuv[:,mask_roi]).all(),(sid,p,f,'frozen mask inconsistent')
                rawproduct=raww[:1024,keep]*rawuv[:1024,keep]
                coef=op@rawproduct
                vals[label][:,pi,iy[keep]-ymin,ix[keep]-xmin]=coef
            tiles_seen.append([int(p),y0,y1,x0,x1,len(iy)])
        for label in keys:
            vals[label][:,pi]=np.where(geomsub,vals[label][:,pi],np.nan)
    qa=[]
    for pi,p in enumerate(pressures):
        cache=V/'diagnostics'/sid/f'omega_products_{p}.npz'
        assert sha(cache.read_bytes())==identity['omega_products'][cache.name]
        with np.load(cache) as z:
            it=int(np.flatnonzero(z['seconds']==wt)[0])
            for k in keys:
                selw=geomsub[iw]
                known=z[k][1,it,ymin:ymax,xmin:xmax][selw]
                new=vals[k][iw,pi][selw]
                good=np.isfinite(known);matches=bool(np.array_equal(good,np.isfinite(new)))
                err=float(np.max(abs(known[good]-new[good]))) if good.any() else None
                qa.append(dict(pressure_pa=int(p),field=k,finite=int(good.sum()),same_mask=matches,max_abs=err))
                assert good.any() and matches and np.allclose(known[good],new[good],rtol=rtol,atol=atol),(sid,wt,p,k,qa[-1])
    tmp=dest/'rate_products.working.npz'
    np.savez_compressed(tmp,seconds=sec,overlap_seconds=np.array([wt],dtype='int64'),latitude=lat[ymin:ymax],longitude=lon[xmin:xmax],
               pressure_Pa=pressures,geometry=geomsub,window=1,**vals)
    os.replace(tmp,file)
    rec=dict(sid=sid,status='PASS',sha256=sha(file.read_bytes()),n_missing_utc=len(m),n_output_utc=len(sec),overlap_seconds=[wt],
             coefficient_overlap=qa,rtol=rtol,atol=atol,window=1,raw_window=1024,stable_mask_length=1088,
             operator_sha256=catalog['operator_sha256'],reference_sha256=identity['reference_sha256'],original_latitude_sha256=identity['latitude_sha256'],
             original_longitude_sha256=identity['longitude_sha256'],raw_start_UTC=plan['raw_start_UTC'],fixed_tile_ranges=tiles_seen,raw_inputs=source_ids,
             elapsed_seconds=round(time.monotonic()-t0,2))
    save(receipt,rec)
    return rec
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    for folder in ['coefficients','checkpoints','tmp']:
        (R/folder).mkdir(exist_ok=True,parents=True)
    sids=sorted(pd.read_csv(T/'MISSING_NATIVE_PRODUCT_REQUESTS.csv').sid.unique())
    if args.limit:sids=sids[:args.limit]
    wall=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(process,s):s for s in sids}
        for f in as_completed(jobs):
            sid=jobs[f]
            try:
                rec=f.result()
            except BaseException:
                save(R/'coefficients'/sid/'failure.json',dict(sid=sid,traceback=traceback.format_exc()))
                for job in jobs:job.cancel()
                raise
            n=sum((R/'coefficients'/s/'receipt.json').is_file() for s in sids)
            save(R/'checkpoints'/'COEFFICIENTS.json',dict(verified=n,target=len(sids),last=sid,elapsed_s=round(time.monotonic()-wall,1)))
            print(n,len(sids),sid,rec['n_missing_utc'],rec['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
