"""Recover regional KE rates for Fig. 6e using frozen MWT identities and masks.

Run with python -B. Modes: pilot, watch, once, final. All outputs are under this
configured rate-output directory. The production functions operate independently
on transform windows. Only the requested window is loaded; unused windows and
unrequested thermodynamic coefficients are NaN and never used in the KE terms.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, zipfile, traceback
from datetime import datetime, timezone
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE.parent/'scripts'))
from configuration import OUTPUT_ROOT, MWT_ROOT
ROOT = Path(os.environ.get('FLTCCF_FIG6E_OUTPUT',OUTPUT_ROOT/'fig6e_rates')).resolve()
OLD = OUTPUT_ROOT
COV = PACKAGE/'data'
BASE = MWT_ROOT
OUT = ROOT/'diagnostics'
import numpy as np
import pandas as pd
import h5py
from theory_MWT_Atmosphere_Direct import MWT_Atmosphere_Direct
from vertical_terms import vertical_terms
METRICS = ['Gamma_K_h','Gamma_K_p','Gamma_K_total']
P = 100.*np.array([400,500,600,700,850,925])
LAY = [1,2,3,4]
EX = [0,2,5]
QA = dict(rtol=1e-12,atol=1e-13)
def dump(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)
def atomic_csv(path, data):
    tmp=path.with_suffix('.tmp')
    data.to_csv(tmp,index=False,float_format='%.17g',lineterminator='\r\n')
    os.replace(tmp,path)
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order='C')).hexdigest()
def maskhash(a):
    return digest(np.packbits(a))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def edges(a):
    return np.r_[a[0]-(a[1]-a[0])/2,(a[:-1]+a[1:])/2,a[-1]+(a[-1]-a[-2])/2]
def geometry(lat,lon,latitude,longitude):
    br=np.radians(lat);lr=np.unwrap(np.radians(lon))
    pc,lc=np.radians([latitude,longitude]);dp=br[:,None]-pc;dl=(lr[None,:]-lc+np.pi)%(2*np.pi)-np.pi
    dist=2*6371000.*np.arcsin(np.sqrt(np.clip(np.sin(dp/2)**2+np.cos(pc)*np.cos(br[:,None])*np.sin(dl/2)**2,0,1)))/1000.
    m=(dist>=0)&(dist<=300)
    if not m.any():raise ValueError('No geometric cells')
    area=6371000.**2*np.abs(np.diff(np.sin(np.clip(edges(br),-np.pi/2,np.pi/2))))[:,None]*np.abs(np.diff(edges(lr)))[None,:]
    yy,xx=np.where(m)
    ys=slice(max(0,int(yy.min())-1),min(len(lat),int(yy.max())+2))
    xs=slice(max(0,int(xx.min())-1),min(len(lon),int(xx.max())+2))
    return m,area,ys,xs
def get_block(h,name,sec,lat,lon):
    """Mirror exact cache intersection, reading only the transform window and local halo."""
    orig=h['coordinates']
    hy=orig['latitude'][:];hx=orig['longitude'][:]
    ti=np.flatnonzero(orig['selected_UTC_seconds'][:]==sec)
    assert len(ti)==1
    hp=orig['omega_pressure_Pa' if name.startswith('omega') else 'pressure_Pa'][:]
    y=np.flatnonzero(np.isin(lat,hy));x=np.flatnonzero(np.isin(lon,hx))
    out=np.full((3,1,len(P),len(lat),len(lon)),np.nan)
    if not len(x) or not len(y):return out
    iy=np.searchsorted(hy,lat[y]);ix=np.searchsorted(hx,lon[x])
    assert np.all(np.diff(iy)==1) and np.all(np.diff(ix)==1) and np.all(np.diff(y)==1) and np.all(np.diff(x)==1)
    sub=h['coefficients/main/'+name][1,ti[0],:,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]
    for k,pressure in enumerate(P):
        hit=np.flatnonzero(hp==pressure)
        if len(hit):out[1,0,k,y[0]:y[-1]+1,x[0]:x[-1]+1]=sub[int(hit[0])]
    return out
def cached_old_omega(sid, sec, shape):
    out={}
    for k in EX:
        p=OLD/'diagnostics'/sid/('omega_products_%d.npz'%P[k])
        with np.load(p,allow_pickle=False) as metadata:
            seconds=metadata['seconds'];ind=np.flatnonzero(seconds==sec)
        assert len(ind)==1
        with zipfile.ZipFile(p) as z:
            for key in ['omega_u','omega_v']:
                with z.open(key+'.npy','r') as f:
                    ver=np.lib.format.read_magic(f)
                    arr_shape,fortran,dtype=np.lib.format._read_array_header(f,ver)
                    assert arr_shape==(3,len(seconds),*shape) and not fortran
                    count=int(np.prod(shape)); slab=count*dtype.itemsize
                    f.seek(f.tell()+(len(seconds)+int(ind[0]))*slab)
                    buf=f.read(slab);assert len(buf)==slab
                    out[key,k]=np.frombuffer(buf,dtype=dtype).reshape(shape)
    return out
def recovered_product(new,key,j,k,lat,lon,ys,xs):
    y=np.flatnonzero(np.isin(lat[ys],new['latitude']));x=np.flatnonzero(np.isin(lon[xs],new['longitude']))
    out=np.full((len(lat[ys]),len(lon[xs])),np.nan)
    if not len(x) or not len(y):return out
    iy=np.searchsorted(new['latitude'],lat[ys][y]);ix=np.searchsorted(new['longitude'],lon[xs][x])
    assert np.all(np.diff(iy)==1) and np.all(np.diff(ix)==1)
    sel=EX.index(k)
    out[np.ix_(y,x)]=new[key][j,sel,iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]
    return out
def run_rate(sid,sec,cy,cx,inc,source,new,old_products=None):
    lat=inc['coordinates/latitude'][:];lon=inc['coordinates/longitude'][:]
    m,area,ys,xs=geometry(lat,lon,cy,cx)
    lat_sub=lat[ys];lon_sub=lon[xs]
    pp=inc['coordinates/pressure_Pa'][:];pi=[int(np.flatnonzero(pp==v)[0]) for v in P]
    c=inc['basic_state/c'][:][pi]
    tref=inc['reference/temperature_mean'][:][pi]
    dc=inc['basic_state/dc_dp'][:][pi]
    co={n:get_block(source,n,sec,lat_sub,lon_sub) for n in ('u','v','uu','uv','vv','omega_u','omega_v')}
    unused=np.full_like(co['u'],np.nan)
    co.update({n:unused for n in ('t','u_t','v_t')})
    if new is not None:
        jj=np.flatnonzero(new['seconds']==sec);assert len(jj)==1
        yy=np.searchsorted(lat,new['latitude']);xx=np.searchsorted(lon,new['longitude'])
        original_geometry=np.zeros_like(m)
        original_geometry[np.ix_(yy,xx)]=new['geometry'][int(jj[0])]
        assert np.array_equal(original_geometry,m),('center/sample geometry mismatch',sid,sec)
        for k in EX:
            for key in ['omega_u','omega_v']:
                co[key][1,0,k]=recovered_product(new,key,int(jj[0]),k,lat,lon,ys,xs)
    else:
        for (key,k),value in old_products.items():
            co[key][1,0,k]=value[ys,xs]
    bs=dict(pressure=P,lon=lon_sub,lat=lat_sub,c=c)
    h=MWT_Atmosphere_Direct(co,bs,mask_spatial_edges=True,return_flux_components=False)['Gamma_K_h'][1,0]
    v=vertical_terms(co['u'],co['v'],co['t'],co['omega_u'],co['omega_v'],unused,P,c,tref,dc_dp=dc)['Gamma_K_p'][1,0]
    both=np.isfinite(h)&np.isfinite(v)
    h=np.where(both,h,np.nan);v=np.where(both,v,np.nan)
    for a in [h,v]:
        a[:,0,:]=np.nan;a[:,-1,:]=np.nan;a[:,:,0]=np.nan;a[:,:,-1]=np.nan
    w=np.where(m,area,0);geom=w.sum()
    result={}
    for key,a in zip(METRICS,[h,v,h+v]):
        layer=np.full(m.shape,np.nan);layer[ys,xs]=np.trapz(a[LAY],P[LAY],axis=0)/35000.
        valid=np.isfinite(layer)&m;va=(valid*w).sum();coverage=va/geom
        result[key]=float((np.where(valid,layer,0)*w).sum()/va) if va>0 and coverage>=.8 else np.nan
        if key==METRICS[0]:
            result.update(valid_area_fraction=float(coverage),valid_mask_sha256=maskhash(valid),geometric_mask_sha256=maskhash(m),n_valid_cells=int(valid.sum()),n_geometric_cells=int(m.sum()))
        else:assert result['valid_mask_sha256']==maskhash(valid)
    return result
def assert_overlap(a,b):
    for field in ['valid_mask_sha256','geometric_mask_sha256','n_valid_cells','n_geometric_cells']:assert a[field]==b[field],(field,a[field],b[field])
    np.testing.assert_allclose([a[n] for n in METRICS],[b[n] for n in METRICS],equal_nan=True,**QA)
    np.testing.assert_allclose(a['valid_area_fraction'],b['valid_area_fraction'],rtol=1e-14,atol=1e-14)
    return max((abs(a[n]-b[n]) for n in METRICS if np.isfinite(a[n])),default=0.)
def load_inputs():
    ledger=pd.read_csv(COV/'EVENT_HOUR_RATE_COVERAGE.csv',float_precision='round_trip')
    missing=pd.read_csv(COV/'MISSING_NATIVE_PRODUCT_REQUESTS.csv',float_precision='round_trip')
    assert list(ledger.groupby('relative_hour').size())==[132]*9
    assert len(missing)==376 and missing.sid.nunique()==117 and len(missing.drop_duplicates(['sid','sec']))==363
    q=pd.read_parquet(COV/'existing_request_values.parquet')
    q['sec']=pd.to_datetime(q.UTC,utc=True).astype('int64')//10**9
    keys=['sid','sec']
    assert not q.duplicated(keys+['metric']).any()
    w=q.pivot(index=keys,columns='metric',values='value')
    meta=['valid_area_fraction','valid_mask_sha256','geometric_mask_sha256','n_valid_cells','n_geometric_cells']
    assert q.groupby(keys)[meta].nunique(dropna=False).le(1).all().all()
    w=w.join(q.groupby(keys)[meta].first())
    centres=pd.read_parquet(COV/'existing_centres.parquet')
    w=w.join(centres.set_index(keys)[['lat','lon']],validate='one_to_one')
    saved=ledger[ledger[METRICS].notna().all(axis=1)]
    b=saved.set_index(keys).join(w,rsuffix='_parquet')
    for n in METRICS:assert np.array_equal(b[n].to_numpy(),b[n+'_parquet'].to_numpy())
    assert len(b)==810 and all(b['valid_mask_sha256']==b['valid_mask_sha256_parquet'])
    return ledger,missing,w
def process_sid(sid,missing,old,pilot=False):
    dest=ROOT/'coefficients'/sid
    incpath=BASE/'addendum_20261023/cache'/sid/'increment.h5'
    legacy=BASE/'cache'/sid/'diagnostics.h5'
    with h5py.File(incpath,'r') as inc:
        lat=inc['coordinates/latitude'][:];lon=inc['coordinates/longitude'][:];tt=inc['coordinates/selected_UTC_seconds'][:]
        if pilot:
            common=old.loc[sid].query('n_valid_cells>0').dropna(subset=METRICS)
            sec=int(common.index[0]); allsec=[sec]; overlap={sec}; new=None
            oldprod=cached_old_omega(sid,sec,(len(lat),len(lon)))
        else:
            with np.load(dest/'rate_products.npz',allow_pickle=False) as z:new={k:z[k] for k in ['seconds','latitude','longitude','pressure_Pa','omega_u','omega_v','overlap_seconds','geometry','window']}
            receipt=json.loads((dest/'receipt.json').read_text());assert receipt['status']=='PASS' and sha(dest/'rate_products.npz')==receipt['sha256'] and int(new['window'])==1
            assert np.array_equal(new['pressure_Pa'],P[EX]);assert set(new['latitude'])<=set(lat) and set(new['longitude'])<=set(lon)
            assert np.all(np.diff(new['latitude'])>0) and np.all(np.diff(new['longitude'])>0)
            allsec=new['seconds'].tolist();overlap=set(np.asarray(new['overlap_seconds']).ravel())
            required=set(missing.loc[missing.sid.eq(sid),'sec'])
            assert set(allsec)==required|overlap and len(allsec)==len(set(allsec)) and not required&overlap
            oldprod=None
        rr=[];check=[]
        unique=missing[missing.sid.eq(sid)].drop_duplicates('sec').set_index('sec')
        for sec in allsec:
            q=old.loc[(sid,sec)] if sec in overlap else unique.loc[sec]
            cy=float(q['lat']);cx=float(q['lon'])
            if sec in tt:
                row=run_rate(sid,sec,cy,cx,inc,inc,new,oldprod)
            else:
                with h5py.File(legacy,'r') as source:
                    for f in ['basic_state/c','reference/temperature_mean','basic_state/dc_dp']:
                        assert np.array_equal(inc[f][:],source[f][:],equal_nan=True)
                    row=run_rate(sid,sec,cy,cx,inc,source,new,oldprod)
            if sec in overlap:
                e=assert_overlap(row,q)
                check.append(dict(sid=sid,sec=int(sec),max_abs_error_Wkg=e,valid_mask_sha256=row['valid_mask_sha256'],n_valid_cells=row['n_valid_cells'],n_geometric_cells=row['n_geometric_cells']))
            else:rr.append(dict(sid=sid,sec=int(sec),lat=cy,lon=cx,**row))
    if pilot:return check
    staged=OUT/'per_sid';staged.mkdir(exist_ok=True)
    atomic_csv(staged/(sid+'.csv'),pd.DataFrame(rr))
    dump(staged/(sid+'.json'),dict(sid=sid,status='PASS',n_new=len(rr),overlaps=check,coefficient_file_sha256=sha(dest/'rate_products.npz')))
    return check
def finalize(ledger,missing,old):
    folder=OUT/'per_sid';sids=sorted(set(missing.sid))
    assert all((folder/(s+'.json')).exists() for s in sids)
    reports=[json.loads((folder/(s+'.json')).read_text()) for s in sids]
    assert all(x['status']=='PASS' and len(x['overlaps'])>=1 for x in reports)
    fresh=pd.concat([pd.read_csv(folder/(s+'.csv'),float_precision='round_trip') for s in sids],ignore_index=True).set_index(['sid','sec'])
    assert len(fresh)==363 and not fresh.index.has_duplicates
    r=ledger.copy();r['old_status']=r.status;r['valid_area_fraction']=np.nan
    for i,q in r.iterrows():
        k=(q.sid,q.sec)
        if q.status=='pressure-product UTC not cached':
            val=fresh.loc[k]
            assert np.isclose(q.lat,val.lat,atol=1e-12,rtol=0) and abs((q.lon-val.lon+180)%360-180)<1e-12
            for f in METRICS+['valid_area_fraction','valid_mask_sha256','geometric_mask_sha256','n_valid_cells','n_geometric_cells']:r.at[i,f]=val[f]
            r.at[i,'status']='recovered valid rate' if np.isfinite(val.Gamma_K_total) else 'recovered below 80% valid area'
        elif k in old.index and q.status!='no accepted center':
            r.at[i,'valid_area_fraction']=old.loc[k,'valid_area_fraction']
    valid=r[METRICS].notna().all(axis=1)
    assert np.array_equal(valid,r[METRICS].notna().any(axis=1))
    assert (r.loc[valid,'valid_area_fraction']>=.8).all() and r.loc[valid,'valid_mask_sha256'].notna().all()
    assert r.status.eq('no accepted center').sum()==1 and r.status.eq('cached but joint regional rates invalid').sum()==1
    np.testing.assert_allclose(r.loc[valid,METRICS[0]]+r.loc[valid,METRICS[1]],r.loc[valid,METRICS[2]],**QA)
    r['unit']='W kg-1';r['window']='2-16 d';r['layer']='500-850 hPa';r['region']='0-300 km';r['variant']='primary'
    atomic_csv(OUT/'FINAL_EVENT_HOUR_RATES.csv',r)
    reread=pd.read_csv(OUT/'FINAL_EVENT_HOUR_RATES.csv',float_precision='round_trip')
    np.testing.assert_array_equal(r[METRICS].to_numpy(),reread[METRICS].to_numpy())
    summary=dict(status='PASS',n_overlap_SIDs=len(reports),n_unique_recovered_times=len(fresh),event_hours=len(r),valid_event_hours=int(valid.sum()),status_counts=r.status.value_counts().to_dict(),max_abs_overlap_Wkg=max(c['max_abs_error_Wkg'] for f in reports for c in f['overlaps']),max_abs_additivity_Wkg=float((r.Gamma_K_total-r.Gamma_K_h-r.Gamma_K_p).abs().max()),rtol=QA['rtol'],atol_Wkg=QA['atol'],csv_sha256=sha(OUT/'FINAL_EVENT_HOUR_RATES.csv'))
    dump(OUT/'FINAL_EVENT_HOUR_QA.json',summary);print(summary,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','once','watch','final']);p.add_argument('--sid');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    ledger,missing,old=load_inputs();sids=sorted(set(missing.sid))
    if args.mode=='pilot':
        v=process_sid(args.sid or sids[0],missing,old,True)
        dump(OUT/'FROZEN_PILOT_OVERLAP.json',v);print(v,flush=True);return
    if args.mode=='final':finalize(ledger,missing,old);return
    target=ROOT/'coefficients'
    while True:
        pending=[s for s in sids if not (OUT/'per_sid'/(s+'.json')).is_file()]
        if not pending:finalize(ledger,missing,old);return
        for sid in pending:
            if not (target/sid/'receipt.json').is_file():continue
            try:checks=process_sid(sid,missing,old);print(sid,checks[0]['max_abs_error_Wkg'],flush=True)
            except Exception:
                (OUT/'diagnostic_failure.txt').write_text(sid+'\n'+traceback.format_exc())
                raise
        if args.mode=='once':break
        time.sleep(5)
if __name__=='__main__':main()
