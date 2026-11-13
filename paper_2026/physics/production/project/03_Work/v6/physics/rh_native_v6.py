from process_v6 import *
from functools import lru_cache
R=6371008.8
NATIVE_SHA=hashlib.sha256((V/'src/native_fields_v5.py').read_bytes()).hexdigest()
def geometry(row):
 obj=dict(UTC=pd.Timestamp(row.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),latitude=float(row.lat),longitude=float(row.lon)%360,radius_km=800.,halo=1,reader_sha=NATIVE_SHA)
 return V/'cache/native_fields'/hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()[:20]
def loadz(p):
 data=Path(p).read_bytes()
 with np.load(io.BytesIO(data),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def checked(p):
 data=Path(p).read_bytes();meta=json.loads(Path(p).with_suffix('.json').read_text());assert hashlib.sha256(data).hexdigest()==meta['npz_sha256']
 with np.load(io.BytesIO(data),allow_pickle=False) as z:
  assert json.loads(str(z['binding_json'].item()))==meta['binding'];return z['values'].copy()
def rh(q,t,p):
 eps=287.0597/461.5250;alpha=np.clip((t-250.16)/(273.16-250.16),0,1)**2
 es=alpha*611.21*np.exp(17.502*(t-273.16)/(t-32.19))+(1-alpha)*611.21*np.exp(22.587*(t-273.16)/(t+.7))
 return 100*p*q/(eps+(1-eps)*q)/es
def shapley(q0,t0,q1,t1,p):
 f00=rh(q0,t0,p);f11=rh(q1,t1,p);f10=rh(q1,t0,p);f01=rh(q0,t1,p)
 cq=.5*((f10-f00)+(f11-f01));ct=.5*((f01-f00)+(f11-f10));return f11-f00,cq,ct
def tests():
 q=np.array([.001,.005,.02]);t=np.array([245.,265.,295.]);d,cq,ct=shapley(q,t,q*1.1,t+2,70000.)
 assert np.max(np.abs(d-cq-ct))<1e-12
 assert np.all(rh(q,t,70000)>0) and np.all(rh(q,t+1,70000)<rh(q,t,70000))
 zero=shapley(q,t,q,t,70000);assert all(np.all(x==0) for x in zero)
 assert np.allclose(rh(q,t,70000)*2,rh(q,t,140000))
 return {'reconstruction_max':float(np.max(np.abs(d-cq-ct))),'identity_zero':True,'pressure_units':'Pa','pressure_contribution_fixed_level':0.0}
def means(a,area,mask):
 g=mask&np.isfinite(a);fraction=area[g].sum()/area[mask].sum() if area[mask].sum() else 0
 return float(np.sum(a[g]*area[g])/area[g].sum()) if fraction>=.9 else np.nan,fraction
def endpoints(row):
 f=geometry(row);g=loadz(f/'sp.npz');g['folder']=str(f)
 for lev in [500,700]:
  for key in ['q','t','r']:g[key+str(lev)]=checked(f/(key+str(lev)+'.npz'))
 return g
def pair(a,b,lev,inner,outer):
 mask0=(a['distance']>=inner)&(a['distance']<outer);mask1=(b['distance']>=inner)&(b['distance']<outer)
 q0,t0,r0=(a[k+str(lev)] for k in ['q','t','r']);q1,t1,r1=(b[k+str(lev)] for k in ['q','t','r']);rhc0=rh(q0,t0,lev*100);rhc1=rh(q1,t1,lev*100)
 full0,frac0=means(rhc0,a['area'],mask0);full1,frac1=means(rhc1,b['area'],mask1);arc0,_=means(r0,a['area'],mask0);arc1,_=means(r1,b['area'],mask1)
 commonlat,i0,i1=np.intersect1d(a['latitude_indices'],b['latitude_indices'],return_indices=True);commonlon,j0,j1=np.intersect1d(a['longitude_indices'],b['longitude_indices'],return_indices=True);ix0=np.ix_(i0,j0);ix1=np.ix_(i1,j1)
 area=a['area'][ix0];assert np.allclose(area,b['area'][ix1]);domain=mask0[ix0]&mask1[ix1];valid=domain&np.isfinite(q0[ix0]+q1[ix1]+t0[ix0]+t1[ix1]);ar=area[domain].sum();overlap0=ar/a['area'][mask0].sum();overlap1=ar/b['area'][mask1].sum()
 base=dict(native_RH_change=full1-full0,archived_RH_change=arc1-arc0,start_RH_calculated=full0,end_RH_calculated=full1,start_RH_archived=arc0,end_RH_archived=arc1,start_valid_fraction=frac0,end_valid_fraction=frac1,overlap_fraction_start=overlap0,overlap_fraction_end=overlap1,overlap_cells=int(domain.sum()),joint_valid_cells=int(valid.sum()))
 d,cq,ct=shapley(q0[ix0],t0[ix0],q1[ix1],t1[ix1],lev*100);err=float(np.nanmax(np.abs(d-cq-ct))) if np.isfinite(d).any() else np.nan
 base['reconstruction_max_pp']=err
 # Local-field estimand is common geographic cells, separate from full along-track mean change.
 for key,arr in [('local_RH_change',d),('q_contribution',cq),('T_contribution',ct)]:base[key]=means(arr,area,domain)[0] if domain.sum()>=10 and min(overlap0,overlap1)>=.25 else np.nan
 base['p_contribution']=0. if np.isfinite(base['local_RH_change']) else np.nan
 base['sampling_support_remainder']=base['native_RH_change']-base['local_RH_change']
 base['rh_archive_difference_change']=base['archived_RH_change']-base['native_RH_change']
 return base
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);args=ap.parse_args();js(O/'RH_physics_unit_checks.json',tests())
 x=pd.read_csv(V/'p45_ready_r1/environment_ready.csv',low_memory=False);x=x[x.module.eq('P4')&x.lag_h.isin([-24,0,24])&np.isfinite(x.lat)&np.isfinite(x.lon)].copy();rows=[];missing=[];t0=time.perf_counter();groups=list(x.groupby(['sid','event_class'],sort=True));groups=groups[:args.limit or None]
 for i,((sid,event),g) in enumerate(groups):
  gg={int(r.lag_h):r for r in g.itertuples()};loaded={}
  for lo,hi in [(-24,0),(0,24)]:
   if not all(t in gg for t in [lo,hi]):continue
   try:
    for lag in [lo,hi]:
     if lag not in loaded:loaded[lag]=endpoints(gg[lag])
   except (FileNotFoundError,KeyError) as e:missing.append(dict(sid=sid,event=event,window=f'{lo}_{hi}',missing=str(e)));continue
   for lev in [500,700]:
    for inner,outer in [(100,300),(300,800),(200,800)]:
     rows.append(dict(sid=sid,boundary=event,window=f'{lo}_{hi}',level_hpa=lev,region=f'{inner}_{outer}km',units='pp',start_UTC=gg[lo].UTC,end_UTC=gg[hi].UTC,**pair(loaded[lo],loaded[hi],lev,inner,outer)))
  if i%50==0:
   pd.DataFrame(rows).to_csv(O/('RH_native_pilot.csv' if args.limit else 'RH_native_SID_sources.csv'),index=False);js(O/'RH_progress.json',{'done':i+1,'groups':len(groups),'rows':len(rows),'missing':len(missing),'seconds':time.perf_counter()-t0});print(i+1,len(groups),len(rows),len(missing),round(time.perf_counter()-t0),flush=True)
 r=pd.DataFrame(rows);r.to_csv(O/('RH_native_pilot.csv' if args.limit else 'RH_native_SID_sources.csv'),index=False);pd.DataFrame(missing).to_csv(O/'RH_missing_cache.csv',index=False)
 if not args.limit:
  fields=['native_RH_change','archived_RH_change','q_contribution','T_contribution','p_contribution','local_RH_change','sampling_support_remainder','rh_archive_difference_change']
  rr=r.melt(id_vars=['sid','boundary','window','level_hpa','region'],value_vars=fields,var_name='component',value_name='value');tab,piv,d,w=summary_table(rr.to_dict('records'),['boundary','window','level_hpa','region','component']);tab.to_csv(O/'RH_component_estimates.csv',index=False);np.savez_compressed(O/'RH_component_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)
 js(O/('RH_pilot_receipt.json' if args.limit else 'RH_receipt.json'),{'status':'COMPLETE','groups':len(groups),'rows':len(r),'missing':missing,'max_shapley_reconstruction_pp':float(r.reconstruction_max_pp.max()),'seconds':time.perf_counter()-t0,'new_raw_reads':0,'within_D_note':'-24h D_to_E availability requires long-enough observed preboundary history; denominator is not all2368','local_domain':'intersection native geographic annuli >=25percent area in both; >=90percent joint q/T coverage; at least10 cells; remainder includes sampling/support, not a sole advection estimate'})
if __name__=='__main__':main()
