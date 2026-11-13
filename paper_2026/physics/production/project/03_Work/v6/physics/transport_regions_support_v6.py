from process_v6 import *
from rh_native_v6 import geometry,loadz,checked,means
from functools import lru_cache
PRE=[-24,-18,-12,-6]
def p4():
 x=pd.read_csv(V/'p45_ready_r1/environment_ready.csv',low_memory=False);x=x[x.module=='P4'];rows=[]
 fields={'moving_tendency':'q_budget_common_moving_tendency_kgkg_s','horizontal_advection':'q_budget_common_horizontal_advection_kgkg_s','vertical_advection':'q_budget_common_vertical_advection_kgkg_s','residual':'q_budget_common_unresolved_residual_kgkg_s','q850_MFC':'q850_moisture_flux_convergence_gkg_day','convergence850':'low_level_convergence_1e5_s'}
 errs=[]
 for (sid,event),g in x.groupby(['sid','event_class']):
  g=g.set_index('lag_h').reindex(range(-24,25,6));cube=[]
  for field,col in fields.items():
   y=g[col].to_numpy(float,copy=True);sup=col+'__support_pass'
   if sup in g:y[~g[sup].fillna(False).astype(bool).to_numpy()]=np.nan
   if field in ['moving_tendency','horizontal_advection','vertical_advection','residual']:y*=86400000.
   cube.append(y)
  cube=np.array(cube);joint=np.isfinite(cube[:4]).all(0);cube[:4,~joint]=np.nan
  if joint.any():errs.append(float(np.max(abs(cube[0,joint]-cube[1,joint]-cube[2,joint]-cube[3,joint]))))
  for j,field in enumerate(fields):
   y=cube[j];units='1e-5 s-1' if field=='convergence850' else 'g kg-1 day-1'
   for k,lag in enumerate(range(-24,25,6)):rows.append(dict(sid=sid,boundary=event,field=field,estimand='lag_'+str(lag),units=units,value=y[k]))
   for name,sl in [('minus24_0',slice(0,5)),('0_24',slice(4,9))]:
    yy=y[sl];val=np.trapezoid(yy,dx=.25) if np.isfinite(yy).all() else np.nan
    rows.append(dict(sid=sid,boundary=event,field=field,estimand='integral_'+name,units='1e-5 s-1 day' if field=='convergence850' else 'g kg-1',value=val))
 r=pd.DataFrame(rows);r.to_csv(O/'P4_transport_SID_sources.csv',index=False);tab,piv,d,w=summary_table(rows,['boundary','field','estimand','units']);tab.to_csv(O/'P4_transport_estimates.csv',index=False);np.savez_compressed(O/'P4_transport_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)
 js(O/'P4_transport_receipt.json',{'max_budget_identity_error_gkg_day':max(errs),'source':'v5 common-support instantaneous scalars','new_raw_reads':0,'n_start_sid':2368,'integrals':'five six-hour instantaneous values, trapezoid; complete temporal support; 850 terms are separate'})
 print('P4 transport complete',len(tab),flush=True)
@lru_cache(maxsize=8)
def climate_tile(name):
 folder=V/'cache/climatology_1991_2020_v5_1'/name
 if not (folder/'complete.json').exists():raise FileNotFoundError(str(folder))
 return loadz(folder/'climatology.npz')
def climq(g,utc,key='q700'):
 t=pd.Timestamp(utc);group=(t.month-1)*4+t.hour//6;out=np.full(g['sp'].shape,np.nan);ilat=g['latitude_indices'];ilon=g['longitude_indices']
 for i in np.unique(ilat//90):
  for j in np.unique(ilon//180):
   z=climate_tile(f'{key}_y{i}_x{j}');a=np.flatnonzero(ilat//90==i);b=np.flatnonzero(ilon//180==j);k=np.flatnonzero(z['groups']==group)
   if len(k)!=1:continue
   out[np.ix_(a,b)]=z['mean'][k[0]][np.ix_(ilat[a]-i*90,ilon[b]-j*180)]
 return out
def regions(limit=0):
 x=pd.read_csv(V/'results/P1_P2_20261009_002220/process_sources.csv',low_memory=False);x=x[x.current117&x.center_kind.eq('actual_track')&x.lag_h.between(-24,24)];req=pd.read_csv(V/'results/P1_P2_20261009_002220/enriched_requests.csv');req=req[req.current117&req.center_kind.eq('actual_track')&req.lag_h.between(-24,24)].copy();folders=x[['request_id','source_folder']].drop_duplicates().set_index('request_id');angles={}
 for eid,g in x[x.lag_h.isin(PRE)&x.variable.isin(['shear_u','shear_v'])].groupby('event_id'):
  z=g.pivot(index='lag_h',columns='variable',values='value').dropna();u,v=z[['shear_u','shear_v']].mean();angles[eid]=np.arctan2(v,u) if len(z)>=3 and np.hypot(u,v)>=2 else np.nan
 # Pure gradient/advection functions only, no netCDF import or source writes.
 tree=ast.parse((V/'src/native_fields_v5.py').read_text());names={'_as_float','_field_grid','spherical_gradient','storm_relative_advection'};env={'np':np,'EARTH_RADIUS_M':6371008.8};exec(compile(ast.Module(body=[z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name in names],type_ignores=[]),'native_pure','exec'),env)
 rows=[];errors=[];started=time.perf_counter();req=req.iloc[:limit or None]
 for i,r in enumerate(req.itertuples()):
  if not np.isfinite(r.lat+r.lon):continue
  folder=Path(folders.loc[r.request_id,'source_folder']);p=loadz(folder/'native_process_800km.npz');g=loadz(geometry(r)/'sp.npz');nat=loadz(folder/'native_map_1500km.npz')
  # Subset map native integer coordinates onto original 800-km grid.
  il={int(k):j for j,k in enumerate(nat['latitude_indices'])};jl={int(k):j for j,k in enumerate(nat['longitude_indices'])};ix=np.ix_([il[int(k)] for k in g['latitude_indices']],[jl[int(k)] for k in g['longitude_indices']]);q=nat['q700'][ix]*1000;rh=nat['r700'][ix]
  fields={'q700':q,'RH700':rh,'horizontal_advection':p['q700_horizontal']*86400000,'vertical_advection':p['q700_vertical']*86400000,'moving_tendency':p['q700_moving_tendency']*86400000,'fixed_tendency':p['q700_fixed_tendency']*86400000,'residual':p['q700_residual']*86400000}
  fields['motion_term']=fields['moving_tendency']-fields['fixed_tendency'];joint=np.isfinite(np.stack([fields[k] for k in ['horizontal_advection','vertical_advection','moving_tendency','residual']])).all(0)
  fields['horizontal_advection_fixed_frame']=fields['horizontal_advection']-fields['motion_term']
  for k in ['horizontal_advection','horizontal_advection_fixed_frame','vertical_advection','moving_tendency','residual','fixed_tendency','motion_term']:fields[k]=np.where(joint,fields[k],np.nan)
  errors.append(float(np.nanmax(abs(fields['moving_tendency']-fields['horizontal_advection']-fields['vertical_advection']-fields['residual']))))
  try:fields['q700_climatology']=climq(g,r.UTC)*1000;fields['q700_anomaly']=q-fields['q700_climatology']
  except FileNotFoundError:fields['q700_climatology']=np.full_like(q,np.nan);fields['q700_anomaly']=np.full_like(q,np.nan)
  qjoint=np.isfinite(q)&np.isfinite(fields['q700_climatology']);fields['q700_raw_common']=np.where(qjoint,q,np.nan);fields['q700_climatology']=np.where(qjoint,fields['q700_climatology'],np.nan)
  phi=np.deg2rad(g['latitude'][:,None]);phi0=np.deg2rad(r.lat);dl=np.deg2rad((g['longitude'][None,:]-r.lon+180)%360-180);bear=np.arctan2(np.cos(phi0)*np.sin(phi)-np.sin(phi0)*np.cos(phi)*np.cos(dl),np.cos(phi)*np.sin(dl))
  ur=nat['u700'][ix]-r.motion_u_ms;vr=nat['v700'][ix]-r.motion_v_ms
  # Outward geodesic tangent expressed in the LOCAL east/north basis at each cell.
  radbear=np.arctan2(np.sin(phi)*np.cos(phi0)*np.cos(dl)-np.cos(phi)*np.sin(phi0),np.cos(phi0)*np.sin(dl))
  fields['radial_wind700']=ur*np.cos(radbear)+vr*np.sin(radbear)
  fields['cyclonic_tangential_wind700']=np.sign(r.lat)*(-ur*np.sin(radbear)+vr*np.cos(radbear))
  fields['z200']=nat['z200'][ix]/9.80665
  try:fields['z200_anomaly']=fields['z200']-climq(g,r.UTC,'z200')/9.80665
  except FileNotFoundError:fields['z200_anomaly']=np.full_like(q,np.nan)
  bins=np.floor(g['distance']/50).astype(int);az=np.full_like(q,np.nan)
  for ib in np.unique(bins):
   m=bins==ib;val,_=means(q,g['area'],m);az[m]=val
  fields['q700_azimuthal']=az;fields['q700_asymmetric']=q-az
  for field in ['radial_wind700','cyclonic_tangential_wind700','z200']:
   az=np.full_like(q,np.nan)
   for ib in np.unique(bins):
    m=bins==ib;val,_=means(fields[field],g['area'],m);az[m]=val
   fields[field+'_azimuthal']=az;fields[field+'_asymmetric']=fields[field]-az
  # Aligned native support for additive total/azimuthal/deviation region means.
  for field in ['q700','radial_wind700','cyclonic_tangential_wind700','z200']:
   shared=np.isfinite(fields[field])&np.isfinite(fields[field+'_azimuthal'])
   fields[field+'_axisymmetric_common_total']=np.where(shared,fields[field],np.nan)
   fields[field+'_azimuthal']=np.where(shared,fields[field+'_azimuthal'],np.nan)
   fields[field+'_asymmetric']=np.where(shared,fields[field+'_asymmetric'],np.nan)
  angle=angles.get(r.event_id,np.nan);sectors=np.floor(((bear-angle+np.pi/4)%(2*np.pi))/(np.pi/2)) if np.isfinite(angle) else np.full_like(bear,np.nan)
  for inner,outer in [(100,300),(300,800)]:
   ring=(g['distance']>=inner)&(g['distance']<outer)
   for sector,sectorname in [(-1,'whole'),(0,'downshear'),(1,'left_of_shear'),(2,'upshear'),(3,'right_of_shear')]:
    mask=ring if sector<0 else ring&(sectors==sector)
    for field,arr in fields.items():
     val,frac=means(arr,g['area'],mask);unit='%' if field=='RH700' else 'g kg-1' if field.startswith('q700') else 'm' if field.startswith('z200') else 'm s-1' if 'wind700' in field else 'g kg-1 day-1'
     rows.append(dict(sid=r.sid,event_id=r.event_id,event_class=r.event_class,lag_h=r.lag_h,region=f'{inner}_{outer}km',sector=sectorname,field=field,units=unit,value=val,valid_fraction=frac,geometric_area_m2=float(g['area'][mask].sum()),angle_rad=angle))
  if i%100==0:print('regions',i+1,len(req),round(time.perf_counter()-started),flush=True)
 r=pd.DataFrame(rows);r.to_csv(O/('regions_common_pilot.csv' if limit else 'regions_common_request_sources.csv'),index=False)
 if not limit:
  paired=[]
  for keys,g in r.groupby(['sid','region','sector','field','units']):
   z=g.pivot(index='lag_h',columns='event_class',values='value').reindex(range(-24,25,6)).reindex(columns=['internal','final']);a=z.to_numpy();joint=np.isfinite(a).all(1)
   for name,value in [('pre_mean',np.mean(a[:4][joint[:4]],0) if joint[:4].sum()>=3 else [np.nan,np.nan]),('integral_0_24',np.trapezoid(a[4:],dx=.25,axis=0) if joint[4:].all() else [np.nan,np.nan])]:
    record=dict(zip(['sid','region','sector','field','units'],keys));event_unit=record['units'];contrast_unit='pp' if record['field']=='RH700' else event_unit
    if name.startswith('integral'):
     contrast_unit=contrast_unit.removesuffix(' day-1') if contrast_unit.endswith(' day-1') else contrast_unit+' day'
     event_unit=event_unit.removesuffix(' day-1') if event_unit.endswith(' day-1') else event_unit+' day'
    record.update(units=contrast_unit,event_units=event_unit,estimand=name,internal=value[0],final=value[1],value=value[0]-value[1]);paired.append(record)
  pd.DataFrame(paired).to_csv(O/'regions_common_SID_pairs.csv',index=False);tab,piv,d,w=summary_table(paired,['region','sector','field','units','estimand']);tab.to_csv(O/'regions_common_estimates.csv',index=False);np.savez_compressed(O/'regions_common_bootstrap.npz',sid=piv.index.to_numpy(str),targets=np.array([str(c) for c in piv.columns]),draws=d,weights=w)
 js(O/('regions_common_pilot_receipt.json' if limit else 'regions_common_receipt.json'),{'requests':len(req),'rows':len(r),'max_budget_error_gkg_day':max(errors),'seconds':time.perf_counter()-started,'raw_reads':0,'sector_reference':'all-along pre mean shear, physical left/right no reflection','axisymmetry':'50-km radial bins native area means of q700, z200, radial and cyclonic tangential storm-relative700wind; total retained; asymmetric component is total minus radial-bin mean; transport itself is retained total','integration_units':'dx=.25 day; rates g/kg/day become g/kg; q/RH integrals g/kg day and pp day'})
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['p4','regions']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();p4() if a.mode=='p4' else regions(a.limit)
