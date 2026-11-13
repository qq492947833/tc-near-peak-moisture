"""P1 native-to-spherical-AEQD maps; no raw ERA5 or unfinished climatology reads."""
from pathlib import Path
import argparse,json,csv,hashlib,os,sys
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=6371008.8;G=9.80665
GRID=np.arange(-1500.,1500.1,100.)*1000
X,Y=np.meshgrid(GRID,GRID);DISPLAY=X*X+Y*Y<=(1500000.)**2
NATIVE=['q700','u700','v700','z200','u200','v200','u850','v850']
FIELDS=['q700_gkg','u700_relative_ms','v700_relative_ms','z200_m','u200_ms','v200_ms','shear_u_ms','shear_v_ms']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def project(lat,lon,lat0,lon0):
 f=np.deg2rad(np.asarray(lat,float));dl=np.deg2rad((np.asarray(lon,float)-lon0+180)%360-180);f0=np.deg2rad(lat0)
 a=np.cos(f)*np.sin(dl);b=np.cos(f0)*np.sin(f)-np.sin(f0)*np.cos(f)*np.cos(dl)
 h=np.clip(np.sin(f0)*np.sin(f)+np.cos(f0)*np.cos(f)*np.cos(dl),-1,1)
 c=np.arccos(h);s=np.sqrt(np.maximum(0,1-h*h));k=np.divide(c,s,out=np.ones_like(c),where=s>1e-10)
 return R*k*a,R*k*b
def inverse(x,y,lat0,lon0):
 x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float));f0=np.deg2rad(lat0);rho=np.hypot(x,y);c=rho/R
 sr=np.divide(np.sin(c),rho,out=np.full_like(rho,1/R),where=rho>0)
 f=np.arcsin(np.clip(np.cos(c)*np.sin(f0)+y*sr*np.cos(f0),-1,1))
 dl=np.arctan2(x*sr,np.cos(f0)*np.cos(c)-y*sr*np.sin(f0))
 return np.rad2deg(f),(lon0+np.rad2deg(dl))%360
def project_vector(lat,lon,u,v,lat0,lon0):
 f=np.deg2rad(np.asarray(lat,float));dl=np.deg2rad((np.asarray(lon,float)-lon0+180)%360-180);f0=np.deg2rad(lat0)
 a=np.cos(f)*np.sin(dl);b=np.cos(f0)*np.sin(f)-np.sin(f0)*np.cos(f)*np.cos(dl)
 h=np.clip(np.sin(f0)*np.sin(f)+np.cos(f0)*np.cos(f)*np.cos(dl),-1,1);s=np.sqrt(np.maximum(0,1-h*h));c=np.arccos(h)
 k=np.divide(c,s,out=np.ones_like(c),where=s>1e-8)
 kh=np.divide(-(s-c*h),s**3,out=np.full_like(s,-1/3),where=s>1e-5)
 ae=np.cos(dl);be=np.sin(f0)*np.sin(dl);he=-np.cos(f0)*np.sin(dl)
 an=-np.sin(f)*np.sin(dl);bn=np.cos(f0)*np.cos(f)+np.sin(f0)*np.sin(f)*np.cos(dl);hn=np.sin(f0)*np.cos(f)-np.cos(f0)*np.sin(f)*np.cos(dl)
 return (k*ae+a*kh*he)*u+(k*an+a*kh*hn)*v,(k*be+b*kh*he)*u+(k*bn+b*kh*hn)*v
def rotate(x,y,angle):
 c=np.cos(angle);s=np.sin(angle);return c*x+s*y,-s*x+c*y
def bilinear(lat,lon,values,target_lat,target_lon,lon0):
 lat=np.asarray(lat,float);lon=lon0+(np.asarray(lon,float)-lon0+180)%360-180
 il=np.argsort(lat);jl=np.argsort(lon);lat=lat[il];lon=lon[jl];v=np.asarray(values,float)[np.ix_(il,jl)]
 a,b=np.broadcast_arrays(np.asarray(target_lat,float),lon0+(np.asarray(target_lon,float)-lon0+180)%360-180)
 ii=np.searchsorted(lat,a,side='right')-1;jj=np.searchsorted(lon,b,side='right')-1
 inside=(a>=lat[0])&(a<=lat[-1])&(b>=lon[0])&(b<=lon[-1])
 ii=np.clip(ii,0,len(lat)-2);jj=np.clip(jj,0,len(lon)-2)
 t=(a-lat[ii])/(lat[ii+1]-lat[ii]);s=(b-lon[jj])/(lon[jj+1]-lon[jj])
 vv=np.stack([v[ii,jj],v[ii+1,jj],v[ii,jj+1],v[ii+1,jj+1]])
 ww=np.stack([(1-t)*(1-s),t*(1-s),(1-t)*s,t*s]);ok=(np.isfinite(vv)|(np.abs(ww)<1e-14)).all(axis=0)&inside
 return np.where(ok,np.sum(np.where(np.isfinite(vv),vv,0)*ww,axis=0),np.nan)
def pre_mean(a):
 a=np.asarray(a,float);n=np.isfinite(a).sum(axis=0)
 return np.divide(np.nansum(a,axis=0),n,out=np.full(n.shape,np.nan),where=n>=3),n
def map_fields(native,request,angle=0,climatology=None):
 la=np.asarray(native['latitude']);lo=np.asarray(native['longitude']);lat,lon=np.meshgrid(la,lo,indexing='ij')
 f={k:np.asarray(native[k],float) for k in NATIVE}
 anomalous=climatology is not None
 if anomalous:f={k:f[k]-climatology[k] for k in NATIVE}
 # In anomaly space (u-c)-(Uclim-c) = u-Uclim; no second motion subtraction.
 cu=0. if anomalous else float(request.get('motion_u_ms') or np.nan);cv=0. if anomalous else float(request.get('motion_v_ms') or np.nan)
 outputs={'q700_gkg':f['q700']*1000,'z200_m':f['z200']/G}
 for names,u,v in [(('u700_relative_ms','v700_relative_ms'),f['u700']-cu,f['v700']-cv),(('u200_ms','v200_ms'),f['u200'],f['v200']),(('shear_u_ms','shear_v_ms'),f['u200']-f['u850'],f['v200']-f['v850'])]:
  u,v=project_vector(lat,lon,u,v,float(request['lat']),float(request['lon']));outputs[names[0]],outputs[names[1]]=rotate(u,v,angle)
 xx,yy=rotate(X,Y,-angle);tla,tlo=inverse(xx,yy,float(request['lat']),float(request['lon']))
 out=np.stack([np.where(DISPLAY,bilinear(la,lo,outputs[k],tla,tlo,float(request['lon'])),np.nan) for k in FIELDS])
 return out
def checked_map(folder,driver_sha,semantics_sha):
 cp=read(folder/'COMPLETE.json');b=cp['binding']
 if cp['status']!='COMPLETE' or b['driver_sha']!=driver_sha or b['semantics_sha']!=semantics_sha:raise ValueError('Source identity mismatch')
 item=next(z for z in cp['outputs'] if z['name']=='native_map_1500km.npz')
 if sha(folder/item['name'])!=item['sha256']:raise ValueError('Native map checksum mismatch')
 with np.load(folder/item['name'],allow_pickle=False) as z:native={k:z[k].copy() for k in z.files}
 return native,b['request'],dict(complete_sha256=sha(folder/'COMPLETE.json'),map_sha256=item['sha256'],folder=str(folder))
def climatology_fields(reader,native,request):
 t=datetime.fromisoformat(request['UTC'].replace('Z','+00:00'));out={};counts={}
 for k in NATIVE:out[k],counts[k]=reader.read(k,t.month,t.hour,native['latitude_indices'],native['longitude_indices'])
 return out,counts
def source_index(launch):
 from derive_thermodynamics_v5 import launch_records
 out={}
 for p in launch_records(launch):
  f=Path(p);cp=read(f/'COMPLETE.json');r=cp['binding']['request']
  if r.get('module')=='P1' and cp['binding']['normalised']['need_map']:
   if r['request_id'] in out and out[r['request_id']]!=f:raise ValueError('Duplicate P1 source request')
   out[r['request_id']]=f
 return out
def event_orientation(event_id,index):
 got=[];sources=[]
 for lag in [-24,-18,-12,-6]:
  key='P1|'+event_id+'|fixed|'+str(lag)
  if key not in index:continue
  with (index[key]/'scalars.csv').open(encoding='utf-8-sig') as f:sc={x['variable']:x for x in csv.DictReader(f)}
  u=sc.get('shear_200_850_u_ms',{});v=sc.get('shear_200_850_v_ms',{})
  def valid(a):return a.get('support_pass','').lower()=='true' and float(a['valid_fraction'])>=.9 and np.isfinite(float(a['value']))
  if valid(u) and valid(v):got.append([float(u['value']),float(v['value'])]);sources.append(key)
 if len(got)<3:return dict(valid=False,reason='PRE_DIRECTION_LESS_THAN_3_OF_4',n_pre=len(got),angle_rad=None,speed_ms=None,sources=sources)
 u,v=np.mean(got,axis=0);speed=float(np.hypot(u,v))
 return dict(valid=speed>=2,reason='PASS' if speed>=2 else 'WEAK_PRE_SHEAR_BELOW_2_MS',n_pre=len(got),angle_rad=float(np.arctan2(v,u)),speed_ms=speed,sources=sources)
def pilot(launch_path):
 from derive_thermodynamics_v5 import atomic,canon,digest
 launch=read(launch_path);index=source_index(launch);sem=read(ROOT/'config/extraction_semantics_v5.json');examples=[];out=ROOT/'audit/p1_spatial'/('real_geometry_'+datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True)
 selected=[]
 for key,p in index.items():
  r=read(p/'COMPLETE.json')['binding']['request']
  if r['center_kind']=='fixed_t0' and int(float(r['lag_h']))==-24:
   hemi='NH' if float(r['lat'])>0 else 'SH'
   if not any(x[0]==hemi for x in selected):selected.append((hemi,key,p))
  if len(selected)==2:break
 for hemi,key,p in selected:
  native,r,identity=checked_map(p,launch['source_identity']['driver_sha256'],digest(sem));orientation=event_orientation(r['event_id'],index)
  angle=orientation['angle_rad'] if orientation['valid'] else 0.
  transformed=map_fields(native,r,angle);lat,lon=np.meshgrid(native['latitude'],native['longitude'],indexing='ij');x,y=project(lat,lon,float(r['lat']),float(r['lon']));la,lo=inverse(x,y,float(r['lat']),float(r['lon']))
  err=max(float(np.max(np.abs(la-lat))),float(np.max(np.abs((lo-lon+180)%360-180))))
  assert err<1e-8 and np.isfinite(transformed[0][DISPLAY]).sum()>500
  with (out/(hemi+'_geometry.npz')).open('wb') as f:np.savez_compressed(f,fields=transformed,x_m=X,y_m=Y,display_mask=DISPLAY,field_names=np.array(FIELDS))
  examples.append(dict(request_id=key,hemisphere=hemi,source=identity,orientation=orientation,roundtrip_max_deg=err,finite_scalar_cells=int(np.isfinite(transformed[0][DISPLAY]).sum())))
 result=dict(status='GEOMETRY_PILOT_PASS_NOT_SCIENTIFIC_COMPOSITE',UTC=datetime.now(timezone.utc).isoformat(),implementation_sha256=sha(__file__),examples=examples,raw_ERA5_reads=0,climatology_not_used=True)
 atomic(out/'verification.json',result);print(canon(result));print(out)

# The preparation and inference interfaces keep current source maps and cohort support explicit.
SUMMARIES=['pre_fixed_minus24_minus6','minus24_fixed','zero_actual','plus24_actual']
PRE=[-24,-18,-12,-6]
def requested_id(event,kind,lag):return 'P1|'+event+'|'+kind+'|'+str(lag)
def load_clim_with_identity(reader,native,request):
 fields,counts=climatology_fields(reader,native,request);identity={}
 for k in NATIVE:
  for i in np.unique(np.asarray(native['latitude_indices'])//90):
   for j in np.unique(np.asarray(native['longitude_indices'])//180):
    name=f'{k}_y{i}_x{j}';folder=reader.root/name
    identity[name]=dict(complete_sha256=sha(folder/'complete.json'),npz_sha256=sha(folder/'climatology.npz'))
 return fields,counts,identity
def event_fields(event,index,launch,semantics_sha,reader=None,request_table=None):
 orientation=event_orientation(event['event_id'],index)
 needed=[('fixed',k) for k in PRE]+[('along',0),('along',24)]
 result={};sources=[];pending=[];climate_ids={};unsupported=[]
 for kind,lag in needed:
  rid=requested_id(event['event_id'],kind,lag)
  if rid not in index:
   
   original=(request_table or {}).get(rid)
   if original is not None and not (np.isfinite(float(original.get('lat') or np.nan)) and np.isfinite(float(original.get('lon') or np.nan))):
    unsupported.append(dict(request_id=rid,reason=original.get('source_support_reason','NO_OBSERVED_POSITION'),scope='legitimate unavailable track point, no extrapolation'))
   else:pending.append(dict(request_id=rid,reason='REQUEST_NOT_IN_COMPLETED_CURRENT_LEDGER'))
   continue
  native,request,source=checked_map(index[rid],launch['source_identity']['driver_sha256'],semantics_sha)
  source.update(request_id=rid);sources.append(source)
  clim=None;clim_reason='CLIMATOLOGY_NOT_REQUESTED_FOR_RAW_PREPARATION'
  if reader is not None:
   try:clim,counts,ident=load_clim_with_identity(reader,native,request);climate_ids.update(ident)
   except FileNotFoundError as exc:clim_reason=str(exc);pending.append(dict(request_id=rid,reason=clim_reason))
  for frame,angle in [('unrotated',0.),('fixed_pre_shear',orientation['angle_rad'] if orientation['valid'] else None)]:
   if angle is None:
    raw=np.full((len(FIELDS),len(GRID),len(GRID)),np.nan);anom=np.full_like(raw,np.nan)
   else:
    raw=map_fields(native,request,angle);anom=map_fields(native,request,angle,clim) if clim is not None else np.full_like(raw,np.nan)
   result[(kind,lag,frame)]=(raw,anom,np.where(np.isfinite(anom)&np.isfinite(raw),raw,np.nan))
 outputs={}
 missing=np.full((len(FIELDS),len(GRID),len(GRID)),np.nan)
 for frame in ['unrotated','fixed_pre_shear']:
  for representation,i in [('raw',0),('anomaly',1),('raw_common',2)]:
   pre=np.stack([result.get(('fixed',lag,frame),(missing,missing,missing))[i] for lag in PRE])
   mean,support=pre_mean(pre)
   outputs[frame+'__'+representation]=np.stack([mean,result.get(('fixed',-24,frame),(missing,missing,missing))[i],result.get(('along',0,frame),(missing,missing,missing))[i],result.get(('along',24,frame),(missing,missing,missing))[i]])
   outputs[frame+'__'+representation+'__pre_time_n']=support
 return outputs,dict(event=event,orientation=orientation,sources=sources,climatology_identities=climate_ids,pending=pending,unsupported=unsupported)
def prepare_events(launch_path,use_climatology=False,limit=0):
 from derive_thermodynamics_v5 import atomic,canon,digest
 launch=read(launch_path);index=source_index(launch);semsha=digest(read(ROOT/'config/extraction_semantics_v5.json'));reader=None
 if use_climatology:
  from climatology_v5 import ClimatologyReader
  reader=ClimatologyReader()
 with (ROOT/'data/p1_event_identity_v5.csv').open(encoding='utf-8-sig') as f:events=list(csv.DictReader(f))
 with (ROOT/'data/environment_requests_v5.csv').open(encoding='utf-8-sig') as f:request_table={r['request_id']:r for r in csv.DictReader(f) if r['module']=='P1'}
 receipt=ROOT/'audit/p1_spatial'/('prepare_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+str(os.getpid()));receipt.mkdir(parents=True)
 records=[]
 for event in events[:limit or None]:
  arrays,meta=event_fields(event,index,launch,semsha,reader,request_table)
  binding=dict(implementation_sha256=sha(__file__),source_launch_sha256=sha(launch_path),event_identity_sha256=sha(ROOT/'data/p1_event_identity_v5.csv'),metadata=meta,summary_labels=SUMMARIES,field_names=FIELDS)
  h=digest(binding)[:20];dest=ROOT/'data/p1_spatial_events'/h;dest.mkdir(parents=True,exist_ok=True);done=dest/'COMPLETE.json'
  if done.exists():
   old=read(done)
   if old['binding']!=binding or old['arrays_sha256']!=sha(dest/'fields.npz'):raise ValueError('P1 spatial checkpoint identity conflict')
   status='REUSED'
  else:
   temp=dest/('fields.'+str(os.getpid())+'.tmp')
   with temp.open('wb') as f:np.savez_compressed(f,**arrays,x_m=X,y_m=Y,display_mask=DISPLAY)
   os.replace(temp,dest/'fields.npz')
   atomic(done,dict(status='COMPLETE_DERIVATION',binding=binding,arrays_sha256=sha(dest/'fields.npz'),raw_ERA5_reads=0,climatology_complete=bool(use_climatology and not meta['pending']),created_utc=datetime.now(timezone.utc).isoformat()))
   status='COMPLETE_DERIVATION'
  rec=dict(event_id=event['event_id'],sid=event['sid'],event_class=event['event_class'],folder=str(dest),complete_sha256=sha(done),status=status,climatology_complete=bool(use_climatology and not meta['pending']),n_pending=len(meta['pending']))
  records.append(rec)
  with (receipt/'events.jsonl').open('a',encoding='utf-8') as f:f.write(canon(rec)+'\n')
  print(canon(rec),flush=True)
 atomic(receipt/'execution.json',dict(status='SOURCE_PREPARATION_ONLY',events=records,expected_events=246,processed=len(records),complete_climatology_events=sum(x['climatology_complete'] for x in records),raw_ERA5_reads=0))
 return receipt
def paired_support(raw,anomaly,raw_same_time_support=None):
 raw=np.asarray(raw,float);anomaly=np.asarray(anomaly,float)
 if raw.shape!=anomaly.shape or raw.shape[1]!=2:raise ValueError('SID x two event arrays required')
 raw_ok=np.isfinite(raw).all(axis=1)
 common_ok=raw_ok&np.isfinite(anomaly).all(axis=1)
 matched=raw if raw_same_time_support is None else np.asarray(raw_same_time_support,float)
 if matched.shape!=raw.shape:raise ValueError('Same-time raw counterpart shape mismatch')
 common_ok &= np.isfinite(matched).all(axis=1)
 return np.where(raw_ok[:,None],raw,np.nan),np.where(common_ok[:,None],matched,np.nan),np.where(common_ok[:,None],anomaly,np.nan)
def paired_components(a):
 ok=np.isfinite(a).all(axis=1);i=np.where(ok,a[:,0],np.nan);f=np.where(ok,a[:,1],np.nan)
 return i,f,i-f  # Fixed sign: internal minus final.
def cohort_arrays(event_records,cohort='current117'):
 bysid={}
 for rec in event_records:
  done=read(Path(rec['folder'])/'COMPLETE.json');ev=done['binding']['metadata']['event']
  if ev[cohort].strip().lower()!='true':continue
  bysid.setdefault(ev['sid'],{})[ev['event_class']]=(rec,done,ev)
 sids=sorted(bysid);rows=[];meta=[];out={k:[] for k in [f+'__'+v for f in ['unrotated','fixed_pre_shear'] for v in ['raw','anomaly','raw_common']]}
 for sid in sids:
  sides=bysid[sid]
  if set(sides)!={'internal','final'}:raise ValueError('Not a complete SID event pair '+sid)
  lats=[float(sides[s][2]['lat']) for s in ['internal','final']]
  hemi='NH' if min(lats)>0 else 'SH' if max(lats)<0 else 'CROSS_HEMISPHERE_OR_EQUATOR'
  meta.append(dict(sid=sid,hemisphere=hemi,internal_event_id=sides['internal'][2]['event_id'],final_event_id=sides['final'][2]['event_id']))
  for k in out:
   pair=[]
   for side in ['internal','final']:
    rec,done,ev=sides[side]
    if sha(Path(rec['folder'])/'fields.npz')!=done['arrays_sha256']:raise ValueError('Event map changed')
    with np.load(Path(rec['folder'])/'fields.npz') as z:pair.append(z[k])
   out[k].append(np.stack(pair))
 return sids,meta,{k:np.stack(v) for k,v in out.items()}
def infer_fields(a,sids,metadata,seed=510081234):
 """One SID draw carries both events, all summaries, fields, hemispheres and grid points."""
 from v5_statistics import bootstrap_weights,paired_mean_draws,summarize_draws,bh
 indices,weights=bootstrap_weights(len(sids),2000,seed);_,_,dif=paired_components(a);summaries=[];draw_blocks={}
 for hemi in ['NH','SH']:
  select=np.array([x['hemisphere']==hemi for x in metadata])
  for fi,name in [(0,'q700_gkg'),(3,'z200_m')]:
   arr=dif[:,:,fi][:,:,:, :].reshape(len(sids),len(SUMMARIES),-1)[:,:,DISPLAY.ravel()]
   arr=np.where(select[:,None,None],arr,np.nan).reshape(len(sids),-1)
   est,draw,n=paired_mean_draws(arr,weights,minimum_n=2)
   draw[:,n<10]=np.nan;draw_blocks[hemi+'__'+name]=draw.reshape(2000,4,int(DISPLAY.sum()))
   for j in range(len(est)):
    z=summarize_draws(est[j],draw[:,j]) if n[j]>=10 else dict(estimate=float(est[j]),lo=np.nan,hi=np.nan,p=np.nan,valid_draws=0)
    summary,cell=divmod(j,int(DISPLAY.sum()));flat=np.flatnonzero(DISPLAY.ravel())[cell];iy,ix=divmod(flat,len(GRID))
    summaries.append(dict(hemisphere=hemi,field=name,summary=SUMMARIES[summary],x_km=GRID[ix]/1000,y_km=GRID[iy]/1000,n_sid=int(n[j]),**z))
 for name in ['q700_gkg','z200_m']:
  ids=[i for i,x in enumerate(summaries) if x['field']==name];q=bh([summaries[i]['p'] for i in ids],family_size=4*2*int(DISPLAY.sum()))
  for i,qq in zip(ids,q):summaries[i]['q']=float(qq);summaries[i]['family_size']=4*2*int(DISPLAY.sum())
 return summaries,draw_blocks,indices,weights

def analyze_prepared(execution_path,output):
 from derive_thermodynamics_v5 import atomic,csvsave
 cfg_path=ROOT/'config/p1_spatial_inference_v5.json'
 if not cfg_path.exists():raise RuntimeError('Inference display-family definition must be frozen before statistical execution')
 cfg=read(cfg_path)
 if cfg['primary_frame']!='fixed_pre_shear' or cfg['primary_representation']!='anomaly' or cfg['family_size_per_scalar']!=5672:raise ValueError('Unknown fixed map family')
 run=read(execution_path);records=run['events']
 if len(records)!=246 or any(not x['climatology_complete'] for x in records):raise RuntimeError('Full246 event preparation and required climatology incomplete; no formal composite')
 output=Path(output)
 if output.exists():raise FileExistsError('Preserve existing spatial analysis; choose new output directory')
 output.mkdir(parents=True);all_rows=[];readiness=[]
 for cohort,expected in [('current117',117),('candidate123',123)]:
  sids,meta,arrays=cohort_arrays(records,cohort)
  if len(sids)!=expected:raise ValueError('Cohort identity count differs from frozen starting cohort')
  csvsave(output/(cohort+'_SID.csv'),meta)
  for frame in ['unrotated','fixed_pre_shear']:
   rawonly,rawcommon,anomaly=paired_support(arrays[frame+'__raw'],arrays[frame+'__anomaly'],arrays[frame+'__raw_common'])
   with (output/(cohort+'_'+frame+'_perSID.npz')).open('wb') as f:np.savez_compressed(f,raw_only=rawonly,raw_common=rawcommon,anomaly_common=anomaly,sid=np.array(sids),fields=np.array(FIELDS),summaries=np.array(SUMMARIES),x_m=X,y_m=Y,display_mask=DISPLAY)
   for view,a in [('raw_only',rawonly),('raw_common',rawcommon),('anomaly_common',anomaly)]:
    rows,draws,indices,weights=infer_fields(a,sids,meta,seed=cfg['seed'])
    primary=frame=='fixed_pre_shear' and view=='anomaly_common'
    for row in rows:
     row.update(cohort=cohort,frame=frame,representation=view,contrast='internal minus final',primary_map_family=primary,CI='paired SID percentile bootstrap,2000 attempted draws')
     if not primary:row.update(p=np.nan,q=np.nan,family_size=0,inference_status='DESCRIPTIVE_INTERVAL_NO_ADDITIONAL_P_OR_Q')
     else:row['inference_status']='MAIN_MAP_FAMILY' if cohort=='current117' else 'PRESPECIFIED_CANDIDATE_SENSITIVITY_FAMILY'
    all_rows.extend(rows)
    with (output/(cohort+'_'+frame+'_'+view+'_draws.npz')).open('wb') as f:np.savez_compressed(f,**draws,indices=indices,weights=weights,sid=np.array(sids))
    internal,final,diff=paired_components(a)
    def mean(x):
     n=np.isfinite(x).sum(axis=0);return np.divide(np.nansum(x,axis=0),n,out=np.full(n.shape,np.nan),where=n>0),n
    mi=[];mf=[];md=[];ns=[]
    for hemisphere in ['NH','SH']:
     selected=np.array([m['hemisphere']==hemisphere for m in meta])
     ai,an=mean(internal[selected]);af,_=mean(final[selected]);ad,_=mean(diff[selected]);mi.append(ai);mf.append(af);md.append(ad);ns.append(an)
    mi,mf,md,n=map(np.stack,[mi,mf,md,ns])
    with (output/(cohort+'_'+frame+'_'+view+'_paired_composites.npz')).open('wb') as f:np.savez_compressed(f,internal=mi,final=mf,difference=md,n_sid=n,hemisphere=np.array(['NH','SH']),fields=np.array(FIELDS),summaries=np.array(SUMMARIES))
 csvsave(output/'map_statistics.csv',all_rows)
 atomic(output/'COMPLETE.json',dict(status='COMPUTED_PENDING_INDEPENDENT_REVIEW',source_execution=str(execution_path),source_execution_sha256=sha(execution_path),config_sha256=sha(cfg_path),implementation_sha256=sha(__file__),primary_contrast='internal minus final',raw_ERA5_reads=0,field_rows=len(all_rows),no_causal_interpretation=True))
 return output

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--launch',type=Path);ap.add_argument('--mode',choices=['pilot','prepare-raw','prepare-climatology','analyze'],default='pilot');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--execution',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
 if args.mode=='analyze':print(analyze_prepared(args.execution,args.output))
 elif args.mode=='pilot':pilot(args.launch)
 else:print(prepare_events(args.launch,args.mode=='prepare-climatology',args.limit))
