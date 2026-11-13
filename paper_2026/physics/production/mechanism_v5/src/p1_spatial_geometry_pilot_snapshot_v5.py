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
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--pilot-launch',type=Path,required=True);args=ap.parse_args();pilot(args.pilot_launch)
