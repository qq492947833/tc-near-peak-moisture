"""P4 linear native-field anomalies on original common area support.
No NC reads, no nonlinear climatological transport proxy, no fit or random draw.
"""
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;U=HERE.parent;ROOT=U.parent
sys.path.insert(0,str(ROOT/'src'))
from native_fields_v5 import area_weighted_support,utc_datetime
sys.path.insert(0,str(U/'input_interpolation/coordinate_alignment_repair_v2'))
from coordinate_labels_v2 import labels
KEYS=('q700','r700','u700','v700');SCALES={'q700':1000.,'r700':1.,'u700':1.,'v700':1.}
UNITS={'q700':'g kg-1','r700':'pp','u700':'m s-1','v700':'m s-1'}
NATIVE_SHA='494cb41141aaa065c340a4cbb51e6c93a699ffcb92415893f3aed22f623786f6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
class PendingInput(Exception):pass

class BoundedClimate:
 def __init__(self,max_bytes=256*1024**2):
  self.root=ROOT/'cache/climatology_1991_2020_v5_1';self.vm=read(ROOT/'config/available_variable_map_v5.json')['paths'];self.cache=OrderedDict();self.max_bytes=max_bytes;self.byte_count=0
  self.plan={(j['key'],j['ilat_tile'],j['ilon_tile']):j for j in read(ROOT/'config/climatology_tiles_v5_1.json')['jobs']}
 def load(self,key,i,j):
  name=f'{key}_y{i}_x{j}';folder=self.root/name;cp=folder/'complete.json';path=folder/'climatology.npz'
  if not cp.exists():raise PendingInput('CLIMATOLOGY_TILE_NOT_COMPLETE: '+name)
  st=(cp.stat().st_mtime_ns,cp.stat().st_size,path.stat().st_mtime_ns,path.stat().st_size)
  if name in self.cache:
   z,identity,prior=self.cache.pop(name)
   if st!=prior:raise ValueError('Completed climate artifact changed during reading')
   self.cache[name]=(z,identity,prior);return z,identity
  c=read(cp);job=self.plan[(key,i,j)]
  expected_signature=hashlib.sha256(json.dumps(dict(job=job,input=self.vm[key]['source_identity'],SP=self.vm['sp']['source_identity'],years=[1991,2020],version=1),sort_keys=True).encode()).hexdigest()
  if c.get('status')!='COMPLETE' or c.get('signature')!=expected_signature or c.get('source_identity')!=self.vm[key]['source_identity'] or c.get('SP_identity')!=self.vm['sp']['source_identity'] or c.get('years')!=[1991,2020]:raise ValueError('Climate source/signature/version conflict')
  with np.load(path,allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
  if str(z['signature'])!=expected_signature or not np.array_equal(z['groups'],np.array(job['groups'])):raise ValueError('Climate group/signature payload conflict')
  if not np.array_equal(labels(z['latitude']),labels(90-np.arange(i*90,min(721,(i+1)*90))*.25)) or not np.array_equal(labels(z['longitude'],True),labels(np.arange(j*180,min(1440,(j+1)*180))*.25,True)):raise ValueError('Climate actual native coordinate labels conflict')
  count=z['count'];expected=z['expected_count'][:,None,None]
  if count.shape!=z['mean'].shape or np.any(count<0) or np.any(count>expected) or np.any(expected<=0) or np.any(np.isfinite(z['mean'])&(count/expected<.9)):raise ValueError('Climate original-time support invalid')
  identity=dict(tile=name,complete=str(cp),complete_sha256=sha(cp),npz=str(path),npz_sha256=sha(path),signature=expected_signature,units=c['units'],years=[1991,2020])
  size=sum(v.nbytes for v in z.values())
  while self.cache and self.byte_count+size>self.max_bytes:
   _,(old,_,_)=self.cache.popitem(last=False);self.byte_count-=sum(v.nbytes for v in old.values())
  self.cache[name]=(z,identity,st);self.byte_count+=size;return z,identity
 def read(self,key,month,hour,ilat,ilon):
  if key not in KEYS or hour not in (0,6,12,18):raise ValueError('Frozen field or UTC conflict')
  group=(month-1)*4+hour//6;ilat=np.asarray(ilat);ilon=np.asarray(ilon);mean=np.full((len(ilat),len(ilon)),np.nan);counts=np.zeros(mean.shape,np.int32);expected=np.zeros(mean.shape,np.int32);identities=[]
  for i in np.unique(ilat//90):
   for j in np.unique(ilon//180):
    z,identity=self.load(key,int(i),int(j));positions=np.flatnonzero(z['groups']==group)
    if len(positions)!=1:raise ValueError('Requested month/UTC group was not planned')
    g=int(positions[0]);a=np.flatnonzero(ilat//90==i);b=np.flatnonzero(ilon//180==j);y=ilat[a]-i*90;x=ilon[b]-j*180
    mean[np.ix_(a,b)]=z['mean'][g][np.ix_(y,x)];counts[np.ix_(a,b)]=z['count'][g][np.ix_(y,x)];expected[np.ix_(a,b)]=z['expected_count'][g];identities.append(dict(identity,group=group,expected_count=int(z['expected_count'][g])))
  return mean,counts,expected,identities

def common_statistics(actual,climate,area,ring):
 if set(actual)!=set(KEYS) or set(climate)!=set(KEYS):raise ValueError('All four fixed fields required')
 common=np.logical_and.reduce([np.isfinite(actual[k])&np.isfinite(climate[k]) for k in KEYS])
 result={}
 for k in KEYS:
  scale=SCALES[k];aa=np.where(common,actual[k]*scale,np.nan);cc=np.where(common,climate[k]*scale,np.nan)
  result[k]=dict(raw_own_support=area_weighted_support(actual[k]*scale,area,ring,min_fraction=.9),raw_common=area_weighted_support(aa,area,ring,min_fraction=.9),climate_common=area_weighted_support(cc,area,ring,min_fraction=.9),anomaly_common=area_weighted_support(aa-cc,area,ring,min_fraction=.9),unit=UNITS[k],raw_unit='%' if k=='r700' else UNITS[k])
 return result,common

def source_anomaly(request,source_complete,source_sha,reader):
 cp=Path(source_complete)
 if sha(cp)!=source_sha:raise ValueError('Native source COMPLETE changed')
 done=read(cp);binding=done['binding']
 if done.get('status')!='COMPLETE' or binding['request']!=request or binding['driver_sha']!=sha(ROOT/'src/extract_environment_v5.py') or binding['semantics_sha']!=digest(read(ROOT/'config/extraction_semantics_v5.json')):raise ValueError('Native extraction request/implementation conflict')
 if request['module']!='P4' or sha(ROOT/'src/native_fields_v5.py')!=NATIVE_SHA:raise ValueError('Fixed P4/native source identity conflict')
 normal=binding['normalised'];geometry=dict(UTC=utc_datetime(normal['UTC']).isoformat()+'Z',latitude=float(normal['lat']),longitude=float(normal['lon'])%360,radius_km=800.,halo=1,reader_sha=NATIVE_SHA);gid=digest(geometry)[:20];folder=ROOT/'cache/native_fields'/gid;gm=read(folder/'geometry.json')
 if gm['binding']['geometry']!=geometry or gm['binding']['sp_identity']!=reader.vm['sp']['source_identity'] or gm['binding']['sp_metadata_sha']!=reader.vm['sp']['metadata_sha256'] or sha(folder/'sp.npz')!=gm['npz_sha256']:raise ValueError('Native geometry/SP conflict')
 with np.load(folder/'sp.npz',allow_pickle=False) as f:
  if json.loads(str(f['binding_json'].item()))!=gm['binding']:raise ValueError('SP embedded source conflict')
  g={k:f[k].copy() for k in f.files if k!='binding_json'}
 if not np.array_equal(labels(g['latitude']),labels(90-g['latitude_indices']*.25)) or not np.array_equal(labels(g['longitude'],True),labels(g['longitude_indices']*.25,True)):raise ValueError('Actual quarter-degree native indices conflict')
 actual={};climate={};counts={};expected={};lineage={};tileids={};t=utc_datetime(normal['UTC']);vm=reader.vm
 for key in KEYS:
  meta=read(folder/(key+'.json'));path=folder/(key+'.npz')
  if meta['binding']['geometry']!=gid or meta['binding']['source_identity']!=vm[key]['source_identity'] or meta['binding']['metadata_sha']!=vm[key]['metadata_sha256'] or sha(path)!=meta['npz_sha256']:raise ValueError('Native field source conflict '+key)
  with np.load(path,allow_pickle=False) as f:
   if json.loads(str(f['binding_json'].item()))!=meta['binding']:raise ValueError('Embedded native field binding conflict')
   raw=f['raw'];mask=f['support_mask'];a=f['values'].copy()
  expected_mask=np.isfinite(raw)&np.isfinite(g['sp'])&(g['sp']>=70000.)
  if not np.array_equal(mask,expected_mask) or not np.array_equal(a,np.where(expected_mask,raw,np.nan),equal_nan=True):raise ValueError('Original SP/finite native mask conflict')
  actual[key]=a;climate[key],counts[key],expected[key],ids=reader.read(key,t.month,t.hour,g['latitude_indices'],g['longitude_indices']);tileids[key]=ids;lineage[key]=dict(npz=str(path),npz_sha256=sha(path),metadata_sha256=sha(folder/(key+'.json')))
 ring=(g['distance']>=200)&(g['distance']<800);statistics,common=common_statistics(actual,climate,g['area'],ring)
 result=dict(request_id=request['request_id'],sid=request['sid'],UTC=request['UTC'],lag_h=request['lag_h'],stage=request.get('stage',''),source_complete=str(cp),source_complete_sha256=source_sha,geometry_id=gid,geometry=geometry,sp_sha256=gm['npz_sha256'],geometry_metadata_sha256=sha(folder/'geometry.json'),native_fields=lineage,climate_tiles=tileids,statistics=statistics,common_valid_cells=int((common&ring).sum()),original_ring_cells=int(ring.sum()),raw_nc_reads=0,nonlinear_process_anomaly=False,wind_anomaly_definition='(u-c)-(monthlyUTC u-c)=u-monthlyUTC u; same for v; no second motion subtraction',cache_bytes=reader.byte_count)
 arrays={key+'__actual':actual[key] for key in KEYS};arrays.update({key+'__climate':climate[key] for key in KEYS});arrays.update({key+'__climate_count':counts[key] for key in KEYS});arrays.update({key+'__climate_expected':expected[key] for key in KEYS});arrays.update(area_m2=g['area'],ring=ring,common=common)
 if sha(cp)!=source_sha:raise ValueError('Source changed during read')
 return result,arrays
