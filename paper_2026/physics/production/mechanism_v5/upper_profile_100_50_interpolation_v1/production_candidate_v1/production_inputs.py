"""Read-only inputs for isolated logp70/inclusive50 production candidates.
No PI calculation, NC read, writes, source modification or production launch.
"""
from pathlib import Path
from functools import lru_cache
import csv,json,hashlib,sys
import numpy as np
HERE=Path(__file__).resolve().parent
U=HERE.parent
ROOT=U.parent
I=U/'input_interpolation'
NATIVE_LEVELS=np.array([1000,925,850,700,600,500,400,300,250,200,150,100,50])
PINS={
 'mapper':('input_interpolation/coordinate_alignment_repair_v2/coordinate_labels_v2.py','d8d300fe1e390bcb98ee556bff09249b290bb6b048b72ffc2b9507b0758772f9'),
 'interpolation':('input_interpolation/interpolation_core.py','5f912d43c8b1fef2b2af9040a167c1f57d86eb009d1f2b9b22c40d4cebc0d225'),
 'axes':('input_interpolation/source_coordinate_axes.npz','20876ca34132c719dab82835a0897f0c77a253552f423b9113041626d494c2bf'),
 'audit':('input_interpolation/four_source_input_audit.json','6bb6330c33f405e9146cc1ff758d7e6dacf1517458351ccf692b6501f505b2b2'),
 'candidate':('candidate_configuration_v1.json','70efa49e4030463f0d1f232c78cf27d44a389a9c979375880ca379a902162952')
}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def number(x):
 try:return float(x)
 except (TypeError,ValueError):return np.nan
def boolean(x):
 s=str(x).strip().lower()
 if s in ('true','1'):return True
 if s in ('false','0',''):return False
 raise ValueError('invalid explicit boolean '+repr(x))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def digest(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

@lru_cache(maxsize=1)
def _context():
 for name,(rel,pin) in PINS.items():
  if sha(U/rel)!=pin:raise ValueError('frozen input dependency changed: '+name)
 catalog=ROOT/'config/available_variable_map_v5.json'
 audit=read(I/'four_source_input_audit.json')
 if sha(catalog)!=audit['catalog_sha256']:raise ValueError('catalog changed')
 # Metadata identity only; no NC content/header re-read and no NC hash.
 for item in audit['files']:
  ident=item['identity'];st=Path(ident['path']).stat()
  if (st.st_size,st.st_mtime_ns,st.st_ino)!=(ident['size'],ident['mtime_ns'],ident['file_id']):
   raise ValueError('original endpoint source identity changed '+item['key'])
 mapper={'__name__':'production_coordinate_mapper','__file__':str(U/PINS['mapper'][0])}
 exec(compile((U/PINS['mapper'][0]).read_text('utf-8'),mapper['__file__'],'exec'),mapper)
 inter={'__name__':'production_interpolation_core','__file__':str(U/PINS['interpolation'][0])}
 exec(compile((U/PINS['interpolation'][0]).read_text('utf-8'),inter['__file__'],'exec'),inter)
 native_path=ROOT/'src/native_fields_v5.py'
 if sha(native_path)!='494cb41141aaa065c340a4cbb51e6c93a699ffcb92415893f3aed22f623786f6':raise ValueError('native support function changed')
 sys.path.insert(0,str(ROOT/'src'))
 from native_fields_v5 import utc_datetime,area_weighted_support
 with np.load(I/'source_coordinate_axes.npz',allow_pickle=False) as z:axes={k:z[k].copy() for k in z.files}
 return dict(vm=read(catalog)['paths'],axes=axes,mapper=mapper,inter=inter,utc=utc_datetime,
  area=area_weighted_support,native_sha=sha(native_path),input_binding={k:dict(path=str(U/v[0]),sha256=v[1]) for k,v in PINS.items()})

def prepare13(original):
 """Retain original arrays; never remove a missing lower prefix."""
 p=np.asarray(original['pressure_hpa'])
 t=np.asarray(original['temperature_K']);q=np.asarray(original['specific_humidity_kgkg'])
 f=np.asarray(original['joint_Tq_valid_fraction'])
 if not np.array_equal(p,NATIVE_LEVELS) or t.shape!=(13,) or q.shape!=(13,) or f.shape!=(13,):
  raise ValueError('original fixed13 shape/pressure mismatch')
 if np.any(np.isfinite(f)&((f<0)|(f>1+1e-12))):raise ValueError('invalid area fraction')
 good=np.isfinite(t)&(t>100)&(t<400)&np.isfinite(q)&(q>=0)&(q<.1)&np.isfinite(f)&(f>=.9)
 return dict(pressure_hpa=p.copy(),temperature_K=np.where(good,t,np.nan),specific_humidity_kgkg=np.where(good,q,np.nan),
  joint_Tq_valid_fraction=f.copy(),valid_mask=good,valid=bool(good.all()),unsupported_levels_hpa=p[~good].tolist())

def add70(prepared,t70=np.nan,q70=np.nan,f70=np.nan):
 # Every original position retained. Unsupported original values stay joint NA.
 return dict(pressure_hpa=np.insert(prepared['pressure_hpa'],12,70),
  temperature_K=np.insert(prepared['temperature_K'],12,t70),
  specific_humidity_kgkg=np.insert(prepared['specific_humidity_kgkg'],12,q70),
  joint_Tq_valid_fraction=np.insert(prepared['joint_Tq_valid_fraction'],12,f70))

def validate_extraction_metadata(binding,request_metadata,entropy_metadata):
 if binding['driver_sha']!='dcf2479262ce857f7856aa4f23aad138168d39291b00875853eb685ce6921c6e':
  raise ValueError('fixed extraction driver identity mismatch')
 semantics=ROOT/'config/extraction_semantics_v5.json'
 if sha(semantics)!='32833280d34d1a33be629308e32323d0218f3310519bb9e0ec6642b5b9d2c5c7':
  raise ValueError('frozen extraction semantics file changed')
 if binding['semantics_sha']!=digest(read(semantics)):raise ValueError('canonical extraction semantics mismatch')
 if request_metadata['binding']!=binding:raise ValueError('request/COMPLETE binding mismatch')
 e=entropy_metadata
 if e['thermodynamics_sha256']!='76cb2a946408791b02587fd7580d2b1df891bddb27d4f2e38f3b274607f1f420':
  raise ValueError('original native entropy implementation mismatch')
 if e['pressure_pa']!=60000 or e['core_radius_km']!=[0,100] or e['environment_radius_km']!=[100,300]:
  raise ValueError('native entropy pressure/spatial definition mismatch')
 if e.get('original_area_support_threshold')!=.9 or e.get('core_q_required') is not False or e.get('environment_q_required') is not True:
  raise ValueError('native entropy support definition mismatch')

def _scalar(r,unit):
 if r['units']!=unit:raise ValueError('unexpected scalar unit '+r['variable'])
 accepted=boolean(r['support_pass']) and number(r['valid_fraction'])>=.9 and np.isfinite(number(r['value']))
 return dict(value=number(r['value']) if accepted else np.nan,raw_value=number(r['value']),accepted=bool(accepted),units=unit,source=r)

def build_profile(request_folder:Path,verified_source:dict)->dict:
 folder=Path(request_folder).resolve()
 if not folder.is_relative_to((ROOT/'data/environment_extract').resolve()):
  raise ValueError('request folder outside original extract root')
 cp=folder/'COMPLETE.json';csha=sha(cp)
 if not verified_source.get('request_id') or verified_source.get('checkpoint_sha256')!=csha:
  raise ValueError('verified_source must bind request_id and COMPLETE checksum')
 complete=read(cp);r=complete['binding']['request'];normal=complete['binding']['normalised']
 if complete['status']!='COMPLETE' or r['request_id']!=verified_source['request_id']:raise ValueError('source request mismatch')
 if not boolean(normal['need_profile']):raise ValueError('NOT_REQUESTED_PROFILE; caller must keep auxiliary denominator separate')
 outputs={z['name']:z['sha256'] for z in complete['outputs']}
 for name in ('profile.npz','profile.csv','scalars.csv','request.json','entropy600_metadata.json'):
  if sha(folder/name)!=outputs[name]:raise ValueError('extracted output changed '+name)
 validate_extraction_metadata(complete['binding'],read(folder/'request.json'),read(folder/'entropy600_metadata.json'))
 with np.load(folder/'profile.npz',allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
 prepared=prepare13(original)
 prows=rows(folder/'profile.csv')
 if len(prows)!=13:raise ValueError('profile CSV length')
 for line,i in zip(prows,range(13)):
  if line['request_id']!=r['request_id']:raise ValueError('profile CSV request mismatch')
  for col,key in [('pressure_hpa','pressure_hpa'),('temperature_K','temperature_K'),('specific_humidity_kgkg','specific_humidity_kgkg'),('joint_Tq_valid_fraction','joint_Tq_valid_fraction')]:
   if not np.isclose(number(line[col]),original[key][i],rtol=1e-12,atol=1e-12,equal_nan=True):raise ValueError('profile CSV/NPZ disagreement')
 sr=rows(folder/'scalars.csv');scalars={z['variable']:z for z in sr}
 if len(scalars)!=len(sr):raise ValueError('duplicate scalar')
 names={'SST_K':('sst_K','K'),'msl_Pa':('msl','Pa'),'shear_200_850_ms':('shear_200_850_vector_magnitude_ms','m s-1'),
  'core600_saturated_entropy_jkgk':('core600_saturated_entropy_J_kgK','J kg-1 K-1'),
  'environment600_entropy_jkgk':('environment600_entropy_J_kgK','J kg-1 K-1')}
 values={name:_scalar(scalars[key],unit) for name,(key,unit) in names.items()}
 surface_valid=values['SST_K']['accepted'] and values['msl_Pa']['accepted'] and 100<values['SST_K']['value']<400 and 20000<values['msl_Pa']['value']<120000
 result=dict(request_id=r['request_id'],module=r['module'],request=r,normalised=normal,
  source_binding=dict(verified_source=verified_source,COMPLETE=str(cp),checkpoint_sha256=csha,profile_sha256=outputs['profile.npz'],profile_csv_sha256=outputs['profile.csv'],scalars_sha256=outputs['scalars.csv'],request_json_sha256=outputs['request.json'],entropy600_metadata_sha256=outputs['entropy600_metadata.json']),
  original_profile=original,original_profile_rows=prows,original13_preserved=True,
  C_profile=add70(prepared),D_profile=add70(prepared),surface={k:values[k]['value'] for k in ('SST_K','msl_Pa')},
  scalar_values=values,scalars=scalars,named_scalars={**{k:values[k]['value'] for k in ('shear_200_850_ms','core600_saturated_entropy_jkgk','environment600_entropy_jkgk')},
  'ventilation_support_confirmed':bool(all(values[k]['accepted'] for k in ('shear_200_850_ms','core600_saturated_entropy_jkgk','environment600_entropy_jkgk')))},unsupported_original_levels_hpa=prepared['unsupported_levels_hpa'],
  ready_for_PI=False,input_quality_status='PENDING',endpoint_cache_attempted=False,raw_nc_reads=0,PI_computed=False,
  headroom_wind_policy='Caller must retain actual-UTC wind lookup from accepted derivation; P1 fixed request wind_kt is t0 and must not be reused blindly.')
 if not prepared['valid']:
  result.update(input_quality_status='STRICT_REQUIRED_ORIGINAL_PROFILE_UNSUPPORTED_NO_PREFIX_REMOVAL',input_reasons=['missing, out-of-domain, or below90percent original joint support'],geometry_binding=None)
  return result
 if not surface_valid:
  result.update(input_quality_status='SURFACE_SUPPORT_OR_DOMAIN_INVALID',input_reasons=['SST ocean support or MSL annular support/domain invalid'],geometry_binding=None)
  return result
 ctx=_context();vm=ctx['vm'];axes=ctx['axes'];mapper=ctx['mapper'];area=ctx['area'];align=mapper['align_native'];labels=mapper['labels']
 geometry=dict(UTC=ctx['utc'](normal['UTC']).isoformat()+'Z',latitude=float(normal['lat']),longitude=float(normal['lon'])%360,radius_km=800.,halo=1,reader_sha=ctx['native_sha'])
 gid=digest(geometry)[:20];cache=ROOT/'cache/native_fields'/gid
 gm=read(cache/'geometry.json')
 if gm['binding']['geometry']!=geometry or sha(cache/'sp.npz')!=gm['npz_sha256']:raise ValueError('geometry or SP cache mismatch')
 with np.load(cache/'sp.npz',allow_pickle=False) as z:
  if json.loads(str(z['binding_json'].item()))!=gm['binding']:raise ValueError('SP embedded binding mismatch')
  g={k:z[k].copy() for k in z.files if k!='binding_json'}
 ring=(g['distance']>=200)&(g['distance']<800)
 native={};lineage=[];qc={};result['endpoint_cache_attempted']=True
 for key in ('t100','t50','q100','q50'):
  meta=read(cache/(key+'.json'))
  if meta['binding']['geometry']!=gid or meta['binding']['source_identity']!=vm[key]['source_identity'] or meta['binding']['metadata_sha']!=vm[key]['metadata_sha256']:raise ValueError('endpoint cache binding mismatch '+key)
  if sha(cache/(key+'.npz'))!=meta['npz_sha256']:raise ValueError('endpoint cache checksum mismatch '+key)
  with np.load(cache/(key+'.npz'),allow_pickle=False) as z:
   if json.loads(str(z['binding_json'].item()))!=meta['binding']:raise ValueError('endpoint embedded binding mismatch')
   raw=z['raw'].copy();a=z['values'].copy();mask=z['support_mask'].copy()
  sy=axes[key+'__latitude'][g['latitude_indices']];sx=axes[key+'__longitude'][g['longitude_indices']]
  if not np.array_equal(labels(g['latitude']),labels(sy)) or not np.array_equal(labels(g['longitude'],True),labels(sx,True)):raise ValueError('source native index coordinate mismatch')
  a=align(a,sy,sx,g['latitude'],g['longitude']);raw=align(raw,sy,sx,g['latitude'],g['longitude']);mask=align(mask,sy,sx,g['latitude'],g['longitude']).astype(bool)
  expected=np.isfinite(raw)&np.isfinite(g['sp'])&(g['sp']>=100*vm[key]['pressure_level_hpa'])
  if not np.array_equal(mask,expected) or not np.array_equal(a,np.where(expected,raw,np.nan),equal_nan=True):raise ValueError('SP support/value mismatch')
  physical=np.isfinite(a)&(a>100)&(a<400) if key.startswith('t') else np.isfinite(a)&(a>=0)&(a<.1)
  native[key]=np.where(physical,a,np.nan)
  qc[key]=dict(missing_or_SPmasked_in_annulus=int((ring&~np.isfinite(a)).sum()),finite_outside_strict_range_in_annulus=int((ring&np.isfinite(a)&~physical).sum()),physical_valid_in_annulus=int((ring&physical).sum()))
  lineage.append(dict(key=key,path=str(cache/(key+'.npz')),sha256=meta['npz_sha256'],binding=meta['binding']))
 tl,ql,joint=ctx['inter']['joint70'](native['t100'],native['t50'],native['q100'],native['q50'],'logp')
 tp,qp,jp=ctx['inter']['joint70'](native['t100'],native['t50'],native['q100'],native['q50'],'linear_p')
 if not np.array_equal(joint,jp):raise ValueError('C/D support differs')
 stats={name:area(a,g['area'],ring,min_fraction=.9) for name,a in [('T_logp_K',tl),('q_logp_kgkg',ql),('T_plinear_K',tp),('q_plinear_kgkg',qp)]}
 if len({x['valid_area_m2'] for x in stats.values()})!=1:raise ValueError('derived70 joint area differs')
 f=stats['T_logp_K']['valid_fraction']
 result['C_profile']=add70(prepared,stats['T_logp_K']['mean'],stats['q_logp_kgkg']['mean'],f)
 result['D_profile']=add70(prepared,stats['T_plinear_K']['mean'],stats['q_plinear_kgkg']['mean'],f)
 ready=all(x['support_pass'] and np.isfinite(x['mean']) for x in stats.values())
 result.update(ready_for_PI=bool(ready),input_quality_status='SUPPORTED_INPUT_NOT_PHYSICAL_PI_ACCEPTANCE' if ready else 'DERIVED70_JOINT_AREA_SUPPORT_INSUFFICIENT',
  input_reasons=[] if ready else ['four endpoint common native support below90percent original annulus'],
  geometry_binding=dict(geometry_id=gid,geometry=geometry,metadata_sha256=sha(cache/'geometry.json'),sp_npz_sha256=gm['npz_sha256']),
  input_dependency_binding=ctx['input_binding'],source_cache=lineage,endpoint_QC=qc,
  derived70_support=stats,derived70_identity='estimated variable values on native grid; NOT native ERA5 70hPa',
  coordinate_policy=dict(source_native_indices_verified=True,roundoff_bound_deg=float(mapper['ROUND_OFF_DEG']),nearest_or_regridding=False))
 return result
