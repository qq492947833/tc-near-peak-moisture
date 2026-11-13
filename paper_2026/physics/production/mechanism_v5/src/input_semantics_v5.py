from _release_paths import release_path as _release_path
from pathlib import Path
import csv,json,sys,re,hashlib,os,traceback
from datetime import datetime,timezone
import numpy as np,netCDF4
ROOT=Path(str(_release_path('mechanism_v5',''))); BASE=Path(str(_release_path('raw_era5','')))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding="utf-8")
meta=json.loads((ROOT/"audit"/"raw_metadata_v5.json").read_text(encoding="utf-8")); by={d["row"]["filename"]:d for d in meta}
# These decoded values were read afresh from all source files in the immediately preceding audit.
comparisons=[]
for wind in "UV":
 for hi,lo in [(200,850),(500,700)]:
  names=[f"{wind}-component-of-wind-{hi}hpa-{lo}hpa-1980-2025.nc",f"{wind}-component-of-wind-{hi}hpa-1980-2025.nc",f"{wind}-component-of-wind-{lo}hpa-1980-2025.nc"]
  dat=[by[n] for n in names]
  row=dict(derived=str(BASE/names[0]),upper=str(BASE/names[1]),lower=str(BASE/names[2]),formula=f"{wind.lower()}{hi}-{wind.lower()}{lo}",same_time=all(d["time"]==dat[0]["time"] for d in dat),same_coordinates=all(d["coordinates"]==dat[0]["coordinates"] for d in dat))
  errors=[];mask_errors=0;count=0
  for a,b,c in zip(*(d["bounded_samples"] for d in dat)):
   v,u,l=[np.asarray(s["decoded_values"],dtype=float) for s in (a,b,c)]
   wanted=u-l;good=np.isfinite(v)&np.isfinite(wanted);mask_errors+=int(np.count_nonzero(np.isfinite(v)!=np.isfinite(wanted)));count+=int(good.sum());errors.extend((v[good]-wanted[good]).tolist())
  mx=max(abs(x) for x in errors);row.update(n_compared=count,max_abs_error=mx,mask_differences=mask_errors,passed=row["same_time"] and row["same_coordinates"] and mx<=1e-10 and mask_errors==0);comparisons.append(row)
dump(ROOT/"audit"/"derived_wind_source_consistency_v5.json",comparisons)
# Unit/level audit, product annotations, exact semantic expectations.
expect={"t":("K","air_temperature"),"q":("kg kg**-1","specific_humidity"),"r":("%","relative_humidity"),"u":("m s-1",None),"v":("m s-1",None),"z":("m**2 s**-2",None),"w":("Pa s**-1",None),"sst":("K",None),"sp":("Pa","surface_air_pressure"),"msl":("Pa",None),"tp":("m",None)}
unitrows=[]
for d in meta:
 vn=d["row"]["variable"];v=d["variables"][vn];a=v["attrs"];match=re.search(r"-(\d+)hpa-1980",d["row"]["filename"])
 level_expected=int(match[1]) if match and not d["row"]["derivation"] else None
 level_ok=level_expected is None or float(a.get("pressure_level_hpa",float("nan")))==level_expected
 unit=a.get("units",""); aliases={"m s-1":{"m s-1","m s**-1"},"kg kg**-1":{"kg kg**-1","kg kg-1","1"},"m**2 s**-2":{"m**2 s**-2","m2 s-2","m^2 s^-2"},"Pa s**-1":{"Pa s**-1","Pa s-1"}}
 allowed=aliases.get(expect.get(vn,(unit,None))[0],{expect.get(vn,(unit,None))[0]})
 unitrows.append(dict(filename=d["row"]["filename"],variable=vn,units=unit,units_pass=unit in allowed,level_hpa=a.get("pressure_level_hpa"),level_expected=level_expected,level_pass=level_ok,step_type=a.get("GRIB_stepType"),data_type=a.get("GRIB_dataType"),legacy_standard_name=a.get("standard_name")))
dump(ROOT/"audit"/"units_levels_products_v5.json",unitrows)
# Optional ORAS5 status: latest manifest and actual annual files override stale failure report.
logs=BASE/"_oras5_ocean_heat_content_for_the_upper_300m_logs"; folder=BASE/"_oras5_ocean_heat_content_for_the_upper_300m_downloads"
manifest=json.loads((logs/"manifest.json").read_text(encoding="utf-8")); yearfiles=sorted(p for p in folder.glob("*.nc") if re.fullmatch(r"ocean_heat_content_for_the_upper_300m-\d{4}\.nc",p.name))
oras=dict(optional=True,required_for_main_v5=False,manifest_path=str(logs/"manifest.json"),manifest_top_fields={k:v for k,v in manifest.items() if k not in ("years","requests","temp_files","part_files","deleted")},year_statuses={y:{k:v for k,v in r.items() if k in ("status","finished","last_error","validated","path")} for y,r in manifest.get("years",{}).items()},annual_files=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in yearfiles],partials=[p.name for p in folder.glob("*.part")],note="old report.json remains failed for1980; latest manifest and readable annual files independently checked; no restart or download",actual_metadata=[])
for p in yearfiles:
 with netCDF4.Dataset(p,"r") as ds:
  vs=[k for k,v in ds.variables.items() if k.startswith("sohtc")]
  timevar=next(k for k in ("time_counter","time","valid_time") if k in ds.variables)
  t=ds[timevar];dt=netCDF4.num2date(t[:],t.units,getattr(t,"calendar","standard"))
  oras["actual_metadata"].append(dict(filename=p.name,variable=vs,units=[getattr(ds[v],"units",None) for v in vs],shape=[list(ds[v].shape) for v in vs],time_count=len(dt),time_start=str(dt[0]),time_end=str(dt[-1]),file_format=ds.file_format))
oras["status"]="PARTIAL_OPTIONAL_NOT_USED"
dump(ROOT/"audit"/"oras5_existence_progress_v5.json",oras)
# No snapshot production diagnostics were restarted.
sp_src=ROOT.parent/"review_completion_v3_2"/"sources"/"canonical_source_validation.json"
sp_validate=json.loads(sp_src.read_text(encoding="utf-8-sig"))
dump(ROOT/"audit"/"authoritative_sp_source_binding_v5.json",dict(source=str(sp_src),sha256=hashlib.sha256(sp_src.read_bytes()).hexdigest(),content=sp_validate,rule="v5 pressure field extraction masks p*100 > sp at identical UTC/gridpoint; retain original spatial area denominator; no use of msl or storm-centre pressure as substitute"))
print(json.dumps(dict(wind_comparisons=comparisons,units_issues=[x for x in unitrows if not(x["units_pass"] and x["level_pass"])],oras_years=[p.name for p in yearfiles],oras_manifest_status=manifest.get("status")),ensure_ascii=False))
