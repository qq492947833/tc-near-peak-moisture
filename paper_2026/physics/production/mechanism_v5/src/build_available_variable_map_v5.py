from _release_paths import release_path as _release_path
from pathlib import Path
import json
ROOT=Path(str(_release_path('mechanism_v5','')))
M=json.loads((ROOT/"audit"/"raw_metadata_v5.json").read_text(encoding="utf-8"))
m={}
for d in M:
 r=d["row"];v=r["variable"];level=r["level_hpa"]
 key=f"{v}{int(float(level))}" if level!="" else (f"{v}_shear_{200 if '200hpa-850hpa' in r['filename'] else 500}_{850 if '200hpa-850hpa' in r['filename'] else 700}" if r["derivation"] else v)
 a=d["variables"][v]
 m[key]=dict(path=r["path"],variable=v,pressure_level_hpa=float(level) if level!="" else None,time_variable=d["time"].get("name"),time_units=d["variables"].get(d["time"].get("name"),{}).get("attrs",{}).get("units"),calendar=d["variables"].get(d["time"].get("name"),{}).get("attrs",{}).get("calendar"),dimensions=a["dimensions"],shape=a["shape"],units=r["units"],latitude_variable="latitude",longitude_variable="longitude",latitude_order="descending",longitude_range="0..359.75",latitude_resolution_deg=.25,longitude_resolution_deg=.25,original_grid=(721,1440),dtype=a["dtype"],chunks=a["chunking"],step_type=r["step_type"],source_level_mask="sp>=100*pressure_level_hpa" if level!="" else "surface:no pressure-level mask",role=r["processing_note"],metadata_sha256=d["metadata_sha256"],source_identity=d["identity"])
out=dict(version="v5.1_same_inputs_P1_P5",mandatory_modules=["P1","P2","P3","P4","P5"],not_a_scientific_result=True,paths=m,raw_time_index_formula="ERA5 time normalized to six-hour index since1980-01-01T00:00Z; verify exact UTC conversion and identity againststoredcoords",static_lsm_note="single2025-12-31T00:00UTCsupport is static and broadcastspatially; do not request eventUTC fromits length1timeaxis",conversions={"z_to_height_m":"z/9.80665","q_to_mixing_ratio_kgkg":"q/(1-q)","t_K_to_C":"t-273.15","tp_to_sampled_hourly_rate_mm_h":"tp*1000 (preceding1h)","slhf_sshf_to_upward_W_m2":"-accumulated_J_m2/3600"})
(ROOT/"config"/"available_variable_map_v5.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(list(m))
