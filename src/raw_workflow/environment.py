"""Bounded original-grid pressure-level annular extraction for one exact request."""
from pathlib import Path
import argparse
import json
import numpy as np
from .native_grid import NativeFieldReader,area_weighted_support

ROOT=Path(__file__).absolute().parents[2]


def variable_map(era5_root,config=None):
    cfg=json.loads(Path(config or ROOT/'config/raw_inputs.json').read_text(encoding='utf-8'))
    fields={k:dict(v,path=str(Path(era5_root)/v['file'])) for k,v in cfg['era5']['fields'].items()}
    return cfg,{'paths':fields}


def extract(era5_root,utc,lat,lon,fields,out,config=None):
    """Apply simultaneous SP support and the original 200–800 km area denominator.

    This entry returns primitive pressure-level fields. It does not substitute
    an atmospheric mask for the separately specified SST or ocean-flux masks.
    """
    cfg,mapping=variable_map(era5_root,config)
    fields=list(dict.fromkeys(fields))
    for field in fields:
        if field not in mapping['paths'] or mapping['paths'][field]['pressure_level_hpa'] is None:
            raise ValueError('This entry requires a listed pressure-level field: '+field)
    inner,outer=cfg['era5']['scalar_annulus_km'];fraction=cfg['era5']['minimum_original_area_fraction']
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=[];arrays={}
    with NativeFieldReader(mapping,max_open=4,chunk_cache_bytes=2*1024**2) as reader:
        ref=reader.read_box('sp',utc,lat,lon,outer,halo=1,apply_sp=False)
        domain=(ref.distance_km>=inner)&(ref.distance_km<outer)
        arrays.update(latitude=ref.latitude,longitude=ref.longitude,cell_area_m2=ref.cell_area_m2,
                      distance_km=ref.distance_km,original_domain=domain,surface_pressure_Pa=ref.raw)
        for field in fields:
            block=reader.read_aligned(field,utc,ref,apply_sp=True)
            stats=area_weighted_support(block.values,ref.cell_area_m2,domain,min_fraction=fraction)
            rows.append({'field':field,'units':block.units,'pressure_level_hpa':block.pressure_level_pa/100,
                         'UTC':block.utc,'latitude':float(lat),'longitude':float(lon),**stats})
            arrays[field]=block.values;arrays[field+'_support']=block.support_mask
        read_operations=list(reader.read_log)
    np.savez_compressed(out/'native_annulus.npz',**arrays)
    payload={'requests':rows,'read_operations':read_operations,
             'spatial_object':'native cell annulus, not interpolated boundary nodes',
             'time_interpolation':False,'new_fit':False}
    (out/'native_annulus.json').write_text(json.dumps(payload,indent=2,allow_nan=True)+'\n',encoding='utf-8')
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--era5-root',required=True);p.add_argument('--utc',required=True)
    p.add_argument('--lat',required=True,type=float);p.add_argument('--lon',required=True,type=float)
    p.add_argument('--field',action='append',required=True);p.add_argument('--out',required=True)
    p.add_argument('--config');a=p.parse_args()
    print(json.dumps(extract(a.era5_root,a.utc,a.lat,a.lon,a.field,a.out,a.config)['requests'],indent=2))
