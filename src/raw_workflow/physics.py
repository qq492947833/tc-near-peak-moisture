"""Explicit bounded entry points for humidity, PI and 700-hPa diagnostics.

Input/output folders are caller supplied. The commands do not launch cohorts,
model fits, downloads or bootstrap resampling. See RAW_INPUTS.md for scope.
"""
from collections import OrderedDict
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .environment import variable_map
from .native_grid import NativeFieldReader


class NativeRequests:
    """Small read-only geometry/field cache, usable by the published functions."""
    def __init__(self, reader, max_geometries=4, max_fields=16):
        self.reader=reader
        self.max_geometries=max_geometries
        self.max_fields=max_fields
        self.geometries=OrderedDict()
        self.fields=OrderedDict()

    @staticmethod
    def key(utc,lat,lon):
        return (pd.Timestamp(utc).isoformat(),float(lat),float(lon)%360.)

    def reference(self,utc,lat,lon,radius_km=800.):
        if radius_km!=800.:
            raise ValueError('These diagnostics use the fixed 800-km bounding box')
        key=self.key(utc,lat,lon)
        if key not in self.geometries:
            self.geometries[key]=self.reader.read_box('sp',utc,lat,lon,800.,halo=1,apply_sp=False)
            if len(self.geometries)>self.max_geometries:
                self.geometries.popitem(last=False)
        self.geometries.move_to_end(key)
        return self.geometries[key]

    def get(self,field,utc,lat,lon):
        key=(field,)+self.key(utc,lat,lon)
        if key not in self.fields:
            self.fields[key]=self.reader.read_aligned(field,utc,self.reference(utc,lat,lon),apply_sp=True)
            if len(self.fields)>self.max_fields:
                self.fields.popitem(last=False)
        self.fields.move_to_end(key)
        return self.fields[key]


def serializable(value):
    if isinstance(value,dict):
        return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [serializable(v) for v in value]
    if isinstance(value,np.generic):
        return serializable(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def rh_request(era5_root,request,config=None):
    """Compute the two levels at the same two UTCs and two positions.

    request keys: sid, boundary, UTC0, UTC1, lat0, lon0, lat1, lon1. Input
    dates must be 24 h apart, representing -24 to 0 h for the defined boundary.
    """
    from .humidity import pair
    if (pd.Timestamp(request['UTC1'])-pd.Timestamp(request['UTC0'])).total_seconds()!=86400:
        raise ValueError('Exactly the prespecified -24 to 0 h window is required')
    _,mapping=variable_map(era5_root,config)
    with NativeFieldReader(mapping,max_open=5,chunk_cache_bytes=2*1024**2) as reader:
        return pair(request,NativeRequests(reader))


def pi_request(profile_npz,scalars,wind_kt=np.nan):
    """Evaluate an explicitly prepared original+augmented environmental profile.

    The NPZ schema is documented in RAW_INPUTS.md. Native-cell SP, SST ocean mask,
    original area support and joint 70-hPa averaging are prerequisites; this API
    does not treat a mean SP as a replacement for grid-cell pressure masking.
    """
    from .potential_intensity import evaluate
    with np.load(profile_npz,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    return evaluate(arrays,scalars,wind_kt)


def budget_request(era5_root,request,config=None):
    """One real 6-hour time, both radial domains, with a supplied prior position.

    Required keys: UTC, lat, lon, lag_h, motion_u_ms, motion_v_ms,
    prior_lat, prior_lon. The prior position is the actual UTC-6 h position.
    Calls for 0,6,12,18,24 h and both events must be integrated together using
    budget_nodes.integrate_pair, which enforces common nodes across all fields.
    """
    from .budget_nodes import calculate
    if request['lag_h'] not in (0,6,12,18,24):
        raise ValueError('A five-time 0--24 h window is required')
    _,mapping=variable_map(era5_root,config)
    with NativeFieldReader(mapping,max_open=7,chunk_cache_bytes=2*1024**2) as reader:
        return calculate(request,NativeRequests(reader),(request['prior_lat'],request['prior_lon']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['rh','budget','pi'])
    p.add_argument('--input',required=True,help='JSON request; PI uses scalar fields')
    p.add_argument('--out',required=True,help='New output directory')
    p.add_argument('--era5-root');p.add_argument('--config')
    p.add_argument('--profile',help='Prepared original+augmented PI NPZ')
    p.add_argument('--wind-kt',type=float,default=np.nan)
    a=p.parse_args();request=json.loads(Path(a.input).read_text(encoding='utf-8'))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if a.mode=='pi':
        if not a.profile: p.error('--profile is required for PI')
        value=pi_request(a.profile,request,a.wind_kt)
    else:
        if not a.era5_root: p.error('--era5-root is required for field calculations')
        value=(rh_request if a.mode=='rh' else budget_request)(a.era5_root,request,a.config)
    (out/(a.mode+'.json')).write_text(json.dumps(serializable(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Calculated '+a.mode+' for the supplied request')


if __name__=='__main__':
    main()
