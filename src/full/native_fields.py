"""Bounded, read-only ERA5 pressure-level extraction and strict interpolation."""
from pathlib import Path
from collections import OrderedDict
from datetime import datetime,timezone
import numpy as np
from netCDF4 import Dataset,date2num

import os
ROOT=Path(os.environ.get('FLTCCF_ERA5_ROOT', 'raw/era5'))
PREFIX={'q':'Specific-humidity','u':'U-component-of-wind','v':'V-component-of-wind',
        'omega':'Vertical-velocity','T':'Temperature','sp':'Surface-pressure'}

def path_for(field,level=None):
    name=PREFIX[field]+('' if level is None else f'-{int(level)}hpa')+'-1980-2025.nc'
    return ROOT/name

def coord_name(ds,candidates):
    for n in candidates:
        if n in ds.variables:return n
    raise ValueError(('Missing coordinate',candidates,str(ds.filepath())))

class NativeReader:
    def __init__(self,max_open=24):
        self.max_open=max_open;self.handles=OrderedDict();self.identities={};self.meta={}
    def close(self):
        for ds in self.handles.values():ds.close()
        self.handles.clear()
    def dataset(self,path):
        path=str(path)
        if path in self.handles:
            ds=self.handles.pop(path);self.handles[path]=ds;return ds
        ds=Dataset(path,'r');self.handles[path]=ds
        st=Path(path).stat()
        self.identities[path]={'path':path,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'access':'read-only; no full-file hash'}
        if len(self.handles)>self.max_open:self.handles.popitem(last=False)[1].close()
        return ds
    def read_window(self,field,level,utc,lat_bounds,lon_ref,lon_bounds):
        path=path_for(field,level);ds=self.dataset(path)
        if str(path) not in self.meta:
            yn=coord_name(ds,['latitude','lat']);xn=coord_name(ds,['longitude','lon']);tn=coord_name(ds,['valid_time','time'])
            names=[n for n,var in ds.variables.items() if yn in var.dimensions and xn in var.dimensions and tn in var.dimensions]
            if len(names)!=1:raise ValueError(('Ambiguous physical variable',names,str(path)))
            self.meta[str(path)]={'yn':yn,'xn':xn,'tn':tn,'vn':names[0],
                'yy':np.asarray(ds[yn][:]),'xx':np.asarray(ds[xn][:]),'times':np.asarray(ds[tn][:])}
        meta=self.meta[str(path)];yn=meta['yn'];xn=meta['xn'];tn=meta['tn'];vn=meta['vn']
        yy=meta['yy'];xx=meta['xx'];tv=ds[tn];var=ds[vn]
        # UTC timestamps must coincide with native endpoints; no nearest-time substitution.
        dt=utc.to_pydatetime() if hasattr(utc,'to_pydatetime') else utc
        if dt.tzinfo is not None:dt=dt.astimezone(timezone.utc).replace(tzinfo=None)
        val=date2num(dt,tv.units,calendar=getattr(tv,'calendar','standard'))
        times=meta['times'];ti=int(np.searchsorted(times,val))
        if ti>=len(times) or abs(float(times[ti])-float(val))>1e-5:raise ValueError(('UTC not present',str(utc),str(path)))
        unwrapped=lon_ref+(xx-lon_ref+180)%360-180
        yi=np.flatnonzero((yy>=lat_bounds[0])&(yy<=lat_bounds[1]));xi=np.flatnonzero((unwrapped>=lon_bounds[0])&(unwrapped<=lon_bounds[1]))
        if len(yi)<3 or len(xi)<3:raise ValueError('Window has insufficient interpolation halo')
        yi=yi[np.argsort(yy[yi])];xi=xi[np.argsort(unwrapped[xi])]
        index=[]
        for dim in var.dimensions:
            if dim==tn:index.append(ti)
            elif dim==yn:index.append(yi)
            elif dim==xn:index.append(xi)
            elif dim in ['pressure_level','level']:
                levels=np.asarray(ds[dim][:]);li=np.flatnonzero(np.isclose(levels,level))
                if len(li)!=1:raise ValueError(('Pressure level mismatch',str(path),levels.tolist(),level))
                index.append(int(li[0]))
            elif len(ds.dimensions[dim])==1:index.append(0)
            else:raise ValueError(('Unhandled non-singleton dimension',dim,str(path)))
        data=var[tuple(index)]
        data=np.asarray(np.ma.filled(data,np.nan),dtype=float)
        dims=[d for d in var.dimensions if d in [yn,xn]]
        if dims==[xn,yn]:data=data.T
        assert data.shape==(len(yi),len(xi))
        return Window(yy[yi],unwrapped[xi],data,lon_ref),{'path':str(path),'variable':vn,'units':getattr(var,'units',None),'native_time_index':ti,'lat_index_bounds':[int(yi.min()),int(yi.max())],'lon_indices_count':len(xi)}

class Window:
    def __init__(self,lat,lon,data,lon_ref):
        self.lat=np.asarray(lat);self.lon=np.asarray(lon);self.data=np.asarray(data);self.lon_ref=lon_ref
    def sample(self,lat,lon,return_corner_min=False):
        lat=np.asarray(lat);lon=self.lon_ref+(np.asarray(lon)-self.lon_ref+180)%360-180
        iy=np.searchsorted(self.lat,lat,side='right')-1;ix=np.searchsorted(self.lon,lon,side='right')-1
        valid=(iy>=0)&(iy<len(self.lat)-1)&(ix>=0)&(ix<len(self.lon)-1)
        iy=np.clip(iy,0,len(self.lat)-2);ix=np.clip(ix,0,len(self.lon)-2)
        fy=(lat-self.lat[iy])/(self.lat[iy+1]-self.lat[iy]);fx=(lon-self.lon[ix])/(self.lon[ix+1]-self.lon[ix])
        c=np.stack([self.data[iy,ix],self.data[iy,ix+1],self.data[iy+1,ix],self.data[iy+1,ix+1]])
        valid=valid&np.isfinite(c).all(axis=0)
        out=(1-fy)*((1-fx)*c[0]+fx*c[1])+fy*((1-fx)*c[2]+fx*c[3])
        out=np.where(valid,out,np.nan)
        if return_corner_min:return out,np.where(valid,np.min(c,axis=0),np.nan)
        return out

    def sample_gradient(self,lat,lon):
        """Bilinear interpolant derivatives per radian, independent area diagnostic."""
        lat=np.asarray(lat);lon=self.lon_ref+(np.asarray(lon)-self.lon_ref+180)%360-180
        iy=np.searchsorted(self.lat,lat,side='right')-1;ix=np.searchsorted(self.lon,lon,side='right')-1
        valid=(iy>=0)&(iy<len(self.lat)-1)&(ix>=0)&(ix<len(self.lon)-1)
        iy=np.clip(iy,0,len(self.lat)-2);ix=np.clip(ix,0,len(self.lon)-2)
        dy=self.lat[iy+1]-self.lat[iy];dx=self.lon[ix+1]-self.lon[ix]
        fy=(lat-self.lat[iy])/dy;fx=(lon-self.lon[ix])/dx
        c=np.stack([self.data[iy,ix],self.data[iy,ix+1],self.data[iy+1,ix],self.data[iy+1,ix+1]])
        valid &= np.isfinite(c).all(axis=0)
        gy=((1-fx)*(c[2]-c[0])+fx*(c[3]-c[1]))/np.radians(dy)
        gx=((1-fy)*(c[1]-c[0])+fy*(c[3]-c[2]))/np.radians(dx)
        return np.where(valid,gy,np.nan),np.where(valid,gx,np.nan)
