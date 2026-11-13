def read_era5_fixed_block(path, variable, start_utc, n_time=512, lat_bounds=(-25., -10.), lon_bounds=(60., 80.), lon_reference=70., time_step_seconds=21600):
    """Read one raw fixed-geographic ERA5 field; retain missing values and native grid.

    Longitude bounds are expressed continuously around lon_reference. Pressure remains
    a file-level property. The return array is (time, ascending latitude, unwrapped
    longitude). No interpolation, geographic resampling or terrain filling is applied.
    """
    import datetime
    import numpy as np
    import h5py
    start=np.datetime64(str(start_utc).replace('Z','').replace('+00:00',''),'s')
    with h5py.File(path,'r') as f:
        tname='time' if 'time' in f else 'valid_time'
        yname='latitude' if 'latitude' in f else 'lat'
        xname='longitude' if 'longitude' in f else 'lon'
        times=f[tname][:]
        units=f[tname].attrs['units'];units=units.decode() if isinstance(units,bytes) else str(units)
        unit,origin=units.split(' since ')
        factor={'seconds':1,'hours':3600,'days':86400}[unit]
        origin=np.datetime64(origin.replace(' ','T'),'s')
        seconds=times.astype(np.float64)*factor
        offset=float((start-origin)/np.timedelta64(1,'s'))
        it=int(np.searchsorted(seconds,offset))
        if it+n_time>len(times) or it==len(times) or seconds[it]!=offset:
            raise ValueError('Requested UTC interval is outside the exact native grid')
        selected=seconds[it:it+n_time]
        if not np.all(np.diff(selected)==time_step_seconds):
            raise ValueError('Duplicate, missing or unexpected time steps')
        lat=np.asarray(f[yname][:]);lon=np.asarray(f[xname][:])
        lon_unwrapped=lon_reference+(lon-lon_reference+180)%360-180
        yi=np.flatnonzero((lat>=min(lat_bounds))&(lat<=max(lat_bounds)))
        xi=np.flatnonzero((lon_unwrapped>=min(lon_bounds))&(lon_unwrapped<=max(lon_bounds)))
        xi=xi[np.argsort(lon_unwrapped[xi])]
        if len(yi)<3 or len(xi)<3:raise ValueError('Domain too small for horizontal derivatives')
        var=f[variable]
        if var.shape!=(len(times),len(lat),len(lon)):raise ValueError('Unexpected dimension order')
        yslice=slice(int(yi.min()),int(yi.max())+1)
        segments=np.split(xi,np.flatnonzero(np.diff(xi)!=1)+1)
        parts=[np.asarray(var[it:it+n_time,yslice,int(seg[0]):int(seg[-1])+1],dtype=np.float64) for seg in segments]
        data=np.concatenate(parts,axis=2)
        lat_out=lat[yslice]
        if lat_out[0]>lat_out[-1]:data=data[:,::-1,:];lat_out=lat_out[::-1]
        for key in ['_FillValue','missing_value']:
            if key in var.attrs:
                for bad in np.asarray(var.attrs[key]).ravel():
                    if np.isfinite(bad):data[data==bad]=np.nan
        if 'scale_factor' in var.attrs:data*=float(np.asarray(var.attrs['scale_factor']).ravel()[0])
        if 'add_offset' in var.attrs:data+=float(np.asarray(var.attrs['add_offset']).ravel()[0])
        physical_units=var.attrs.get('units','');physical_units=physical_units.decode() if isinstance(physical_units,bytes) else str(physical_units)
    utc=origin+selected.astype('timedelta64[s]')
    return {'field':data,'latitude':lat_out,'longitude':lon_unwrapped[xi],
            'UTC':utc,'units':physical_units,'time_step_seconds':int(time_step_seconds),
            'source_path':str(path),'source_variable':variable,'native_time_start_index':it}
