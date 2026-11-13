def MWT_Prepare_Direct(u, v, temperature, lon, lat, pressure, omega=None,
                       omega_pressure=(50000., 70000., 85000.), geopotential=None,
                       temperature_mean=None, geopotential_reference=None,
                       valid_mask=None, surface_pressure=None, surface_margin_pa=0.,
                       pressure_unit='Pa', include_vertical_gradients=True):
    """Prepare direct MWT diagnostics without inventing missing omega levels.

    All four-dimensional fields use (time, pressure, lat, lon), SI units.
    u/v/T/Phi retain all supplied pressure levels; omega retains its own levels.
    Surface pressure has shape (time,lat,lon), Pa. A column is valid only if
    all times are finite and above ground. Invalid columns become NaN, never 0.
    valid_mask optionally further restricts MWT fixed support (pressure,lat,lon);
    it does not restrict instantaneous reference averaging support.
    A supplied temperature_mean and geopotential_reference must be the SAME
    fixed profiles for both events of a SID. Otherwise construct one area/time
    mean from this input; callers must pass a common long input for the SID.
    Reference averaging uses each time's above-ground valid area; it does not
    require an entire pressure layer to have a permanently valid MWT column.
    reference_weight_fraction records the actual space-time area coverage.
    APE uses the 13-level reference derivatives before any omega-level selection.
    Returns data,basic_state; pass data to unchanged MWT with physical norm and
    nan_policy='propagate'. Functions are independent, NumPy-only, no file I/O.
    """
    import numpy as np
    def real(x, name):
        if np.iscomplexobj(x): raise TypeError(name+' must be real')
        a=np.asarray(np.ma.asarray(x,dtype=float).filled(np.nan))
        if np.isinf(a).any(): raise ValueError(name+' has infinity')
        return a
    def coord(x,name,n):
        a=real(x,name)
        if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError(name+' coordinate shape/finite')
        return a
    def derivative(x,c,axis):
        return np.gradient(x,c,axis=axis,edge_order=2 if len(c)>=3 else 1)
    arrays={k:real(x,k) for k,x in [('u',u),('v',v),('temperature',temperature)]}
    shape=arrays['u'].shape
    if len(shape)!=4 or any(a.shape!=shape for a in arrays.values()): raise ValueError('u/v/T shape mismatch')
    if shape[0]<4 or shape[0]&(shape[0]-1): raise ValueError('time length must be a power of two')
    if min(shape[1:])<3: raise ValueError('at least 3 pressure/lat/lon points required')
    lon=coord(lon,'lon',shape[3]);lat=coord(lat,'lat',shape[2]);p=coord(pressure,'pressure',shape[1])
    if pressure_unit=='hPa': p=p*100
    elif pressure_unit!='Pa': raise ValueError('pressure_unit must be Pa or hPa')
    lr=np.unwrap(np.deg2rad(lon));br=np.deg2rad(lat)
    for x in (lr,br,p):
        if not (np.all(np.diff(x)>0) or np.all(np.diff(x)<0)): raise ValueError('coordinates must be strictly monotonic')
    if np.ptp(lr)>=2*np.pi-1e-10 or np.any(np.abs(lat)>=90) or np.any(p<=0): raise ValueError('invalid grid/pressure')
    if np.any(arrays['temperature'][np.isfinite(arrays['temperature'])]<=0): raise ValueError('T must be Kelvin')
    fixed=np.logical_and.reduce([np.isfinite(a).all(axis=0) for a in arrays.values()])
    reference_valid=np.ones(shape,dtype=bool)
    if geopotential is not None:
        phi=real(geopotential,'geopotential')
        if phi.shape!=shape: raise ValueError('geopotential shape mismatch')
        fixed &= np.isfinite(phi).all(axis=0)
    if surface_pressure is not None:
        sp=real(surface_pressure,'surface_pressure')
        if sp.shape!=(shape[0],shape[2],shape[3]): raise ValueError('surface_pressure shape mismatch')
        above=(p[None,:,None,None]+float(surface_margin_pa) < sp[:,None,:,:]) & np.isfinite(sp[:,None,:,:])
        fixed &= above.all(axis=0)
        reference_valid &= above
    if valid_mask is not None:
        vm=np.asarray(valid_mask)
        if vm.shape!=shape[1:] or vm.dtype!=np.bool_: raise ValueError('valid_mask must be fixed pressure/lat/lon bool')
        fixed &= vm
    def edges(c):
        return np.r_[c[0]-(c[1]-c[0])/2,(c[1:]+c[:-1])/2,c[-1]+(c[-1]-c[-2])/2]
    area=np.abs(np.diff(np.sin(np.clip(edges(br),-np.pi/2,np.pi/2))))[:,None]*np.abs(np.diff(edges(lr)))[None,:]
    def mean_profile(a):
        valid=reference_valid & np.isfinite(a)
        weights=area[None,None,:,:]*valid
        denominator=weights.sum(axis=(0,2,3))
        # Invalid cells are excluded from weighted sums, never sent as zeros to MWT.
        numerator=np.sum(np.where(valid,a,0.)*weights,axis=(0,2,3))
        means=np.divide(numerator,denominator,out=np.full(len(p),np.nan),where=denominator>0)
        complete=valid.all(axis=(0,2,3))
        # Preserve the supplied function's summation order on complete levels.
        means[complete]=np.sum(a[:,complete].mean(axis=0)*area,axis=(-2,-1))/area.sum()
        return means
    tbar=mean_profile(arrays['temperature']) if temperature_mean is None else real(temperature_mean,'temperature_mean')
    if tbar.shape!=(len(p),) or np.count_nonzero(np.isfinite(tbar))<3 or np.any(tbar[np.isfinite(tbar)]<=0): raise ValueError('fewer than three valid reference levels or invalid Kelvin values')
    gas,cp,g=287.058,1005.,9.8
    dtbar=derivative(tbar,p,0)
    lapse=g*p/(gas*tbar)*dtbar
    stability=g/cp-lapse
    valid_stability=stability>0
    c=np.divide(g,tbar*stability,out=np.full_like(tbar,np.nan),where=valid_stability)
    dc=derivative(c,p,0)
    data={k:np.where(fixed[None,:,:,:],arrays[k],np.nan) for k in ('u','v')}
    data['t']=np.where(fixed[None,:,:,:],arrays['temperature']-tbar[None,:,None,None],np.nan)
    for name,l,r in [('uu','u','u'),('uv','u','v'),('vv','v','v'),('u_t','u','t'),('v_t','v','t')]: data[name]=data[l]*data[r]
    phi_ref=None
    if geopotential is not None:
        if geopotential_reference is None:
            phi_mean=mean_profile(phi)
            ref_valid=np.isfinite(tbar)&np.isfinite(phi_mean)
            if not ref_valid.any(): raise ValueError('no valid geopotential reference anchor')
            anchor=int(np.flatnonzero(ref_valid)[np.argmax(p[ref_valid])])
            phi_ref=np.full(len(p),np.nan)
            left=anchor
            right=anchor
            while left>0 and ref_valid[left-1]: left-=1
            while right+1<len(p) and ref_valid[right+1]: right+=1
            # Only the contiguous valid segment containing the reference anchor.
            segment=slice(left,right+1)
            values=np.r_[0.,-gas*np.cumsum(.5*(tbar[left:right]+tbar[left+1:right+1])*np.diff(np.log(p[segment])))]
            values+=phi_mean[anchor]-values[anchor-left]
            phi_ref[segment]=values
        else: phi_ref=real(geopotential_reference,'geopotential_reference')
        if phi_ref.shape!=(len(p),) or not np.isfinite(phi_ref).any(): raise ValueError('geopotential reference unavailable')
        data['phi']=np.where(fixed[None,:,:,:],phi-phi_ref[None,:,None,None],np.nan)
    gradients={}
    if include_vertical_gradients:
        for k in ('u','v','t'):
            gradients['d'+k+'_dp']=derivative(data[k],p,1)
        if geopotential is not None:
            gradients['hydrostatic_residual']=derivative(np.where(fixed[None,:,:,:],phi,np.nan),p,1)+gas*arrays['temperature']/p[None,:,None,None]
    op=real(omega_pressure,'omega_pressure')
    if op.ndim!=1 or not np.isfinite(op).all() or np.any(op<=0) or len(set(op.tolist()))!=len(op): raise ValueError('invalid omega pressure (Pa)')
    oi=[]
    for q in op:
        hit=np.flatnonzero(p==q)
        if len(hit)!=1: raise ValueError('omega pressure must exactly match a wind/T pressure')
        oi.append(int(hit[0]))
    oi=np.array(oi,dtype=int)
    omega_valid=np.zeros((len(op),shape[2],shape[3]),dtype=bool)
    if omega is not None:
        w=real(omega,'omega')
        if w.shape!=(shape[0],len(op),shape[2],shape[3]): raise ValueError('omega shape mismatch')
        omega_valid=fixed[oi]&np.isfinite(w).all(axis=0)
        data['omega']=np.where(omega_valid[None,:,:,:],w,np.nan)
        for k in ('u','v','t'): data['omega_'+k]=data['omega']*data[k][:,oi,:,:]
    basic_state={'lon':lon.copy(),'lat':lat.copy(),'pressure':p.copy(),'omega_pressure':op.copy(),
                 'omega_indices':oi,'temperature_mean':tbar,'dtemperature_mean_dp':dtbar,'lapse_rate':lapse,
                 'stability':stability,'c':c,'dc_dp':dc,'valid_stability':valid_stability,
                 'data_shape':np.array(shape,dtype=int),'valid_mask':fixed,'omega_valid_mask':omega_valid,
                 'area_weights':area,'gas_constant':np.array(gas),'gravity':np.array(g),
                 'reference_weight_fraction':np.sum(area[None,None,:,:]*(reference_valid & np.isfinite(arrays['temperature'])),axis=(0,2,3))/(shape[0]*area.sum()),
                 'reference_valid_sample_count':np.sum(reference_valid & np.isfinite(arrays['temperature']),axis=(0,2,3))}
    if phi_ref is not None: basic_state['geopotential_reference']=phi_ref
    basic_state.update(gradients)
    return data,basic_state
