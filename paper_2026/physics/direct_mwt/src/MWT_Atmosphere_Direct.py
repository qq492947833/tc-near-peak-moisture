def MWT_Atmosphere_Direct(coefficients, basic_state, dt=21600., gamma_form='conservative',
                          mask_spatial_edges=True, return_flux_components=True):
    """Direct pressure-coordinate diagnostics; no missing omega recovery.

    coefficients are physical-normalization MWT coefficients, not reconstructions.
    u/v/t/phi and uu/uv/vv/u_t/v_t: (window,time,pressure,lat,lon).
    omega/omega_u/omega_v/omega_t: same axes, only omega_pressure levels.
    Original-field products must be transformed before this function is called.
    Outputs at full pressure levels: KE/APE (J kg-1), time tendencies,
    horizontal flux convergence, horizontal Gamma contributions and W_PG
    (rates W kg-1; horizontal fluxes m3 s-3). Common-level outputs: conversion
    (W kg-1), Q_K_p/Q_A_p/Q_P_p (Pa m2 s-3). Boundary inputs are Q(500)-Q(850)
    divided by g (W m-2) or 35000 Pa (W kg-1). No closed budget/residual claimed.
    gamma_form='conservative' preserves supplied Atmosphere Eq46/62 differencing;
    optional 'bracket' evaluates the equivalent analytic canonical form and is
    for finite-difference method sensitivity, not new observations.
    Derivatives use the supplied geographic grid before neighborhood sampling.
    NaN columns remain NaN. One outer grid row/column is masked by default.
    The caller must ensure identical MWT settings/reference/time support.
    """
    import numpy as np
    def real(a,n):
        if np.iscomplexobj(a): raise TypeError(n+' must be real')
        a=np.asarray(np.ma.asarray(a,dtype=float).filled(np.nan))
        if np.isinf(a).any(): raise ValueError(n+' has infinity')
        return a
    if not np.isfinite(dt) or dt<=0: raise ValueError('dt must be seconds >0')
    if gamma_form not in ('conservative','bracket'): raise ValueError('unknown gamma form')
    names=('u','v','t','uu','uv','vv','u_t','v_t')
    f={k:real(coefficients[k],k) for k in names}
    shape=f['u'].shape
    if len(shape)!=5 or shape[0]!=3 or shape[1]<3 or min(shape[-2:])<3: raise ValueError('coefficient shape invalid')
    if any(x.shape!=shape for x in f.values()): raise ValueError('horizontal coefficient shape mismatch')
    p=real(basic_state['pressure'],'pressure');lon=real(basic_state['lon'],'lon');lat=real(basic_state['lat'],'lat')
    if p.shape!=(shape[2],) or lat.shape!=(shape[3],) or lon.shape!=(shape[4],): raise ValueError('coordinate shape mismatch')
    lr=np.unwrap(np.deg2rad(lon));br=np.deg2rad(lat)
    for x in (p,lr,br):
        if not np.isfinite(x).all() or not (np.all(np.diff(x)>0) or np.all(np.diff(x)<0)): raise ValueError('invalid coordinates')
    if np.any(p<=0) or np.any(np.abs(lat)>=90): raise ValueError('pressure/grid invalid')
    c=real(basic_state['c'],'c')
    if c.shape!=p.shape or np.any(c[np.isfinite(c)]<=0): raise ValueError('c must be positive on valid levels')
    c3=c[:,None,None];radius=6.371e6;gas=287.058;gravity=9.8
    cosine=np.cos(br)[:,None];metric=np.tan(br)[:,None]/radius
    def d(x,coords,axis): return np.gradient(x,coords,axis=axis,edge_order=2)
    def dx(x): return d(x,lr,-1)/(radius*cosine)
    def dy(x): return d(x,br,-2)/radius
    def dy_div(x): return d(x*cosine,br,-2)/(radius*cosine)
    uc,vc,tc=(f[k] for k in ('u','v','t'))
    kuu,kuv,kvv=(f[k] for k in ('uu','uv','vv'))
    qkx=.5*(uc*kuu+vc*kuv);qky=.5*(uc*kuv+vc*kvv)
    divqk=dx(qkx)+dy_div(qky)
    advkh=-(uc*dx(kuu)+vc*dx(kuv)+uc*dy_div(kuv)+vc*dy_div(kvv))+metric*(uc*kuv-vc*kuu)
    qax=.5*c3*tc*f['u_t'];qay=.5*c3*tc*f['v_t']
    divqa=dx(qax)+dy_div(qay)
    advah=-c3*tc*(dx(f['u_t'])+dy_div(f['v_t']))
    if gamma_form=='conservative':
        gkh=advkh+divqk;gah=advah+divqa
    else:
        gkh=.5*(kuu*dx(uc)-uc*dx(kuu)+kuv*dx(vc)-vc*dx(kuv)
                 +kuv*dy(uc)-uc*dy_div(kuv)+kvv*dy(vc)-vc*dy_div(kvv))
        gkh+=metric*(uc*kuv-vc*kuu)
        gah=.5*c3*(f['u_t']*dx(tc)-tc*dx(f['u_t'])+f['v_t']*dy(tc)-tc*dy_div(f['v_t']))
    ke=.5*(uc**2+vc**2);ape=.5*c3*tc**2
    result={'KE':ke,'APE':ape,'dKE_dt':d(ke,np.arange(shape[1])*float(dt),1),
            'dAPE_dt':d(ape,np.arange(shape[1])*float(dt),1),
            'transport_K_h':-divqk,'transport_A_h':-divqa,
            'Gamma_K_h':gkh,'Gamma_A_h':gah}
    if return_flux_components: result.update(Q_K_lon=qkx,Q_K_lat=qky,Q_A_lon=qax,Q_A_lat=qay)
    phic=None
    if 'phi' in coefficients:
        phic=real(coefficients['phi'],'phi')
        if phic.shape!=shape: raise ValueError('phi shape mismatch')
        result['W_PG']=-(uc*dx(phic)+vc*dy(phic))
        if return_flux_components: result.update(Q_P_lon=uc*phic,Q_P_lat=vc*phic)
    if 'omega' in coefficients:
        op=real(basic_state['omega_pressure'],'omega_pressure')
        oi=np.asarray(basic_state['omega_indices'],dtype=int)
        if not np.array_equal(p[oi],op): raise ValueError('omega/full-pressure identity mismatch')
        common_shape=(shape[0],shape[1],len(op),shape[3],shape[4])
        wf={k:real(coefficients[k],k) for k in ('omega','omega_u','omega_v','omega_t')}
        if any(x.shape!=common_shape for x in wf.values()): raise ValueError('omega coefficient shape mismatch')
        uq,vq,tq=uc[:,:,oi,:,:],vc[:,:,oi,:,:],tc[:,:,oi,:,:]
        result['C_A_to_K']=-gas/op[:,None,None]*tq*wf['omega']
        result['Q_K_p']=.5*(uq*wf['omega_u']+vq*wf['omega_v'])
        result['Q_A_p']=.5*c[oi,None,None]*tq*wf['omega_t']
        if phic is not None: result['Q_P_p']=wf['omega']*phic[:,:,oi,:,:]
        if 50000. in op and 85000. in op:
            i0=int(np.flatnonzero(op==50000.)[0]);i1=int(np.flatnonzero(op==85000.)[0])
            for label in ('K','A','P'):
                key='Q_'+label+'_p'
                if key in result:
                    dif=result[key][:,:,i0,:,:]-result[key][:,:,i1,:,:]
                    result['vertical_input_'+label+'_Wm2']=dif/gravity
                    result['vertical_input_'+label+'_Wkg']=dif/35000.
    if mask_spatial_edges:
        for a in result.values():
            a[...,0,:]=np.nan;a[...,-1,:]=np.nan;a[...,:,0]=np.nan;a[...,:,-1]=np.nan
    return result
