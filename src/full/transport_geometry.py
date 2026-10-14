"""Spherical moving-control-volume geometry. SI units throughout."""
import numpy as np
A=6371000.0
G0=9.80665

def ll_xyz(lat,lon):
    lat,lon=np.broadcast_arrays(np.radians(lat),np.radians(lon))
    return np.stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=-1)

def xyz_ll(x):
    x=np.asarray(x);x=x/np.linalg.norm(x,axis=-1,keepdims=True)
    return np.degrees(np.arcsin(np.clip(x[...,2],-1,1))),np.degrees(np.arctan2(x[...,1],x[...,0]))

def basis(x):
    lat,lon=map(np.radians,xyz_ll(x))
    east=np.stack((-np.sin(lon),np.cos(lon),np.zeros_like(lon)),axis=-1)
    north=np.stack((-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)),axis=-1)
    return east,north

def slerp(a,b,f):
    a=np.asarray(a);b=np.asarray(b);theta=np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1))
    if np.any(theta>np.pi-1e-8):raise ValueError('Antipodal centre interpolation is ambiguous')
    f=np.broadcast_to(np.asarray(f),theta.shape)
    wa=(1-f)*np.sinc((1-f)*theta/np.pi)/np.sinc(theta/np.pi)
    wb=f*np.sinc(f*theta/np.pi)/np.sinc(theta/np.pi)
    out=wa[...,None]*a+wb[...,None]*b
    return out/np.linalg.norm(out,axis=-1,keepdims=True)

def circle(k,r_m,n_az=360):
    """Fixed geodesic radius; theta is clockwise from local north."""
    alpha=r_m/A;theta=np.arange(n_az)*2*np.pi/n_az
    e,n=basis(k);out=np.cos(theta)[:,None]*n+np.sin(theta)[:,None]*e
    x=np.cos(alpha)*k+np.sin(alpha)*out
    normal=(np.cos(alpha)*x-k)/np.sin(alpha)
    dl=np.full(n_az,A*np.sin(alpha)*2*np.pi/n_az)
    return x,normal,dl

def boundary_normal_speed(k,kdot,x,r_m):
    # Differentiate k(t).x(t)=cos(r/a). Only normal velocity affects flux.
    return A*np.sum(kdot*x,axis=-1)/np.sin(r_m/A)

def annulus_grid(k,r_inner,r_outer,n_radial=16,n_az=180):
    """Gauss-Legendre in cos(radius/a); exact spherical area weights."""
    z,w=np.polynomial.legendre.leggauss(n_radial)
    lo=np.cos(r_outer/A);hi=np.cos(r_inner/A)
    c=(hi+lo)/2+z*(hi-lo)/2;wc=w*(hi-lo)/2
    e,n=basis(k);theta=np.arange(n_az)*2*np.pi/n_az
    tang=np.cos(theta)[:,None]*n+np.sin(theta)[:,None]*e
    x=c[:,None,None]*k+np.sqrt(1-c*c)[:,None,None]*tang[None,:,:]
    da=np.broadcast_to((A*A*wc*2*np.pi/n_az)[:,None],(n_radial,n_az)).copy()
    return x.reshape(-1,3),da.ravel()

def flux_levels(q,v_normal,dl):
    """kg/s/Pa at native pressures; incomplete circumference remains NaN."""
    q=np.asarray(q);v_normal=np.asarray(v_normal)
    inward=np.sum(q*np.maximum(-v_normal,0)*dl,axis=-1)/G0
    outward=np.sum(q*np.maximum(v_normal,0)*dl,axis=-1)/G0
    return {'in':inward,'out':outward,'net':inward-outward}

def column_integral(values,p_pa):
    values=np.asarray(values);p_pa=np.asarray(p_pa)
    if p_pa.ndim!=1 or len(p_pa)<2 or len(values)!=len(p_pa) or not np.isfinite(p_pa).all() or not np.all(np.diff(p_pa)>0):
        raise ValueError('Pressure must be strictly increasing with matching first data axis')
    if np.max(p_pa)<2000:raise ValueError('Pressure must be in Pa, not hPa')
    return np.trapezoid(values,p_pa,axis=0)

def vertical_exchange(q_top,q_bottom,omega_top,omega_bottom,da):
    return np.sum((q_top*omega_top-q_bottom*omega_bottom)*da)/G0

def integrate_trajectory(x,p,t_start,t_end,step_seconds,field):
    """Earth-fixed vector RK4 using absolute wind; pressure velocity is Pa/s.

    Caller handles interpolation, validity, and trajectory termination. This
    primitive never substitutes zeros or extrapolates missing field values.
    """
    x=np.asarray(x,dtype=float).copy();p=np.asarray(p,dtype=float).copy()
    if not np.isfinite(step_seconds) or step_seconds<=0:raise ValueError('Positive finite RK4 step required')
    t=float(t_start);sgn=np.sign(t_end-t_start)
    def rhs(xx,pp,tt):
        xx=xx/np.linalg.norm(xx,axis=-1,keepdims=True)
        v,w=field(xx,pp,tt)
        v=np.asarray(v);v=v-np.sum(v*xx,axis=-1,keepdims=True)*xx
        return v/A,np.asarray(w)
    while sgn*(t_end-t)>1e-7:
        dt=sgn*min(abs(t_end-t),abs(step_seconds))
        a,b=rhs(x,p,t);c,d=rhs(x+dt*a/2,p+dt*b/2,t+dt/2)
        e,f=rhs(x+dt*c/2,p+dt*d/2,t+dt/2);g,h=rhs(x+dt*e,p+dt*f,t+dt)
        x=x+dt*(a+2*c+2*e+g)/6;p=p+dt*(b+2*d+2*f+h)/6
        x=x/np.linalg.norm(x,axis=-1,keepdims=True);t+=dt
    return x,p
