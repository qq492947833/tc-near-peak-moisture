"""Native-grid 700-hPa tendency and advection, with all signed budget terms.

The residual is the algebraic remainder, not an independent closure test.
Inputs are co-located q in kg/kg, wind in m/s, omega in Pa/s and degrees.
Temporal neighbors are actual +/-6-hour fields at the same grid positions.
"""
import numpy as np
from .native_grid import spherical_gradient, vertical_advection_700


def terms(q600,q700,q850,u700,v700,omega700,q_before,q_after,
          latitude,longitude,motion_u_ms,motion_v_ms):
    """Return g kg-1 day-1 fields and the simultaneous finite support mask.

    Motion is the contemporaneous observed storm translation at the sampled
    time even when the environmental center is held at its event-start position.
    """
    q=np.asarray(q700,float)
    gx,gy=spherical_gradient(q,latitude,longitude)
    fixed=(np.asarray(q_after)-np.asarray(q_before))/43200.*86400000.
    motion=(motion_u_ms*gx+motion_v_ms*gy)*86400000.
    horizontal_fixed=-(np.asarray(u700)*gx+np.asarray(v700)*gy)*86400000.
    horizontal=horizontal_fixed+motion
    vertical=vertical_advection_700(q600,q,q850,omega700)*86400000.
    moving=fixed+motion
    fields=dict(moving_tendency=moving,fixed_tendency=fixed,motion_term=motion,
                horizontal_advection=horizontal,horizontal_advection_fixed_frame=horizontal_fixed,
                vertical_advection=vertical,residual=moving-horizontal-vertical)
    common=np.isfinite(np.stack(list(fields.values()))).all(axis=0)
    return fields,common
