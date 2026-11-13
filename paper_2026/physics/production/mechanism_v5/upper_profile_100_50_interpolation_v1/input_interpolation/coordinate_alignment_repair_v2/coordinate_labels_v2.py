"""Native 0.25 degree coordinate identity, with arithmetic-roundoff canonicalization.
This is not geographic nearest selection: labels must fall within a floating
operation bound of an exact quarter-degree integer and that integer must exist.
"""
import numpy as np
# Existing longitude expression: center+((source-center+180)%360-180).
# The range of its finite operands is bounded by720 degrees. Six arithmetic/
# remainder operations get a conservative16ULP bound at that operand scale.
ROUND_OFF_DEG=16.*np.spacing(np.float64(720.))
def labels(coordinate,longitude=False):
 raw=np.asarray(coordinate,dtype=np.float64)
 if not np.isfinite(raw).all():raise ValueError('nonfinite coordinate label')
 a=np.mod(raw,360.) if longitude else raw
 k=np.rint(a*4.);canonical=k/4.
 if np.any(np.abs(a-canonical)>ROUND_OFF_DEG):raise ValueError('coordinate not on native quarter-degree lattice within arithmetic roundoff')
 if longitude:k=np.mod(k,1440.)
 return k.astype(np.int64)
def align_native(values,source_lat,source_lon,target_lat,target_lon):
 a=np.asarray(np.ma.filled(np.ma.asarray(values,dtype=float),np.nan),dtype=np.float64)
 sy=labels(source_lat);sx=labels(source_lon,True);ty=labels(target_lat);tx=labels(target_lon,True)
 if a.shape!=(len(sy),len(sx)) or len(set(sy))!=len(sy) or len(set(sx))!=len(sx):raise ValueError('shape/duplicate native coordinate')
 iy={v:i for i,v in enumerate(sy)};ix={v:i for i,v in enumerate(sx)}
 try:ii=[iy[v] for v in ty];jj=[ix[v] for v in tx]
 except KeyError as e:raise ValueError('actual native coordinate missing; no nearest selection') from e
 return a[np.ix_(ii,jj)]
