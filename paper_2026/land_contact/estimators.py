"""Paired-SID and matched-energy estimators reused from the frozen production routines."""
import numpy as np
import pandas as pd

SEED = 650117
REPS = 2000

def bootstrap_summary(values, indices):
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 1 and indices.shape[1] == len(v)
    if not np.isfinite(v).all():
        return {'n': len(v), 'finite_n': int(np.isfinite(v).sum()), 'estimate': None,
                'low': None, 'high': None, 'status': 'NA conditional quantity undefined; no draw replacement'}
    boot = v[indices].mean(axis=1)
    return {'n': len(v), 'finite_n': len(v), 'estimate': float(v.mean()),
            'low': float(np.quantile(boot, .025)), 'high': float(np.quantile(boot, .975)),
            'status': 'COMPUTED'}


def equal_weights(sid,event,mult=None):
    """Equal bins within event, events within SID; bootstrap SID multiplicity."""
    df=pd.DataFrame({'sid':sid,'event':event})
    nbin=df.groupby(['sid','event']).event.transform('size').to_numpy()
    nevent=df.groupby('sid').event.transform('nunique').to_numpy()
    weight=1./nbin/nevent
    if mult is not None:weight*=np.array([mult[s] for s in sid])
    return weight


def summarize(estimate,draws):
    a=np.asarray(draws,float);a=a[np.isfinite(a)]
    if not np.isfinite(estimate) or not len(a):return dict(estimate=estimate,lower=np.nan,upper=np.nan,p_centered=np.nan,valid_bootstrap=0,failed_bootstrap=REPS)
    lo,hi=np.quantile(a,[.025,.975])
    return dict(estimate=float(estimate),lower=float(lo),upper=float(hi),p_centered=float((1+np.count_nonzero(np.abs(a-estimate)>=abs(estimate)))/(len(a)+1)),valid_bootstrap=len(a),failed_bootstrap=REPS-len(a))


def raw_calc(sid,event,a,b):
    valid=np.isfinite(a)&np.isfinite(b)
    df=pd.DataFrame({'sid':sid[valid],'event':event[valid],'rise':b[valid],'decay':a[valid]})
    ev=df.groupby(['sid','event'])[['rise','decay']].mean();by=ev.groupby('sid').mean().sort_index()
    delta=(by.rise-by.decay).to_numpy();ix=np.random.default_rng(SEED).integers(0,len(by),size=(REPS,len(by)))
    draw=delta[ix].mean(axis=1)
    return dict(**summarize(delta.mean(),draw),n_sid=len(by),n_events=len(ev),n_bin_pairs=int(valid.sum()),rise_mean=float(by.rise.mean()),decay_mean=float(by.decay.mean()),sid_ESS=float(len(by)),row_ESS=float((equal_weights(sid[valid],event[valid]).sum())**2/np.sum(equal_weights(sid[valid],event[valid])**2))),draw
