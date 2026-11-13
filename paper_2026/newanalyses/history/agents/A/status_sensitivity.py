# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
"""Audit-requested status sensitivity; inherited primary risk definition unchanged."""
from pathlib import Path
import runpy,json
O=Path(__file__).resolve().parent
bootstrap(O/'temp')
import numpy as np
import pandas as pd
m=pd.read_csv(O/'sid_denominators.csv',keep_default_na=False,na_values=['']).sort_values('sid').reset_index(drop=True);r=pd.read_csv(O/'risk_intervals_6h.csv',keep_default_na=False,na_values=['']);basins=['NA','EP','WP','NI','SI','SP'];ix={s:i for i,s in enumerate(m.sid)};bi={b:i for i,b in enumerate(basins)}
strict=r.tropical_start.eq(True)&r.tropical_end.eq(True);a=np.zeros((len(m),6,4))
for (s,b),g in r.groupby(['sid','basin_end']):
 q=g[g.tropical_start.eq(True)&g.tropical_end.eq(True)];a[ix[s],bi[b]]=[g.hours.sum(),g.return_event.sum(),q.hours.sum(),q.return_event.sum()]
iy=[np.flatnonzero(m.genesis_year.to_numpy()==yr) for yr in range(1982,2024)];rng=np.random.default_rng(2026110901);draw=[]
for k in range(2000):
 yy=np.concatenate([(rng.integers(0,42)+np.arange(3))%42 for j in range(14)]);w=np.zeros(len(m))
 for j in yy:np.add.at(w,rng.choice(iy[j],len(iy[j]),replace=True),1)
 z=np.einsum('n,nbk->bk',w,a);draw.append(z[:,3]/z[:,2]*1000)
draw=np.array(draw);z=a.sum(axis=0);out=[]
for j,b in enumerate(basins):
 ci=np.quantile(draw[:,j],[.025,.975]);out.append(dict(basin=b,primary_hours=z[j,0],primary_events=z[j,1],strict_tropical_hours=z[j,2],strict_tropical_events=z[j,3],strict_rate_per1000h=z[j,3]/z[j,2]*1000,strict_rate_ci_low=ci[0],strict_rate_ci_high=ci[1],excluded_hours=z[j,0]-z[j,2],excluded_events=z[j,1]-z[j,3]))
pd.DataFrame(out).to_csv(O/'risk_status_sensitivity.csv',index=False);r[~strict].to_csv(O/'risk_nontropical_endpoint_intervals.csv',index=False);np.save(O/'risk_status_sensitivity_draws.npy',draw)
rep=dict(status='COMPLETE_STATUS_SENSITIVITY',primary_rule='Inherited legal=finite wind endpoints plus6h interval plus termination bound; no tropical-status screen within retained full lifecycle',excluded_intervals=int((~strict).sum()),excluded_hours=int(r.loc[~strict,'hours'].sum()),excluded_events=int(r.loc[~strict,'return_event'].sum()),strict_events=int(r.loc[strict,'return_event'].sum()),strict_hours=int(r.loc[strict,'hours'].sum()),strict_sid=r.loc[strict,'sid'].nunique(),bootstrap=2000,role='Sensitivity only. Primary frozen counts/risk set untouched. Full-record opportunity convention is distinct from requiring both endpoints to carry a tropical flag.')
(O/'STATUS_SENSITIVITY_RECEIPT.json').write_text(json.dumps(rep,indent=2),encoding='utf8');print(json.dumps(rep,indent=2));print(pd.DataFrame(out).to_string(index=False))
