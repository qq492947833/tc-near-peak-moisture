
"""Prespecified P1 background comparison. Consumes verified paired sources only."""
from pathlib import Path
import argparse,json,hashlib,datetime,sys,time
import numpy as np
import pandas as pd
from p4_p5_analysis_v5 import phase_standardization,clean
from v5_statistics import summarize_draws
PROCESSES=['shear200_850','shear500_850','shear200_500','midRH','q700_horizontal_advection','q700_vertical_advection','PIheadroom','ventilation','SST']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def writej(p,x):Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def yes(v):
 if v is True or str(v).lower() in ('true','1'):return True
 if v is False or str(v).lower() in ('false','0'):return False
 raise ValueError('invalid boolean '+str(v))
def finite(v):
 try:return float(v)
 except (ValueError,TypeError):return np.nan
def event_vector(row,levels):
 lon=np.deg2rad(finite(row.lon));month=pd.Timestamp(row.t0).month
 x=[1.,finite(row.lat),np.sin(lon),np.cos(lon),np.sin(2*np.pi*month/12),np.cos(2*np.pi*month/12),finite(row.wind_t0_kt),np.log1p(finite(row.age_h)),finite(row.delta_v24_backward),finite(row.land_contact)]
 return x+[float(str(row.basin_genesis)==b) for b in levels[1:]]
def selftest(out):
 rng=np.random.default_rng(42);n=60;sid=np.array(['S%03d'%i for i in range(n)])
 x=rng.normal(size=n);A=np.c_[np.ones(n),x];B=np.c_[np.ones(n),x+.2]
 a=2*x;b=3+2*(x+.2)
 r=phase_standardization(sid,a,b,A,B,draws=32,seed=5)
 assert r['point']['success'];assert abs(r['point']['estimate']-3)<1e-9
 assert abs(r['raw_original_valid_population']['estimate']-3.4)<1e-9
 assert np.nanmax(abs(r['adjusted']['draws']-3))<1e-8
 # Independently solve the observed pooled design for exactly its saved support.
 k=r['point']['support_row_mask'];X=np.c_[np.r_[np.zeros(k.sum()),np.ones(k.sum())],np.vstack([A[k],B[k]])]
 beta=np.linalg.lstsq(X,np.r_[a[k],b[k]],rcond=None)[0]
 assert abs(beta[0]-r['point']['estimate'])<1e-10
 inv=phase_standardization(sid,b,a,B,A,draws=8,seed=5);assert abs(inv['point']['estimate']+3)<1e-9
 Ar=np.c_[A,A[:,1]*1e8];Br=np.c_[B,B[:,1]*1e8]
 red=phase_standardization(sid,a,b,Ar,Br,draws=8,seed=5);assert abs(red['point']['estimate']-3)<1e-8
 sep=phase_standardization(sid,a,b,A,np.c_[np.ones(n),x+100],draws=8,seed=5)
 assert not sep.get('point',{}).get('success',False)
 assert yes('False') is False and yes('True') is True
 writej(out,dict(status='PASS_SYNTHETIC_WRAPPER_ONLY',checks=8,created=now(),code_sha256=sha(__file__),not_observed_science=True))
def run(a):
 start=now();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 config=root/'config/p1_background_implementation_v1.json';events=Path(a.events);pairs=Path(a.pairs)
 binding={'pairs':{'path':str(pairs),'sha256':sha(pairs)},'events':{'path':str(events),'sha256':sha(events)},'config_sha256':sha(config),'wrapper_sha256':sha(__file__),'statistic_code_sha256':sha(Path(__file__).with_name('p4_p5_analysis_v5.py'))}
 old=out/'binding.json'
 if old.exists():
  if json.loads(old.read_text('utf-8'))!=binding:raise ValueError('existing input binding differs')
 else:writej(old,binding)
 e=pd.read_csv(events,keep_default_na=False);e=e[e.current117.map(yes)].copy()
 if e.duplicated(['sid','event_class']).any():raise ValueError('duplicate event')
 levels=sorted(e.basin_genesis.unique());by={(r.sid,r.event_class):r for r in e.itertuples()}
 p=pd.read_csv(pairs);p=p[p.summary.eq('pre_mean')].copy()
 if p.center_policy.nunique()!=1 or p.variable.nunique()!=9:raise ValueError('wrong primary premean source')
 output=[]
 for var,g in p.groupby('variable',sort=True):
  if g.sid.duplicated().any():raise ValueError('duplicate SID per target')
  g=g.sort_values('sid');sid=g.sid.to_numpy(str)
  if any((s,'internal') not in by or (s,'final') not in by for s in sid):raise ValueError('nonprimary SID')
  W0=np.asarray([event_vector(by[s,'final'],levels) for s in sid]);W1=np.asarray([event_vector(by[s,'internal'],levels) for s in sid])
  basin=np.array([str(by[s,'internal'].basin_genesis) for s in sid])
  dest=out/var;dest.mkdir(exist_ok=True)
  if (dest/'COMPLETE.json').exists():
   done=json.loads((dest/'COMPLETE.json').read_text('utf-8'))
   if all(sha(dest/n)==h for n,h in done['outputs'].items()):
    output.append(json.loads((dest/'summary.json').read_text('utf-8')));continue
   raise ValueError('completed result fingerprint mismatch')
  r=phase_standardization(sid,g.final.to_numpy(float),g.internal.to_numpy(float),W0,W1,basin,draws=2000,seed=510082026)
  g.to_csv(dest/'paired_source.csv',index=False)
  names=['intercept','latitude','longitude_sin','longitude_cos','month_sin','month_cos','wind_t0_kt','log1p_age_h','past24windchange','land_contact']+['basin_'+b for b in levels[1:]]
  pd.DataFrame(np.vstack([W0,W1]),columns=names).assign(sid=np.r_[sid,sid],event_class=['final']*len(sid)+['internal']*len(sid)).to_csv(dest/'design.csv',index=False)
  row=dict(variable=var,units=str(g.units.iloc[0]),status=r['status'],contrast='internal minus final',n_raw=int(np.isfinite(g.difference.to_numpy(float)).sum()),n_background_complete=r['n_sid_background_complete'],n_support=r.get('n_sid_support',r.get('point',{}).get('n_sid_support')),CI_method='SID percentile; support and outcome refitted each draw; descriptive no P/q',requested_draws=2000)
  raw=r['raw_original_valid_population'];row.update(raw_estimate=raw['estimate'])
  # cluster_mean stores draws, and its own valid SID membership; never subtract endpoints.
  for label,z in [('raw',raw),('raw_same_support',r.get('raw_same_support',{})),('adjusted',r.get('adjusted',{}))]:
   if 'draws' in z and np.isfinite(z.get('estimate',np.nan)):
    ss=summarize_draws(z['estimate'],z['draws']);row.update({label+'_'+k:ss[k] for k in ('estimate','lo','hi','valid_draws')})
   else:row.update({label+'_'+k:None for k in ('estimate','lo','hi','valid_draws')})
  row['adjusted_failed_draws']=r.get('bootstrap_failure_count',2000)
  row['inference_scope']='restricted successful-draw interval; target not supported in all resamples' if row['adjusted_failed_draws'] else 'prespecified supported descriptive target'
  np.savez_compressed(dest/'source_and_draws.npz',SID=sid,W_final=W0,W_internal=W1,final=g.final.to_numpy(float),internal=g.internal.to_numpy(float),
   adjusted_draws=r.get('adjusted',{}).get('draws',np.full(2000,np.nan)),raw_same_support_draws=r.get('raw_same_support',{}).get('draws',np.full(2000,np.nan)),
   raw_draws=raw['draws'],draw_SID_order=r.get('sampled_sid_order',np.array([],dtype=str)),SID_draw_indices=r.get('sampled_sid_indices',np.empty((0,0),int)))
  pd.DataFrame(r.get('draw_records',[])).to_csv(dest/'draw_status.csv',index=False)
  detail={k:v for k,v in r.items() if k not in ['adjusted','raw_same_support','raw_original_valid_population','sampled_sid_indices','sampled_sid_order','draw_records']}
  writej(dest/'support_and_fit.json',detail);writej(dest/'summary.json',row)
  files=['paired_source.csv','design.csv','source_and_draws.npz','draw_status.csv','support_and_fit.json','summary.json']
  writej(dest/'COMPLETE.json',dict(created=now(),outputs={n:sha(dest/n) for n in files},status='COMPUTED_PENDING_INDEPENDENT_REVIEW'))
  output.append(row)
 pd.DataFrame(output).to_csv(out/'background_contrasts.csv',index=False,encoding='utf-8-sig')
 writej(out/'execution.json',dict(start=start,end=now(),exit_code=0,command=sys.argv,status='COMPUTED_PENDING_INDEPENDENT_REVIEW',input_bindings=binding,targets=len(output)))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root');ap.add_argument('--events');ap.add_argument('--pairs');ap.add_argument('--out');ap.add_argument('--selftest')
 args=ap.parse_args()
 if args.selftest:selftest(args.selftest)
 else:run(args)
