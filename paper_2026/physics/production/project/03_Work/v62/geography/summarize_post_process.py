"""SID-level subsequent process summaries; unknown outcome is its own group."""
from pathlib import Path
import json,datetime
import numpy as np,pandas as pd
from geography_tables import O,read,bools,sha

def mean_and_draw(v,mask,w):
    v=np.asarray(v,float);mask=np.asarray(mask,bool)&np.isfinite(v);den=w[:,mask].sum(1);d=np.full(len(w),np.nan)
    np.divide(w[:,mask]@v[mask],den,out=d,where=den>0)
    n=int(mask.sum());return n,float(v[mask].mean()) if n else np.nan,(np.nanquantile(d,[.025,.975]) if n>=3 else [np.nan,np.nan]),d

def main():
    receipt=json.loads((O/'post_extraction_receipt.json').read_text());assert receipt['status']=='COMPLETE'
    req=read(O/'post_process_requests.csv');d=read(O/'post_process_native_sources.csv');sids=req[['sid','region_at_t0','outcome_group']].drop_duplicates().sort_values('sid').reset_index(drop=True)
    assert sids.sid.is_unique and len(d.request_id.unique())==int(bools(req.observed_position).sum())
    for folder in d.source_folder.unique():
        f=Path(folder);z=json.loads((f/'COMPLETE.json').read_text());assert sha(f/'values.json')==z['values_sha256']
    indices=np.random.default_rng(62112026).integers(0,len(sids),size=(2000,len(sids)),dtype=np.int32);w=np.array([np.bincount(ix,minlength=len(sids)) for ix in indices],np.int16)
    np.savez_compressed(O/'post_process_SID_bootstrap_indices.npz',sid=sids.sid.to_numpy(str),indices=indices)
    sid_rows=[];estimates=[];contrasts=[];curve=[];curvesources=[];draws={};audit=[];budget_masks={}
    budget_fields=['fixed_tendency','motion_term','moving_tendency','horizontal_advection','horizontal_advection_fixed_frame','vertical_advection','residual']
    for (radial,field),z in d.groupby(['radial','field']):
        wide=z.pivot(index='sid',columns='lag_h',values='value').reindex(index=sids.sid,columns=[0,6,12,18,24]);a=wide.to_numpy();complete=np.isfinite(a).all(1)
        # Fixed SID support over time, independently defined for q, the joint
        # budget family and 850 fields. Invalid times do not become zero rates.
        a[~complete]=np.nan
        if field in budget_fields:budget_masks[(radial,field)]=complete.copy()
        for i,sid in enumerate(sids.sid):
            for j,lag in enumerate([0,6,12,18,24]):curvesources.append(dict(sid=sid,radial=radial,field=field,lag_h=lag,value=a[i,j],full_five_time_support=bool(complete[i]),outcome_group=sids.outcome_group.iloc[i]))
        unit=z.units.iloc[0];integral=np.trapz(a,dx=.25,axis=1)
        targets={'change_0_24':a[:,4]-a[:,0]} if field=='q700' else {'integral_0_24':integral}
        for target,v in targets.items():
            outunit='g kg-1' if field!='convergence850' else '1e-5 s-1 day'
            for i,sid in enumerate(sids.sid):sid_rows.append(dict(sid=sid,radial=radial,field=field,estimand=target,value=v[i],units=outunit,full_five_time_support=bool(complete[i]),outcome_group=sids.outcome_group.iloc[i]))
            groupdraw={}
            for group in ['known_return','complete_nonreturn','unknown']:
                mask=sids.outcome_group.eq(group).to_numpy();n,point,ci,dv=mean_and_draw(v,mask,w)
                estimates.append(dict(region='SCS_box',radial=radial,field=field,estimand=target,outcome_group=group,n_sid=n,estimate=point,ci_low=ci[0],ci_high=ci[1],B_valid=int(np.isfinite(dv).sum()),units=outunit,bootstrap='2000 SID percentile; original outcome groups fixed'))
                key='integral|'+radial+'|'+field+'|'+group;draws[key]=dv;groupdraw[group]=(n,point,dv)
            nr,pr,dr=groupdraw['known_return'];nn,pn,dn=groupdraw['complete_nonreturn'];diff=dr-dn;ci=np.nanquantile(diff,[.025,.975]) if min(nr,nn)>=3 else [np.nan,np.nan]
            contrasts.append(dict(region='SCS_box',radial=radial,field=field,estimand=target,n_return=nr,n_complete_nonreturn=nn,estimate=pr-pn,ci_low=ci[0],ci_high=ci[1],B_valid=int(np.isfinite(diff).sum()),units=outunit,contrast='known return minus complete nonreturn; subsequent process association'));draws['contrast|'+radial+'|'+field]=diff
        for group in ['known_return','complete_nonreturn','unknown']:
            mask=sids.outcome_group.eq(group).to_numpy()
            for j,lag in enumerate([0,6,12,18,24]):
                n,point,ci,dv=mean_and_draw(a[:,j],mask,w);curve.append(dict(region='SCS_box',radial=radial,field=field,outcome_group=group,lag_h=lag,n_sid=n,estimate=point,ci_low=ci[0],ci_high=ci[1],B_valid=int(np.isfinite(dv).sum()),units=unit,full_five_time_support=True));draws['curve|'+radial+'|'+field+'|'+group+'|'+str(lag)]=dv
        audit.append(dict(radial=radial,field=field,planned_sid=len(sids),five_time_valid_sid=int(complete.sum()),missing_sid=int((~complete).sum())))
    for radial in ['inner','outer']:
        for field in budget_fields:assert np.array_equal(budget_masks[(radial,field)],budget_masks[(radial,'moving_tendency')]),(radial,field)
    for name,rows in [('post_process_SID_estimands',sid_rows),('post_process_group_estimates',estimates),('post_process_return_contrasts',contrasts),('post_process_curves',curve),('post_process_curve_SID_sources',curvesources),('post_process_support',audit)]:pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
    pd.DataFrame(draws).to_csv(O/'post_process_bootstrap_draws.csv',index_label='draw')
    # Native support algebra is checked again after time integration.
    tab=pd.DataFrame(sid_rows);p=tab[tab.estimand.eq('integral_0_24')].pivot(index=['sid','radial'],columns='field',values='value')
    closure=(p.moving_tendency-p.horizontal_advection-p.vertical_advection-p.residual).dropna();movement=(p.moving_tendency-p.fixed_tendency-p.motion_term).dropna()
    assert np.max(abs(closure))<1e-10 and np.max(abs(movement))<1e-10
    rec=dict(status='COMPLETE_PENDING_REVIEW',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),n_SID=len(sids),known_groups=sids.outcome_group.value_counts().to_dict(),native_rows=len(d),estimand_rows=len(sid_rows),comparison='Within fixed SCS origin cohort, known return versus complete 48h nonreturn; unknown retained separately',support='Every curve and integral uses the same complete five-time SID support for that field/radial band. All seven 700 budget terms share native support; 850 is independent.',max_integral_closure_error=float(abs(closure).max()),max_moving_transform_error=float(abs(movement).max()),bootstrap_draws=2000,source_sha256={str(p):sha(p) for p in [Path(__file__),O/'post_process_native_sources.csv',O/'post_process_requests.csv']})
    (O/'post_summary_receipt.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(json.dumps(rec))

if __name__=='__main__':main()
