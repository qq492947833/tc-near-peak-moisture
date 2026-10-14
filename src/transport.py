"""Moving-boundary moisture exchange, paired samples, and saved-fit reaggregation."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from src.components import integrated_components, bootstrap_summary
ROOT=Path(__file__).absolute().parents[1]
DATA=ROOT/'source_data/transport'
def rd(p): return pd.read_csv(p,keep_default_na=False,na_values=['','NaN','nan'])
def eq(a,b,rtol=1e-11,atol=.1):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape, 'Comparison shapes differ'
    assert np.array_equal(np.isnan(a),np.isnan(b)), 'NaN support differs'
    assert not np.isinf(a).any() and not np.isinf(b).any()
    np.testing.assert_allclose(a,b,rtol=rtol,atol=atol,equal_nan=True)

def check_draw(bank):
    sid,indices=bank['sid'],bank['indices']
    assert sid.ndim==1 and len(np.unique(sid))==len(sid)
    assert indices.shape==(2000,len(sid)) and np.issubdtype(indices.dtype,np.integer)
    assert (indices>=0).all() and (indices<len(sid)).all()
def departures(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    events=rd(DATA/'event_identity_current117.csv')
    ids=rd(DATA/'pair117_identity_support_ledger.csv')
    assert len(events)==234 and events.sid.nunique()==117
    assert events.groupby('sid').event_class.agg(set).apply(lambda x:x=={'internal','final'}).all()
    records=[]
    for meta in json.loads((DATA/'nodes/index.json').read_text()):
        p=DATA/'nodes'/meta['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
        row=events[events.event_id.eq(meta['event_id'])].iloc[0]
        assert row.sid==meta['sid'] and row.event_class==meta['event_class']
        assert pd.Timestamp(row.t0)==pd.Timestamp(meta['t0'])
        with np.load(p) as z:
            assert np.array_equal(z['lag_h'],np.arange(-24,25,6))
            assert np.array_equal(z['pressure_Pa'],[50000,60000,70000,85000,92500])
            valid=np.isfinite(z['q'])&(z['q']>=0)&np.isfinite(z['u'])&np.isfinite(z['v'])&np.isfinite(z['sp_corner_min'][:,None,:])&(z['sp_corner_min'][:,None,:]>=z['pressure_Pa'][None,:,None])
            assert np.array_equal(valid,z['valid'])
            eq(z['a']-z['b'],z['vn'],1e-13,1e-13)
            records.append({'sid':row.sid,'event_id':row.event_id,'event_class':row.event_class,'post':bool(valid[4:].all()),'pre':bool(valid[:5].all()),'both':bool(valid.all())})
    support=pd.DataFrame(records).groupby('sid')[['pre','post','both']].all()
    assert support.post.sum()==80 and support.both.sum()==78
    assert set(support.index[support.post])==set(ids.loc[ids.member_80_exchange,'sid'])
    assert set(support.index[support.both])==set(ids.loc[ids.member_78_common_windows,'sid'])
    b=rd(DATA/'background/event_support_membership.csv')
    assert len(b)==160 and b.sid.nunique()==80
    selected=b.groupby('sid').selected_pair.all()
    assert selected.sum()==74
    pd.DataFrame(records).to_csv(out/'event_node_support.csv',index=False)
    support.to_csv(out/'pair_node_support.csv')
    return {'event_count':234,'paired_sid':117,'post_complete':80,'pre_complete':int(support.pre.sum()),'common_windows':78,'background_common':74}
def t1(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    t=rd(DATA/'t1/T1_event_UTC_columns.csv')
    ids=rd(DATA/'event_identity_current117.csv')
    missing=rd(DATA/'t1/T1_missing_centres.csv')
    assert ids.event_id.is_unique
    keys=['event_id','lag_h']
    assert not t.duplicated(keys).any() and not missing.duplicated(keys).any()
    present=set(t[keys].itertuples(index=False,name=None))
    absent=set(missing[keys].itertuples(index=False,name=None))
    expected={(event,lag) for event in ids.event_id for lag in range(-24,25,6)}
    assert not present&absent and present|absent==expected
    identity=ids.set_index('event_id')
    for frame in [t,missing]:
        matched=identity.loc[frame.event_id]
        assert np.array_equal(frame.sid,matched.sid)
        assert np.array_equal(frame.event_class,matched.event_class)
        assert np.array_equal(pd.to_datetime(frame.t0,utc=True),pd.to_datetime(matched.t0,utc=True))
    assert np.array_equal(pd.to_datetime(t.UTC,utc=True),pd.to_datetime(t.t0,utc=True)+pd.to_timedelta(t.lag_h,unit='h'))
    result=[]
    for row in ids.itertuples():
        z=t[t.event_id.eq(row.event_id)].set_index('lag_h')
        for window,lags in [('pre',np.arange(-24,1,6)),('post',np.arange(0,25,6))]:
            for name in [f'r{r}_{k}_kg_s' for r in [100,300] for k in ['in','out','net']]+['side_net_kg_s']:
                values=z[name].reindex(lags).to_numpy(float)
                val=np.trapezoid(values,lags*3600.) if np.isfinite(values).all() else np.nan
                result.append({'sid':row.sid,'event_class':row.event_class,'metric':window+'_'+name.replace('_kg_s',''),'value':val})
        for lag in [0,24]:
            result.append({'sid':row.sid,'event_class':row.event_class,'metric':f'stock_{lag}h','value':float(z.stock_kg.get(lag,np.nan))})
        stock=z.stock_kg.reindex([0,24]).to_numpy(float)
        result.append({'sid':row.sid,'event_class':row.event_class,'metric':'stock_change_0_24','value':stock[1]-stock[0]})
    pairs=pd.DataFrame(result).pivot(index=['sid','metric'],columns='event_class',values='value').reset_index()
    old=rd(DATA/'t1/T1_SID_paired_values.csv').sort_values(['sid','metric'])
    pairs=pairs.sort_values(['sid','metric'])
    assert pairs[['sid','metric']].to_records(index=False).tolist()==old[['sid','metric']].to_records(index=False).tolist()
    eq(pairs[['internal','final']],old[['internal','final']])
    summaries=[]
    target=rd(DATA/'t1/T1_summary.csv')
    for metric,d in pairs.groupby('metric'):
        b=np.load(DATA/f't1/T1_bootstrap_{metric}.npz')
        d=d.set_index('sid').loc[b['sid']]
        v=d[['internal','final']].to_numpy(float);assert np.isfinite(v).all()
        draws=v[b['indices']].mean(1);diff=draws[:,0]-draws[:,1]
        q=np.quantile(diff,[.025,.975]);est=np.mean(v[:,0]-v[:,1])
        ref=target[target.metric.eq(metric)].iloc[0]
        eq([est,*q],[ref.difference,ref.low,ref.high])
        summaries.append({'metric':metric,'units':'kg','n':len(d),'temporary':v[:,0].mean(),'final':v[:,1].mean(),'difference':est,'low':q[0],'high':q[1]})
    pd.DataFrame(summaries).to_csv(out/'T1_summary.csv',index=False)
    pairs.to_csv(out/'T1_paired_values.csv',index=False)
    # T3 uses a shared fixed SID draw for every complete radial/budget quantity.
    c=rd(DATA/'t1/T3_common_support_event_values.csv')
    draw=np.load(DATA/'t1/T3_joint_bootstrap.npz');rows=[]
    check_draw(draw)
    reference=rd(DATA/'t1/T3_summary.csv')
    support=rd(DATA/'t1/support_by_SID.csv')
    assert len(set(draw['sid']))==len(draw['sid'])==80
    assert set(draw['sid'])==set(support.loc[support.T3_common,'sid'])
    assert reference.metric.is_unique and set(c.metric)==set(reference.metric)
    assert not c.duplicated(['sid','metric']).any()
    # Float64 CSV values and <=80-term means; humidity tolerances are in g kg-1.
    tolerances={'kg':.1,'g kg-1':1e-10}
    for (metric,units),d in c.groupby(['metric','units']):
        assert set(d.sid)==set(draw['sid'])
        ref=reference[reference.metric.eq(metric)].iloc[0]
        assert units==ref.units and units in tolerances
        atol=tolerances[units]
        v=d.set_index('sid').reindex(draw['sid'])[['internal','final']].to_numpy(float)
        eq(d.set_index('sid').loc[draw['sid'],'difference'].to_numpy(float),v[:,0]-v[:,1],atol=atol)
        if units=='kg':
            original=pairs[pairs.metric.eq(metric)].set_index('sid').loc[draw['sid'],['internal','final']].to_numpy(float)
            eq(v,original,atol=atol)
        else:
            assert np.isfinite(v).all(), 'T3 humidity requires complete paired support'
        good=np.isfinite(v).all(1);dd=v[good,0]-v[good,1]
        assert int(good.sum())==ref.n
        eq(v[good].mean(0),[ref.internal,ref.final],atol=atol)
        row={'metric':metric,'units':units,'n':int(good.sum()),'difference':dd.mean()}
        eq(row['difference'],ref.difference,atol=atol)
        if good.all():
            q=np.quantile((v[draw['indices'],0]-v[draw['indices'],1]).mean(1),[.025,.975])
            eq(q,[ref.low,ref.high],atol=atol);row.update(low=q[0],high=q[1],interval_rebuilt=True)
        else:
            supported=d.set_index('sid').loc[np.asarray(draw['sid'])[good]]
            banks=[]
            for path in sorted((DATA/'t1').glob('T1_bootstrap_*.npz')):
                bank=np.load(path)
                if set(bank['sid'])==set(supported.index):banks.append(bank)
            assert banks, 'Original subset draw identity is required'
            bank=banks[0]
            check_draw(bank)
            assert all(np.array_equal(z['indices'],bank['indices']) and np.array_equal(z['sid'],bank['sid']) for z in banks)
            sv=supported.loc[bank['sid'],['internal','final']].to_numpy(float)
            q=np.quantile((sv[bank['indices'],0]-sv[bank['indices'],1]).mean(1),[.025,.975])
            eq(q,[ref.low,ref.high],atol=atol);row.update(low=q[0],high=q[1],interval_rebuilt=True)

        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'T3_summary.csv',index=False)
    return pd.DataFrame(summaries)
def components(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    idx=json.loads((DATA/'nodes/index.json').read_text());by={}
    for e in idx: by.setdefault(e['sid'],{})[e['event_class']]=e['file']
    primary=np.load(DATA/'components/bootstrap_primary80.npz')
    common=np.load(DATA/'components/bootstrap_common_Sstar.npz')
    rows=[]
    for cohort,bank,windows in [('post80',primary,[('0_24h',slice(4,9))]),('common78',common,[('pre_minus24_0h',slice(0,5)),('post_0_24h',slice(4,9))])]:
        for sid in bank['sid']:
            with np.load(DATA/'nodes'/by[sid]['internal']) as a,np.load(DATA/'nodes'/by[sid]['final']) as b:
                for window,k in windows:
                    assert a['valid'][k].all() and b['valid'][k].all()
                    args=[a['q'][k],b['q'][k],a['a'][k],b['a'][k],a['b'][k],b['b'][k]]
                    for kind in ['in','out','net']:
                        values=integrated_components(*args,a['lag_h'][k]*3600.,a['pressure_Pa'],a['dl_m'],kind)
                        rows.append({'sid':sid,'cohort':cohort,'window':window,'kind':kind,**values})
    data=pd.DataFrame(rows);values=list(values);summ=[]
    for (cohort,window,kind),g in data.groupby(['cohort','window','kind']):
        bank=primary if cohort=='post80' else common
        ref=rd(DATA/'components'/('paired_flux_components_0_24_summary.csv' if cohort=='post80' else 'common_prepost_four_cells_summary.csv'))
        for quantity in values:
            v=g.set_index('sid').loc[bank['sid'],quantity].to_numpy()
            s=bootstrap_summary(v,bank['indices'])
            target=ref[ref.window.eq(window)&ref.kind.eq(kind)&ref.quantity.eq(quantity)].iloc[0]
            eq([s['estimate'],s['low'],s['high']],[target.estimate,target.low,target.high])
            summ.append({'cohort':cohort,'window':window,'kind':kind,'quantity':quantity,'units':'kg',**s})
    data.to_csv(out/'boundary_components.csv',index=False)
    pd.DataFrame(summ).to_csv(out/'component_summary.csv',index=False)
    # Direct change uses identical pre/post SID sets.
    original=rd(DATA/'components/common_prepost_paired_change_contrasts_summary.csv');changes=[]
    for kind,g in data[data.cohort.eq('common78')].groupby('kind'):
        pre=g[g.window.eq('pre_minus24_0h')].set_index('sid').loc[common['sid']]
        post=g[g.window.eq('post_0_24h')].set_index('sid').loc[common['sid']]
        theta=post.difference.to_numpy()-pre.difference.to_numpy()
        s=bootstrap_summary(theta,common['indices'])
        z=original[original.kind.eq(kind)&original.quantity.eq('Theta')].iloc[0]
        eq([s['estimate'],s['low'],s['high']],[z.estimate,z.low,z.high])
        changes.append({'kind':kind,'quantity':'Theta','units':'kg',**s})
    pd.DataFrame(changes).to_csv(out/'direct_window_change.csv',index=False)
    # Fixed quadrature sensitivity: reaggregate every retained factor and component.
    sensitivity=rd(DATA/'components/quadrature_sensitivity.csv')
    reference=rd(DATA/'components/quadrature_sensitivity_summary.csv')
    checks=[]
    for (factor,kind),g in sensitivity.groupby(['factor','kind']):
        assert set(g.sid)==set(primary['sid'])
        for quantity in values:
            s=bootstrap_summary(g.set_index('sid').loc[primary['sid'],quantity].to_numpy(),primary['indices'])
            z=reference[reference.factor.eq(factor)&reference.kind.eq(kind)&reference.quantity.eq(quantity)].iloc[0]
            eq([s['estimate'],s['low'],s['high']],[z.estimate,z.low,z.high])
            checks.append({'factor':factor,'kind':kind,'quantity':quantity,**s})
    pd.DataFrame(checks).to_csv(out/'quadrature_summary.csv',index=False)
    return pd.DataFrame(summ)
def background(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=rd(DATA/'t1/T1_SID_paired_values.csv');target=rd(DATA/'background/exchange_contrasts.csv')
    fixed=np.load(DATA/'background/fixed_draw_identity.npz')
    sid=fixed['sid'];ix=fixed['indices'];assert ix.shape==(2000,74)
    result=[]
    for model in ['geographical','geographical_state']:
        diag=json.loads((DATA/f'background/{model}_diagnostic.json').read_text())
        w=rd(DATA/f'background/{model}_weights_design.csv')
        X=w[diag['columns']].to_numpy();y=w.event_class.eq('internal').to_numpy()
        assert np.array_equal(w.sid[y],sid) and np.array_equal(w.sid[~y],sid)
        pr=expit(X@np.array(diag['fit']['coef']));ow=np.where(y,1-pr,pr)
        eq(pr,w.propensity,1e-13,1e-14)
        f=np.load(DATA/f'background/{model}_refit_draws.npz')
        status=rd(DATA/f'background/{model}_draw_status.csv')
        assert np.array_equal(f['indices'],ix) and np.array_equal(status.draw,np.arange(2000))
        assert status.valid.all()
        Y0=np.column_stack([source[source.metric.eq(m)].set_index('sid').loc[sid,'final'] for m in f['metrics']])
        Y1=np.column_stack([source[source.metric.eq(m)].set_index('sid').loc[sid,'internal'] for m in f['metrics']])
        rebuilt=[]
        for draw,coef in zip(ix,f['coefficients']):
            n=np.bincount(draw,minlength=len(sid));p=expit(X@coef);weights=np.where(y,1-p,p)*np.r_[n,n]
            rebuilt.append(np.average(Y1,axis=0,weights=weights[y])-np.average(Y0,axis=0,weights=weights[~y]))
        rebuilt=np.asarray(rebuilt);eq(rebuilt,f['balanced'],atol=.05)
        for j,metric in enumerate(f['metrics']):
            v=source[source.metric.eq(metric)].set_index('sid').loc[fixed['original_sid']]
            vd=v.internal.to_numpy()-v.final.to_numpy()
            for label,pt,boot in [
                ('original80_raw',vd.mean(),vd[fixed['original_indices']].mean(1)),
                ('restricted_raw',(Y1-Y0)[:,j].mean(),(Y1-Y0)[ix,j].mean(1)),
                ('balanced',np.average(Y1[:,j],weights=ow[y])-np.average(Y0[:,j],weights=ow[~y]),rebuilt[:,j])]:
                lo,hi=np.quantile(boot,[.025,.975])
                z=target[target.model.eq(model)&target.metric.eq(metric)&target.target.eq(label)].iloc[0]
                eq([pt,lo,hi],[z.estimate,z.low,z.high])
                result.append({'model':model,'metric':metric,'target':label,'units':'kg','n':80 if label=='original80_raw' else 74,'estimate':pt,'low':lo,'high':hi,'attempts':2000,'failed':0})
    result=pd.DataFrame(result);result.to_csv(out/'background_exchange.csv',index=False)
    return result


