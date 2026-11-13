def bools(x):
    if pd.api.types.is_bool_dtype(x): return x.fillna(False)
    return x.astype(str).str.lower().isin(['true','1','1.0'])

def fold(sid): return int(hashlib.sha256(('510082026|'+str(sid)).encode()).hexdigest()[:12],16)%5

def ridge(X, y, w, Xt, names):
    mu=np.average(X,axis=0,weights=w); sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=w))
    keep=sd>1e-12; Z=(X[:,keep]-mu[keep])/sd[keep]; Zt=(Xt[:,keep]-mu[keep])/sd[keep]
    ym=np.average(y,weights=w)
    beta=np.linalg.solve(Z.T@(w[:,None]*Z)+np.eye(keep.sum()),Z.T@(w*(y-ym)))
    prediction=ym+Zt@beta
    model={'columns':names,'train_mean':mu.tolist(),'train_sd':sd.tolist(),'retained':keep.tolist(),'beta_standardized':beta.tolist(),'outcome_train_mean':float(ym),'alpha':1.0,'rank':int(np.linalg.matrix_rank(Z)),'n_columns_retained':int(keep.sum())}
    return prediction,model

def bootstrap_indices(sids,seed,prefix):
    rng=np.random.default_rng(seed);ids=rng.integers(0,len(sids),size=(2000,len(sids)),dtype=np.int32)
    np.savez_compressed(BASE/(prefix+'_draw_indices.npz'),indices=ids,sids=np.array(sids,dtype=str),seed=np.array(seed))
    return ids

def a2(d,ledger,s):
    rows=[];ch=[]
    for sid,g in d.groupby('sid',sort=True):
        legal=g.legal.to_numpy();w=g.valid_h.to_numpy(float)
        first=g.lmi_time.iloc[0];last=g.lmi_last_time.iloc[0]
        intervals=[('genesis_to_firstLMI',g.genesis_time.iloc[0],first),('first_to_lastLMI',first,last),('lastLMI_to_termination',last,g.termination_time.iloc[0])]
        for rep,stages in REP.items():
            for stage in stages:
                mask=legal&g[stage].to_numpy(bool);h=float(w[mask].sum())
                common={'sid':sid,'representation':rep,'stage':stage,'controlled_eligible':bool(ledger.loc[sid,'controlled_eligible']),'covered_h':h,'interval_n':int(mask.sum()),'A_signed_unchanged':float(s.loc[sid,'A_signed'])}
                rows.append(dict(common,variable='covered_h',value=h,valid_h=h,unit='h'))
                for var,unit in zip(VARS,UNITS):
                    val=g[var+'_value'].to_numpy(float);q=bools(g[var+'_qc']).to_numpy()&np.isfinite(val)&(g[var+'_valid_fraction'].to_numpy(float)>=.9);take=mask&q
                    vh=w[take].sum();mean=float(np.average(val[take],weights=w[take])) if vh>0 else np.nan
                    rows.append(dict(common,variable=var,value=mean,valid_h=float(vh),unit=unit))
                for name,start,end in intervals:
                    overlap=np.maximum(0,(np.minimum(g.interval_end.array.asi8,end.value)-np.maximum(g.UTC.array.asi8,start.value))/3.6e12)
                    ch.append(dict(common,chronology=name,chronology_overlap_h=float(overlap[mask].sum())))
    tab=pd.DataFrame(rows);chron=pd.DataFrame(ch)
    tab.to_parquet(BASE/'A2_stage_metrics.parquet',index=False);tab.to_csv(BASE/'A2_stage_metrics.csv',index=False)
    chron.to_csv(BASE/'A2_chronology_allocation.csv',index=False)
    check('chronology_covers_stages',np.allclose(chron.groupby(['sid','representation','stage']).chronology_overlap_h.sum().to_numpy(),chron.groupby(['sid','representation','stage']).covered_h.first().to_numpy()))
    eligible=sorted(ledger.index[ledger.controlled_eligible]);draw=bootstrap_indices(eligible,620122026,'A2');out=[];drawrows=[];pairbank=[]
    choices=[('D-pooledDE','A4-F','D','A3-F','DE'),('E-pooledDE','A4-F','E','A3-F','DE'),('F-R','A4-F','F','A4-R','R'),('Res-post_R_tail','A4-F','Res','A4-R','post_R_tail')]
    for name,r1,st1,r0,st0 in choices:
        for var in ['covered_h']+VARS:
            x=tab[(tab.representation==r1)&(tab.stage==st1)&(tab.variable==var)].set_index('sid').value.reindex(eligible).to_numpy(float)
            y=tab[(tab.representation==r0)&(tab.stage==st0)&(tab.variable==var)].set_index('sid').value.reindex(eligible).to_numpy(float)
            z=x-y;good=np.isfinite(z);vals=z[draw];n=np.isfinite(vals).sum(axis=1);mean=np.divide(np.nansum(vals,axis=1),n,out=np.full(2000,np.nan),where=n>0)
            ci=np.nanpercentile(mean,[2.5,97.5]);out.append({'contrast':name,'variable':var,'n_paired_sid':int(good.sum()),'estimate':float(np.nanmean(z)),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'zero_difference_sid':int((z==0).sum()),'B':2000,'failed_draws':int(np.isnan(mean).sum())})
            drawrows.extend({'draw':i,'contrast':name,'variable':var,'estimate':a,'paired_sample_n':int(n[i])} for i,a in enumerate(mean))
            pairbank.extend({'sid':sid,'contrast':name,'variable':var,'left':a,'right':b,'difference':c} for sid,a,b,c in zip(eligible,x,y,z))
    pd.DataFrame(out).to_csv(BASE/'A2_paired_contrasts.csv',index=False)
    pd.DataFrame(drawrows).to_parquet(BASE/'A2_bootstrap_estimates.parquet',index=False)
    pd.DataFrame(pairbank).to_parquet(BASE/'A2_paired_bank.parquet',index=False)
    report=[]
    for rep,stages in REP.items():
        report.append({'representation':rep,'separate_D_E':('D' in stages),'recurrence_retained_in_maturity':('F' in stages),'original_sid_preserved':True,'later_occupancy_and_bridges_explicit':'R/L/B decomposition available' if 'F' in stages else 'later occupancy and bridge times reassigned to post-R tail','controlled_sid_n':len(eligible)})
    pd.DataFrame(report).to_csv(BASE/'A2_reportability.csv',index=False)
    # Identity checks are structural; identical contrasts across the independent factor must remain exact.
    piv=tab.pivot(index=['sid','stage','variable'],columns='representation',values='value')
    for a,b,stages in [('A4-F','A4-R',['D','E']),('A3-F','A3-R',['DE']),('A4-F','A3-F',['F','Res']),('A4-R','A3-R',['R','post_R_tail'])]:
        z=piv[piv.index.get_level_values('stage').isin(stages)]
        check('factor_independence_'+a+'_'+b,np.allclose(z[a],z[b],equal_nan=True))
    return tab,pd.DataFrame(out)

def design_matrix(g,method):
    basin=g.basin_genesis.fillna('UNKNOWN').astype(str);basin=basin.where(basin.isin(['EP','NA','NI','SI','SP','WP','SA']),'UNKNOWN')
    z={f'basin_{b}':(basin==b).astype(float).to_numpy() for b in ['NA','NI','SI','SP','WP','SA','UNKNOWN']}
    for name,val in [('latitude',g.lat),('latitude2',g.lat**2),('lon_sin',g.lon_sin),('lon_cos',g.lon_cos),('month_sin',g.month_sin),('month_cos',g.month_cos),('genesis_year',g.genesis_year)]: z[name]=val.to_numpy(float)
    if method=='actual_DE':z['actual_E_indicator']=g.E.to_numpy(float)
    if method=='time_midpoint':z['time_midpoint_indicator']=(g.u>=.5).to_numpy(float)
    if method=='continuous_progress':z['progress']=g.u.to_numpy(float);z['progress2']=g.u.to_numpy(float)**2
    if method=='continuous_age':z['age_days']=g.age_h_a3.to_numpy(float)/24;z['age_days2']=(g.age_h_a3.to_numpy(float)/24)**2
    return np.column_stack(list(z.values())),list(z)

def a3(d,all_sids):
    d=d.copy();d['age_h_a3']=(d.UTC-d.genesis_time).dt.total_seconds()/3600
    d['u']=d.age_h_a3/((d.maturity_time-d.genesis_time).dt.total_seconds()/3600)
    valid=np.isfinite(d[['lat','lon','month_sin','month_cos','lon_sin','lon_cos','genesis_year','u','age_h_a3']].to_numpy(float)).all(axis=1)
    for var in ['X2_shear_ms','X3_mid_rh_pct']:
        d['usable_'+var]=bools(d[var+'_qc'])&np.isfinite(d[var+'_value'])&(d[var+'_valid_fraction']>=.9)
    de=d.DE&d.legal; support=de&valid&d.usable_X2_shear_ms&d.usable_X3_mid_rh_pct&(d.u>=0)&(d.u<1)
    d['a3_support']=support;d['a3_exclusion']=np.select([~d.DE,~d.legal,~valid,~d.usable_X2_shear_ms,~d.usable_X3_mid_rh_pct],[ 'outside_DE','illegal_interval','nonfinite_covariates','shear_QC_or_missing','RH_QC_or_missing'],default='included')
    d[['sid','UTC','D','E','legal','a3_support','a3_exclusion','usable_X2_shear_ms','usable_X3_mid_rh_pct']].to_parquet(BASE/'A3_row_support_ledger.parquet',index=False)
    counts=d.groupby('sid').agg(DE_legal_n=('DE',lambda x:0))
    counts['DE_legal_n']=d[de].groupby('sid').size();counts['common_n']=d[support].groupby('sid').size();counts['shear_only_usable_n']=d[de&valid&d.usable_X2_shear_ms].groupby('sid').size();counts['RH_only_usable_n']=d[de&valid&d.usable_X3_mid_rh_pct].groupby('sid').size()
    for st in ['D','E']:counts[st+'_common_n']=d[support&d[st]].groupby('sid').size()
    counts=counts.fillna(0).astype(int);counts['fold']=[fold(sid) for sid in counts.index];counts.reset_index().to_csv(BASE/'A3_SID_support.csv',index=False)
    g=d[support].copy().reset_index(drop=True);g['fold']=g.sid.map(fold);g['weight']=g.valid_h/g.groupby('sid').valid_h.transform('sum')
    check('A3_eachSIDweight1',np.allclose(g.groupby('sid').weight.sum(),1));check('A3_all5folds',set(g.fold)==set(range(5)))
    preds=[];models=[];membership=[];oof=np.full((len(g),5,2),np.nan)
    targets=['X2_shear_ms','X3_mid_rh_pct']
    for k in range(5):
        train=g.fold!=k;test=g.fold==k;trids=set(g.loc[train,'sid']);teids=set(g.loc[test,'sid'])
        check('A3_disjoint_fold'+str(k),not trids&teids)
        membership.extend({'fold':k,'sid':sid,'role':'train' if sid in trids else 'heldout'} for sid in sorted(trids|teids))
        for mi,method in enumerate(METHODS):
            X,names=design_matrix(g,method)
            for yi,var in enumerate(targets):
                p,m=ridge(X[train],g.loc[train,var+'_value'].to_numpy(float),g.loc[train,'weight'].to_numpy(float),X[test],names)
                oof[test,mi,yi]=p;m.update(fold=k,method=method,response=var,train_sid_n=len(trids),test_sid_n=len(teids),train_rows=int(train.sum()),test_rows=int(test.sum()));models.append(m)
    check('A3_no_missing_prediction',np.isfinite(oof).all())
    check('A3_actual_midpoint_df_equal',all(len(m['columns'])==15 for m in models if m['method'] in ['actual_DE','time_midpoint']))
    dump('A3_fitted_models.json',models);pd.DataFrame(membership).to_csv(BASE/'A3_fold_membership.csv',index=False)
    p=g[['sid','UTC','fold','weight','valid_h','D','E','u','age_h_a3','lat','lon','basin_genesis','genesis_year','month_sin','month_cos','lon_sin','lon_cos']].copy()
    for yi,var in enumerate(targets):
        p[var+'_observed']=g[var+'_value'].to_numpy(float)
        for mi,method in enumerate(METHODS):p[var+'__'+method]=oof[:,mi,yi]
    p.to_parquet(BASE/'A3_OOF_predictions.parquet',index=False)
    loss=[]
    for yi,var in enumerate(targets):
        y=g[var+'_value'].to_numpy(float)
        for mi,method in enumerate(METHODS):
            err=y-oof[:,mi,yi];q=pd.DataFrame({'sid':g.sid,'mse':g.weight*err**2,'mae':g.weight*abs(err)}).groupby('sid').sum()
            for sid,r in q.iterrows():loss.append({'sid':sid,'response':var,'method':method,'mse':r.mse,'mae':r.mae,'fold':fold(sid)})
    losses=pd.DataFrame(loss);losses.to_csv(BASE/'A3_SID_loss_bank.csv',index=False)
    sids=sorted(g.sid.unique());draw=bootstrap_indices(sids,620112026,'A3');stats=[];drawstats=[];aggregate=[]
    for var in targets:
        q=losses[losses.response==var].pivot(index='sid',columns='method',values='mse').reindex(sids)
        mae=losses[losses.response==var].pivot(index='sid',columns='method',values='mae').reindex(sids)
        means=q.to_numpy()[draw].mean(axis=1);names=q.columns.tolist();actual=q.actual_DE.to_numpy();ai=names.index('actual_DE')
        for method in METHODS:
            aggregate.append({'response':var,'method':method,'sid_n':len(sids),'row_n':len(g),'mse':q[method].mean(),'rmse':np.sqrt(q[method].mean()),'mae':mae[method].mean()})
        for ref in METHODS:
            if ref=='actual_DE':continue
            ri=names.index(ref);delta=means[:,ai]-means[:,ri];rdelta=np.sqrt(means[:,ai])-np.sqrt(means[:,ri]);ci=np.percentile(delta,[2.5,97.5]);rci=np.percentile(rdelta,[2.5,97.5])
            stats.append({'response':var,'contrast':'actual_DE-'+ref,'sid_n':len(sids),'row_n':len(g),'delta_mse':float(np.mean(actual-q[ref].to_numpy())),'ci_low':ci[0],'ci_high':ci[1],'delta_rmse':float(np.sqrt(q.actual_DE.mean())-np.sqrt(q[ref].mean())),'rmse_ci_low':rci[0],'rmse_ci_high':rci[1],'delta_mae':float((mae.actual_DE-mae[ref]).mean()),'B':2000,'fixed_OOF':True})
            drawstats.extend({'draw':j,'response':var,'contrast':'actual_DE-'+ref,'delta_mse':delta[j],'delta_rmse':rdelta[j]} for j in range(2000))
    pd.DataFrame(stats).to_csv(BASE/'A3_paired_contrasts.csv',index=False);pd.DataFrame(aggregate).to_csv(BASE/'A3_aggregate_losses.csv',index=False);pd.DataFrame(drawstats).to_parquet(BASE/'A3_bootstrap_estimates.parquet',index=False)
    # Independent score reconstruction uses saved prediction columns, not the fit routine.
    saved=pd.read_parquet(BASE/'A3_OOF_predictions.parquet')
    for var in targets:
        for method in METHODS:
            rec=((saved[var+'_observed']-saved[var+'__'+method])**2*saved.weight).groupby(saved.sid).sum()
            bank=losses[(losses.response==var)&(losses.method==method)].set_index('sid').mse
            check('A3_saved_prediction_loss_'+var+'_'+method,np.allclose(rec.sort_index(),bank.sort_index(),rtol=1e-12,atol=1e-12))
    return pd.DataFrame(stats),pd.DataFrame(aggregate),counts
