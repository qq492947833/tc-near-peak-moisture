def fit_trend(years,values,metric,maxlags=3,per_decade=True,min_n=8):
    years=np.asarray(years,dtype=float);values=np.asarray(values,dtype=float)
    valid=np.isfinite(values)&np.isfinite(years);x=years[valid];y=values[valid]
    scale=(10 if per_decade else 1)*(100 if metric in PROPORTIONS else 1)
    unit=('pp' if metric in PROPORTIONS else 'h' if metric.endswith('_h') else 'kt' if metric=='LMI_kt' else 'cyclones' if metric=='complex_count' else 'A')+('/decade' if per_decade else '/year')
    row=dict(metric=metric,n_years=len(x),start_year=int(x.min()) if len(x) else np.nan,end_year=int(x.max()) if len(x) else np.nan,missing_years=';'.join(str(int(v)) for v in years[~valid]),maxlags=maxlags,use_correction=True,use_t=True,kernel='Bartlett',df=len(x)-2,unit=unit,status='ok' if len(x)>=min_n else 'insufficient_years')
    if len(x)<min_n:return {**row,'slope':np.nan,'ci_low':np.nan,'ci_high':np.nan,'p_value':np.nan,'intercept':np.nan},pd.DataFrame()
    center=x.mean();X=np.column_stack([np.ones(len(x)),x-center]);fit=sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'maxlags':maxlags,'use_correction':True,'kernel':'bartlett'},use_t=True)
    ci=fit.conf_int(alpha=.05);cov=np.asarray(fit.cov_params());pred=X@fit.params
    se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,cov,X),0));crit=student_t.ppf(.975,len(x)-2)
    row.update(slope=fit.params[1]*scale,ci_low=ci[1,0]*scale,ci_high=ci[1,1]*scale,p_value=fit.pvalues[1],intercept=fit.params[0],year_center=center,slope_native_per_year=fit.params[1],cov_00=cov[0,0],cov_01=cov[0,1],cov_11=cov[1,1],t_critical=crit)
    fitted=pd.DataFrame({'metric':metric,'year':x.astype(int),'observed':y,'fitted':pred,'mean_ci_low':pred-crit*se,'mean_ci_high':pred+crit*se})
    return row,fitted
