"""Deterministic output curation from saved diagnostics/statistics, no model fitting."""
from pathlib import Path
import json,os,hashlib
from configuration import OUTPUT_ROOT as ROOT, initialize_output
for k in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR'):os.environ[k]=str(ROOT/'tmp')
import numpy as np,pandas as pd
DEFINITIONS={
 'KE':'Horizontal kinetic-energy reservoir',
 'APE':'Dry available-potential-energy reservoir about the frozen reference state',
 'C_A_to_K':'Same-window dry APE-to-horizontal-KE conversion; positive supplies KE',
 'Gamma_K_h':'All-source horizontal contribution to canonical horizontal-KE transfer',
 'Gamma_K_p':'All-source pressure-direction contribution to canonical horizontal-KE transfer',
 'Gamma_K_total':'Sum of horizontal and pressure-direction canonical horizontal-KE transfer',
 'Gamma_A_h':'All-source horizontal contribution to canonical dry-APE transfer',
 'Gamma_A_p':'All-source pressure-direction contribution to canonical dry-APE transfer',
 'Gamma_A_total':'Sum of horizontal and pressure-direction canonical dry-APE transfer',
 'APE_ref_correction':'Separate dry-APE reference-gradient correction, one-half T_hat MWT(omega T-prime) dc/dp',
 'APE_compressibility':'Separate dry-APE thermodynamic reference term R/(p Tbar) T_hat MWT(omega T-prime)',
 'vertical_input_K_Wkg':'Net vertical KE input through 500/850-hPa boundaries per unit layer mass',
 'vertical_input_A_Wkg':'Net vertical dry-APE input through 500/850-hPa boundaries per unit layer mass',
 'vertical_input_P_Wkg':'Net vertical pressure-flux input through 500/850-hPa boundaries per unit layer mass'}
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x
def save(p,x):Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf8')
def require_fields(frame,fields,label):
    missing=sorted(set(fields)-set(frame.columns))
    if missing:raise AssertionError(f'{label}: missing required source fields: {missing}')

def export_support_ledgers(unit_values,matched_values,raw):
    """Retain complete frozen matched units; derive statuses without inference."""
    identity=['comparison','variant','window','region','level','metric']
    directions={'pre_event':'temporary departure minus start of decay',
        'wind_matched_raw':'reintensification minus decay at matched wind',
        'rise_mean':'actual valley-to-return time mean',
        'rise_rate_integral':'actual valley-to-return trapezoid integral in seconds',
        'rise_endpoint_change':'return minus valley'}
    def describe(frame):
        frame=frame.copy()
        frame['definition']=frame.metric.map(DEFINITIONS)
        frame['scale']=frame.window.map({0:'>16 d',1:'2–16 d',2:'<2 d'})
        frame['direction']=frame.comparison.map(directions)
        assert frame[['definition','scale','direction']].notna().all().all()
        return frame
    units=describe(unit_values)
    units['source_path']=str(ROOT/'statistics/unit_support_and_values.parquet')
    units.to_csv(ROOT/'outputs/UNIT_SUPPORT_AND_VALUES.csv.gz',index=False,encoding='utf8',float_format='%.17g',compression='gzip')
    excluded=units[~np.isfinite(units.value)].copy()
    excluded.to_csv(ROOT/'outputs/SUPPORT_EXCLUSIONS.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    matched=matched_values.copy();matched['comparison']='wind_matched_raw'
    require_fields(matched,[*identity,'sid','event_id','pair_id','value','valid','failure_reason'],'complete matched support')
    if matched.duplicated([*identity,'pair_id']).any():raise AssertionError('Repeated frozen matched unit')
    group_sizes=matched.groupby(identity,dropna=False).size()
    if not group_sizes.eq(165).all():raise AssertionError('Each matched diagnostic must retain all 165 frozen matched units')
    if not np.array_equal(matched.valid.to_numpy(bool),np.isfinite(matched.value)):
        raise AssertionError('Matched validity differs from finite target values')
    target_meta=raw.loc[raw.comparison.eq('wind_matched_raw'),[*identity,'unit']]
    if target_meta.duplicated(identity).any():raise AssertionError('Repeated matched summary identity')
    matched=matched.merge(target_meta,on=identity,how='left',validate='many_to_one')
    assert matched.unit.notna().all()
    matched=describe(matched);matched['source_path']=str(ROOT/'statistics/matched_unit_values.parquet')
    matched.to_csv(ROOT/'outputs/MATCHED_UNIT_SUPPORT_AND_VALUES.csv.gz',index=False,encoding='utf8',float_format='%.17g',compression='gzip')
    matched_excluded=matched[~matched.valid].copy()
    matched_excluded.to_csv(ROOT/'outputs/MATCHED_SUPPORT_EXCLUSIONS.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    # The existing wind-matched event table only contains events with at least
    # one valid bin. Use all matched rows for complete event and SID accounting.
    eligible=units[units.comparison.ne('wind_matched_raw')].copy()
    eligible['unit_kind']=np.where(eligible.comparison.eq('pre_event'),'paired_event_target','return_event_target')
    matched['unit_kind']='original_shared_wind_bin'
    ledger=pd.concat([eligible,matched],ignore_index=True)
    ledger['valid_unit']=np.isfinite(ledger.value)
    if not np.array_equal(ledger.valid.to_numpy(bool),ledger.valid_unit):
        raise AssertionError('Saved event validity differs from finite target values')
    ledger['unavailable_reason']=ledger.failure_reason.fillna('').where(~ledger.valid_unit,'')
    group_keys=[*identity,'unit','unit_kind','sid','event_id']
    event=ledger.groupby(group_keys,dropna=False,sort=True).agg(
        n_eligible_units=('valid_unit','size'),n_valid_units=('valid_unit','sum'),
        unavailable_reasons=('unavailable_reason',lambda x:'; '.join(sorted(set(str(v) for v in x if v)))),
        source_path=('source_path','first')).reset_index()
    event['n_invalid_units']=event.n_eligible_units-event.n_valid_units
    event['included_in_estimate']=event.n_valid_units.gt(0)
    event['support_status']=np.select([event.n_valid_units.eq(0),event.n_valid_units.eq(event.n_eligible_units)],['EXCLUDED','FULLY_RETAINED'],default='PARTIALLY_RETAINED')
    event=describe(event)
    event.to_csv(ROOT/'outputs/TARGET_EVENT_SUPPORT.csv.gz',index=False,encoding='utf8',compression='gzip')
    sid=event.groupby([*identity,'unit','unit_kind','sid'],dropna=False,sort=True).agg(
        n_eligible_event_targets=('event_id','size'),
        n_retained_event_targets=('included_in_estimate','sum'),
        n_eligible_units=('n_eligible_units','sum'),n_valid_units=('n_valid_units','sum'),
        unavailable_reasons=('unavailable_reasons',lambda x:'; '.join(sorted(set(str(v) for v in x if v)))),
        source_path=('source_path','first')).reset_index()
    sid['n_invalid_units']=sid.n_eligible_units-sid.n_valid_units
    sid['n_excluded_event_targets']=sid.n_eligible_event_targets-sid.n_retained_event_targets
    sid['included_in_estimate']=sid.n_retained_event_targets.gt(0)
    sid['support_status']=np.select([sid.n_valid_units.eq(0),sid.n_valid_units.eq(sid.n_eligible_units)],['EXCLUDED','FULLY_RETAINED'],default='PARTIALLY_RETAINED')
    sid=describe(sid)
    sid.to_csv(ROOT/'outputs/TARGET_SID_SUPPORT.csv.gz',index=False,encoding='utf8',compression='gzip')
    sid_n=sid.groupby(identity,dropna=False).included_in_estimate.sum().rename('ledger_n_sid')
    event_n=event.groupby(identity,dropna=False).included_in_estimate.sum().rename('ledger_n_events')
    check=raw.merge(pd.concat([sid_n,event_n],axis=1).reset_index(),on=identity,how='left',validate='one_to_one')
    check.loc[check.comparison.eq('pre_event'),'ledger_n_events']*=2
    failures=check[~check.n_sid.eq(check.ledger_n_sid)|~check.n_events.eq(check.ledger_n_events)]
    qa={'status':'PASS' if failures.empty else 'FAIL','matched_rows':len(matched),
        'matched_targets':len(group_sizes),'matched_units_per_target':165,
        'matched_unavailable_rows':len(matched_excluded),'event_target_rows':len(event),
        'sid_target_rows':len(sid),'partially_retained_sid_target_rows':int(sid.support_status.eq('PARTIALLY_RETAINED').sum()),
        'summary_count_checks':len(check),'summary_count_mismatches':clean(failures[[*identity,'n_sid','ledger_n_sid','n_events','ledger_n_events']].to_dict('records')),
        'status_definition':'EXCLUDED means no valid unit; PARTIALLY_RETAINED means at least one valid and one invalid unit, and the SID/event remains included; FULLY_RETAINED means all eligible units valid.',
        'pre_event_count_definition':'One ledger event target represents one within-SID paired contrast; raw n_events counts both event-class members.'}
    save(ROOT/'audit/SUPPORT_LEDGER_QA.json',qa)
    if not failures.empty:raise AssertionError('Support ledgers disagree with saved summary sample counts')
    return units,excluded,matched,matched_excluded,event,sid,qa
def section(title,df,cols,note=''):
    columns=[]
    for key in cols:
        typ='integer' if key.startswith('n_') or key in ('window','family_size') else 'number' if key in ('estimate','ci_lower','ci_upper','p','q','temporary_mean','decay_mean','rise_mean') else 'text'
        width=16 if typ!='text' else 24
        if key in ('definition','direction','source_path','reason'):width=55
        columns.append(dict(key=key,label=key.replace('_',' '),type=typ,width=width))
    for k in cols:
        if k not in df:df=df.assign(**{k:None})
    return dict(title=title,note=note,columns=columns,rows=clean(df[cols].replace({np.nan:None}).to_dict('records')))
def main():
    initialize_output()
    raw=pd.read_parquet(ROOT/'statistics/raw_summary.parquet')
    require_fields(raw,['comparison','variant','metric','window','level','region','unit',
        'n_sid','n_events','estimate','ci_lower','ci_upper','p','q','family_id',
        'n_attempted','n_estimable','n_failed','direction','source_path'],'raw statistics')
    raw['definition']=raw.metric.map(DEFINITIONS)
    assert raw.definition.notna().all()
    raw['scale']=raw.window.map({0:'>16 d',1:'2–16 d',2:'<2 d'})
    raw['spatial_representation']='per unit mass'
    raw.to_csv(ROOT/'outputs/MWT_ALL_STATISTICS.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    unit_values=pd.read_parquet(ROOT/'statistics/unit_support_and_values.parquet')
    matched_values=pd.read_parquet(ROOT/'statistics/matched_unit_values.parquet')
    unit_values,excluded,matched_values,matched_excluded,event_ledger,sid_ledger,support_qa=export_support_ledgers(unit_values,matched_values,raw)
    # Unit-area representation is a deterministic rescaling of the same layer
    # estimand and paired draws, not a new test family or additional evidence.
    factor=35000./9.8
    col=raw[raw.level.eq('500-850')].copy();col['spatial_representation']='pressure-column integral (dp/g)'
    for name in ['estimate','ci_lower','ci_upper','temporary_mean','temporary_ci_lower','temporary_ci_upper','decay_mean','decay_ci_lower','decay_ci_upper','rise_mean','rise_ci_lower','rise_ci_upper']:
        if name in col:col[name]*=factor
    col['unit']=col.unit.map({'W kg-1':'W m-2','J kg-1':'J m-2'})
    col['definition']='Pressure-column representation of: '+col.definition
    col['column_mass_kg_m2']=factor
    col.to_csv(ROOT/'outputs/MWT_COLUMN_INTEGRALS.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    olddraw=np.load(ROOT/'statistics/raw_bootstrap_draws.npz')
    colkeys=set(col.draw_key)
    np.savez_compressed(ROOT/'statistics/column_integral_draws.npz',**{key:olddraw[key]*factor for key in olddraw.files if key[:24] in colkeys})
    pd.DataFrame([dict(metric=k,definition=v,level_mean_to_column_mass=factor) for k,v in DEFINITIONS.items()]).to_csv(ROOT/'outputs/METRIC_DEFINITIONS.csv',index=False,encoding='utf-8-sig')
    adj=pd.read_parquet(ROOT/'statistics/adjusted/ADJUSTED_WIND_MATCHED.parquet').rename(columns={'lower':'ci_lower','upper':'ci_upper','p_centered':'p','q_BH':'q','valid_bootstrap':'n_estimable','failed_bootstrap':'n_failed','n_bin_pairs':'n_matched_units','source_file':'source_path'})
    require_fields(adj,['target','metric','window','n_sid','n_events','estimate','ci_lower',
        'ci_upper','p','q','family_id','n_attempted','n_estimable','n_failed','source_path'],
        'adjusted statistics')
    adj['definition']=adj.metric.map(DEFINITIONS);adj['scale']=adj.window.map({0:'>16 d',1:'2–16 d'});adj['level']='500-850';adj['unit']='W kg-1';adj['direction']='reintensification minus decay at matched wind';adj['region']='r0_300'
    assert adj.definition.notna().all(),'Unknown adjusted diagnostic definition'
    adj.to_csv(ROOT/'outputs/MWT_ADJUSTED_WIND_MATCHED.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    old_support_path=ROOT/'statistics/adjusted/OLD_METRICS_ON_NEW_SUPPORT.csv'
    (ROOT/'outputs/OLD_METRICS_ON_NEW_SUPPORT.csv').write_bytes(old_support_path.read_bytes())
    old_support=pd.read_csv(old_support_path,float_precision='round_trip')
    old_support['scale']=old_support.window.map({0:'>16 d',1:'2–16 d'})
    old_support['definition']=old_support.metric.map({**DEFINITIONS,'APE':'Dry available potential energy from the original 700-hPa diagnostic'})
    old_support['comparison']='reintensification minus decay at matched wind'
    old_support['summary_source_path']=str(old_support_path)
    # Complete CSV views have explicit estimand, support, units and source fields.
    for comp,file in [('pre_event','PRE_EVENT'),('rise_mean','REINTENSIFICATION_MEANS'),('rise_endpoint_change','REINTENSIFICATION_STORAGE_CHANGES'),('rise_rate_integral','REINTENSIFICATION_RATE_INTEGRALS'),('wind_matched_raw','WIND_MATCHED_RAW')]:
        raw[raw.comparison.eq(comp)].to_csv(ROOT/'outputs'/f'{file}.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    sens_path=ROOT/'statistics/sensitivity_summary.csv'
    sens=pd.read_csv(sens_path,float_precision='round_trip') if sens_path.exists() else pd.DataFrame()
    require_fields(sens,['sensitivity_id','alternative_variant','reference_variant',
        'alternative_level','reference_level','metric','window','region','comparison',
        'unit','n_sid','n_events','estimate','ci_lower','ci_upper','p','q','family_id',
        'n_attempted','n_estimable','n_failed','direction','source_path'],'numerical sensitivity')
    sens['definition']=sens.metric.map(DEFINITIONS)
    sens['scale']=sens.window.map({0:'>16 d',1:'2–16 d',2:'<2 d'})
    assert sens.definition.notna().all(),'Unknown sensitivity diagnostic definition'
    assert sens.scale.notna().all(),'Unknown sensitivity scale window'
    sens.to_csv(ROOT/'outputs/NUMERICAL_SENSITIVITY.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    save(ROOT/'audit/CURATION_QA.json',dict(raw_rows=len(raw),column_rows=len(col),adjusted_rows=len(adj),sensitivity_rows=len(sens),column_mass_factor=factor,three_windows=sorted(raw.window.unique().tolist())))
    print(json.dumps(dict(raw=len(raw),column=len(col),adjusted=len(adj),sensitivity=len(sens))))
if __name__=='__main__':main()
