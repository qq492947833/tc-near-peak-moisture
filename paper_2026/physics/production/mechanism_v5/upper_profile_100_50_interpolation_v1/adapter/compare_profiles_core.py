"""Fixed four-configuration finite profile comparison; no fits or physical release."""
import numpy as np
from pi_endpoint_trace import TracedPI,PARAMETERS
P13=np.array([1000,925,850,700,600,500,400,300,250,200,150,100,50.],dtype=float)
P14=np.array([1000,925,850,700,600,500,400,300,250,200,150,100,70,50.],dtype=float)
CONFIGURATIONS={
 'A_13_original':dict(P='pressure_hpa',T='temperature_K',q='specific_humidity_kgkg',support='joint_Tq_valid_fraction',inclusive=False,interpolation='none'),
 'B_14_logp_original':dict(P='pressure_aug_hpa',T='T_logp_K',q='q_logp_kgkg',support='joint_fraction_aug',inclusive=False,interpolation='linear_in_log_pressure_on_variable_value'),
 'C_14_logp_inclusive50':dict(P='pressure_aug_hpa',T='T_logp_K',q='q_logp_kgkg',support='joint_fraction_aug',inclusive=True,interpolation='linear_in_log_pressure_on_variable_value'),
 'D_14_plinear_inclusive50':dict(P='pressure_aug_hpa',T='T_plinear_K',q='q_plinear_kgkg',support='joint_fraction_aug',inclusive=True,interpolation='linear_in_pressure_on_variable_value_single_sensitivity')}

def evaluate_profile(arrays,SST_K,msl_Pa):
 results={};summaries=[];pathrows=[]
 for name,cfg in CONFIGURATIONS.items():
  source=dict(configuration=name,interpolation=cfg['interpolation'],endpoint_inclusive=cfg['inclusive'],PI_parameters=dict(PARAMETERS))
  try:
   p,t,q,area=(np.asarray(arrays[cfg[k]],dtype=float).copy() for k in ('P','T','q','support'))
   expected=P13 if name.startswith('A_') else P14
   if not np.array_equal(p,expected):raise ValueError('INPUT_LAYER_LIST_DIFFERS_FROM_FROZEN13_OR14')
   if area.shape!=p.shape or not np.all(np.isfinite(area)&(area>=.9)&(area<=1+1e-12)):raise ValueError('REQUIRED_LAYER_ORIGINAL_AREA_SUPPORT_FAIL_NO_PREFIX_REMOVAL')
   calc=TracedPI(cfg['inclusive']);r=calc.run(t,q,p,float(SST_K),float(msl_Pa))
   if r['parameters']!=PARAMETERS:raise AssertionError('fixed parameter identity')
   source.update(input_pressure_hpa=p,input_joint_area_fraction=area,input_T_K=t,input_q_kgkg=q,SST_K=float(SST_K),msl_Pa=float(msl_Pa))
   r.update(source);results[name]=r
   row=dict(configuration=name,execution='CALCULATED',numerical_success=r['numerical_success'],physical_acceptance='PENDING_INDEPENDENT_REVIEW',**r['outputs'],n_CAPE_calls=len(r['all_CAPE_calls']),n_any_iteration_top_positive=sum(z['top_positive_buoyancy'] for z in r['all_CAPE_calls']),n_any_iteration_top_exactly_neutral=sum(z['top_exactly_neutral'] for z in r['all_CAPE_calls']))
   for path in ('CAPEA','CAPEM','CAPEMS'):
    calls=[z for z in r['all_CAPE_calls'] if z['path']==path];last=calls[-1] if calls else None
    row[path+'_final_top_status']=last['top_boundary_status'] if last else 'NO_CALL'
    row[path+'_final_zero_interior']=last['selected_zero_strictly_interior'] if last else False
    row[path+'_final_top_hpa']=last['actual_top_hpa'] if last else np.nan
    row[path+'_final_top_TVRDIF_K']=last['top_TVRDIF_K'] if last else np.nan
    row[path+'_final_CAPED_jkg']=last['CAPED_jkg'] if last else np.nan
   for call in r['all_CAPE_calls']:
    bracket=call['selected_LNB_bracket'] or {}
    pathrows.append(dict(configuration=name,call_index=call['call_index'],path=call['path'],iteration=call['iteration'],final_used_iteration=call['final_used_iteration'],parcel_T_K=call['parcel_T_K'],parcel_r_kgkg=call['parcel_r_kgkg'],parcel_P_hpa=call['parcel_P_hpa'],actual_top_hpa=call['actual_top_hpa'],top_TVRDIF_K=call['top_TVRDIF_K'],top_positive_buoyancy=call['top_positive_buoyancy'],top_exactly_neutral=call['top_exactly_neutral'],selected_zero_bracketed=call['selected_zero_bracketed'],selected_zero_strictly_interior=call['selected_zero_strictly_interior'],top_boundary_status=call['top_boundary_status'],CAPED_jkg=call['CAPED_jkg'],CAPED_before_zero_clip_jkg=call['CAPED_before_zero_clip_jkg'],TOB_K=call['TOB_K'],LNB_hpa=call['LNB_hpa'],IFLAG=call['IFLAG'],lower_crossing_pressure_hpa=bracket.get('lower_pressure_hpa'),upper_crossing_pressure_hpa=bracket.get('upper_pressure_hpa'),zero_pressure_hpa=bracket.get('zero_pressure_hpa')))
  except (ValueError,KeyError) as exc:
   r=dict(source,execution='INPUT_REJECTED_NO_CALCULATION',failure_reason=str(exc),numerical_success=False,physical_acceptance='NOT_SUPPORTED_BY_INPUT',all_CAPE_calls=[])
   results[name]=r;row=dict(configuration=name,execution=r['execution'],failure_reason=str(exc),numerical_success=False,physical_acceptance=r['physical_acceptance'],PI_ms=np.nan,PMIN_hpa=np.nan,IFL=np.nan,TO_K=np.nan,OTL_hpa=np.nan,n_CAPE_calls=0)
  summaries.append(row)
 return results,summaries,pathrows
