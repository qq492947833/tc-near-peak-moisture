"""Environmental-profile PI with per-call physical QC and signed ventilation.

The profile is the area mean, not the mean of gridpoint PI. Missing required
native levels remain missing. Failure at a physical gate is retained explicitly.
The primary calculation is log-pressure 70-hPa interpolation, inclusive 50 hPa;
the only alternate calculation is pressure-linear 70-hPa interpolation.
"""
from pathlib import Path
import ast, copy
import numpy as np
from .pi_trace import TracedPI, PARAMETERS, UP, PINNED_SHA, sha
from .pi_profile import P13, P14
from .entropy import pseudoadiabatic_entropy, assemble_ventilation
HERE=Path(__file__).resolve().parent

class TracedRMW(TracedPI):

    def __init__(self):
        super().__init__(include_endpoint=True)
        self.pi_return_state = {}
        tree = ast.parse(Path(UP.__file__).read_text(encoding='utf-8'))
        fn = copy.deepcopy(next((x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_pi_numba')))
        fn.decorator_list = []
        calls = sorted([n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and (n.func.id == 'cape')], key=lambda n: n.lineno)
        assert len(calls) == 3
        for call, label in zip(calls, ('CAPEA', 'CAPEM', 'CAPEMS')):
            call.func.id = '_call_cape'
            call.args = [ast.Constant(label), ast.Call(func=ast.Attribute(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]), attr='get', ctx=ast.Load()), args=[ast.Constant('NP'), ast.Constant(-1)], keywords=[])] + call.args

        class ReturnOnly(ast.NodeTransformer):

            def visit_Return(self, node):
                return ast.Return(value=ast.Call(func=ast.Name(id='_return_pi', ctx=ast.Load()), args=[node.value, ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[]))
        fn = ReturnOnly().visit(fn)
        ns = dict(vars(UP))
        ns.update(_call_cape=self.call_cape, _return_pi=self.return_pi)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(HERE / 'production_thermodynamics.py') + '::return_only', 'exec'), ns)
        self.pi_numeric = ns['_pi_numba']

    def return_pi(self, outputs, state):
        self.pi_return_state = {key: state.get(key, np.nan) for key in ('PM', 'PNEW', 'NP', 'RS0')}
        return outputs

    def run(self, *args, **kwargs):
        self.pi_return_state = {}
        r = super().run(*args, **kwargs)
        r['pi_return_state'] = self.pi_return_state
        final = {p: next((c for c in reversed(r['all_CAPE_calls']) if c['path'] == p), None) for p in ('CAPEA', 'CAPEM', 'CAPEMS')}
        m, s = (final['CAPEM'], final['CAPEMS'])
        rmw = {}
        if m and s:
            rmw = dict(rmw_pressure_used_hpa=m['parcel_P_hpa'], rmw_temperature_k=m['parcel_T_K'], rmw_mixing_ratio_kgkg=m['parcel_r_kgkg'], rmw_saturated_mixing_ratio_kgkg=s['parcel_r_kgkg'], rmw_pressure_next_hpa=self.pi_return_state.get('PM', np.nan), rmw_iterations=self.pi_return_state.get('NP', np.nan), same_iteration=m['iteration'] == s['iteration'], same_pressure=m['parcel_P_hpa'] == s['parcel_P_hpa'], parcel_temperatures_equal=m['parcel_T_K'] == s['parcel_T_K'], environment_temperature_matches_lowest=m['parcel_T_K'] == float(np.asarray(args[0])[0]), saturated_temperature_matches_SST=s['parcel_T_K'] == float(args[3]))
        r['rmw_actual_final_state'] = rmw
        return r

def call_gate(call):
    reasons = []
    p = np.asarray(call.get('effective_P_hpa', []), float)
    d = np.asarray(call.get('density_temperature_difference_profile_K', []), float)
    levels = call.get('levels', [])
    by = {int(x['index']): x for x in levels}
    if call.get('IFLAG') != 1:
        reasons.append('CALL_IFLAG_NOT_1')
    if not np.array_equal(p, P14):
        reasons.append('CALL_DISCRETE_PRESSURE_NOT_FIXED14')
    if d.shape != p.shape or not np.all(np.isfinite(d)):
        reasons.append('NONFINITE_OR_MISSING_BUOYANCY_NODES')
    if set(by) != set(range(len(p))):
        reasons.append('MISSING_ACTUALLY_EVALUATED_NODES')
    for x in levels:
        if not all((np.isfinite(x.get(k, np.nan)) for k in ('parcel_temperature_k', 'parcel_mixing_ratio_kgkg', 'parcel_density_temperature_k', 'environment_density_temperature_k', 'density_temperature_difference_k', 'buoyancy_ms2'))):
            reasons.append('NONFINITE_ACTUAL_LEVEL_STATE')
            break
    if len(p) and len(d) == len(p) and all((j in by for j in range(len(p)))):
        if not np.array_equal(d, np.array([by[j]['density_temperature_difference_k'] for j in range(len(p))])):
            reasons.append('BUOYANCY_TRACE_DISAGREEMENT')
        if d[-1] > 0:
            reasons.append('TOP_POSITIVE_UNBRACKETED')
        if d[-1] == 0:
            reasons.append('NEUTRAL_AT_BOUNDARY')
    else:
        reasons.append('UNRESOLVED_TOP')
    if reasons:
        return dict(accepted=False, status='UNSUPPORTED', reasons=sorted(set(reasons)))
    positive = np.flatnonzero(d[1:] > 0) + 1
    zero_branch = len(positive) == 0 and call.get('INB') == 0 and (call.get('LNB_hpa') == 0) and (call.get('CAPED_jkg') == 0)
    if zero_branch:
        okay = call.get('path') in ('CAPEA', 'CAPEM')
        return dict(accepted=okay, status='NO_RESOLVED_POSITIVE_ABOVE_FIRST_ZERO_SENTINEL' if okay else 'CAPEMS_NO_POSITIVE_OUTFLOW', reasons=[] if okay else ['CAPEMS_REQUIRES_SUPPORTED_POSITIVE_OUTFLOW'], first_node_positive=bool(d[0] > 0), above_first_positive_count=0, lnb_is_real_outflow=False)
    if not len(positive):
        return dict(accepted=False, status='NO_POSITIVE_BRANCH_INCONSISTENT', reasons=['NO_POSITIVE_BUT_NOT_ORIGINAL_ZERO_BRANCH'])
    highest = int(positive[-1])
    b = call.get('selected_LNB_bracket')
    okay = highest < len(p) - 1 and d[highest + 1] <= 0 and (call.get('INB') == highest) and (b is not None) and (b['lower_index'] == highest) and (b['upper_index'] == highest + 1) and (p[-1] < b['zero_pressure_hpa'] < p[0]) and (call.get('selected_zero_strictly_interior') is True) and (call.get('LNB_hpa') == b['zero_pressure_hpa'])
    return dict(accepted=bool(okay), status='SUPPORTED_HIGHEST_POSITIVE_INTERIOR_ZERO' if okay else 'UNSUPPORTED_HIGHEST_POSITIVE_ZERO', reasons=[] if okay else ['HIGHEST_POSITIVE_BRACKET_OR_ZERO_NOT_SUPPORTED'], first_node_positive=bool(d[0] > 0), above_first_positive_count=int(len(positive)), lnb_is_real_outflow=bool(okay))

def physical_gate(result):
    calls = result.get('all_CAPE_calls', [])
    gates = [dict(call_index=c['call_index'], path=c['path'], iteration=c['iteration'], final_used_iteration=c['final_used_iteration'], **call_gate(c)) for c in calls]
    raw = result.get('outputs', {})
    reasons = []
    if not calls:
        reasons.append('NO_CAPE_CALLS')
    if any((not x['accepted'] for x in gates)):
        reasons.append('ONE_OR_MORE_ACTUAL_CAPE_CALLS_UNSUPPORTED')
    if set((c['path'] for c in calls)) != {'CAPEA', 'CAPEM', 'CAPEMS'}:
        reasons.append('MISSING_CAPE_PATH')
    if raw.get('IFL') != 1 or not all((np.isfinite(raw.get(k, np.nan)) for k in ('PI_ms', 'PMIN_hpa', 'IFL', 'TO_K', 'OTL_hpa'))) or raw.get('PI_ms', 0) <= 0:
        reasons.append('EXISTING_STRICT_NONPOSITIVE_NONFINITE_OR_IFL_FAILURE')
    rmw = result.get('rmw_actual_final_state', {})
    if not all((rmw.get(k) is True for k in ('same_iteration', 'same_pressure', 'environment_temperature_matches_lowest', 'saturated_temperature_matches_SST'))):
        reasons.append('FINAL_ACTUAL_RMW_PATH_MISMATCH')
    return dict(accepted=not reasons, reasons=reasons, calls=gates, criterion='All actual CAPE calls supported; zero sentinel only CAPEA/CAPEM; strictly positive PI')

def evaluate(arrays, scalars, wind_kt=np.nan):
    out = {}
    for name, Tkey, qkey in [('C_14_logp_inclusive50', 'T_logp_K', 'q_logp_kgkg'), ('D_14_plinear_inclusive50', 'T_plinear_K', 'q_plinear_kgkg')]:
        empty = dict(pi_vmax_ms=np.nan, pi_pmin_hpa=np.nan, pi_outflow_temperature_K=np.nan, pi_outflow_pressure_hpa=np.nan, thermal_headroom_ms=np.nan, raw_signed_index=np.nan, chi600=np.nan, entropy_deficit_jkgk=np.nan, sea_air_entropy_difference_jkgk=np.nan, ventilation_index=np.nan, pi_valid=False, signed_estimable=False, standard_ventilation_domain_valid=False)
        r = dict(configuration=name, authoritative=dict(empty), raw_trace=None)
        try:
            p13 = np.asarray(arrays['pressure_hpa'])
            p = np.asarray(arrays['pressure_aug_hpa'])
            t = np.asarray(arrays[Tkey])
            q = np.asarray(arrays[qkey])
            support = np.asarray(arrays['joint_fraction_aug'])
            if not np.array_equal(p13, P13) or not np.array_equal(p, P14):
                raise ValueError('FIXED_ORIGINAL13_OR_AUGMENTED14_LAYER_IDENTITY')
            ot = np.asarray(arrays['temperature_K'])
            oq = np.asarray(arrays['specific_humidity_kgkg'])
            of = np.asarray(arrays['joint_Tq_valid_fraction'])
            joint = np.isfinite(ot) & (ot > 100) & (ot < 400) & np.isfinite(oq) & (oq >= 0) & (oq < 0.1) & np.isfinite(of) & (of >= 0.9)
            for original, aug in [('temperature_K', Tkey), ('specific_humidity_kgkg', qkey), ('joint_Tq_valid_fraction', 'joint_fraction_aug')]:
                expected = np.asarray(arrays[original]) if original == 'joint_Tq_valid_fraction' else np.where(joint, np.asarray(arrays[original]), np.nan)
                if not np.array_equal(expected, np.delete(np.asarray(arrays[aug]), 12), equal_nan=True):
                    raise ValueError('ORIGINAL13_JOINT_QC_VALUES_NOT_PRESERVED')
            if not np.all(np.isfinite(support) & (support >= 0.9) & (support <= 1 + 1e-12)):
                raise ValueError('REQUIRED_ORIGINAL_AREA_SUPPORT_FAIL_NO_PREFIX_REMOVAL')
            calc = TracedRMW()
            raw = calc.run(t, q, p, float(scalars['SST_K']), float(scalars['msl_Pa']))
            r['raw_trace'] = raw
            gate = physical_gate(raw)
            r['physical_gate'] = gate
            if gate['accepted']:
                v = raw['outputs']
                rmw = raw['rmw_actual_final_state']
                pp = rmw['rmw_pressure_used_hpa']
                sb = float(pseudoadiabatic_entropy(rmw['rmw_temperature_k'], rmw['rmw_mixing_ratio_kgkg'], pp * 100))
                ss = float(pseudoadiabatic_entropy(scalars['SST_K'], rmw['rmw_saturated_mixing_ratio_kgkg'], pp * 100))
                pi = dict(vmax_ms=v['PI_ms'], pi_valid=True, sea_air_entropy_difference_jkgk=ss - sb)
                vi = assemble_ventilation(scalars.get('shear_200_850_ms', np.nan), scalars.get('core600_saturated_entropy_jkgk', np.nan), scalars.get('environment600_entropy_jkgk', np.nan), pi, support_confirmed=scalars.get('ventilation_support_confirmed') is True)
                r['authoritative'].update(pi_vmax_ms=v['PI_ms'], pi_pmin_hpa=v['PMIN_hpa'], pi_outflow_temperature_K=v['TO_K'], pi_outflow_pressure_hpa=v['OTL_hpa'], pi_valid=True, thermal_headroom_ms=v['PI_ms'] - float(wind_kt) * 1852 / 3600 if np.isfinite(wind_kt) else np.nan, entropy_bl_jkgk=sb, entropy_sst_sat_jkgk=ss, sea_air_entropy_difference_jkgk=ss - sb, raw_signed_index=vi['raw_signed_index'], chi600=vi['chi600'], entropy_deficit_jkgk=vi['entropy_deficit_jkgk'], ventilation_index=vi['ventilation_index'], signed_estimable=bool(np.isfinite(vi['raw_signed_index'])), standard_ventilation_domain_valid=vi['valid'], ventilation_quality_status=vi['quality_status'], **rmw)
                r['status'] = 'PHYSICALLY_SUPPORTED'
            else:
                r['status'] = 'PHYSICAL_FAIL_AUTHORITATIVE_NA'
        except (ValueError, KeyError) as exc:
            r.update(status='INPUT_NOT_SUPPORTED_AUTHORITATIVE_NA', failure_reason=str(exc))
        r['authoritative']['pi_quality_status'] = r['status']
        r['authoritative']['pi_iflag'] = r['raw_trace']['outputs']['IFL'] if r['raw_trace'] else None
        r['authoritative']['ventilation_quality_status'] = r['authoritative'].get('ventilation_quality_status', r['status'])
        out[name] = r
    return out
