"""Explicit inclusive-50-hPa CAPE endpoint and per-call buoyancy trace.

The pinned upstream source is retained unchanged. The selected calculation
includes its upper endpoint by replacing the three P/T/R upper slices N with
N+1. All other numerical statements remain as in the pinned implementation.
"""
from pathlib import Path
import ast, copy, hashlib, importlib, os
import numpy as np
HERE=Path(__file__).resolve().parent
# Pure Python is portable across the documented environments. A caller with a
# compatible Numba installation may explicitly opt in before importing.
os.environ.setdefault('TCPYPI_DISABLE_NUMBA','1')
UP=importlib.import_module(__package__+'._vendor.tcpyPI.pi')
PINNED_SHA='34c823867b14f82503c3bb868e7ee19d008d927c219fb9514c600757a3bed022'
PARAMETERS=dict(CKCD=.9,ascent_flag=1,diss_flag=1,V_reduc=.8,ptop=50.,miss_handle=1,outflow_source_flag=0)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(y) for y in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and (not np.isfinite(x)):
        return None
    return x

def endpoint_tree(include_endpoint=False, instrument=True):
    raw = Path(UP.__file__).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINNED_SHA:
        raise ValueError('pinned upstream PI source changed')
    tree = ast.parse(raw.decode('utf-8'))
    fn = copy.deepcopy(next((x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'cape')))
    fn.decorator_list = []
    changed = []

    class Edit(ast.NodeTransformer):

        def visit_Assign(self, node):
            if include_endpoint and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) and (node.value.value.id in ('P', 'T', 'R')) and isinstance(node.value.slice, ast.Slice) and isinstance(node.value.slice.upper, ast.Name) and (node.value.slice.upper.id == 'N'):
                changed.append(node.value.value.id)
                node.value.slice.upper = ast.BinOp(left=ast.Name(id='N', ctx=ast.Load()), op=ast.Add(), right=ast.Constant(1))
            target = node.targets[0]
            if instrument and isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and (target.value.id == 'TVRDIF'):
                assert isinstance(node.value, ast.BinOp) and isinstance(node.value.right, ast.Name)
                env = node.value.right.id
                return [node, ast.Expr(value=ast.Call(func=ast.Name(id='_trace_level', ctx=ast.Load()), args=[ast.Name(id=n, ctx=ast.Load()) for n in ('j', 'TG', 'RG', 'TLVR', env)], keywords=[]))]
            return node

        def visit_Return(self, node):
            if instrument:
                return ast.Return(value=ast.Call(func=ast.Name(id='_finish_cape', ctx=ast.Load()), args=[node.value, ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[]))
            return node
    fn = Edit().visit(fn)
    if include_endpoint and changed != ['P', 'T', 'R']:
        raise ValueError('endpoint transformation did not match exactly3 slices')
    return (ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), changed)

class TracedPI:

    def __init__(self, include_endpoint=False):
        self.include_endpoint = bool(include_endpoint)
        self.calls = []
        self.active = None
        capetree, self.endpoint_changes = endpoint_tree(include_endpoint)
        ns = dict(vars(UP))
        ns.update(_trace_level=self.trace_level, _finish_cape=self.finish_cape)
        exec(compile(capetree, str(HERE / 'pi_endpoint_trace.py') + '::cape', 'exec'), ns)
        self.cape_numeric = ns['cape']
        self.cape_tree = capetree
        tree = ast.parse(Path(UP.__file__).read_text(encoding='utf-8'))
        fn = copy.deepcopy(next((x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_pi_numba')))
        fn.decorator_list = []
        calls = sorted([n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and (n.func.id == 'cape')], key=lambda n: n.lineno)
        if len(calls) != 3:
            raise ValueError('expected three actual CAPE call sites')
        for call, label in zip(calls, ('CAPEA', 'CAPEM', 'CAPEMS')):
            call.func.id = '_call_cape'
            call.args = [ast.Constant(label), ast.Call(func=ast.Attribute(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]), attr='get', ctx=ast.Load()), args=[ast.Constant('NP'), ast.Constant(-1)], keywords=[])] + call.args
        ns['_call_cape'] = self.call_cape
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(HERE / 'pi_endpoint_trace.py') + '::pi', 'exec'), ns)
        self.pi_numeric = ns['_pi_numba']

    def trace_level(self, j, TG, RG, parcel_trho, env_trho):
        self.active['levels'].append(dict(index=int(j), parcel_temperature_k=float(TG), parcel_mixing_ratio_kgkg=float(RG), parcel_density_temperature_k=float(parcel_trho), environment_density_temperature_k=float(env_trho), density_temperature_difference_k=float(parcel_trho - env_trho), buoyancy_ms2=float(9.80665 * (parcel_trho - env_trho) / env_trho)))

    def call_cape(self, path, iteration, TP, RP, PP, T, R, P, *args, **kwargs):
        self.active = dict(call_index=len(self.calls), path=path, iteration=int(iteration), parcel_T_K=float(TP), parcel_r_kgkg=float(RP), parcel_P_hpa=float(PP), input_P_hpa=np.asarray(P).copy(), levels=[])
        return self.cape_numeric(TP, RP, PP, T, R, P, *args, **kwargs)

    def finish_cape(self, result, state):
        r = self.active
        for key in ('P', 'T', 'R', 'TVRDIF'):
            if key in state:
                r[{'P': 'effective_P_hpa', 'T': 'environment_T_K', 'R': 'environment_r_kgkg', 'TVRDIF': 'density_temperature_difference_profile_K'}[key]] = np.asarray(state[key]).copy()
        for key in ('INB', 'jmin', 'nlvl', 'PLCL', 'PINB', 'PA', 'NA', 'PAT'):
            if key in state:
                r[key] = state[key]
        r.update(CAPED_jkg=result[0], TOB_K=result[1], LNB_hpa=result[2], IFLAG=result[3], endpoint_inclusive=self.include_endpoint)
        pressures = np.asarray(r.get('effective_P_hpa', []))
        evaluated = {x['index']: x for x in r['levels']}
        brackets = []
        for j in range(max(0, len(pressures) - 1)):
            if j in evaluated and j + 1 in evaluated:
                a = evaluated[j]['density_temperature_difference_k']
                b = evaluated[j + 1]['density_temperature_difference_k']
                if a > 0 and b <= 0:
                    zero = (pressures[j + 1] * a - pressures[j] * b) / (a - b)
                    brackets.append(dict(lower_index=j, upper_index=j + 1, lower_pressure_hpa=float(pressures[j]), upper_pressure_hpa=float(pressures[j + 1]), lower_TVRDIF_K=a, upper_TVRDIF_K=b, zero_pressure_hpa=float(zero)))
        r['positive_to_nonpositive_zero_brackets'] = brackets
        crossings = []
        for j in range(max(0, len(pressures) - 1)):
            if j in evaluated and j + 1 in evaluated:
                a = evaluated[j]['density_temperature_difference_k']
                b = evaluated[j + 1]['density_temperature_difference_k']
                if a > 0 and b <= 0 or (a < 0 and b >= 0):
                    crossings.append(dict(lower_pressure_hpa=float(pressures[j]), upper_pressure_hpa=float(pressures[j + 1]), lower_TVRDIF_K=a, upper_TVRDIF_K=b, direction='POSITIVE_TO_NONPOSITIVE' if a > 0 else 'NEGATIVE_TO_NONNEGATIVE', zero_pressure_hpa=float((pressures[j + 1] * a - pressures[j] * b) / (a - b))))
        r['all_sign_crossings'] = crossings
        r['CAPED_before_zero_clip_jkg'] = state['PA'] + state.get('PAT', 0.0) - state['NA'] if 'PA' in state and 'NA' in state else None
        r['solver_no_positive_buoyancy_LNB_zero'] = bool(state.get('INB') == 0 and result[2] == 0)
        r['TOB_definition'] = 'original environmental T linearly interpolated in pressure at selected PINB, not parcel or density temperature'
        top = len(pressures) - 1
        r['actual_top_hpa'] = float(pressures[-1]) if len(pressures) else None
        r['top_evaluated'] = top in evaluated
        r['top_TVRDIF_K'] = evaluated[top]['density_temperature_difference_k'] if top in evaluated else None
        r['top_positive_buoyancy'] = bool(top in evaluated and evaluated[top]['density_temperature_difference_k'] > 0)
        inb = int(r.get('INB', -1))
        r['selected_LNB_bracket'] = next((x for x in brackets if x['lower_index'] == inb), None)
        r['selected_zero_bracketed'] = r['selected_LNB_bracket'] is not None
        r['top_exactly_neutral'] = bool(top in evaluated and evaluated[top]['density_temperature_difference_k'] == 0.0)
        r['selected_zero_strictly_interior'] = bool(r['selected_LNB_bracket'] is not None and pressures[-1] < r['selected_LNB_bracket']['zero_pressure_hpa'] < pressures[0])
        r['top_boundary_status'] = 'NO_POSITIVE_BUOYANCY_LNB_ZERO' if r['solver_no_positive_buoyancy_LNB_zero'] else 'TOP_NOT_EVALUATED' if top not in evaluated else 'TOP_POSITIVE_UNBRACKETED' if r['top_positive_buoyancy'] else 'NEUTRAL_AT_BOUNDARY' if r['top_exactly_neutral'] else 'SELECTED_ZERO_BRACKETED' if r['selected_zero_bracketed'] else 'NO_POSITIVE_LNB_OR_OTHER_SUPPORT_FAILURE'
        self.calls.append(r)
        self.active = None
        return result

    def run(self, T_K, q_kgkg, P_hpa, SST_K, MSL_Pa, **overrides):
        self.calls = []
        p = np.asarray(P_hpa, float).copy()
        t = np.asarray(T_K, float).copy()
        q = np.asarray(q_kgkg, float).copy()
        if p.ndim != 1 or p.shape != t.shape or p.shape != q.shape or (len(p) < 3):
            raise ValueError('profile shape')
        if not np.all(np.isfinite(p) & (p > 0)) or not np.all(np.diff(p) < 0):
            raise ValueError('strict descending unique hPa required')
        if p[-1] != 50.0 or np.sum(p == 50.0) != 1:
            raise ValueError('real50hPa endpoint must occur exactly once')
        if not np.all(np.isfinite(t) & (t > 100) & (t < 400) & np.isfinite(q) & (q >= 0) & (q < 0.1)):
            raise ValueError('MISSING_OR_INVALID_REQUIRED_T_Q_NO_PREFIX_REMOVAL')
        if not np.isfinite(SST_K) or not 100 < SST_K < 400 or (not np.isfinite(MSL_Pa)) or (not 20000 < MSL_Pa < 120000):
            raise ValueError('surface units/support invalid')
        cfg = dict(PARAMETERS)
        cfg.update(overrides)
        if cfg != PARAMETERS:
            raise ValueError('all frozen PI physical parameters must remain exactly fixed')
        out = self.pi_numeric(float(SST_K - 273.15), float(MSL_Pa / 100), p, t - 273.15, 1000 * q / (1 - q), **cfg)
        last = {path: next((r for r in reversed(self.calls) if r['path'] == path), None) for path in ('CAPEA', 'CAPEM', 'CAPEMS')}
        for call in self.calls:
            call['final_used_iteration'] = last.get(call['path']) is call
        summary = {k: dict(actual_top_hpa=v['actual_top_hpa'], top_TVRDIF_K=v['top_TVRDIF_K'], top_positive_buoyancy=v['top_positive_buoyancy'], selected_zero_bracketed=v['selected_zero_bracketed'], top_exactly_neutral=v['top_exactly_neutral'], selected_zero_strictly_interior=v['selected_zero_strictly_interior'], top_boundary_status=v['top_boundary_status'], LNB_hpa=v['LNB_hpa']) if v else None for k, v in last.items()}
        return dict(outputs=dict(zip(('PI_ms', 'PMIN_hpa', 'IFL', 'TO_K', 'OTL_hpa'), out)), parameters=cfg, endpoint_inclusive=self.include_endpoint, all_CAPE_calls=self.calls, final_CAPE_paths=summary, numerical_success=bool(out[2] == 1 and np.all(np.isfinite(out))), physical_acceptance='NOT_DECIDED_BY_THIS_ADAPTER', input_q_conversion='r_gkg=1000*q/(1-q),once', input_temperature_conversion='TC=TK-273.15,once')
