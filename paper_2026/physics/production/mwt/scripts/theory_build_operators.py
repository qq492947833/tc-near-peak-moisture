from _release_paths import release_path as _release_path
from pathlib import Path
import numpy as np, json, time, hashlib
W=Path(str(_release_path('mwt','')))
ns={};exec((W/'source_package'/'MWT.py').read_text(encoding='utf-8-sig'),ns);mwt=ns['MWT']
ns={};exec((W/'scripts'/'theory_MWT_Apply_Operator.py').read_text(encoding='utf-8-sig'),ns);apply=ns['MWT_Apply_Operator']
rng=np.random.default_rng(20261022);report=[]
for N,j0,j1 in [(512,4,7),(1024,5,8)]:
    start=time.perf_counter();operator=mwt(np.eye(N),j0=j0,j1=j1,return_reconstruction=False);build=time.perf_counter()-start
    np.save(W/'cache'/f'mwt_operator_N{N}_j{j0}_{j1}_symmetric_physical.npy',operator)
    settings={'N':N,'j0':j0,'j1':j1,'boundary':'symmetric','normalization':'physical','source_sha256':hashlib.sha256((W/'source_package'/'MWT.py').read_bytes()).hexdigest(),'build_seconds':build,'operator_sha256':hashlib.sha256((W/'cache'/f'mwt_operator_N{N}_j{j0}_{j1}_symmetric_physical.npy').read_bytes()).hexdigest()}
    (W/'cache'/f'mwt_operator_N{N}_j{j0}_{j1}_symmetric_physical.json').write_text(json.dumps(settings,indent=2))
    # Include realistic offsets and large raw-product amplitudes, plus NaN column.
    x=rng.normal(size=(N,256))*20+1000;x[:,0]=250;x[:,1]=100000
    direct=mwt(x,j0=j0,j1=j1,return_reconstruction=False)
    dense=apply(x,operator)
    diff=float(np.max(abs(direct-dense)));maxrel=float(np.max(abs(direct-dense)/(1+abs(direct))))
    selected=np.arange(N//2-9,N//2+10)
    t0=time.perf_counter();small=apply(x,operator,time_indices=selected);elapsed=time.perf_counter()-t0
    nodata=x.copy();nodata[3,5]=np.nan
    nd=apply(nodata,operator,time_indices=selected)
    row={'N':N,'max_abs_difference':diff,'max_scaled_difference':maxrel,'passed':bool(np.allclose(direct,dense,atol=1e-8,rtol=1e-11)),
         'select_matches_full':bool(np.allclose(small,dense[:,selected],atol=1e-8,rtol=1e-11)),'selected_full_max_abs_difference':float(np.max(abs(small-dense[:,selected]))),'nan_column_preserved':bool(np.isnan(nd[:,:,5]).all()),'selected_apply_seconds_256_series':elapsed,'build_seconds':build}
    report.append(row)
(W/'validation'/'theory_operator_equivalence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
