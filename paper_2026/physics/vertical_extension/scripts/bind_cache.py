"""Bind a completed extraction's reusable products before optional reaggregation."""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
import pandas as pd
from configuration import OUTPUT_ROOT as ROOT, MWT_ROOT as W, initialize_output


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2), b''):h.update(b)
    return h.hexdigest()


def bind_cache():
    initialize_output()
    request=pd.read_parquet(ROOT/'inputs_manifest/requests.parquet')
    operator_sha=sha(W/'cache/mwt_operator_N1024_j5_8_symmetric_physical.npy')
    rows=[]
    for sid in sorted(request.sid.unique()):
        folder=ROOT/'diagnostics'/sid
        receipt_path=folder/'COMPLETE.json'
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        assert receipt['sid']==sid and receipt['status']=='COMPLETE'
        assert sha(folder/'request_values.parquet')==receipt['output_sha256']
        selected=request[request.sid.eq(sid)].sort_values('UTC')
        seconds=pd.to_datetime(selected.UTC,utc=True).astype('int64').to_numpy()//10**9
        with h5py.File(W/'addendum_20261023/cache'/sid/'increment.h5','r') as h:
            pressure=h['coordinates/pressure_Pa'][:]
            ii=[int(np.flatnonzero(pressure==p)[0]) for p in [40000,50000,60000,70000,85000,92500]]
            reference=np.r_[h['basic_state/c'][:][ii],h['reference/temperature_mean'][:][ii],h['basic_state/dc_dp'][:][ii]]
            identity={'operator_sha256':operator_sha,'reference_sha256':hashlib.sha256(reference.tobytes()).hexdigest(),'latitude_sha256':hashlib.sha256(h['coordinates/latitude'][:].tobytes()).hexdigest(),'longitude_sha256':hashlib.sha256(h['coordinates/longitude'][:].tobytes()).hexdigest(),'seconds_sha256':hashlib.sha256(seconds.tobytes()).hexdigest()}
            shape=(3,len(seconds),len(h['coordinates/latitude']),len(h['coordinates/longitude']))
        paths=[folder/f'omega_products_{p}.npz' for p in [40000,60000,92500]]
        for p in paths:
            with np.load(p) as z:
                assert np.array_equal(z['seconds'],seconds)
                for field in ['omega','omega_u','omega_v','omega_t']:assert z[field].shape==shape
        rows.append({'sid':sid,'cache':[{'path':str(p),'size':p.stat().st_size,'sha256':sha(p)} for p in paths],'source_receipt_sha256':sha(receipt_path),'original_identity':identity,'raw_inputs':receipt.get('raw_inputs',[])})
    target=ROOT/'inputs_manifest/TASK_CACHE_BEFORE_DERIVATION_V2.json'
    target.write_text(json.dumps({'sids':rows},indent=2),encoding='utf-8')
    return target


if __name__ == '__main__': print(bind_cache())
