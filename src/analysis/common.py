from pathlib import Path
import json, os
import numpy as np
import pandas as pd

def repository():
    p=Path(__file__).resolve().parents[2]
    return Path("\\\\?\\"+str(p)) if os.name=="nt" and not str(p).startswith("\\\\?\\") else p

def read(path):
    return pd.read_csv(path,keep_default_na=False,na_values=["","nan","NaN"])

def boolean(values):
    return pd.Series(values).astype(str).str.lower().isin(["true","1"]).to_numpy()

def close(actual,expected,atol=1e-9,rtol=1e-9):
    a=np.asarray(actual,dtype=float); b=np.asarray(expected,dtype=float)
    if not np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True):
        raise AssertionError(f"Numeric disagreement: maximum absolute error {np.nanmax(np.abs(a-b))}")
    return float(np.nanmax(np.abs(a-b))) if a.size and np.isfinite(a-b).any() else 0.0

def interval(values):
    x=np.asarray(values,float); good=x[np.isfinite(x)]
    return np.quantile(good,[.025,.975]) if len(good) else np.array([np.nan,np.nan])

def mean_draws(values,indices,batch=100):
    values=np.asarray(values,float)
    return np.concatenate([values[indices[i:i+batch]].mean(1) for i in range(0,len(indices),batch)])

def output(topic,out=None):
    p=Path(out) if out is not None else repository()/"outputs"/topic
    p.mkdir(parents=True,exist_ok=True)
    return p

def save_table(frame,path):
    frame.to_csv(path,index=False)

