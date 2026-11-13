import numpy as np
import pandas as pd
CORE=["X1_sst_c","X2_shear_ms","X3_mid_rh_pct","X4_d200_1e5s","X5_minus_omega500_pa_s","X6_signed_vo850_1e5s"]

def make_design(d, extra_dv=False):
    d=d.copy(); m=pd.to_datetime(d.entry_UTC,utc=True).dt.month; d['month_sin']=np.sin(2*np.pi*(m-1)/12); d['month_cos']=np.cos(2*np.pi*(m-1)/12); d['year_center']=pd.to_numeric(d.genesis_year,errors='coerce')-2002.5; d['lat_z']=pd.to_numeric(d.entry_lat,errors='coerce'); d['lon_sin']=np.sin(np.deg2rad(pd.to_numeric(d.entry_lon,errors='coerce'))); d['lon_cos']=np.cos(np.deg2rad(pd.to_numeric(d.entry_lon,errors='coerce'))); d['wind_z']=pd.to_numeric(d.entry_wind_kt,errors='coerce'); d['motion_z']=pd.to_numeric(d.entry_motion_ms_v3,errors='coerce'); d['age_z']=pd.to_numeric(d.entry_age_h,errors='coerce'); d['land_entry']=pd.to_numeric(d.entry_land_current_v3,errors='coerce'); d['dv24_z']=pd.to_numeric(d.entry_dv24_backward,errors='coerce'); d['past24_land_z']=pd.to_numeric(d.entry_past_land_contact_v3,errors='coerce')
    controls=['month_sin','month_cos','year_center','lat_z','lon_sin','lon_cos','wind_z','age_z','land_entry']+(['motion_z','dv24_z','past24_land_z'] if extra_dv else [])
    cat=pd.get_dummies(d.basin_genesis.astype('string'),prefix='basin',drop_first=True,dtype=float); stage=pd.get_dummies(d.stage.astype('string'),prefix='stage',drop_first=True,dtype=float); Xbase=pd.concat([d[controls],cat,stage],axis=1); names=list(Xbase.columns)
    X=pd.concat([Xbase]+[d[[x]].rename(columns={x:x}) for x in CORE]+[d[x].mul(d.stage.eq(s).astype(float)).rename(f'{x}_{s}') for x in CORE for s in ['E','F','Res']],axis=1); return d,X,list(X.columns),controls,list(cat.columns),list(stage.columns)
