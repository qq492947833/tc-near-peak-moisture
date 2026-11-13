"""Bounded regression check against three supplied, real frozen cyclone tracks."""
from pathlib import Path
import json,sys,inspect
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'lifecycle'))
from lifecycle_core import identify_tc_crisis_lifecycle

def main():
    source=ROOT/'lifecycle/input'
    master=pd.read_csv(source/'main_sid.csv',keep_default_na=False,na_values=[''])
    assert len(master)==2340 and master.sid.is_unique
    assert set(master.sid)==set((source/'main_sid.txt').read_text().splitlines())
    tracks=pd.read_csv(source/'example_preprocessed_tracks.csv',keep_default_na=False,na_values=[''])
    checked=0
    for sid,g in tracks.groupby('SID',sort=False):
        g=g.sort_values('ISO_TIME')
        expected=master.set_index('sid').loc[sid]
        kwargs=dict(time=g.ISO_TIME,lat=g.LAT,lon=g.LON_180,wind=g.WIND,pressure=g.PRES,basin=g.BASIN,is_tropical=g.IS_TROPICAL,dist2land=g.DIST2LAND,landfall=g.LANDFALL,wind_filled_flag=g.WIND_WAS_FILLED,sid=sid)
        result=identify_tc_crisis_lifecycle(**kwargs)['summary']
        signature=inspect.signature(identify_tc_crisis_lifecycle)
        names=list(signature.parameters)
        assert names[names.index('maturity_progress')+1]=='escalation_rate_threshold'
        assert names[names.index('escalation_rate_threshold')+1]=='resolution_rate_threshold'
        positional=[kwargs.get(name,param.default) for name,param in signature.parameters.items()]
        positional[names.index('escalation_rate_threshold')]=10000
        by_position=identify_tc_crisis_lifecycle(*positional)
        by_keyword=identify_tc_crisis_lifecycle(**kwargs)
        assert set(by_position)==set(by_keyword)=={'summary','track'}
        assert set(by_position['summary'])==set(by_keyword['summary'])
        pd.testing.assert_frame_equal(by_position['track'],by_keyword['track'],check_exact=True)
        assert by_position['summary']['escalation_time']==result['escalation_time']
        for key in ['genesis_time','escalation_time','maturity_time','resolution_time','termination_time','lmi_time','lmi_last_time']:
            assert pd.Timestamp(result[key])==pd.Timestamp(expected[key]),(sid,key)
            checked+=1
        for key in ['genesis_wind','lmi','mature_threshold_wind','development_duration_hours','escalation_duration_hours','maturity_duration_hours','resolution_duration_hours','lifecycle_asymmetry_time']:
            assert np.isclose(result[key],float(expected[key]),rtol=0,atol=1e-12,equal_nan=True),(sid,key)
            checked+=1
        for old_rate in [0,10,10000]:
            assert identify_tc_crisis_lifecycle(**kwargs,escalation_rate_threshold=old_rate)['summary']['escalation_time']==result['escalation_time']
    print(json.dumps({'status':'PASS','frozen_main_SID':len(master),'example_SID':tracks.SID.nunique(),'scientific_field_checks':checked,'legacy_E_rate_parameter_ignored':True}))

if __name__=='__main__':main()
