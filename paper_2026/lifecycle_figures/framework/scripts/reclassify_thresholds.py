"""Executed 18-configuration classification loop with portable input/output handling.

The original candidate extraction includes 1982-2025; final manuscript summaries
are restricted to the frozen 1982-2023 master by recompute_primary_summaries.py.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'lifecycle'))
from lifecycle_core import identify_tc_crisis_lifecycle
from track_arguments import args_from_group

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    manifest=pd.read_csv(ROOT/'source_data/de_threshold/manifest.csv',keep_default_na=False)
    assert len(manifest)==18
    x=pd.read_parquet(ROOT/'source_data/preprocessed_candidate_tracks.parquet');groups=[]
    for sid,g in x.groupby('SID',sort=False):
        g=g.sort_values('ISO_TIME').reset_index(drop=True);gen=g.loc[g.IS_TROPICAL&g.WIND.notna()&g.WIND.ge(35),'ISO_TIME']
        if len(gen) and not 1982<=gen.iloc[0].year<=2025:continue
        groups.append((sid,args_from_group(sid,g)))
    assert len(groups)==int(manifest.n_total_sid.iloc[0])
    f=identify_tc_crisis_lifecycle
    for r in manifest.to_dict('records'):
        name=r['experiment_id'];folder=args.out/name;folder.mkdir()
        cfg={k:float(r[k]) for k in ['escalation_progress','maturity_progress','min_peak_gain']}
        if r['is_baseline']:
            q=pd.read_parquet(ROOT/'source_data/de_threshold'/name/'QC.parquet')
            s=pd.read_parquet(ROOT/'source_data/de_threshold'/name/'summary.parquet')
            t=pd.read_parquet(ROOT/'source_data/de_threshold'/name/'observations.parquet')
        else:
            qc=[];tt=[]
            for sid,arguments in groups:
                z=f(**arguments,**cfg);qc.append(z['summary'])
                if z['summary']['valid_lifecycle']:
                    tr=z['track'][['time','stage_code']].copy();tr.insert(0,'sid',sid);tt.append(tr)
            q=pd.DataFrame(qc);s=q[q.valid_lifecycle.eq(1)].copy();t=pd.concat(tt,ignore_index=True)
        for label,frame in [('QC',q),('summary',s),('observations',t)]:
            expected=pd.read_parquet(ROOT/'source_data/de_threshold'/name/(label+'.parquet'))
            pd.testing.assert_frame_equal(frame,expected,check_exact=True)
            frame.to_parquet(folder/(label+'.parquet'),index=False)
        print(json.dumps(dict(experiment=name,valid_SID=len(s),frozen_comparison='PASS')))

if __name__=='__main__':main()
