"""Descriptive first-return status sensitivity; original P3 target unchanged."""
from pathlib import Path
import json,datetime
import numpy as np,pandas as pd
from geography_tables import O,read,bools,regions,sha

e=read(O/'G3_departure_risk_SID_source.csv');rows=[]
e[bools(e.known_other_return)|bools(e.known_unknown_state_return)].to_csv(O/'G3_nontropical_first_return_cases.csv',index=False)
for definition in ['same_origin_cohort_first_return_tropical','exclude_known_nontropical_first_return_SID']:
    d=e if definition.startswith('same') else e[~bools(e.known_other_return)]
    for region,mask in regions(d):
        z=d[mask];n=len(z);nt=int(bools(z.known_tropical_return).sum());other=int(bools(z.known_other_return).sum());unknownstatus=int(bools(z.known_unknown_state_return).sum());non=int(bools(z.known_nonreturn).sum());unknown=int(z.recovery48.isna().sum())
        assert nt+other+unknownstatus+non+unknown==n
        rows.append(dict(definition=definition,region=region,n_risk=n,tropical_first_return=nt,nontropical_first_return=other,unknown_status_first_return=unknownstatus,complete_without_return=non,unknown_followup=unknown,strict_confirmed_fraction=nt/n if n else np.nan,strict_first_return_lower_bound=nt/n if n else np.nan,strict_first_return_upper_bound=(nt+unknown+unknownstatus)/n if n else np.nan,interpretation='Descriptive first-return status endpoint; known nontropical first returns remain a separate category. No policy-model refit or relabeling of original outcomes.'))
pd.DataFrame(rows).to_csv(O/'G3_strict_tropical_status_sensitivity.csv',index=False)
rec=dict(status='COMPLETE_DESCRIPTIVE_STATUS_SENSITIVITY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_risk_n=len(e),tropical_first_return=int(bools(e.known_tropical_return).sum()),nontropical_first_return=int(bools(e.known_other_return).sum()),state_pairs=e[bools(e.known_other_return)][['sid','return_USA_STATUS','return_NATURE']].to_dict('records'),original_P3_unchanged=True,source_sha256={str(p):sha(p) for p in [Path(__file__),O/'G3_departure_risk_SID_source.csv']})
(O/'G3_strict_status_receipt.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(json.dumps(rec))
