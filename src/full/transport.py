"""Read the frozen native ERA5 windows using the original spherical transport functions."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from . import native_fields
from . import transport_native as core
ROOT=Path(__file__).absolute().parents[2]
def run(era5_root,out,event_id=None,lag=None,all_events=False):
    assert all_events or (event_id is not None and lag is not None)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    native_fields.ROOT=Path(era5_root);core.W=out
    base=ROOT/'source_data/transport'
    events=core.rows(base/'events_frozen.csv');centres=core.rows(base/'centres_frozen.csv')
    if event_id is not None:events=[e for e in events if e['event_id']==event_id]
    assert events
    request={}
    for row in centres:request.setdefault(row['event_id'],{})[int(row['lag_h'])]=row
    reader=native_fields.NativeReader();records=[]
    try:
        for e in events:
            for h in (core.LAGS if lag is None else [int(lag)]):
                centre=core.centre_geometry(request[e['event_id']],h)
                if centre is None:
                    records.append({'event_id':e['event_id'],'lag_h':h,'status':'missing genuine centre'})
                    continue
                windows,meta=core.get_windows(reader,e,h,centre)
                d=core.evaluate(windows,centre)
                d.update(event_id=e['event_id'],lag_h=h,UTC=centre[2]['UTC'])
                records.append(d)
                (out/'native_exchange.json').write_text(json.dumps(records,indent=2,allow_nan=True),encoding='utf8')
    finally:reader.close()
    return records
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--era5-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--event-id');p.add_argument('--lag',type=int);p.add_argument('--all-events',action='store_true')
    a=p.parse_args();run(a.era5_root,a.out,a.event_id,a.lag,a.all_events)

