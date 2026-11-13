"""Independent point/source and regional readback; no event-effect inspection."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import sys
import h5py
import numpy as np
import pandas as pd

ADD = Path(__file__).absolute().parents[1]
WORK = ADD.parent
sys.path.insert(0, str(WORK / 'scripts'))
sys.path.insert(0, str(WORK / 'source_package'))
from read_era5_fixed_block import read_era5_fixed_block
from MWT import MWT


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def validate(sid, source_point=False):
    folder = ADD / 'cache' / sid
    receipt = json.loads((folder / 'completed.json').read_text())
    if receipt['status'] != 'COMPLETE_EXACT_REQUEST_MWT_INCREMENT' or receipt['sid'] != sid:
        raise ValueError('Increment receipt absent or wrong identity')
    checks = []; output_identities = {}
    for name, checksum in receipt['outputs_sha256'].items():
        if sha(folder / name) != checksum:
            raise ValueError('Increment output checksum mismatch: ' + name)
        info=(folder/name).stat()
        output_identities[name]={'sha256':checksum,'bytes':info.st_size,'mtime_ns':info.st_mtime_ns}
    points = pd.read_csv(ADD / 'source_data/rise_unique_point_requests.csv', dtype={'sid':str})
    points = points[points.sid.eq(sid)].copy()
    points['UTC'] = pd.to_datetime(points.UTC, utc=True)
    frame = pd.read_parquet(folder / 'regional_metrics.parquet')
    if set(frame.request_id) != set(points.request_id):
        raise ValueError('Actual requested point set changed')
    if frame.duplicated(['request_id','scheme','region','metric','window','level']).any():
        raise ValueError('Repeated diagnostic rows')
    if set(frame.scheme) != {'main','shift16d'} or frame.metric.nunique() != 25:
        raise ValueError('Incomplete direct diagnostics')
    missing = points[~(points.position_available & points.actual_track_point & ~points.position_extrapolated & np.isfinite(points.lat) & np.isfinite(points.lon))]
    if len(missing) and not frame[frame.request_id.isin(missing.request_id)].value.isna().all():
        raise ValueError('Missing location produced a valid diagnostic')
    maximum_coefficient_error = 0.; new_source_rows = 0
    with h5py.File(folder / 'increment.h5','r') as h:
        lat = h['coordinates/latitude'][:]; lon = h['coordinates/longitude'][:]
        p = h['coordinates/pressure_Pa'][:]; op = h['coordinates/omega_pressure_Pa'][:]
        times = h['coordinates/selected_UTC_seconds'][:]
        selected = h['coordinates/selected_union_indices'][:]
        assert np.array_equal(p, np.array([50,100,150,200,250,300,400,500,600,700,850,925,1000])*100)
        assert np.array_equal(op, [50000,70000,85000])
        assert h['reference/temperature_mean'].shape == (13,)
        if receipt['plan']['existing_completed_cache']:
            olddir = Path(receipt['plan']['legacy_cache'])
            if sha(olddir / 'completed.json') != receipt['plan']['legacy_receipt_sha256']:
                raise ValueError('Legacy receipt identity changed')
            with h5py.File(olddir / 'diagnostics.h5','r') as old:
                for key in ['temperature_mean','geopotential_reference','reference_weight_fraction','reference_valid_sample_count']:
                    assert np.array_equal(h['reference'][key][:],old['reference'][key][:],equal_nan=True),key
                oldlat=old['coordinates/latitude'][:]; oldlon=old['coordinates/longitude'][:]
                oldtimes=old['coordinates/selected_UTC_seconds'][:]
                common = np.intersect1d(times, oldtimes)
                for second in common[[0,len(common)//2,-1]] if len(common) else []:
                    i = int(np.flatnonzero(times==second)[0]); j=int(np.flatnonzero(oldtimes==second)[0])
                    for oy, ox in [(len(oldlat)//2,len(oldlon)//2),(len(oldlat)//4,len(oldlon)//4)]:
                        iy=int(np.flatnonzero(lat==oldlat[oy])[0]); ix=int(np.flatnonzero(lon==oldlon[ox])[0])
                        for scheme in ['main','shift16d']:
                            for key in ['u','t','uu','omega','omega_t']:
                                a=h['coefficients'][scheme][key][:,i,:,iy,ix]
                                b=old['coefficients'][scheme][key][:,j,:,oy,ox]
                                assert np.array_equal(a,b,equal_nan=True),(scheme,key,second)
                checks.append({'scope':'deterministic common-time/source-grid coefficient nodes','old_reference_exact':True,'saved_coefficients_bit_identical':True})
        if source_point:
            if receipt['plan']['existing_completed_cache']:
                with h5py.File(Path(receipt['plan']['legacy_cache'])/'diagnostics.h5','r') as old:
                    new_source_rows=int((~np.isin(times,old['coordinates/selected_UTC_seconds'][:])).sum())
                if sid=='1982005S10105' and new_source_rows<1:
                    raise ValueError('Fixed increment pilot does not exercise a genuinely new UTC operator row')
            else:new_source_rows=len(times)
            pi = int(np.flatnonzero(p==70000)[0]); oi=int(np.flatnonzero(op==70000)[0])
            valid=h['support/fixed_1088_pressure_grid'][pi]&h['support/fixed_1088_omega_grid'][oi]
            valid[:1]=False;valid[-1:]=False;valid[:,:1]=False;valid[:,-1:]=False
            candidates=np.argwhere(valid)
            if not len(candidates): raise ValueError('No predetermined valid 700-hPa raw source point')
            order=np.argsort(np.sum((candidates-np.array([len(lat)//2,len(lon)//2]))**2,axis=1),kind='stable')
            iy,ix=candidates[order[0]]
            inventory=json.loads((WORK/'reports/data_era5_inventory.json').read_text())
            source={(r['field'],r['level_hpa']):r for r in inventory}
            raw={}
            for key in ['u','t','omega']:
                meta=source[(key,700)]
                b=read_era5_fixed_block(meta['path'],meta['variable'],receipt['plan']['raw_start_UTC'],1088,
                    lat_bounds=(lat[iy-1],lat[iy+1]),lon_bounds=(lon[ix-1],lon[ix+1]),lon_reference=receipt['plan']['domain']['lon_reference'])
                cy=int(np.flatnonzero(b['latitude']==lat[iy])[0]);cx=int(np.flatnonzero(b['longitude']==lon[ix])[0])
                raw[key]=b['field'][:,cy,cx]
            raw['t']-=h['reference/temperature_mean'][pi]
            raw['uu']=raw['u']**2;raw['omega_u']=raw['omega']*raw['u'];raw['omega_t']=raw['omega']*raw['t']
            for scheme,offset in [('main',0),('shift16d',64)]:
                for key in raw:
                    core=MWT(raw[key][offset:offset+1024],j0=5,j1=8,boundary='symmetric',normalization='physical',return_reconstruction=False)
                    pressure_index=oi if key.startswith('omega') else pi
                    got=h['coefficients'][scheme][key][:,:,pressure_index,iy,ix]
                    expected=core[:,selected-offset]
                    error=float(np.max(np.abs(got-expected)))
                    maximum_coefficient_error=max(maximum_coefficient_error,error)
                    # Fixed double-precision rounding allowance; no empirical relaxation.
                    if not np.allclose(got,expected,rtol=5e-12,atol=5e-12):
                        raise ValueError('Original source MWT mismatch: '+key)
                    checks.append({'scope':'actual_source_original_MWT','scheme':scheme,'field':key,'maximum_absolute_difference':error,
                                   'all_selected_UTC_rows_checked':len(times),'genuinely_new_UTC_rows_checked':new_source_rows,
                                   'latitude':float(lat[iy]),'longitude':float(lon[ix]),'pressure_Pa':70000})
        # Recompute six primary quantities without the production diagnostic or sampler.
        point=points[points.position_available & np.isfinite(points.lat) & np.isfinite(points.lon)].sort_values('UTC').iloc[0]
        second=point.UTC.value//10**9;ti=int(np.flatnonzero(times==second)[0]);c=h['basic_state/c'][:]
        def edges(x):return np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
        area=6371000.**2*np.abs(np.diff(np.sin(np.clip(edges(np.radians(lat)),-np.pi/2,np.pi/2))))[:,None]*np.abs(np.diff(edges(np.unwrap(np.radians(lon)))))[None,:]
        latrad=np.radians(lat[:,None]);center=np.radians(float(point.lat));dl=np.radians((lon[None,:]-float(point.lon)+180)%360-180)
        distance=6371000.*2*np.arcsin(np.sqrt(np.clip(np.sin((latrad-center)/2)**2+np.cos(latrad)*np.cos(center)*np.sin(dl/2)**2,0,1)))
        region=distance<=300000.; geometric=float(area[region].sum())
        for scheme in ['main','shift16d']:
            cf=h['coefficients'][scheme]
            def at(key,pressure):
                coord=op if key.startswith('omega') else p
                ip=int(np.flatnonzero(coord==pressure)[0])
                return cf[key][1,ti,ip,:,:]
            u=at('u',70000);v=at('v',70000);t=at('t',70000);omega=at('omega',70000)
            fields={'KE':.5*(u*u+v*v),'APE':.5*c[np.flatnonzero(p==70000)[0]]*t*t,'C_A_to_K':-287.058/70000*t*omega}
            for symbol in ['K','A','P']:
                flux=[]
                for pressure in [50000,85000]:
                    if symbol=='K':q=.5*(at('u',pressure)*at('omega_u',pressure)+at('v',pressure)*at('omega_v',pressure))
                    elif symbol=='A':q=.5*c[np.flatnonzero(p==pressure)[0]]*at('t',pressure)*at('omega_t',pressure)
                    else:q=at('omega',pressure)*at('phi',pressure)
                    flux.append(q)
                fields['vertical_input_'+symbol+'_Wkg']=(flux[0]-flux[1])/35000
            for metric,field in fields.items():
                finite=np.isfinite(field)&region
                valid_area=float(area[finite].sum());fraction=valid_area/geometric
                expected=float(np.sum(field[finite]*area[finite])/valid_area) if fraction>=.8 else np.nan
                rows=frame[frame.request_id.eq(point.request_id)&frame.scheme.eq(scheme)&frame.region.eq('r0_300')&frame.window.eq(1)&frame.metric.eq(metric)]
                rows=rows[rows.pressure_hpa.eq(700)] if metric in ['KE','APE','C_A_to_K'] else rows
                if len(rows)!=1:raise ValueError('Unexpected primary metric row cardinality')
                actual=float(rows.iloc[0].value)
                if not np.isclose(expected,actual,rtol=5e-12,atol=5e-12,equal_nan=True):raise ValueError('Independent region mismatch: '+metric)
                if not np.isclose(fraction,float(rows.iloc[0].valid_area_fraction),rtol=5e-13,atol=5e-13):raise ValueError('Valid area changed')
                checks.append({'scope':'independent_formula_and_area_mean','scheme':scheme,'metric':metric,
                               'maximum_absolute_difference':abs(expected-actual) if np.isfinite(expected) else None,
                               'request_id':str(point.request_id),'NaN_support_equal':True})
    report={'status':'VALIDATED_INCREMENT_STATED_SCOPE','sid':sid,'completed_receipt_sha256':sha(folder/'completed.json'),
            'validator_sha256':sha(__file__),'actual_source_original_MWT_checked':source_point,
            'request_count':len(points),'rows':len(frame),'checks':checks,'output_identities':output_identities,
            'maximum_source_coefficient_difference':maximum_coefficient_error,
            'genuinely_new_source_UTC_rows_checked':new_source_rows,
            'scope':'Implementation/source/identity readback, not budget closure or evidence for a group difference',
            'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    target=ADD/'validation/increment_cache'/sid
    target.mkdir(parents=True,exist_ok=True)
    (target/'validated.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sid',required=True);parser.add_argument('--source-point',action='store_true')
    args=parser.parse_args();validate(args.sid,args.source_point)
