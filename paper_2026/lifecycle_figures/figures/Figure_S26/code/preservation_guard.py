"""Source-to-artist verification for the authorized 20261027 restructuring."""
from pathlib import Path
import json, numpy as np, pandas as pd, hashlib

METRICS=['KE','APE','C_A_to_K','vertical_input_K_Wkg','vertical_input_A_Wkg','vertical_input_P_Wkg']
def verify(fig,fid,out):
    pkg=Path(__file__).resolve().parents[1]
    index=json.loads((pkg/'inputs.json').read_text(encoding='utf8'))
    frames={}
    for k,info in index.items():
        p=pkg/info['relative_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
        d=pd.read_csv(p,float_precision='round_trip',keep_default_na=False,na_values=['','nan','NaN'],low_memory=False)
        if 'level'in d:d['level']=d['level'].astype(str)
        d['_source_csv_line']=np.arange(len(d))+2
        frames[k]=d
    def primary(d):
        mask=d.metric.isin(METRICS)&d.window.eq(1)
        if 'scheme'in d:mask&=d.scheme.eq('main')
        if 'region'in d:mask&=d.region.eq('r0_300')
        return mask&((d.metric.isin(METRICS[:3])&d.level.eq('700'))|(d.metric.isin(METRICS[3:])&d.level.eq('500-850')))
    by={a.get_gid():a for a in fig.axes};records=[]
    def check_line(line,x,y):
        assert np.array_equal(np.asarray(line.get_xdata(),float),np.asarray(x,float))
        assert np.array_equal(np.asarray(line.get_ydata(),float),np.asarray(y,float))
    if fid=='Figure_S16':
        d=frames['coverage'];mask=primary(d);series=[]
        for i,metric in enumerate(METRICS):
            q=d[mask&d.metric.eq(metric)].sort_values('lag_h');a=by['a'+str(i+1)]
            check_line(a.lines[0],q.lag_h,q.paired_valid_sid)
            series.append(q.paired_valid_sid.tolist())
            records.append(dict(panel=a.get_gid(),metric=metric,quantity='paired_valid_sid (sample support count, not an energy value)',source='coverage',source_lines=q._source_csv_line.tolist(),x=q.lag_h.tolist(),y=q.paired_valid_sid.tolist(),numeric_offsets=False))
        assert all(v==series[0]for v in series)
        specs=[('b','reference_size','fixed_joint_aboveground_finite_area_fraction'),('c','reference_size','c'),('d','hydro_size','residual_over_reference_scale')]
        for panel,key,field in specs:
            q=frames[key].sort_values('pressure_hpa');a=by[panel]
            check_line(a.lines[0],q[field+'_median'],q.pressure_hpa)
            # fill_betweenx constructs a closed polygon from unaltered min/max.
            paths=a.collections[0].get_paths();v=paths[0].vertices
            for r in q.itertuples():
                lo=getattr(r,field+'_minimum');hi=getattr(r,field+'_maximum')
                assert any(np.array_equal(p,[lo,r.pressure_hpa])for p in v)
                assert any(np.array_equal(p,[hi,r.pressure_hpa])for p in v)
            records.append(dict(panel=panel,source=key,source_lines=q._source_csv_line.tolist(),fields=[field+'_median',field+'_minimum',field+'_maximum','pressure_hpa'],all_line_and_interval_vertices_exact=True))
    elif fid=='Figure_S17':
        d=frames['pre'];d=d[d.scheme.eq('main')&d.region.eq('r0_300')]
        for i,metric in enumerate(METRICS[:3]):
            q=d[d.metric.eq(metric)];array=q.pivot(index='level',columns='window',values='estimate').reindex(index=['500','700','850'],columns=[0,1,2]).to_numpy(float)
            assert np.array_equal(np.asarray(by['abc'[i]].images[0].get_array()),array,equal_nan=True)
            records.append(dict(panel='abc'[i],metric=metric,source='pre',source_lines=q[q.level.isin(['500','700','850'])]._source_csv_line.tolist(),array=array.tolist(),display_transform='native data; decimal multiplier on labels only'))
        for i,metric in enumerate(METRICS[3:]):
            q=d[d.metric.eq(metric)&d.level.eq('500-850')].sort_values('window')
            a=by['d'+str(i+1)]
            dots=[l for l in a.lines if l.get_marker()=='o'];assert len(dots)==3
            segs=[s for col in a.collections for s in col.get_segments()]
            for j,(_,r)in enumerate(q.iterrows()):
                y=(1-j)*.30;check_line(dots[j],[r.estimate],[y])
                assert any(np.array_equal(s,np.array([[r.lower,y],[r.upper,y]]))for s in segs)
                records.append(dict(panel=a.get_gid(),old_panel='def'[i],metric=metric,window=int(r.window),source='pre',source_line=int(r._source_csv_line),estimate=float(r.estimate),lower=float(r.lower),upper=float(r.upper),n_sid=int(r.n_sid),x_coordinate_transform='identity',y_coordinate='categorical window slot; no numeric jitter',original_estimate_exact=True,original_interval_exact=True))
    elif fid=='Figure_S18':
        d=frames['sensitivity'];d=d[primary(d)]
        fig.canvas.draw()
        for i,metric in enumerate(METRICS):
            a=by['abcdef'[i]]
            q=[d[d.metric.eq(metric)&d.sensitivity.eq(k)].iloc[0]for k in ['input_shift','radius_500']]
            motion=frames['motion'][frames['motion'].metric.eq(metric)].iloc[0]
            values=[(r.alternative_minus_main,r.direct_difference_lower,r.direct_difference_upper,int(r.n_sid),'sensitivity',int(r._source_csv_line))for r in q]
            values.append((motion.adjustment,motion.adjustment_lower,motion.adjustment_upper,int(motion.n_sid),'motion',int(motion._source_csv_line)))
            dots=[l for l in a.lines if l.get_marker()=='o'];assert len(dots)==3
            segs=[s for col in a.collections for s in col.get_segments()]
            for j,(v,lo,hi,n,key,line)in enumerate(values):
                check_line(dots[j],[v],[j])
                assert any(np.array_equal(s,np.array([[lo,j],[hi,j]]))for s in segs)
                width=abs(a.transData.transform((hi,j))[0]-a.transData.transform((lo,j))[0])
                marker_px=dots[j].get_markersize()*fig.dpi/72
                records.append(dict(panel=a.get_gid(),metric=metric,scenario=['input_shift','radius_500','motion_adjustment'][j],source=key,source_line=line,estimate=float(v),lower=float(lo),upper=float(hi),n_sid=n,interval_width_pt=width/fig.dpi*72,marker_width_pt=dots[j].get_markersize(),interval_to_marker_ratio=width/marker_px,axis_width_pt=a.bbox.width/fig.dpi*72,axis_min=float(a.get_xlim()[0]),axis_max=float(a.get_xlim()[1]),original_estimate_exact=True,original_interval_exact=True))
    else:raise ValueError(fid)
    result=dict(figure=fid,status='PASS',method='packaged frozen CSV -> each native artist -> exact values and intervals',inputs={k:dict(path=str(pkg/v['relative_path']),sha256=v['sha256'])for k,v in index.items()},records=records,scientific_recalculation=False)
    (Path(out)/'source_artist_crosswalk.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result
