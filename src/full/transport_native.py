"""Single-reader native-field T1 calculation, checkpointed per event and UTC."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,csv,json,time,hashlib,argparse,traceback
import numpy as np
from .transport_geometry import A,G0,ll_xyz,xyz_ll,basis,circle,annulus_grid,boundary_normal_speed,flux_levels,column_integral
from .native_fields import NativeReader,Window
W=Path(os.environ.get('FLTCCF_FULL_OUTPUT', 'outputs/raw_transport'))
P=np.array([500.,600.,700.,850.,925.]);PP=P*100
LAGS=list(range(-24,25,6))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def atom_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def utc(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def good(r):
    try:return np.isfinite(float(r['lat'])) and np.isfinite(float(r['lon'])) and r['actual_track_point']=='True' and r['position_extrapolated']=='False'
    except (ValueError,TypeError):return False
def centre_geometry(rr,lag):
    here=rr.get(lag)
    if here is None or not good(here):return None
    k=ll_xyz(float(here['lat']),float(here['lon']));derivs=[]
    for offset in [-6,6]:
        other=rr.get(lag+offset)
        if other is None or not good(other):continue
        ko=ll_xyz(float(other['lat']),float(other['lon']));dot=np.clip(k@ko,-1,1);angle=np.arccos(dot)
        if angle<1e-10:derivs.append(np.zeros(3))
        else:derivs.append((ko-dot*k)/np.sin(angle)*angle/(offset*3600.))
    if not derivs:return None
    return k,np.mean(derivs,axis=0),here,'two_adjacent_arcs' if len(derivs)==2 else 'one_genuine_adjacent_arc'

def get_windows(reader,event,lag,centre):
    key=event['event_id'].replace('|','_');folder=W/'cache/T1_native'/key;folder.mkdir(parents=True,exist_ok=True)
    cache=folder/f'{lag:+03d}.npz';identity=folder/f'{lag:+03d}.json'
    if cache.exists():
        assert identity.exists(),'Incomplete raw-cache checkpoint'
        meta=json.loads(identity.read_text(encoding='utf-8'))
        assert sha(cache)==meta['cache_sha256']
        with np.load(cache) as z:
            windows={name:Window(z[name+'_lat'],z[name+'_lon'],z[name],meta['lon_ref']) for name in meta['fields']}
        return windows,meta
    k,kdot,row,motion=centre;lat=float(row['lat']);lon=float(row['lon']);padding=300e3/A*180/np.pi+1.
    lat_bounds=[lat-padding,lat+padding];lon_padding=padding/max(.1,np.cos(np.radians(abs(lat)+padding)))
    lon_bounds=[lon-lon_padding,lon+lon_padding]
    windows={};provenance={};data={}
    for field,level in [('sp',None)]+[(f,p) for p in P for f in ['q','u','v']]:
        name=field if level is None else field+str(int(level))
        win,info=reader.read_window(field,level,utc(row['UTC']),lat_bounds,lon,lon_bounds)
        windows[name]=win;provenance[name]=info
        data[name]=win.data;data[name+'_lat']=win.lat;data[name+'_lon']=win.lon
    tmp=cache.with_suffix('.pending.npz');np.savez_compressed(tmp,**data)
    with tmp.open('r+b') as f:os.fsync(f.fileno())
    os.replace(tmp,cache)
    meta={'sid':event['sid'],'event_id':event['event_id'],'UTC':row['UTC'],'lag_h':lag,'lon_ref':lon,
          'lat_bounds':lat_bounds,'lon_bounds_unwrapped':lon_bounds,'fields':list(windows),'inputs':provenance,'cache_sha256':sha(cache)}
    atom_json(identity,meta);return windows,meta

def evaluate(windows,centre,n_az=360,n_radial=16,area_az=180):
    k,kdot,row,motion=centre;level_rows=[];column_rows=[];boundary={};moving={}
    ax,da=annulus_grid(k,100e3,300e3,n_radial,area_az);alat,alon=xyz_ll(ax)
    sp_area,spmin=windows['sp'].sample(alat,alon,True)
    # Native sp cells touching the annulus are also checked, to avoid missing a
    # terrain cell between quadrature nodes. This is a conservative fixed halo.
    sw=windows['sp'];yy,xx=np.meshgrid(sw.lat,sw.lon,indexing='ij');sx=ll_xyz(yy,xx)
    rr=np.arccos(np.clip(sx@k,-1,1))*A;halo=A*np.radians(.25)*np.sqrt(2)
    touching=(rr>=100e3-halo)&(rr<=300e3+halo)
    body_sp_complete=bool(np.isfinite(sw.data[touching]).all() and np.all(sw.data[touching]>=PP[-1]))
    area_support=bool(body_sp_complete and np.isfinite(spmin).all() and np.all(spmin>=PP[-1]))
    stocks=[];absolute_div=[];mass_div=[]
    for radius in [100e3,300e3]:
        bx,bn,dl=circle(k,radius,n_az);blat,blon=xyz_ll(bx);east,north=basis(bx)
        cb=boundary_normal_speed(k,kdot,bx,radius);_,spb=windows['sp'].sample(blat,blon,True)
        vals={x:[] for x in ['in','out','net']};move=[]
        for p,p_pa in zip(P,PP):
            q=windows['q'+str(int(p))].sample(blat,blon)
            u=windows['u'+str(int(p))].sample(blat,blon);v=windows['v'+str(int(p))].sample(blat,blon)
            valid=np.isfinite(q)&(q>=0)&np.isfinite(u)&np.isfinite(v)&np.isfinite(spb)&(spb>=p_pa)
            vn=(u[:,None]*east+v[:,None]*north)*bn;vn=vn.sum(-1)-cb
            q=np.where(valid,q,np.nan);vn=np.where(valid,vn,np.nan)
            f=flux_levels(q,vn,dl)
            for name in vals:vals[name].append(float(f[name]))
            move.append(float(np.sum(q*cb*dl)/G0))
            level_rows.append({'radius_km':radius/1000,'pressure_hPa':p,'complete':bool(valid.all()),'valid_nodes':int(valid.sum()),'total_nodes':n_az,
                'in_kg_s_Pa':float(f['in']),'out_kg_s_Pa':float(f['out']),'net_kg_s_Pa':float(f['net']),
                'q_mean_kg_kg':float(np.mean(q)),'relative_normal_mean_m_s':float(np.mean(vn)),
                'line_length_m':float(dl.sum()),'boundary_normal_min_m_s':float(cb.min()),'boundary_normal_max_m_s':float(cb.max())})
        boundary[radius]={name:column_integral(vals[name],PP) for name in vals}
        moving[radius]=column_integral(move,PP)
        column_rows.append({'radius_km':radius/1000,**{name+'_kg_s':float(value) for name,value in boundary[radius].items()},'complete':bool(np.isfinite(list(boundary[radius].values())).all())})
    for p,p_pa in zip(P,PP):
        qwin=windows['q'+str(int(p))];uwin=windows['u'+str(int(p))];vwin=windows['v'+str(int(p))]
        q=qwin.sample(alat,alon);u=uwin.sample(alat,alon);v=vwin.sample(alat,alon)
        valid=area_support and np.isfinite(q).all() and np.all(q>=0) and np.isfinite(u).all() and np.isfinite(v).all()
        if not valid:stocks.append(np.nan);absolute_div.append(np.nan);mass_div.append(np.nan);continue
        qy,qx=qwin.sample_gradient(alat,alon);uy,ux=uwin.sample_gradient(alat,alon);vy,vx=vwin.sample_gradient(alat,alon)
        phi=np.radians(alat);cos=np.cos(phi);sin=np.sin(phi)
        div=(u*qx+q*ux+cos*(v*qy+q*vy)-sin*q*v)/(A*cos)
        div_air=(ux+cos*vy-sin*v)/(A*cos)
        stocks.append(float(np.sum(q*da)/G0));absolute_div.append(float(np.sum(div*da)/G0));mass_div.append(float(np.sum(div_air*da)/G0))
    stock=column_integral(stocks,PP)
    side=boundary[300e3]['net']-boundary[100e3]['net']
    area_flux=-column_integral(absolute_div,PP)+moving[300e3]-moving[100e3]
    gross=sum(boundary[r][x] for r in boundary for x in ['in','out'])
    return {'level_flux':level_rows,'column_flux':column_rows,'stock_kg':float(stock),'stock_complete':bool(np.isfinite(stock)),
            'stock_per_native_pressure_kg_Pa':stocks,'area_m2':float(da.sum()),'side_net_kg_s':float(side),
            'independent_area_net_kg_s':float(area_flux),'divergence_check_gross_normalized_error':float(abs(side-area_flux)/max(gross,1)),
            'horizontal_air_mass_divergence_kg_s':float(column_integral(mass_div,PP)),
            'vertical_exchange_kg_s':None,'residual_kg_s':None,'vertical_budget_status':'vertical boundary exchange is not included in this lateral-exchange calculation',
            'centre_speed_m_s':float(A*np.linalg.norm(kdot)),'centre_derivative':motion,'quadrature':[n_az,n_radial,area_az]}
