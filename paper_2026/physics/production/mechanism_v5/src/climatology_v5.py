"""Requested native-tile monthly same-UTC climatology, 1991–2020.
Read-only source files. Resume completed tiles or partial accumulators; no global field allocation.
This does NOT compute PI of climatology, causal effects, or derivative-of-mean as mean transport.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse,json,hashlib,os,time,traceback
import numpy as np,pandas as pd,netCDF4
V=Path(__file__).resolve().parents[1]
ROOT=V/"cache"/"climatology_1991_2020_v5_1";ROOT.mkdir(parents=True,exist_ok=True)
KEYS=["q700","r700","u700","v700","z200","u200","v200","u850","v850"]
def now():return datetime.now(timezone.utc).isoformat()
def jwrite(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding="utf-8");os.replace(temp,p)
def npwrite(p,**x):
    p=Path(p);temp=p.with_name(p.name+f".{os.getpid()}.tmp")
    with temp.open("wb") as f:np.savez_compressed(f,**x)
    os.replace(temp,p)
def truth(s):
    q=s.astype(str).str.strip().str.lower()
    if not q.isin(["true","false","1","0"]).all():raise ValueError("nonboolean map flag")
    return q.isin(["true","1"])
def geometry_indices(la,lo,radius=1500):
    lat=90-np.arange(721)*.25;lon=np.arange(1440)*.25
    d=radius*1000/6371008.8;deg=np.rad2deg(d)
    il=np.flatnonzero((lat>=max(-90,la-deg))&(lat<=min(90,la+deg)))
    il=np.arange(max(0,il.min()-2),min(721,il.max()+3))
    half=180 if abs(la)+deg>=90 else np.rad2deg(np.arcsin(np.clip(np.sin(d)/np.cos(np.deg2rad(la)),0,1)))
    dl=(lon-lo+180)%360-180
    jl=np.flatnonzero(np.abs(dl)<=half+.5)
    return il,jl
def plan():
    d=pd.read_csv(V/"data/environment_requests_v5.csv",keep_default_na=False,na_values=[""],low_memory=False)
    d=d.loc[(d.module.eq("P1")&truth(d.need_map))|d.module.isin(["P4","P5"])].dropna(subset=["lat","lon"])
    tiles={}
    for r in d[["module","UTC","lat","lon"]].drop_duplicates().itertuples(index=False):
        t=pd.Timestamp(r.UTC);g=(t.month-1)*4+t.hour//6
        il,jl=geometry_indices(r.lat,r.lon,1500 if r.module=="P1" else 800)
        for i in np.unique(il//90):
            for j in np.unique(jl//180):
                for k in (KEYS if r.module=="P1" else [k for k in KEYS if k!="z200"]):tiles.setdefault((k,int(i),int(j)),set()).add(g)
    # Optional applicability screen has one predefined WP JASO upper-level background domain.
    for k in ["z200","u200","v200"]:
        for i in np.unique(np.flatnonzero((90-np.arange(721)*.25>=20)&(90-np.arange(721)*.25<=60))//90):
            for j in np.unique(np.flatnonzero((np.arange(1440)*.25>=100)&(np.arange(1440)*.25<=180))//180):
                tiles.setdefault((k,int(i),int(j)),set()).update((m-1)*4+h for m in [7,8,9,10] for h in range(4))
    jobs=[dict(key=k,ilat_tile=i,ilon_tile=j,groups=sorted(g)) for (k,i,j),g in sorted(tiles.items())]
    spec=dict(version="native_requested_climatology_v1",created=now(),years=[1991,2020],calendar="actual UTC calendar month/hour",group_encoding="(month-1)*4+hour/6",native_tile=[90,180],minimum_time_support=.9,SP="sameUTC/cellSP>=level andfinite; original requestedtimecount denominator",keys=KEYS,jobs=jobs,estimated_final_means_counts_bytes=sum(len(x["groups"])*90*180*12 for x in jobs))
    path=V/"config/climatology_tiles_v5_1.json"
    if path.exists():
        old=json.loads(path.read_text(encoding="utf-8"))
        if old["jobs"]!=jobs:raise RuntimeError("existing climatology plan differs; preserve and review")
        return old
    jwrite(path,spec);print(json.dumps({k:v for k,v in spec.items() if k!="jobs"},default=str),flush=True);print("jobs",len(jobs),flush=True)
    return spec
def ident(job):return f"{job['key']}_y{job['ilat_tile']}_x{job['ilon_tile']}"
def run_job(job,vm,max_batches=0,batch=64):
    key=job["key"];cfg=vm[key];dest=ROOT/ident(job);dest.mkdir(exist_ok=True)
    config_signature=hashlib.sha256(json.dumps(dict(job=job,input=cfg["source_identity"],SP=vm["sp"]["source_identity"],years=[1991,2020],version=1),sort_keys=True).encode()).hexdigest()
    done=dest/"complete.json";partial=dest/"accumulator.npz";output=dest/"climatology.npz"
    if done.exists():
        z=json.loads(done.read_text())
        if z["signature"]!=config_signature:raise RuntimeError("completed tile dependencies differ")
        with np.load(output) as x:
            assert x["mean"].shape==x["count"].shape and x["mean"].shape[0]==len(job["groups"])
        return dict(tile=ident(job),status="REUSED",elapsed_s=0)
    yl=job["ilat_tile"]*90;yh=min(721,yl+90);xl=job["ilon_tile"]*180;xh=min(1440,xl+180)
    groups=np.array(job["groups"]);lookup={int(x):i for i,x in enumerate(groups)}
    shape=(len(groups),yh-yl,xh-xl)
    sums=np.zeros(shape,np.float64);counts=np.zeros(shape,np.int32)
    start=time.monotonic();batches=0
    with netCDF4.Dataset(cfg["path"],"r") as ds,netCDF4.Dataset(vm["sp"]["path"],"r") as sp:
        tv=ds[cfg["time_variable"]];t=np.asarray(tv[:],dtype=float)
        begin=int(np.searchsorted(t,netCDF4.date2num(datetime(1991,1,1),tv.units,getattr(tv,"calendar","standard"))))
        end=int(np.searchsorted(t,netCDF4.date2num(datetime(2021,1,1),tv.units,getattr(tv,"calendar","standard"))))
        assert end-begin==43832
        dates=pd.date_range("1991-01-01","2020-12-31 18:00:00",freq="6h",tz="UTC")
        gi=np.array((dates.month-1)*4+dates.hour//6)
        expected=np.bincount(gi,minlength=48)[groups]
        stv=sp[vm["sp"]["time_variable"]]
        sb=int(np.searchsorted(np.asarray(stv[:]),netCDF4.date2num(datetime(1991,1,1),stv.units,getattr(stv,"calendar","standard"))))
        assert sb==begin
        nxt=begin
        if partial.exists():
            with np.load(partial) as x:
                if str(x["signature"])!=config_signature:raise RuntimeError("partial dependencies differ")
                sums=x["sums"];counts=x["counts"];nxt=int(x["next_index"])
        var=ds[cfg["variable"]];var.set_var_chunk_cache(16*1024**2,1009,.5)
        sv=sp[vm["sp"]["variable"]];sv.set_var_chunk_cache(16*1024**2,1009,.5)
        level=cfg.get("pressure_level_hpa")
        for a in range(nxt,end,batch):
            b=min(end,a+batch);gg=gi[a-begin:b-begin]
            need=np.isin(gg,groups)
            if np.any(need):
                raw=np.asarray(np.ma.filled(var[a:b,yl:yh,xl:xh],np.nan),dtype=float)
                valid=np.isfinite(raw)
                if level is not None:
                    spr=np.asarray(np.ma.filled(sv[a:b,yl:yh,xl:xh],np.nan),dtype=float)
                    valid &= np.isfinite(spr)&(spr>=max(100,float(level)*100))
                for g in np.unique(gg[need]):
                    q=(gg==g);ii=lookup[int(g)];ok=valid[q]
                    sums[ii]+=np.sum(np.where(ok,raw[q],0.),axis=0)
                    counts[ii]+=np.sum(ok,axis=0,dtype=np.int32)
            batches+=1;nxt=b
            if batches%100==0 or b==end or (max_batches and batches>=max_batches):
                npwrite(partial,sums=sums,counts=counts,next_index=np.array(nxt),signature=np.array(config_signature))
                jwrite(dest/"progress.json",dict(status="ACCUMULATING",next_index=nxt,end_index=end,updated=now(),pid=os.getpid(),elapsed_s=time.monotonic()-start))
            if max_batches and batches>=max_batches and b<end:
                return dict(tile=ident(job),status="PARTIAL_NOT_SCIENTIFIC_OUTPUT",elapsed_s=time.monotonic()-start,next_index=nxt)
        fractions=counts/expected[:,None,None]
        means=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
        means=np.where(fractions>=.9,means,np.nan)
        npwrite(output,mean=means,count=counts,expected_count=expected,groups=groups,latitude=np.asarray(ds["latitude"][yl:yh]),longitude=np.asarray(ds["longitude"][xl:xh]),signature=np.array(config_signature))
    result=dict(tile=ident(job),status="COMPLETE",signature=config_signature,created=now(),elapsed_s=time.monotonic()-start,groups=groups.tolist(),units=cfg["units"],source_identity=cfg["source_identity"],SP_identity=vm["sp"]["source_identity"],years=[1991,2020],max_count=int(counts.max()),output=str(output))
    jwrite(done,result)
    return result
class ClimatologyReader:
    def __init__(self,root=ROOT):self.root=Path(root);self.cache={}
    def read(self,key,month,hour,ilat,ilon):
        group=(int(month)-1)*4+int(hour)//6
        if int(hour) not in [0,6,12,18]:raise ValueError("UTC notsynoptic")
        out=np.full((len(ilat),len(ilon)),np.nan);cnt=np.zeros(out.shape,np.int32)
        for i in np.unique(np.array(ilat)//90):
            for j in np.unique(np.array(ilon)//180):
                name=f"{key}_y{i}_x{j}";p=self.root/name
                if not (p/"complete.json").exists():raise FileNotFoundError("required climatology not complete: "+str(p))
                if name not in self.cache:
                    with np.load(p/"climatology.npz") as z:self.cache[name]={k:z[k] for k in z.files}
                z=self.cache[name];g=np.flatnonzero(z["groups"]==group)
                if len(g)!=1:raise ValueError(f"group {group} notrequested in {name}")
                a=np.flatnonzero(np.array(ilat)//90==i);b=np.flatnonzero(np.array(ilon)//180==j)
                yi=np.array(ilat)[a]-int(i)*90;xi=np.array(ilon)[b]-int(j)*180
                out[np.ix_(a,b)]=z["mean"][g[0]][np.ix_(yi,xi)]
                cnt[np.ix_(a,b)]=z["count"][g[0]][np.ix_(yi,xi)]
        return out,cnt
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--plan-only",action="store_true");ap.add_argument("--run",action="store_true");ap.add_argument("--limit",type=int,default=1);ap.add_argument("--max-batches",type=int,default=0);ap.add_argument("--key");ap.add_argument("--shard-index",type=int,default=0);ap.add_argument("--shard-count",type=int,default=1)
    args=ap.parse_args();spec=plan()
    if not args.run:return
    vm=json.loads((V/"config/available_variable_map_v5.json").read_text())["paths"]
    jobs=[x for i,x in enumerate(spec["jobs"]) if i%args.shard_count==args.shard_index and (not args.key or x["key"]==args.key)]
    if args.limit>0:jobs=jobs[:args.limit]
    receipt=V/"logs"/f"climatology_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{os.getpid()}.jsonl"
    for j in jobs:
        try:r=run_job(j,vm,args.max_batches)
        except Exception:
            r=dict(tile=ident(j),status="ERROR",traceback=traceback.format_exc(),utc=now())
            with receipt.open("a",encoding="utf-8") as f:f.write(json.dumps(r)+"\n")
            raise
        with receipt.open("a",encoding="utf-8") as f:f.write(json.dumps(r,default=str)+"\n")
        print(json.dumps(r,default=str),flush=True)
if __name__=="__main__":main()
