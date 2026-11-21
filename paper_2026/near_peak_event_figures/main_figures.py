"""Render frozen FLTCCF display data at 180 mm. Source paths are supplied by the caller."""
from pathlib import Path
import argparse, hashlib, json, zipfile, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from PIL import Image, ImageChops
import numpy as np
import pandas as pd
from io import BytesIO

BLUE, RUST, GREEN, INK, GREY = "#287FA0", "#BC693F", "#428C79", "#33454F", "#8899A2"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8.5,"axes.labelsize":8,
                     "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.linewidth":.65,"axes.labelcolor":INK,"text.color":INK,
                     "xtick.color":INK,"ytick.color":INK,"axes.edgecolor":GREY,"lines.linewidth":1.2,
                     "xtick.major.size":2.3,"ytick.major.size":2.3,"pdf.fonttype":42,"axes.unicode_minus":True,
                     "savefig.facecolor":"white","legend.frameon":False})
parser=argparse.ArgumentParser()
parser.add_argument("--repo",type=Path,required=True)
parser.add_argument("--source-assets",type=Path,default=Path(__file__).resolve().parent/"inputs"/"plot_assets.zip")
parser.add_argument("--out",type=Path,default=Path("out"))
parser.add_argument("--figures",nargs="+",type=int,default=[1,2,3,4,5])
args=parser.parse_args()
root=args.repo/"paper_2026/lifecycle_figures/figures"
out=args.out
out.mkdir(exist_ok=True,parents=True)
sources={}
values={}
def remember(key,raw):
    sources[key]=hashlib.sha256(raw).hexdigest()
    return raw
def read(path,**kw):
    return pd.read_csv(BytesIO(remember(str(path),path.read_bytes())),float_precision="round_trip",**kw)
def archive(path):
    with zipfile.ZipFile(args.source_assets) as z:
        return remember(f"{args.source_assets}!{path}",z.read(path))
def fdata(n,fn):
    return read(root/f"Figure_{n:02}"/fn)
def fig(h):
    return plt.figure(figsize=(180/25.4,h/25.4),dpi=100)
def axis(f,x,y,w,h): # top-down physical millimetres
    H=f.get_figheight()*25.4
    return f.add_axes([x/180,1-(y+h)/H,w/180,h/H])
def title(ax,lbl,name):
    ax.set_title(f"$\\mathbf{{{lbl}}}$  {name}",loc="left",pad=5)
def zero(ax):
    ax.axhline(0,ls=":",lw=.7,color=GREY,zorder=0)
def timing(ax,lab=False,base="Hours from departure"):
    ax.axvline(0,lw=.65,ls=":",color=GREY,zorder=0)
    ax.set_xlim(-25.5,25.5)
    ax.set_xticks([-24,-12,0,12,24])
    if lab: ax.set_xlabel(base)
def curve(ax,x,v,lo,hi,color,label=None,ls="-",marker="o"):
    x,v,lo,hi=map(lambda z:np.array(z,dtype=float),[x,v,lo,hi])
    assert np.isfinite(v).all() and np.all(lo<=hi)
    ax.fill_between(x,lo,hi,color=color,alpha=.16,lw=0)
    ax.plot(x,v,c=color,ls=ls,marker=marker,ms=2.3,label=label)
def save(f,num):
    name=f"Figure_{num:02}"
    f.canvas.draw()
    bboxes=[list(np.round(np.array(a.get_position().bounds)*[180,f.get_figheight()*25.4,180,f.get_figheight()*25.4],3)) for a in f.axes]
    f.savefig(out/(name+".png"),dpi=900)
    plt.close(f)
    with Image.open(out/(name+".png")) as im:
        rgb=im.convert("RGB")
    rgb.save(out/(name+".png"),dpi=(900,900),optimize=False)
    rgb.save(out/(name+".tiff"),compression="tiff_lzw",dpi=(900,900))
    with Image.open(out/(name+".tiff")) as t:
        assert t.mode=="RGB" and t.tag_v2[259]==5 and ImageChops.difference(rgb,t).getbbox() is None
    print(name,rgb.size,flush=True)
def forests(ax,rows,labels,unit):
    y=np.arange(len(rows))[::-1];v,lo,hi=np.array(rows,dtype=float).T
    assert np.all(lo<=v) and np.all(v<=hi)
    ax.axvline(0,ls=":",color=GREY,lw=.8)
    ax.errorbar(v,y,xerr=[v-lo,hi-v],fmt="o",ms=4,color=BLUE,capsize=2.5,lw=1.15)
    ax.set(yticks=y,yticklabels=labels,ylim=(-.6,len(y)-.4),xlabel=unit)
    ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
def draw1():
    import shapefile
    from matplotlib.collections import LineCollection
    s=fdata(1,"input/lifecycle_schematic.csv")
    k=json.loads(remember("schematic JSON",(root/"Figure_01/input/lifecycle_schematic.json").read_bytes()))
    ev=fdata(1,"input/all_return_events.csv").query("ledger_eligible")
    bas=read(root/"Figure_01/input/basin_recurrence_proportions.csv",keep_default_na=False).query("mode=='new'")
    assert len(ev)==133 and ev.sid.nunique()==123 and bas.eligible_SID.sum()==2340 and bas.recurrent_SID.sum()==123
    f=fig(130);a=axis(f,14,11,158,51)
    title(a,"a","Near-peak departures and return")
    a.plot(s.hours,s.wind_kt,c=INK,marker="o",ms=3)
    level=k["I090_kt"]
    a.axhline(level,c=GREY,lw=.75,ls="--",zorder=0)
    a.text(1,level+2.5,"Near-peak threshold",fontsize=8,color=GREY)
    for low,high in k["episode_ranges_h"]:
        a.axvspan(low,high,color=GREEN,alpha=.14,lw=0,zorder=0)
    a.text(39,118,"Episode containing\nfirst LMI",ha="center",va="center",color=GREEN,fontsize=8)
    for tx,ty,h,txt,align in [(43,48,48,"Temporary departure","right"),(53,51,54,"Trough","center"),(64,116,60,"Near-peak return","center"),(80,46,72,"Final departure","right")]:
        a.annotate(txt,xy=(h,s.set_index("hours").loc[h,"wind_kt"]),xytext=(tx,ty),ha=align,fontsize=8,color=INK,arrowprops=dict(arrowstyle="-",lw=.65,color=GREY))
    a.set(xlim=(-4,82),ylim=(30,130),xticks=[0,12,24,36,48,60,72],yticks=[35,60,85,100],xlabel="Time from 35-kt start (h)",ylabel="Wind speed (kt)")
    b=axis(f,12,82,100,35)
    prefix="paper_2026/lifecycle_figures/figures/Figure_01/input/coastline/ne_110m_coastline."
    lines=[]
    sf=shapefile.Reader(shp=BytesIO(archive(prefix+"shp")),shx=BytesIO(archive(prefix+"shx")))
    for shape in sf.iterShapes():
        pts=np.array(shape.points);pts[:,0]%=360
        for start,end in zip(list(shape.parts),list(shape.parts)[1:]+[len(pts)]):
            arr=pts[start:end]
            lines.extend(seg for seg in np.split(arr,np.flatnonzero(abs(np.diff(arr[:,0]))>180)+1) if len(seg)>1)
    b.add_collection(LineCollection(lines,colors="#A6B6BC",linewidths=.45))
    for lon in [30,100,180]: b.plot([lon,lon],[0,65],"--",color=GREY,lw=.4)
    for lon in [10,135,290]: b.plot([lon,lon],[-60,0],"--",color=GREY,lw=.4)
    b.axhline(0,color=GREY,lw=.4,ls="--")
    for i,m,c,lab in [(1,"o",BLUE,"First"),(2,"D",RUST,"Second"),(3,"^","#725B91","Third or later")]:
        z=ev[ev.return_order.eq(i) if i<3 else ev.return_order.ge(i)]
        b.scatter(z.return_lon%360,z.return_lat,c=c,marker=m,s=11 if i==1 else 20,edgecolors="white",linewidths=.35,label=lab)
    for basin,x,y in [("NA",335,42),("EP",243,5),("WP",150,48),("NI",74,37),("SI",68,-41),("SP",205,-40)]:
        b.text(x,y,basin,fontsize=7.5,color="#60777E",ha="center",va="center")
    b.set(xlim=(0,360),ylim=(-60,65),aspect="equal",xticks=[0,90,180,270,360],xticklabels=["0°","90°E","180°","90°W","0°"],yticks=[-60,0,60],yticklabels=["60°S","0°","60°N"])
    title(b,"b","Near-peak return locations");b.legend(loc="upper center",bbox_to_anchor=(.50,-.19),ncol=3,columnspacing=.9,handletextpad=.2)
    c=axis(f,128,80,45,37);title(c,"c","Observed fraction")
    y=np.arange(6)[::-1];c.barh(y,bas.percentage,color=BLUE,height=.48)
    for ypos,row in zip(y,bas.itertuples()):
        c.text(row.percentage+.13,ypos,f"{row.percentage:.2f}%  {row.recurrent_SID}/{row.eligible_SID}",va="center",fontsize=7.7)
    c.set(yticks=y,yticklabels=bas.basin_genesis,xlim=(0,15),xticks=[0,4,8],xlabel="Recurrent cyclones (%)")
    values["numbers"]=[len(ev),ev.sid.nunique(),int(bas.eligible_SID.sum())];save(f,1)
def draw2():
    tr=fdata(3,"inputs/direct/fig5_trajectories.csv")
    env=fdata(3,"inputs/direct/environmental_contrasts.csv")
    f=fig(148);cols=[15,74,133];w=41
    dat=[("midRH","Mid-level RH","RH (%)","ΔRH (pp)"),("shear200_850","Deep-layer shear","Shear (m s⁻¹)","ΔShear (m s⁻¹)"),("PI_headroom","PI − V","PI − V (m s⁻¹)","Δ(PI − V) (m s⁻¹)")]
    for i,(v,t,u,du) in enumerate(dat):
        for j,ser in enumerate([["internal","final"],["difference"]]):
            ax=axis(f,cols[i],12+j*46,w,27);title(ax,chr(97+i+j*3),t if not j else t+" difference")
            for name in ser:
                d=tr[(tr.variable==v)&(tr.series==name)].sort_values("lag_h")
                assert len(d)==9
                curve(ax,d.lag_h,d.estimate,d.lo,d.hi,{"internal":BLUE,"final":RUST,"difference":INK}[name],{"internal":"Temporary departure","final":"Decay","difference":"Paired difference"}[name],ls="--" if name=="final" else "-")
            timing(ax,j==1);ax.set_ylabel(du if j else u);ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            if j: zero(ax)
    f.legend([Line2D([],[],color=BLUE,marker="o",ms=3),Line2D([],[],color=RUST,ls="--",marker="o",ms=3)],["Temporary departure","Decay"],loc="upper center",bbox_to_anchor=(.52,.996),ncol=2)
    sub=env[(env.coverage=="coverage_117")&(env.variable=="midRH")].set_index("model")
    geo=sub.loc["geographical"];state=sub.loc["geographical_state"]
    rows=[[geo.original_raw,geo.original_lo,geo.original_hi],[geo.overlap_raw,geo.overlap_raw_lo,geo.overlap_raw_hi],[geo.standardized,geo.lo,geo.hi],[state.standardized,state.lo,state.hi]]
    g=axis(f,70,109,97,27);title(g,"g","Pre-departure RH contrast")
    forests(g,rows,["All pairs (117)","Common support (105)","Geography and season (105)","Additional storm state (105)"],"Temporary − decay (percentage points)")
    values["g"]=np.array(rows).tolist();values["PI_support"]=tr[(tr.variable=="PI_headroom")&(tr.series=="difference")][["lag_h","n_sid"]].values.tolist()
    save(f,2)

def draw3():
    from matplotlib.patches import Circle,FancyArrowPatch
    from matplotlib.colors import LinearSegmentedColormap
    pair=fdata(4,"inputs/direct/paired_flux_components.csv")
    pressure=fdata(4,"inputs/direct/pressure_components.csv")
    adj=fdata(4,"inputs/direct/exchange_background80.csv")
    where="cohort=='original_post80' and window=='0_24h' and factor==1 and sector=='all'"
    allz=pair.query(where+" and layer=='all'")
    f=fig(140);a=axis(f,10,10,160,40);title(a,"a","Moving-boundary exchange over 0–24 h (10¹² kg)")
    a.set(xlim=(0,160),ylim=(0,40),aspect="equal");a.axis("off")
    for x,which,col,lab in [(40,"temporary",BLUE,"Temporary departure"),(117,"final",RUST,"Decay")]:
        q=allz[allz.quantity==which].set_index("kind").estimate/1e12
        a.add_patch(Circle((x,19),11,facecolor="#F5F8FA",edgecolor=col,lw=1.1))
        a.text(x,36,lab,ha="center",c=col,weight="bold")
        a.text(x,20,f"300 km\nNet {q['net']:.2f}",ha="center",va="center")
        for k,yy,left,right in [("in",26,x-27,x-8.5),("out",12,x+8.5,x+27)]:
            mean=q[k];a.add_patch(FancyArrowPatch((left,yy),(right,yy),arrowstyle="-|>",mutation_scale=mean*1.2,lw=mean*.36,color=col))
            a.text((left+right)/2+(-2 if k=="in" else 2),yy+2.6,("In " if k=="in" else "Out ")+f"{mean:.2f}",ha="center")
    b=axis(f,26,63,38,33);title(b,"b","Net inward by layer")
    p=pressure.query(where+" and kind=='net' and quantity in ['temporary','final']")
    order=["500-600_hPa","600-700_hPa","700-850_hPa","850-925_hPa"]
    z=p.pivot(index="layer",columns="quantity",values="estimate").loc[order,["temporary","final"]].to_numpy()/1e12
    cmap=LinearSegmentedColormap.from_list("signed",["#DBAA8D","#FAFAF8","#9FC3D2"])
    b.imshow(z,vmin=-3.4,vmax=3.4,cmap=cmap,aspect="auto")
    for i in range(4):
        for j in range(2): b.text(j,i,f"{z[i,j]:.2f}".replace("-","−"),ha="center",va="center",size=8.5)
    b.set(xticks=[0,1],xticklabels=["Temporary","Decay"],yticks=range(4),yticklabels=[l.replace("_hPa","").replace("-","–") for l in order],ylabel="Layer (hPa)",xlabel="Net inward (10¹² kg)");b.tick_params(length=0)
    c=axis(f,113,63,57,33);title(c,"c","Gross exchange differences")
    cd=allz.query("quantity=='difference'").set_index("kind").loc[["in","out"],["estimate","low","high"]]/1e12
    forests(c,cd.values,["Gross inward","Gross outward"],"Temporary − decay (10¹² kg)")
    d=axis(f,67,113,102,17);title(d,"d","Net inward exchange over 0–24 h")
    q=adj[adj.metric=="post_r300_net"];raw=q[q.model=="geographical"].set_index("target")
    dd=pd.concat([raw.loc[["original80_raw","restricted_raw","balanced"]],q[(q.model=="geographical_state")&(q.target=="balanced")].set_index("target")])
    assert list(dd.n_target)==[80,74,74,74]
    forests(d,dd[["estimate","low","high"]].values/1e12,["All pairs (80)","Common support (74)","Geography and season (74)","Additional storm state (74)"],"Temporary − decay (10¹² kg)")
    values["gross_diff_CI"]=cd.values.tolist();values["net_diff_CI"]=(dd[["estimate","low","high"]].values/1e12).tolist();save(f,3)
def draw4():
    p=root/"Figure_05";meta=json.loads((p/"inputs.json").read_text())
    fn=meta["time"]["relative_path"];data=archive(f"paper_2026/lifecycle_figures/figures/Figure_05/{fn}")
    assert hashlib.sha256(data).hexdigest()==meta["time"]["sha256"]
    time=pd.read_csv(BytesIO(data),float_precision="round_trip",dtype={"level":str})
    point=fdata(5,"inputs/S10a_primary_six.csv").set_index("metric")
    f=fig(138);cols=[15,74,133];w=41
    defs=[("KE","Horizontal KE"),("APE","Dry APE"),("C_A_to_K","APE-to-KE conversion"),("vertical_input_P_Wkg","Pressure-flux input"),("vertical_input_K_Wkg","Vertical KE input"),("vertical_input_A_Wkg","Vertical APE input")]
    for i,(metric,label) in enumerate(defs):
        row,col=divmod(i,3);y=20+60*row
        z=time[(time.scheme=="main")&(time.region=="r0_300")&(time.window==1)&(time.metric==metric)&(time.level==("700" if i<3 else "500-850"))&(time.lag_h.notna())].sort_values("lag_h")
        assert len(z)==9,(metric,len(z),time.columns)
        factor=1 if i<2 else 1e4
        a=axis(f,cols[col],y,w,21);b=axis(f,cols[col],y+27,w,16.5)
        title(a,chr(97+i),label);a.plot(z.lag_h,z.temporary_mean*factor,c=BLUE);a.plot(z.lag_h,z.final_mean*factor,c=RUST,ls="--")
        a.set_xlim(-26,38);a.set_xticks([]);a.axvline(0,lw=.6,ls=":",c=GREY)
        a.set_ylabel("J kg⁻¹" if i<2 else "10⁻⁴ W kg⁻¹")
        a.yaxis.set_major_locator(MaxNLocator(nbins=3));b.yaxis.set_major_locator(MaxNLocator(nbins=3))
        lo=z.lower.where(z.n_sid>=2).values*factor;hi=z.upper.where(z.n_sid>=2).values*factor
        b.plot(z.lag_h,z.estimate*factor,c=INK);b.fill_between(z.lag_h.to_numpy(),lo,hi,color=INK,alpha=.16,lw=0)
        t=point.loc[metric];assert t.n_sid==117
        b.errorbar(34,t.estimate*factor,yerr=[[factor*(t.estimate-t.lower)],[factor*(t.upper-t.estimate)]],fmt="D",color=INK,capsize=2,ms=3)
        zero(b);b.axvline(0,lw=.6,ls=":",c=GREY);b.set(xlim=(-26,38),xticks=[-24,0,24,34],xticklabels=["−24","0","24","Pre"],ylabel="Δ",xlabel="Hours from departure" if row else "")
        values[metric]={"level":str(t.level),"unit":str(t.units),"n_sid_pre":int(t.n_sid),"pre":t[["estimate","lower","upper"]].tolist()}
    f.legend([Line2D([],[],c=BLUE),Line2D([],[],c=RUST,ls="--"),Line2D([],[],c=INK)],["Temporary departure","Decay","Δ Temporary − decay"],loc="upper center",bbox_to_anchor=(.52,.999),ncol=3,columnspacing=1)
    f.text(.52,1-9.5/138,"700 hPa  ·  2–16 d",ha="center",va="center",color=INK,weight="bold")
    f.text(.52,1-70/138,"500–850 hPa layer  ·  2–16 d",ha="center",va="center",color=INK,weight="bold")
    save(f,4)
def draw5():
    t=fdata(6,"inputs/MAIN_RISE_COMPARISON_source.csv")
    gvals=fdata(6,"inputs/panel_g_source_rows.csv").sort_values("window")
    rates=read(args.repo/"paper_2026/physics/vertical_extension/fig6e_rates/data/FIG6E_RATE_ESTIMATES.csv").query("support=='nine_time_common'")
    assert len(rates)==27 and set(rates.n_sid)=={117} and set(rates.n_events)=={126} and set(rates.n_draws)=={2000}
    pivot=rates.pivot(index="relative_hour",columns="metric",values="estimate")
    assert np.allclose(pivot.Gamma_K_total,pivot.Gamma_K_h+pivot.Gamma_K_p,rtol=0,atol=1e-14)
    f=fig(147);cols=[15,74,133];w=41
    defs=[("wind","Wind speed","kt",BLUE,1),("KE","Horizontal KE","J kg⁻¹",BLUE,1),("APE","Dry APE","J kg⁻¹",GREEN,1),("C_A_to_K","APE-to-KE conversion","10⁻⁴ W kg⁻¹",RUST,1e4),("rates","KE canonical transfer","10⁻⁴ W kg⁻¹",INK,1e4),("vertical_input_K_Wkg","Vertical KE input","10⁻⁴ W kg⁻¹",BLUE,1e4)]
    for i,(metric,label,unit,c,factor) in enumerate(defs):
        row,col=divmod(i,3);a=axis(f,cols[col],11+row*46,w,27);title(a,chr(97+i),label)
        if metric=="rates":
            for m,co,lab,sym in [("Gamma_K_h",BLUE,"Horizontal","o"),("Gamma_K_p",GREEN,"Pressure direction","s"),("Gamma_K_total",INK,"Total","D")]:
                s=rates[rates.metric==m].sort_values("relative_hour");curve(a,s.relative_hour,s.estimate*factor,s.ci_lower*factor,s.ci_upper*factor,co,lab,marker=sym)
            handles,labels=a.get_legend_handles_labels()
            f.legend(handles,labels,loc="upper center",bbox_to_anchor=(.51,.337),ncol=3,columnspacing=1.1)
        else:
            s=t[t.metric==metric].sort_values("relative_hour");assert len(s)==9
            curve(a,s.relative_hour,s.estimate*factor,s.lower*factor,s.upper*factor,c)
        timing(a,row==1,"Time from trough (h)");a.set_ylabel(unit)
        a.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if i>=3: zero(a)
    g=axis(f,53,119,110,16.5);title(g,"g","Wind-matched dry APE contrast at 700 hPa")
    assert list(gvals.window)==[0,1,2] and set(gvals.n_sid)=={108} and set(gvals.n_events)=={115}
    forests(g,gvals[["estimate","lower","upper"]].values,[">16 d","2–16 d","<2 d"],"Reintensification − decay (J kg⁻¹)")
    values["g"]=gvals[["window","estimate","lower","upper","n_events","n_sid"]].values.tolist()
    values["e"]=rates[["relative_hour","metric","n_events","n_sid","estimate","ci_lower","ci_upper"]].values.tolist();save(f,5)
if __name__=="__main__":
    for n in args.figures:
        sources.clear();values.clear()
        {1:draw1,2:draw2,3:draw3,4:draw4,5:draw5}[n]()


