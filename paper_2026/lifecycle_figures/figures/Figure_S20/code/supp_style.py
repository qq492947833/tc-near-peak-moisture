"""Supplementary display-only styling and immutable numerical-object guard."""
import json,hashlib,os,sys,re
from pathlib import Path
import numpy as np
from display_format import format_number,apply_numeric_format
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return str(v)
    return v
def data_state(fig):
    axes=[]
    for ax in fig.axes:
        cs=[]
        for c in ax.collections:
            z=dict(class_=type(c).__name__,offsets=np.asarray(c.get_offsets()),array=None if c.get_array() is None else np.ma.filled(np.ma.asarray(c.get_array(),dtype=float),np.nan),paths=[p.vertices for p in c.get_paths()])
            if hasattr(c,'get_segments'):z['segments']=c.get_segments()
            if hasattr(c,'get_coordinates'):z['coordinates']=c.get_coordinates()
            cs.append(z)
        axes.append(dict(gid=ax.get_gid(),xlim=ax.get_xlim(),ylim=ax.get_ylim(),
            lines=[dict(x=l.get_xdata(),y=l.get_ydata()) for l in ax.lines],
            images=[dict(array=np.ma.filled(np.ma.asarray(im.get_array(),dtype=float),np.nan),extent=im.get_extent(),norm=type(im.norm).__name__,vmin=im.norm.vmin,vmax=im.norm.vmax) for im in ax.images],
            collections=cs,patches=[dict(class_=type(p).__name__,vertices=p.get_path().vertices,transform=p.get_patch_transform().get_matrix()) for p in ax.patches]))
    return clean(axes)
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def audit_and_style(fig,fid,out):
    from matplotlib.colors import ListedColormap
    from matplotlib.text import Text
    from matplotlib.ticker import FixedLocator
    out=Path(out);before=data_state(fig)
    if os.environ.get('SUPP_CURRENT_BASELINE_CAPTURE')=='1':
        dump(out/'scientific_objects.json',before);dump(out/'capture_receipt.json',dict(figure=fid,sha256=digest(before),status='BASELINE_NUMERIC_CAPTURE_ONLY',no_render=True));raise SystemExit(0)
    baseline=Path(__file__).resolve().parents[1]/'reference/scientific_objects.json'
    expected=json.loads(baseline.read_text(encoding='utf-8'))
    if before!=expected:
        dump(out/'scientific_difference.json',dict(expected=expected,current=before))
        raise AssertionError('Frozen scientific objects changed: '+fid)
    # Lighten only annotated raster-cell palettes, keeping each original norm/data.
    changed_images=[]
    for ax in fig.axes:
        if ax.images and len(ax.texts)>3:
            for im in ax.images:
                rgba=im.get_cmap()(np.linspace(0,1,256));rgba[:,:3]=.52*rgba[:,:3]+.48
                cm=ListedColormap(rgba,name=im.get_cmap().name+'_readable')
                cm.set_bad('#E5E8EA');im.set_cmap(cm);changed_images.append(ax.get_gid())
            for t in ax.texts:
                if re.search(r'\d',t.get_text()) and t.get_transform()==ax.transData:t.set_color('black')
    # All visible axis numerals and matrix numbers use the same display policy.
    axis_rules=apply_numeric_format(fig)
    # Common cell/colorbar exponents prevent tiny-value labels from colliding.
    from decimal import Decimal
    from matplotlib.ticker import FuncFormatter,MaxNLocator,FixedLocator
    from display_format import exponent_text
    scales={}
    if fid=='Figure_S06':scales={'c':-3,'d':-3}
    if fid=='Figure_S17':scales={'c':-5,'d':-5,'e':-6,'f':-5}
    for ax in fig.axes:
        gid=ax.get_gid()
        if gid in scales and ax.images:
            exponent=scales[gid];values=np.asarray(ax.images[0].get_array(),dtype=float)
            for t in ax.texts:
                x,y=t.get_position()
                if t.get_transform()==ax.transData and isinstance(x,(int,float,np.number)) and isinstance(y,(int,float,np.number)) and 0<=round(y)<values.shape[0] and 0<=round(x)<values.shape[1] and abs(x-round(x))<1e-7 and abs(y-round(y))<1e-7:
                    suffix='†' if t.get_text().endswith('†') else ''
                    raw=values[round(y),round(x)]
                    t.set_text(format_number(Decimal(str(raw)).scaleb(-exponent),decimals=4)+suffix)
            title=ax.get_title(loc='left') or ax.get_title()
            ax.set_title(title+'\nCells '+exponent_text(exponent),loc='left')
            if ax.get_title():ax.set_title('',loc='center')
    # Fixed colorbar tick strings are regenerated from their actual locations.
    for ax in fig.axes:
        if ax.get_label()!='<colorbar>':continue
        horizontal=ax.get_position().width>ax.get_position().height
        axis=ax.xaxis if horizontal else ax.yaxis
        lim=ax.get_xlim() if horizontal else ax.get_ylim();lo,hi=sorted(lim)
        ticks=MaxNLocator(nbins=4).tick_values(lo,hi)
        ticks=[v for v in ticks if lo-1e-14<=v<=hi+1e-14]
        # Keep logarithmic count bars on their original power-of-ten labels.
        if (ax.get_xscale() if horizontal else ax.get_yscale())=='log':continue
        axis.set_major_locator(FixedLocator(ticks))
        exponent=-3 if fid=='Figure_S06' else 0
        if fid=='Figure_S17':
            # The matching image exposes its own colorbar; no data rescaling.
            for panel in fig.axes:
                if panel.get_gid() in scales and panel.images and getattr(panel.images[0],'colorbar',None) is not None and panel.images[0].colorbar.ax is ax:
                    exponent=scales[panel.get_gid()]
        if not exponent:
            nonzero=[abs(float(v)) for v in ticks if v!=0]
            if nonzero and min(nonzero)<.001 and max(nonzero)<.1:exponent=int(np.floor(np.log10(max(nonzero))))
        axis.set_major_formatter(FuncFormatter(lambda x,p,e=exponent:format_number(Decimal(str(x)).scaleb(-e))))
        label=ax.get_xlabel() if horizontal else ax.get_ylabel()
        label=re.sub(r' \(×10[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+\)','',label)
        if exponent:label+=' ('+exponent_text(exponent)+')'
        if horizontal:ax.set_xlabel(label)
        else:ax.set_ylabel(label)
    if fid=='Figure_S05':
        for ax in fig.axes:
            if ax.get_gid() in ('b','c'):
                pos=ax.get_position();ax.set_position([pos.x0,pos.y0+.04,pos.width,pos.height])
        fig.legends[0].set_bbox_to_anchor((.55,.405))
    if fid=='Figure_S12':
        for ax in fig.axes:
            if ax.get_gid() in ('a','b'):
                pos=ax.get_position();ax.set_position([pos.x0,pos.y0+.025,pos.width,pos.height-.025])
        next(ax for ax in fig.axes if ax.get_gid()=='b').set_xlabel('Annulus (km)')
    if fid=='Figure_S15':fig.subplots_adjust(top=.86)
    for ax in fig.axes:
        for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_color('black')
    fig.canvas.draw()
    assert before==data_state(fig), 'Style changed a data object: '+fid
    dump(out/'scientific_objects.json',before)
    dump(out/'display_preservation.json',dict(figure=fid,status='PASS',baseline_data_sha256=digest(expected),current_data_sha256=digest(before),all_data_objects_equal=True,axes=len(before),data_recalculation=False,annotated_heatmaps_lightened=changed_images,normalization_unchanged=True,axis_display_rules=axis_rules))

