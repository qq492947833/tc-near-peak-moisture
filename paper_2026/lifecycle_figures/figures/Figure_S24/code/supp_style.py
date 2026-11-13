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
    from preservation_guard import verify
    out=Path(out);before=data_state(fig)
    proof=verify(fig,fid,out)
    changed_images=[]
    for ax in fig.axes:
        if ax.images and len(ax.texts)>3:
            for im in ax.images:
                rgba=im.get_cmap()(np.linspace(0,1,256));rgba[:,:3]=.52*rgba[:,:3]+.48
                cm=ListedColormap(rgba,name=im.get_cmap().name+'_readable');cm.set_bad('#E5E8EA')
                im.set_cmap(cm);changed_images.append(ax.get_gid())
    rules=apply_numeric_format(fig)
    for ax in fig.axes:
        for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_color('black')
    fig.canvas.draw()
    assert before==data_state(fig),'Display style altered scientific coordinates: '+fid
    dump(out/'scientific_objects.json',before)
    dump(out/'display_preservation.json',dict(figure=fid,status='PASS',validation='EXACT_FROZEN_SOURCE_TO_ARTIST_CROSSWALK',baseline_layout_restructured=fid in ['Figure_S16','Figure_S17'],source_artist_crosswalk='source_artist_crosswalk.json',current_data_sha256=digest(before),style_step_all_data_objects_equal=True,data_recalculation=False,annotated_heatmaps_lightened=changed_images,axis_display_rules=rules))
