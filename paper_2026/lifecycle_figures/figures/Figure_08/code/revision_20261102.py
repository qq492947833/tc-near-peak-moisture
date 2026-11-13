"""20261102 layout-only changes; input values and axis scales are frozen."""
from pathlib import Path
import json
import numpy as np
from matplotlib.text import Text

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def science(fig):
    return clean([dict(gid=a.get_gid(),xlim=a.get_xlim(),ylim=a.get_ylim(),xticks=a.get_xticks(),yticks=a.get_yticks(),
      lines=[dict(x=l.get_xdata(),y=l.get_ydata()) for l in a.lines],
      images=[dict(values=im.get_array(),lo=im.norm.vmin,hi=im.norm.vmax,cmap=im.cmap.name) for im in a.images],
      collections=[dict(array=c.get_array(),offsets=c.get_offsets(),paths=[p.vertices for p in c.get_paths()]) for c in a.collections]) for a in fig.axes])

def geometry(fig):
    fig.canvas.draw();r=fig.canvas.get_renderer();w,h=fig.get_size_inches()*25.4
    out=[]
    for i,a in enumerate(fig.axes):
        p=a.get_position();b=a.get_tightbbox(r)
        out.append(dict(id=a.get_gid() or f'axis_{i}',is_colorbar=getattr(a,'_colorbar',None)is not None,
          rect_mm=np.array(p.bounds)*[w,h,w,h],visible_mm=None if b is None else np.array(b.bounds)/fig.dpi*25.4,
          ylabel=a.get_ylabel(),ylabel_anchor_px=a.yaxis.label.get_transform().transform(a.yaxis.label.get_position()),
          texts=[dict(text=t.get_text(),position=t.get_position(),bbox_mm=np.array(t.get_window_extent(r).bounds)/fig.dpi*25.4) for t in a.texts if t.get_visible()]))
    return clean(dict(size_mm=[w,h],axes=out))

def apply(fig,fid,out):
    out=Path(out);fig.canvas.draw();before=science(fig);geom_before=geometry(fig)
    by={a.get_gid():a for a in fig.axes if a.get_gid()}
    w,h=fig.get_size_inches()*25.4
    if fid=='Figure_03':
        # Retain physical heights; add 3 mm above row 2. Widen each upper
        # matrix by 4.5 mm so the middle gap shrinks 9 mm with fixed outer edges.
        allpos={a:np.asarray(a.get_position().bounds)*[w,h,w,h] for a in fig.axes}
        upper=[by['a'],by['b']]
        bars=[a for a in fig.axes if getattr(a,'_colorbar',None)is not None]
        upper += [a for a in bars if a.get_position().y0>.55]
        left=[by['a'],by['c']]+[a for a in bars if a.get_position().x0<.5]
        right=[by['b'],by['d']]+[a for a in bars if a.get_position().x0>.5]
        fig.set_size_inches(w/25.4,(h+3)/25.4,forward=False)
        for a,p in allpos.items():
            if a in upper:p[1]+=3
            if a in left:p[2]+=4.5
            if a in right:p[0]-=4.5;p[2]+=4.5
            a.set_axes_locator(None);a.set_position(p/[w,h+3,w,h+3])
    elif fid=='Figure_06':
        bars=[a for a in fig.axes if getattr(a,'_colorbar',None)is not None]
        cb={}
        for a in bars:
            m=a._colorbar.mappable
            for pid in ['b','c']:
                if m in by[pid].collections or m in by[pid].images:cb[pid]=a
        assert set(cb)=={'b','c'}
        baseline=max(a.get_position().y0 for a in cb.values())
        thick=cb['b'].get_position().height
        for pid,a in cb.items():
            p=a.get_position();q=by[pid].get_position()
            a.set_axes_locator(None);a.set_position([q.x0+(q.width-p.width)/2,baseline,p.width,thick])
    elif fid=='Figure_09':
        r=fig.canvas.get_renderer()
        for group in [('a','c','e'),('b','d')]:
            axes=[by[p] for p in group]
            edge=min(t.get_window_extent(r).x0 for a in axes for t in a.get_yticklabels() if t.get_visible() and t.get_text())
            half=max(a.yaxis.label.get_window_extent(r).width for a in axes)/2
            anchor=(edge-3*fig.dpi/72-half)/fig.bbox.width
            for a in axes:
                p=a.get_position();a.yaxis.set_label_coords((anchor-p.x0)/p.width,.5)
    elif fid=='Figure_12':
        a=by['a'];matches=[t for t in a.texts if t.get_text().replace('\n',' ') in ['Final departure','Final depature']]
        assert len(matches)==1
        t=matches[0];t.set_text('Final departure');t.set_position(((139+a.get_xlim()[1])/2,10.8))
    else:raise ValueError(fid)
    fig.canvas.draw();assert science(fig)==before,'Layout revision changed data, scales, ticks, or norms'
    result=dict(figure=fid,scientific_objects_unchanged=True,before=geom_before,after=geometry(fig),no_new_event_marker=True)
    (out/'LOCAL_LAYOUT_20261102.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    d=out/'local_previews';d.mkdir(exist_ok=True)
    fig.savefig(d/'full.png',dpi=250)
    for pid,a in by.items():
        if getattr(a,'_colorbar',None)is not None:continue
        b=a.get_tightbbox(fig.canvas.get_renderer()).expanded(1.04,1.08).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(d/(pid+'.png'),dpi=300,bbox_inches=b,pad_inches=.03)
