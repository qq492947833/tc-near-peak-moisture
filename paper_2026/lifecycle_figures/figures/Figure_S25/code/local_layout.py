"""Nov 2 local geometry only; each measurement uses rendered visible objects."""
from pathlib import Path
import json
import numpy as np
from matplotlib.transforms import Bbox

def _clean(x):
    if isinstance(x, dict): return {k:_clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [_clean(v) for v in x]
    if isinstance(x,np.ndarray):return _clean(x.tolist())
    if isinstance(x,np.generic):return x.item()
    return x

def measure(fig):
    fig.canvas.draw(); r=fig.canvas.get_renderer(); w,h=fig.get_size_inches()*25.4
    return dict(size_mm=[w,h],axes=[dict(id=a.get_gid(),position_mm=np.array(a.get_position().bounds)*[w,h,w,h],tight_mm=np.array(a.get_tightbbox(r).bounds)/fig.dpi*25.4,xlim=a.get_xlim(),ylim=a.get_ylim(),xticks=a.get_xticks(),yticks=a.get_yticks(),xlabel=a.get_xlabel(),ylabel=a.get_ylabel())for a in fig.axes])

def apply(fig,fid,out):
    before=measure(fig); w,h=fig.get_size_inches()*25.4
    by={a.get_gid():a for a in fig.axes if a.get_gid()}; bars=[a for a in fig.axes if getattr(a,'_colorbar',None)]
    exemption=[];change=''
    if fid=='Figure_S13':
        # Add a dedicated variable-key band, without changing any map/scatter size.
        add=15.0; new_h=h+add
        old=[(a,np.array(a.get_position().bounds)*[w,h,w,h])for a in fig.axes]
        texts=[(t,t.get_position())for t in fig.texts]
        fig.set_size_inches(w/25.4,new_h/25.4,forward=False)
        for a,p in old:
            a.set_axes_locator(None)
            a.set_position([p[0]/w,(p[1]+(add if a.get_gid() in 'abcdefgh' else 0))/new_h,p[2]/w,p[3]/new_h])
        for t,(x,y)in texts:t.set_position((x,(y*h+add)/new_h))
        # Both numerical keys are centered on the first three map columns only.
        left=by['e'].get_position().x0;right=by['g'].get_position().x1
        bar_width=(right-left)*.62
        q_y=by['e'].get_position().y0-5.5/new_h
        bars[0].set_position([(left+right-bar_width)/2,q_y,bar_width,2.0/new_h])
        bars[1].set_position([(left+right-bar_width)/2,10.73/new_h,bar_width,2.0/new_h])
        change='q700 key moved beneath e–g; z200 key beneath m–o, each centered on map columns a–c; added 15 mm between variable blocks while retaining every panel size.'
        exemption=[dict(kind='vertical-gutter',reason='Two variable blocks deliberately separated by a quantitative q700 colorbar; within-block row gaps remain equal.')]
    elif fid=='Figure_S17':
        group=[by['d1'],by['d2'],by['d3']]
        fig.canvas.draw();r=fig.canvas.get_renderer()
        boxes=[a.get_tightbbox(r).transformed(fig.transFigure.inverted())for a in group]
        visible=Bbox.union(boxes);target=(by['a'].get_position().x0+by['c'].get_position().x1)/2
        dx=target-(visible.x0+visible.x1)/2
        for a in group:
            p=a.get_position();a.set_position([p.x0+dx,p.y0,p.width,p.height])
        change=f'd group including diagnostic names, ticks, units and intervals translated {dx*w:.6f} mm horizontally; scales, zero positions within axes and limits unchanged.'
    else:raise ValueError(fid)
    after=measure(fig)
    before_limits=[(a['xlim'],a['ylim']) for a in before['axes']]
    after_limits=[(a['xlim'],a['ylim']) for a in after['axes']]
    assert before_limits==after_limits
    Path(out,'LOCAL_LAYOUT_20261102.json').write_text(json.dumps(_clean(dict(figure=fid,before=before,after=after,change=change,limits_exact=True,exemptions=exemption)),indent=2,ensure_ascii=False),encoding='utf8')
    return exemption
