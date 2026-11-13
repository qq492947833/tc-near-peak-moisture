"""Current display-label clarity; no inference, scale, geometry or style changes."""
import json,hashlib
from pathlib import Path
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator,FixedFormatter
from supp_style import data_state

def apply_labels(fig,fid,out):
    before=data_state(fig); changes=[]
    axes={a.get_gid():a for a in fig.axes if a.get_gid()}
    geom=[list(a.get_position().bounds) for a in fig.axes]
    norms=[(type(c.norm).__name__,c.norm.vmin,c.norm.vmax,c.cmap.name) for a in fig.axes for c in a.collections]
    def edit(t,new):
        old=t.get_text()
        if old!=new:t.set_text(new);changes.append({'old':old,'new':new})
    def ticks(ax,which,labels):
        axis=ax.yaxis if which=='y' else ax.xaxis
        old=[t.get_text() for t in axis.get_ticklabels()]
        axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
        axis.set_major_formatter(FixedFormatter(labels))
        changes.append({'panel':ax.get_gid(),'axis':which,'old':old,'new':labels})
    if fid=='Figure_S18':
        for key in 'abcdef':ticks(axes[key],'y',['Input start\nshifted 16 d','0–500 km\nregion','Wind/translation\nadjustment'])
    elif fid=='Figure_S27':
        edit(axes['a']._left_title,'Stage-duration associations')
        edit(axes['b']._left_title,'Development fraction at fixed D + E duration')
    elif fid=='Figure_S28':
        mapping={'Window-specific support':'Window-specific cyclones','Common-SID support':'Same cyclones across windows'}
        for key in 'ab':
            for t in axes[key].texts:
                if t.get_text() in mapping:edit(t,mapping[t.get_text()])
            ticks(axes[key],'x',['12 h','24 h','Full\nsegment','12 h','24 h','Full\nsegment'])
    fig.canvas.draw()
    after=data_state(fig)
    assert before==after,(fid,'numeric artists changed')
    assert geom==[list(a.get_position().bounds) for a in fig.axes]
    assert norms==[(type(c.norm).__name__,c.norm.vmin,c.norm.vmax,c.cmap.name) for a in fig.axes for c in a.collections]
    dump=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    (Path(out)/'semantic_preservation.json').write_text(json.dumps({'status':'PASS','figure':fid,'changes':changes,'before_sha256':hashlib.sha256(dump(before).encode()).hexdigest(),'after_sha256':hashlib.sha256(dump(after).encode()).hexdigest(),'data_CI_limits_norms_unchanged':True,'axes_positions_unchanged':True},ensure_ascii=False,indent=2),encoding='utf-8')
