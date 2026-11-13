"""Label-only revision; frozen values, intervals, axes, norms and colours are guarded."""
from pathlib import Path
import hashlib, json
import numpy as np
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FixedFormatter
from supp_style import data_state, clean

def science(fig):
    return clean(dict(data=data_state(fig), axes=[dict(
        gid=ax.get_gid(), xticks=ax.get_xticks(), yticks=ax.get_yticks(),
        xscale=ax.get_xscale(), yscale=ax.get_yscale(),
        images=[dict(norm=type(m.norm).__name__,vmin=m.norm.vmin,vmax=m.norm.vmax,cmap=m.cmap.name) for m in ax.images],
        collections=[dict(norm=type(m.norm).__name__,vmin=m.norm.vmin,vmax=m.norm.vmax,cmap=m.cmap.name,face=m.get_facecolors(),edge=m.get_edgecolors()) for m in ax.collections],
        patches=[dict(face=p.get_facecolor(),edge=p.get_edgecolor()) for p in ax.patches]) for ax in fig.axes]))

def apply_labels(fig,fid,out):
    out=Path(out);fig.canvas.draw();before=science(fig);changes=[]
    axes={a.get_gid():a for a in fig.axes if a.get_gid()}
    fonts_before=[t.get_fontsize() for t in fig.findobj(Text)]
    def text(t,new):
        old=t.get_text()
        if old!=new:t.set_text(new);changes.append(dict(old=old,new=new))
    def ticks(ax,which,labels):
        axis=ax.xaxis if which=='x' else ax.yaxis
        old=[t.get_text() for t in axis.get_ticklabels()]
        axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
        axis.set_major_formatter(FixedFormatter(labels))
        changes.append(dict(panel=ax.get_gid(),axis=which,old=old,new=labels))
    if fid=='Figure_S15':
        ticks(axes['a'],'y',['All pairs (80)\nUnadjusted','Geographic overlap (74)\nUnadjusted','Same 74 pairs\nLocation/season adjusted','Same 74 pairs\nAlso storm-state adjusted'])
        for k in 'abc':text(axes[k].xaxis.label,'Temporary − Decay (10¹² kg)')
    elif fid=='Figure_S23':
        ticks(axes['c'],'x',['All valid pairs\nUnadjusted','Overlap subset\nUnadjusted','Overlap subset\nCovariate-adjusted'])
    else:raise ValueError(fid)
    fig.canvas.draw();after=science(fig)
    assert before==after,(fid,'Data, confidence intervals, coordinates, ticks, norm or colours changed')
    assert fonts_before==[t.get_fontsize() for t in fig.findobj(Text)],'Font sizes changed'
    digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    (out/'semantic_preservation.json').write_text(json.dumps(dict(status='PASS',figure=fid,changes=changes,before_sha256=digest(before),after_sha256=digest(after),data_CI_limits_ticks_norms_colours_unchanged=True,font_sizes_unchanged=True),ensure_ascii=False,indent=2),encoding='utf-8')
    # QA render from the same Matplotlib figure, including every panel at readable size.
    fig.savefig(out/'semantic_whole_QA.png',dpi=230)
    for k,ax in axes.items():
        if len(k)==1 and k.isalpha():
            box=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.05,1.06)
            fig.savefig(out/f'semantic_panel_{k}_QA.png',dpi=300,bbox_inches=box)
