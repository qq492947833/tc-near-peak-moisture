"""Visible nomenclature only; frozen event IDs and all scientific artists stay intact."""
import json,re,hashlib
import numpy as np
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

def geometry(fig):
    out=[]
    for a in fig.axes:
        out.append(dict(position=list(a.get_position().bounds),xlim=list(a.get_xlim()),ylim=list(a.get_ylim()),xticks=a.get_xticks().tolist(),yticks=a.get_yticks().tolist(),
          lines=[dict(x=np.asarray(l.get_xdata()).tolist(),y=np.asarray(l.get_ydata()).tolist(),color=l.get_color(),lw=l.get_linewidth(),ls=l.get_linestyle())for l in a.lines],
          collections=[dict(paths=[p.vertices.tolist()for p in c.get_paths()],offsets=np.asarray(c.get_offsets()).tolist(),array=None if c.get_array() is None else np.ma.filled(c.get_array(),np.nan).tolist(),norm=[c.norm.vmin,c.norm.vmax],facecolor=c.get_facecolor().tolist(),edgecolor=c.get_edgecolor().tolist())for c in a.collections],
          images=[dict(array=np.asarray(i.get_array()).tolist(),norm=[i.norm.vmin,i.norm.vmax])for i in a.images],
          patches=[dict(vertices=p.get_path().vertices.tolist(),matrix=p.get_patch_transform().get_matrix().tolist(),face=p.get_facecolor(),edge=p.get_edgecolor())for p in a.patches]))
    return json.dumps(out,sort_keys=True,default=str)

def rename(old):
    new=old.replace('depature','departure').replace('Depature','Departure')
    new=re.sub(r'\b[Ff]inal departure\b','Decay',new)
    new=re.sub(r'\b[Ff]inal\b','Decay',new)
    new=new.replace('temporary − Decay','Temporary − Decay').replace('Before temporary vs Decay','Before Temporary vs Decay')
    return new

def apply(fig,figure_id,out):
    fig.canvas.draw()
    before=geometry(fig); changes=[]
    for t in list(fig.findobj(Text)):
        old=t.get_text(); normalized=old.replace('\n',' ')
        if figure_id=='Figure_12' and normalized in ['Final departure','Final depature']:
            changes.append(dict(old=old,new='',action='remove duplicate label below existing Decay stage; no geometry or anchor change',position=list(t.get_position())))
            t.remove();continue
        new=rename(old)
        if new!=old:
            changes.append(dict(old=old,new=new,position=list(t.get_position()),fontsize=t.get_fontsize()))
            t.set_text(new)
    # FixedFormatter/FuncFormatter can restore old strings during draw; wrap only
    # affected category formatters, leaving their locator and numeric positions intact.
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            for which in ['major','minor']:
                formatter=getattr(axis,'get_'+which+'_formatter')()
                ticks=getattr(axis,'get_'+which+'_ticks')()
                if any(rename(formatter(t.get_loc(),i))!=formatter(t.get_loc(),i) for i,t in enumerate(ticks)):
                    def mapped(value,pos=None,original=formatter):return rename(original(value,pos))
                    getattr(axis,'set_'+which+'_formatter')(FuncFormatter(mapped))
    fig.canvas.draw()
    after=geometry(fig)
    assert before==after,'Terminology edit changed scientific geometry'
    assert changes,'Expected visible terminology change missing'
    out.joinpath('terminology_revision.json').write_text(json.dumps(dict(figure_id=figure_id,changes=changes,scientific_geometry_identical=True,geometry_sha256=hashlib.sha256(before.encode()).hexdigest(),internal_identifiers_unchanged=True),ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf8')
