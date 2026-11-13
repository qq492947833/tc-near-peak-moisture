"""Display strings only; frozen data, interval paths, axes scales and norms guarded."""
import json, hashlib
from pathlib import Path
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FixedFormatter
from supp_style import data_state

def apply_labels(fig,fid,out):
    out=Path(out); before=data_state(fig); changes=[]
    axes={a.get_gid():a for a in fig.axes if a.get_gid()}
    def edit(t,new):
        old=t.get_text()
        if old!=new: t.set_text(new); changes.append({'old':old,'new':new})
    def ticks(ax,which,labels):
        axis=ax.xaxis if which=='x' else ax.yaxis
        old=[t.get_text() for t in axis.get_ticklabels()]
        axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
        axis.set_major_formatter(FixedFormatter(labels))
        changes.append({'panel':ax.get_gid(),'axis':which,'old':old,'new':labels})
    if fid=='Figure_S05':
        mapping={'A4-F':'Separate D/E · enclosing F','A4-R':'Separate D/E · continuous R','A3-F':'Pooled D/E · enclosing F','A3-R':'Pooled D/E · continuous R'}
        for t in axes['a'].texts:
            if t.get_text() in mapping: edit(t,mapping[t.get_text()])
        for key in ['b','c']:edit(axes[key].xaxis.label,axes[key].get_xlabel().replace('SID mean squared error','Cyclone mean squared error'))
        edit(axes['d']._left_title,'Shear prediction error'); edit(axes['e']._left_title,'RH prediction error')
    elif fid=='Figure_S06':
        edit(axes['a'].yaxis.label,'Intensification progress I')
        edit(axes['a'].xaxis.label,'Near-peak progress I / wind gain (kt)')
        ticks(axes['b'],'x',['Own\nsubset','Main\n0.90','Alt.','Own\nsubset','Main\n0.90','Alt.'])
        edit(axes['b']._left_title,'Threshold and subset sensitivity')
        for key in ['c','d']:ticks(axes[key],'x',['0.85','0.95','LMI\n− 5 kt'])
    elif fid=='Figure_S07':
        labels=['Main','No 6-h-only return','Valid bridge ends','No Freddy','No WP','No EP','No SI','No NA','No SP','No NI']
        for key in ['c','d']: ticks(axes[key],'y',labels)
        for key in ['e','f']:edit(axes[key].xaxis.label,'Leave-one-out change (10⁻⁴ reference SD)')
        for ax in fig.axes:
            if ax.get_xlabel()=='Deleted-SID results per bin (log colour scale)':edit(ax.xaxis.label,'Leave-one-cyclone-out results per bin')
    fig.canvas.draw()
    after=data_state(fig)
    assert before==after,(fid,'science data changed')
    dump=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    (out/'semantic_preservation.json').write_text(json.dumps({'status':'PASS','figure':fid,'changes':changes,'before_sha256':hashlib.sha256(dump(before).encode()).hexdigest(),'after_sha256':hashlib.sha256(dump(after).encode()).hexdigest(),'data_CI_limits_norms_unchanged':True},ensure_ascii=False,indent=2),encoding='utf-8')

