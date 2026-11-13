"""November 2 local display edits. No scientific input or analysis changes."""
from pathlib import Path
import json, hashlib
import numpy as np
from matplotlib.text import Text
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FixedFormatter

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x
def write(p,x): Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf8')
def isbar(a): return getattr(a,'_colorbar',None) is not None
def science(fig):
    return clean([dict(gid=a.get_gid(), xlim=a.get_xlim(),ylim=a.get_ylim(),
      xticks=a.get_xticks(),yticks=a.get_yticks(),
      lines=[dict(x=l.get_xdata(),y=l.get_ydata(),c=l.get_color(),ls=l.get_linestyle()) for l in a.lines],
      images=[dict(values=np.ma.filled(np.ma.asarray(m.get_array(),dtype=float),np.nan),norm=[type(m.norm).__name__,m.norm.vmin,m.norm.vmax],cmap=m.cmap.name) for m in a.images],
      collections=[dict(values=c.get_array(),offsets=c.get_offsets(),paths=[p.vertices for p in c.get_paths()],face=c.get_facecolor(),edge=c.get_edgecolor(),norm=[type(c.norm).__name__,c.norm.vmin,c.norm.vmax]) for c in a.collections],
      patches=[dict(path=p.get_path().vertices,transform=p.get_patch_transform().get_matrix(),face=p.get_facecolor()) for p in a.patches]) for a in fig.axes if not isbar(a)])
def inventory(fig):
    fig.canvas.draw();r=fig.canvas.get_renderer();w,h=fig.get_size_inches()*25.4
    axes=[]
    for i,a in enumerate(fig.axes):
        tight=a.get_tightbbox(r).transformed(fig.dpi_scale_trans.inverted())
        item=dict(panel=a.get_gid() or str(i),bar=isbar(a),rect_mm=np.asarray(a.get_position().bounds)*[w,h,w,h],tight_mm=np.asarray(tight.bounds)*25.4,xlim=a.get_xlim(),ylim=a.get_ylim(),xlabel=a.get_xlabel(),ylabel=a.get_ylabel(),xticks=a.get_xticks(),yticks=a.get_yticks(),texts=[t.get_text() for t in a.findobj(Text) if t.get_visible() and t.get_text()])
        if isbar(a):
            cb=a._colorbar;item['colorbar']=dict(norm=type(cb.norm).__name__,range=[cb.norm.vmin,cb.norm.vmax],ticks=cb.get_ticks(),labels=[t.get_text() for t in a.get_xticklabels()])
        axes.append(item)
    by={d['panel']:d for d in axes}
    gaps={}
    for left,right in [('a','b'),('c','d'),('e','f')]:
        if left in by and right in by:
            l,rr=by[left],by[right]
            gaps[left+right]=dict(plot_horizontal_mm=rr['rect_mm'][0]-sum(l['rect_mm'][[0,2]]),visible_horizontal_mm=rr['tight_mm'][0]-sum(l['tight_mm'][[0,2]]))
    for up,dn in [('a','c'),('c','e'),('b','d'),('d','f')]:
        if up in by and dn in by:
            u,d=by[up],by[dn]
            gaps[up+dn]=dict(plot_vertical_mm=u['rect_mm'][1]-sum(d['rect_mm'][[1,3]]),visible_vertical_mm=u['tight_mm'][1]-sum(d['tight_mm'][[1,3]]))
    return clean(dict(size_mm=[w,h],axes=axes,gaps=gaps,figure_text=[t.get_text() for t in fig.texts if t.get_visible()],legends=[dict(title=l.get_title().get_text(),labels=[t.get_text() for t in l.get_texts()]) for l in fig.legends]))

def six_grid(fig,fid):
    """Translate rows/columns in mm; physical plot areas, scales and fonts fixed."""
    fig.canvas.draw();w,h=fig.get_size_inches()*25.4
    dx=9.;step={'Figure_S18':4.,'Figure_S21':4.5,'Figure_S30':4.5}[fid]
    dh=step*2;neww=w-dx;newh=h+dh
    axes=[(a,np.asarray(a.get_position().bounds)*[w,h,w,h]) for a in fig.axes]
    texts=[(t,t.get_position()) for t in fig.texts]
    legends=[(l,l.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds) for l in fig.legends]
    fig.set_size_inches(neww/25.4,newh/25.4,forward=False)
    for a,p in axes:
        gid=a.get_gid()
        row={'a':2,'b':2,'c':1,'d':1,'e':0,'f':0}.get(gid,0)
        right=gid in ['b','d','f'] or (gid not in ['a','c','e'] and p[0]>w*.5)
        # Numeric colorbar belongs to e and moves with e, retaining its width.
        x=p[0]-(dx if right else 0)
        a.set_axes_locator(None);a.set_position([x/neww,(p[1]+row*step)/newh,p[2]/neww,p[3]/newh])
    for t,(x,y) in texts:
        sy=dh if y>.8 else (step if y>.4 else 0)
        sx=dx if x>.65 else (dx/2 if x>.35 else 0)
        t.set_position(((x*w-sx)/neww,(y*h+sy)/newh))
    for l,(x,y,ww,hh) in legends:
        sy=dh if y>.8 else 0
        sx=dx if x>.65 else (dx/2 if x>.35 else 0)
        l.set_bbox_to_anchor(((x*w-sx)/neww,(y*h+sy)/newh,ww*w/neww,hh*h/newh),transform=fig.transFigure)

def apply(fig,fid,out):
    out=Path(out);fig.canvas.draw();before=inventory(fig);core= science(fig);changes=[];removed=[]
    if fid in ['Figure_S18','Figure_S21','Figure_S30']:
        six_grid(fig,fid);changes.append('Both columns moved 9 mm closer; each row gap increased 4 or 4.5 mm; axes dimensions and map aspect preserved.')
    if fid=='Figure_S23':
        cb=next(a._colorbar for a in fig.axes if isbar(a))
        assert isinstance(cb.norm,LogNorm) and cb.norm.vmin==1 and cb.norm.vmax==85
        ticks=[1,3,9,28,85]
        cb.set_ticks(ticks);cb.set_ticklabels([str(x) for x in ticks]);cb.ax.minorticks_off()
        changes.append('LogNorm(1,85) unchanged; five explicit count labels at 1,3,9,28,85 in near-uniform logarithmic positions.')
    if fid=='Figure_S27':
        for t in list(fig.findobj(Text)):
            text=t.get_text()
            if text.startswith('Outline:') or text=='Each row uses its own native-unit scale' or text in ['Per fixed environmental SD (%)','Stage-duration response per fixed environmental SD (%)']:
                removed.append(text);t.set_visible(False)
        cb=next(a._colorbar for a in fig.axes if isbar(a))
        cb.set_label('Response per fixed environmental SD (%)',fontsize=7.5,labelpad=2)
        changes.append('Upper-right long scale text moved to concise colorbar label; outline/CI and native-axis explanations removed as existing-caption duplicates.')
    fig.canvas.draw();after=inventory(fig)
    if fid=='Figure_S27':
        from audit_panel_alignment import require_matplotlib_panel_alignment
        sub=[a for a in fig.axes if str(a.get_gid()).startswith('b_')]
        require_matplotlib_panel_alignment(fig,axes=sub,column_groups=[[a.get_gid() for a in sub]],json_out=out/'NOV02_b_alignment.json',tolerance_pt=1.5,strict=True)
    assert science(fig)==core,'Scientific values, scales, marks, ticks or CI changed.'
    if fid in ['Figure_S18','Figure_S21','Figure_S30']:
        for old,new in zip(before['axes'],after['axes']):
            assert np.allclose(old['rect_mm'][2:],new['rect_mm'][2:],atol=1e-8),(fid,old['panel'],'plot size changed')
        for key in ['ab','cd','ef']:assert after['gaps'][key]['plot_horizontal_mm']<before['gaps'][key]['plot_horizontal_mm']
        for key in ['ac','ce','bd','df']:assert after['gaps'][key]['plot_vertical_mm']>before['gaps'][key]['plot_vertical_mm']
    write(out/'NOV02_LAYOUT_AUDIT.json',dict(figure=fid,before=before,after=after,changes=changes,removed_text=removed,scientific_values_limits_ticks_and_marks_unchanged=True))
    preview=out/'nov02_previews';preview.mkdir(exist_ok=True)
    fig.savefig(preview/'full.png',dpi=230)
    for i,a in enumerate(fig.axes):
        if isbar(a):continue
        box=a.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.03,1.08)
        fig.savefig(preview/(str(a.get_gid() or i)+'.png'),dpi=250,bbox_inches=box)
