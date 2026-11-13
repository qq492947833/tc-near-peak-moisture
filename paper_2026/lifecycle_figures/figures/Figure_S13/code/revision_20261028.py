"""20261028 display-only stage names, true numerical color keys and measured layout."""
from pathlib import Path
import json,re,hashlib
import numpy as np
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator
from matplotlib.colors import Normalize,TwoSlopeNorm,LogNorm
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v)for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float)and not np.isfinite(x):return None
    return x
def dump(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False),encoding='utf8')
def isbar(ax):return getattr(ax,'_colorbar',None)is not None
def scientific(fig):
    return [dict(gid=ax.get_gid(),lines=[dict(x=l.get_xdata(),y=l.get_ydata())for l in ax.lines],
      images=[dict(array=np.ma.filled(np.ma.asarray(im.get_array(),dtype=float),np.nan))for im in ax.images],
      collections=[dict(array=c.get_array(),offsets=c.get_offsets(),paths=[p.vertices for p in c.get_paths()])for c in ax.collections])
      for ax in fig.axes if not isbar(ax)]
def trim_top(fig,upper,mm):
    fig.canvas.draw();w,h=fig.get_size_inches()*25.4
    old=[(a,np.asarray(a.get_position().bounds)*[w,h,w,h])for a in fig.axes]
    ft=[(t,t.get_position())for t in fig.texts]
    fl=[(l,l.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds)for l in fig.legends]
    fig.set_size_inches(w/25.4,(h-mm)/25.4,forward=False)
    for a,p in old:
        a.set_axes_locator(None);a.set_position([p[0]/w,(p[1]-(mm if a in upper else 0))/(h-mm),p[2]/w,p[3]/(h-mm)])
    for t,(x,y)in ft:t.set_position((x,(y*h-mm)/(h-mm)))
    for l,(x,y,ww,hh)in fl:l.set_bbox_to_anchor((x,(y*h-mm)/(h-mm),ww,hh*h/(h-mm)),transform=fig.transFigure)
def apply(fig,fid,out):
    out=Path(out);fig.canvas.draw()
    before=clean(scientific(fig));changes=[]
    # Res is an internal frozen category; only final display Text changes.
    for t in fig.findobj(Text):
        old=t.get_text()
        new=re.sub(r'\bResolution\b','Decay',old)
        new=re.sub(r'\bRes\b(?!\.)','Dec',new)
        if new!=old:t.set_text(new);changes.append({'before':old,'after':new})
    # All continuous zero-centred linear keys show real endpoints and midpoint ticks.
    bars=[]
    for ax in fig.axes:
        cb=getattr(ax,'_colorbar',None)
        if cb is None:continue
        norm=cb.norm;lo=float(norm.vmin);hi=float(norm.vmax)
        if isinstance(norm,LogNorm):continue
        if lo<0<hi and np.isclose(-lo,hi,rtol=1e-10):
            ticks=np.linspace(lo,hi,5);cb.set_ticks(ticks)
        elif lo>=0:
            ticks=np.linspace(lo,hi,5);cb.set_ticks(ticks)
        else:ticks=cb.get_ticks()
        bars.append(ax)
    by={ax.get_gid():ax for ax in fig.axes if ax.get_gid()}
    if fid=='Figure_03':
        upper=[by['a'],by['b']]
        upper += [ax for ax in bars if ax.get_position().y0>.55]
        trim_top(fig,upper,2)
    if fid=='Figure_04':
        g=by['g'];ca=bars[-1];p=ca.get_position();ca.set_position([p.x0,g.get_position().y0-.068,p.width,p.height])
    if fid=='Figure_05':
        ca=bars[0];p=ca.get_position();ca.set_position([p.x0,p.y0+7/(fig.get_figheight()*25.4),p.width,p.height])
    if fid=='Figure_08':
        # Common centerline close to widest tick field, with a real 3pt gap.
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for prefix in [('a','c','e'),('b','d','f')]:
            group=[a for a in fig.axes if (a.get_gid()or'').split('_')[0]in prefix]
            left=min(t.get_window_extent(renderer).x0 for a in group for t in a.get_yticklabels()if t.get_visible()and t.get_text())
            label_width=max(a.yaxis.label.get_window_extent(renderer).width for a in group)
            center=(left-3*fig.dpi/72-label_width/2)/fig.bbox.width
            for a in group:
                p=a.get_position();y=.58 if (a.get_gid()or'')[0]in 'cdef' and a.get_gid().endswith('mean') else .5
                a.yaxis.set_label_coords((center-p.x0)/p.width,y)
    if fid=='Figure_S16':
        trim_top(fig,[by['a'+str(i)]for i in range(1,7)],6)
    if fid=='Figure_S17':
        a=by['a'];letter_x=a.get_position().x0-18/(fig.get_figwidth()*72)
        for t in fig.texts:
            if t.get_text()=='d':t.set_position((letter_x,.525))
            if t.get_text()=='Layer inputs, 500–850 hPa':t.set_position((a.get_position().x0,.525))
        for l in fig.legends:l.set_bbox_to_anchor((.965,.533))
    fig.canvas.draw()
    assert clean(scientific(fig))==before,'Display revision modified plotted scientific values'
    # Existing formatters retain per-key exponent and scientific-unit semantics.
    from display_format import apply_numeric_format
    apply_numeric_format(fig)
    # Reassert exact key positions in case a formatter chose interior ticks.
    for ax in bars:
        cb=ax._colorbar;lo,hi=float(cb.norm.vmin),float(cb.norm.vmax)
        if not isinstance(cb.norm,LogNorm)and (lo>=0 or (lo<0<hi and np.isclose(-lo,hi))):
            from matplotlib.ticker import FuncFormatter
            from display_format import format_number
            cb.set_ticks(np.linspace(lo,hi,5))
            label=ax.get_xlabel()or ax.get_ylabel()
            exp=0
            match=re.search(r'×10([⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹]+)',label)
            if match:exp=int(match.group(1).translate(str.maketrans('⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹','-+0123456789')))
            cb.formatter=FuncFormatter(lambda v,pos,e=exp:format_number(v/(10**e),decimals=4))
            cb.update_ticks()
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    inventory=[];colour=[];preview=out/'final_previews';preview.mkdir(exist_ok=True)
    for i,ax in enumerate(fig.axes):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        gid=ax.get_gid()or('colorbar_'+str(i)if isbar(ax)else'axis_'+str(i))
        p=ax.get_position()
        item=dict(panel=gid,rect_mm=np.asarray(p.bounds)*([fig.get_figwidth()*25.4,fig.get_figheight()*25.4]*2),
           xlim=ax.get_xlim(),ylim=ax.get_ylim(),xticks=ax.get_xticks(),yticks=ax.get_yticks(),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),
           title=ax.get_title(loc='left')or ax.get_title(),ylabel_bbox_px=ax.yaxis.label.get_window_extent(renderer).bounds,
           texts=[t.get_text()for t in ax.findobj(Text)if t.get_visible()and t.get_text()],
           legend=[]if ax.get_legend()is None else[t.get_text()for t in ax.get_legend().get_texts()])
        inventory.append(item)
        cb=getattr(ax,'_colorbar',None)
        if cb:
            arr=cb.mappable.get_array();finite=np.asarray([])if arr is None else np.ma.asarray(arr).compressed()
            finite=finite[np.isfinite(finite)]
            src=[a.get_gid()for a in fig.axes if cb.mappable in a.images or cb.mappable in a.collections]
            colour.append(dict(key=gid,panels=src,semantic=ax.get_xlabel()or ax.get_ylabel(),norm=type(cb.norm).__name__,
               vmin=cb.norm.vmin,vmax=cb.norm.vmax,ticks=cb.get_ticks(),tick_labels=[t.get_text()for t in (ax.get_xticklabels()if cb.orientation=='horizontal'else ax.get_yticklabels())],
               cmap=cb.cmap.name,data_range=None if not len(finite)else[finite.min(),finite.max()],
               endpoints_present=bool(np.isclose(cb.get_ticks()[0],cb.norm.vmin)and np.isclose(cb.get_ticks()[-1],cb.norm.vmax)),
               clipped_values=int(np.sum((finite<cb.norm.vmin)|(finite>cb.norm.vmax)))if len(finite)else None))
        if not cb:
            box=ax.get_tightbbox(renderer).expanded(1.04,1.08).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(preview/(gid+'.png'),dpi=250,bbox_inches=box,pad_inches=.03)
    fig.savefig(preview/'full.png',dpi=250)
    displayed=[t.get_text()for t in fig.findobj(Text)if t.get_visible()and t.get_text()]
    residual=[t for t in displayed if re.search(r'\bResolution\b|\bRes\b(?!\.)',t)]
    assert not residual,residual
    dump(out/'REVISION_DISPLAY_AUDIT.json',dict(figure=fid,changes=changes,stage_replacement_count=len(changes),display_residuals=residual,
       data_objects_unchanged=True,axes=inventory,colorbars=colour,figure_legends=[[t.get_text()for t in l.get_texts()]for l in fig.legends],
       display_size_mm=fig.get_size_inches()*25.4,internal_Res_category_preserved=True))
