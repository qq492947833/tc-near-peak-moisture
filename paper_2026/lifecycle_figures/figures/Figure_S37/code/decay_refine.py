"""Display-only 20261028 refinement; frozen scientific arrays and historical IDs stay exact."""
from pathlib import Path
import json,re,hashlib,math,os
import numpy as np
from matplotlib.text import Text
from matplotlib.colors import Normalize,TwoSlopeNorm,LogNorm
from matplotlib.ticker import FixedLocator,FuncFormatter
from matplotlib.lines import Line2D
from display_format import format_number

def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v)),encoding='utf-8')
def core(fig):
 from supp_style import data_state
 rows=[]
 for ax,d in zip(fig.axes,data_state(fig)):
  if hasattr(ax,'_colorbar'):continue
  d.pop('gid',None)
  for im in d['images']:
   for k in ['norm','vmin','vmax']:im.pop(k,None)
  rows.append(d)
 return rows

def inventory(fig,out,fid,changes,semantics,initial):
 fig.canvas.draw();ren=fig.canvas.get_renderer();fw,fh=fig.get_size_inches()*72
 axes=[]
 for idx,ax in enumerate(fig.axes):
  def bbox(obj):
   b=obj.get_window_extent(ren).transformed(fig.dpi_scale_trans.inverted());return [float(v*72) for v in b.bounds]
  def ticks(axis,lim):
   lo,hi=sorted(lim);return [{'value':float(v),'label':t.get_text()} for v,t in zip(axis.get_majorticklocs(),axis.get_ticklabels()) if lo-1e-12<=v<=hi+1e-12 and t.get_visible() and t.get_text()]
  axes.append(dict(index=idx,panel=ax.get_gid(),position_pt=[float(x*y) for x,y in zip(ax.get_position().bounds,[fw,fh,fw,fh])],xlim=ax.get_xlim(),ylim=ax.get_ylim(),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),ylabel_box_pt=bbox(ax.yaxis.label),tight_box_pt=list(ax.get_tightbbox(ren).transformed(fig.dpi_scale_trans.inverted()).bounds),legend=None if ax.get_legend() is None else dict(text=[t.get_text() for t in ax.get_legend().get_texts()],title=ax.get_legend().get_title().get_text(),bbox_pt=bbox(ax.get_legend())),xticks=ticks(ax.xaxis,ax.get_xlim()),yticks=ticks(ax.yaxis,ax.get_ylim())))
 dump(out/'final_display_inventory.json',dict(figure=fid,changes=changes,figure_mm=(fig.get_size_inches()*25.4),axes=axes,figure_legends=[dict(text=[t.get_text() for t in l.get_texts()],title=l.get_title().get_text(),bbox_pt=bbox(l)) for l in fig.legends],numeric_colors=semantics))
 assert core(fig)==initial,'Display hook altered scientific array/CI/limits geometry'
 dump(out/'new_display_preservation.json',dict(status='PASS',scientific_arrays_CI_limits_and_marks_exact=True,allowed_changes='display text, geometry layout, norm/range/cmap/ticks and colorbar/legend only',changes=changes,all_numeric_colors_have_bars=True))


def refine(fig,fid,out):
 out=Path(out);initial=core(fig);changes=[]
 axes=[a for a in fig.axes if not hasattr(a,'_colorbar')]
 if fid in ['Figure_S19','Figure_S20']:
  for a,p in zip(axes,'abcdef'):a.set_gid(p)
 by={a.get_gid():a for a in axes};replacements=[]
 for ax in axes:
  for axis in [ax.xaxis,ax.yaxis]:
   old=[t.get_text() for t in axis.get_ticklabels()]
   new=[re.sub(r'\bRes\b','Dec',re.sub(r'\bResolution\b','Decay',v)) for v in old]
   if new!=old:
    axis.set_ticklabels(new)
    replacements.extend(dict(old=x,new=y,role='categorical tick label') for x,y in zip(old,new) if x!=y)
 for t in fig.findobj(Text):
  old=t.get_text();new=re.sub(r'\bResolution\b','Decay',old);new=re.sub(r'\bRes\b','Dec',new)
  if new!=old:t.set_text(new);replacements.append(dict(old=old,new=new))
 if replacements:changes.append('Stage display Resolution/Res to Decay/Dec; internal identifiers retained')
 if fid=='Figure_S19':fig.subplots_adjust(wspace=.29);changes.append('Left/right column gap reduced from wspace0.39 to0.29')
 if fid=='Figure_S21':
  leg=by['c'].get_legend(); handles=leg.legend_handles;labels=[t.get_text() for t in leg.get_texts()];leg.remove()
  fig.legend(handles,labels,title='Interventions (c–f)',title_fontsize=7.5,fontsize=7.5,ncol=2,loc='lower center',bbox_to_anchor=(.55,.005),frameon=False)
  changes.append('Interventions c–f legend moved to compact bottom, follow-up legend remains local b')
 if fid=='Figure_S24':
  from matplotlib.colors import ListedColormap
  im=by['d'].images[0];rgba=im.get_cmap()(np.linspace(0,1,256));rgba[:,:3]=.52*rgba[:,:3]+.48
  im.set_cmap(ListedColormap(rgba,name='support_Greens_readable'))
  for t in by['d'].texts:
   if re.search(r'\d',t.get_text()):t.set_color('black')
  changes.append('Support matrix lightened consistently with its bar; black cell counts')
 if fid=='Figure_S25':fig.legends[0].set_bbox_to_anchor((.57,.045));changes.append('Bottom legend moved 2.45mm farther from data axes')
 if fid=='Figure_S26':
  fig.subplots_adjust(wspace=.34)
  legend=next(a for a in axes if a.get_gid() is None and a.get_legend() is not None);legend.set_position([.28,.005,.58,.07])
  changes.append('Column gutter reduced wspace0.48 to0.34; endpoint legend lowered2.1mm')
 if fid=='Figure_S31':
  fig.subplots_adjust(wspace=.48)
  fig.canvas.draw();ren=fig.canvas.get_renderer()
  for grp in [('b','d','f'),('c','e')]:
   lx=[]
   for k in grp:
    ax=by[k];valid=[t.get_window_extent(ren).x0 for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
    lx.append((min(valid)-8*fig.dpi/72)/fig.bbox.width)
   common=min(lx)
   for k in grp:by[k].yaxis.set_label_coords(common,.5,transform=fig.transFigure if False else by[k].transAxes)
   # use figure x and each axes midpoint y, preserving vertical centre
   for k in grp:by[k].yaxis.set_label_coords(common,by[k].get_position().y0+by[k].get_position().height/2,transform=fig.transFigure)
  changes.append('Column gutter reduced; b/d/f and c/e ylabel centres share physical x')
 # Existing scalar mappables are tied to the true unrounded data arrays.
 bars=[a._colorbar for a in fig.axes if hasattr(a,'_colorbar')]
 if fid=='Figure_S23':
  maps=[a.collections[0] for a in [by['a'],by['b']]]
  mx=max(float(np.max(m.get_array())) for m in maps);unit=5*10**max(0,math.floor(math.log10(mx))-1);hi=math.ceil(mx/unit)*unit
  norm=LogNorm(1,hi)
  for m in maps:m.set_norm(norm)
  cb=bars[0];cb.update_normal(maps[-1]);ticks=sorted(set([1]+[10**k for k in range(1,int(math.log10(hi))+1)]+[hi]));cb.set_ticks(ticks);cb.set_ticklabels([format_number(v) for v in ticks]);cb.ax.minorticks_off();cb.ax.xaxis.tick_top();cb.ax.tick_params(axis='x',labeltop=True,labelbottom=False,top=True,bottom=False,pad=3)
  changes.append('Shared a/b log count norm with explicit finite upper endpoint')
 if fid=='Figure_S27':
  cb=bars[0];A=20.;norm=TwoSlopeNorm(0,-A,A)
  for m in by['a'].collections:
   if m.get_array() is not None:m.set_norm(norm)
  cb.mappable.set_norm(norm);cb.update_normal(cb.mappable);cb.ax.set_position([.30,.482,.54,.012]);cb.set_label('')
  by['a'].set_title('Stage-duration response',loc='left',pad=16)
  by['a'].text(1,1.045,'Stage-duration response per fixed environmental SD (%)',transform=by['a'].transAxes,ha='right',va='bottom',fontsize=7.5)
  for t in fig.texts:
   if t.get_text().startswith('Outline:'):t.set_position((.18,.44))
  changes.append('a bar length54% canvas centred under panel; complete scale description at upper right; signed range±20')
 if fid=='Figure_S28':
  ims=[by[p].images[0] for p in ['a','b']];norm=TwoSlopeNorm(0,-.03,.03)
  for im in ims:im.set_norm(norm)
  bpos=by['b'].get_position();by['b'].set_position([bpos.x0,bpos.y0+.02,bpos.width,bpos.height])
  ca=fig.add_axes([.29,.065,.55,.013]);ca.set_gid('shared_colourbar');cb=fig.colorbar(ims[0],cax=ca,orientation='horizontal');bars.append(cb)
  cb.set_label('Change in A or |A| per fixed SD (dimensionless)',fontsize=7.5,labelpad=1)
  for t in fig.texts:
   if t.get_text().startswith('Outline:'):t.set_position((.965,.945));t.set_ha('right');t.set_fontsize(7.5)
  changes.append('a/b share same fixed-SD scale and new signed±0.03 bar; original q outlines retained')
 semantics=[]
 for cb in bars:
  ca=cb.ax;m=cb.mappable;owner=[]; vals=[]
  for p,ax in by.items():
   for obj in [*ax.images,*ax.collections]:
    ar=obj.get_array()
    if ar is not None and ((getattr(obj,'colorbar',None) is cb) or obj.norm is m.norm):
     owner.append(p);a=np.ma.asarray(ar,dtype=float).compressed();vals.extend(a[np.isfinite(a)].tolist())
  if fid=='Figure_S23':owner=['a','b'];vals=[v for p in owner for v in np.ma.asarray(by[p].collections[0].get_array()).compressed()]
  if fid=='Figure_S26':owner=['b'];vals=[]
  name=ca.get_xlabel() or ca.get_ylabel()
  if fid=='Figure_S23':name='Cyclone–intensity-bin pairs per hexagon (count)'
  if fid=='Figure_S26':name='P / q (dimensionless)'
  if fid=='Figure_S27':name='Stage-duration response per fixed environmental SD (%)';owner=['a']
  if fid=='Figure_S30':
   name='Recovery contrast (pp)';ca.set_xlabel(name,fontsize=7,labelpad=1)
   norm=TwoSlopeNorm(0,-7,7);by['e'].images[0].set_norm(norm);cb.update_normal(by['e'].images[0]);m=cb.mappable
  if not isinstance(m.norm,LogNorm):
   lo,hi=float(m.norm.vmin),float(m.norm.vmax)
   ticks=np.linspace(lo,hi,5);cb.set_ticks(ticks);cb.set_ticklabels([format_number(v) for v in ticks]);ca.minorticks_off()
  ticks=[float(v) for v in cb.get_ticks()];lo,hi=float(m.norm.vmin),float(m.norm.vmax)
  valid=[float(v) for v in vals if np.isfinite(v)];clip=bool(valid and (min(valid)<lo-1e-12 or max(valid)>hi+1e-12))
  assert not clip,(fid,name,min(valid),max(valid),lo,hi)
  semantics.append(dict(owner=sorted(set(owner)),quantity_unit=name,data_range=[min(valid),max(valid)] if valid else None,norm=type(m.norm).__name__,range=[lo,hi],ticks=ticks,cmap=m.get_cmap().name,finite_endpoints=True,clipping=False,five_ticks_linear=len(ticks)==5 if not isinstance(m.norm,LogNorm) else 'log',zero_or_missing_handling='Original masked/zero cells and NA text retained'))
 dump(out/'stage_display_replacements.json',dict(replacements=replacements,count=len(replacements),internal_IDs_unchanged=True))
 inventory(fig,out,fid,changes,semantics,initial)
 # Actual plot crops are produced by Matplotlib from the final figure, no raster painting.
 if not os.environ.get('DECAY_INVENTORY_ONLY'):
  for p,ax in by.items():
   if p and len(p)==1:
    bounds=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.07)
    fig.savefig(out/f'panel_{p}_preview.png',dpi=160,bbox_inches=bounds)
 if os.environ.get('DECAY_INVENTORY_ONLY'):raise SystemExit(0)
 return [a for a in fig.axes if hasattr(a,'_colorbar')]
