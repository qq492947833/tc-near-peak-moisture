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
 out=Path(out);initial=core(fig);changes=[];replacements=[]
 axes=[a for a in fig.axes if not hasattr(a,'_colorbar')];by={a.get_gid():a for a in axes}
 for ax in axes:
  for axis in (ax.xaxis,ax.yaxis):
   old=[t.get_text() for t in axis.get_ticklabels()]
   new=[re.sub(r'\bRes\b','Dec',re.sub(r'\bResolution\b','Decay',t)) for t in old]
   if new!=old:
    axis.set_ticklabels(new);replacements.extend(dict(old=x,new=y,role='tick') for x,y in zip(old,new) if x!=y)
 for t in fig.findobj(Text):
  old=t.get_text();new=re.sub(r'\bRes\b','Dec',re.sub(r'\bResolution\b','Decay',old))
  if old!=new:t.set_text(new);replacements.append(dict(old=old,new=new))
 if replacements:changes.append('Visible stage Decay/Dec, unchanged historical source IDs')
 bars=[a._colorbar for a in fig.axes if hasattr(a,'_colorbar')]
 groups=[]
 if fid=='Figure_S10':
  bounds=[4.,11.,5.,.22,12.,.11]
  names=['RH contrast (pp)','PI − V contrast (m s⁻¹)','PI contrast (m s⁻¹)','Signed entropy ratio (dimensionless)','Entropy-deficit contrast (J kg⁻¹ K⁻¹)','Signed ventilation (dimensionless)']
  for p,cb,hi,label in zip('abcdef',bars,bounds,names):groups.append(([p],[by[p].images[0]],cb,hi,label))
 if fid=='Figure_S11':groups=[(['f'],[by['f'].images[0]],bars[0],1.06,'Shear contrast (m s⁻¹)')]
 if fid=='Figure_S12':
  groups=[(['b'],[by['b'].images[0]],bars[0],.75,'Integrated contrast (g kg⁻¹)')]
  by['c'].patches[0].set_facecolor('#617F46');by['c'].patches[1].set_facecolor('#D5615A')
  changes.append('c: green Lateral input and coral Stock change are direct-labelled categories; neither colour encodes missingness or a continuous value')
 if fid=='Figure_S13':
  for keys,cb,hi,label in [('abcefg',bars[0],2.,'q700 anomaly contrast (g kg⁻¹)'),('ijkmno',bars[1],40.,'z200 anomaly contrast (m)')]:
   groups.append((list(keys),[by[p].collections[0] for p in keys],cb,hi,label))
 if fid=='Figure_S14':groups=[(list('abc'),[by[p].images[0] for p in 'abc'],bars[0],5.2,'Interval-integrated water-vapour mass (10¹² kg)')]
 semantics=[]
 for owner,maps,cb,hi,label in groups:
  vals=np.concatenate([np.ma.asarray(m.get_array(),dtype=float).compressed() for m in maps]);vals=vals[np.isfinite(vals)]
  assert np.max(np.abs(vals))<=hi+1e-12,(fid,label,vals.min(),vals.max(),hi)
  norm=TwoSlopeNorm(0,-hi,hi)
  for m in maps:m.set_norm(norm)
  cb.update_normal(maps[0]);ticks=np.linspace(-hi,hi,5);cb.set_ticks(ticks);cb.set_ticklabels([format_number(v) for v in ticks]);cb.ax.minorticks_off()
  cb.set_label(label,fontsize=7.0 if fid=='Figure_S10' else 7.4,labelpad=2)
  # Replace former unit-only side text by the complete label on the colour bar.
  if fid in ('Figure_S11','Figure_S12'):
   for t in list(cb.ax.texts):t.remove()
   cb.set_label('')
   side='Shear contrast\n(m s⁻¹)' if fid=='Figure_S11' else 'Integrated contrast\n(g kg⁻¹)'
   cb.ax.text(-.14,.5,side,transform=cb.ax.transAxes,ha='right',va='center',fontsize=7.4)
   if fid=='Figure_S11':by['f'].set_xlabel('Centre choice')
  semantics.append(dict(owner=owner,quantity_unit=label,data_range=[float(vals.min()),float(vals.max())],norm=type(norm).__name__,range=[-hi,hi],ticks=ticks.tolist(),cmap=maps[0].get_cmap().name,all_owners_same_norm=all(m.norm is norm for m in maps),all_owners_same_cmap=all(m.get_cmap().name==maps[0].get_cmap().name for m in maps),finite_endpoints=True,clipping=False,five_ticks_linear=True,zero_or_missing_handling='Original mask and NA preserved; zero is genuine zero'))
  changes.append('Shared signed 5-tick scale for '+','.join(owner)+': '+str(-hi)+' to '+str(hi))
 dump(out/'stage_display_replacements.json',dict(replacements=replacements,count=len(replacements),internal_IDs_unchanged=True))
 inventory(fig,out,fid,changes,semantics,initial)
 dump(out/'categorical_artist_inventory.json',[dict(panel=ax.get_gid(),line_colors=[dict(label=l.get_label(),color=str(l.get_color())) for l in ax.lines],patch_colors=[list(p.get_facecolor()) for p in ax.patches],local_legend=[] if ax.get_legend() is None else [t.get_text() for t in ax.get_legend().get_texts()],direct_xticklabels=[t.get_text() for t in ax.get_xticklabels()]) for ax in axes])
 for p,ax in by.items():
  if p and len(p)==1:
   bb=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.07)
   fig.savefig(out/f'panel_{p}_preview.png',dpi=180,bbox_inches=bb)
 return [cb.ax for cb in bars]
