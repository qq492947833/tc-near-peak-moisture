from pathlib import Path
import json,numpy as np
from matplotlib.colors import to_hex
from matplotlib.text import Text
def record_figure(fig,fid,out):
 out=Path(out);preview=out/'panel_previews';preview.mkdir(exist_ok=True);fig.canvas.draw();renderer=fig.canvas.get_renderer();axes=[]
 def legend(l):
  if l is None:return None
  return {'text':[t.get_text()for t in l.get_texts()],'bbox_pixels':list(l.get_window_extent(renderer).bounds),'font_sizes':[t.get_fontsize()for t in l.get_texts()]}
 for i,ax in enumerate(fig.axes):
  gid=ax.get_gid()or str(i)
  axes.append({'panel':gid,'title':ax.get_title(loc='left')or ax.get_title(),'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),'position':list(ax.get_position().bounds),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'xscale':ax.get_xscale(),'yscale':ax.get_yscale(),'xticks':ax.get_xticks().tolist(),'yticks':ax.get_yticks().tolist(),'xticklabels':[t.get_text()for t in ax.get_xticklabels()],'yticklabels':[t.get_text()for t in ax.get_yticklabels()],'legend':legend(ax.get_legend()),'line_colors':[{'label':l.get_label(),'color':str(l.get_color()),'linestyle':l.get_linestyle()}for l in ax.lines],'patch_colors':list(dict.fromkeys(to_hex(p.get_facecolor())for p in ax.patches)),'colormaps':[{'name':im.get_cmap().name,'vmin':im.norm.vmin,'vmax':im.norm.vmax}for im in ax.images],'texts':[t.get_text()for t in ax.texts]})
  if ax.get_label()!='<colorbar>':
   box=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.03,1.08)
   fig.savefig(preview/('panel_'+str(gid)+'.png'),dpi=250,bbox_inches=box)
 for info,ax in zip(axes,fig.axes):
  info['axison']=ax.axison
  info['actual_tick_visibility']={}
  for name,axis,limits in [('x',ax.xaxis,ax.get_xlim()),('y',ax.yaxis,ax.get_ylim())]:
   lo,hi=sorted(limits);rows=[]
   for tick in axis.get_major_ticks():
    value=float(tick.get_loc())
    if not lo-1e-10<=value<=hi+1e-10:continue
    rows.append({'value':value,'mark1':bool(ax.axison and axis.get_visible() and tick.tick1line.get_visible() and tick.tick1line.get_markersize()>0),'mark2':bool(ax.axison and axis.get_visible() and tick.tick2line.get_visible() and tick.tick2line.get_markersize()>0),'label1_visible':bool(ax.axison and axis.get_visible() and tick.label1.get_visible() and tick.label1.get_text()),'label2_visible':bool(ax.axison and axis.get_visible() and tick.label2.get_visible() and tick.label2.get_text()),'label1_text':tick.label1.get_text(),'label2_text':tick.label2.get_text()})
   info['actual_tick_visibility'][name]=rows
 fig.savefig(out/'review_full.png',dpi=220)
 (out/'legend_color_tick_inventory.json').write_text(json.dumps({'figure':fid,'axes':axes,'figure_legends':[legend(l)for l in fig.legends],'figure_text':[t.get_text()for t in fig.texts]},ensure_ascii=False,indent=2,default=str),encoding='utf8')
