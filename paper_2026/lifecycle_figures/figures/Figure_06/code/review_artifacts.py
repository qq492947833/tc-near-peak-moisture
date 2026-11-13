"""Native-object geometry and inventory for this display-only revision."""
from pathlib import Path
import json, numpy as np
from matplotlib.text import Text
from matplotlib.transforms import Bbox

def capture(fig,fid,out):
    out=Path(out);fig.canvas.draw();renderer=fig.canvas.get_renderer()
    if fid=='Figure_08':
        # The common x coordinate follows the widest final formatted tick field
        # in each figure column; no data coordinate changes.
        for index in (0,2):
            group=[fig.axes[j]for j in [index,index+1,index+4,index+5,index+8,index+9]]
            xpos=min(a.yaxis.label.get_window_extent(renderer).x0 for a in group)/fig.bbox.width
            for a in group:a.yaxis.set_label_coords(xpos,0.5,transform=fig.transFigure if False else a.transAxes)
            for a in group:
                p=a.get_position()
                a.yaxis.set_label_coords((xpos-p.x0)/p.width,0.5)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
    data=[]
    panels=out/'panel_previews';panels.mkdir(exist_ok=True)
    for i,ax in enumerate(fig.axes):
        gid=ax.get_gid()or f'axis_{i}'
        bbox=ax.get_position()
        record=dict(id=gid,label=ax.get_label(),bounds_fraction=list(bbox.bounds),bounds_pt=[float(x)for x in np.array(bbox.bounds)*np.array([fig.get_figwidth(),fig.get_figheight()]*2)*72],
          xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),xscale=ax.get_xscale(),yscale=ax.get_yscale(),
          xticks=[float(x)for x in ax.get_xticks()],yticks=[float(x)for x in ax.get_yticks()],
          xticklabels=[x.get_text()for x in ax.get_xticklabels()],yticklabels=[x.get_text()for x in ax.get_yticklabels()],
          xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),
          ylabel_bbox_px=list(ax.yaxis.label.get_window_extent(renderer).bounds),
          lines=[dict(label=l.get_label(),color=l.get_color(),ls=l.get_linestyle(),marker=l.get_marker(),ms=l.get_markersize(),x=np.asarray(l.get_xdata()).tolist(),y=np.asarray(l.get_ydata()).tolist())for l in ax.lines],
          image_colormaps=[dict(cmap=im.cmap.name,vmin=im.norm.vmin,vmax=im.norm.vmax)for im in ax.images],
          collection_colors=[dict(class_=type(c).__name__,cmap=c.cmap.name,vmin=c.norm.vmin,vmax=c.norm.vmax,colors=c.get_facecolor().tolist())for c in ax.collections],
          legends=[]if ax.get_legend()is None else [x.get_text()for x in ax.get_legend().get_texts()])
        data.append(record)
        if ax.get_label() not in ('<colorbar>','<closeout-colorbar>'):
            box=ax.get_tightbbox(renderer).expanded(1.04,1.07).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(panels/(gid+'.png'),dpi=250,bbox_inches=box,pad_inches=0.03)
    top=dict(figure=fid,axes=data,figure_legends=[dict(text=[t.get_text()for t in l.get_texts()],bbox_px=list(l.get_window_extent(renderer).bounds))for l in fig.legends])
    (out/'tick_color_legend_inventory.json').write_text(json.dumps(top,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    return top
