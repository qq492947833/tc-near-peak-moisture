"""Accepted late color-key/geometry amendments, applied to native plot objects.

Geometries follow other_matrix_minimal_keys.py and the later compact(fig) repair.
Color limits are read from actual plotted/source values, never copied estimates.
"""
def apply(fig,figure_id,rows,out,m):
    import pandas as pd
    from display_format import format_number, nice_symmetric_limit
    by={a.get_gid():a for a in fig.axes if a.get_gid()};keys=[]
    if figure_id=='Figure_04':
        oldw,oldh=fig.get_size_inches();newh=oldh+10/25.4
        axes=[(a,a.get_position().frozen())for a in fig.axes]
        texts=[(t,t.get_position())for t in fig.texts]
        legs=[(l,l.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds)for l in fig.legends]
        fig.set_size_inches(oldw,newh,forward=False);ratio=oldh/newh;shift=(newh-oldh)/newh
        for a,p in axes:a.set_position([p.x0,p.y0*ratio+shift,p.width,p.height*ratio])
        for t,(x,y)in texts:t.set_position((x,y*ratio+shift))
        for l,(x,y,w,h)in legs:l.set_bbox_to_anchor((x,y*ratio+shift,w,h*ratio),transform=fig.transFigure)
        source=next(r for r in rows if r['id']=='g')
        saved=pd.read_csv(out/source['saved_selected_table'],float_precision='round_trip')
        values=saved[['original_raw','overlap_raw','standardized']].to_numpy(float)
        v=4.6;assert float(abs(values).max())<=v;norm=m.TwoSlopeNorm(0,-v,v)
        ca=fig.add_axes([.32,.050,.46,.0075],label='<closeout-colorbar>')
        cb=fig.colorbar(m.mpl.cm.ScalarMappable(norm=norm,cmap=m.CMAP),cax=ca,orientation='horizontal',ticks=[-v,0,v])
        cb.ax.set_xticklabels([format_number(-v),'0',format_number(v)]);label='g: Pre-event RH contrast (pp)'
    elif figure_id=='Figure_06':
        a=by['b'];mesh=a.collections[0];p=a.get_position();label=a.get_xlabel();a.set_xlabel('');v=float(mesh.norm.vmax)
        ca=fig.add_axes([p.x0,p.y0-.050,p.width,.010],label='<closeout-colorbar>')
        cb=fig.colorbar(mesh,cax=ca,orientation='horizontal',ticks=[-v,0,v]);cb.ax.set_xticklabels([format_number(-v),'0',format_number(v)])
    else:return []
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=7.5,length=2,pad=2);cb.set_label(label,fontsize=7.5,labelpad=2);keys.append(ca)
    if figure_id=='Figure_06':
        fig.canvas.draw();w,h=fig.get_size_inches()*25.4;upper=[by['a']]
        old=[[x*w,y*h,ww*w,hh*h]for x,y,ww,hh in [a.get_position().bounds for a in fig.axes]]
        gap=min(a.get_tightbbox(fig.canvas.get_renderer()).y0/fig.dpi*25.4 for a in upper)-max(a.get_tightbbox(fig.canvas.get_renderer()).y1/fig.dpi*25.4 for a in fig.axes if a not in upper)
        assert gap>12
        fig.set_size_inches(w/25.4,(h-10)/25.4,forward=False)
        for a,p in zip(fig.axes,old):
            a.set_axes_locator(None);shift=10 if a in upper else 0
            a.set_position([p[0]/w,(p[1]-shift)/(h-10),p[2]/w,p[3]/(h-10)])
    return keys
