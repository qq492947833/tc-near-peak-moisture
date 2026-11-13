"""Display-only reader labels; frozen estimates and uncertainty remain untouched."""
from pathlib import Path
import json
import numpy as np

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def science(fig):
    return clean([dict(panel=a.get_gid(),xlim=a.get_xlim(),ylim=a.get_ylim(),
        xticks=a.get_xticks(),yticks=a.get_yticks(),
        lines=[dict(x=l.get_xdata(),y=l.get_ydata())for l in a.lines],
        collections=[dict(array=c.get_array(),offsets=c.get_offsets(),paths=[p.vertices for p in c.get_paths()],
                          vmin=c.norm.vmin,vmax=c.norm.vmax,cmap=c.cmap.name)for c in a.collections],
        patches=[dict(vertices=p.get_path().vertices,matrix=p.get_patch_transform().get_matrix(),
                      face=p.get_facecolor(),edge=p.get_edgecolor())for p in a.patches])
        for a in fig.axes])

def apply(fig,out):
    out=Path(out);fig.canvas.draw();before=science(fig)
    oldw,oldh=fig.get_size_inches()*25.4
    original=[dict(panel=a.get_gid(),rect_mm=np.asarray(a.get_position().bounds)*[oldw,oldh,oldw,oldh])
              for a in fig.axes]
    # Add only lower annotation room. All a-f physical sizes and top coordinates stay fixed.
    extra=8.0;newh=oldh+extra
    axes=[(a,np.asarray(a.get_position().bounds)*[oldw,oldh,oldw,oldh])for a in fig.axes]
    text=[(t,t.get_position())for t in fig.texts]
    legends=[(l,l.get_bbox_to_anchor().transformed(fig.transFigure.inverted()).bounds)for l in fig.legends]
    fig.set_size_inches(oldw/25.4,newh/25.4,forward=False)
    for a,(x,y,w,h) in axes:
        a.set_axes_locator(None);dy=extra
        if a.get_gid()=='g':dy-=2
        if getattr(a,'_colorbar',None)is not None:dy-=8
        a.set_position([x/oldw,(y+dy)/newh,w/oldw,h/newh])
    for t,(x,y)in text:t.set_position((x,(y*oldh+extra)/newh))
    for l,(x,y,w,h)in legends:l.set_bbox_to_anchor((x,(y*oldh+extra)/newh,w,h*oldh/newh),transform=fig.transFigure)
    by={a.get_gid():a for a in fig.axes if a.get_gid()}
    by['a'].set_title('Mid-level relative humidity',loc='left',pad=13)
    g=by['g']
    g.set_title('Mid-level humidity difference before weakening',loc='left',pad=13+2*72/25.4)
    for t in g.texts:
        if t.get_text()=='g' and hasattr(t,'xyann'):
            x,y=t.xyann;t.xyann=(x,y+2*72/25.4)
    g.set_ylabel('')
    g.set_xticks([0,1,2,3])
    g.set_xticklabels(['Unadjusted','Unadjusted','Adjusted for location\nand season',
                       'Adjusted for\nstorm characteristics'])
    # Bracket annotations express sample groups without adding data artists.
    for x0,x1,label in [(-.47,.47,'All event pairs (117)'),(.53,3.47,'Geographic overlap subset (105)')]:
        for xx,yy in [([x0,x1],[.965,.965]),([x0,x0],[.965,1.035]),([x1,x1],[.965,1.035])]:
            g.annotate('',xy=(xx[1],yy[1]),xytext=(xx[0],yy[0]),xycoords=g.get_xaxis_transform(),
                       arrowprops=dict(arrowstyle='-',color='#677B86',lw=.65,shrinkA=0,shrinkB=0),
                       annotation_clip=False)
        g.text((x0+x1)/2,1.075,label,transform=g.get_xaxis_transform(),
               ha='center',va='bottom',fontsize=8,color='#233A46',clip_on=False)
    for t in g.texts:
        if t.get_text()=='3':t.set_text('3.00')
    for a in fig.axes:
        cb=getattr(a,'_colorbar',None)
        if cb is not None:cb.set_label('Relative humidity difference (percentage points)',fontsize=7.5,labelpad=2)
    fig.canvas.draw();after=science(fig)
    assert before==after,'Reader-label revision altered frozen numeric artists, limits, ticks or colour mapping'
    current=[dict(panel=a.get_gid(),rect_mm=np.asarray(a.get_position().bounds)*[oldw,newh,oldw,newh])
             for a in fig.axes]
    report=dict(status='PASS_FROZEN_SCIENTIFIC_ARTISTS',before_size_mm=[oldw,oldh],after_size_mm=[oldw,newh],
        before_geometry=original,after_geometry=current,scientific_artists_exact=True,
        permitted_geometry='8 mm extra lower canvas; a-f unchanged in physical size and distance from top; g tiles 2 mm lower; colorbar 8 mm lower',
        groups=[dict(label='All event pairs (117)',tile_indices=[0]),
                dict(label='Geographic overlap subset (105)',tile_indices=[1,2,3])],
        estimates_display=['3.85','4.23','3.00','1.72'],
        ci_statement='Frozen CIs preserved in input and selected tables; g tiles encode estimates, not significance. Last adjusted CI crosses zero.',
        variable='Equal-weight 700/500 hPa relative humidity',
        contrast='Before temporary departure minus before start of decay',
        window='-24 to -6 h with existing common-observation rules',
        weighting='Location/season and storm-state sample weighting, not climatology subtraction')
    (out/'SEMANTIC_REVISION_QA.json').write_text(json.dumps(clean(report),ensure_ascii=False,indent=2),encoding='utf8')
    preview=out/'semantic_previews';preview.mkdir(exist_ok=True)
    for ax in fig.axes:
        if not ax.get_gid():continue
        box=ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.03,1.06).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(preview/(ax.get_gid()+'.png'),dpi=250,bbox_inches=box,pad_inches=.03)
    fig.savefig(preview/'full.png',dpi=250)
