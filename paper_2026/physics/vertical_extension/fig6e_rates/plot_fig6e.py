"""Draw the standalone Fig. 6e rate panel from a verified nine-time summary."""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STYLE = [
    ('Gamma_K_h','Horizontal','#167B9E'),
    ('Gamma_K_p','Vertical','#428C79'),
    ('Gamma_K_total','Total','#233A46'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(source, qa_file, output, dpi=900):
    stats = json.loads(qa_file.read_text(encoding='utf-8'))
    if stats['status']!='PASS' or sha(source)!=stats['plot_data_sha256']:
        raise ValueError('Rate table differs from the verified source')
    if dpi<600:
        raise ValueError('The final panel must have at least 600 dpi at its nominal size')
    tab=pd.read_csv(source,float_precision='round_trip').query("support == 'nine_time_common'")
    assert len(tab)==27 and tab.n_sid.nunique()==tab.n_events.nunique()==1
    assert tab.n_sid.iloc[0]==stats['n_fixed_sid'] and tab.n_events.iloc[0]==stats['n_fixed_events']
    assert set(tab.unit)=={'W kg-1'}
    assert set(tab.region)=={'0-300 km'} and set(tab.layer)=={'500-850 hPa'} and set(tab.window)=={'2-16 d'}
    assert set(tab.n_draws)=={2000} and set(tab.metric)=={x[0] for x in STYLE}
    assert np.isfinite(tab[['estimate','ci_lower','ci_upper']]).all().all()
    output = Path(output).resolve()
    output.mkdir(exist_ok=True, parents=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    from PIL import Image, ImageChops
    output.mkdir(exist_ok=True,parents=True)
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
                         'axes.titlesize':9.2,'axes.labelsize':8.5,'axes.linewidth':0.65,
                         'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
                         'axes.spines.right':False,'axes.spines.top':False,'savefig.facecolor':'white'})
    # Width of the native Fig. 6 left-column panel. The legacy six panels were rendered at 900 dpi.
    size_inches=(3144/900,1927/900)
    fig=plt.figure(figsize=size_inches,dpi=dpi)
    ax=fig.add_axes([0.232,0.214,0.737,0.573])
    fig.text(0.060,0.939,'e',fontweight='bold',fontsize=10.5,va='center')
    fig.text(0.232,0.939,'KE cross-scale transfer',fontsize=9.2,va='center')
    ax.axvline(0, color='#9EAAAF',ls=(0,(3,3)),lw=0.75,zorder=0)
    ax.axhline(0, color='#A9B3B7',lw=0.65,zorder=0)
    for index,(metric,label,color) in enumerate(STYLE):
        sub=tab.loc[tab.metric==metric].sort_values('relative_hour')
        x=sub.relative_hour.to_numpy(float)
        assert np.array_equal(x,np.arange(-24,25,6))
        y,lo,hi=sub[['estimate','ci_lower','ci_upper']].to_numpy(float).T
        assert (lo<=hi).all()
        ax.fill_between(x,lo,hi,color=color,alpha=0.16,lw=0,zorder=1+0.1*index)
        ax.plot(x,y,color=color,marker='o',ms=3,lw=(1.6 if index==2 else 1.5),
                label=label,zorder=3+0.1*index)
    magnitude=np.max(np.abs(tab[['ci_lower','ci_upper']].to_numpy(float)))
    exponent=int(np.floor(np.log10(magnitude))) if magnitude else 0
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    def tick(v, _):
        s=f'{v/(10.0**exponent):g}'
        return s.replace('-','−')
    ax.yaxis.set_major_formatter(FuncFormatter(tick))
    scale=r'$10^{'+str(exponent)+r'}$ ' if exponent else ''
    ax.set_ylabel('KE canonical transfer rate\n('+scale+r'W kg$^{-1}$)')
    ax.set_xlim(-25,25)
    ax.set_xticks([-24,-12,0,12,24])
    ax.set_xlabel('Time from trough (h)')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),borderaxespad=0,borderpad=0,
              frameon=False,ncol=3,handlelength=1.1,handletextpad=0.35,columnspacing=0.8)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    frame=fig.bbox
    low,high=ax.get_ylim()
    ytexts=[tick.label1 for tick in ax.yaxis.get_major_ticks() if low<=tick.get_loc()<=high]
    objects=[ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ytexts,ax.legend_,*fig.texts]
    for obj in objects:
        box=obj.get_window_extent(renderer)
        assert box.x0>=-1 and box.x1<=frame.width+1 and box.y0>=-1 and box.y1<=frame.height+1, (obj,box)
    png=output/'Figure_06e_KE_transfer_rate.png'
    tiff=png.with_suffix('.tiff')
    fig.savefig(png,dpi=dpi)
    
    plt.close(fig)
    with Image.open(png) as im:
        if 'A' in im.getbands():
            assert im.getchannel('A').getextrema()==(255,255)
        rgb=im.convert('RGB')
    rgb.save(png,dpi=(dpi,dpi))
    rgb.save(tiff,dpi=(dpi,dpi),compression='tiff_lzw')
    with Image.open(png) as a, Image.open(tiff) as b:
        assert a.mode==b.mode=='RGB' and a.size==b.size
        assert ImageChops.difference(a,b).getbbox() is None
        w,h=a.size
    receipt=dict(status='RENDERED',unit='W kg-1',display_power10=exponent,dpi=dpi,width_px=w,height_px=h,
                 nominal_width_inches=size_inches[0],nominal_height_inches=size_inches[1],
                 dpi_at_nominal_width=w/size_inches[0],n_sid=int(tab.n_sid.iloc[0]),n_events=int(tab.n_events.iloc[0]),
                 source_sha256=sha(source),png_sha256=sha(png),tiff_sha256=sha(tiff),
                 equal_decoded_pixels=True)
    (output/'FIGURE_06e_FILE_QA.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return png,tiff


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/FIG6E_RATE_ESTIMATES.csv')
    p.add_argument('--qa',type=Path,default=ROOT/'data/CURVE_STATISTICS_QA.json')
    p.add_argument('--out',type=Path,default=Path('fig6e_figure'))
    p.add_argument('--dpi',type=int,default=900)
    args=p.parse_args()
    print(*draw(args.source,args.qa,args.out,args.dpi),sep='\n')

