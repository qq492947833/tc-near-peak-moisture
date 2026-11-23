"""Render the final g panel from frozen estimates and retain the native r04 panels a–f."""
from pathlib import Path
from io import BytesIO
import csv, hashlib, json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

INPUTS=Path(__file__).resolve().parent/'inputs'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def box(artist, f):
    return (np.asarray(artist.get_window_extent(f.canvas.get_renderer()).extents)*25.4/f.dpi).tolist()

def render(number,args,fig,axis,panel_title,zero,BLUE,INK):
    dpi=900
    out=args.out
    name=f'Figure_{number:02d}'
    H=142 if number==2 else 147
    left,right=15,174 if number==2 else 177
    f=fig(H)
    g=axis(f,left,105.7 if number==2 else 108.5,right-left,25.8 if number==2 else 28.5)
    title='Pre-departure RH contrast' if number==2 else 'Wind-matched dry APE'
    panel_title(g,'g',title)
    zero(g)
    g.tick_params(axis='y',pad=1.4)
    if number==2:
        data=args.repo/'paper_2026/lifecycle_figures/figures/Figure_03/inputs/direct/environmental_contrasts.csv'
        submit={r['model']:r for r in read(data) if r['coverage']=='coverage_117' and r['variable']=='midRH'}
        labels=['All pairs','Common support','Geo-season','+Storm state']
        entries=[('geographical','original_raw','original_lo','original_hi','original_pairs'),
                 ('geographical','overlap_raw','overlap_raw_lo','overlap_raw_hi','overlap_pairs'),
                 ('geographical','standardized','lo','hi','overlap_pairs'),
                 ('geographical_state','standardized','lo','hi','overlap_pairs')]
        rows=[]
        for label,(model,*columns) in zip(labels,entries):
            r=submit[model]
            row=dict(zip(['estimate','lower','upper','n_pairs'],[r[c] for c in columns]))
            rows.append(dict(display=label,**row))
        assert [int(r['n_pairs']) for r in rows]==[117,105,105,105]
        centers=np.arange(4)
        bounds=np.asarray([[float(r[k]) for k in ('estimate','lower','upper')] for r in rows])
        heights,low,high=bounds.T
        g.bar(centers,heights,width=.63,color=BLUE,alpha=.9,zorder=2)
        g.errorbar(centers,heights,yerr=[heights-low,high-heights],fmt='none',ecolor=INK,capsize=3,lw=1.05,zorder=3)
        g.set(xticks=centers,xticklabels=labels,xlim=(-.55,3.55),ylim=(-2,6.5),yticks=[-2,0,2,4,6],ylabel='RH contrast\n(percentage points)')
        assert low[-1]<0<high[-1]
        legend=None
    else:
        data=INPUTS/'wind_matched_dry_APE.csv'
        ids={f'ENERGY_ADJ:{i:05d}' for i in range(101,110)}
        rows=[r for r in read(data) if r['record_id'] in ids]
        assert len(rows)==9 and {r['record_id'] for r in rows}==ids
        assert all(r['sample']=='all_events' and r['estimand']=='matched_rise_minus_Resolution' and r['region']=='r0_300' and r['n_sid']=='108' and r['n_events']=='115' and r['metric']=='APE' and r['scheme']=='main' and r['support_mode']=='native_diagnostic' and r['units']=='J kg-1' and r['source_table']=='B/results/REUSED_BASELINE.csv' for r in rows)
        assert {r['level'] for r in rows}=={'500','700','850'} and {r['window'] for r in rows}=={'0','1','2'}
        table={ (int(r['level']),int(r['window'])):r for r in rows}
        assert len(table)==9 and table[700,0]['estimate']=='3.898006021438687'
        centers=np.arange(3)
        handles=[]
        for j,(level,col) in enumerate([(500,'#287FA0'),(700,'#63A58C'),(850,'#B17E59')]):
            values=np.asarray([[float(table[level,w][k]) for k in ('estimate','lower','upper')] for w in range(3)])
            heights,low,high=values.T
            x=centers + (j-1)*.225
            handles.append(g.bar(x,heights,width=.195,color=col,zorder=2,label=f'{level} hPa'))
            g.errorbar(x,heights,yerr=[heights-low,high-heights],fmt='none',ecolor=INK,capsize=2.7,lw=1.05,zorder=3)
        g.set(xticks=centers,xticklabels=['>16 d','2–16 d','<2 d'],xlim=(-.55,2.55),ylim=(-8,10),yticks=[-8,-4,0,4,8],ylabel='Reintensification − decay\n(J kg$^{-1}$)')
        legend=g.legend(handles=handles,loc='lower right',bbox_to_anchor=(1,1.025),ncol=3,borderaxespad=0,handlelength=1.0,handletextpad=.4,columnspacing=1.05,borderpad=.05)
        bounds=np.asarray([[float(r[k]) for k in ('estimate','lower','upper')] for r in rows])
    lower,upper=g.get_ylim()
    assert (bounds[:,1]<=bounds[:,0]).all() and (bounds[:,2]>=bounds[:,0]).all()
    assert min(bounds[:,1])>lower and max(bounds[:,2])<upper
    g.yaxis.set_label_coords(-.059,.5)
    f.canvas.draw()
    ylabel=g.yaxis.label
    b=ylabel.get_window_extent(f.canvas.get_renderer())
    margin=1.4*f.dpi/25.4
    if b.x0<margin:
        point=ylabel.get_transform().transform(ylabel.get_position())
        point[0]+=margin-b.x0
        g.yaxis.set_label_coords(*f.transFigure.inverted().transform(point),transform=f.transFigure)
    f.canvas.draw()
    geometry=box(g,f)
    assert abs(geometry[0]-left)<.01 and abs(geometry[2]-right)<.01
    legend_bbox=box(legend,f) if legend else None
    title_bbox=box(g._left_title,f)
    if legend:
        assert title_bbox[2]+2<legend_bbox[0] and legend_bbox[1]>geometry[3]
    for label in list(g.get_xticklabels())+list(g.get_yticklabels())+[g.yaxis.label,g._left_title]:
        b=box(label,f)
        assert b[0]>=0 and b[1]>=0 and b[2]<=180 and b[3]<=H,(name,b)
    stream=BytesIO()
    f.savefig(stream,format='png',dpi=dpi)
    plt.close(f)
    stream.seek(0)
    tile=INPUTS/(name+'_af.png')
    info=json.loads((INPUTS/'upper_panels_provenance.json').read_text(encoding='utf-8'))[name]
    assert hashlib.sha256(tile.read_bytes()).hexdigest()==info['crop_png_sha256']
    with Image.open(tile) as frozen, Image.open(stream) as layer:
        merged=layer.convert('RGB')
        assert list(frozen.size)==info['upper_panel_crop_xyxy'][2:] and merged.width==frozen.width==6377
        cut=frozen.height
        assert merged.crop((0,cut-3,merged.width,cut+3)).getextrema()==((255,255),)*3
        assert frozen.crop((0,cut-3,frozen.width,cut)).getextrema()==((255,255),)*3
        merged.paste(frozen,(0,0))
        merged.save(out/(name+'.png'),dpi=(dpi,dpi))
        merged.save(out/(name+'.tiff'),compression='tiff_lzw',dpi=(dpi,dpi))
    if args.qa:
        args.qa.mkdir(exist_ok=True,parents=True)
        report=dict(upper_panel_source=info,estimates=rows,g_axes_extents_mm=geometry,title_mm=title_bbox,legend_mm=legend_bbox,y_range=[lower,upper])
        (args.qa/(name+'_g.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
