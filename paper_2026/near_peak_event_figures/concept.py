from pathlib import Path
import argparse,json,csv,hashlib,sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,Ellipse,FancyArrowPatch
from PIL import Image
import pandas as pd
INK='#233A46'; TEAL='#428C79'; BLUE='#167B9E'; OCHRE='#CC7445'; GRAY='#CAD6DC'
def label(a,x,y,s,**kw):
 a.text(x,y,s,color=kw.pop('color',INK),fontsize=kw.pop('fontsize',8.6),ha=kw.pop('ha','center'),va='center',**kw)
def box(a,x,y,w,h,s,face,edge):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.022',fc=face,ec=edge,lw=1))
 label(a,x+w/2,y+h/2,s,fontsize=9)
def arrow(a,x,y,X,Y,color=INK,style='-|>',lw=1.3):
 a.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle=style,mutation_scale=9,linewidth=lw,color=color,shrinkA=0,shrinkB=0))
def load_evidence(source):
 d=source/'Figure_08'/'input'; result={}
 def row(path,selector):
  f=pd.read_csv(path); k=f.loc[selector(f)]; assert len(k)==1,(path,len(k))
  return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),row=k.iloc[0].dropna().to_dict())
 result['shear']=row(d/'departure_environment.csv',lambda x:(x.variable=='shear200_850')&(x.model=='geographical')&(x.coverage=='coverage_116'))
 result['humidity']=row(d/'radial_moisture.csv',lambda x:(x.region=='300_800km')&(x.sector=='whole')&(x.field=='q700')&(x.estimand=='pre_mean'))
 result['outward']=row(d/'paired_flux_components.csv',lambda x:(x.kind=='out')&(x.layer=='all')&(x.sector=='all')&(x.quantity=='difference')&(x.cohort=='original_post80')&(x.factor==1)&(x.window=='0_24h'))
 for w in ['D_pre','D_post']:
  result[w]=row(d/'common_prepost.csv',lambda x:(x.kind=='out')&(x.quantity==w))
 result['conversion']=row(d/'ascent_energy.csv',lambda x:(x.content_id=='SI_RISE_ENERGY_rates')&(x.scheme=='main')&(x.support_mode=='native_diagnostic')&(x.metric=='C_A_to_K')&(x.level.astype(str)=='700')&(x.window==1)&(x.region=='r0_300')&(x.estimand=='rise_rate_integral')&(x['sample']=='all_events'))
 for i in range(3):
  result[str(i)]=row(source/'Figure_06'/'inputs'/'panel_g_source_rows.csv',lambda x:(x.metric=='APE')&(x.window==i))
 assert result['shear']['row']['original_raw']<0 and result['humidity']['row']['estimate']>0
 assert all(result[k]['row']['estimate']<0 for k in ('outward','D_pre','D_post','1','2'))
 assert all(result[k]['row']['estimate']>0 for k in ('0','conversion'));return result
def draw(source,out,qa=None):
    """Use the published cyclone and reservoir vector shapes at their original proportions."""
    evidence=load_evidence(source)
    sys.path.insert(0,str(source/'Figure_08'/'code'))
    import scene_20261026_preserved as symbols
    import scene as original
    d=source/'Figure_08'/'input'
    flux=pd.read_csv(d/'paired_flux_components.csv')
    sel=flux[(flux.kind=='out')&(flux.layer=='all')&(flux.sector=='all')&(flux.factor==1)&
             (flux.cohort=='original_post80')&(flux.window=='0_24h')].set_index('quantity')
    assert sel.loc['humidity','estimate']>0 and sel.loc['relative_motion','estimate']<0
    pre=pd.read_csv(d/'energy_primary.csv').set_index('metric')
    assert all(pre.loc[k,'estimate']<0 for k in ('KE','APE','C_A_to_K'))
    joint_path=source.parent.parent/'physics/vertical_extension/plots/Figure_S26_source.csv'
    joint=pd.read_csv(joint_path,float_precision='round_trip').query("panel=='c' and window==1 and region=='r0_300' and variant=='joint_with_transfer'").set_index('row')
    assert list(joint.index)==[1,2,3] and joint.level.eq('500-850').all()
    assert joint.n_events.eq(132).all() and joint.n_sid.eq(122).all()
    assert joint.estimate.iloc[0]>0 and joint.estimate.iloc[1]>0 and joint.estimate.iloc[2]<0
    evidence['layer_integrals']=dict(file=joint_path.name,sha256=hashlib.sha256(joint_path.read_bytes()).hexdigest(),rows=joint[['metric','estimate','ci_lower','ci_upper','unit','n_events','n_sid']].to_dict('records'))
    out.mkdir(exist_ok=True,parents=True)
    mpl.rcParams.update({'font.family':'Arial','font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True})
    width,height=180,112
    f=plt.figure(figsize=(width/25.4,height/25.4))
    def canvas(name,x,y,w,h):
        ax=f.add_axes([x/width,y/height,w/width,h/height]);ax.set_gid(name)
        ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off')
        return ax
    a=canvas('a',3,87,174,23);b=canvas('b',3,3,98,79);c=canvas('c',107,3,70,79)
    def txt(ax,x,y,words,**kw):
        return ax.text(x,y,words,fontsize=kw.pop('size',8.3),weight=kw.pop('weight','normal'),
                       color=kw.pop('color',INK),ha=kw.pop('ha','center'),va='center',linespacing=1.1,zorder=24,**kw)
    for ax,letter,heading,H in [(a,'a','Near-peak sequence',23),(b,'b','Moving-boundary moisture exchange',79),(c,'c','Multiscale energetics',79)]:
        txt(ax,0,H-1.8,f'{letter}  {heading}',weight='bold',size=9,ha='left')
    labels=[('Reference near-peak\nepisode',TEAL),('Temporary\ndeparture',BLUE),('Wind trough',INK),
            ('Near-peak\nreturn',TEAL),('Final\ndeparture',OCHRE)]
    stations=[12,51,86,122,157]
    for i,(x,(words,color)) in enumerate(zip(stations,labels)):
        a.scatter([x],[4],s=24,c=color,zorder=5,edgecolors='white',linewidths=.3)
        txt(a,x,10,words,color=color)
        if i+1<len(stations):original.arrow(a,(x+2,4),(stations[i+1]-2,4),'sequence_'+str(i),GRAY,.9,6)
    for x,cat,col,outer,loss,delta in [(24,'Temporary',BLUE,'Moister outer annulus','Less gross export',5.5),
                                       (74,'Decay',OCHRE,'Drier outer annulus','More gross export',10.)]:
        txt(b,x,69,cat,weight='bold',color=col)
        txt(b,x,63.5,'Weaker shear' if cat=='Temporary' else 'Stronger shear')
        original.arrow(b,(x-5,58),(x+delta,58),cat+'_shear','#627A85',1.0,6)
        b.add_patch(Ellipse((x,29),42,10.5,facecolor='#D8E8E2',edgecolor='#AAC8C0',lw=.7,zorder=0))
        symbols.cylinder(b,x,50,31,10.5,col,cat)
        sign=-1 if cat=='Temporary' else 1
        original.arrow(b,(x+sign*9,34.2),(x+sign*19,34.2),cat+'_normal_velocity',col,1.25,7)
        txt(b,x,20.7,outer,size=8.15);txt(b,x,15,loss,color=col,size=8.15)
    txt(b,49,53.3,'300 km boundary',size=8)
    txt(b,49,45,'(u − c) · n',size=8);txt(b,49,39.5,'Outward normal',size=8)
    txt(b,49,8.3,'Annuli: 100–300 km and 300–800 km',color='#4F7072',size=8)
    txt(b,49,2.3,'Gross outward contrast: humidity (+), relative motion (−)',size=8.05)
    txt(c,35,69,'Before departure · 700 hPa',weight='bold',size=8.6)
    txt(c,35,62.7,'Temporary − decay, 2–16 d',size=8.1)
    txt(c,35,57.1,'Lower dry APE, KE and conversion',size=8.1)
    c.plot([2,68],[53,53],color=GRAY,lw=.75)
    txt(c,35,48.1,'During reintensification',weight='bold',size=8.6)
    txt(c,35,42.7,'500–850 hPa · 2–16 d',size=8.1)
    original.reservoir(c,15,34,'APE');original.reservoir(c,55,34,'KE')
    original.arrow(c,(24,34),(46,34),'positive_conversion',OCHRE,1.4,7)
    txt(c,35,38,'Conversion (+)',size=8)
    txt(c,35,26.7,'Vertical KE input (+) · Total KE transfer (−)',size=8.0)
    c.plot([2,68],[22.8,22.8],color=GRAY,lw=.75)
    txt(c,35,18.4,'Wind-matched dry APE · 700 hPa',weight='bold',size=8.6)
    txt(c,35,12.6,'Reintensification − decay',size=8.1)
    for x,period,sign in [(11,'>16 d','Higher'),(35,'2–16 d','Lower'),(59,'<2 d','Lower')]:
        txt(c,x,7.4,period,size=8.1);txt(c,x,2.25,sign,color=TEAL if sign=='Higher' else INK,weight='bold',size=8.1)
    f.canvas.draw();renderer=f.canvas.get_renderer()
    boxes={ax.get_gid():(np.array(ax.get_window_extent(renderer).extents)/f.dpi*25.4).tolist() for ax in f.axes}
    error=max(abs(boxes['b'][i]-boxes['c'][i]) for i in (1,3));assert error<=.3
    if qa:
        qa.mkdir(parents=True,exist_ok=True);f.savefig(qa/'Figure_06.pdf',dpi=180);f.savefig(qa/'Figure_06.svg')
        (qa/'Figure_06_geometry.json').write_text(json.dumps(dict(size_mm=[width,height],axes_mm=boxes,max_edge_error_mm=error),indent=2))
        (qa/'Figure_06_data.json').write_text(json.dumps(evidence,indent=2,default=str))
    stem=out/'Figure_06';f.savefig(stem.with_suffix('.png'),dpi=900,facecolor='white');plt.close(f)
    with Image.open(stem.with_suffix('.png')) as im:rgb=im.convert('RGB')
    rgb.save(stem.with_suffix('.png'),dpi=(900,900));rgb.save(stem.with_suffix('.tiff'),compression='tiff_lzw',dpi=(900,900))
    print(stem.name,rgb.size)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--qa',type=Path)
    args=p.parse_args();draw(args.source,args.output,args.qa)

