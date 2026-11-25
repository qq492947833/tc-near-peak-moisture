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
TEMPORARY_COLOR='#2F6B9A'
def draw_wind_sequence(ax, source):
    base = source / 'Figure_01/input'
    df = pd.read_csv(base/'lifecycle_schematic.csv')
    k = json.loads((base/'lifecycle_schematic.json').read_text())
    vm = k['genesis_kt'] + .90*(k['lmi_kt']-k['genesis_kt'])
    assert vm == k['I090_kt']
    reference, returned = k['episode_ranges_h']
    t = df.hours.to_numpy(); v = df.wind_kt.to_numpy()
    # Display the near-peak part of the original illustrative six-hour history.
    keep = t >= reference[0]; t,v = t[keep],v[keep]
    assert t[np.argmin(abs(t-k['valley_h']))]==k['valley_h']
    assert np.array_equal(v>=vm,[True,True,False,False,True,True,False,False])
    # Event times are six-hour observations; interpolation locates no event boundary.
    td=float(t[(t>reference[1]) & (v<vm)][0]);tr=float(k['valley_h'])
    ret=float(returned[0]);fd=float(k['Decay_onset_h']);end=float(t[-1])
    assert td<tr<ret<fd<end and all(x in t for x in (td,tr,ret,fd))
    cols=[TEMPORARY_COLOR,'#2A9D6F','#C96A3A']
    X=lambda h: 9+(np.asarray(h)-t[0])*158/(end-t[0])
    Y=lambda kts: 4.1+(np.asarray(kts)-65)*.37
    for left,right,col in [(td,tr,cols[0]),(tr,ret,cols[1]),(fd,end,cols[2])]:
        ax.add_patch(mpl.patches.Rectangle((X(left),2.1),X(right)-X(left),16.4,
                                           fc=col,ec='none',alpha=.12,zorder=0))
    ax.plot([8,170],[Y(vm)]*2,ls=(0,(3.,2.4)),lw=.8,color='#99A9AC',zorder=1)
    for left,right,col in [(t[0],td,INK),(td,tr,cols[0]),(tr,ret,cols[1]),(ret,fd,INK),(fd,end,cols[2])]:
        sel=(t>=left)&(t<=right)
        ax.plot(X(t[sel]),Y(v[sel]),lw=1.65,color=col,solid_joinstyle='round',solid_capstyle='round',zorder=3)
    for time,col in zip([td,tr,ret,fd],[cols[0],cols[1],cols[1],cols[2]]):
        ax.plot(X(time),Y(v[t==time][0]),'o',ms=3.5,mec='white',mew=.45,mfc=col,zorder=5)
    # Band-centred labels sit outside the shaded intervals. The top pair share a baseline.
    for left,right,y,label,col in [(td,tr,20.89,'Temporary departure',cols[0]),
                                   (tr,ret,-2.40,'Reintensification',cols[1]),
                                   (fd,end,20.89,'Final decay',cols[2])]:
        ax.text((X(left)+X(right))/2,y,label,ha='center',va='baseline',fontsize=8.5,color=col)
    ax.text(25.5,Y(vm)-3.1,'Near-peak',ha='right',va='center',fontsize=8.5,color=INK)
    return dict(hours=t.tolist(),wind_kt=v.tolist(),threshold_kt=vm,
                observed_events_h=[td,tr,ret,fd],band_edges_h=[[td,tr],[tr,ret],[fd,end]])


def _bc_text(ax, x, y, s, *, ha='center', weight='normal', color='#233A46', va='center'):
    return ax.text(x,y,s,ha=ha,va=va,weight=weight,color=color,fontfamily='Arial',
                   fontsize=8.5,zorder=22,clip_on=False)


def _bc_arrow(ax,x,y,X,Y,color,lw,ms):
    from matplotlib.patches import FancyArrowPatch
    arrow=FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',color=color,linewidth=lw,
                         mutation_scale=ms,shrinkA=0,shrinkB=0,joinstyle='round',zorder=14)
    ax.add_patch(arrow)


def draw_environment(ax,scene):
    """Pre-departure outer moisture and shear; following-day inner moistening."""
    from matplotlib.patches import Ellipse
    ink,teal,blue,ochre='#233A46','#428C79',TEMPORARY_COLOR,'#C96A3A'
    _bc_text(ax,4.3,43.75,'Environment and moisture',ha='left',weight='bold',va='top')
    for x,name,col,rim,fill,shear,sw in [
            (22,'Temporary',blue,'#74A998','#A9CFC2','Weaker shear',10.0),
            (68,'Decay',ochre,'#BBCFC7','#E9F2EE','Stronger shear',17.6)]:
        _bc_text(ax,x,36.4,name,weight='bold',color=col)
        _bc_text(ax,x,31.5,shear)
        _bc_arrow(ax,x-sw/2,28.5,x+sw/2,28.5,ink,1.05,6.7)
        top,base,radius=22.5,9.4,10.4
        ax.add_patch(Ellipse((x,base),35.2,10.5,fc=fill,ec=rim,lw=.7,zorder=0))
        ax.add_patch(Ellipse((x,base),2*radius,2*.28*radius,fc='white',ec='none',zorder=.5))
        scene.cylinder(ax,x,top,base,radius,col,name)
    _bc_arrow(ax,15.7,8.2,15.7,14.6,teal,1.3,7.4)
    _bc_text(ax,20.2,2.3,'Vert. adv. (+) · 0–24 h',color=teal)
    # The first anchor lies between the core and 300-km rim; the second on the outer annulus.
    ax.plot([63.0,61.2,58.9],[7.9,5.1,4.1],c='#8CA2A6',lw=.55,zorder=11)
    ax.plot([80.9,82.5,82.5],[7.3,5.6,4.2],c='#8CA2A6',lw=.55,zorder=11)
    _bc_text(ax,56.2,2.1,'100–300 km');_bc_text(ax,81.5,2.1,'300–800 km')


def draw_exchange(ax,source):
    """Original matched 80-pair, full-layer 0–24 h moving-boundary totals (kg)."""
    from pathlib import Path
    from matplotlib.patches import Polygon, Rectangle
    import pandas as pd
    path=Path(source)/'Figure_08/input/paired_flux_components.csv'
    df=pd.read_csv(path,float_precision='round_trip')
    f=df[(df.cohort=='original_post80')&(df.window=='0_24h')&(df.factor==1)&
         (df.layer=='all')&(df.sector=='all')].set_index(['kind','quantity'],verify_integrity=True)
    def lookup(kind,quantity):
        row=f.loc[(kind,quantity)]
        assert row.n==80 and row.finite_n==80 and row.units=='kg'
        return float(row.estimate)
    values={cat:{k:lookup(k,cat)/1e12 for k in ['in','out','net']} for cat in ['temporary','final']}
    deltas={k:lookup(k,'difference')/1e12 for k in ['in','out','net']}
    for k in deltas:assert abs(values['temporary'][k]-values['final'][k]-deltas[k])<1e-12
    parts={k:lookup('out',k)/1e12 for k in ['humidity','relative_motion']}
    assert abs(sum(parts.values())-deltas['out'])<1e-12
    # Positive gross exchange values retain their physical direction in both classes.
    _bc_text(ax,4.3,33.75,'Boundary exchange / 300 km',ha='left',weight='bold',va='top')
    colx=[33.,55.,78.]
    for x,t in zip(colx,['Gross inward','Gross outward','Net inward']):_bc_text(ax,x,27.1,t)
    width_per_Tkg=.19   # shaft thickness in mm per 10^12 kg; one scale for all six totals
    def horizontal(x,y,sgn,h,col):
        pts=[(-7.6,-h/2),(4.6,-h/2),(4.6,-h/2-.72),(7.6,0),(4.6,h/2+.72),(4.6,h/2),(-7.6,h/2)]
        ax.add_patch(Polygon([(x+sgn*u,y+v) for u,v in pts],closed=True,fc=col,ec='none',zorder=3))
    # Short meridians mark the moving boundary; left is the control-volume interior.
    for x in colx:
        ax.plot([x,x],[13.,24.2],color='#ACBFBE',lw=.8,zorder=1)
    for cat,title,y,edge in [('temporary','Temporary',21.5,TEMPORARY_COLOR),('final','Decay',15.8,'#C96A3A')]:
        _bc_text(ax,1.0,y,title,ha='left',weight='bold',color=edge)
        for x,k,sgn,col in zip(colx,['in','out','net'],[-1,1,-1],['#438E87','#C88153','#167B9E']):
            horizontal(x,y,sgn,values[cat][k]*width_per_Tkg,col)
    ax.plot([.8,91.1],[12.3,12.3],color='#CAD6DC',lw=.7)
    _bc_text(ax,1.,6.15,'Export contrast',ha='left',weight='bold')
    # Signed rectangular bars denote outward-contrast contributions, not transport directions.
    base=8.7; mm_per_Tkg=5.7
    ax.plot([34.,69.],[base,base],color='#83999C',lw=.75,zorder=2)
    for x,key,label,tx in [(38,'humidity','q (+)',46.9),(64,'relative_motion','(u−c)·n (−)',78.0)]:
        length=parts[key]*mm_per_Tkg
        ax.add_patch(Rectangle((x-2.3,base),4.6,length,facecolor='#5D8893' if length>0 else '#427E78',
                               edgecolor='none',zorder=3))
        _bc_text(ax,tx,5.45,label)
    return dict(source='Figure_08/input/paired_flux_components.csv',support_pairs=80,
                window_hours=[0,24],layer_hPa=[500,925],unit='10^12 kg',values=values,
                contrasts=deltas,outward_components=parts,shaft_mm_per_Tkg=width_per_Tkg,
                decomposition_mm_per_Tkg=mm_per_Tkg)



def draw_energy(ax, csv_path):
    """Layer (Pre and Recovery) and pressure-surface wind-matched contrasts; lengths in mm."""
    import csv
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
    ink, teal, blue, ochre, gray = '#233A46', '#428C79', '#167B9E', '#CC7445', '#CAD6DC'
    def txt(x,y,t,ha='center',bold=False,color=ink):
        return ax.text(x,y,t,ha=ha,va='center',family='Arial',fontsize=8.5,
                       color=color,weight='bold' if bold else 'normal',zorder=6)
    def path(x,y,xx,yy,color=gray,lw=.7):
        ax.plot([x,xx],[y,yy],color=color,lw=lw,solid_capstyle='round',zorder=2)
    def box(x,y,w,h,fc,ec,rad=1.3,lw=.75):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rad}',
                                    facecolor=fc,edgecolor=ec,lw=lw,zorder=3))
    def flow(x,y,xx,yy,col,lw,ms):
        artist=FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=ms,
                              shrinkA=0,shrinkB=0,color=col,lw=lw,zorder=4)
        ax.add_patch(artist);return artist
    def capsule(x,w,t):
        box(x,55.8,w,5.8,'#FFFFFF','#AABFC1',2.8,.65)
        txt(x+w/2,58.7,t)
    # The gray brace stops at the Recovery row. Its labels do not apply to the 700-hPa bars.
    txt((.6+73)/2,77.1,'500–850 hPa · 2–16 d',bold=True)
    path(72.3,73.8,75,73.8); path(75,73.8,75,24.7);path(75,24.7,72.3,24.7)
    path(.6,50.5,73,50.5); path(.6,23.0,73,23.0)
    txt(.8,70.6,'Pre','left',True)
    box(.45,52.6,44.1,15.5,'#F4F8F8','#8BA5A8',1.0)
    # The inset groups only the pressure-level comparisons, away from the layer interface.
    box(1.45,54.35,38.25,12.3,'#FFFFFF','#C7D6D7',.8,.55)
    txt(20.6,64.4,'700 hPa')
    capsule(1.9,12,'APE ↓');capsule(14.7,10.5,'KE ↓');capsule(26,13.2,'Conv. ↓')
    # Side port represents the layer integral across both pressure boundaries.
    path(44.55,55.9,44.55,63.5,blue,1.05)
    flow(53,63.0,42.5,63.0,teal,1.8,7.8)
    txt(58.7,67.0,'Pressure flux (+)')
    flow(42.5,56.2,53,56.2,ochre,1.0,6.8)
    txt(62.3,55.1,'Vert. KE (−)')
    txt(.8,48.1,'Recovery','left',True)
    y_flow=37.3; y_label=43.0; box_h=7.8
    box(13,y_flow-box_h/2,15.3,box_h,'#E8F1ED',teal,2)
    box(49,y_flow-box_h/2,11.5,box_h,'#E8F1F5',blue,2)
    txt(20.65,y_flow,'Dry APE');txt(54.75,y_flow,'KE')
    flows=[(flow(3.8,y_flow,13,y_flow,teal,1.0,6.6),'Transfer (+)'),
           (flow(28.4,y_flow,48.85,y_flow,teal,2.35,8.8),'Conversion (+)'),
           (flow(60.65,y_flow,72.65,y_flow,ochre,1.6,7.6),'Transfer (−)')]
    flow(54.75,25.4,54.75,y_flow-box_h/2-.1,teal,1.0,6.6)
    ax.figure.canvas.draw()
    alignment={}
    for arrow,label in flows:
        bbox=arrow.get_path().get_extents()
        mid=(bbox.x0+bbox.x1)/2
        tx=ax.text(mid,y_label,label,ha='center',va='baseline',fontsize=8.5,color=ink,zorder=6)
        alignment[label]=dict(x_mid_mm=float(mid),y_label_baseline_mm=y_label,y_flow_mm=y_flow)
    txt(52,26.5,'Vert. input (+)','right')
    txt(.8,19.8,'Wind-matched','left',True)
    txt(73,19.8,'700 hPa','right')
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        rows=sorted((r for r in csv.DictReader(f) if r['level']=='700' and r['metric']=='APE'),
                    key=lambda r:int(r['window']))
    assert [r['window'] for r in rows]==['0','1','2']
    assert all((r['region'],r['n_sid'],r['n_events'],r['units'])==('r0_300','108','115','J kg-1') for r in rows)
    vals=[float(r['estimate']) for r in rows]
    y0,mm_per_unit=8.3,2.03
    path(13.0,y0,73.1,y0,'#7D919B',.75)
    txt(10.4,y0,'ΔAPE','right')
    for x,val,t in zip([28,48,66],vals,['>16 d','2–16 d','<2 d']):
        color=teal if val>0 else ochre
        ax.add_patch(Rectangle((x-3.75,y0),7.5,val*mm_per_unit,ec='none',fc=color,zorder=3))
        txt(x,2.15,t)
    # Conversion at matched wind speed in the slow window, distinct from the layer integral.
    txt(28,y0+vals[0]*mm_per_unit+3.1,'Conv. ↑')
    return {'matched_values_J_kg':vals, 'matched_n_sid':108,'matched_n_events':115,
            'flow_alignment':alignment,'bar_zero_mm':y0,'bar_mm_per_J_kg':mm_per_unit}



def collect_visible_labels(f):
    import re
    strings=[t.get_text() for a in f.axes for t in a.texts if t.get_visible() and t.get_alpha()!=0 and t.get_text().strip()]
    assert all('$' not in s for s in strings)
    return dict(labels=strings,english_words=sum(len(re.findall('[A-Za-z]+',s)) for s in strings),nonspace_chars=sum(len(''.join(s.split())) for s in strings))

def draw(source,out,qa=None):
    from matplotlib.font_manager import FontProperties,findfont
    source=Path(source).resolve()
    sys.path.insert(0,str(source/'Figure_08/code'))
    import scene_20261026_preserved as symbols
    font_path=findfont(FontProperties(family='Arial'),fallback_to_default=False)
    mpl.rcParams.update({'font.family':'Arial','font.size':8.5,'pdf.fonttype':42,'axes.unicode_minus':True,'savefig.bbox':None})
    width,height=180,112
    f=plt.figure(figsize=(width/25.4,height/25.4),facecolor='white')
    def canvas(name,x,y,w,h):
        ax=f.add_axes([x/width,y/height,w/width,h/height]);ax.set_gid(name);ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off');return ax
    a=canvas('a',3,87,174,23);b=canvas('b',3,38,92,44);c=canvas('c',3,3,92,34);d=canvas('d',101,3,76,79)
    for ax,t,h in [(a,'a',23),(b,'b',44),(c,'c',34),(d,'d',79)]:
        ax.text(0,h-.25,t,va='top',ha='left',color=INK,fontsize=8.5,weight='bold')
    wind=draw_wind_sequence(a,source);draw_environment(b,symbols)
    exchange=draw_exchange(c,source)
    bars=draw_energy(d,Path(__file__).resolve().parent/'inputs/wind_matched_dry_APE.csv')
    f.canvas.draw()
    boxes={ax.get_gid():(ax.get_window_extent().extents/f.dpi*25.4).tolist() for ax in f.axes}
    for i in (0,2): assert abs(boxes['b'][i]-boxes['c'][i])<1e-6
    assert abs(boxes['b'][3]-boxes['d'][3])+abs(boxes['c'][1]-boxes['d'][1])<1e-6
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if qa:
        qa=Path(qa);qa.mkdir(exist_ok=True,parents=True)
        data=dict(canvas_mm=[width,height],axes_mm=boxes,wind=wind,boundary_exchange=exchange,matched=bars,text=collect_visible_labels(f),Arial_sha256=hashlib.sha256(Path(font_path).read_bytes()).hexdigest(),matplotlib=mpl.__version__,numpy=np.__version__,pandas=pd.__version__)
        (qa/'figure6_render.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    stem=out/'Figure_06'
    f.savefig(stem.with_suffix('.png'),dpi=900,facecolor='white',bbox_inches=None)
    f.savefig(stem.with_suffix('.pdf'),facecolor='white',bbox_inches=None,metadata={'CreationDate':None,'ModDate':None})
    with Image.open(stem.with_suffix('.png')) as im:rgb=im.convert('RGB')
    rgb.save(stem.with_suffix('.png'),dpi=(900,900));rgb.save(stem.with_suffix('.tiff'),dpi=(900,900),compression='tiff_lzw')
    plt.close(f)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--qa',type=Path)
    a=p.parse_args();draw(a.source,a.output,a.qa)

