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
    # Linear segments cross the threshold exactly once between adjacent samples.
    crosses=[t[i]+(vm-v[i])*(t[i+1]-t[i])/(v[i+1]-v[i])
             for i in range(len(t)-1) if (v[i]>=vm)!=(v[i+1]>=vm)]
    down,up,final = crosses
    assert down < k['valley_h'] < up < final
    X=lambda h: 9+(np.asarray(h)-t[0])*158/(t[-1]-t[0])
    Y=lambda s: 3.2+(np.asarray(s)-65)*.42
    ax.plot([8,170],[Y(vm)]*2,ls=(0,(3.,2.4)),lw=.8,color='#99A9AC',zorder=1)
    for u,z,col in [(t[0],down,INK),(down,k['valley_h'],BLUE),(k['valley_h'],up,TEAL),(up,final,INK),(final,t[-1],OCHRE)]:
        h=np.r_[u,t[(t>u)&(t<z)],z]; ax.plot(X(h),Y(np.interp(h,t,v)),lw=1.65,color=col,
            solid_joinstyle='round',solid_capstyle='round',zorder=3)
    for x,y,s,col in [(65,19.4,'Temporary departure',BLUE),(86,4.2,'Reintensification',TEAL),
                      (139,4.8,'Final decay',OCHRE),(170,18.0,'Near-peak',INK)]:
        ax.text(x,y,s,ha='right' if s=='Near-peak' else 'center',va='center',fontsize=8.5,color=col)
    return dict(hours=t.tolist(),wind_kt=v.tolist(),threshold_kt=vm,illustrative_crossings_h=crosses,
                trough_h=k['valley_h'],first_return_observation_h=returned[0],final_departure_observation_h=k['Decay_onset_h'])




def draw_moisture(ax, scene):
    """Qualitative comparison of two equal-sized moving control volumes (local mm)."""
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
    ink, teal, blue, orange = '#233A46', '#428C79', '#167B9E', '#CC7445'
    def text(x, y, s, **k):
        return ax.text(x, y, s, ha=k.pop('ha','center'), va='center', color=k.pop('color',ink),
                       fontfamily='Arial', fontsize=8.5, zorder=22, **k)
    def directed(start, end, col, lw, ms, gid):
        p=FancyArrowPatch(start, end, arrowstyle='-|>', color=col, lw=lw,
                          mutation_scale=ms, shrinkA=0, shrinkB=0, zorder=15,
                          capstyle='round', joinstyle='round')
        p.set_gid(gid); ax.add_patch(p)
    def leader(x, y):
        ax.plot(x,y,color='#8CA2A6',lw=.65,zorder=11,solid_capstyle='round')
    def badge(cx, y, width, s, fc, ec):
        ax.add_patch(FancyBboxPatch((cx-width/2,y-2.85),width,5.7,
                     boxstyle='round,pad=0,rounding_size=2.65',fc=fc,ec=ec,lw=.9,zorder=15))
        text(cx,y,s)
    # Cylinder's bottom and its two radius markers mark 300 and 100 km.
    top, base, radius = 55.4, 31.2, 12.0
    for x, name, edge, ring, rim in [(22,'Temporary',blue,'#A9CFC2','#74A998'),
                                      (70,'Decay',orange,'#E9F2EE','#BBCFC7')]:
        text(x,72.4,name,weight='bold',color=edge)
        ax.add_patch(Ellipse((x,base),40.4,12.2,fc=ring,ec=rim,lw=.75,zorder=0))
        ax.add_patch(Ellipse((x,base),2*radius,2*.28*radius,fc='white',ec='none',zorder=.5))
        scene.cylinder(ax,x,top,base,radius,edge,name)
    # Same direction of environmental shear, ordinal lengths only.
    text(46,65.1,'Shear')
    directed((16.4,65.1),(27.6,65.1),ink,1.25,7.2,'temporary_shear')
    directed((59.7,65.1),(80.3,65.1),ink,1.25,7.2,'decay_shear')
    # An identical 300-km boundary is identified once, at the sides of the volumes.
    text(46,46.6,'300 km')
    leader([34.7,39],[45.6,46.2]); leader([53,57.3],[46.2,45.6])
    # Both exports leave the volume; the gross outward paired contrast is unadjusted.
    directed((30.8,30.0),(39.5,21.8),orange,1.15,7.6,'temporary_export')
    directed((61.2,30.0),(52.5,21.8),orange,2.55,10.2,'decay_export')
    badge(36.9,15.6,10.8,'q (+)','#FBF9F3','#CABE9E')
    badge(55.1,15.6,21.5,'(u−c)·n (−)','#F3F8F6','#81ABA0')
    # Local moistening contrast at 700 hPa. The arrow is not vertical wind velocity.
    directed((16.3,29.0),(16.3,36.8),teal,1.45,8,'inner_vertical_advection')
    text(10.0,23.1,'Vert. adv. (+)',color=teal)
    # The two leaders touch the inner and outer annuli, respectively, on the same base.
    leader([73.4,74.2,76.6],[29.4,25.6,23.9])
    text(77.0,21.1,'100–300 km')
    leader([86.4,89.0,89.0,86.8],[30.2,28.2,16.8,15.2])
    text(76.7,15.2,'300–800 km')



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
    txt(.8,47.3,'Recovery','left',True)
    txt(38.3,42.2,'Conversion (+)')
    box(13,33.4,15.3,7.8,'#E8F1ED',teal,2)
    box(49.0,33.4,11.5,7.8,'#E8F1F5',blue,2)
    txt(20.65,37.3,'Dry APE');txt(54.75,37.3,'KE')
    flow(1.4,37.3,13,37.3,teal,1.0,6.6)
    flow(28.4,37.3,48.85,37.3,teal,2.35,8.8)
    sink=flow(60.65,37.3,72.65,37.3,ochre,1.6,7.6)
    flow(54.75,25.4,54.75,33.3,teal,1.0,6.6)
    txt(10,31.0,'Transfer (+)')
    ax.figure.canvas.draw()
    bounds=sink.get_path().get_extents()
    txt((bounds.x0+bounds.x1)/2,43.25,'Transfer (−)')
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
            'bar_zero_mm':y0,'bar_mm_per_J_kg':mm_per_unit}



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
    a=canvas('a',3,87,174,23);b=canvas('b',3,3,92,79);c=canvas('c',101,3,76,79)
    for ax,t,h in [(a,'a',23),(b,'b',79),(c,'c',79)]:ax.text(0,h-.25,t,va='top',ha='left',color=INK,fontsize=8.5,weight='bold')
    wind=draw_wind_sequence(a,source);draw_moisture(b,symbols)
    bars=draw_energy(c,Path(__file__).resolve().parent/'inputs/wind_matched_dry_APE.csv')
    f.canvas.draw()
    boxes={ax.get_gid():(ax.get_window_extent().extents/f.dpi*25.4).tolist() for ax in f.axes}
    assert max(abs(boxes['b'][i]-boxes['c'][i]) for i in (1,3))<1e-6
    assert max(abs(boxes['a'][0]-boxes['b'][0]),abs(boxes['a'][2]-boxes['c'][2]),abs(boxes['c'][0]-boxes['b'][2]-6))<1e-6
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if qa:
        qa=Path(qa);qa.mkdir(exist_ok=True,parents=True)
        data=dict(canvas_mm=[width,height],axes_mm=boxes,wind=wind,matched=bars,text=collect_visible_labels(f),Arial_sha256=hashlib.sha256(Path(font_path).read_bytes()).hexdigest(),matplotlib=mpl.__version__,numpy=np.__version__,pandas=pd.__version__)
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

