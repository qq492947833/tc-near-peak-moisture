"""Render supplementary figures 7, 8, 11, 14 and 15 from saved figure data.

Axes are laid out in mm at a printed width of 170.67 mm. Inputs retain the
saved estimates and interval endpoints. No estimation or resampling is run.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Wedge,Circle
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm,LinearSegmentedColormap
from PIL import Image
BLUE,ORANGE,INK,GRAY,GREEN,PURPLE='#167B9E','#CC7445','#233A46','#7F9099','#418D79','#9069A3'
LIGHT=LinearSegmentedColormap.from_list('annulus',['#DBAA8D','#FAFAF8','#9FC3D2'])
W=170.67
REG=['100_300km','300_800km']; POLICY=['shear200_850','midRH']
def fig(h):
 f=plt.figure(figsize=(W/25.4,h/25.4),dpi=120); f.ph=h;f.boxes={};return f
def ax(f,tag,x,y,w,h):
 a=f.add_axes((x/W,(f.ph-y-h)/f.ph,w/W,h/f.ph));f.boxes[tag]=(a,x,y,w,h);return a
def txt(f,x,y,s,**kw):
 return f.text(x/W,1-y/f.ph,s,va='center',**kw)
def title(f,k,x,y,s):
 txt(f,x,y,k,fontsize=11,fontweight='bold');txt(f,x+4.7,y,s,fontsize=9.8,linespacing=1.0)
def leg(f,handles,labels,x,y,cols=1):
 return f.legend(handles,labels,loc='upper center',bbox_to_anchor=(x/W,1-y/f.ph),ncol=cols,borderaxespad=0,frameon=False,handlelength=1.25,handletextpad=.45,columnspacing=1.2,labelspacing=.27,fontsize=9)
def fills(colors):return [Patch(fc=c,ec='none') for c in colors]
def lines(colors):return [Line2D([],[],lw=1.35,marker='o',ms=3,color=c) for c in colors]
def zero(a,v=False):
 if v:a.axvline(0,color='#A0ADB2',lw=.65,ls=':',zorder=0)
 else:a.axhline(0,color='#A0ADB2',lw=.65,zorder=0)
def ci(a,x,y,l,h,color):
 a.plot(x,y,'o-',color=color,ms=3.2,lw=1.35)
 a.fill_between(x,l,h,color=color,alpha=.15,lw=0)
 assert np.array_equal(a.lines[-1].get_ydata(),y)
def numbers(series):return np.asarray(series,dtype=float)
def row(t,**keys):
 for k,v in keys.items():t=t[t[k].eq(v)]
 assert len(t)==1,keys
 return t.iloc[0]
def CB(f,k,x,y,w,n,cmap,ticks,unit):
 ca=ax(f,k,x,y,w,1.9);cb=f.colorbar(mpl.cm.ScalarMappable(norm=n,cmap=cmap),cax=ca,orientation='horizontal',ticks=ticks);cb.outline.set_visible(False);cb.set_ticklabels([format(float(v),'.4f').rstrip('0').rstrip('.').replace('-', '−') for v in ticks])
 cb.set_label(unit,fontsize=9,labelpad=1);cb.ax.tick_params(length=2,pad=1.4)
def rings(a):
 for r in (100,300,800):a.add_patch(Circle((0,0),r,fill=False,ec='#82959D',lw=.5))
 a.set_aspect('equal');a.axis('off')
def S7(d):
 f=fig(162);title(f,'a',7,3.6,'Pre-event humidity: Temporary − Decay'); n=TwoSlopeNorm(0,-.6,.6)
 for j,(k,lab) in enumerate(zip(['q700_raw_common','q700_climatology','q700_anomaly'],['Observed','Background','Anomaly'])):
  x=30+44*j;a=ax(f,'a'+str(j),x,13,28,28);txt(f,x+14,10,lab,ha='center')
  for reg,r,ww in zip(REG,[300,800],[200,500]):
   val=row(d['annulus_state'],field=k,estimand='pre_mean',region=reg,sector='whole').estimate
   a.add_patch(Wedge((0,0),r,0,360,width=ww,facecolor=LIGHT(n(val)),edgecolor='white',lw=.6))
  rings(a);a.plot(0,0,'.',ms=2.5,c=INK);a.set(xlim=(-850,850),ylim=(-850,850))
 CB(f,'cb_a',60,43,50,n,LIGHT,[-.6,-.3,0,.3,.6],'Specific humidity difference (g kg⁻¹)')
 for j,(field,lab,u) in enumerate([('q700_asymmetric','Humidity by shear sector','g kg⁻¹'),('cyclonic_tangential_wind700_asymmetric','Wind by shear sector','m s⁻¹')]):
  x=9+84*j;title(f,'bc'[j],x-2,59,lab);a=ax(f,'bc'[j],x+10,64,52,44);sectors=['downshear','left_of_shear','upshear','right_of_shear'];q=d['sector_state'];q=q[q.field.eq(field)&q.estimand.eq('pre_mean')&q.sector.isin(sectors)]; assert len(q)==8 and q.n_sid.eq(103).all()
  lim=np.ceil(max(abs(q.estimate).max(),abs(q.original_estimate).max())/.005)*.005;n=TwoSlopeNorm(0,-lim,lim);c=plt.get_cmap('RdBu_r')
  for reg,r,ww in zip(REG,[300,800],[200,500]):
   for i,sec in enumerate(sectors):
    val=row(q,region=reg,sector=sec).estimate;a.add_patch(Wedge((0,0),r,45+90*i,135+90*i,width=ww,facecolor=c(n(val)),edgecolor='white',lw=1.05))
  rings(a);a.set(xlim=(-1300,1300),ylim=(-1100,1100))
  a.annotate('',(0,960),(0,810),arrowprops=dict(arrowstyle='->',color=INK,lw=.9))
  a.text(0,990,'Downshear',ha='center',va='bottom');a.text(0,-900,'Upshear',ha='center',va='top');a.text(-870,0,'Left',ha='right',va='center');a.text(870,0,'Right',ha='left',va='center')
  ticks=np.linspace(-lim,lim,5)
  CB(f,'cb_'+'bc'[j],x+6,109,60,n,c,ticks,'Temporary − Decay ('+u+')')
 title(f,'d',7,126,'Moisture budget, 0–24 h');a=ax(f,'d',34,131,127,10)
 cats=['horizontal_advection','vertical_advection','residual','moving_tendency']
 z=np.array([[row(d['budget'],region=reg,field=k).estimate for k in cats] for reg in REG])
 a.pcolormesh(np.arange(5)-.5,np.arange(3)-.5,z,cmap=LIGHT,norm=TwoSlopeNorm(0,-1,1),edgecolors='white',linewidth=1.0)
 a.set(xticks=range(4),xticklabels=['Horizontal','Vertical','Residual','Net tendency'],yticks=[0,1],yticklabels=['100–300 km','300–800 km'],ylim=(1.5,-.5))
 for s in a.spines.values():s.set_visible(False)
 a.tick_params(length=0,pad=2)
 for i,j in np.ndindex(z.shape):a.text(j,i,format(z[i,j],'.2f').rstrip('0').rstrip('.').replace('-','−'),ha='center',va='center')
 CB(f,'cb_d',57,148,65,TwoSlopeNorm(0,-1,1),LIGHT,[-1,-.5,0,.5,1],'Integrated difference (g kg⁻¹)')
 return f
def inline(f,titles,x=None):
 x=x or [15.5,73.2,131];w=37.8
 return [pair(f,l,n,xx,w) for l,n,xx in zip('abc',titles,x)]
def pair(f,l,t,x,w):
 title(f,l,x-9.3,5.6,t);return ax(f,l,x,15,w,43)
def S8(d):
 f=fig(87);a,b,c=inline(f,['Moving moisture\ntendency','Boundary transport\nover time','Outward transport\nby layer'])
 k=['fixed_tendency','motion_term','moving_tendency'];x=np.arange(3)
 for i,(pol,col) in enumerate(zip(['all_along','all_fixed'],[BLUE,ORANGE])):
  t=d['frame'];t=t[t.policy.eq(pol)].set_index('field').loc[k]; y=numbers(t.estimate);a.bar(x+(i-.5)*.36,y,.31,color=col);a.errorbar(x+(i-.5)*.36,y,yerr=[y-t.lo,t.hi-y],fmt='none',color=INK,capsize=2,lw=.75)
 zero(a);a.set(xticks=x,xticklabels=['Fixed frame','Translation','Moving'],ylabel='Contrast (g kg⁻¹)'); a.tick_params(axis='x',labelrotation=27)
 for pol,col in [('internal',BLUE),('final',ORANGE)]:
  t=d['time_flux'];t=t[t.side.eq(pol)&t.kind.eq('net')].sort_values('lag_h');ci(b,numbers(t.lag_h),numbers(t.mean_kg_s)/1e8,numbers(t.low_kg_s)/1e8,numbers(t.high_kg_s)/1e8,col)
 b.set(xticks=[0,6,12,18,24],xlabel='Time after departure (h)',ylabel='Net inward flux (10⁸ kg s⁻¹)')
 layers=['500-600_hPa','600-700_hPa','700-850_hPa','850-925_hPa'];x=np.arange(4)
 for i,(k,col) in enumerate(zip(['humidity','relative_motion','difference'],[GREEN,'#BC567D','#333D49'])):
  t=d['layer_flux'];t=t[t.quantity.eq(k)].set_index('layer').loc[layers];c.bar(x+(i-1)*.24,numbers(t.estimate)/1e12,.203,color=col)
 zero(c);c.set(xticks=x,xticklabels=['500–\n600','600–\n700','700–\n850','850–\n925'],xlabel='Pressure interval (hPa)',ylabel='Outward contrast (10¹² kg)')
 leg(f,fills([BLUE,ORANGE]),['Along track','Fixed centre'],34,70);leg(f,lines([BLUE,ORANGE]),['Temporary','Decay'],92,70);leg(f,fills([GREEN,'#BC567D','#333D49']),['Humidity','Motion','Total'],150,70)
 return f
def S11(d):
 f=fig(105)
 for j,(k,t,y,col) in enumerate([('midRH','Mid-level humidity','Relative humidity (%)','#428C79'),('shear200_850','Deep-layer shear','Shear (m s⁻¹)','#82649A'),('PI','Potential intensity','PI (m s⁻¹)',BLUE),('PI_headroom','PI − V','PI − V (m s⁻¹)','#437F9C')]):
  x=[16.5,101][j%2];top=[12,60][j//2];title(f,'abcd'[j],x-12,top-6,t);a=ax(f,'abcd'[j],x,top,64,32)
  z=d['history'];z=z[z.metric.eq(k)&z.radial.eq('200-800')].sort_values('relative_hour');assert z.relative_hour.tolist()==list(range(-24,25,6))
  zero(a,True);ci(a,numbers(z.relative_hour),numbers(z.estimate),numbers(z.lower),numbers(z.upper),col);a.set(xlim=(-25,25),xticks=[-24,-12,0,12,24],ylabel=y)
  if j>1:a.set_xlabel('Time from trough (h)')
 return f
def S14(d):
 f=fig(82);a,b,c=inline(f,['Recovery within\n48 h','Wind change\nover 24 h','Exposure and\ndensity support'])
 methods=['gcomp','onestep','ipw'];markers=['o','s','^'];cols=[INK,GRAY,PURPLE]
 for aa,o,label in [(a,'recovery48_pp','Recovery difference (pp)'),(b,'delta24_kt','Wind difference (kt)')]:
  for i,p in enumerate(POLICY):
   for j,m in enumerate(methods):
    r=row(d['effect'],policy=p,outcome=o,method=m);aa.errorbar(r.estimate,1-i+(.2-.2*j),xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt=markers[j],ms=3.6,color=cols[j],capsize=2,lw=.9)
  aa.set(yticks=[1,0],yticklabels=['Shear','RH'],ylim=(-.5,1.5),xlabel=label);zero(aa,True)
 for j,p in enumerate(POLICY):
  r=row(d['support'],policy=p,outcome=48);c.bar(j-.16,r.coverage*100,.28,color=INK);c.bar(j+.16,r.ratio_ESS/2231*100,.28,color='#C8D1D8')
 c.set(xticks=[0,1],xticklabels=['Shear\n−2 m s⁻¹','RH\n+5 pp'],ylim=(0,100),ylabel='Cohort fraction (%)')
 leg(f,[Line2D([],[],color=col,marker=m,lw=.9,ms=4) for m,col in zip(markers,cols)],['G-formula','One-step','Density weighting'],60,71,3)
 leg(f,fills([INK,'#C8D1D8']),['Changed exposure','Ratio ESS / n'],148,69)
 return f
def S15(d):
 f=fig(83);a,b,c=inline(f,['Follow-up\noutcomes','Recovery\ncalibration','Policy response\nonly'])
 cnt=d['counts']['count'].astype(int).to_list();assert cnt==[309,1312,610];left=0;colors=[GREEN,'#88959B','#E1E6E8']
 for j,(n,col) in enumerate(zip(cnt,colors)):
  a.barh(0,n,left=left,height=.32,color=col);a.text(left+n/2,.23,str(n),va='bottom',ha='center');left+=n
 a.set(xlim=(0,2231),ylim=(-.5,.5),yticks=[],xticks=[0,1000,2000],xlabel='Departure cohort (cyclones)');a.spines['left'].set_visible(False)
 for p,col in zip(POLICY,[BLUE,ORANGE]):
  q=d['cal'];q=q[q.policy.eq(p)&q.outcome.eq(48)].sort_values('bin');b.plot(q.predicted*100,q.observed*100,'o-',lw=1.3,ms=2.9,color=col)
  q=d['missing'];q=q[q.policy.eq(p)&q.outcome.eq(48)].sort_values('offset');c.plot(q.offset,q.estimate,'o-',color=col,lw=1.3,ms=3.1)
 b.plot([0,60],[0,60],ls=':',c=GRAY,lw=.8);b.set(xlim=(0,60),ylim=(0,60),xticks=[0,15,30,45,60],yticks=[0,15,30,45,60],xlabel='Predicted (%)',ylabel='Weighted observed (%)')
 c.set(xlabel='Log-odds offset',ylabel='Recovery contrast (pp)',xticks=[-2,-1,0,1,2]);zero(c);zero(c,True)
 leg(f,fills(colors),['Return','Complete nonreturn','Unknown'],33,68)
 leg(f,lines([BLUE,ORANGE]),['Shear −2 m s⁻¹','RH +5 pp'],119,71,2)
 return f
def render(folder,out,dpi=900):
 folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True);d={}
 for k,v in json.loads((folder/'manifest.json').read_text()).items():
  p=folder/(k+'.csv');assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];d[k]=pd.read_csv(p,float_precision='round_trip')
 mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9.8,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.size':2.6,'ytick.major.size':2.6,'xtick.major.pad':2,'ytick.major.pad':2,'axes.labelpad':2,'axes.edgecolor':INK,'savefig.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})
 audit={}
 for i,fn in [(7,S7),(8,S8),(11,S11),(14,S14),(15,S15)]:
  f=fn(d);f.canvas.draw();name=f'Figure_S{i:02d}';r=f.canvas.get_renderer();bb=f.bbox
  area={k:[round(z*25.4/f.dpi,3) for z in a.get_window_extent(r).bounds] for k,(a,*_) in f.boxes.items()}
  if i in (8,14,15):
   assert all(np.ptp([area[k][j] for k in 'abc'])<.01 for j in (1,2,3))
  if i==11:
   assert all(np.ptp([area[k][j] for k in 'abcd'])<.01 for j in (2,3))
   for key,j in [('ac',0),('bd',0),('ab',1),('cd',1)]:assert abs(area[key[0]][j]-area[key[1]][j])<.01
  beyond=[]
  hidden={id(v) for a in f.axes if not a.axison for v in [*a.get_xticklabels(),*a.get_yticklabels()]}
  for t in f.findobj(mpl.text.Text):
   if id(t) in hidden:continue
   if t.get_visible() and t.get_text() and t.axes is not None and t in t.axes.get_xticklabels()+t.axes.get_yticklabels() and not t.axes.axison:continue
   if t.get_visible() and t.get_text():
    box=t.get_window_extent(r);extent=box.extents
    if np.isfinite(extent).all() and (min(extent[:2])<-.5 or extent[2]>bb.x1+.5 or extent[3]>bb.y1+.5):beyond.append(t.get_text())
  audit[name]={'width_mm':W,'height_mm':f.ph,'body_font_pt':9,'dpi':dpi,'axes_xywh_mm':area,'text_outside_canvas':beyond}
  f.savefig(out/(name+'_preview.png'),dpi=155);f.savefig(out/(name+'.png'),dpi=dpi);plt.close(f)
  with Image.open(out/(name+'.png')) as im:
   im.save(out/(name+'.tiff'),compression='tiff_lzw',dpi=(dpi,dpi))
   with Image.open(out/(name+'.tiff')) as ti:assert im.size==ti.size and im.tobytes()==ti.tobytes()
  print(name,area,'overflow',beyond,flush=True)
 (out/'axes_geometry.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,default=Path(__file__).parent/'inputs');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--dpi',type=int,default=900);a=p.parse_args();render(a.input_dir,a.output_dir,a.dpi)

