"""Rebuild the 11 selected supplementary figures from the v2.1.5 plotting inputs."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.ticker import MaxNLocator
from PIL import Image
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--figures',type=int,nargs='+',default=list(range(1,12)))
parser.add_argument('--out',type=Path,default=ROOT/'outputs')
args=parser.parse_args()
repo=args.repo/'paper_2026'
legacy=repo/'lifecycle_figures/figures'
select=repo/'selected_evidence'
out=args.out
out.mkdir(exist_ok=True,parents=True)

dpi=900
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(mod)
 return mod
b=load_module('group_b',select/'layout/group_b/render_selected.py')
read=lambda p:pd.read_csv(p,float_precision='round_trip')
data={}
for k,v in json.loads((select/'layout/group_b/inputs/manifest.json').read_text()).items():
 p=select/'layout/group_b/inputs'/(k+'.csv')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
 data[k]=read(p)
def figure(height):
 f=plt.figure(figsize=(180/25.4,height/25.4),dpi=135)
 f.ph=height;f.boxes={}
 return f
b.fig=figure
def rect(f,l,title,x,t,w,h):
 b.title(f,l,x-7,t-6,title)
 return b.ax(f,l,x,t,w,h)
def style():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'axes.labelpad':2,'xtick.major.pad':2,'ytick.major.pad':2,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','axes.unicode_minus':True})
def near_peak_definitions():
 import matplotlib.patches as patches
 pkg=legacy/'Figure_S01'/'input'
 z=read(pkg/'lifecycle_schematic.csv')
 k=json.loads((pkg/'lifecycle_schematic.json').read_text())
 x,y=z.hours,z.wind_kt
 f,axes=plt.subplots(3,1,figsize=(180/25.4,139/25.4),sharex=True,sharey=True)
 f.subplots_adjust(left=.105,right=.98,top=.955,bottom=.10,hspace=.43)
 titles=['Yokoyama & Takayabu (2008)','Song et al. (2024)','Near-peak departures and returns in this study']
 for a,letter,title in zip(axes,'abc',titles):
  a.plot(x,y,marker='o',ms=2.8,lw=1.2,c='#33454F',zorder=3)
  a.set(ylabel='Wind speed (kt)',ylim=(30,115),xlim=(-2,81),yticks=[35,60,85,100],xticks=[0,12,24,36,48,60,72])
  a.set_title(letter+'   '+title,loc='left',fontsize=9.5,pad=4)
 for a,threshold,intervals,label in zip(axes,[.8*k['lmi_kt'],k['lmi_kt']-5,k['I090_kt']], [[(36,48),(60,66)],[(36,42)],k['episode_ranges_h']],['>80% of peak intensity','Within 5 kt of LMI','90% of the gain from initial wind to LMI']):
  a.axhline(threshold,ls='--',lw=.75,c='#82969F',zorder=0)
  a.text(.035,.89,label,transform=a.transAxes,va='center',fontsize=8)
  for low,high in intervals:a.axvspan(low,high,color='#428C79',alpha=.18,lw=0,zorder=0)
 a=axes[0];a.axvline(54,c='#8B7158',ls=':',lw=.8)
 a.text(54,46,'Segment restart',fontsize=8,ha='center')
 axes[1].text(55,48,'First continuous mature interval',fontsize=8,ha='center')
 axes[2].text(29,49,'Episode containing first LMI',fontsize=8,color='#33454F')
 axes[2].text(63,53,'Return',fontsize=8,color='#33454F',ha='center')
 axes[2].set_xlabel('Time from 35-kt start (h)')
 return f

def thresholds():
 z=read(ROOT/'inputs/counts.csv')
 vals=np.array([[int(b.row(z,escalation_progress=t,min_peak_gain=g,maturity_progress=m).n_valid) for g in [15,20] for m in [.85,.9,.95]] for t in [.2,.25,.3]])
 f=figure(72);a=rect(f,'a','Eligible cyclones',22,12,135,29)
 rgba=plt.cm.Blues(np.linspace(0,1,256));rgba[:,:3]=.52*rgba[:,:3]+.48
 cmap=ListedColormap(rgba);norm=Normalize(2120,2340);a.imshow(vals,cmap=cmap,norm=norm,aspect='auto')
 a.set(yticks=range(3),yticklabels=['0.20','0.25','0.30'],xticks=range(6),xticklabels=[f'{m:.2f} / {g}' for g in [15,20] for m in [.85,.9,.95]],ylabel='Progress threshold',xlabel='Near-peak threshold / minimum gain (kt)')
 for i,j in np.ndindex(vals.shape):a.text(j,i,str(vals[i,j]),ha='center',va='center')
 b.CB(f,'scale',65,58,47,norm,cmap,[2120,2175,2230,2285,2340],'Number of cyclones')
 return f
def radial():
 # Retain the two selected native scientific objects, deleting the full sector block.
 import inspect
 body=inspect.getsource(b.S7)
 i=body.index(' for j,(field,lab,u)');j=body.index(" title(f,'d'",i)
 body=(body[:i]+body[j:]).replace('def S7(d):','def selected_radial(d):').replace('fig(162)','fig(100)').replace("'d',7,126","'b',7,62").replace("'d',34,131","'b',34,68").replace("'cb_d',57,148","'cb_b',57,85")
 ctx=vars(b).copy();exec(body,ctx)
 return ctx['selected_radial'](data)
def histories():
 f=figure(72)
 for j,(k,title,label,col) in enumerate([('midRH','Mid-level humidity','Relative humidity (%)','#428C79'),('shear200_850','Deep-layer shear','Shear (m s⁻¹)','#82649A'),('PI','Potential intensity','PI (m s⁻¹)',b.BLUE)]):
  a=rect(f,'abc'[j],title,15.7+57*j,16,39.3,38)
  z=data['history'];z=z[z.metric.eq(k)&z.radial.eq('200-800')].sort_values('relative_hour')
  assert z.relative_hour.tolist()==list(range(-24,25,6))
  b.zero(a,True);b.ci(a,z.relative_hour.to_numpy(float),z.estimate.to_numpy(float),z.lower.to_numpy(float),z.upper.to_numpy(float),col)
  a.set(xlim=(-25,25),xticks=[-24,-12,0,12,24],ylabel=label)
 f.text(.50,.055,'Time from trough (h)',ha='center')
 return f
def forest(a,y,e,lo,hi,col):
 assert lo<=e<=hi
 a.errorbar(e,y,xerr=[[e-lo],[hi-e]],fmt='o',ms=4,lw=1.1,capsize=2.2,color=col)
def prepost():
 f=figure(73); axes=[rect(f,l,t,15+57*j,14,39,39) for j,(l,t) in enumerate(zip('abc',['Outward exchange','Gross outward','Net inward']))]
 q=read(legacy/'Figure_S23/inputs/direct/paired_flux_components.csv')
 q=q[q.cohort.eq('original_post80')&q.window.eq('0_24h')&q.factor.eq(1)&q.sector.eq('all')&q.layer.eq('all')&q.kind.eq('out')].set_index('quantity').loc[['humidity','relative_motion']]
 y=q.estimate.to_numpy()/1e12; a=axes[0]
 a.bar(range(2),y,.5,color=[b.GREEN,b.PURPLE]);a.errorbar(range(2),y,yerr=[y-q.low/1e12,q.high/1e12-y],fmt='none',color=b.INK,lw=1,capsize=2)
 a.set(xticks=range(2),xticklabels=['Humidity','Relative\nmotion'],ylabel='Contrast (10¹² kg)');b.zero(a)
 d=read(legacy/'Figure_S23/individual/common_prepost_paired_change_contrasts.csv')
 for a,k,col in zip(axes[1:],['out','net'],[b.PURPLE,b.GREEN]):
  sub=d[d.kind.eq(k)];assert len(sub)==sub.sid.nunique()==78
  x=sub.D_pre.to_numpy()/1e12;y=sub.D_post.to_numpy()/1e12
  lim=np.array([min(x.min(),y.min()),max(x.max(),y.max())]);lim+=np.array([-1,1])*np.ptp(lim)*.07
  a.plot(lim,lim,'--',lw=.7,c=b.GRAY);b.zero(a);b.zero(a,True);a.scatter(x,y,s=13,c=col,alpha=.68,linewidths=.3,edgecolors='white')
  a.set(xlim=lim,ylim=lim,xlabel='Pre-event (10¹² kg)',ylabel='Post-event (10¹² kg)')
  stat=read(legacy/'Figure_S23/inputs/direct/common_prepost.csv')
  pre=b.row(stat,kind=k,quantity='D_pre');post=b.row(stat,kind=k,quantity='D_post')
  assert pre.n==post.n==78
  a.errorbar(pre.estimate/1e12,post.estimate/1e12,xerr=[[(pre.estimate-pre.low)/1e12],[(pre.high-pre.estimate)/1e12]],yerr=[[(post.estimate-post.low)/1e12],[(post.high-post.estimate)/1e12]],fmt='D',color=b.INK,capsize=2,ms=3.7,lw=.9)
  a.xaxis.set_major_locator(MaxNLocator(4));a.yaxis.set_major_locator(MaxNLocator(5))
 return f
def sampling():
 f=figure(149);q=read(select/'data/s22_altlevels.csv')
 labs=q.window.drop_duplicates().tolist()
 mat=np.array([[b.row(q,pressure_hPa=pr,window=k).estimate for k in labs] for pr in [500,850]])
 a=rect(f,'a','Dry APE paired difference by pressure and scale',30,11,105,21)
 lim=max(abs(mat.flatten()));im=a.imshow(mat,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
 a.set(xticks=range(3),xticklabels=labs,yticks=[0,1],yticklabels=['500','850'],ylabel='Pressure (hPa)')
 for i,j in np.ndindex(mat.shape):
  rgba=im.cmap(im.norm(mat[i,j]));lum=np.dot(rgba[:3],[.2126,.7152,.0722])
  a.text(j,i,f'{mat[i,j]:.3f}'.replace('-','−'),color='white' if lum<.49 else 'black',ha='center',va='center')
 cb=f.colorbar(im,cax=b.ax(f,'cb',141,11,2.6,21));cb.set_label('J kg⁻¹');cb.locator=MaxNLocator(4);cb.update_ticks()
 d=read(ROOT/'inputs/S10c_common_support_direct_sensitivities.csv');d['level']=d.level.astype(str)
 motion=read(legacy/'Figure_S26/inputs/S10e_motion_intensity_targets.csv')
 for j,(k,title) in enumerate(zip(['KE','APE','C_A_to_K','vertical_input_K_Wkg'],['Horizontal KE','Dry APE','APE → KE conversion','Layer KE input'])):
  a=rect(f,'bcde'[j],title,30+84*(j%2),55+48*(j//2),48,28)
  s=d[d.metric.eq(k)&d.window.eq(1)&d.level.eq('500-850' if j==3 else '700')]
  if 'region' in s:s=s[s.region.eq('r0_300')]
  if 'scheme' in s:s=s[s.scheme.eq('main')]
  factor=1e4 if j>=2 else 1.
  for i,measure in enumerate(['input_shift','radius_500']):
   r=b.row(s,sensitivity=measure);forest(a,i,r.alternative_minus_main*factor,r.direct_difference_lower*factor,r.direct_difference_upper*factor,b.INK)
  r=b.row(motion,metric=k);forest(a,2,r.adjustment*factor,r.adjustment_lower*factor,r.adjustment_upper*factor,b.INK)
  b.zero(a,True);a.set(yticks=range(3),yticklabels=['Input shifted\n16 d','0–500 km\nregion','Wind/speed\nadjustment'],ylim=(2.5,-.5),xlabel='Contrast change ('+('10⁻⁴ W kg⁻¹' if j>=2 else 'J kg⁻¹')+')')
  a.xaxis.set_major_locator(MaxNLocator(4))
 return f
def rise_moisture():
 f=figure(117);pkg=legacy/'Figure_S28/inputs'
 ds=read(pkg/'RISE_MOISTURE_PROCESS_states_source.csv');di=read(pkg/'RISE_MOISTURE_PROCESS_integrals_source.csv')
 for j,(r,region) in enumerate([('inner','100–300 km'),('outer','300–800 km')]):
  a=rect(f,'ab'[j],'Rise integrals · '+region,17+85*j,14,59,30)
  q=di[di.radial.eq(r)&di.estimand.eq('rise_rate_integral')&di['sample'].eq('all_events')].set_index('metric').loc[['horizontal_advection','vertical_advection','residual','moving_tendency']]
  y=q.estimate.to_numpy();a.bar(range(4),y,.58,color=['#7C8C95','#428C79','#AB685E','white'],edgecolor=['none','none','none','#233A46'],linewidth=.85);a.set_ylim(-5.1,5.7)
  a.errorbar(range(4),y,yerr=[y-q.lower,q.upper-y],fmt='none',ecolor=b.INK,capsize=2,lw=.8);b.zero(a)
  a.set(ylabel='Integral (g kg⁻¹)',xticks=range(4),xticklabels=['Horiz.\nadv.','Vert.\nadv.','Residual','Moving\ntendency'])
  a=rect(f,'cd'[j],'Wind-matched humidity · '+region,17+85*j,71,59,30)
  t=ds[ds.radial.eq(r)];assert t.sid.nunique()==len(t)==108
  x=t.matched_Resolution.to_numpy();y=t.matched_rise.to_numpy();lo=np.floor(ds[['matched_Resolution','matched_rise']].min().min());hi=np.ceil(ds[['matched_Resolution','matched_rise']].max().max());lim=[lo,hi]
  a.plot(lim,lim,'--',c=b.GRAY,lw=.8);a.scatter(x,y,s=13,alpha=.59,c=(b.BLUE,b.GREEN)[j],edgecolors='none')
  a.set(xlim=lim,ylim=lim,xlabel='Decay q (g kg⁻¹)',ylabel='Reintensification q (g kg⁻¹)')
 return f
def integrals():
 f=figure(72);d=read(select/'data/plot_source.csv')
 for j,(key,title,labels) in enumerate([('c','Conversion and\nvertical input',['APE → KE','KE input']),('a','KE canonical\ntransfer',['Horiz.','Vert.','Total']),('b','Dry APE canonical\ntransfer',['Horiz.','Vert.','Total'])]):
  a=rect(f,'abc'[j],title,15.5+57*j,17,39,35)
  t=d[d.panel.eq(key)].sort_values('row').iloc[:len(labels)]
  assert t.n_sid.eq(122).all() and t.n_events.eq(132).all() and t.unit.eq('J kg-1').all()
  for i,(_,r) in enumerate(t.iterrows()):
   col=([b.ORANGE,b.BLUE] if j==0 else [b.BLUE,b.GREEN,b.INK])[i];a.errorbar(i,r.estimate,yerr=[[r.estimate-r.ci_lower],[r.ci_upper-r.estimate]],color=col,fmt='o',ms=4,capsize=2,lw=1)
  b.zero(a);a.set(xticks=range(len(labels)),xticklabels=labels,xlim=(-.5,len(labels)-.5),ylabel='Integral (J kg⁻¹)')
 return f
def save(f,n):
 name=f'Figure_S{n:02d}';f.canvas.draw();renderer=f.canvas.get_renderer()
 geom=[[float(x*25.4/f.dpi) for x in a.get_window_extent(renderer).bounds] for a in f.axes]
 png=out/(name+'.png');f.savefig(png,dpi=dpi);plt.close(f)
 with Image.open(png) as im: rgb=im.convert('RGB')
 rgb.save(png,dpi=(dpi,dpi));rgb.save(out/(name+'.tiff'),dpi=(dpi,dpi),compression='tiff_lzw')
 print('Exported',name,flush=True)
for n in args.figures:
 style()
 funcs={1:near_peak_definitions,2:thresholds,3:radial,4:lambda:b.S8(data),5:prepost,6:sampling,7:histories,8:rise_moisture,9:integrals,10:lambda:b.S14(data),11:lambda:b.S15(data)}
 save(funcs[n](),n)

