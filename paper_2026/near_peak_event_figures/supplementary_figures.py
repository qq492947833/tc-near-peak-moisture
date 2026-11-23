"""Rebuild the 10 selected supplementary figures from the v2.1.5 plotting inputs."""
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
parser.add_argument('--diagnostics',type=Path,help='Optional directory for figure geometry, previews and the plotted threshold rows.')
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--figures',type=int,nargs='+',default=list(range(1,11)))
parser.add_argument('--out',type=Path,default=ROOT/'outputs')
args=parser.parse_args()
repo=args.repo/'paper_2026'
legacy=repo/'lifecycle_figures/figures'
select=repo/'selected_evidence'
out=args.out
out.mkdir(exist_ok=True,parents=True)

dpi=900
qa=args.diagnostics
if qa is not None: qa.mkdir(exist_ok=True,parents=True)
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
b.W=180

def title_at_base_size(f, key, x, y, label):
 b.txt(f, x, y, key, fontsize=9, fontweight='bold')
 b.txt(f, x+4.7, y, label, fontsize=9, linespacing=1.0)

b.title=title_at_base_size
def rect(f,l,title,x,t,w,h):
 b.title(f,l,x-7,t-6,title)
 return b.ax(f,l,x,t,w,h)
def style():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'axes.labelpad':2,'xtick.major.pad':2,'ytick.major.pad':2,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','axes.unicode_minus':True})
def thresholds():
 path=ROOT/'inputs/threshold_duration_summary.csv'
 assert hashlib.sha256(path.read_bytes()).hexdigest()=='c36ae8eef6ac068c168ee7ed81426ee6c8e7f84827d5a7cbbbddc5033f521b3d'
 s=read(path)
 f=figure(82); b.txt(f,90,4.5,'Sensitivity to the near-peak threshold',ha='center',fontsize=9)
 b.leg(f,b.fills([b.BLUE,b.ORANGE]),['Later near-peak occupancy','Weakening–recovery bridge'],90,8,2)
 rec=[]
 for j,(cut,prefix,nid) in enumerate([(0.85,'beta085',150),(.95,'beta095',109)]):
  a=rect(f,'ab'[j],f'Near-peak threshold {cut:.2f}',18+89*j,27,61,39)
  for i,(metric,col) in enumerate([('L',b.BLUE),('B',b.ORANGE)]):
   keys=[prefix+'_'+t for t in ['native_multi','fixed90_multi_main','fixed90_multi_alternate']]
   q=s[s.statistic.eq('mean')&s.metric.eq(metric)].set_index('scope').loc[keys]
   assert list(q.n_sid)==[nid,123,123] and q.B_valid.eq(2000).all() and q.unit.eq('h').all()
   x=np.arange(3)+(i-.5)*.34;a.bar(x,q.estimate,.29,color=col)
   a.errorbar(x,q.estimate,yerr=[q.estimate-q.ci_low,q.ci_high-q.estimate],fmt='none',ecolor=b.INK,capsize=2,lw=1);rec.append(q.reset_index())
  a.set(xticks=range(3),xticklabels=['Own subset','Main 0.90','Alternative'],ylim=(0,46),ylabel='Mean duration (h)')
 if qa is not None: pd.concat(rec,ignore_index=True).to_csv(qa/'S01_plotted_rows.csv',index=False,float_format='%.17g')
 return f
def radial():
 from matplotlib.patches import Wedge
 from matplotlib.colors import TwoSlopeNorm
 f=figure(113);b.title(f,'a',7,4,'Pre-departure specific humidity, 700 hPa: Temporary − decay')
 norm=TwoSlopeNorm(0,-.6,.6)
 for j,(k,head) in enumerate(zip(['q700_raw_common','q700_climatology','q700_anomaly'],['Observed','Background','Anomaly'])):
  center=40+50*j;ax=b.ax(f,'a'+str(j),center-14,13,28,28);b.txt(f,center,10,head,ha='center')
  vv=[]
  for reg,rad,width in zip(b.REG,[300,800],[200,500]):
   row=b.row(data['annulus_state'],field=k,region=reg,sector='whole',estimand='pre_mean');assert row.n_sid==117
   vv.append(row.estimate);ax.add_patch(Wedge((0,0),rad,0,360,width=width,facecolor=b.LIGHT(norm(row.estimate)),edgecolor='white',lw=.6))
  b.rings(ax);ax.set(xlim=(-850,850),ylim=(-850,850))
  label=' / '.join(f'{v:+.2f}'.replace('-','−') for v in vv);b.txt(f,center,43,label,ha='center',fontsize=8.5)
 b.txt(f,90,49,'100–300 / 300–800 km',ha='center',fontsize=8.5)
 cbar_left, cbar_width = 60, 60  # mm in the final 180 mm figure
 b.CB(f,'cb_a',cbar_left,53,cbar_width,norm,b.LIGHT,np.linspace(-.6,.6,5),'Specific humidity difference (g kg⁻¹)')
 b.title(f,'b',7,72,'Moisture-budget integral, 0–24 h: Temporary − decay');a=b.ax(f,'b',31,78,127,13);cats=['horizontal_advection','vertical_advection','residual','moving_tendency']
 q=np.array([[b.row(data['budget'],region=r,field=k).estimate for k in cats] for r in b.REG]);assert data['budget'].n_sid.eq(105).all()
 a.pcolormesh(np.arange(5)-.5,np.arange(3)-.5,q,cmap=b.LIGHT,norm=TwoSlopeNorm(0,-1,1),edgecolors='white',linewidth=1.)
 a.set(xticks=range(4),xticklabels=['Horizontal','Vertical','Residual','Net tendency'],yticks=range(2),yticklabels=['100–300 km','300–800 km'],ylim=(1.5,-.5));a.tick_params(length=0)
 for sp in a.spines.values():sp.set_visible(False)
 for i,j in np.ndindex(q.shape):a.text(j,i,f'{q[i,j]:.2f}'.replace('-','−'),va='center',ha='center')
 b.CB(f,'cb_b',cbar_left,99,cbar_width,TwoSlopeNorm(0,-1,1),b.LIGHT,np.linspace(-1,1,5),'Integrated difference (g kg⁻¹)')
 return f


def histories():
 f=figure(72)
 for j,(k,title,label,col) in enumerate([('midRH','Mid-level humidity','Relative humidity (%)','#428C79'),('shear200_850','Deep-layer shear','Shear (m s⁻¹)','#82649A'),('PI','Potential intensity','PI (m s⁻¹)',b.BLUE)]):
  a=rect(f,'abc'[j],title,15.7+57*j,16,39.3,38)
  z=data['history'];z=z[z.metric.eq(k)&z.radial.eq('200-800')].sort_values('relative_hour')
  assert z.relative_hour.tolist()==list(range(-24,25,6))
  b.zero(a,True);b.ci(a,z.relative_hour.to_numpy(float),z.estimate.to_numpy(float),z.lower.to_numpy(float),z.upper.to_numpy(float),col)
  a.set(xlim=(-25,25),xticks=[-24,-12,0,12,24],ylabel=label,xlabel='Time from trough (h)')

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
  a.set(xlim=lim,ylim=lim,xlabel='Pre-event contrast (10¹² kg)',ylabel='Post-event contrast (10¹² kg)')
  stat=read(legacy/'Figure_S23/inputs/direct/common_prepost.csv')
  pre=b.row(stat,kind=k,quantity='D_pre');post=b.row(stat,kind=k,quantity='D_post')
  assert pre.n==post.n==78
  a.errorbar(pre.estimate/1e12,post.estimate/1e12,xerr=[[(pre.estimate-pre.low)/1e12],[(pre.high-pre.estimate)/1e12]],yerr=[[(post.estimate-post.low)/1e12],[(post.high-post.estimate)/1e12]],fmt='D',color=b.INK,capsize=2,ms=3.7,lw=.9)
  a.xaxis.set_major_locator(MaxNLocator(4));a.yaxis.set_major_locator(MaxNLocator(5))
 return f
def sampling():
 f=figure(155);q=read(select/'data/s22_altlevels.csv')
 labs=q.window.drop_duplicates().tolist()
 mat=np.array([[b.row(q,pressure_hPa=pr,window=k).estimate for k in labs] for pr in [500,850]])
 a=rect(f,'a','Pre-departure dry APE: Temporary − decay',27,11,138,21)
 lim=max(abs(mat.flatten()));im=a.imshow(mat,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
 a.set(xticks=range(3),xticklabels=labs,yticks=[0,1],yticklabels=['500','850'],ylabel='Pressure (hPa)')
 for i,j in np.ndindex(mat.shape):
  rgba=im.cmap(im.norm(mat[i,j]));lum=np.dot(rgba[:3],[.2126,.7152,.0722])
  a.text(j,i,f'{mat[i,j]:.3f}'.replace('-','−'),color='white' if lum<.49 else 'black',ha='center',va='center')
 cb=f.colorbar(im,cax=b.ax(f,'cb',68,42,56,1.8),orientation='horizontal');cb.set_label('Dry APE difference (J kg⁻¹)');cb.locator=MaxNLocator(4);cb.update_ticks()
 d=read(ROOT/'inputs/S10c_common_support_direct_sensitivities.csv');d['level']=d.level.astype(str)
 motion=read(legacy/'Figure_S26/inputs/S10e_motion_intensity_targets.csv')
 for j,(k,title) in enumerate(zip(['KE','APE','C_A_to_K','vertical_input_K_Wkg'],['Horizontal KE','Dry APE','APE → KE conversion','Layer KE input'])):
  a=rect(f,'bcde'[j],title,27+85*(j%2),67+45*(j//2),53,24)
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
  y=q.estimate.to_numpy();a.bar(range(4),y,.58,color=['#7C8C95','#428C79','#AB685E','white'],edgecolor=['none','none','none','#233A46'],linewidth=.85)
  if r == 'outer':
   assert q.n_sid.eq(121).all() and q.n_events.eq(130).all()
   assert q.lower.ge(-1).all() and q.upper.le(1).all()
   a.set_ylim(-1, 1)
   a.set_yticks([-1, -.5, 0, .5, 1], labels=['−1', '−0.5', '0', '0.5', '1'])
  else:
   a.set_ylim(-5.1,5.7)
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
 for j,(key,title,labels) in enumerate([('c','Conversion and\nvertical input',['APE → KE','Vertical KE\ninput']),('a','KE canonical\ntransfer',['Horiz.','Pressure','Total']),('b','Dry APE canonical\ntransfer',['Horiz.','Pressure','Total'])]):
  a=rect(f,'abc'[j],title,15.5+57*j,17,39,35)
  t=d[d.panel.eq(key)].sort_values('row').iloc[:len(labels)]
  assert t.n_sid.eq(122).all() and t.n_events.eq(132).all() and t.unit.eq('J kg-1').all()
  for i,(_,r) in enumerate(t.iterrows()):
   col=([b.ORANGE,b.BLUE] if j==0 else [b.BLUE,b.GREEN,b.INK])[i];a.errorbar(i,r.estimate,yerr=[[r.estimate-r.ci_lower],[r.ci_upper-r.estimate]],color=col,fmt='o',ms=4,capsize=2,lw=1)
  b.zero(a);a.set(xticks=range(len(labels)),xticklabels=labels,xlim=(-.5,len(labels)-.5),ylabel='Integral (J kg⁻¹)')
 return f
def legend(f,handles,labels,center,y,n,size=8.25):
 return f.legend(handles,labels,loc='upper center',bbox_to_anchor=(center/180,1-y/f.ph),ncol=n,frameon=False,borderaxespad=0,handlelength=1,handletextpad=.30,columnspacing=.63,labelspacing=.2,fontsize=size)
def move_ax(f,key,x,top,w,h):
 a=f.boxes[key][0];a.set_position((x/180,1-(top+h)/f.ph,w/180,h/f.ph));f.boxes[key]=(a,x,top,w,h);return a
def rowlegends():
 f=b.S8(data)
 for lg in list(f.legends):lg.remove()
 for j,l in enumerate('abc'):
  x=16+58*j;move_ax(f,l,x,23,39,43)
  for k,t in enumerate(f.texts[j*2:j*2+2]):t.set_position(((x-7+4.7*k)/180,1-6/f.ph))
 legend(f,b.fills([b.BLUE,b.ORANGE]),['Track','Fixed'],35.5,13,2)
 legend(f,b.lines([b.BLUE,b.ORANGE]),['Temporary','Decay'],93.5,13,2)
 legend(f,b.fills([b.GREEN,'#BC567D','#333D49']),['Humidity','Motion','Total'],151.5,13,3)
 a=f.boxes['c'][0];layers=['500-600_hPa','600-700_hPa','700-850_hPa','850-925_hPa']
 for i,k in enumerate(['humidity','relative_motion','difference']):
  q=data['layer_flux'];z=q[q.quantity.eq(k)].set_index('layer').loc[layers]
  a.errorbar(np.arange(4)+(i-1)*.24,z.estimate/1e12,yerr=[(z.estimate-z.low)/1e12,(z.high-z.estimate)/1e12],fmt='none',capsize=1.5,color=b.INK,lw=.65)
 return f
def recovery():
 f=b.S14(data)
 for lg in list(f.legends):lg.remove()
 for j,l in enumerate('abc'):move_ax(f,l,17+58*j,16,38,40)
 handles=[plt.Line2D([],[],color=col,marker=m,lw=.9,ms=4) for m,col in zip(['o','s','^'],[b.INK,b.GRAY,b.PURPLE])]
 legend(f,handles,['G-formula','One-step','Density weighting'],65,73,3)
 legend(f,b.fills([b.INK,'#C8D1D8']),['Changed exposure','Ratio ESS / n'],150,71,1)
 return f
def outcomes():
 f=b.S15(data)
 for lg in list(f.legends):lg.remove()
 a,c1,c2=[move_ax(f,l,15+60*j,16,41,40) for j,l in enumerate('abc')]
 legend(f,b.fills([b.GREEN,'#88959B','#E1E6E8']),['Return','Nonreturn','Unknown'],35.5,73,3)
 legend(f,b.lines([b.BLUE,b.ORANGE]),['Shear −2 m s⁻¹','RH +5 pp'],125.5,73,2)
 return f
def save(f,n):
 name=f'Figure_S{n:02d}';f.canvas.draw();r=f.canvas.get_renderer()
 boxes={k:[float(v*25.4/f.dpi) for v in a.get_window_extent(r).bounds] for k,(a,*_) in f.boxes.items()}
 def align(labs,dims):
  assert all(np.ptp([boxes[k][j] for k in labs])<=.3 for j in dims),(name,labs,boxes)
 if n in [1,3,4,6,8,9,10]:align('ab' if n==1 else 'abc',[1,2,3])
 if n==5:
  align('bc',[1,2,3]);align('de',[1,2,3]);align('bd',[0,2]);align('ce',[0,2]);assert abs(boxes['a'][0]-boxes['b'][0])<.3 and abs(boxes['a'][0]+boxes['a'][2]-boxes['c'][0]-boxes['c'][2])<.3
 if n==7:align('ab',[1,2,3]);align('cd',[1,2,3]);align('ac',[0,2]);align('bd',[0,2])
 if n==2:
  align(['a0','a1','a2'],[1,2,3]);align(['cb_a','cb_b'],[0,2,3]);assert abs(boxes['cb_a'][0]+boxes['cb_a'][2]/2-boxes['a1'][0]-boxes['a1'][2]/2)<.3
 if qa is not None: (qa/(name+'_geometry.json')).write_text(json.dumps(boxes,indent=2),encoding='utf8')
 if qa is not None:
  f.savefig(qa/(name+'.pdf'));f.savefig(qa/(name+'_preview.png'),dpi=145)
 png=out/(name+'.png');f.savefig(png,dpi=dpi);plt.close(f)
 with Image.open(png) as im:rgb=im.convert('RGB')
 rgb.save(png,dpi=(dpi,dpi));tif=out/(name+'.tiff');rgb.save(tif,dpi=(dpi,dpi),compression='tiff_lzw')
 with Image.open(tif) as im:assert im.mode=='RGB' and im.tobytes()==rgb.tobytes()
 print('Exported',name,rgb.size,flush=True)
for n in args.figures:
 style()
 funcs={1:thresholds,2:radial,3:rowlegends,4:prepost,5:sampling,6:histories,7:rise_moisture,8:integrals,9:recovery,10:outcomes}
 save(funcs[n](),n)

