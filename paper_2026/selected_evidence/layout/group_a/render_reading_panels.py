"""Render reading-SI panels 2–6 from the versioned frozen plotting inputs."""
from pathlib import Path
import argparse, os, json, hashlib, sys
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter
from PIL import Image
def cli():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dpi',type=int,default=900);p.add_argument('--figures',nargs='+',type=int,default=[2,3,4,5,6]);return p.parse_args()
B='#167B9E';O='#CC7445';INK='#233A46';C=['#437F9C','#2C9F9A','#C89D38','#C05A52'];ST=['D','E','F','Dec'];basins=['NA','EP','WP','NI','SI','SP'];cm=LinearSegmentedColormap.from_list('dry_signed',['#DBAA8D','#FAFAF8','#9FC3D2']);fmt=FuncFormatter(lambda x,pos:f'{x:g}'.replace('-','−'))
def style():
 plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'legend.frameon':False,'savefig.facecolor':'white','figure.facecolor':'white','axes.axisbelow':True})
def set_title(a,letter,title):
 a.set_gid(letter);a.set_title(title,loc='left',pad=6,fontsize=9.5,weight='normal')
 a.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-15,6),textcoords='offset points',va='baseline',ha='left',weight='bold',fontsize=10,annotation_clip=False)
def grid(a,axis='y'):a.grid(axis=axis,color='#E4E9EB',lw=.5)
def rowplot(f,upper,h,left=.80,right=4.02,w=2.52):
 def axes(x):return f.add_axes((x/6.72,upper/f.get_figheight(),w/6.72,h/f.get_figheight()))
 return axes(left),axes(right)
def err(a,x,y,lo,hi,vert=False,color=B,marker='o',label=None):
 vals=y if vert else x;dx=np.array([np.asarray(vals)-lo,hi-np.asarray(vals)]).reshape(2,-1);kw={'yerr':dx} if vert else {'xerr':dx}
 return a.errorbar(x,y,**kw,fmt=marker,ms=4,color=color,lw=1,capsize=2,label=label,zorder=3)
def legend_rows(f,handles,y,n):
 f.legend(handles=handles,loc='center',bbox_to_anchor=(.52,y/f.get_figheight()),ncol=n,handlelength=1.5,handletextpad=.4,columnspacing=1.1)
def ECDF(a,values,color,label):
 x=np.sort(np.asarray(values));a.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=color,label=label,lw=1.15)
def build2(read):
 f=plt.figure(figsize=(6.72,6.65));a,b=rowplot(f,4.78,1.39);c,d=rowplot(f,2.55,1.35);e=f.add_axes([.8/6.72,.43/6.65,5.74/6.72,1.37/6.65])
 z=read('duration');q=read('fractions').set_index('basin_genesis').loc[basins];assert len(z)==2340;left=np.zeros(6)
 for i,k in enumerate(['development_duration_hours','escalation_duration_hours','maturity_duration_hours','resolution_duration_hours']):
  ECDF(a,z[k],C[i],ST[i]);b.barh(range(6),q[k],left=left,color=C[i],height=.65);left+=q[k].to_numpy()
 a.set(xlim=(0,None),ylim=(0,1),xlabel='Stage duration (h)',ylabel='Cumulative fraction');grid(a);b.set(yticks=range(6),yticklabels=basins,xlim=(0,1),ylim=(5.5,-.5),xlabel='Mean fraction of lifetime');set_title(a,'a','Stage durations');set_title(b,'b','Stage composition by basin')
 legend_rows(f,list(a.lines),6.53,4)
 z=read('phases');left=np.zeros(3);hh=[]
 for st,col in zip(['D','E','F','Res'],C):
  q=z[z.stage.eq(st)].set_index('segment').loc[['pre','middle','post']];v=q.mean_covered_h.to_numpy();u=q.mean_uncovered_h.to_numpy();c.barh(range(3),v,left=left,color=col,height=.6);left+=v
  mask=u>1e-9;c.barh(np.arange(3)[mask],u[mask],left=left[mask],height=.6,color='white',edgecolor=col,lw=.8);left+=u;hh.append(Patch(facecolor=col,label=ST[len(hh)]))
 legend_rows(f,hh+[Patch(facecolor='white',edgecolor='#819097',label='Uncovered')],4.24,5)
 c.set(yticks=range(3),yticklabels=['Before first\nLMI','First–last\nLMI','After last\nLMI'],xlim=(0,82),xlabel='Mean elapsed time (h)');c.invert_yaxis();grid(c,'x');set_title(c,'c','Peak-centred stage allocation')
 z=read('cdf');assert len(z)==2340;d.step(z.A,z.empirical_CDF,where='post',color=INK,lw=1.15);d.axvline(0,lw=.6,color='0.6');d.set(xlim=(-1,1),ylim=(0,1),xlabel='Signed asymmetry A',ylabel='Cumulative fraction');grid(d);set_title(d,'d','Lifecycle time asymmetry')
 z=read('trends').set_index('metric').loc[['D_h','E_h','F_h','Resolution_h']]
 for i,(_,row) in enumerate(z.iterrows()):err(e,row.slope,i,row.ci_low,row.ci_high,color=C[i],marker=['o','s','^','D'][i])
 e.axvline(0,color='0.55',lw=.7);e.set(xlim=(-7.5,2.8),yticks=range(4),yticklabels=ST,ylim=(3.5,-.5),xlabel='Stage-duration trend (h decade$^{-1}$)');grid(e,'x');set_title(e,'e','Stage-duration trends, 1982–2023');return f
def build3(read):
 f=plt.figure(figsize=(6.72,5.52));a,b=rowplot(f,3.49,1.55);c,d=rowplot(f,1.10,1.59)
 data=read('windows');cols=['#354F5E','#8B9EA8',B,O,'#629786','#9574A5'];labs=['R','Longest','U','F','Later L','Bridge B']
 for j,ax in enumerate([a,b]):
  z=data[data.main_complete & (data.multi if j else True)];assert len(z)==[2340,123][j]
  for k,color,lab in zip(['R','R_long','U','F','L','B'],cols,labs):ECDF(ax,z['T_'+k+'_h'],color,lab)
  ax.set(xlabel='Covered duration (h)',ylabel='Cumulative fraction',ylim=(0,1));grid(ax);set_title(ax,'ab'[j],['All common histories','Recurrent subset'][j])
 legend_rows(f,list(a.lines),3.00,6)
 z=read('counts');vals=np.array([[int(z[z.escalation_progress.eq(t)&z.min_peak_gain.eq(g)&z.maturity_progress.eq(m)].n_valid.iloc[0]) for g in [15,20] for m in [.85,.90,.95]] for t in [.2,.25,.3]])
 rgba=plt.cm.Blues(np.linspace(0,1,256));rgba[:,:3]=.52*rgba[:,:3]+.48;im=c.imshow(vals,cmap=ListedColormap(rgba),aspect='auto',norm=Normalize(2120,2340))
 c.set(yticks=range(3),yticklabels=['0.20','0.25','0.30'],xticks=range(6),xticklabels=[str(m)+'\n'+str(g) for g in [15,20] for m in [.85,.9,.95]],ylabel='Intensification progress I',xlabel='Near-peak progress I / wind gain (kt)');set_title(c,'c','Valid cyclones')
 for i,j in np.ndindex(vals.shape):c.text(j,i,str(vals[i,j]),ha='center',va='center',fontsize=9)
 cb=f.colorbar(im,cax=f.add_axes([.80/6.72,.37/5.52,2.52/6.72,.07/5.52]),orientation='horizontal',ticks=[2120,2175,2230,2285,2340]);cb.set_label('Valid cyclones (count)',labelpad=1);cb.ax.tick_params(pad=1,length=2)
 z=read('sensitivity');scopes=[p+s for p in ['beta085_','beta095_'] for s in ['native_multi','fixed90_multi_main','fixed90_multi_alternate']]
 for j,(k,color) in enumerate(zip(['L','B'],[B,O])):
  q=z[z.metric.eq(k)&z.statistic.eq('mean')].set_index('scope').loc[scopes];x=np.arange(6)+(j-.5)*.32
  d.bar(x,q.estimate,.29,color=color,label=['Later L','Bridge B'][j]);d.errorbar(x,q.estimate,yerr=[q.estimate-q.ci_low,q.ci_high-q.estimate],fmt='none',ecolor=INK,capsize=2,lw=.8)
 d.set(xticks=range(6),xticklabels=['Own\nsubset','Main\n0.90','Alt.']*2,ylabel='Mean duration (h)',ylim=(0,52));set_title(d,'d','Threshold and subset sensitivity');grid(d)
 for j in [0,3]:
  d.plot([j-.38,j-.38,j+2.38,j+2.38],[-.22,-.25,-.25,-.22],transform=d.get_xaxis_transform(),clip_on=False,color=INK,lw=.65);d.text(j+1,-.28,['0.85','0.95'][j//3],ha='center',va='top',transform=d.get_xaxis_transform(),fontsize=9)
 d.legend(loc='upper right',ncol=2,fontsize=9,columnspacing=.6,handlelength=1,handletextpad=.35);return f

def build4(read):
 f=plt.figure(figsize=(6.72,4.55));a,b=rowplot(f,2.71,1.49);c,d=rowplot(f,.43,1.49)
 q=read('basins').set_index('basin').loc[basins];yy=np.arange(6);err(a,q.rate_per1000h,yy,q.rate_low,q.rate_high);a.set(yticks=yy,yticklabels=basins,ylim=(5.45,-.55),xlim=(-.08,2.9),xticks=[0,1,2],xlabel='Returns per 1,000 cyclone-hours');set_title(a,'a','Return rate per opportunity')
 order=['NA','EP','WP','SI'];q=read('annual').set_index('estimand').loc[order+['GLOBAL anchored-return SID fraction']];err(b,q.estimate,np.arange(5),q.lo,q.hi,color='#437F9C');b.axvline(0,c='.6',lw=.7);b.set(yticks=range(5),yticklabels=order+['Global'],ylim=(4.7,-.7),xlim=(-2.6,3.7),xticks=[-2,0,2],xlabel='Trend (pp decade$^{-1}$)');set_title(b,'b','Annual recurrence trends')
 z=read('envtrend');yy=np.arange(4)
 for j,key in enumerate(['actual_environment','fixed_reference_environment_state_unchanged','actual_minus_fixed_environment']):
  ax=c if j<2 else d;q=z[z.estimand.eq(key)].set_index('basin').loc[order];err(ax,q.estimate_pp_per_decade,yy+[-.13,.13,0][j],q.ci_low,q.ci_high,color=[B,O,B][j],marker=['o','s','o'][j],label=['Actual','Fixed environment',None][j]);ax.set(yticks=yy,yticklabels=order,ylim=(3.5,-.5))
 for ax in [c,d]:ax.axvline(0,color=INK,ls='--',lw=.7,zorder=0)
 c.set(xlim=(-18,8),xticks=[-15,-10,-5,0,5],xlabel='Probability trend (pp decade$^{-1}$)');d.set(xlim=(-2,6),xticks=[-2,0,2,4,6],xlabel='Actual − fixed (pp decade$^{-1}$)')
 set_title(c,'c','Actual and fixed-environment trends');set_title(d,'d','Environmental standardization contrast')
 f.legend(*c.get_legend_handles_labels(),ncol=2,loc='center',bbox_to_anchor=(.52,2.23/4.55),handlelength=1.3,handletextpad=.45,columnspacing=1.0);return f
def build5(read):
 f=plt.figure(figsize=(6.72,4.65));a,b=rowplot(f,2.85,1.42,left=1.07,right=4.45,w=2.09);c,d=rowplot(f,.48,1.51,left=1.07,right=4.45,w=2.09)
 z=read('contrasts');u=read('losses')
 for j,(ax,lower,var,name) in enumerate([(a,c,'X3_mid_rh_pct','RH'),(b,d,'X2_shear_ms','Shear')]):
  q=z[z.variable.eq(var)].set_index('contrast').loc[['D-pooledDE','E-pooledDE','F-R','Res-post_R_tail']]
  for y,(_,v) in enumerate(q.iterrows()):err(ax,v.estimate,y,v.ci_low,v.ci_high,color=['#437F9C','#428C79','#BD9145','#AB685E'][y])
  ax.axvline(0,color='#9aa6aa',lw=.75,zorder=0);ax.set(yticks=range(4),yticklabels=['D − pooled D/E','E − pooled D/E','F − R','Dec − post-R tail'],ylim=(3.5,-.5),xlabel='RH contrast (pp)' if not j else 'Shear contrast (m s$^{-1}$)');grid(ax,'x');ax.margins(x=.16);set_title(ax,'ab'[j],name+': stage contrasts')
  v=u[u.response.eq(var)].set_index('contrast').loc[['actual_DE-'+k for k in ['coarse','time_midpoint','continuous_progress','continuous_age']]]
  lower.bar(range(4),v.delta_rmse,.65,color=['#A6708C','#596572'][j]);lower.errorbar(range(4),v.delta_rmse,yerr=[v.delta_rmse-v.rmse_ci_low,v.rmse_ci_high-v.delta_rmse],fmt='none',ecolor=INK,capsize=2,lw=.9);lower.axhline(0,color='#738690',lw=.8);lower.set(xticks=range(4),xticklabels=['Pooled','Midpoint','Progress','Age'],ylabel='Actual − reference RMSE\n'+['(RH percentage points)','(m s$^{-1}$)'][j]);set_title(lower,'cd'[j],name+' prediction error');grid(lower)
 return f
def build6(read):
 f=plt.figure(figsize=(6.72,5.46));a,b=rowplot(f,3.53,1.41,left=1.02,right=4.31,w=2.22);c,d=rowplot(f,1.02,1.41,left=1.02,right=4.31,w=2.22);groups=['D_to_E','E_to_F'];labs=['D → E','E → F']
 def heat(ax,vals,ylabels,xlabels,lim,label,ybar):
  v=np.asarray(vals);im=ax.pcolormesh(np.arange(v.shape[1]+1)-.5,np.arange(v.shape[0]+1)-.5,v,cmap=cm,vmin=-lim,vmax=lim,edgecolors='white',lw=1.25);ax.set(yticks=range(v.shape[0]),yticklabels=ylabels,xticks=range(v.shape[1]),xticklabels=xlabels,ylim=(v.shape[0]-.5,-.5));ax.tick_params(length=0,pad=3)
  for spine in ax.spines.values():spine.set_visible(False)
  for i,j in np.ndindex(v.shape):ax.text(j,i,('%0.2f'%v[i,j]).rstrip('0').rstrip('.').replace('-','−'),ha='center',va='center',fontsize=9)
  q=ax.get_position();cb=f.colorbar(im,cax=f.add_axes([q.x0,ybar/f.get_figheight(),q.width,.065/f.get_figheight()]),orientation='horizontal',ticks=np.linspace(-lim,lim,5));cb.ax.xaxis.set_major_formatter(fmt);cb.ax.tick_params(pad=1,length=2);cb.set_label(label,labelpad=1);cb.outline.set_visible(False)
 z=read('humidity');z=z[z.field.eq('q700')&z.window.eq('-24_0')].set_index(['component','boundary']);v=[[z.loc[(k,g),'estimate'] for g in groups] for k in ['raw_change','climate_change','anomaly_change']]
 heat(a,v,['Observed','Background','Anomaly'],labs,.3,'Humidity change (g kg$^{-1}$)',2.98);set_title(a,'a','Humidity along the track')
 z=read('rh');z=z[z.scope.eq('primary25')].set_index(['variable','level_hpa','boundary']);v=[[z.loc[(k,lev,g),'estimate'] for lev in [700,500] for g in groups] for k in ['time_q_contribution','time_T_contribution','domain_symmetric','fixed_support_endpoint_change']]
 heat(b,v,['Humidity','Temperature','Geography','Total'],['700 hPa\nD → E','700 hPa\nE → F','500 hPa\nD → E','500 hPa\nE → F'],5,'RH change (pp)',2.98);set_title(b,'b','RH change: time and location')
 z=read('anomaly');z=z[z.field.eq('r700')&z.quantity.eq('anomaly_common')]
 for i,g in enumerate(groups):
  q=z[z.boundary.eq(g)].sort_values('lag_h');x=q.lag_h.to_numpy();c.fill_between(x,q.ci_low.to_numpy(),q.ci_high.to_numpy(),color=[B,O][i],alpha=.1,lw=0);c.plot(x,q.estimate,color=[B,O][i],marker=['o','s'][i],ms=2.6,lw=1.0,label=labs[i])
 c.axvline(0,color='#91A1A9',lw=.7,ls=':');c.set(xticks=[-24,-12,0,12,24],xlabel='Time from transition (h)',ylabel='RH anomaly (pp)');grid(c);set_title(c,'c','RH anomaly along the track');f.legend(*c.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.32,.32/5.46),ncol=2,handletextpad=.4,columnspacing=1.1,handlelength=1.4)
 z=read('pi').set_index(['component','boundary']);v=[[z.loc[(k,g),'estimate'] for g in groups] for k in ['PI','minus_V','headroom']]
 heat(d,v,['ΔPI','−ΔV','Δ(PI − V)'],labs,20,'Change (m s$^{-1}$)',.56);set_title(d,'d','Potential intensity and wind')
 return f
def main():
 args=cli();out=args.output;out.mkdir(exist_ok=True,parents=True);sources=json.loads((args.inputs/'sources.json').read_text())
 @lru_cache(None)
 def read(k):
  from io import BytesIO
  raw=(args.inputs/(k+'.csv')).read_bytes();assert hashlib.sha256(raw).hexdigest()==sources[k]['sha256'],k
  return pd.read_csv(BytesIO(raw),keep_default_na=False,na_values=['','NaN','nan'],float_precision='round_trip')
 style();qa={}
 for i in args.figures:
  f=globals()['build'+str(i)](read);f.canvas.draw();ax=[a for a in f.axes if a.get_gid()];geom={a.get_gid():[round(x*72,3) for x in a.get_window_extent().transformed(f.dpi_scale_trans.inverted()).bounds] for a in ax}
  for a,b in [('a','b'),('c','d')]:assert geom[a][1::2]==geom[b][1::2]
  for a,b in [('a','c'),('b','d')]:assert geom[a][::2]==geom[b][::2]
  name='Figure_S%02d'%i;png=out/(name+'.png');tiff=out/(name+'.tiff');f.savefig(png,dpi=args.dpi)
  with Image.open(png) as im:
   im.save(tiff,compression='tiff_lzw',dpi=(args.dpi,args.dpi));shape=im.size
   with Image.open(tiff) as second:assert im.tobytes()==second.tobytes()
  f.savefig(out/(name+'_preview.png'),dpi=145)
  qa[name]=dict(axes_bounds_pt=geom,inches=list(f.get_size_inches()),pixels=shape,dpi=args.dpi,source_sha256={k:v['sha256'] for k,v in sources.items()},raster_pair_equal=True)
  (out/'figure_layout_QA.json').write_text(json.dumps(qa,indent=2));plt.close(f);print(name,shape,flush=True)
if __name__=='__main__':main()

