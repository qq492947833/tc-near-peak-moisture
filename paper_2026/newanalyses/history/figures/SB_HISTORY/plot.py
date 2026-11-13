# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
"""Render only new supplementary plots from frozen computed CSVs; no inference."""
from pathlib import Path
import runpy,sys,json,hashlib
O=Path(__file__).resolve().parent
bootstrap(O/'temp')
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image
FONT=font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
INK='#233A46';BLUE='#167B9E';ORANGE='#CC7445';GREY='#809198';PURPLE='#9069A3'
mpl.rcParams.update({'font.family':'Arial','font.size':8.2,'axes.labelsize':8.2,'axes.titlesize':8.7,'axes.titleweight':'normal','xtick.labelsize':8.2,'ytick.labelsize':8.2,'legend.fontsize':8.0,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'FLTCCF_new_supplement_20261109','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'legend.frameon':False,'savefig.facecolor':'white','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.unicode_minus':True})
Q=O/'qa';Q.mkdir(exist_ok=True);records=[]
def read(name):return pd.read_csv(O/'source_data'/name,keep_default_na=False,na_values=['','nan','NaN'])
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def title(ax,letter,label):
 ax.set_gid(letter);ax.set_title(label,loc='left',pad=11)
 ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-26,11),textcoords='offset points',fontsize=10,fontweight='bold',va='baseline',ha='left')
def style(ax,ylabels,ylim=None):
 ax.set_yticks(np.arange(len(ylabels)),ylabels);ax.set_ylim(len(ylabels)-.55,-.55) if ylim is None else ax.set_ylim(*ylim)
 ax.tick_params(axis='y',length=0,pad=5);ax.tick_params(axis='x',length=3,pad=3)
 ax.grid(False);ax.set_axisbelow(True)
def forest(ax,est,lo,hi,y,color=BLUE,marker='o',label=None,ms=4.5):
 est=np.asarray(est,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
 assert np.isfinite(est).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
 assert np.all(lo<=est+1e-12) and np.all(hi>=est-1e-12)
 ax.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt=marker,markersize=ms,color=color,ecolor=color,elinewidth=.9,capsize=2,capthick=.8,lw=0,label=label,zorder=3)
def record(panel,df,**extra):
 z=df.copy();z.insert(0,'panel',panel)
 for k,v in extra.items():z[k]=v
 records.append(z)

fig,axes=plt.subplots(3,2,figsize=(7.086614173228347,8.070866141732284),dpi=300)
fig.subplots_adjust(left=.13,right=.97,bottom=.075,top=.94,wspace=.47,hspace=.77)
a,b,c,d,e,f=axes.flat
if O.name=='SA_BASIN':
 v=read('basin_estimates_ci.csv').set_index('basin').loc[['NA','EP','WP','NI','SI','SP']];yy=np.arange(6)
 for ax,col,label,letter,heading,scale in [(a,'event_count','Return events (count)','a','Return events',1),(b,'risk_hours','Observable cyclone-hours (×1,000)','b','Observable opportunity',1000)]:
  val=v[col].to_numpy(float)/scale;ax.barh(yy,val,height=.57,color=BLUE,alpha=.88,zorder=2);style(ax,v.index);ax.set_xlim(0,56);ax.set_xticks([0,20,40]);ax.set_xlabel(label);title(ax,letter,heading)
  for y,x in zip(yy,val):ax.annotate(f'{x:.0f}' if scale==1 else f'{x:.1f}',(x,y),xytext=(4,0),textcoords='offset points',va='center',ha='left',fontsize=8.2)
  record(letter,v.reset_index()[['basin',col]],display_divisor=scale,interval='descriptive frozen count or hours')
 forest(c,v.rate_per1000h,v.rate_low,v.rate_high,yy);style(c,v.index);c.set_xlim(-.08,2.9);c.set_xticks([0,1,2]);c.set_xlabel('Returns per 1,000 cyclone-hours');title(c,'c','Return rate per opportunity');record('c',v.reset_index()[['basin','rate_per1000h','rate_low','rate_high']])
 diff=read('basin_probability_differences.csv');contrasts=['NA-EP','NA-WP','NA-SI','EP-WP','EP-SI','WP-SI'];yl=[x.replace('-',' − ') for x in contrasts]
 for model,col,mark,off,lab in [('unadjusted',GREY,'s',-.24,'Raw'),('season_state',ORANGE,'^',0,'State'),('season_state_environment',BLUE,'o',.24,'+ Env')]:
  z=diff[diff.model.eq(model)].set_index('contrast').loc[contrasts];forest(d,z.difference_pp,z.ci_low,z.ci_high,yy+off,color=col,marker=mark,label=lab,ms=3.7);record('d',z.reset_index())
 style(d,yl,(5.6,-1.5));d.axvline(0,ymax=.85,color=INK,lw=.7,ls=(0,(3,3)),zorder=1);d.set_xlim(-16,22);d.set_xticks([-10,0,10,20]);d.set_xlabel('Difference (percentage points)');title(d,'d','Genesis-basin probability differences')
 d.legend(loc='upper center',ncol=3,columnspacing=.75,handletextpad=.25,handlelength=1,borderaxespad=.15)
 z=read('environment_associations.csv');scale=np.array([5.,5.,1.]);est=np.exp(z.log_odds_per_unit.to_numpy()*scale);lo=np.exp(z.ci_low.to_numpy()*scale);hi=np.exp(z.ci_high.to_numpy()*scale)
 if np.any(lo<=0) or np.any(est<=0) or np.any(hi<=0):raise ValueError('Odds-ratio log axis requires strictly positive values')
 forest(e,est,lo,hi,np.arange(3));style(e,['RH (+5 points)','Shear (+5 m/s)','SST (+1 °C)']);e.set_xscale('log');e.axvline(1,color=INK,lw=.7,ls=(0,(3,3)),zorder=1);e.set_xlim(.2,3.5);e.set_xticks([.25,.5,1,2],['0.25','0.5','1','2']);e.minorticks_off();e.set_xlabel('Odds ratio (stated increase)');title(e,'e','Initial environment associations');record('e',z,display_odds_ratio=est,display_ci_low=lo,display_ci_high=hi,increment=scale)
 z=diff[diff.model.eq('change_after_environment')].set_index('contrast').loc[contrasts];forest(f,z.difference_pp,z.ci_low,z.ci_high,yy);style(f,yl);f.axvline(0,color=INK,lw=.7,ls=(0,(3,3)),zorder=1);f.set_xlim(-9,9);f.set_xticks([-8,-4,0,4,8]);f.set_xlabel('State − environment (percentage points)');title(f,'f','Change after environment adjustment');record('f',z.reset_index())
elif O.name=='SB_HISTORY':
 z=read('global_equal_year_P4_minus_P1_decomposition.csv').set_index('component').loc[['total_within','total_composition','global_change']];forest(a,z.estimate_pp,z.ci_low,z.ci_high,np.arange(3));style(a,['Within basin','Basin mix','Global']);a.axvline(0,color=INK,lw=.7,ls=(0,(3,3)),zorder=1);a.set_xlim(-1.5,8);a.set_xticks([0,2,4,6,8]);a.set_xlabel('P4 − P1 difference (percentage points)');title(a,'a','Global change in return proportion');record('a',z.reset_index())
 hist=read('environment_history_trends.csv');basins=['NA','EP','WP','SI']
 for ax,letter,var,heading,label,xlim,ticks in [(b,'b','X3_mid_rh_pct','Humidity history within basins','RH change (percentage points / decade)',(-1.8,4),[-1,0,1,2,3]),(c,'c','X2_shear_ms','Shear history within basins','Shear change (m/s / decade)',(-2.5,1.3),[-2,-1,0,1]),(d,'d','X1_sst_c','SST history within basins','SST change (°C / decade)',(-.25,.55),[-.2,0,.2,.4])]:
  z=hist[hist.variable.eq(var)].set_index('basin').loc[basins];forest(ax,z.estimate,z.ci_low,z.ci_high,np.arange(4));style(ax,basins);ax.axvline(0,color=INK,lw=.7,ls=(0,(3,3)),zorder=1);ax.set_xlim(*xlim);ax.set_xticks(ticks);ax.set_xlabel(label);title(ax,letter,heading);record(letter,z.reset_index())
 tab=read('same_model_environment_only_trends.csv')
 for key,col,mark,off,lab in [('actual_environment',BLUE,'o',-.13,'Actual'),('fixed_reference_environment_state_unchanged',ORANGE,'s',.13,'Fixed environment')]:
  z=tab[tab.estimand.eq(key)].set_index('basin').loc[basins];forest(e,z.estimate_pp_per_decade,z.ci_low,z.ci_high,np.arange(4)+off,color=col,marker=mark,label=lab,ms=4);record('e',z.reset_index())
 style(e,basins,(3.5,-1));e.axvline(0,ymax=.84,color=INK,lw=.7,ls=(0,(3,3)),zorder=1);e.set_xlim(-18,8);e.set_xticks([-15,-10,-5,0,5]);e.set_xlabel('Probability trend (percentage points / decade)');title(e,'e','Actual and fixed-environment trends');e.legend(loc='upper center',ncol=2,handlelength=1,handletextpad=.25,columnspacing=.8,borderaxespad=.15)
 z=tab[tab.estimand.eq('actual_minus_fixed_environment')].set_index('basin').loc[basins];forest(f,z.estimate_pp_per_decade,z.ci_low,z.ci_high,np.arange(4));style(f,basins);f.axvline(0,color=INK,lw=.7,ls=(0,(3,3)),zorder=1);f.set_xlim(-2,6);f.set_xticks([-2,0,2,4,6]);f.set_xlabel('Actual − fixed (percentage points / decade)');title(f,'f','Environmental standardization contrast');record('f',z.reset_index())
else:raise RuntimeError('Unrecognized new-figure directory')

# Exact final canvas, with no tight bounding-box crop or post-export scaling.
fig.canvas.draw()
geometry=[]
for ax in axes.flat:
 geometry.append(dict(panel=ax.get_gid(),bbox_figure_fraction=list(ax.get_position().bounds),bbox_points=[v*72 for v in ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).bounds],xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),xscale=ax.get_xscale(),xticks=ax.get_xticks().tolist(),yticks=ax.get_yticks().tolist(),xtick_text=[t.get_text() for t in ax.get_xticklabels()],ytick_text=[t.get_text() for t in ax.get_yticklabels()],title=ax.get_title(loc='left'),xlabel=ax.get_xlabel(),normalization='none; source probabilities/CI retained; explicit count-hours/OR display transforms logged in plotted_values'))
dump(Q/'native_geometry.json',{'font_file':FONT,'width_mm':180,'height_mm':205,'dpi':900,'geometry':geometry})
pd.concat(records,ignore_index=True,sort=False).to_csv(O/'plotted_values.csv',index=False)
for suffix in ['pdf','svg']:
 fig.savefig(O/f'{O.name}.{suffix}',format=suffix,facecolor='white')
fig.savefig(O/f'{O.name}.png',dpi=900,facecolor='white')
fig.savefig(O/f'{O.name}.tiff',dpi=900,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
pdf=O/f'{O.name}.pdf'
with Image.open(O/f'{O.name}.png') as im1,Image.open(O/f'{O.name}.tiff') as im2:
 same=np.array_equal(np.asarray(im1.convert('RGBA')),np.asarray(im2.convert('RGBA')));info={'pixels':list(im1.size),'png_dpi':[float(v) for v in im1.info.get('dpi',[])],'tiff_dpi':[float(v) for v in im2.info.get('dpi',[])],'pixel_identical':bool(same),'true_native_dpi':900,'width_mm':180,'height_mm':205};assert same;dump(Q/'raster_identity.json',info)
# Python/PyMuPDF clip renders are only QA previews, never submission replacements.
import fitz
doc=fitz.open(pdf);page=doc[0];width,height=page.rect.width,page.rect.height
page.get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False).save(Q/'preview.png')
for ax in axes.flat:
 bb=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted());rect=fitz.Rect(max(0,bb.x0*width-5),max(0,(1-bb.y1)*height-16),min(width,bb.x1*width+5),min(height,(1-bb.y0)*height+5));pix=page.get_pixmap(matrix=fitz.Matrix(2,2),clip=rect,alpha=False);pix.save(Q/f'panel_{ax.get_gid()}.png')
doc.close();plt.close(fig)
