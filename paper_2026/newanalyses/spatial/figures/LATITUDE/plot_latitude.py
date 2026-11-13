# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
"""Only the new four-panel latitude supplement; finalized source CSVs required."""
from pathlib import Path
import runpy,sys,json,hashlib
O=Path(__file__).resolve().parent
bootstrap(O/'temp')
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from matplotlib.lines import Line2D
from PIL import Image
import fitz

S=O/'source_data';Q=O/'qa';Q.mkdir(exist_ok=True)
binding=json.loads((O/'FINAL_STATISTICAL_BINDING.json').read_text(encoding='utf8'))
assert binding['accepted_version']=='v2_fold_supported'
assert binding['complete_bootstrap_attempts_per_hemisphere']==1000
assert binding['training_sid']==1245 and binding['reference_sid']==322
assert binding['all_final_sources_hash_verified'] is True
for item in binding['sources']:
 assert hashlib.sha256((O/item['snapshot']).read_bytes()).hexdigest()==item['sha256']
def read(name):return pd.read_csv(S/name,keep_default_na=False,na_values=['','NaN','nan'],float_precision='round_trip')
def dump(name,obj):(Q/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
counts=read('latitude_bin_display_counts.csv');risks=read('plotted_standardized_probabilities.csv')
display=counts[counts.displayed].copy()
assert display.actual_return_events.sum()==133 and display.historical_entry_sid.sum()==2340
assert risks.shape[0]==8 and set(risks.model)=={'M0_state','M1_state_joint_environment'}
assert np.isfinite(risks[['signed_latitude','probability_pct','ci_low_pct','ci_high_pct']]).all().all()
assert (risks.ci_low_pct>=0).all() and (risks.ci_high_pct<=100).all() and (risks.ci_low_pct<=risks.ci_high_pct).all()
assert all(np.isclose(risks[risks.hemisphere.eq(h)].n_reference_sid,n).all() for h,n in [('NH',202),('SH',120)])
arial=font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
INK='#233A46';BLUE='#167B9E';ENV='#B66042';GREY='#BECBD1';UNKNOWN='#D8B568'
mpl.rcParams.update({'font.family':'Arial','font.size':8.2,'axes.labelsize':8.2,'axes.titlesize':8.7,'xtick.labelsize':8.2,'ytick.labelsize':8.2,'legend.fontsize':8.0,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'FLTCCF_latitude_20261110','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'legend.frameon':False,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.unicode_minus':True})
fig,axes=plt.subplots(2,2,figsize=(7.086614173228347,6.496062992125984),dpi=300)
fig.subplots_adjust(left=.12,right=.97,bottom=.10,top=.88,wspace=.48,hspace=.85)
a,b,c,d=axes.flat
def heading(ax,letter,text):
 ax.set_gid(letter);ax.set_title(text,loc='left',pad=34)
 ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-29,34),textcoords='offset points',ha='left',va='baseline',fontsize=10,fontweight='bold')
for ax in axes.flat:
 ax.tick_params(length=3,pad=3);ax.grid(False)

heading(a,'a','Observed return locations')
a.bar(display.bin_mid,display.actual_return_events,width=8.7,color=BLUE,zorder=2)
a.set(xlim=(-50,50),ylim=(0,60),xlabel='Actual-return latitude (°)',ylabel='Return events (count)')
a.set_xticks([-40,-20,0,20,40]);a.set_yticks([0,20,40,60])
heading(b,'b','48-h outcomes by entry latitude')
bottom=np.zeros(len(display))
for field,label,color in [('observed_return48_sid','Returned',BLUE),('known_nonreturn48_sid','No return',GREY),('unknown48_sid','Unknown',UNKNOWN)]:
 values=display[field].to_numpy(float);b.bar(display.bin_mid,values,bottom=bottom,width=8.7,label=label,color=color,zorder=2);bottom+=values
b.set(xlim=(-50,50),ylim=(0,max(bottom)*1.16),xlabel='Entry latitude (°)',ylabel='Cyclones (count)')
b.set_xticks([-40,-20,0,20,40]);b.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,integer=True))
b.legend(loc='upper left',bbox_to_anchor=(.01,.99),borderaxespad=.1,handlelength=1.2,handletextpad=.4,labelspacing=.25)

# Exact supported signed coordinates: no x offsets or synthetic latitude grid.
expected={'NH':{'risk_low':15.8,'risk_high':23.0},'SH':{'risk_low':-15.0,'risk_high':-20.2}}
prob_max=min(100.,max(20.,np.ceil(risks.ci_high_pct.max()/10)*10))
legend=[Line2D([],[],marker='o',markerfacecolor='white',markeredgecolor=BLUE,markeredgewidth=1.1,markersize=6,linestyle='none',label='State'),Line2D([],[],marker='s',markerfacecolor=ENV,markeredgecolor=ENV,markersize=3.8,linestyle='none',label='State + environment')]
drawn_probabilities=[]
for ax,hem,letter,title in [(c,'NH','c','Northern Hemisphere'),(d,'SH','d','Southern Hemisphere')]:
 heading(ax,letter,title)
 for model,color,marker,size,face,linestyle,zorder in [('M0_state',BLUE,'o',6,'white','solid',3),('M1_state_joint_environment',ENV,'s',3.8,ENV,(0,(2,2)),4)]:
  selected=risks[risks.hemisphere.eq(hem)&risks.model.eq(model)].sort_values('signed_latitude')
  assert len(selected)==2 and set(selected.target)=={'risk_low','risk_high'}
  xx=selected.signed_latitude.to_numpy(float)
  assert np.array_equal(xx,np.array([expected[hem][t] for t in selected.target]))
  # The interval lines are independent of point geometry: original percentile
  # intervals are retained even if a point lies outside its percentile interval.
  interval_artist=ax.vlines(xx,selected.ci_low_pct,selected.ci_high_pct,color=color,linestyle=linestyle,linewidth=.95,zorder=2)
  for yy in [selected.ci_low_pct,selected.ci_high_pct]:ax.plot(xx,yy,linestyle='none',marker='_',markersize=7 if model=='M0_state' else 4.5,color=color,markeredgewidth=.9,zorder=2)
  point_artist,=ax.plot(xx,selected.probability_pct,linestyle='none',marker=marker,markersize=size,markerfacecolor=face,markeredgecolor=color,markeredgewidth=1.1,zorder=zorder)
  drawn_probabilities.append({'panel':letter,'hemisphere':hem,'model':model,'x':point_artist.get_xdata().tolist(),'y':point_artist.get_ydata().tolist(),'interval_segments':[segment.tolist() for segment in interval_artist.get_segments()],'marker':marker,'x_offset':0})
 coords=sorted(expected[hem].values());ax.set_xlim(coords[0]-2.2,coords[-1]+2.2);ax.set_xticks(coords,[f'{abs(v):.1f}°'+('N' if v>0 else 'S') for v in coords])
 ax.set_ylim(0,prob_max);ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4));ax.set_xlabel('Entry latitude');ax.set_ylabel('48-h return probability (%)')
 ax.legend(handles=legend,loc='lower left',bbox_to_anchor=(0,1.015),borderaxespad=0,handlelength=1.3,handletextpad=.4,labelspacing=.25)

fig.canvas.draw()
geometry=[]
for ax in axes.flat:
 geometry.append({'panel':ax.get_gid(),'axes_bbox_fraction':list(ax.get_position().bounds),'axes_bbox_pt':[x*72 for x in ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).bounds],'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'xticks':ax.get_xticks().tolist(),'yticks':ax.get_yticks().tolist(),'xtick_text':[x.get_text() for x in ax.get_xticklabels()],'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),'normalization':'none: census counts or saved probability percentages; actual signed latitude coordinates unchanged'})
dump('native_geometry.json',{'dimensions_mm':[180,165],'Arial_font_file':arial,'minimum_configured_pt':8,'axes':geometry,'drawn_probabilities':drawn_probabilities,'c_d_x_offsets':0,'counts_are_separate_populations':True})
for suffix in ['pdf','svg']:fig.savefig(O/f'Figure_LATITUDE.{suffix}',facecolor='white')
fig.savefig(O/'Figure_LATITUDE.png',dpi=900,facecolor='white')
fig.savefig(O/'Figure_LATITUDE.tiff',dpi=900,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
pdf=O/'Figure_LATITUDE.pdf'
with Image.open(O/'Figure_LATITUDE.png') as p,Image.open(O/'Figure_LATITUDE.tiff') as t:
 same=np.array_equal(np.asarray(p.convert('RGBA')),np.asarray(t.convert('RGBA')));assert same
 dump('raster_identity.json',{'pixels':list(p.size),'pixel_identical':bool(same),'png_dpi':[float(x) for x in p.info['dpi']],'tiff_dpi':[float(x) for x in t.info['dpi']],'native_export_dpi':900})
doc=fitz.open(pdf);page=doc[0];page.get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False).save(Q/'preview.png')
for ax in axes.flat:
 bb=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted());rect=fitz.Rect(max(0,bb.x0*page.rect.width-8),max(0,(1-bb.y1)*page.rect.height-5),min(page.rect.width,bb.x1*page.rect.width+5),min(page.rect.height,(1-bb.y0)*page.rect.height+5));page.get_pixmap(matrix=fitz.Matrix(2.5,2.5),clip=rect,alpha=False).save(Q/f'panel_{ax.get_gid()}.png')
doc.close();plt.close(fig)
