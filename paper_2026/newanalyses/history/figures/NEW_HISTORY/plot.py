# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from pathlib import Path
import sys,os,json,hashlib
R=Path(__file__).resolve().parent
bootstrap(R/'temp')
import numpy as np,pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'svg.hashsalt':'fltccf-history-20261109'})
SRC=R.parents[1]/'backup/Figure_07.png';im=Image.open(SRC).convert('RGBA');assert im.size==(6377,4606)
# User-authorized complete crop: original c label/title/axes/uncertainty remain untouched.
crop=np.asarray(im)[2380:4606,:,:].copy();Image.fromarray(crop).save(R/'original_panel_c.png')
w=6377;h=5400;dpi=900;fig=plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
a=fig.add_axes([.155,.53,.34,.36]);b=fig.add_axes([.64,.53,.34,.36]);a.set_gid('a');b.set_gid('b')
p=pd.read_csv(R/'period_basin_numerators_denominators.csv',keep_default_na=False,na_values=['']);basins=['NA','EP','WP','NI','SI','SP'];periods=['P1','P2','P3','P4'];colors=['#437F9C','#68A8A2','#BD9145','#AB685E'];markers=['o','s','^','D']
selected=[]
for j,(period,color,marker) in enumerate(zip(periods,colors,markers)):
 d=p[p.period.eq(period)&p.basin.isin(basins)].set_index('basin').loc[basins];assert d.n_sid.gt(0).all();x=np.arange(6)+(j-1.5)*.17
 a.errorbar(x,d.rate_pp,yerr=[d.rate_pp-d.lo_wilson,d.hi_wilson-d.rate_pp],fmt=marker,ms=3.7,capsize=1.8,color=color,lw=.8,label=period,zorder=3);selected.append(d.reset_index())
a.set(xticks=range(6),xticklabels=basins,xlim=(-.6,5.6),ylabel='Cyclones with a return (%)',xlabel='Genesis basin',ylim=(-.8,22));a.set_yticks([0,5,10,15,20]);a.grid(axis='y',color='#E5E7E9',lw=.55,zorder=0)
a.set_title('Returns across four periods',loc='left',pad=12);a.annotate('a',(0,1),xycoords='axes fraction',xytext=(-20,10),textcoords='offset points',weight='bold',fontsize=11,va='bottom')
a.legend(loc='lower left',bbox_to_anchor=(-.04,1.14),ncol=4,frameon=False,handletextpad=.3,columnspacing=.6,borderaxespad=0)
t=pd.read_csv(R/'annual_trends_and_basin_interactions.csv',keep_default_na=False,na_values=['']);keys=['NA','EP','WP','SI','GLOBAL anchored-return SID fraction'];z=t.set_index('estimand').loc[keys];y=np.arange(5)
b.errorbar(z.estimate,y,xerr=[z.estimate-z.lo,z.hi-z.estimate],fmt='o',ms=4.5,color='#437F9C',ecolor='#437F9C',capsize=2.3,lw=1.1);b.axvline(0,color='#999999',lw=.8,zorder=0);b.set(yticks=y,yticklabels=['NA','EP','WP','SI','Global'],ylim=(4.7,-.7),xlim=(-2.6,3.7),xticks=[-2,0,2],xlabel='Trend (pp decade⁻¹)');b.set_title('Annual recurrence trends',loc='left',pad=12);b.annotate('b',(0,1),xycoords='axes fraction',xytext=(-20,10),textcoords='offset points',weight='bold',fontsize=11,va='bottom')
# Raster-preserving placement at native900dpi; no redrawing or repainting panel c.
fig.figimage(crop,xo=0,yo=0,origin='upper',zorder=-10)
fig.canvas.draw()
fig.savefig(R/'Figure_NEW_HISTORY.png',dpi=900)
fig.savefig(R/'Figure_NEW_HISTORY.tiff',dpi=900,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(R/'Figure_NEW_HISTORY.pdf',dpi=900,metadata={'CreationDate':None,'ModDate':None})
fig.savefig(R/'Figure_NEW_HISTORY.svg',dpi=900,metadata={'Date':None})
out=np.asarray(Image.open(R/'Figure_NEW_HISTORY.png').convert('RGBA'));assert out.shape[:2]==(h,w);assert np.array_equal(out[-len(crop):],crop),'Protected c pixels changed'
with Image.open(R/'Figure_NEW_HISTORY.png') as aa,Image.open(R/'Figure_NEW_HISTORY.tiff') as bb:assert np.array_equal(np.asarray(aa),np.asarray(bb))
Image.fromarray(out).resize((1594,1350),Image.Resampling.LANCZOS).save(R/'preview.png')
Image.fromarray(out[:3174]).resize((1594,794),Image.Resampling.LANCZOS).save(R/'panel_ab_preview.png')
pd.concat(selected).to_csv(R/'panel_a_source.csv',index=False);z.reset_index().to_csv(R/'panel_b_source.csv',index=False)
receipt={'status':'RENDERED_PENDING_QA','size_mm':[w/dpi*25.4,h/dpi*25.4],'pixels':[w,h],'dpi':900,'a_rows':24,'b_rows':5,'c_crop_box':[0,2380,6377,4606],'c_crop_native_pixels_identical':True,'c_original_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'c_text':'protected original raster, not new PDF glyphs','c_label_and_title_retained':True,'unchanged_original_c_pixels':int(crop.shape[0]*crop.shape[1]),'no_SA_denominator':True,'no_new_statistics':True,'PNG_TIFF_same_pixels':True}
(R/'render_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt,indent=2))


