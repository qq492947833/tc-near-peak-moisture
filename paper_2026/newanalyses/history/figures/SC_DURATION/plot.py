# Public path adapter; the scientific functions below retain their original implementation.
import sys as _sys
from pathlib import Path as _Path
_PACKAGE=next(p for p in _Path(__file__).resolve().parents if (p/'release_paths.py').is_file())
_sys.path.insert(0,str(_PACKAGE))
from release_paths import ADDENDUM, UNIFIED, PACKAGE, bootstrap
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parent
bootstrap(R/'temp')
import numpy as np,pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'fltccf-duration-20261109','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
fig,axs=plt.subplots(3,2,figsize=(7.086614173228346,7.086614173228346),dpi=300);fig.subplots_adjust(left=.18,right=.98,bottom=.09,top=.94,wspace=.76,hspace=.72)
colors=['#7A868D','#437F9C','#418D79','#BD9145'];markers=['o','s','D','^'];targets=['actual_composition','fixed_basin_season','fixed_basin_season_state','fixed_basin_season_state_environment'];labels=['Actual distribution','Basin / season','+ State','+ Environment'];allrows=[]
d=pd.read_csv(R/'standardized_duration_trends.csv',keep_default_na=False,na_values=[''])
for k,(stage,title,lims) in enumerate([('D','Development',(-3,4.5)),('E','Intensification',(-4,1)),('F','Near-peak persistence',(-4.7,1)),('Res','Decay',(-11.2,1.3))]):
 a=axs.flat[k];a.set_gid(chr(97+k));g=d[d.stage.eq(stage)].set_index('target').loc[targets];assert len(g)==4 and g.n_sid.nunique()==1
 for i,(target,color,marker) in enumerate(zip(targets,colors,markers)):
  r=g.loc[target];a.errorbar(r.slope,i,xerr=[[r.slope-r.ci_low],[r.ci_high-r.slope]],fmt=marker,color=color,ms=4,capsize=2,lw=1)
 a.axvline(0,color='#A4AAAE',lw=.7,zorder=0);a.set(yticks=range(4),yticklabels=labels,ylim=(3.6,-.6),xlim=lims,xlabel='Trend (h decade⁻¹)');a.xaxis.set_major_locator(MaxNLocator(4));a.set_title(title,loc='left',pad=12);a.annotate(chr(97+k),(0,1),xycoords='axes fraction',xytext=(-17,10),textcoords='offset points',weight='bold',fontsize=11,va='bottom');g=g.reset_index();g.insert(0,'panel',chr(97+k));allrows.append(g)
a=axs[2,0];a.set_gid('e');g=pd.read_csv(R/'F_components_original_N1_replayed.csv').set_index('component').loc[['R','L','B']]
for i,(label,color) in enumerate(zip(['R','L','B'],['#437F9C','#418D79','#BD9145'])):
 r=g.loc[label];a.errorbar(r.estimate,i,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',color=color,ms=4,capsize=2,lw=1)
a.axvline(0,color='#A4AAAE',lw=.7);a.set(yticks=range(3),yticklabels=['First episode R','Later occupancy L','Bridge B'],ylim=(2.6,-.6),xlim=(-3.6,1),xticks=[-3,-2,-1,0,1],xlabel='Trend (h decade⁻¹)');a.set_title('Near-peak components',loc='left',pad=12);a.annotate('e',(0,1),xycoords='axes fraction',xytext=(-17,10),textcoords='offset points',weight='bold',fontsize=11,va='bottom');g.reset_index().to_csv(R/'panel_e_source.csv',index=False)
a=axs[2,1];a.set_gid('f');g=pd.read_csv(R/'Dec_observation_trends_HAC.csv').set_index('metric').loc[['elapsed_h','covered_h','tail_missing_wind_h','elapsed_h_no_missing_wind_tail']]
for i,(metric,color) in enumerate(zip(g.index,['#AB685E','#437F9C','#7A868D','#418D79'])):
 r=g.loc[metric];a.errorbar(r.slope,i,xerr=[[r.slope-r.ci_low],[r.ci_high-r.slope]],fmt='o',color=color,ms=4,capsize=2,lw=1)
a.axvline(0,color='#A4AAAE',lw=.7);a.set(yticks=range(4),yticklabels=['Elapsed','Wind-covered','Tail portion','No tail gaps'],ylim=(3.6,-.6),xlim=(-7.5,.6),xticks=[-6,-4,-2,0],xlabel='Trend (h decade⁻¹)');a.set_title('Decay-record coverage',loc='left',pad=12);a.annotate('f',(0,1),xycoords='axes fraction',xytext=(-17,10),textcoords='offset points',weight='bold',fontsize=11,va='bottom');g.reset_index().to_csv(R/'panel_f_source.csv',index=False)
pd.concat(allrows).to_csv(R/'panels_a_d_source.csv',index=False)
fig.canvas.draw()
fig.savefig(R/'Figure_SC_DURATION.png',dpi=900)
fig.savefig(R/'Figure_SC_DURATION.tiff',dpi=900,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(R/'Figure_SC_DURATION.pdf',dpi=900,metadata={'CreationDate':None,'ModDate':None})
fig.savefig(R/'Figure_SC_DURATION.svg',dpi=900,metadata={'Date':None})
fig.savefig(R/'preview.png',dpi=300)
for a in axs.flat:
 box=a.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.03,1.08);fig.savefig(R/('panel_'+a.get_gid()+'.png'),bbox_inches=box,dpi=300)
with Image.open(R/'Figure_SC_DURATION.png') as a,Image.open(R/'Figure_SC_DURATION.tiff') as b:assert np.array_equal(np.asarray(a),np.asarray(b));pixels=list(a.size)
receipt={'status':'RENDERED_PENDING_QA','size_mm':[180,180],'pixels':pixels,'dpi':900,'panels':6,'primary_rows':16,'e_rows':3,'f_rows':4,'source_estimates_reused':True,'new_statistics_computed':False,'families':'a-d new C16; e original N1 unchanged; f separate recording sensitivities','e_overlap_audit':'Old SI7d containsL/B restriction matrix, not R/L/B decomposition forest. New e directly compares first/late/bridge contributions using original values.','PNG_TIFF_same_pixels':True}
(R/'render_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(receipt,indent=2))

