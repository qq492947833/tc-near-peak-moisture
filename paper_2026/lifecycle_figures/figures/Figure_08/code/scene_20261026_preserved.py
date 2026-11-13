"""Deterministic editable geometry; all cloud ribbons are Python vector objects.

The old reference is used for visual comparison only, never loaded as artwork.
Frozen CSV evidence checks are inherited; no scientific calculation is changed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath
SRC=Path(__file__).resolve().parents[1]/'input'
OUT=SRC
INK='#233F4D'; GREY='#6E8B97'; BLUE='#2D85A6'; ORANGE='#C57B51'
STAGE=['#467F9B','#458E81','#BD9346','#AE7066']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
 'font.size':7.4,'axes.titlesize':8.3,'svg.fonttype':'none','pdf.fonttype':42,
 'svg.hashsalt':'FLTCCF-mechanism-vector-20261026','savefig.facecolor':'white'})
def label(ax,x,y,s,size=7.4,color=INK,bold=False):
    return ax.text(x,y,s,fontsize=size,color=color,ha='center',va='center',fontweight='bold' if bold else 'normal',linespacing=1.25,zorder=20)
def arrow(ax,p,q,gid,color=GREY,lw=1,size=7):
    a=FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=size,linewidth=lw,color=color,zorder=12);a.set_gid(gid);ax.add_patch(a)
def cloud(ax,cx,cy,r,gid):
    ax.add_patch(Ellipse((cx,cy-.85),2.12*r,.92*r,facecolor='#96B9C7',alpha=.24,edgecolor='none',zorder=5))
    for arm in range(4):
        t=np.linspace(.2,5.35,110)+arm*np.pi/2
        radius=np.linspace(.18,1,110)*r;w=np.linspace(.19,.035,110)*r
        xy=np.c_[np.r_[cx+(radius+w)*np.cos(t),(cx+(radius-w)*np.cos(t))[::-1]],np.r_[cy+.43*(radius+w)*np.sin(t),(cy+.43*(radius-w)*np.sin(t))[::-1]]]
        for shift,col,z in [(-.40,'#A9C9D4',6),(0,'#EFF8FA',7)]:
            p=Polygon(xy+[0,shift],closed=True,facecolor=col,edgecolor='none',zorder=z);p.set_gid(gid+'_arm_'+str(arm)+'_'+str(z));ax.add_patch(p)
        for theta in np.linspace(1.1,4.2,7):
            f=(theta-.2)/5.15;rr=(.18+.82*f)*r;ang=theta+arm*np.pi/2;size=(.12-.055*f)*r
            ax.add_patch(Ellipse((cx+rr*np.cos(ang),cy+.43*rr*np.sin(ang)+.2),size*2,size,facecolor='white',edgecolor='none',zorder=8))
    ax.add_patch(Ellipse((cx,cy),.31*r,.14*r,facecolor='#8DB4C4',edgecolor='none',zorder=10))
def cylinder(ax,cx,top,bottom,r,col,gid):
    ry=r*.28
    ax.add_patch(Polygon([(cx-r,bottom),(cx-r,top),(cx+r,top),(cx+r,bottom)],facecolor=col,alpha=.075,edgecolor='none',zorder=1))
    ax.add_patch(Ellipse((cx,bottom),2*r,2*ry,facecolor=col,alpha=.10,edgecolor='none',zorder=1))
    for k in range(12):
        x=cx-r+2*r*k/12
        ax.add_patch(Polygon([(x,bottom),(x,top),(x+2*r/12,top),(x+2*r/12,bottom)],facecolor='white',alpha=.025+.055*(1-abs(k-5.5)/5.5),edgecolor='none',zorder=2))
    t=np.linspace(0,np.pi,100)
    for yy in [top,bottom]:ax.plot(cx+r*np.cos(t),yy+ry*np.sin(t),color=col,lw=.8,ls=(0,(4,3)),alpha=.7,zorder=3)
    for xx in [cx-r,cx+r]:ax.plot([xx,xx],[bottom,top],color=col,lw=.9,zorder=4)
    t=np.linspace(np.pi,2*np.pi,100)
    for yy in [top,bottom]:ax.plot(cx+r*np.cos(t),yy+ry*np.sin(t),color=col,lw=1,zorder=9)
    ax.add_patch(Ellipse((cx,bottom),2*r/3,2*ry/3,facecolor='none',edgecolor=col,lw=.75,zorder=9))
    cloud(ax,cx,top-11,r*.82,gid+'_storm')
def verify_evidence(dpi):
    # Original frozen row selectors are checked directly, without any fitting.
    bg=pd.read_csv(SRC/'background.csv');rh=pd.read_csv(SRC/'RH.csv');pi=pd.read_csv(SRC/'PI.csv')
    flux=pd.read_csv(SRC/'paired_flux_components.csv');prepost=pd.read_csv(SRC/'common_prepost.csv');matched=pd.read_csv(SRC/'matched_intensity.csv')
    checks={'background':bg[(bg.field=='q700')&(bg.component=='climate_change')&(bg.window=='-24_0')],
      'humidity_RH':rh[(rh.scope=='primary25')&(rh.variable=='time_q_contribution')],
      'PI_minus_V':pi[pi.component=='headroom'],
      'outward_components':flux[(flux.cohort=='original_post80')&(flux.kind=='out')&(flux.layer=='all')&(flux.sector=='all')&(flux.factor==1)],
      'outward_two_windows':prepost[(prepost.kind=='out')&prepost.quantity.isin(['D_pre','D_post'])],
      'net_post_departure':prepost[(prepost.kind=='net')&prepost.quantity.isin(['temporary_post','final_post'])],
      'RH_timing':matched[(matched.process=='midRH')&matched.role.isin(['raw','slope'])]}
    assert all(len(v)>0 for v in checks.values())
    for k in ['background','PI_minus_V','outward_two_windows','RH_timing']:assert checks[k].estimate.lt(0).all()
    for k in ['humidity_RH','net_post_departure']:assert checks[k].estimate.gt(0).all()
    z=checks['outward_components'];assert len(z)==7
    assert z[z.quantity=='difference'].estimate.lt(0).all() and z[z.quantity=='humidity'].estimate.gt(0).all() and z[z.quantity=='relative_motion'].estimate.lt(0).all()
    support=rh[(rh.scope=='primary25')&(rh.variable=='support_change')];assert support[['estimate','ci_low','ci_high']].eq(0).all().all()
    ledger={'stage_sequence':'D/E/F/Dec ordering only; no measured rate',
      'sampling_track':'Sampling-region translation, not drying causing intensification',
      'outward_temporary':'Observed gross outward transport direction; paired_flux_components.csv and common_prepost.csv; no arrow magnitude encoding',
      'outward_final':'Same transport definition and arrow width as temporary class',
      'RH_asymmetry_link':'Undirected dotted association from matched_intensity.csv, not causal',
      'signs':{'drier_background':'background.csv, q700 climate_change -24_0 < 0','RH_humidity_plus':'RH.csv primary25 time_q_contribution > 0','narrow_PI_minus_V':'PI.csv headroom < 0','gross_outward_contrast':'paired_flux_components.csv original_post80/out/all/all/factor1 difference < 0, humidity > 0, relative_motion < 0','both_windows':'common_prepost.csv out D_pre and D_post < 0','post_peak_RH':'matched_intensity.csv midRH raw and slope < 0'},
      'old_raster_or_vector_geometry_imported':False,'no_AI_image_generator':True,'new_statistical_inference':False}
    ledger['frozen_rows']={k:v.to_dict('records') for k,v in checks.items()}
    (OUT/'mechanism_evidence.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def build(dpi):
    verify_evidence(dpi)
    fig=plt.figure(figsize=(180/25.4,132/25.4))
    gs=fig.add_gridspec(1,4,left=.035,right=.965,bottom=.085,top=.84,wspace=.22)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1:3]);c=fig.add_subplot(gs[0,3]);axes=[a,b,c]
    for ax,gid in zip(axes,['a','b','c']):
        ax.set_gid(gid);ax.axis('off');p=ax.get_position();ax.set(xlim=(0,p.width*180),ylim=(0,p.height*132))
    wa,wb,wc=[ax.get_xlim()[1] for ax in axes];h=a.get_ylim()[1]
    sea=MPath([(.025,.245),(.10,.20),(.88,.17),(.975,.235),(.975,.54),(.70,.59),(.25,.57),(.025,.51),(.025,.245)],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY])
    fig.patches.append(PathPatch(sea,transform=fig.transFigure,facecolor='#EAF4F6',edgecolor='none',zorder=-2))
    for i,(name,col) in enumerate(zip(['Development','Intensification','Mature','Decay'],STAGE)):
        x=.045+i*.237;fig.text(x+.102,.949,name,fontsize=7.5,color=col,fontweight='bold',ha='center',va='center')
        fig.lines.append(plt.Line2D([x,x+.204],[.92,.92],transform=fig.transFigure,color=col,lw=2.3,solid_capstyle='round'))
        if i<3:
            p=FancyArrowPatch((x+.208,.92),(x+.23,.92),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=5,color='#B5C5CC',lw=.7);p.set_gid('stage_sequence_'+str(i));fig.patches.append(p)
    for ax,title in zip(axes,['Intensification setting','Near-peak departures','Lifecycle timing']):
        ax.text(0,1.035,ax.get_gid()+'  '+title,transform=ax.transAxes,ha='left',va='bottom',fontweight='bold',fontsize=8,color=INK)
    label(a,wa/2,h-7,'D → E / E → F',7.5);label(a,wa/2,h-17,'Drier sampled\nbackground',7.6)
    for cx,cy,col in [(wa*.29,h*.59,'#CDE3E3'),(wa*.70,h*.65,'#DCE6D8')]:
        a.add_patch(Ellipse((cx,cy),wa*.59,wa*.29,facecolor=col,edgecolor='#94B1BB',lw=.6,alpha=.8,zorder=1))
        a.add_patch(Ellipse((cx,cy),wa*.59/4,wa*.29/4,facecolor='#EAF4F6',edgecolor='#94B1BB',lw=.45,zorder=2));cloud(a,cx,cy+1,wa*.15,'sampling_'+str(cx))
    arrow(a,(wa*.38,h*.58),(wa*.61,h*.63),'sampling_track',lw=.8)
    label(a,wa/2,h*.46,'200–800 km',7.1);label(a,wa/2,h*.28,'Humidity contribution\nto RH (+)',7.6,color='#397F80');label(a,wa/2,7,'Wind growth\nnarrows PI − V',7.5)
    cx1,cx2=wb*.25,wb*.75;r=min(16,wb*.205)
    label(b,cx1,h-7,'Temporary',7.9,color=BLUE,bold=True);label(b,cx2,h-7,'Final',7.9,color=ORANGE,bold=True)
    for cx,col,tag in [(cx1,BLUE,'temporary'),(cx2,ORANGE,'final')]:cylinder(b,cx,h*.76,h*.38,r,col,tag)
    arrow(b,(cx1-r*.72,h*.44),(cx1-r*1.27,h*.44),'outward_temporary',BLUE,1.8,10);arrow(b,(cx2+r*.72,h*.44),(cx2+r*1.27,h*.44),'outward_final',ORANGE,1.8,10)
    label(b,cx1,h*.29,'Less gross\noutward exchange',7.2,color=BLUE,bold=True);label(b,cx2,h*.29,'More gross\noutward exchange',7.2,color=ORANGE,bold=True)
    label(b,wb/2,h*.87,'300 km boundary',7,color=GREY)
    label(b,wb/2,h*.17,'Specific-humidity component (+)',7.3,color='#397F80');label(b,wb/2,h*.09,'Relative-motion component (−)',7.3,color=GREY);label(b,wb/2,h*.015,'Partial cancellation',7.3,bold=True)
    label(c,wc/2,h-7,'Comparable intensity',7.2)
    for y,word,rh,col,gid in [(h*.65,'Pre-peak','Higher RH','#BEDFD7','pre_peak'),(h*.28,'Post-peak','Lower RH','#DCE3DF','post_peak')]:
        label(c,wc*.43,y+h*.12,word,7.6,bold=True);c.add_patch(Ellipse((wc*.43,y),wc*.78,wc*.43,facecolor=col,edgecolor='none',zorder=1));cloud(c,wc*.43,y,wc*.29,gid);label(c,wc*.43,y-h*.13,rh,7.2)
    c.plot([wc*.88,wc*.95,wc*.95,wc*.6],[h*.53,h*.53,h*.035,h*.035],color=GREY,lw=.7,ls=(0,(2,3)),gid='RH_asymmetry_link')
    c.plot([wc*.88,wc*.95],[h*.15,h*.15],color=GREY,lw=.7,ls=(0,(2,3)),gid='RH_contrast_lower_link');label(c,wc*.43,h*.035,'A',9.5,bold=True)
    emit(fig,row_groups=[['a','b','c']],column_groups=[])
