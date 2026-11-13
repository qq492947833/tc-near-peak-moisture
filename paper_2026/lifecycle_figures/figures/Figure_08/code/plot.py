"""Independent native geometry from approved scientific claims; no legacy asset geometry."""
from pathlib import Path
import json
import numpy as np,pandas as pd,matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,FancyArrowPatch
SRC=Path(__file__).resolve().parents[1]/'input'
OUT=SRC
INK='#233A46';BLUE='#167B9E';ORANGE='#CC7445';GREY='#7D929C'
STAGE=['#437F9C','#428C79','#BD9145','#AD675E']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.titlesize':9,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'S23-native-20261026','savefig.facecolor':'white'})
def text(ax,x,y,s,fs=8,ha='center',color=INK,weight='normal'):
    return ax.text(x,y,s,ha=ha,va='center',fontsize=fs,color=color,weight=weight)
def arrow(ax,start,end,gid,color=GREY,style='-|>',lw=.9):
    p=FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=8,lw=lw,color=color);p.set_gid(gid);ax.add_patch(p);return p
def build(dpi):
    bg=pd.read_csv(SRC/'background.csv');rh=pd.read_csv(SRC/'RH.csv');pi=pd.read_csv(SRC/'PI.csv')
    flux=pd.read_csv(SRC/'paired_flux_components.csv');prepost=pd.read_csv(SRC/'common_prepost.csv')
    matched=pd.read_csv(SRC/'matched_intensity.csv')
    assert bg[(bg.field=='q700')&(bg.component=='climate_change')&(bg.window=='-24_0')].estimate.lt(0).all()
    assert rh[(rh.scope=='primary25')&(rh.variable=='time_q_contribution')].estimate.gt(0).all()
    assert pi[pi.component=='headroom'].estimate.lt(0).all()
    z=flux[(flux.cohort=='original_post80')&(flux.kind=='out')&(flux.layer=='all')&(flux.sector=='all')&(flux.factor==1)]
    assert z[z.quantity=='difference'].estimate.lt(0).all() and z[z.quantity=='humidity'].estimate.gt(0).all() and z[z.quantity=='relative_motion'].estimate.lt(0).all()
    assert len(z)==7
    assert prepost[(prepost.kind=='out')&prepost.quantity.isin(['D_pre','D_post'])].estimate.lt(0).all()
    assert prepost[(prepost.kind=='net')&prepost.quantity.isin(['temporary_post','final_post'])].estimate.gt(0).all()
    assert matched[(matched.process=='midRH')&matched.role.isin(['raw','slope'])].estimate.lt(0).all()
    support=rh[(rh.scope=='primary25')&(rh.variable=='support_change')]
    assert support[['estimate','ci_low','ci_high']].eq(0).all().all()
    fig=plt.figure(figsize=(180/25.4,150/25.4))
    # Three complementary claim domains. These dimensions are independently defined.
    a=fig.add_axes([.045,.10,.43,.70]);b=fig.add_axes([.535,.46,.43,.34]);c=fig.add_axes([.535,.10,.43,.25])
    for ax,gid in [(a,'a'),(b,'b'),(c,'c')]:
        ax.set_gid(gid);ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
        ax.text(0,1.025,gid,transform=ax.transAxes,fontsize=10.5,weight='bold',ha='left',va='bottom',color=INK)
    # Top stage guide is sequence only, not a measured rate or causal claim.
    for i,(lab,col) in enumerate(zip(['Development','Intensification','Mature','Decay'],STAGE)):
        x=.045+i*.235
        fig.patches.append(Rectangle((x,.90),.215,.025,transform=fig.transFigure,facecolor=col,edgecolor='none'))
        fig.text(x+.1075,.945,lab,ha='center',va='center',fontsize=7.6,color=col)
        if i<3:
            p=FancyArrowPatch((x+.218,.9125),(x+.233,.9125),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=6,lw=.7,color=GREY)
            p.set_gid('stage_sequence_'+str(i));fig.patches.append(p)
    text(a,50,98,'D → E and E → F',fs=9,weight='bold')
    text(a,50,90,'Moving 200–800 km sampling regions',fs=7.8)
    # Annuli are schematic sampling domains, with exact 1:4 radius ratio.
    for x,col in [(25,'#C8DDDA'),(75,'#E4ECEB')]:
        a.add_patch(Circle((x,69),18,facecolor=col,edgecolor=GREY,lw=.8))
        a.add_patch(Circle((x,69),4.5,facecolor='white',edgecolor=GREY,lw=.8))
    arrow(a,(45,69),(55,69),'moving_sampling_domain')
    text(a,50,49,'Drier background along the track',fs=8)
    text(a,50,35,'Humidity contributes positively to RH\non common grid cells',fs=8)
    text(a,50,21,'Fixed support separates time and location',fs=8)
    text(a,50,7,'Wind growth narrows PI − V',fs=8)
    text(b,50,99,'Paired departure histories',fs=9,weight='bold')
    text(b,25,84,'Temporary',color=BLUE,fs=8)
    text(b,75,84,'Final',color=ORANGE,fs=8)
    # 300-km outer boundary and 100-km inner circle, same geometry in both classes.
    for x,col,tag in [(25,BLUE,'temporary'),(75,ORANGE,'final')]:
        b.add_patch(Circle((x,52),16,fill=False,edgecolor=col,lw=1.0,ls=(0,(3,2))))
        b.add_patch(Circle((x,52),16/3,fill=False,edgecolor=col,lw=.7))
        for dy in [-1,1]:
            arrow(b,(x,52+dy*9),(x,52+dy*25),'outward_'+tag+'_'+str(dy),color=col)
    text(b,50,15,'300 km boundary · 500–925 hPa layer',fs=7.5)
    text(b,50,1,'Less gross outward exchange after temporary departure',fs=7.0)
    text(c,50,99,'Post-peak environments and timing',fs=9,weight='bold')
    text(c,50,79,'Lower RH at comparable intensity',fs=8)
    c.plot([50,50],[51,67],color=GREY,lw=1.2,ls=(0,(1,2)),gid='association_no_direction')
    text(c,50,38,'Duration asymmetry A',fs=8)
    text(c,50,15,'Association',fs=7.5,color=GREY)
    # Keep interpretation outside scientific connectors.
    fig.text(.045,.055,'Outward-exchange contrast occurs in both adjacent 24 h windows;\nboth classes have positive mean net inward exchange after departure.',fontsize=7.2,ha='left',va='center',color=INK)
    fig.text(.965,.018,'Schematic geometry; arrows show direction, not magnitude.',fontsize=7,ha='right',va='bottom',color=GREY)
    emit(fig,row_groups=[],column_groups=[['b','c']])
