"""Editable synthesis from frozen results. No fits, generated art or raster assets."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Polygon,FancyArrowPatch,PathPatch
from matplotlib.path import Path as MPath
import scene_20261026_preserved as old
SRC=Path(__file__).resolve().parents[1]/'input';OUT=SRC
INK='#233F4D';GREY='#6E8B97';BLUE='#2D85A6';ORANGE='#C57B51';HUM='#397F80';ENERGY='#745C9F'
STAGE=old.STAGE
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7.3,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'FLTCCF-mechanism-vector-20261027','savefig.facecolor':'white'})

def text(ax,x,y,s,size=7.3,color=INK,bold=False,ha='center',va='center'):
 return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,fontweight='bold' if bold else 'normal',linespacing=1.23,zorder=25)
def arrow(ax,p,q,gid,color=GREY,lw=1,size=7,style='-|>'):
 a=FancyArrowPatch(p,q,arrowstyle=style,mutation_scale=size,linewidth=lw,color=color,zorder=12);a.set_gid(gid);ax.add_patch(a);return a

def evidence():
 old.SRC=SRC;old.OUT=OUT;old.verify_evidence(900)
 ledger=json.loads((OUT/'mechanism_evidence.json').read_text(encoding='utf8'))
 for rec in json.loads((SRC.parent/'mechanism_additional_inputs.json').read_text(encoding='utf8')):
  assert hashlib.sha256((SRC/rec['name']).read_bytes()).hexdigest()==rec['sha256']
 tables={k:pd.read_csv(SRC/k) for k in ['energy_primary.csv','ascent_energy.csv','ascent_environment.csv','departure_environment.csv','radial_moisture.csv']}
 energy=tables['energy_primary.csv'];assert len(energy)==6
 for metric in ['KE','APE','C_A_to_K','vertical_input_K_Wkg']:
  q=energy[energy.metric==metric];assert len(q)==1 and q['upper'].iloc[0]<0
 q=energy[energy.metric=='vertical_input_P_Wkg'];assert len(q)==1 and q.lower.iloc[0]>0
 q=energy[energy.metric=='vertical_input_A_Wkg'];assert len(q)==1 and q.lower.iloc[0]<0<q.upper.iloc[0]
 rise=tables['ascent_energy.csv'];rise=rise[(rise.window==1)&(rise.region=='r0_300')&(rise.scheme=='main')&(rise['sample']=='all_events')]
 select=[]
 for metric,estimand,lev in [('APE','rise_endpoint_change','700'),('C_A_to_K','rise_rate_integral','700'),('vertical_input_K_Wkg','rise_rate_integral','500-850'),('KE','rise_endpoint_change','700')]:
  content='SI_RISE_ENERGY_stores' if estimand=='rise_endpoint_change' else 'SI_RISE_ENERGY_rates'
  q=rise[(rise.metric==metric)&(rise.estimand==estimand)&(rise.level.astype(str)==lev)&(rise.content_id==content)];assert len(q)==1
  if metric!='KE':assert q.lower.iloc[0]>0
  else:assert q.lower.iloc[0]<0<q.upper.iloc[0]
  select.extend(q.to_dict('records'))
 dep=tables['departure_environment.csv'];dep=dep[dep.model=='geographical']
 env=[]
 for variable,coverage,sign in [('midRH','coverage_117',1),('shear200_850','coverage_116',-1)]:
  q=dep[(dep.variable==variable)&(dep.coverage==coverage)];assert len(q)==1
  assert sign*q.original_raw.iloc[0]>0;env.extend(q.to_dict('records'))
 radial=tables['radial_moisture.csv'];vert=radial[(radial.region=='100_300km')&(radial.sector=='whole')&(radial.field=='vertical_advection')&(radial.estimand=='integral_0_24')];assert len(vert)==1 and vert.lo.iloc[0]>0
 outer=radial[(radial.region=='300_800km')&(radial.sector=='whole')&(radial.field=='q700')&(radial.estimand=='pre_mean')];assert len(outer)==1 and outer.lo.iloc[0]>0
 ledger.update(design_version='20261027_physical_synthesis',no_new_scientific_calculation=True,
  energy_before_departure=energy.to_dict('records'),energy_during_return=select,departure_environment=env,inner_vertical_contribution=vert.to_dict('records'),outer_moisture=outer.to_dict('records'),
  module_results_map={'a':'MAIN paragraphs18–25; Fig1/Fig3; lifecycle arrows denote event order within one SID, Mature includes temporary weakening and return.','b':'MAIN paragraphs27–34; Figs4–7; temporary versus final departure raw paired contrasts, not a new adjusted causal effect.','c':'MAIN paragraphs35–40; Figs8–9; lower pre-departure stores/conversion/KE input and higher pressure input; APE-input uncertainty retained; actual ascent endpoint/integral evidence.','d':'MAIN paragraphs44–51; Fig11/S23; matched-wind RH contrast and undirected association with duration asymmetry.'},
  arrow_ledger={'a_stage_and_return':'Chronological sequence only, no proportional time','b_outward':'Gross outward direction; equal width/length, no inferred magnitude scaling','b_vertical_contrast':'Signed annotation only; positive paired vertical-advection contribution does not assert stronger upward air motion','b_decomposition':'Signed humidity and relative-normal-motion terms in exact product decomposition, not cyclone translation alone','c_conversion':'Positive actual APE-to-KE conversion; comparison labels from frozen pre-departure and ascent results','c_layer_KE':'Separate signed layer-input annotation, no arrow into 700-hPa KE; not a closed local KE budget','d_dotted':'Undirected association; no causal direction'},
  semantic_colors={'blue_orange':'Temporary/final class only','teal':'Humidity/moisture quantities','slate':'Relative motion or neutral geometry','violet':'Energy pathways','stage_colors':'Local stage guide only'},
  limitations_in_caption=['Stage and scene geometry schematic; arrows do not scale magnitude','Vertical-advection contribution is not a humidity profile','No inferred moisture-to-energy causal arrow','APE layer-input contrast spans zero','KE endpoint change during ascent spans zero and is not labelled an increase'])
 (OUT/'mechanism_evidence.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2,default=str),encoding='utf8')

def reservoir(ax,x,y,label,sub=None):
 ax.add_patch(Ellipse((x,y),16,9,facecolor='#EEE8F5',edgecolor=ENERGY,lw=.8,zorder=3))
 text(ax,x,y,label,8,color=ENERGY,bold=True)
 if sub:text(ax,x,y-7,sub,7,color=INK)

def build(dpi):
 evidence()
 fig=plt.figure(figsize=(180/25.4,174/25.4))
 specs=[('a',[.04,.805,.92,.175]),('b',[.04,.295,.55,.46]),('c',[.63,.295,.33,.46]),('d',[.04,.045,.92,.205])]
 axs={}
 for gid,pos in specs:
  ax=fig.add_axes(pos);ax.set_gid(gid);ax.axis('off');ax.set(xlim=(0,pos[2]*180),ylim=(0,pos[3]*174));axs[gid]=ax
 a,b,c,d=[axs[x] for x in 'abcd'];wa,wb,wc,wd=[ax.get_xlim()[1] for ax in [a,b,c,d]]
 text(a,0,29,'a  Lifecycle pathways',8.4,bold=True,ha='left')
 for name,x0,x1,col in [('Development',0,30,STAGE[0]),('Intensification',34,69,STAGE[1]),('Mature',74,128,STAGE[2]),('Decay',139,wa,STAGE[3])]:
  text(a,(x0+x1)/2,23.5,name,7.4,color=col,bold=True);a.plot([x0,x1],[18.5,18.5],color=col,lw=2.4,solid_capstyle='round')
 for x0,x1 in [(30.5,33),(69.5,73),(128.5,138)]:arrow(a,(x0,18.5),(x1,18.5),'chronological_stage',GREY,.75,5)
 a.plot([84,84,121],[18.5,5.4,5.4],color=BLUE,lw=1.2);arrow(a,(121,5.4),(121,18),'near_peak_return',BLUE,1.2,6)
 text(a,102.5,10.8,'Temporary departure',7.1,color=BLUE);text(a,122,1.2,'Return',7.1,color=BLUE)
 text(a,137,7.5,'Final\ndeparture',7.1,color=ORANGE)
 text(a,33,7.4,'Local moistening;\ndrier sampled background',7.2,color=HUM)
 # Paired physical scenes share the same geometry; category colors never encode another variable.
 text(b,0,79,'b  Moisture pathways near departure',8.2,bold=True,ha='left')
 for cx,tag,col,env,loss in [(24,'Temporary',BLUE,'Moister\nWeaker vertical wind shear','Less outward exchange'),(75,'Final',ORANGE,'Drier\nStronger vertical wind shear','More outward exchange')]:
  text(b,cx,71,tag,8.1,color=col,bold=True);text(b,cx,64.3,env,6.9)
  # Wind vectors are schematic context only; relative magnitude is carried by direct words.
  arrow(b,(cx-6,59),(cx+6,59),tag+'_upper_wind',GREY,.75,5)
  arrow(b,(cx-3,56),(cx+3,56),tag+'_lower_wind',GREY,.75,5)
  b.add_patch(Ellipse((cx,21),46,12,facecolor='#E1F0ED',edgecolor='#AFCEC6',lw=.6,zorder=0))
  old.cylinder(b,cx,47,23,12.5,col,tag)
  direction=-1 if tag=='Temporary' else 1
  arrow(b,(cx+direction*9,29),(cx+direction*21,29),tag+'_gross_outward',col,1.5,8)
  text(b,cx,13.0,loss,7.1,color=col,bold=True)
  text(b,cx+direction*18,33,'u − c',6.4,color=GREY)
 text(b,49.5,50.8,'Moving 300 km boundary',7,color=GREY)
 text(b,24,17.2,'300–800 km',6.5,color=HUM)
 text(b,75,17.2,'300–800 km',6.5,color=HUM)
 text(b,49.5,38.0,'Vertical contribution',6.7,color=HUM)
 text(b,49.5,33.6,'Temporary − final > 0',6.5,color=HUM)
 text(b,49.5,29.3,'100–300 km',6.8,color=GREY)
 text(b,39.0,6.5,'Humidity (+)',7.25,color=HUM,ha='right');text(b,43.0,6.5,'+',7.25,color=GREY);text(b,47.0,6.5,'relative motion (−)',7.25,color=GREY,ha='left')
 text(b,49.5,0.5,'Reduced export; larger net inward exchange',7.1,bold=True)
 # Energy diagnostics are physical reservoirs/conversion, with transport contrasts kept distinct.
 text(c,0,79,'c  Energy pathways',8.2,bold=True,ha='left')
 text(c,wc/2,71,'Before temporary vs final',7.2,bold=True)
 text(c,wc/2,66,'700 hPa · 2–16 days',7,color=GREY)
 reservoir(c,13,57,'APE','Lower');reservoir(c,46,57,'KE','Lower')
 arrow(c,(22,57),(37,57),'positive_APE_to_KE_before_departure',ENERGY,1.3,7)
 text(c,wc/2,43.5,'Lower APE → KE conversion',7.0,color=ENERGY)
 c.plot([2,wc-2],[38.5,38.5],color='#B6A9C8',lw=.8)
 text(c,wc/2,36,'500–850 hPa layer inputs',7.0,color=GREY)
 text(c,8.5,29.5,'APE\nUncertain',6.9);text(c,27.4,29.5,'KE\nLower',6.9);text(c,49.2,29.5,'Pressure\nHigher',6.9)
 c.plot([0,wc],[23,23],color='#D9E1E4',lw=.6)
 text(c,wc/2,19.8,'During near-peak return',7.2,color=BLUE,bold=True)
 reservoir(c,13,10,'APE');reservoir(c,46,10,'KE')
 arrow(c,(22,10),(37,10),'positive_ascent_APE_to_KE_integral',ENERGY,1.3,7)
 text(c,13,2.5,'Increase',7.0,color=ENERGY)
 text(c,29.6,16,'Conversion (+)',6.9,color=ENERGY)
 text(c,46,2.5,'Layer KE input (+)',6.6,color=ENERGY)
 text(c,46,-2.0,'500–850 hPa',6.6,color=GREY)
 # The third Results section remains part of the synthesis, with an explicitly undirected link.
 text(d,0,34,'d  Before and after peak intensity',8.2,bold=True,ha='left')
 text(d,40,26.8,'At comparable wind speed',7.1,color=GREY)
 for cx,title,rh,col in [(22,'Pre-peak','Higher RH','#BBDCD2'),(66,'Post-peak','Lower RH','#DEE6E3')]:
  text(d,cx,21.2,title,7.3,bold=True);d.add_patch(Ellipse((cx,11.3),34,14,facecolor=col,edgecolor='none',zorder=1));old.cloud(d,cx,12,11,title);text(d,cx,1.6,rh,7.2,color=HUM)
 d.plot([87,98,98,107],[12,12,18,18],color=GREY,lw=.85,ls=(0,(2,3)),gid='undirected_RH_duration_asymmetry')
 text(d,132,22,'Duration asymmetry A',7.8,bold=True)
 text(d,132,13.5,'Pre-peak versus\npost-peak duration',7.2)
 text(d,132,4.4,'Associated environmental contrast',6.9,color=GREY)
 # Undirected guide binds the two near-peak diagnostic panels to Mature; not a causal path.
 from matplotlib.lines import Line2D
 for xx,yy in [([.04+74/180,.04+74/180],[.805+18.5/174,.784]),([.04,.96],[.784,.784]),([.04,.04],[.784,.778]),([.96,.96],[.784,.778])]:
  line=Line2D(xx,yy,transform=fig.transFigure,color='#C9D5D9',lw=.6,zorder=0);line.set_gid('noncausal_near_peak_time_guide');fig.add_artist(line)
 emit(fig,row_groups=[['b','c']],column_groups=[['a','d']])
