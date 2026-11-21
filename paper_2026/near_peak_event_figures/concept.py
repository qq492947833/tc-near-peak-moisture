from pathlib import Path
import argparse,json,csv,hashlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,Ellipse,FancyArrowPatch
from PIL import Image
import pandas as pd
INK='#233A46'; TEAL='#428C79'; BLUE='#167B9E'; OCHRE='#CC7445'; GRAY='#CAD6DC'
def label(a,x,y,s,**kw):
 a.text(x,y,s,color=kw.pop('color',INK),fontsize=kw.pop('fontsize',8.6),ha=kw.pop('ha','center'),va='center',**kw)
def box(a,x,y,w,h,s,face,edge):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.022',fc=face,ec=edge,lw=1))
 label(a,x+w/2,y+h/2,s,fontsize=9)
def arrow(a,x,y,X,Y,color=INK,style='-|>',lw=1.3):
 a.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle=style,mutation_scale=9,linewidth=lw,color=color,shrinkA=0,shrinkB=0))
def load_evidence(source):
 d=source/'Figure_08'/'input'; result={}
 def row(path,selector):
  f=pd.read_csv(path); k=f.loc[selector(f)]; assert len(k)==1,(path,len(k))
  return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),row=k.iloc[0].dropna().to_dict())
 result['shear']=row(d/'departure_environment.csv',lambda x:(x.variable=='shear200_850')&(x.model=='geographical')&(x.coverage=='coverage_116'))
 result['humidity']=row(d/'radial_moisture.csv',lambda x:(x.region=='300_800km')&(x.sector=='whole')&(x.field=='q700')&(x.estimand=='pre_mean'))
 result['outward']=row(d/'paired_flux_components.csv',lambda x:(x.kind=='out')&(x.layer=='all')&(x.sector=='all')&(x.quantity=='difference')&(x.cohort=='original_post80')&(x.factor==1)&(x.window=='0_24h'))
 for w in ['D_pre','D_post']:
  result[w]=row(d/'common_prepost.csv',lambda x:(x.kind=='out')&(x.quantity==w))
 result['conversion']=row(d/'ascent_energy.csv',lambda x:(x.content_id=='SI_RISE_ENERGY_rates')&(x.scheme=='main')&(x.support_mode=='native_diagnostic')&(x.metric=='C_A_to_K')&(x.level.astype(str)=='700')&(x.window==1)&(x.region=='r0_300')&(x.estimand=='rise_rate_integral')&(x['sample']=='all_events'))
 for i in range(3):
  result[str(i)]=row(source/'Figure_06'/'inputs'/'panel_g_source_rows.csv',lambda x:(x.metric=='APE')&(x.window==i))
 assert result['shear']['row']['original_raw']<0 and result['humidity']['row']['estimate']>0
 assert all(result[k]['row']['estimate']<0 for k in ('outward','D_pre','D_post','1','2'))
 assert all(result[k]['row']['estimate']>0 for k in ('0','conversion'));return result
def draw(source,out):
 out.mkdir(exist_ok=True,parents=True); evidence=load_evidence(source)

 mpl.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True})
 fig,axes=plt.subplots(1,3,figsize=(180/25.4,80/25.4),gridspec_kw=dict(wspace=.12))
 fig.subplots_adjust(left=.014,right=.985,bottom=.055,top=.965)
 for a,c,title in zip(axes,'abc',['Near-peak sequence','Departure and decay','Multiscale energetics']):
  a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
  label(a,0,.965,c,ha='left',weight='bold',fontsize=10);label(a,.11,.965,title,ha='left',weight='bold',fontsize=9.6)
  a.plot([0,1],[.90,.90],color=GRAY,lw=.8)
 a,b,c=axes
 for y,t,face,edge in [(.76,'Episode containing first LMI','#ECF4F1',TEAL),(.59,'Temporary departure','#EEF3F5',GRAY),(.42,'Trough and reintensification','#F4F7F8',GRAY),(.25,'Near-peak return','#ECF4F1',TEAL),(.08,'Final departure','#F7F3EE',GRAY)]:
  box(a,.037,y,.926,.087,t,face,edge)
 for y in [.75,.58,.41,.24]:arrow(a,.50,y,.50,y-.058)
 label(b,.5,.81,'Temporary departure',weight='bold')
 label(b,.5,.735,'relative to decay onset')
 for x,outer,name,wid in [(.25,'#66A697','Temporary',.11),(.74,'#A6B9BE','Decay',.20)]:
  b.add_patch(Ellipse((x,.51),.32,.25,fc='white',ec=outer,lw=2))
  b.add_patch(Ellipse((x,.51),.14,.11,ec='#637E84',fc='#EDF3F4',lw=.9))
  for direct in [-1,1]:
   arrow(b,x+direct*.042,.51,x+direct*(.052+wid),.51,color='#5B7582',lw=1.3)
  label(b,x,.327,name,fontsize=8.3)
 label(b,.5,.222,'Weaker deep-layer shear')
 label(b,.5,.151,'Higher outer-annulus humidity')
 label(b,.5,.080,'Smaller gross outward exchange')
 label(c,.5,.81,'Dry APE at matched winds',weight='bold')
 label(c,.5,.736,'Reintensification − decay')
 for y,window,i in [(.62,'>16 d',0),(.52,'2–16 d',1),(.42,'<2 d',2)]:
  label(c,.13,y,window,ha='left')
  pos=evidence[str(i)]['row']['estimate']>0
  label(c,.62,y,'Higher' if pos else 'Lower',ha='left',weight='bold',color=TEAL if pos else INK)
 c.plot([.09,.91],[.335,.335],color=GRAY,lw=.8)
 label(c,.5,.263,'2–16 d conversion during ascent',fontsize=8.4)
 box(c,.02,.07,.39,.104,'Dry APE','#E8F1ED',TEAL)
 box(c,.69,.07,.285,.104,'KE','#E8F1F5',BLUE)
 arrow(c,.435,.122,.663,.122,color=OCHRE,lw=2)
 stem=out/'Figure_06'
 for ext in ['png']:
  fig.savefig(stem.with_suffix('.'+ext),dpi=900,facecolor='white')
 plt.close(fig)
 with Image.open(stem.with_suffix('.png')) as im: rgb=im.convert('RGB')
 rgb.save(stem.with_suffix('.png'),dpi=(900,900));rgb.save(stem.with_suffix('.tiff'),compression='tiff_lzw',dpi=(900,900))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();draw(a.source,a.output)

