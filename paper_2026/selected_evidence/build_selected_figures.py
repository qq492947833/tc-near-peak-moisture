"""Compose selected native panels from their accepted raster assets. No scientific inference is performed."""
from pathlib import Path
import argparse, csv, json, re, hashlib, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
from scipy import ndimage
from spec import S, PANEL_ORDER
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from functools import lru_cache
import pandas as pd
P=argparse.ArgumentParser();P.add_argument('--source',type=Path,required=True);P.add_argument('--data',type=Path,required=True);P.add_argument('--output',type=Path,required=True);P.add_argument('--numbers',nargs='*',type=int,default=list(range(18)));a=P.parse_args()
D=a.source;out=a.output;out.mkdir(exist_ok=True,parents=True); dat=a.data
ORIG_FROM_RES=(592,483); WHITE=(255,255,255); W=6100; PAD=40; GAP=65
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'axes.unicode_minus':True})
# Individual origins in the parent thumbnail; adjusted below by finding the actual freestanding glyph.
GL={
'S02':[(55,36),(347,36)],'S03':[(70,42),(356,42),(70,223)],
'S04':[(49,28),(292,28)],'S05':[(63,24),(297,24)],
'S07':{'c':(40,174)},'S32':{'b':(306,27)},'S33':{'e':(28,326),'f':(237,326)},
'S08':{'c':(80,227),'d':(378,227)},'S09':{'d':(43,309),'e':(256,309)},
'S10':{'a':(77,15),'b':(318,15),'d':(318,177)},'S11':{'d':(327,263)},
'S12':{'a':(68,38),'b':(68,272)},'S13':{'a':(39,42),'b':(329,42)},
'S15':{'c':(40,176)},'S16':{'d':(266,170),'f':(266,324)},
'S23':{'b':(334,15),'a':(85,12),'c':(85,177),'d':(333,177)},'S24':{'b':(249,19),'c':(24,168),'e':(24,317),'f':(249,317)},
'S27':{'a':(62,37),'b':(354,37),'c':(62,237)},'S28':{'a':(54,23),'c':(54,177)},'S29':{'b':(232,35)}
}
# Row order indexes into S[n]. All legacy multi-panel row tiles retain their shared legends.
ROWS={2:[[0],[1]],3:[[0],[1]],4:[[0,1],[2,3]],5:[[0],[1]],6:[[0],[1,2]],7:[[0],[1]],8:[[0,1],[2]],10:[[0],[1]],11:[[0,1],[2,3]],14:[[0],[1]],15:[[0,1],[2]]}
@lru_cache(maxsize=3)
def orig(k):return Image.open(D/('Figure_'+k+'.png')).convert('RGB')
def sha(im):return hashlib.sha256(im.tobytes()).hexdigest()
def scale(im):return min(592/im.width,483/im.height,1)
def boxpx(im,b):return tuple(int(round(v/scale(im))) for v in b)
def crop(im,b):
 if not b:return im.copy()
 x,y,u,v=boxpx(im,b);return im.crop((max(0,x),max(0,y),min(im.width,u),min(im.height,v)))
def paint(fig,tag):
 buf=__import__('io').BytesIO();fig.savefig(buf,format='png',dpi=900,facecolor='white');plt.close(fig);buf.seek(0);return Image.open(buf).convert('RGB')
def new4():
 im=orig('04').copy(); sc=scale(im)
 # Recreate only c and d from the frozen estimates and intervals; leave a and b and their full scales native.
 pair=pd.read_csv(dat/'paired_flux_components.csv',float_precision='round_trip');adj=pd.read_csv(dat/'exchange_background80.csv',float_precision='round_trip')
 c=pair.query("cohort=='original_post80' and window=='0_24h' and factor==1 and layer=='all' and sector=='all' and quantity=='difference'").set_index('kind').loc[['in','out']]
 d=pd.concat([adj.query("model=='geographical' and metric=='post_r300_net'").set_index('target').loc[['original80_raw','restricted_raw','balanced']],adj.query("model=='geographical_state' and metric=='post_r300_net' and target=='balanced'").set_index('target')])
 assert len(c)==2 and len(d)==4
 def region(x0,y0,x1,y1,rows,labels,title,ch):
  b=boxpx(im,(x0,y0,x1,y1)); w=b[2]-b[0];h=b[3]-b[1];im.paste(WHITE,b); f=plt.figure(figsize=(w/900,h/900),dpi=900)
  f.text(.025,.955,ch,weight='bold',fontsize=9.2,va='top');f.text(.085,.955,title,fontsize=8.2,va='top')
  ax=f.add_axes([.36 if ch=='d' else .35,.25,.60,.56]);y=np.arange(len(rows))[::-1]
  v=rows.estimate.to_numpy()/1e12;lo=rows.low.to_numpy()/1e12;hi=rows.high.to_numpy()/1e12
  ax.axvline(0,lw=.7,ls=':',color='.5');ax.errorbar(v,y,xerr=[v-lo,hi-v],fmt='o',color='#267C91',capsize=2.3,lw=1.1,ms=3.4)
  ax.set(yticks=y,yticklabels=labels,xlabel='Temporary − decay (10¹² kg)',ylim=(-.6,len(y)-.4));ax.tick_params(labelsize=7);ax.xaxis.label.set_size(7.4);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
  tile=paint(f,ch);im.paste(tile,(b[0],b[1]));return [dict(panel=ch,kind=str(idx),estimate=float(r.estimate),lo=float(r.low),hi=float(r.high)) for idx,r in rows.iterrows()]
 vals=region(236,149,479,320,c,['Gross inward','Gross outward'],'Direct paired differences','c')
 vals+=region(12,319,479,477,d,['All pairs (80), unadjusted','Overlap (74), unadjusted','Location and season','Location, season and storm state'],'Net inward water-vapour exchange, 0–24 h','d')
 src=orig('04');mask=ImageChops.difference(src,im);assert mask.getbbox()
 (out/'Figure_04_values.json').write_text(json.dumps(vals,indent=2))
 return im
def new9():
 # Preserve b,c intact and retain only the first two native bars in a. The retained two bars are re-rendered with their intervals.
 im=orig('S20').copy();sc=scale(im)
 pair=pd.read_csv(dat/'paired_flux_components.csv',float_precision='round_trip')
 z=pair.query("cohort=='original_post80' and window=='0_24h' and factor==1 and layer=='all' and sector=='all' and kind=='out'").set_index('quantity').loc[['humidity','relative_motion']]
 b=(0,0,im.width,int(190/sc));f=plt.figure(figsize=(b[2]/900,b[3]/900));f.text(.081,.95,'a',fontweight='bold',fontsize=10,va='top');f.text(.16,.95,'Composition of the outward-exchange difference',fontsize=9,va='top');ax=f.add_axes([.13,.23,.81,.56])
 v=z.estimate.values/1e12;lo=z.low.values/1e12;hi=z.high.values/1e12
 ax.bar([0,1],v,yerr=[v-lo,hi-v],width=.46,color=['#308077','#9273a3'],error_kw=dict(capsize=2.7,lw=.8));ax.axhline(0,c='0.4',lw=.6);ax.set(xticks=[0,1],xticklabels=['Specific humidity','Relative normal motion'],ylabel='Temporary − decay (10¹² kg)');ax.tick_params(labelsize=7);ax.yaxis.label.set_size(7.4)
 im.paste(WHITE,b);im.paste(paint(f,'a'),(0,0));return im
def new13():
 # Existing native a,b are untouched. Panel c drops its duplicated third series.
 im=orig('S26').copy();p=pd.read_csv(dat/'plot_source.csv',float_precision='round_trip');z=p.query("panel=='c'").sort_values('row').iloc[:2];assert len(z)==2 and set(z.n_sid)=={122} and set(z.n_events)=={132}
 b=(round(im.width*.667),0,im.width,im.height);f=plt.figure(figsize=((b[2]-b[0])/900,b[3]/900));f.text(.02,.91,'c',fontweight='bold',fontsize=10.5);f.text(.22,.91,'Selected KE terms',fontsize=9)
 ax=f.add_axes([.23,.26,.69,.60]);ax.axhline(0,color='#9CAAB0',ls=(0,(3,3)),lw=.8)
 for i,(_idx,row) in enumerate(z.iterrows()):
  ax.errorbar(i,row.estimate,yerr=[[row.estimate-row.ci_lower],[row.ci_upper-row.estimate]],fmt='o',ms=4.5,capsize=3,lw=1.2,c=['#CC7445','#437F9C'][i])
 ax.set(xlim=(-.55,1.55),ylim=(-18,40),yticks=[-15,0,15,30],xticks=[0,1],xticklabels=['APE-to-KE\nconversion','Vertical\nKE input'],ylabel='Time integral (J kg⁻¹)');ax.tick_params(axis='x',length=0,pad=6)
 im.paste(WHITE,b);im.paste(paint(f,'c'),(b[0],0))
 # Move each complete native column, changing only its independent panel letter.
 edges=[0,2126,4251,im.width];glyph=[(98,143,177,251),(2208,143,2291,251),(4298,143,4368,251)]
 letters=[im.crop(g) for g in glyph];original=im.copy();out=Image.new('RGB',im.size,'white');dx=0
 for new_idx,j in enumerate(['abc'.index(c) for c in PANEL_ORDER[13].lstrip('0123456789')]):
  bounds=(edges[j],0,edges[j+1],im.height);tile=original.crop(bounds);g=glyph[j];spot=(g[0]-edges[j],g[1],g[2]-edges[j],g[3]);tile.paste(WHITE,spot);tile.paste(letters[new_idx],(spot[0],spot[1]));out.paste(tile,(dx,0));dx+=tile.width
 assert dx==im.width
 return out
def nearestglyph(k,ch,im):
 sc=scale(im);h=GL[k];cx,cy=h[ch] if isinstance(h,dict) else h[ord(ch)-97];local=(max(0,int((cx-11)/sc)),max(0,int((cy-12)/sc)),int((cx+12)/sc),int((cy+12)/sc))
 arr=np.asarray(im.crop(local)); lab,n=ndimage.label(np.min(arr,axis=2)<190);lbs=ndimage.find_objects(lab)
 cand=[]
 for sl in lbs:
  if not sl:continue
  sy,sx=sl;w=(sx.stop-sx.start)*sc;hh=(sy.stop-sy.start)*sc
  if not (2.0<w<15 and 3<hh<17): continue
  x=(sx.start+sx.stop)/2+local[0];y=(sy.start+sy.stop)/2+local[1];cand.append(((sc*x-cx)**2+(sc*y-cy)**2,(sx.start+local[0],sy.start+local[1],sx.stop+local[0],sy.stop+local[1])))
 if not cand:raise ValueError((k,ch,'cannot locate label',cx,cy))
 return min(cand)[1]
def relabel(k,im,b,old,new):
 if old==new:return
 src=orig(k);g=nearestglyph(k,old,src); bp=boxpx(src,b) if b else (0,0,src.width,src.height);x,y,x1,y1=(g[0]-bp[0],g[1]-bp[1],g[2]-bp[0],g[3]-bp[1]);assert 0<=x<x1<im.width and 0<=y<y1<im.height,(k,old,new,(x,y,x1,y1),im.size)
 b=(max(0,x-5),max(0,y-5),x1+5,y1+5);im.paste(WHITE,b); fontpath=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight='bold'))
 # Match visible glyph height for the old letter, retaining baseline and weight.
 fg=ImageFont.truetype(fontpath,100);ymin,ymax=fg.getbbox(old)[1::2];pt=round((y1-y)*100/(ymax-ymin));font=ImageFont.truetype(fontpath,max(12,pt));x0,t0,_,_=font.getbbox(new);_,to,_,_=font.getbbox(old);ImageDraw.Draw(im).text((x-x0,y-to),new,font=font,fill=(24,32,38),stroke_width=0)
def addlegend(im,src,box,where):
 patch=crop(orig(src),box);bbox=ImageChops.difference(patch,Image.new('RGB',patch.size,'white')).getbbox()
 if not bbox:return im
 patch=patch.crop(bbox)
 if patch.width>im.width:patch.thumbnail((im.width,patch.height),Image.Resampling.LANCZOS)
 ret=Image.new('RGB',(im.width,im.height+patch.height+25),'white');ret.paste(im,(0,patch.height+25 if where=='top' else 0));ret.paste(patch,((im.width-patch.width)//2,0 if where=='top' else im.height+25));return ret
def redraw_altlevels():
 # The three 700-hPa reference cells are removed; two off-level rows come directly from the source numeric matrix supplied with the release.
 p=pd.read_csv(dat/'s22_altlevels.csv',float_precision='round_trip'); assert len(p)==6
 a=p.pivot(index='pressure_hPa', columns='window',values='estimate').reindex(index=[500,850],columns=['>16 d','2–16 d','<2 d']); assert a.notna().all().all()
 f=plt.figure(figsize=(3.18,1.98));f.text(.02,.92,'a',fontsize=10,weight='bold');f.text(.14,.92,'Dry available potential energy',fontsize=8.5)
 ax=f.add_axes([.21,.31,.72,.47]);h=ax.imshow(a.values,cmap='RdBu_r',vmin=-5,vmax=5,aspect='auto',interpolation='nearest');ax.set(xticks=range(3),xticklabels=a.columns,yticks=[0,1],yticklabels=['500','850'],ylabel='Pressure (hPa)');ax.tick_params(length=0)
 for i in range(2):
  for j in range(3):ax.text(j,i,f'{a.iloc[i,j]:.3f}',ha='center',va='center',fontsize=7)
 cb=f.colorbar(h,cax=f.add_axes([.25,.16,.62,.026]),orientation='horizontal');f.text(.55,.025,'Paired difference (J kg⁻¹)',fontsize=7,ha='center',va='bottom');cb.ax.tick_params(labelsize=6.5);return paint(f,'a')
def build(n):
 if n==0:return new4()
 if n==9:return new9()
 if n==13:return new13()
 tiles=[]; nchar=0
 if n in [1,12,16,17]:return orig(S[n][0][0]).copy()
 for idx,(k,b,chars) in enumerate(S[n]):
  if n==10 and idx==0:tile=redraw_altlevels()
  else:
   tile=crop(orig(k),b)
   for z,ch in enumerate(chars):relabel(k,tile,b,ch,chr(97+nchar+z))
  if k=='S11':tile=addlegend(tile,'S11',(151,0,283,22),'top')
  if k=='S15':tile=addlegend(tile,'S15',(164,0,323,20),'top')
  if k=='S28' and chars=='c':tile=addlegend(tile,'S28',(163,466,326,480),'bottom')
  if k=='S29':
   ylab=crop(orig(k),(0,25,22,249));strip=Image.new('RGB',(tile.width+ylab.width,tile.height),'white');strip.paste(ylab,(0,0));strip.paste(tile,(ylab.width,0));tile=strip
  nchar+=len(chars);tiles.append(tile)
 rows=[]
 for row in ROWS[n]:
  items=[tiles[j] for j in row];avail=W-2*PAD-(len(row)-1)*GAP
  ratio=min(1,avail/sum(t.width for t in items))
  pieces=[t.resize((int(t.width*ratio),int(t.height*ratio)),Image.Resampling.LANCZOS) if ratio<.99999 else t for t in items];h=max(i.height for i in pieces);w=sum(i.width for i in pieces)+(len(pieces)-1)*GAP;can=Image.new('RGB',(W,h),'white');x=(W-w)//2
  for t in pieces:can.paste(t,(x,0));x+=t.width+GAP
  rows.append(can)
 return stack(rows)
def stack(rows):
 canvas=Image.new('RGB',(W,2*PAD+sum(t.height for t in rows)+GAP*(len(rows)-1)),'white');y=PAD
 for t in rows:canvas.paste(t,(0,y));y+=t.height+GAP
 return canvas
if __name__=='__main__':
 import shutil
 for n in a.numbers:
  name='Figure_04' if n==0 else f'Figure_S{n:02d}';im=build(n)
  if n in [1,12,16,17]:
   orig_name='Figure_'+S[n][0][0]
   for ext in ['png','tiff']:shutil.copyfile(D/(orig_name+'.'+ext),out/(name+'.'+ext))
  else:
   im.save(out/(name+'.png'),dpi=(900,900));im.save(out/(name+'.tiff'),compression='tiff_lzw',dpi=(900,900))
  with Image.open(out/(name+'.png')) as p,Image.open(out/(name+'.tiff')) as t:assert ImageChops.difference(p.convert('RGB'),t.convert('RGB')).getbbox() is None
  preview=im.copy();preview.thumbnail((1000,1300));preview.save(out/(name+'.jpg'),quality=90)
  print(n,im.size,flush=True)
