"""Render this figure from packaged frozen results; no fitting or resampling."""
from pathlib import Path
import argparse,os,sys,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--dpi',type=int,default=900);args=p.parse_args()
if args.dpi!=900:raise ValueError('This frozen publication contract uses 900 dpi.')
PKG=Path(__file__).resolve().parent;OUT=Path(args.output_dir).resolve();OUT.mkdir(parents=True,exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME'):
    dest=OUT/'cache'/key.lower();dest.mkdir(parents=True,exist_ok=True);os.environ[key]=str(dest)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
environment=PKG.parents[1]/'environment'
if (environment/'bootstrap.py').exists():
    sys.path.insert(0,str(environment))
    from bootstrap import apply
    apply(OUT)
sys.path.insert(0,str(PKG/'code'))
from supp_style import audit_and_style
from decay_refine import refine
if os.environ.get('SUPP_CAPTURE_BASELINE')=='1':
    import baseline_render_source as m
else:
    import render_source as m
import pandas as pd
import numpy as np
from PIL import Image
C=json.loads((PKG/'config.json').read_text(encoding='utf-8'));INPUTS=json.loads((PKG/'inputs.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for i in INPUTS.values():
    if sha(PKG/i['relative_path'])!=i['sha256']:raise ValueError('Frozen input changed: '+i['relative_path'])
ROWS=[];USED=set()
def load(key):
    if key not in INPUTS:raise KeyError('Unpackaged data input: '+key)
    d=INPUTS[key];q=pd.read_csv(PKG/d['relative_path'],keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip',low_memory=False)
    q['_source_csv_row_index']=q.index;q['_source_csv_line']=np.arange(len(q))+2;q.attrs['source_key']=key;USED.add(key);return q
def table(fig,panel,df,tag):
    q=df.copy();fn='panel_'+str(panel)+'_'+str(len(ROWS))+'.csv';q.to_csv(OUT/fn,index=False,float_format='%.17g')
    ROWS.append({'id':str(panel),'role':tag,'source_key':df.attrs.get('source_key'),'fields':list(q.columns),'selected_rows':len(q),'saved_selected_table':fn,'sha256':sha(OUT/fn)})
    return df
def record(fig,panel,src,q,role,transform='identity'):
    q=q.copy();q.attrs['source_key']=src;return table(fig,panel,q,role+'; transform='+transform)
def log(fig,panel,src,q,meaning=''):return record(fig,panel,src,q,meaning)
def legacy(key):return key,load(key)
def export(fig,name,*a,**kw):
    # Apply approved text changes to the original Matplotlib text objects before export.
    for t in fig.findobj(m.Text):
        tx=t.get_text()
        for old,new in C.get('text_replacements',{}).items():tx=tx.replace(old,new)
        t.set_text(tx)
    if C.get('add_colourbar'):
        s=C['add_colourbar'];ax=next(x for x in fig.axes if x.get_gid()==s['panel']);im=ax.images[0];ca=fig.add_axes(s['cax']);ca.set_gid(s['panel']+'_colourbar');v=float(im.norm.vmax)
        cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[-v,0,v]);cb.ax.set_xticklabels([f'−{v:.3f}','0',f'{v:.3f}']);cb.ax.tick_params(labelsize=7,length=2,width=.5,pad=1.5);cb.outline.set_linewidth(.5);cb.ax.text(1.04,.5,s['unit'],transform=cb.ax.transAxes,ha='left',va='center',fontsize=7,color=m.INK)
    audit_and_style(fig,C['figure_id'],OUT)
    new_colorbar_axes=refine(fig,C['figure_id'],OUT)
    from terminology_20261103 import apply as revise_terminology
    revise_terminology(fig,C['figure_id'],OUT)
    fig.canvas.draw()
    from audit_panel_alignment import require_matplotlib_panel_alignment
    opts={k:v for k,v in kw.items() if k in ['axes','panel_ids','row_groups','column_groups','exemptions','exclude_axes']}
    excluded=list(opts.get('exclude_axes',[]))+list(kw.get('exclude',[]))+new_colorbar_axes
    if C.get('add_colourbar'):excluded.append(ca)
    if excluded:opts['exclude_axes']=excluded
    if C['figure_id']=='Figure_09':opts.update(row_groups=[['a','b'],['c','d'],['e','f']],column_groups=[['a','c','e'],['b','d','f']])
    if C['figure_id']=='Figure_S21':
        for t in fig.axes[0].texts:
            if t.get_text()=='Return':t.set_ha('left');t.set_x(0)
    require_matplotlib_panel_alignment(fig,json_out=OUT/'alignment.json',overlay_svg=OUT/'alignment.svg',tolerance_pt=1.5,strict=True,**opts)
    objects=[]
    for k,ax in enumerate(fig.axes):
        collections=[]
        for col in ax.collections:
            item={'class':type(col).__name__,'offsets':np.asarray(col.get_offsets()).tolist(),'paths':[p.vertices.tolist() for p in col.get_paths()],'array':None if col.get_array() is None else np.ma.filled(col.get_array(),np.nan).tolist(),'facecolor':col.get_facecolor().tolist(),'edgecolor':col.get_edgecolor().tolist(),'linewidth':col.get_linewidth().tolist(),'alpha':col.get_alpha(),'norm':{'class':type(col.norm).__name__,'vmin':col.norm.vmin,'vmax':col.norm.vmax},'cmap':col.cmap.name}
            if hasattr(col,'get_segments'):item['segments']=[np.asarray(x).tolist() for x in col.get_segments()]
            if hasattr(col,'get_coordinates'):item['coordinates']=np.asarray(col.get_coordinates()).tolist()
            collections.append(item)
        patches=[{'class':type(p).__name__,'path_vertices':p.get_path().vertices.tolist(),'path_codes':None if p.get_path().codes is None else p.get_path().codes.tolist(),'transform_matrix':p.get_patch_transform().get_matrix().tolist(),'facecolor':list(p.get_facecolor()),'edgecolor':list(p.get_edgecolor()),'linewidth':p.get_linewidth()}for p in ax.patches]
        objects.append({'panel':ax.get_gid() or str(k),'position':list(ax.get_position().bounds),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),'title':ax.get_title(),'left_title':ax.get_title(loc='left'),'lines':[{'x':np.asarray(l.get_xdata()).tolist(),'y':np.asarray(l.get_ydata()).tolist(),'label':l.get_label()} for l in ax.lines],'images':[{'array':np.asarray(im.get_array()).tolist(),'vmin':im.norm.vmin,'vmax':im.norm.vmax,'norm':type(im.norm).__name__,'cmap':im.cmap.name} for im in ax.images],'collections':collections,'patches':patches})
    (OUT/'render_objects.json').write_text(json.dumps(objects,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    texts=[{'text':t.get_text(),'size_pt':t.get_fontsize()} for t in fig.findobj(m.Text) if t.get_visible() and t.get_text().strip()]
    (OUT/'render_text.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2),encoding='utf-8')
    for ext in ['png','tiff','pdf','svg']:
        opts={'dpi':900}
        if ext=='tiff':opts['pil_kwargs']={'compression':'tiff_lzw'}
        if ext in ('pdf','svg'):opts['metadata']={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None}
        fig.savefig(OUT/(C['figure_id']+'.'+ext),**opts)
    fig.savefig(OUT/'preview.png',dpi=150)
    m.plt.close(fig)
    with Image.open(OUT/(C['figure_id']+'.png')) as im:sz=list(im.size);dpi=[float(x) for x in im.info['dpi']]
    receipt={'figure_id':C['figure_id'],'status':'RENDERED_PENDING_VISUAL_QA','no_new_inference':True,'used_input_keys':sorted(USED),'panels':ROWS,'pixels':sz,'dpi':dpi,'figure_inches':list(fig.get_size_inches()),'files':{e:{'path':C['figure_id']+'.'+e,'sha256':sha(OUT/(C['figure_id']+'.'+e))} for e in ['png','tiff','pdf','svg']},'no_AI_assets':True}
    (OUT/'rebuild_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
m.rd=m.source=m.read=m.external=load;m.legacy=legacy;m.table=table;m.record=record;m.log=log;m.export=m.finish=export
m.DATA=m.D=PKG/'inputs';m.TABLES=m.REPORTS=m.REP=m.OUT=OUT
getattr(m,C['function'])(*C.get('arguments',[]))
