"""Portable frozen-data figure entry point; no inference or image compositing."""
from pathlib import Path
import argparse, os, sys, json, hashlib, ast, re, inspect, platform, importlib.util

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',required=True)
parser.add_argument('--dpi',type=int,default=900)
args=parser.parse_args()
R=Path(args.output_dir).resolve(); R.mkdir(parents=True,exist_ok=True)
for sub in ['tmp','cache/matplotlib','panels']:(R/sub).mkdir(parents=True,exist_ok=True)
os.environ.update(TEMP=str(R/'tmp'),TMP=str(R/'tmp'),TMPDIR=str(R/'tmp'),
 MPLCONFIGDIR=str(R/'cache/matplotlib'),XDG_CACHE_HOME=str(R/'cache'),
 PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.dont_write_bytecode=True
bootstrap_info=None
envdir=ROOT.parents[1]/'environment'
if (envdir/'bootstrap.py').exists():
    sys.path.insert(0,str(envdir))
    from bootstrap import apply
    bootstrap_info=apply(R)
print('IMPORT_BEGIN '+ROOT.name,flush=True)
vendor=ROOT.parents[1]/'environment/vendor'
if vendor.is_dir():sys.path.insert(0,str(vendor))
sys.path.insert(0,str(ROOT/'code'))
# Scoped OS metadata uses the running interpreter/native environment and avoids
# platform's WMI query during dependency initialization. Restored after imports.
original_system,original_machine=platform.system,platform.machine
if sys.platform=='win32' and bootstrap_info is None:
    platform.system=lambda:'Windows'
    platform.machine=lambda:os.environ.get('PROCESSOR_ARCHITEW6432',os.environ.get('PROCESSOR_ARCHITECTURE','AMD64'))
try:
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle, Wedge, Circle
    from matplotlib.lines import Line2D
    from matplotlib.colors import TwoSlopeNorm, Normalize, LogNorm, LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator
    from matplotlib.text import Text
    from PIL import Image
    from audit_panel_alignment import require_matplotlib_panel_alignment
    from supp_style import audit_and_style
finally:
    platform.system,platform.machine=original_system,original_machine

print('IMPORT_COMPLETE '+ROOT.name,flush=True)
config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
FIGURE=config['figure_id']; DATA=ROOT/'inputs'; U=DATA
SOURCES={};LEDGER=[];EXPORTS=[];SOURCE_KEYS={}
MAP={k:ROOT/v for k,v in config['aliases'].items()}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for row in manifest['inputs']:
    assert sha(ROOT/row['relative_path'])==row['sha256'],row['relative_path']
    SOURCE_KEYS[str((ROOT/row['relative_path']).resolve())]=row

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'unified-master-2340',
 'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.65,'legend.frameon':False})

def load(path):
    p=Path(path).resolve(); assert p.is_relative_to(ROOT/'inputs'),p
    SOURCES[str(p.relative_to(ROOT))]=sha(p)
    d=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN'],float_precision='round_trip')
    d.attrs['source']=str(p.relative_to(ROOT))
    return d
def source(key):return load(MAP[key])
rd=read=external=source
def legacy(key):return key,source(key)

def table(fig,panel,data,tag):
    data=data.copy();path=R/'panels'/f'{len(LEDGER):02d}_{panel}_{re.sub(r"[^A-Za-z0-9_-]","_",tag)}.csv'
    data.to_csv(path,index=False,float_format='%.17g')
    frame=inspect.currentframe().f_back
    expression=''.join(inspect.getframeinfo(frame,context=1).code_context or []).strip()
    sample={}
    for col in data.columns:
        if col.lower() in ['n','n_sid','n_sids','n_pairs','n_events','valid_draws','bootstrap_successful','n_years']:
            vals=data[col].dropna().unique().tolist();sample[col]=[x.item() if hasattr(x,'item') else x for x in vals]
    if 'sid' in data:sample['distinct_observed_SIDs']=int(data.sid.nunique())
    for col in ['event_id','pair_id']:
        if col in data:sample['distinct_'+col]=int(data[col].nunique())
    units=sorted(data['units'].dropna().astype(str).unique()) if 'units' in data else []
    LEDGER.append({'panel':panel,'role':tag,'rows':len(data),'fields':list(data.columns),
      'input_relative_path':data.attrs.get('source'), 'filter':{'executed_code':expression,'entry_function':config['entry_function'],'selected_rows':str(path.relative_to(R)),'selected_rows_sha256':sha(path)},
      'sample_scope':sample,'units_from_source':units,'table_path':str(path.relative_to(R))})
    return data
def log(name,panel,src,data,meaning=''):
    d=data.copy()
    if src in MAP:d.attrs['source']=str(MAP[src].relative_to(ROOT))
    return table(name,panel,d,meaning or 'original_statistics')
def record(fig,panel,src,q,role,transform='identity'):return log(fig,panel,src,q,role)
def finish(fig,name,**kwargs):return export(fig,name,**kwargs)

def numeric_artists(fig):
    axes=[]
    for ax in fig.axes:
        item={'id':ax.get_gid(),'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),'title':ax.get_title(loc='left'),
         'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'texts':[t.get_text() for t in ax.texts],
         'lines':[],'collections':[],'patches':[]}
        for line in ax.lines:
            item['lines'].append({'x':np.asarray(line.get_xdata()).tolist(),'y':np.asarray(line.get_ydata()).tolist()})
        for c in ax.collections:
            vals=c.get_array()
            item['collections'].append({'type':type(c).__name__,'offsets':np.asarray(c.get_offsets()).tolist(),'values':None if vals is None else np.asarray(vals).tolist()})
        for p in ax.patches:
            if isinstance(p,Rectangle):item['patches'].append({'type':'Rectangle','xy':list(p.get_xy()),'width':p.get_width(),'height':p.get_height()})
            elif isinstance(p,Wedge):item['patches'].append({'type':'Wedge','center':list(p.center),'r':p.r,'theta1':p.theta1,'theta2':p.theta2,'width':p.width})
        axes.append(item)
    return axes

def export(fig,name,dpi=None,exclude=(),exemptions=(),exclude_axes=(),row_groups=None,**unused):
    dpi=args.dpi if dpi is None else dpi
    assert dpi>=600,'Requested final raster DPI must be at least 600'
    if config['legacy_identity']=='SI_recurrence_robustness':
        a=next(ax for ax in fig.axes if ax.get_gid()=='b')
        assert all(p.get_height()>0 for p in a.patches)
        a.set_yscale('log');a.set(xticks=range(5),xlim=(-.5,4.5),ylim=(.8,3500),ylabel='Cyclones (log scale)')
    for t in fig.findobj(Text):
        text=t.get_text().replace('Escalation','Intensification').replace('Maturity','Near-peak persistence')
        text=re.sub(r'PI[-–‑ ]headroom','PI − V',text)
        t.set_text(text)
    audit_and_style(fig,FIGURE,R)
    fig.canvas.draw()
    layout={} if row_groups is None else {'row_groups':row_groups}
    require_matplotlib_panel_alignment(fig,json_out=R/'alignment.json',overlay_svg=R/'alignment.svg',
      exclude_axes=list(exclude)+list(exclude_axes),exemptions=exemptions,tolerance_pt=1.5,strict=True,**layout)
    (R/'numerical_objects.json').write_text(json.dumps(numeric_artists(fig),ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    (R/'display_text.json').write_text(json.dumps([t.get_text() for t in fig.findobj(Text) if t.get_text()],ensure_ascii=False,indent=2),encoding='utf-8')
    fig.savefig(R/(FIGURE+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(R/(FIGURE+'.svg'),metadata={'Date':None})
    fig.savefig(R/(FIGURE+'.png'),dpi=dpi)
    fig.savefig(R/(FIGURE+'.tiff'),dpi=dpi,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(R/'preview.png',dpi=180)
    with Image.open(R/(FIGURE+'.png')) as a,Image.open(R/(FIGURE+'.tiff')) as b:
        assert a.size==b.size and np.array_equal(np.asarray(a),np.asarray(b))
        pixels=list(a.size);res=list(a.info['dpi'])
    record={'figure':FIGURE,'size_mm':list(fig.get_size_inches()*25.4),'pixels':pixels,'dpi':res,
      'effective_dpi_in_baseline_document':min(pixels[i]/config['baseline']['display_inches'][i] for i in range(2)),
      'PNG_TIFF_same_pixels':True,'minimum_object_font_pt':min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text()),
      'files':{ext:{'path':FIGURE+'.'+ext,'sha256':sha(R/(FIGURE+'.'+ext))} for ext in ['png','tiff','pdf','svg']}}
    assert record['effective_dpi_in_baseline_document']>=600
    EXPORTS.append(record);plt.close(fig)
    print('RENDERED '+FIGURE,flush=True)

exec(compile((ROOT/('code/baseline_plot.py' if os.environ.get('SUPP_CAPTURE_BASELINE')=='1' else 'code/plot.py')).read_text(encoding='utf-8'),str(ROOT/('code/baseline_plot.py' if os.environ.get('SUPP_CAPTURE_BASELINE')=='1' else 'code/plot.py')),'exec'),globals())
globals()[config['entry_function']]()
for relative,h in SOURCES.items():assert sha(ROOT/relative)==h
assert len(EXPORTS)==1,EXPORTS
panels=[]
for item in LEDGER:
    ids=re.split(r'[_ ,]+',item['panel'])
    for pid in ids:
        if not pid:continue
        panels.append({'id':pid,'input_files':[item['input_relative_path']] if item['input_relative_path'] else [x['relative_path'] for x in manifest['inputs']],
          'fields':item['fields'],'filter':item['filter'],'sample_scope':item['sample_scope'],
          'units':item['units_from_source'] or {'source':'scientific caption and source-level axis strings','caption':config['caption']},
          'normalization':{'executed_entry':config['entry_function'],'role':item['role'],'caption':config['caption']},
          'difference_direction':{'role':item['role'],'caption':config['caption']},
          'weights':{'source':'Frozen upstream estimates; plotting code never refits or recomputes weights. See caption and copied source fields.','caption':config['caption']},
          'CI_source':{'copied_fields':[c for c in item['fields'] if any(k in c.lower() for k in ['ci','lower','upper','bootstrap','low','high'])],'caption':config['caption']},
          'statistical_generation_provenance':{'frozen_source_inputs':manifest['inputs'],'original_plot_code':manifest['provenance'],'statistical_recalculation':False},
          'selected_data_sha256':item['filter']['selected_rows_sha256']})
manifest.update(status='RENDERED_PENDING_INDEPENDENT_NUMERICAL_TEXT_VISUAL_QA',panels=panels,outputs=EXPORTS[0],
 QA={'alignment':json.loads((R/'alignment.json').read_text()),'raster_pair_equal':True,'numerical_objects':'numerical_objects.json','display_text':'display_text.json','visual_review':'PENDING','PDF_text_and_collision':'PENDING'},
 runtime={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':mpl.__version__,'Pillow':Image.__version__,'temporary_root':str(R/'tmp'),'scoped_platform_WMI_avoidance':sys.platform=='win32'})
(R/'render_receipt.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
(R/'panel_ledger.json').write_text(json.dumps(LEDGER,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps({'status':manifest['status'],'figure':FIGURE,'panels':len(panels),'source_files':len(SOURCES),'output':str(R)},ensure_ascii=False),flush=True)
