"""Portable deterministic rendering from packaged frozen data; no scientific fitting."""
from pathlib import Path
import argparse,os,sys,json,hashlib,math,platform,importlib.util
PKG=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=PKG/'rebuilt');ap.add_argument('--dpi',type=int,default=900);ap.add_argument('--dependency-path',type=Path)
args=ap.parse_args();OUT=args.output_dir.resolve();assert args.dpi>=600
OUT.mkdir(parents=True,exist_ok=True)
for d in ['tmp','cache/matplotlib','qa','previews']:(OUT/d).mkdir(parents=True,exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TEMP=str(OUT/'tmp'),TMP=str(OUT/'tmp'),TMPDIR=str(OUT/'tmp'),MPLCONFIGDIR=str(OUT/'cache/matplotlib'),XDG_CACHE_HOME=str(OUT/'cache'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
vendor=args.dependency_path or PKG.parents[1]/'environment/vendor'
if vendor.exists():sys.path.insert(0,str(vendor))
sys.path.insert(0,str(PKG/'code'))
from bootstrap import apply
apply(OUT)
import numpy as np,pandas as pd,matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from audit_panel_alignment import require_matplotlib_panel_alignment
from supp_style import audit_and_style
CFG=json.loads((PKG/'config.json').read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x
def dump(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def snapshot(fig):
    allaxes=[]
    for ax in fig.axes:
        d=dict(id=ax.get_gid(),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),title=ax.get_title(loc='left'),xlim=ax.get_xlim(),ylim=ax.get_ylim(),xticks=ax.get_xticks(),yticks=ax.get_yticks(),position=ax.get_position().bounds,
          texts=[dict(text=t.get_text(),font_pt=t.get_fontsize(),family=t.get_fontfamily(),color=t.get_color()) for t in ax.findobj(Text) if t.get_text()],
          lines=[dict(x=l.get_xdata(),y=l.get_ydata(),color=l.get_color(),lw=l.get_linewidth(),ls=l.get_linestyle(),marker=l.get_marker(),ms=l.get_markersize()) for l in ax.lines],
          collections=[dict(offsets=c.get_offsets(),array=c.get_array(),paths=[p.vertices for p in c.get_paths()],facecolor=c.get_facecolor(),edgecolor=c.get_edgecolor()) for c in ax.collections],
          patches=[dict(path=p.get_path().vertices,transform=p.get_patch_transform().get_matrix(),face=p.get_facecolor(),edge=p.get_edgecolor()) for p in ax.patches])
        allaxes.append(d)
    return clean(dict(axes=allaxes,figure_text=[dict(text=t.get_text(),fontsize=t.get_fontsize()) for t in fig.texts]))
def emit(fig,name=None,dpi=None,exclude=(),row_groups=None,column_groups=None):
    name=CFG['figure_id'];audit_and_style(fig,name,OUT);fig.canvas.draw();kwargs={}
    if row_groups is not None:kwargs['row_groups']=row_groups
    if column_groups is not None:kwargs['column_groups']=column_groups
    alignment=require_matplotlib_panel_alignment(fig,json_out=OUT/'qa/alignment.json',overlay_svg=OUT/'qa/alignment.svg',exclude_axes=list(exclude),tolerance_pt=1.5,strict=True,**kwargs)
    dump(OUT/'numerical_objects.json',snapshot(fig))
    for ext in ['pdf','svg','png','tiff']:
        kw={'dpi':args.dpi}
        if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
        if ext=='pdf':kw['metadata']={'CreationDate':None,'ModDate':None}
        if ext=='svg':kw['metadata']={'Date':None}
        fig.savefig(OUT/(name+'.'+ext),**kw)
    fig.savefig(OUT/'previews/full.png',dpi=300)
    for i,ax in enumerate(fig.axes):
        if ax in exclude or ax.get_label()=='<colorbar>':continue
        box=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.08)
        fig.savefig(OUT/'previews'/('panel_'+str(ax.get_gid() or i)+'.png'),dpi=300,bbox_inches=box)
    with Image.open(OUT/(name+'.png')) as a,Image.open(OUT/(name+'.tiff')) as b:
        assert a.size==b.size and a.mode==b.mode and a.tobytes()==b.tobytes()
        pixels=list(a.size);dpi=list(a.info['dpi'])
    files={ext:dict(relative_path=name+'.'+ext,sha256=sha(OUT/(name+'.'+ext))) for ext in ['png','tiff','pdf','svg']}
    ref=PKG/'reference/baseline.png';comparison=dict(method='Source AST and frozen input identity; numeric objects exported before writing; raster comparison is QA only')
    if ref.exists():
        with Image.open(ref) as a,Image.open(OUT/(name+'.png')) as b:
            comparison.update(baseline_size=list(a.size),rebuilt_size=list(b.size))
            if a.size==b.size:
                aa=np.asarray(a.convert('RGB'));bb=np.asarray(b.convert('RGB'))
                comparison.update(pixel_identical=bool(np.array_equal(aa,bb)),different_pixel_count=int(np.any(aa!=bb,axis=2).sum()),mean_absolute_channel_difference=float(np.abs(aa.astype(float)-bb).mean()))
    dump(OUT/'render_manifest.json',dict(status='RENDERED_PENDING_INDEPENDENT_QA',figure_id=name,files=files,size_mm=fig.get_size_inches()*25.4,pixels=pixels,dpi=dpi,png_tiff_identical=True,alignment=alignment.get('verdict'),comparison=comparison,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=mpl.__version__),inputs=[dict(path=str(p.relative_to(PKG)),sha256=sha(p)) for p in (PKG/'input').rglob('*') if p.is_file()],new_statistics=False,no_generative_image_service=True,no_raster_input_to_plot=True))
    plt.close(fig)
spec=importlib.util.spec_from_file_location('plot',PKG/('code/baseline_plot.py' if os.environ.get('SUPP_CAPTURE_BASELINE')=='1' else 'code/plot.py'));p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.SRC=PKG/'input';p.OUT=OUT;p.QA=OUT/'qa';p.ROOT=OUT;p.emit=emit
if CFG['kind']=='structural':p.save=emit
p.build(args.dpi)
print(json.dumps(dict(status='RENDERED',figure=CFG['figure_id'],output=str(OUT))))
