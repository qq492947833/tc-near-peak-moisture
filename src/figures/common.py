"""Shared source-data readers and vector/raster export for the NPJ figures."""
from pathlib import Path
import os, sys, json, ast, hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Ellipse, Arc
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.text import Text
from PIL import Image
import shapefile
from shapely.geometry import LineString, box
ROOT=Path(os.environ.get('FLTCCF_REPO_ROOT',str(Path(__file__).absolute().parents[2])))
DATA=ROOT/'source_data/figures'
OUTPUT=ROOT/'outputs/figures'
DPI=int(os.environ.get('FLTCCF_FIGURE_DPI','601'))
FORMATS=os.environ.get('FLTCCF_FIGURE_FORMATS','png,tiff,pdf,svg').split(',')
MANIFEST=[]
SELECTED=None
def read_csv(path):
    return pd.read_csv(path,keep_default_na=False,na_values=['','nan','NaN'],low_memory=False)
def export(fig,name,**kwargs):
    if SELECTED is not None and name not in SELECTED:
        plt.close(fig);return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for t in fig.findobj(Text):t.set_fontfamily('DejaVu Sans')
    fig.canvas.draw()
    for ext in FORMATS:
        kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        if ext=='pdf':kw['metadata']={'CreationDate':None,'ModDate':None}
        if ext=='svg':kw['metadata']={'Date':None}
        fig.savefig(OUTPUT/(name+'.'+ext),dpi=DPI,**kw)
    axes=[dict(id=a.get_gid(),bounds=list(a.get_position().bounds),xlim=list(a.get_xlim()),ylim=list(a.get_ylim()),xlabel=a.get_xlabel(),ylabel=a.get_ylabel(),title=a.get_title(loc='left')) for a in fig.axes]
    numeric=[]
    for ax in fig.axes:
        rows={'lines':[],'collections':[],'patches':[]}
        for line in ax.lines:rows['lines'].append([np.asarray(line.get_xdata()).tolist(),np.asarray(line.get_ydata()).tolist()])
        for col in ax.collections:
            row={}
            if hasattr(col,'get_offsets'):row['offsets']=np.asarray(col.get_offsets()).tolist()
            if hasattr(col,'get_segments'):row['segments']=[np.asarray(x).tolist() for x in col.get_segments()]
            if col.get_array() is not None:row['array']=np.asarray(col.get_array()).tolist()
            rows['collections'].append(row)
        for p in ax.patches:
            if isinstance(p,Rectangle):rows['patches'].append([p.get_x(),p.get_y(),p.get_width(),p.get_height()])
        numeric.append(rows)
    numeric_hash=hashlib.sha256(json.dumps(numeric,sort_keys=True,allow_nan=True,default=str).encode()).hexdigest()
    MANIFEST.append(dict(figure=name,size_mm=list(fig.get_size_inches()*25.4),dpi=DPI,axes=axes,numeric_artist_sha256=numeric_hash))
    plt.close(fig)
    print('Rendered '+name,flush=True)
def preserve_bound(job):
    return None
class Job:
    def __init__(self,name):self.name=name;self.data={};self.mapping=[];self.rows=[];self.p=OUTPUT
    def load(self,binding):
        r=json.loads((DATA/'legacy/bindings.json').read_text(encoding='utf-8'))[binding]
        p=DATA/r['file']
        if binding not in self.data:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
            self.data[binding]=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,keep_default_na=False,low_memory=False)
        return self.data[binding].copy()
    def bind(self,*args,**kwargs):return None
    def save(self,fig,*args,**kwargs):export(fig,LEGACY_NAMES[self.name])
LEGACY_NAMES={'SF-A':'FigS1','SF-B':'FigS2','SF-D':'FigS4','SF-E':'FigS5','SF-M':'FigS7','SF-F':'FigS9','SF-L':'FigS15','SF-N':'FigS26'}
