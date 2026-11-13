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
    from terminology_20261103 import apply as revise_terminology
    revise_terminology(fig,FIGURE,R)
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
    fig.savefig(R/'preview.png',dpi=300)
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

from display_format import format_number,format_group
MASTER_SHA = '0b4c4308543fb4e7f5f32911d6b654a5fc4e720b8617fc53ce46a59b2c117b37'
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
TEMP = BLUE = '#167B9E'
FINAL = ORANGE = '#CC7445'
INK = '#233A46'
MUTED = '#677B86'
GREEN = '#418D79'
RINGS = ['100_300km', '300_800km']
RINGLABELS = ['100–300 km', '300–800 km']
CMAP = LinearSegmentedColormap.from_list('signed_native', [ORANGE, '#FAFAF7', BLUE])
VAR = ['X1_sst_c', 'X2_shear_ms', 'X3_mid_rh_pct', 'X4_d200_1e5s', 'X5_minus_omega500_pa_s', 'X6_signed_vo850_1e5s']
VL = ['SST', 'Deep shear', 'Mid-level RH', 'Divergence', 'Ascent', 'Vorticity']
BS = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']
BC = dict(zip(BS, ['#437F9C', '#BD9145', '#428C79', '#9D82AB', '#AB685E', '#67A4AC']))
BASINS = BS
WIDTH_MM = 180
WIDTH_IN = 180 / 25.4
GRID = '#DDE3E7'
SCOPES = ['main_all', 'exclude_single_6h', 'strict_bridge_tropical', 'without_Freddy2023', 'without_basin_WP', 'without_basin_EP', 'without_basin_SI', 'without_basin_NA', 'without_basin_SP', 'without_basin_NI']

def pick(df, **conditions):
    d = df
    for k, v in conditions.items():
        d = d[d[k].eq(v)]
    assert len(d) == 1, (conditions, len(d))
    return d.iloc[0]

def decorate(ax, letter, title, pad=16):
    ax.set_gid(letter)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-16, pad - 4), textcoords='offset points', weight='bold', fontsize=10.5, ha='left', va='bottom')
    ax.set_title(title, loc='left', pad=pad, fontweight='normal')

def grid(rows=2, cols=2, height=170, left=0.13, hspace=0.72, wspace=0.45):
    fig, axs = plt.subplots(rows, cols, figsize=(180 / 25.4, height / 25.4), squeeze=False)
    fig.subplots_adjust(left=left, right=0.965, bottom=0.11, top=0.91, wspace=wspace, hspace=hspace)
    return (fig, axs)

def bars(ax, data, keys, groups, labelmap=None, colors=None, xkey='field', groupkey='policy', est='estimate', low='lo', high='hi', scale=1, uncertainty=True):
    xx = np.arange(len(keys))
    width = 0.72 / len(groups)
    for i, g in enumerate(groups):
        sub = data[data[groupkey].eq(g)].set_index(xkey).loc[keys]
        e = sub[est].to_numpy(float) / scale
        color = (colors or {}).get(g, [BLUE, ORANGE, GREY][i % 3])
        pos = xx + (i - (len(groups) - 1) / 2) * width
        ax.bar(pos, e, width * 0.86, color=color, label=(labelmap or {}).get(g, g), zorder=2)
        if uncertainty:
            lo = sub[low].to_numpy(float) / scale
            hi = sub[high].to_numpy(float) / scale
            valid = np.isfinite(lo) & np.isfinite(hi)
            ax.errorbar(pos[valid], e[valid], yerr=[e[valid] - lo[valid], hi[valid] - e[valid]], fmt='none', ecolor=INK, elinewidth=0.75, capsize=2, zorder=3)
    ax.axhline(0, color='#9BA5AA', lw=0.65, zorder=1)
    ax.set_xticks(xx)
    ax.yaxis.set_major_locator(MaxNLocator(4))

def heat(ax, values, xlabels, ylabels, vmax=None, fmt='.2f', cmap='RdBu_r', mark=None):
    values = np.asarray(values, float)
    vmax = max(np.nanmax(np.abs(values)), 1e-10) if vmax is None else vmax
    im = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0, pad=6)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                text = format_number(values[i, j])
                ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if abs(values[i, j]) > 0.63 * vmax else INK)
                if mark is not None and mark[i, j]:
                    ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, lw=1.2, zorder=5))
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im

def waterfall(ax, vals, labels, colors=None, errors=None):
    """Show one exact decomposition; final item is a saved total, not another addend."""
    vals = np.asarray(vals, float)
    assert np.isclose(vals[:-1].sum(), vals[-1], rtol=1e-09, atol=1e-10), (vals, vals[:-1].sum())
    pos = np.arange(len(vals))
    start = 0.0
    for i, v in enumerate(vals):
        b = 0 if i == len(vals) - 1 else start
        color = (colors or [BLUE, ORANGE, GREY, INK])[i]
        ax.bar(i, v, 0.65, bottom=b, color=color, zorder=2)
        if i < len(vals) - 1:
            start += v
            ax.plot([i + 0.33, i + 0.67], [start, start], color='#ABB4B9', lw=0.7)
        if errors is not None:
            lo, hi = errors[i]
            ax.errorbar(i, b + v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor=INK, capsize=2, lw=0.7)
    ax.axhline(0, color='#9BA5AA', lw=0.65)
    ax.set_xticks(pos, labels)
    edges = np.r_[0.0, np.cumsum(vals[:-1]), vals[-1]]
    lo, hi = (float(edges.min()), float(edges.max()))
    pad = max((hi - lo) * 0.12, 1e-08)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(MaxNLocator(4))

def fig16():
    name = 'FigS16'
    src = 'thematic/fig6_regions.csv'
    d = read(src)
    sectors = ['downshear', 'left_of_shear', 'upshear', 'right_of_shear']
    angles = [(45, 135), (135, 225), (225, 315), (315, 405)]
    fig, ax = grid(1, 2, 103, left=0.09, hspace=0.5, wspace=0.27)
    cbaxes = []
    for j, (field, title, unit) in enumerate([('q700_asymmetric', 'Moisture azimuthal deviation', 'g kg⁻¹'), ('cyclonic_tangential_wind700_asymmetric', 'Circulation azimuthal deviation', 'm s⁻¹')]):
        z = d[d.field.eq(field) & d.estimand.eq('pre_mean') & d.sector.isin(sectors)]
        vmax = max(abs(z.original_estimate))
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        for ring, inner, outer in [('100_300km', 100, 300), ('300_800km', 300, 800)]:
            for sector, (t1, t2) in zip(sectors, angles):
                r = pick(z, region=ring, sector=sector)
                ax[0, j].add_patch(Wedge((0, 0), outer, t1, t2, width=outer - inner, facecolor=plt.get_cmap('RdBu_r')(norm(r.estimate)), edgecolor='white', lw=1.8))
        for radius in [100, 300, 800]:
            ax[0, j].add_patch(Circle((0, 0), radius, fill=False, edgecolor='#ADB7BC', lw=0.6))
        ax[0, j].annotate('', xy=(0, 990), xytext=(0, 820), arrowprops={'arrowstyle': '->', 'color': INK, 'lw': 1.4})
        ax[0, j].text(0, 1060, 'Downshear', ha='center', fontsize=8)
        ax[0, j].text(0, -1000, 'Upshear', ha='center', fontsize=8)
        ax[0, j].text(-930, 0, 'Left', ha='right', va='center', fontsize=8)
        ax[0, j].text(930, 0, 'Right', ha='left', va='center', fontsize=8)
        ax[0, j].set(xlim=(-1250, 1250), ylim=(-1100, 1150))
        ax[0, j].set_aspect('equal')
        ax[0, j].axis('off')
        decorate(ax[0, j], chr(97 + j), title.replace(' azimuthal', '\nazimuthal'))
        ax[0, j].apply_aspect()
        p = ax[0, j].get_position()
        ca = fig.add_axes([p.x0 + 0.04*p.width, 0.1, 0.92*p.width, 0.025])
        cbaxes.append(ca)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), cax=ca, orientation='horizontal', ticks=np.arange(-2, 3) * (0.04 if j == 0 else 0.2))
        cb.set_label('Temporary − final (' + unit + ')')
        log(name, chr(97 + j), src, z, 'Real four shear-relative sectors and two annuli; CI/n retained in statistics')
    finish(fig, name, exclude=cbaxes)

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
