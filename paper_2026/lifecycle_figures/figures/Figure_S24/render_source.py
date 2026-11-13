from display_format import format_number,format_group
from pathlib import Path
import os, sys, json, hashlib, re, zipfile, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, LogNorm
from matplotlib.patches import Circle, Wedge, Rectangle, FancyArrowPatch, Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PIL import Image
BLUE = TEMP = '#167B9E'
ORANGE = FINAL = '#CC7445'
INK = '#233A46'
MUTED = '#677B86'
TEAL, PURPLE, GREY = ('#2C8078', '#9069A3', '#7F9099')
GREEN = '#418D79'
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
CMAP = LinearSegmentedColormap.from_list('signed_native', [ORANGE, '#FAFAF7', BLUE])
WIDTH_MM = 180
WIDTH_IN = 180 / 25.4
GRID = '#DDE3E7'
LEDGER = []
FILES = {}
CAPTIONS = {}
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'savefig.facecolor': 'white'})
STYLE = {'temporary': '#287FA0', 'final': '#BC693F', 'difference': '#33454F', 'windows': ['#6D84A7', '#337C7D', '#A87846'], 'dpi': 900, 'width_mm': 180}
LABELS = {'KE': 'Kinetic energy', 'APE': 'Dry available potential energy', 'C_A_to_K': 'APE → KE conversion', 'vertical_input_K_Wkg': 'Layer KE input', 'vertical_input_A_Wkg': 'Layer APE input', 'vertical_input_P_Wkg': 'Layer pressure-flux input'}
UNITS = {'KE': 'J kg⁻¹', 'APE': 'J kg⁻¹', 'C_A_to_K': 'W kg⁻¹', 'vertical_input_K_Wkg': 'W kg⁻¹', 'vertical_input_A_Wkg': 'W kg⁻¹', 'vertical_input_P_Wkg': 'W kg⁻¹'}
PRIMARY = ['KE', 'APE', 'C_A_to_K', 'vertical_input_K_Wkg', 'vertical_input_A_Wkg', 'vertical_input_P_Wkg']

def primary_mask(frame):
    mask = frame.metric.isin(PRIMARY) & frame.window.eq(1)
    if 'scheme' in frame:
        mask &= frame.scheme.eq('main')
    if 'region' in frame:
        mask &= frame.region.eq('r0_300')
    return mask & (frame.metric.isin(PRIMARY[:3]) & frame.level.eq('700') | frame.metric.isin(PRIMARY[3:]) & frame.level.eq('500-850'))

def display_intervals(frame, lower='lower', upper='upper', n='n_sid'):
    """Presentation copy only; original bootstrap outputs remain untouched."""
    copy = frame.copy()
    copy['interval_interpretable'] = copy[n].ge(2) & np.isfinite(copy[lower]) & np.isfinite(copy[upper])
    copy['display_lower'] = copy[lower].where(copy.interval_interpretable)
    copy['display_upper'] = copy[upper].where(copy.interval_interpretable)
    copy['evidence_status'] = np.select([copy[n].eq(0), copy[n].eq(1), copy.interval_interpretable], ['unavailable', 'one SID descriptive value; interval not interpretable', 'pointwise SID sampling interval'], default='interval unavailable')
    return copy

def setup():
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.6, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'axes.unicode_minus': True})
    return (plt, None)

def decorate(ax, title, letter=None):
    from matplotlib.ticker import MaxNLocator
    ax.set_title(title, loc='left', pad=7)
    if letter:
        ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-18, 8), textcoords='offset points', fontweight='bold', fontsize=9, va='bottom')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(length=2.5, width=0.6, pad=2)

def interval(ax, estimate, lower, upper, n, y=0, color=None):
    color = color or STYLE['difference']
    if not np.isfinite(estimate):
        return
    if n >= 2 and np.isfinite(lower) and np.isfinite(upper):
        ax.hlines(y, lower, upper, color=color, lw=1.1)
        ax.plot([lower, upper], [y, y], '|', color=color, ms=4)
    ax.plot(estimate, y, 'o', ms=3.8, color=color, mfc=color if n >= 2 else 'white')
import types
inputs = types.SimpleNamespace(PRIMARY=PRIMARY, primary_mask=primary_mask, display_intervals=display_intervals)

def draw_support(frames):
    plt, _ = setup()
    fig = plt.figure(figsize=(180 / 25.4, 132 / 25.4))
    outer = fig.add_gridspec(2,1,left=.105,right=.975,top=.925,bottom=.115,hspace=.55,height_ratios=[1,1.05])
    top = outer[0].subgridspec(2,3,hspace=.65,wspace=.42)
    axes=[];ids=[];cov=frames['coverage'];mask=inputs.primary_mask(cov)
    colors=['#287FA0','#BC693F','#4E8462','#876B9D','#80754F','#6B7F8B']
    short=['KE','APE','APE → KE','Layer KE','Layer APE','Layer pressure']
    for i,(metric,col,label) in enumerate(zip(inputs.PRIMARY,colors,short)):
        ax=fig.add_subplot(top[i//3,i%3]);axes.append(ax);ids.append('a'+str(i+1))
        data=cov[mask & cov.metric.eq(metric)].sort_values('lag_h')
        ax.plot(data.lag_h,data.paired_valid_sid,color=col,lw=1.05,marker='o',ms=2.5,label=label)
        ax.set(xlim=(-26,26),ylim=(100,120),xticks=[-24,0,24],yticks=[100,110,120])
        ax.set_title(label,loc='left',pad=6,fontsize=8)
        ax.tick_params(length=2.5,width=.6,pad=2)
        if i%3==0:ax.set_ylabel('Paired SID')
        else:ax.tick_params(axis='y',labelleft=False)
        if i>=3:ax.set_xlabel('Hours from departure')
        if i==0:ax.annotate('a',(0,1),xycoords='axes fraction',xytext=(-18,8),textcoords='offset points',fontweight='bold',fontsize=9,va='bottom')
    lower=outer[1].subgridspec(1,3,wspace=.5)
    ref=frames['reference_size'].sort_values('pressure_hpa')
    hydro=frames['hydro_size'].sort_values('pressure_hpa')
    specs=[(ref,'fixed_joint_aboveground_finite_area_fraction','Fixed-mask coverage','Area fraction'),(ref,'c','Dry-APE reference coefficient','J kg⁻¹ K⁻²'),(hydro,'residual_over_reference_scale','Dry-hydrostatic mismatch','Relative to Rd T̄/p')]
    for i,(data,field,title,unit) in enumerate(specs):
        ax=fig.add_subplot(lower[0,i]);axes.append(ax);ids.append('bcd'[i])
        ax.plot(data[field+'_median'],data.pressure_hpa,color=STYLE['difference'],marker='o',ms=2.5,lw=1)
        ax.fill_betweenx(data.pressure_hpa.to_numpy(float),data[field+'_minimum'].to_numpy(float),data[field+'_maximum'].to_numpy(float),color=STYLE['difference'],alpha=.14,linewidth=0)
        ax.set_ylim(1025,25);ax.set_xlabel(unit);ax.set_ylabel('Pressure (hPa)'if i==0 else'')
        if field=='c':
            ax.set_xscale('log')
            from matplotlib.ticker import ScalarFormatter,NullFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(unit+' (log scale)')
        decorate(ax,title,'bcd'[i])
        ax.set_yticks([100,300,500,700,1000])
        if i==0:ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1])
        if i==2:ax.set_xticks([0,.05,.1,.15,.2])
    rows=[ids[:3],ids[3:6],ids[6:]]
    cols=[[ids[i],ids[i+3]]for i in range(3)]
    return fig,dict(axes=axes,panel_ids=ids,row_groups=rows,column_groups=cols)

def build():
    frames = {k: load(k) for k in ['coverage', 'reference_size', 'hydro_size']}
    for k, d in frames.items():
        if 'level' in d:
            d['level'] = d['level'].astype(str)
        table('Figure_S16', k, d, 'Frozen precomputed display source; selected below by original plotting statements')
    fig, layout = draw_support(frames)
    for a, i in zip(layout['axes'], layout['panel_ids']):
        a.set_gid(i)
    export(fig, 'Figure_S16', **layout)
