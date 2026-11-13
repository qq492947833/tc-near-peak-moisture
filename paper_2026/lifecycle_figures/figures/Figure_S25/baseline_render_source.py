from pathlib import Path
import os,sys,json,hashlib,re,zipfile,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm,Normalize,LogNorm
from matplotlib.patches import Circle,Wedge,Rectangle,FancyArrowPatch,Ellipse,Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PIL import Image
BLUE=TEMP='#167B9E';ORANGE=FINAL='#CC7445';INK='#233A46';MUTED='#677B86'
TEAL,PURPLE,GREY='#2C8078','#9069A3','#7F9099';GREEN='#418D79'
STAGES={'D':'#437F9C','E':'#428C79','F':'#BD9145','Res':'#AB685E'}
CMAP=LinearSegmentedColormap.from_list('signed_native',[ORANGE,'#FAFAF7',BLUE])
WIDTH_MM=180; WIDTH_IN=180/25.4; GRID='#DDE3E7'; LEDGER=[]; FILES={}; CAPTIONS={}
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'savefig.facecolor':'white'})

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
inputs=types.SimpleNamespace(PRIMARY=PRIMARY,primary_mask=primary_mask,display_intervals=display_intervals)

def draw_levels(frames):
    import matplotlib.colors as colors
    plt, _ = setup()
    fig, axes = plt.subplots(3, 2, figsize=(180 / 25.4, 171 / 25.4), layout=None)
    fig.subplots_adjust(left=0.13, right=0.9, top=0.94, bottom=0.08, hspace=0.68, wspace=0.66)
    color_axes = []
    for i, (metric, ax) in enumerate(zip(inputs.PRIMARY, axes.ravel())):
        data = frames['pre']
        data = data[data.scheme.eq('main') & data.region.eq('r0_300') & data.metric.eq(metric)]
        levels = ['500', '700', '850'] if i < 3 else ['500-850']
        array = data.pivot(index='level', columns='window', values='estimate').reindex(index=levels, columns=[0, 1, 2]).to_numpy(float)
        limit = float(np.nanmax(np.abs(array))) if np.isfinite(array).any() else 1.0
        if limit == 0:
            limit = 1.0
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#D9DDDF')
        im = ax.imshow(np.ma.masked_invalid(array), origin='upper', aspect='auto', cmap=cmap, norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        for y in range(len(levels)):
            for x in range(3):
                value = array[y, x]
                text = 'NA' if not np.isfinite(value) else format(value, '.2g').replace('-', '−')
                ax.text(x, y, text, ha='center', va='center', fontsize=7.5, color='white' if np.isfinite(value) and abs(value) > 0.62 * limit else '#243741')
        ax.set_xticks([0, 1, 2], ['>16 d', '2–16 d', '<2 d'])
        ax.set_yticks(range(len(levels)), levels)
        ax.set_ylabel('hPa' if i < 3 else 'Layer (hPa)')
        decorate(ax, LABELS[metric], 'abcdef'[i])
        ax.set_yticks(range(len(levels)), levels)
        bar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.05)
        bar.ax.set_ylabel('Δ ' + UNITS[metric])
        color_axes.append(bar.ax)
    return (fig, {'axes': list(axes.ravel()), 'panel_ids': list('abcdef'), 'exclude_axes': color_axes, 'row_groups': [['a', 'b'], ['c', 'd'], ['e', 'f']], 'column_groups': [['a', 'c', 'e'], ['b', 'd', 'f']]})
def build():
    frames={k:load(k) for k in ['pre']}
    for k,d in frames.items():
        if "level" in d:d["level"]=d["level"].astype(str)
        table("Figure_S17", k, d, "Frozen precomputed display source; selected below by original plotting statements")
    fig,layout=draw_levels(frames)
    for a,i in zip(layout["axes"],layout["panel_ids"]):a.set_gid(i)
    export(fig,"Figure_S17",**layout)
