from pathlib import Path
from display_format import format_number, nice_symmetric_limit
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

def draw_core(frames):
    plt, _ = setup()
    fig = plt.figure(figsize=(180 / 25.4, 198 / 25.4))
    grid = fig.add_gridspec(3, 2, left=0.115, right=0.975, top=0.918, bottom=0.065, hspace=0.37, wspace=0.38)
    axes = []
    ids = []
    rows = []
    for i, metric in enumerate(inputs.PRIMARY):
        row = i // 2 * 2
        col = i % 2
        letter = 'abcdef'[i]
        cell = grid[i // 2, col].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.14)
        ax = fig.add_subplot(cell[0, 0])
        delta = fig.add_subplot(cell[1, 0])
        axes.extend([ax, delta])
        ids.extend([letter + '_mean', letter + '_difference'])
        data = frames['time'][inputs.primary_mask(frames['time']) & frames['time'].metric.eq(metric)].sort_values('lag_h')
        point = frames['primary'][frames['primary'].metric.eq(metric)].iloc[0]
        ax.plot(data.lag_h, data.temporary_mean, color=STYLE['temporary'], lw=1.25, label='Temporary departure')
        ax.plot(data.lag_h, data.final_mean, color=STYLE['final'], lw=1.25, ls='--', label='Final departure')
        ax.axvline(0, color='#969EA2', lw=0.6, ls=':')
        ax.set_xlim(-26, 39)
        ax.set_xticks([-24, -12, 0, 12, 24])
        ax.set_xticklabels([])
        ax.set_ylabel(UNITS[metric])
        decorate(ax, LABELS[metric], letter)
        display = inputs.display_intervals(data)
        delta.axhline(0, color='#989FA3', lw=0.6)
        delta.axvline(0, color='#ADB4B8', lw=0.6, ls=':')
        delta.plot(display.lag_h, display.estimate, color=STYLE['difference'], lw=1)
        delta.fill_between(display.lag_h.to_numpy(float), display.display_lower.to_numpy(float), display.display_upper.to_numpy(float), color=STYLE['difference'], alpha=0.16, linewidth=0)
        if np.isfinite(point.estimate):
            if point.n_sid >= 2 and np.isfinite(point.lower) and np.isfinite(point.upper):
                delta.vlines(34, point.lower, point.upper, color='black', lw=1.1)
                delta.plot([34, 34], [point.lower, point.upper], '_', color='black', ms=4)
            delta.plot(34, point.estimate, 'D', color='black', mfc='black' if point.n_sid >= 2 else 'white', ms=3)
        delta.axvline(28, color='#D6DADD', lw=0.6)
        delta.set_xlim(-26, 39)
        delta.set_xticks([-24, 0, 24, 34], ['−24', '0', '24', 'Pre'])
        delta.set_ylabel('Δ ' + UNITS[metric])
        decorate(delta, '')
        if row == 4:
            delta.set_xlabel('Hours from departure')
    # Display-only coordination: four uniformly spaced major ticks per y-axis.
    # Autoscaled data/interval extents are preserved; bounds only expand outward.
    coordinated_ticks = {
        'a_mean': [70, 80, 90, 100],
        'a_difference': [-20, -10, 0, 10],
        'b_mean': [2, 3, 4, 5],
        'b_difference': [-4.5, -3, -1.5, 0],
        'c_mean': [0.00015, 0.00030, 0.00045, 0.00060],
        'c_difference': [-0.00045, -0.00030, -0.00015, 0],
        'd_mean': [-0.00005, 0, 0.00005, 0.00010],
        'd_difference': [-0.0002, -0.0001, 0, 0.0001],
        'e_mean': [-0.000135, -0.000120, -0.000105, -0.000090],
        'e_difference': [0, 0.000025, 0.000050, 0.000075],
        'f_mean': [-0.0006, -0.0002, 0.0002, 0.0006],
        'f_difference': [0, 0.00025, 0.00050, 0.00075],
    }
    for axis, panel_id in zip(axes, ids):
        bounds = axis.get_ylim()
        ticks = coordinated_ticks[panel_id]
        axis.set_yticks(ticks)
        axis.set_ylim(min(bounds[0], min(ticks)), max(bounds[1], max(ticks)))
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, bbox_to_anchor=(0.54, 0.994), frameon=False)
    for i in range(3):
        rows.extend([[ids[i * 4], ids[i * 4 + 2]], [ids[i * 4 + 1], ids[i * 4 + 3]]])
    columns = [[ids[i] for i in [0, 4, 8]], [ids[i] for i in [2, 6, 10]], [ids[i] for i in [1, 5, 9]], [ids[i] for i in [3, 7, 11]]]
    return (fig, {'axes': axes, 'panel_ids': ids, 'row_groups': rows, 'column_groups': columns})
def build():
    frames={k:load(k) for k in ['time', 'primary']}
    for k,d in frames.items():
        if "level" in d:d["level"]=d["level"].astype(str)
        table("Figure_08", k, d, "Frozen precomputed display source; selected below by original plotting statements")
    fig,layout=draw_core(frames)
    for a,i in zip(layout["axes"],layout["panel_ids"]):a.set_gid(i)
    export(fig,"Figure_08",**layout)
