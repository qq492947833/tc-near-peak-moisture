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
CMAP=LinearSegmentedColormap.from_list('light_signed',['#DBAA8D','#FAFAF8','#9FC3D2'])
WIDTH_MM=180; WIDTH_IN=180/25.4; GRID='#DDE3E7'; LEDGER=[]; FILES={}; CAPTIONS={}
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'savefig.facecolor':'white'})

BLUE, ORANGE, INK = ('#167B9E', '#CC7445', '#233A46')
TEAL, PURPLE, GREY = ('#2C8078', '#9069A3', '#7F9099')
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
CMAP = LinearSegmentedColormap.from_list('light_signed', ['#DBAA8D', '#FAFAF8', '#9FC3D2'])
WIDTH_MM = 180
FINAL_DPI = 900
PREVIEW_RESOLUTION = 180
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'npj-20261021-main1-6', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'legend.frameon': False, 'savefig.facecolor': 'white'})

def zero(ax):
    ax.axhline(0, color='#9BA5AA', lw=0.65, zorder=0)
    ax.grid(axis='y', color='#E8EDEF', lw=0.5, zorder=0)
    ax.set_axisbelow(True)

def fig3(dpi):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 154 / 25.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.72, 0.4])
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.105, wspace=0.55, hspace=0.75)
    src = 'direct/fig5_trajectories.csv'
    tr = rd(src)
    variables = [('midRH', 'Relative humidity', 'RH (%)', 'ΔRH (pp)'), ('shear200_850', 'Deep-layer shear', 'Shear (m s⁻¹)', 'ΔShear (m s⁻¹)'), ('PI_headroom', 'PI headroom', 'PI − V (m s⁻¹)', 'ΔHeadroom (m s⁻¹)')]
    for j, (var, heading, yl, dyl) in enumerate(variables):
        for r in [0, 1]:
            ax = fig.add_subplot(gs[r, j])
            letter = chr(97 + r * 3 + j)
            for s in ['internal', 'final'] if r == 0 else ['difference']:
                z = tr[tr.variable.eq(var) & tr.series.eq(s)].sort_values('lag_h')
                col = {'internal': BLUE, 'final': ORANGE, 'difference': INK}[s]
                ax.plot(z.lag_h, z.estimate, color=col, lw=1.4)
                ax.fill_between(z.lag_h, z.lo, z.hi, color=col, alpha=0.16, lw=0)
                record('Fig3', letter, src, z, 'Absolute trajectory' if r == 0 else 'Saved direct paired difference and its own interval')
            ax.axvline(0, color=GREY, lw=0.65, ls='--')
            if r:
                zero(ax)
                ax.set_xlabel('Time from departure (h)')
            ax.set(xticks=[-24, 0, 24], ylabel=yl if r == 0 else dyl)
            title(ax, letter, heading if r == 0 else 'Temporary − final')
    fig.legend([Line2D([0], [0], color=c, lw=1.5) for c in [BLUE, ORANGE]], ['Temporary departure', 'Final departure'], loc='upper center', bbox_to_anchor=(0.55, 0.98), ncol=2)
    ax = fig.add_subplot(gs[2, :])
    src = 'direct/environmental_contrasts.csv'
    z = rd(src)
    z = z[z.variable.eq('midRH')]
    geo = z[z.model.eq('geographical')].iloc[0]
    state = z[z.model.eq('geographical_state')].iloc[0]
    values = [geo.original_raw, geo.overlap_raw, geo.standardized, state.standardized]
    tiles(ax, values, ['Original', 'Overlap, raw', 'Geography + season', '+ Cyclone state'], 'Pre-event ΔRH\n(pp)')
    title(ax, 'g', 'Humidity contrast on comparable measured backgrounds')
    record('Fig3', 'g', src, z, 'All original, restricted and adjusted means, CI, model/support fields preserved')
    export(fig, 'Fig3', dpi)

def celltext(ax, x, y, v, norm, fontsize=8.5):
    ax.text(x, y, fmt(v), ha='center', va='center', fontsize=fontsize, color='black')

def fmt(v):
    return format_number(v, decimals=2)

def norm_for(z, limit=None):
    m = nice_symmetric_limit(max(0.01, float(np.nanmax(abs(np.asarray(z, float)))))) if limit is None else limit
    return TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)

def title(ax, letter, text):
    ax.set_gid(letter)
    ax.set_title(text, loc='left', pad=13)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-20, 12), textcoords='offset points', fontsize=10, weight='bold', va='bottom')

def tiles(ax, values, labels, unit):
    z = np.asarray(values, float)
    norm = norm_for(z, 4.6)
    assert np.nanmax(abs(z)) <= 4.6
    ax.set(xlim=(-0.5, len(z) - 0.5), ylim=(-0.5, 0.5), xticks=range(len(z)), xticklabels=labels, yticks=[])
    for j, v in enumerate(z):
        ax.add_patch(Rectangle((j - 0.47, -0.34), 0.94, 0.68, facecolor=CMAP(norm(v)), edgecolor='white'))
        celltext(ax, j, 0, v, norm, 9)
    ax.tick_params(length=0, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel(unit, labelpad=10)