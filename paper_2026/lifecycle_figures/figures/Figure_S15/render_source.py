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

def celltext(ax, x, y, v, norm, fontsize=8.5):
    ax.text(x, y, fmt(v), ha='center', va='center', fontsize=fontsize, color='black')

def fig4(dpi):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 160 / 25.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    fig.subplots_adjust(left=0.16, right=0.95, top=0.88, bottom=0.16, wspace=0.16, hspace=0.95)
    src = 'direct/fig6_regions.csv'
    d = rd(src)
    regions = ['100_300km', '300_800km']
    pre = d[d.sector.eq('whole') & d.estimand.eq('pre_mean') & d.field.isin(['q700_raw_common', 'q700_climatology', 'q700_anomaly'])]
    vals = pre.estimate.to_numpy()
    norm = norm_for(vals, 0.6)
    axes = []
    for j, (field, name) in enumerate([('q700_raw_common', 'Observed'), ('q700_climatology', 'Background'), ('q700_anomaly', 'Anomaly')]):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        ax.set_gid('a' + str(j))
        q = pre[pre.field.eq(field)].set_index('region').loc[regions]
        for (radius, width), r in zip([(300, 200), (800, 500)], q.itertuples()):
            ax.add_patch(Wedge((0, 0), radius, 0, 360, width=width, facecolor=CMAP(norm(r.estimate)), edgecolor='white', lw=1))
        for rad in [100, 300, 800]:
            ax.add_patch(Circle((0, 0), rad, fill=False, ec='#66777F', lw=0.45))
        ax.plot(0, 0, '.', color=INK, ms=3)
        ax.set(xlim=(-850, 850), ylim=(-850, 850))
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
        ax.set_title(name, pad=9, fontsize=9)
    axes[0].annotate('a', (0, 1), xycoords='axes fraction', xytext=(-20, 12), textcoords='offset points', fontsize=10, weight='bold', va='bottom')
    fig.text(0.55, 0.965, 'Pre-event humidity: temporary − final', ha='center', fontsize=9.5)
    # Annulus definitions are preserved in the figure caption.
    cax = fig.add_axes([0.31, 0.515, 0.49, 0.012], label='<colorbar>')
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation='horizontal', ticks=[-0.6, 0, 0.6])
    cb.outline.set_visible(False)
    cb.set_label('Specific humidity difference (g kg⁻¹)', fontsize=8, labelpad=3)
    cb.ax.tick_params(labelsize=8, length=2)
    record('Fig4', 'a', src, pre, 'Uniform colours encode the two actual saved annular averages; actual 100/300/800 km radial geometry')
    ax = fig.add_subplot(gs[1, :])
    cats = ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency']
    z = d[d.sector.eq('whole') & d.estimand.eq('integral_0_24') & d.field.isin(cats)]
    vals = [[float(z[z.region.eq(r) & z.field.eq(k)].estimate.iloc[0]) for k in cats] for r in regions]
    cax2 = matrix(fig, ax, vals, ['100–300 km', '300–800 km'], ['Horizontal', 'Vertical', 'Residual', 'Net tendency'], '0–24 h integrated difference (g kg⁻¹)', limit=1.0)
    title(ax, 'b', 'Moisture-budget accounting')
    record('Fig4', 'b', src, z, 'All four original terms; net = horizontal + vertical + residual; original shared budget support')
    export(fig, 'Fig4', dpi, exclude=[cax, cax2])

def fmt(v):
    return format_number(v, decimals=2)

def norm_for(z, limit=None):
    m = nice_symmetric_limit(max(0.01, float(np.nanmax(abs(np.asarray(z, float)))))) if limit is None else limit
    return TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m)

def title(ax, letter, text):
    ax.set_gid(letter)
    ax.set_title(text, loc='left', pad=13)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-20, 12), textcoords='offset points', fontsize=10, weight='bold', va='bottom')

def matrix(fig, ax, values, rows, cols, unit, limit=None, values_text=True):
    z = np.asarray(values, float)
    norm = norm_for(z, limit)
    mesh = ax.pcolormesh(np.arange(z.shape[1] + 1) - 0.5, np.arange(z.shape[0] + 1) - 0.5, z, cmap=CMAP, norm=norm, edgecolors='white', linewidth=2)
    ax.set(xlim=(-0.5, z.shape[1] - 0.5), ylim=(z.shape[0] - 0.5, -0.5), xticks=np.arange(z.shape[1]), xticklabels=cols, yticks=np.arange(z.shape[0]), yticklabels=rows)
    ax.tick_params(length=0, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)
    if values_text:
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                celltext(ax, j, i, z[i, j], norm)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.062, pos.width, 0.01], label='<colorbar>')
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[-norm.vmax, 0, norm.vmax])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, pad=2, labelsize=8)
    cb.set_label(unit, labelpad=2, fontsize=8)
    return cax