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
    norm = norm_for(z, 1.2)
    assert np.nanmax(abs(z)) <= 1.2
    ax.set(xlim=(-0.5, len(z) - 0.5), ylim=(-0.5, 0.5), xticks=range(len(z)), xticklabels=labels, yticks=[])
    for j, v in enumerate(z):
        ax.add_patch(Rectangle((j - 0.47, -0.34), 0.94, 0.68, facecolor=CMAP(norm(v)), edgecolor='white'))
        celltext(ax, j, 0, v, norm, 9)
    ax.tick_params(length=0, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel(unit, labelpad=10)

def fluxrows(src):
    d = rd(src)
    return d[d.cohort.eq('original_post80') & d.window.eq('0_24h') & d.factor.eq(1) & d.sector.eq('all')]

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

def fig5(dpi):
    d = fluxrows('direct/paired_flux_components.csv')
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 174 / 25.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.8, 0.35])
    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.1, hspace=0.9, wspace=0.45)
    a = fig.add_subplot(gs[0, :])
    a.set(xlim=(0, 10), ylim=(0, 5))
    a.axis('off')
    pos = a.get_position()
    xy_unit_ratio = pos.width * fig.get_figwidth() / 10 / (pos.height * fig.get_figheight() / 5)
    for x, kind, col, lab in [(2.2, 'temporary', BLUE, 'Temporary'), (7.4, 'final', ORANGE, 'Final')]:
        q = d[d.layer.eq('all') & d.quantity.eq(kind)].set_index('kind')
        a.add_patch(Ellipse((x, 2.6), 1.52, 1.52 * xy_unit_ratio, facecolor='#F6F8F9', edgecolor=col, lw=1.2))
        a.text(x, 4.35, lab, ha='center', color=col, fontsize=9.5, weight='bold')
        a.text(x, 2.8, '300 km', ha='center', fontsize=8.5)
        a.text(x, 2.35, 'Net ' + format_number(q.loc['net', 'estimate'] / 1000000000000.0, decimals=2), ha='center', fontsize=8.5)
        for k, yy in [('in', 3.5), ('out', 1.9)]:
            mean = float(q.loc[k, 'estimate']) / 1000000000000.0
            start, end = ((x - 1.7, yy), (x - 0.05, yy)) if k == 'in' else ((x + 0.05, yy), (x + 1.7, yy))
            a.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=mean * 1.5, linewidth=mean * 0.48, color=col, shrinkA=0, shrinkB=0, alpha=0.9))
            a.text(x - 1.35 if k == 'in' else x + 1.35, yy + 0.4, ('In ' if k == 'in' else 'Out ') + format_number(mean, decimals=2), ha='center', fontsize=8.5)
        record('Fig5', 'a', 'direct/paired_flux_components.csv', q.reset_index(), 'Saved class means; arrow shaft width proportional to mean exchange', 'kg / 10^12')
    a.text(5, 0.75, 'Exchange over 0–24 h (10¹² kg)', ha='center', fontsize=8.5)
    title(a, 'a', 'Exchange through a moving boundary')
    b = fig.add_subplot(gs[1, 0])
    layers = fluxrows('direct/pressure_components.csv')
    order = ['500-600_hPa', '600-700_hPa', '700-850_hPa', '850-925_hPa']
    z = layers[layers.kind.eq('net') & layers.quantity.isin(['temporary', 'final'])]
    vals = np.array([[float(z[z.layer.eq(k) & z.quantity.eq(q)].estimate.iloc[0]) / 1000000000000.0 for q in ['temporary', 'final']] for k in order])
    norm = norm_for(vals, 3.4)
    assert np.nanmax(abs(vals)) <= 3.4
    mesh = b.pcolormesh([-0.5, 0.5, 1.5], [-0.5, 0.5, 1.5, 2.5, 3.5], vals, cmap=CMAP, norm=norm, edgecolors='white', lw=2)
    b.set(xlim=(-0.5, 1.5), ylim=(3.5, -0.5), xticks=[0, 1], xticklabels=['Temporary', 'Final'], yticks=range(4), yticklabels=['500–600', '600–700', '700–850', '850–925'], ylabel='Pressure layer (hPa)', xlabel='Net inward mass (10¹² kg)')
    b.tick_params(length=0)
    for i, y in enumerate(range(4)):
        for j in range(2):
            celltext(b, j, y, vals[i, j], norm)
    for s in b.spines.values():
        s.set_visible(False)
    title(b, 'b', 'Net exchange by pressure layer')
    record('Fig5', 'b', 'direct/pressure_components.csv', z, 'Layer-integrated amount; equal-height categorical display of unchanged native pressure layers; no interpolation', 'kg / 10^12')
    c = fig.add_subplot(gs[1, 1])
    z = d[d.layer.eq('all') & d.quantity.eq('difference')].set_index('kind').loc[['in', 'out', 'net']].reset_index()
    cb = matrix(fig, c, z.estimate.to_numpy()[:, None] / 1000000000000.0, ['Gross inward', 'Gross outward', 'Net inward'], ['Temporary − final'], 'Paired contrast (10¹² kg)', limit=1.2)
    title(c, 'c', 'Direct paired differences')
    record('Fig5', 'c', 'direct/paired_flux_components.csv', z, 'Saved direct differences and their original percentile intervals', 'kg / 10^12')
    ax = fig.add_subplot(gs[2, :])
    src = 'direct/exchange_background80.csv'
    z = rd(src)
    z = z[z.metric.eq('post_r300_net')]
    q = pd.concat([z[z.model.eq('geographical') & z.target.eq(t)] for t in ['original80_raw', 'restricted_raw', 'balanced']] + [z[z.model.eq('geographical_state') & z.target.eq('balanced')]])
    tiles(ax, q.estimate.to_numpy() / 1000000000000.0, ['Original', 'Support, raw', 'Geography + season', '+ Event state'], 'Net contrast\n(10¹² kg)')
    title(ax, 'd', 'Comparison on measured backgrounds')
    record('Fig5', 'd', src, q, 'Distinct original/restricted/weighted targets, intervals and draw support retained', 'kg / 10^12')
    export(fig, 'Fig5', dpi, exclude=[cb])