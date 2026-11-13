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

VAR = ['X1_sst_c', 'X2_shear_ms', 'X3_mid_rh_pct', 'X4_d200_1e5s', 'X5_minus_omega500_pa_s', 'X6_signed_vo850_1e5s']
VL = ['SST', 'Deep shear', 'Mid-level RH', 'Divergence', 'Ascent', 'Vorticity']
BS = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']
BC = dict(zip(BS, ['#437F9C', '#BD9145', '#428C79', '#9D82AB', '#AB685E', '#67A4AC']))
CAPTIONS = {}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
SCOPES = ['main_all', 'exclude_single_6h', 'strict_bridge_tropical', 'without_Freddy2023', 'without_basin_WP', 'without_basin_EP', 'without_basin_SI', 'without_basin_NA', 'without_basin_SP', 'without_basin_NI', 'expanded_estimable', 'expanded_added_estimable']

def title(ax, label, text):
    ax.set_gid(label)
    ax.set_title(text, loc='left', pad=12)
    ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(-18, 12), textcoords='offset points', weight='bold', fontsize=10, va='baseline')

def grid(rows, cols, height=175, left=0.14, bottom=0.1, top=0.92, hspace=0.72, wspace=0.42):
    fig, axs = plt.subplots(rows, cols, figsize=(180 / 25.4, height / 25.4), squeeze=False)
    fig.subplots_adjust(left=left, right=0.975, bottom=bottom, top=top, hspace=hspace, wspace=wspace)
    return (fig, axs)

def s13():
    name = 'FigS13'
    d = source('physics/FigS8_panel_source.csv')
    specs = [('midRH', 'Relative humidity', 'RH percentage points'), ('PI_headroom', 'PI headroom', 'm s⁻¹'), ('PI_vmax', 'Potential intensity', 'm s⁻¹'), ('signed_chi', 'Signed entropy ratio', 'χ'), ('signed_entropy_deficit', 'Signed entropy deficit', 'J kg⁻¹ K⁻¹'), ('Tang_Emanuel_ventilation', 'Signed ventilation', 'Signed index')]
    fig, ax = grid(3, 2, 192, left=0.19, wspace=0.7, hspace=1.15, top=0.91, bottom=0.14)
    policies = ['all_along', 'all_fixed', 'historical_mixed']
    cbs = []
    for i, (var, tl, unit) in enumerate(specs):
        a = ax.flat[i]
        q = table(name, chr(97 + i), d[d.variable.eq(var)], 'sampling_policies')
        vals = q.pivot(index='policy', columns='estimand', values='estimate').loc[policies, ['pre_mean', 'change_0_24']]
        limit = float(np.nanmax(np.abs(vals.to_numpy())))
        im = matrix(a, vals, ['Pre-mean', 'Change'], ['Along track', 'Fixed centre', 'Mixed centre'], limit, fmt='.3g')
        title(a, chr(97 + i), tl)
        p = a.get_position()
        cb = fig.add_axes([p.x0 + 0.02, p.y0 - 0.074, p.width - 0.04, 0.01])
        cbs.append(cb)
        cbar = fig.colorbar(im, cax=cb, orientation='horizontal', ticks=[-limit, 0, limit], format='%.2g')
        cbar.set_label(unit, fontsize=7.6, labelpad=2)
        cb.tick_params(labelsize=7.5, pad=2)
    finish(fig, name, exclude_axes=cbs)

def matrix(ax, values, xlabels, ylabels, limit, zero_mark=None, fmt='.3f'):
    v = np.asarray(values, float)
    masked = np.ma.masked_invalid(v)
    cm = mpl.colormaps['RdBu_r'].copy()
    cm.set_bad('#E5E8EA')
    im = ax.imshow(masked, cmap=cm, norm=TwoSlopeNorm(0, -limit, limit), aspect='auto')
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0, pad=6)
    ax.set_xticks(np.arange(-0.5, len(xlabels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels)), minor=True)
    ax.grid(which='minor', color='white', lw=3)
    ax.tick_params(which='minor', length=0)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            text = 'NA' if not np.isfinite(v[i, j]) else format(v[i, j], fmt)
            if zero_mark is not None and bool(zero_mark[i, j]):
                text += '†'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if np.isfinite(v[i, j]) and abs(v[i, j]) > 0.65 * limit else INK)
    return im