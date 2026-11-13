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

def grid(rows, cols, height=175, left=0.14, bottom=0.1, top=0.92, hspace=0.72, wspace=0.42):
    fig, axs = plt.subplots(rows, cols, figsize=(180 / 25.4, height / 25.4), squeeze=False)
    fig.subplots_adjust(left=left, right=0.975, bottom=bottom, top=top, hspace=hspace, wspace=wspace)
    return (fig, axs)

def s12():
    name = 'FigS12'
    ap = DATA / 'spatial/current117_fixed_pre_shear_anomaly_common_paired_composites.npz'
    bp = DATA / 'spatial/current117_fixed_pre_shear_raw_common_paired_composites.npz'
    gp = DATA / 'spatial/geometry.npz'
    A = dict(np.load(ap))
    B = dict(np.load(bp))
    G = dict(np.load(gp))
    mask = G['display_mask']
    X = G['x_m'] / 1000
    Y = G['y_m'] / 1000
    original = external('04_Data_and_Code/github_ready/source_values/accepted_v51/p1_spatial_v1/r02/map_source_data.csv')
    infer = external('03_Work/v6/physics/spatial_centered_null_full_family_B199999.csv')
    table(name, 'all', original, 'original_pointwise_intervals')
    table(name, 'all', infer, 'frozen_199999_null_inference')
    fig, axs = grid(4, 4, 221, left=0.115, top=0.9, bottom=0.13, hspace=0.4, wspace=0.43)
    rows = []
    cbs = []
    for ri, (fi, hi) in enumerate([(0, 0), (0, 1), (3, 0), (3, 1)]):
        field = 'q700' if fi == 0 else 'z200'
        hemi = ['NH', 'SH'][hi]
        for ci in range(4):
            a = axs[ri, ci]
            lab = chr(97 + ri * 4 + ci)
            a.set_gid(lab)
            if ci < 3:
                si = ci + 1
                limit = 2 if fi == 0 else 40
                val = A['difference'][hi, si, fi]
                a.pcolormesh(X, Y, np.ma.masked_where(~mask, val), shading='nearest', cmap='RdBu_r', vmin=-limit, vmax=limit, rasterized=True)
                a.set(xlim=(-1500, 1500), ylim=(-1500, 1500), aspect='equal')
                a.set_xticks([-1000, 0, 1000])
                a.set_yticks([-1000, 0, 1000])
                n = A['n_sid'][hi, si, fi][mask]
                nt = str(n.min()) if n.min() == n.max() else f'{n.min()}–{n.max()}'
                a.set_title(f'{lab}  n={nt}', loc='left', fontsize=7.6, pad=5)
                for iy, ix in zip(*np.where(mask)):
                    rows.append({'panel': lab, 'source_file': str(ap), 'source_sha256': hashlib.sha256(ap.read_bytes()).hexdigest(), 'array_index': f'{hi},{si},{fi},{iy},{ix}', 'hemisphere': hemi, 'field': str(A['fields'][fi]), 'summary': str(A['summaries'][si]), 'x_km': X[iy, ix], 'y_km': Y[iy, ix], 'estimate': val[iy, ix], 'n_sid': A['n_sid'][hi, si, fi, iy, ix]})
            else:
                raw = B['difference'][hi, 0, fi][mask]
                anom = A['difference'][hi, 0, fi][mask]
                lim = 2 if fi == 0 else 100
                a.scatter(raw, anom, s=3, color=TEMP, alpha=0.35, edgecolors='none', rasterized=True)
                a.plot([-lim, lim], [-lim, lim], color='#9AA8AE', ls='--', lw=0.7)
                a.axhline(0, color='#BCC6CB', lw=0.5)
                a.axvline(0, color='#BCC6CB', lw=0.5)
                a.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')
                a.set_xticks([-lim, 0, lim])
                a.set_yticks([-lim, 0, lim])
                a.set_title(lab + '  709 cells', loc='left', fontsize=7.6, pad=5)
                for (iy, ix), r, v in zip(zip(*np.where(mask)), raw, anom):
                    rows.append({'panel': lab, 'source_file': str(ap) + '; ' + str(bp), 'hemisphere': hemi, 'field': str(A['fields'][fi]), 'summary': str(A['summaries'][0]), 'x_km': X[iy, ix], 'y_km': Y[iy, ix], 'raw_estimate': r, 'anomaly_estimate': v, 'n_raw': B['n_sid'][hi, 0, fi, iy, ix], 'n_anomaly': A['n_sid'][hi, 0, fi, iy, ix]})
            a.tick_params(labelsize=6.7, pad=2, labelleft=ci in [0, 3], labelbottom=ri == 3)
            if ci == 0:
                a.set_ylabel(field + ' · ' + hemi + '\nCross-shear (km)', fontsize=7.6)
            if ci == 3:
                a.set_ylabel('Anomaly', fontsize=7.5)
                a.yaxis.set_label_coords(-0.25, 0.5)
    for ci, txt in enumerate(['−24 h\nFixed t0 centre', '0 h\nActual centre', '+24 h\nActual centre', 'Pre-mean\nRaw vs anomaly']):
        p = axs[0, ci].get_position()
        fig.text(p.x0 + p.width / 2, 0.95, txt, ha='center', va='center', fontsize=8)
    for ci in range(4):
        axs[3, ci].set_xlabel('Raw contrast' if ci == 3 else 'Along shear (km)', fontsize=7.4)
    for x, lim, lab in [(0.16, 2, 'q700 anomaly contrast (g kg⁻¹)'), (0.6, 40, 'z200 anomaly contrast (m)')]:
        ca = fig.add_axes([x, 0.05, 0.3, 0.011])
        cbs.append(ca)
        fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(-lim, lim), cmap='RdBu_r'), cax=ca, orientation='horizontal', ticks=[-lim, 0, lim]).set_label(lab, fontsize=7.4)
    table(name, 'a_p', pd.DataFrame(rows), 'all_displayed_cells')
    finish(fig, name, exclude_axes=cbs)