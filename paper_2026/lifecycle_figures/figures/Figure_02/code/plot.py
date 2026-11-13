from pathlib import Path
from display_format import format_number, nice_symmetric_limit
import json,hashlib
import numpy as np,pandas as pd,matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.text import Text
SRC=Path(__file__).resolve().parents[1]/'input'
OUT=SRC
QA=SRC
WIDTH_MM = 180
FINAL_DPI = 900
PREVIEW_RESOLUTION = 180
STAGES = ['D', 'E', 'F', 'Res']
COLORS = {'D': '#437F9C', 'E': '#2C9F9A', 'F': '#C89D38', 'Res': '#C05A52'}
MARKERS = {'D': 'o', 'E': 's', 'F': '^', 'Res': 'D'}
BLUE = '#247EB1'
ORANGE = '#D0783F'
GREY = '#9B9B9B'
INK = '#000000'
VARS = ['X1_sst_c', 'X2_shear_ms', 'X3_mid_rh_pct', 'X4_d200_1e5s', 'X5_minus_omega500_pa_s', 'X6_signed_vo850_1e5s']
VL = ['SST', 'Deep shear', 'Mid-level RH', 'Divergence', 'Ascent', 'Vorticity']
BASINS = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']
BC = dict(zip(BASINS, ['#437F9C', '#BD9145', '#428C79', '#9D82AB', '#AB685E', '#67A4AC']))
CMAP = LinearSegmentedColormap.from_list('light_signed', ['#DBAA8D', '#FAFAF8', '#9FC3D2'])
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'svg.hashsalt': 'fltccf-structural-20261025'})
def read(name):
    return pd.read_csv(SRC / name, keep_default_na=False, na_values=['', 'nan', 'NaN'], float_precision='round_trip')

def title(ax, letter, text):
    ax.set_gid(letter)
    ax.set_title(text, loc='left', pad=10)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-21, 9), textcoords='offset points', weight='bold', fontsize=10, va='bottom')

def grid(ax, axis='y'):
    ax.grid(axis=axis, color='#E6E6E6', lw=0.5, zorder=0)
    ax.set_axisbelow(True)

def fig2(dpi):
    raw = read('raw_stage_summary.csv')
    pairs = read('raw_adjacent_contrasts.csv')
    adj = read('adjusted_adjacent_summary.csv')
    adj = adj[adj.scheme.eq('geography')]
    ref = read('reference_scale.csv').set_index('variable')
    assert len(raw) == 24 and len(pairs) == 18 and (len(adj) == 18)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 140 / 25.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.12])
    fig.subplots_adjust(left=0.155, right=0.98, top=0.895, bottom=0.145, hspace=0.38, wspace=0.24)
    a = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
    for j, st in enumerate(STAGES):
        d = raw[raw.stage.eq(st)].set_index('variable').loc[VARS]
        x = np.arange(6) + (j - 1.5) * 0.15
        e = d.estimate_standardized.to_numpy()
        lo = d.ci_low_standardized.to_numpy()
        hi = d.ci_high_standardized.to_numpy()
        a.errorbar(x, e, yerr=[e - lo, hi - e], fmt=MARKERS[st], color=COLORS[st], ms=4.1, capsize=2, lw=0.9, label=st, zorder=3)
    a.axhline(0, color='#898989', lw=0.65, zorder=0)
    a.set_xticks(range(6), ['Sea surface\ntemperature', '200–850-hPa\nshear', '700/500-hPa\nrelative humidity', '200-hPa\ndivergence', '500-hPa\nascent (−ω)', '850-hPa\ncyclonic vorticity'])
    a.set(xlim=(-0.55, 5.55), ylabel='(Stage mean − reference mean)\n/ reference SD')
    grid(a)
    title(a, 'a', 'Stage environments')
    fig.legend(*a.get_legend_handles_labels(), ncol=4, loc='upper center', bbox_to_anchor=(0.56, 0.992), handlelength=1.3, columnspacing=2.2)
    rows = []
    allbounds = []
    for transition in ['E-D', 'F-E', 'Res-F']:
        d = pairs[pairs.contrast.eq(transition)].set_index('variable').loc[VARS]
        z = adj[adj.contrast.eq(transition)].set_index('variable').loc[VARS]
        for i, var in enumerate(VARS):
            sd = float(ref.loc[var, 'reference_sd'])
            assert np.isclose(sd, d.loc[var, 'reference_sd']) and np.isclose(sd, z.loc[var, 'reference_sd'])
            for key, point, lo, hi, n in [('Raw, full support', d.loc[var, 'estimate'], d.loc[var, 'ci_low'], d.loc[var, 'ci_high'], d.loc[var, 'n_pairs']), ('Raw, common support', z.loc[var, 'raw_matched_estimate'], z.loc[var, 'raw_on_support_ci_low'], z.loc[var, 'raw_on_support_ci_high'], z.loc[var, 'n_pairs']), ('Adjusted, common support', z.loc[var, 'estimate'], z.loc[var, 'ci_low'], z.loc[var, 'ci_high'], z.loc[var, 'n_pairs'])]:
                rows.append(dict(transition=transition, variable=var, series=key, estimate=point / sd, ci_low=lo / sd, ci_high=hi / sd, n_sid=int(n), fixed_reference_sd=sd))
                allbounds.extend([lo / sd, hi / sd])
    display = pd.DataFrame(rows)
    display.to_csv(OUT / 'Figure_02_display_values.csv', index=False, float_format='%.17g')
    low = min(allbounds)
    high = max(allbounds)
    pad = (high - low) * 0.07
    limits = (low - pad, high + pad)
    for j, (ax, tr, lab) in enumerate(zip(axes, ['E-D', 'F-E', 'Res-F'], ['E − D', 'F − E', 'Res − F'])):
        for k, (series, col, marker) in enumerate(zip(['Raw, full support', 'Raw, common support', 'Adjusted, common support'], [GREY, BLUE, ORANGE], ['o', 's', 'D'])):
            d = display[display.transition.eq(tr) & display.series.eq(series)].set_index('variable').loc[VARS]
            e = d.estimate.to_numpy()
            lo = d.ci_low.to_numpy()
            hi = d.ci_high.to_numpy()
            y = np.arange(6) + (k - 1) * 0.23
            ax.errorbar(e, y, xerr=[e - lo, hi - e], fmt=marker, color=col, ms=3, capsize=1.5, lw=0.85, zorder=3)
        ax.axvline(0, color='#777777', lw=0.7, zorder=0)
        ax.set(xlim=limits, ylim=(5.6, -0.6))
        ax.set_yticks(range(6), VL if j == 0 else [''] * 6)
        ax.set_xlabel('Difference (reference SD)' if j == 1 else '', labelpad=5)
        grid(ax, 'x')
        title(ax, chr(98 + j), lab)
    handles = [Line2D([0], [0], marker=m, color=c, lw=0.85, ms=3, label=l) for c, m, l in zip([GREY, BLUE, ORANGE], ['o', 's', 'D'], ['Raw, full support', 'Raw, common support', 'Adjusted, common support'])]
    fig.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0.55, 0.025), handlelength=1.5, columnspacing=1.2)
    save(fig, 'Figure_02_stage_environment_candidate', dpi)

def build(dpi): fig2(dpi)
