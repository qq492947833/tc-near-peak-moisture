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

def fig11(dpi):
    allocation = read('S2_e_time_allocation.csv')
    cdf = read('S2_f_time_asymmetry.csv')
    trends = read('primary_trends.csv').set_index('metric').loc[['D_h', 'E_h', 'F_h', 'Resolution_h']]
    assert len(allocation) == 12 and len(cdf) == 2340
    assert allocation.n_sid.eq(2340).all() and trends.n_years.eq(42).all()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 130 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    fig.subplots_adjust(left=0.155, right=0.98, top=0.865, bottom=0.13, hspace=0.50, wspace=0.40)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, :])
    left = np.zeros(3)
    for st in STAGES:
        q = allocation[allocation.stage.eq(st)].set_index('segment').loc[['pre', 'middle', 'post']]
        covered = q.mean_covered_h.to_numpy(float)
        uncovered = q.mean_uncovered_h.to_numpy(float)
        a.barh(range(3), covered, left=left, color=COLORS[st], height=0.6)
        left += covered
        visible = uncovered > 1e-09
        a.barh(np.arange(3)[visible], uncovered[visible], left=left[visible], color='white', edgecolor=COLORS[st], lw=0.8, height=0.6)
        left += uncovered
    a.set_yticks(range(3), ['Before first\nLMI', 'First–last\nLMI', 'After last\nLMI'])
    a.invert_yaxis()
    a.set(xlabel='Mean elapsed time (h)', xlim=(0, 82))
    grid(a, 'x')
    title(a, 'a', 'Peak-centred stage allocation')
    handles = [Patch(facecolor=COLORS[st], label=st) for st in STAGES] + [Patch(facecolor='white', edgecolor=GREY, label='Uncovered')]
    fig.legend(handles=handles, ncol=5, loc='upper center', bbox_to_anchor=(0.56, 0.995), handlelength=1.4, columnspacing=1.5)
    b.step(cdf.A, cdf.empirical_CDF, where='post', color=INK, lw=1.2)
    b.axvline(0, color=GREY, lw=0.7)
    grid(b)
    b.set(xlabel='Signed asymmetry A', ylabel='Cumulative fraction', xlim=(-1, 1), ylim=(0, 1))
    title(b, 'b', 'Lifecycle time asymmetry')
    for i, st in enumerate(STAGES):
        q = trends.iloc[i]
        c.errorbar(q.slope, i, xerr=[[q.slope - q.ci_low], [q.ci_high - q.slope]], fmt=MARKERS[st], color=COLORS[st], ms=4, capsize=2, lw=1.2, zorder=3)
    c.axvline(0, color=GREY, lw=0.75)
    c.set(yticks=range(4), yticklabels=['Dec' if st=='Res' else st for st in STAGES], ylim=(3.5, -0.5), xlim=(-7.5, 2.8), xlabel='Stage-duration trend (h decade⁻¹)')
    grid(c, 'x')
    title(c, 'c', 'Stage-duration trends, 1982–2023')
    trends.reset_index().to_csv(OUT / 'Figure_11_frozen_trend_values.csv', index=False, float_format='%.17g')
    save(fig, 'Figure_11_lifecycle_asymmetry_candidate', dpi)

def build(dpi): fig11(dpi)
