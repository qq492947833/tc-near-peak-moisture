from pathlib import Path
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

def figs2(dpi):
    dur = read('S2_a_duration.csv')
    fraction = read('S2_b_stage_fraction.csv').set_index('basin_genesis').loc[BASINS]
    base = read('S2_c_d_asymmetry.csv')
    assert len(dur) == len(base) == 2340
    cols = ['development_duration_hours', 'escalation_duration_hours', 'maturity_duration_hours', 'resolution_duration_hours']
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, 145 / 25.4))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.89, bottom=0.15, wspace=0.38, hspace=0.6)
    a, b, c, d = axs.flat
    for st, col in zip(STAGES, cols):
        values = dur[col].to_numpy(float)
        assert len(values) == 2340 and np.isfinite(values).all()
        x = np.sort(values)
        a.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=COLORS[st], lw=1.2, label=st)
    a.set(xlabel='Stage duration (h)', ylabel='Cumulative fraction', ylim=(0, 1), xlim=(0, None))
    grid(a)
    title(a, 'a', 'Stage durations · 2,340 cyclones')
    fig.legend(*a.get_legend_handles_labels(), ncol=4, loc='upper center', bbox_to_anchor=(0.56, 0.995), columnspacing=2.0)
    left = np.zeros(6)
    for st, col in zip(STAGES, cols):
        b.barh(range(6), fraction[col], left=left, color=COLORS[st], height=0.65)
        left += fraction[col].to_numpy(float)
    b.set(yticks=range(6), yticklabels=BASINS, ylim=(5.5, -0.5), xlabel='Mean fraction of lifetime', xlim=(0, 1))
    title(b, 'b', 'Basin stage allocation')
    for j, ax in enumerate([c, d]):
        for basin in BASINS:
            z = base.loc[base.basin_genesis.eq(basin), 'lifecycle_asymmetry_time'].to_numpy(float)
            x = np.sort(np.abs(z) if j else z)
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=BC[basin], lw=1.1)
        ax.set(xlabel='Absolute asymmetry |A|' if j else 'Signed asymmetry A', ylabel='Cumulative fraction', xlim=(0, 1) if j else (-1, 1), ylim=(0, 1))
        title(ax, chr(99 + j), ['Signed asymmetry', 'Absolute asymmetry'][j])
        grid(ax)
        if not j:
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    fig.legend(handles=[Line2D([0], [0], color=BC[b], label=b, lw=1.2) for b in BASINS], ncol=6, loc='lower center', bbox_to_anchor=(0.56, 0.02), columnspacing=1.5)
    save(fig, 'Legacy_Figure_S02_distributions_abcd_candidate', dpi)

def build(dpi): figs2(dpi)
