from pathlib import Path
import json,hashlib
from decimal import Decimal,ROUND_HALF_UP
import numpy as np,pandas as pd,matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection
from matplotlib.text import Annotation,Text
SRC=Path(__file__).resolve().parents[1]/'input'
SOURCE=SRC;OLD=SRC;MASTER=True;COLLECT=[];RECORDS=[]
C=json.loads((SRC/'manuscript_figures.json').read_text(encoding='utf8'))
P=C['palette'];INK=P['ink'];BLUE=P['return1'];ORANGE=P['return2'];PURPLE=P['return3'];GREY='#809198'
BASINS=C['figure1']['basin_order']
MASTER_SIDS=set((SRC/'main_sid.txt').read_text(encoding='utf8').splitlines()) if (SRC/'main_sid.txt').exists() else set()
def tracked(p):return Path(p)
def read_table(p):
    d=pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN']);d.attrs['source']=str(p);return d
def record(*args):pass
def save(fig,name,final):COLLECT.append((fig,name))
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'mwt-addendum-manuscript-figures', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'legend.frameon': False, 'savefig.facecolor': 'white', 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
def title(ax, letter, label, pad=12):
    ax.set_gid(letter)
    ax.set_title(label, loc='left', pad=pad)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-21, pad), textcoords='offset points', fontsize=10, weight='bold', va='baseline')

import types
f=types.SimpleNamespace(read_table=read_table,record=record,title=title,P=P)
def main():
    stages = f.read_table(SRC/'raw_stage_summary.csv')
    comparisons = f.read_table(SRC/'A2_paired_contrasts.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.25))
    fig.subplots_adjust(left=0.17, right=0.975, top=0.92, bottom=0.105, wspace=0.7, hspace=0.65)
    variables = [('X3_mid_rh_pct', 'Mid-level relative humidity', 'RH (%)', 'RH contrast (percentage points)'), ('X2_shear_ms', 'Deep-layer vertical shear', 'Shear (m s$^{-1}$)', 'Shear contrast (m s$^{-1}$)')]
    order = ['D', 'E', 'F', 'Res']
    colors = [f.P[k] for k in order]
    labels = ['D − pooled D/E', 'E − pooled D/E', 'F − R', 'Res − post-R tail']
    contrast_order = ['D-pooledDE', 'E-pooledDE', 'F-R', 'Res-post_R_tail']
    for column, (var, label, axis_label, contrast_label) in enumerate(variables):
        ax = axes[0, column]
        d = stages[stages.variable.eq(var)].set_index('stage').loc[order].reset_index()
        f.record('Framework_environment', 'ab'[column], d, 'Original stage means and SID-percentile intervals on frozen main sample')
        f.title(ax, 'ab'[column], label)
        for x, row in d.iterrows():
            ax.errorbar(x, row.estimate, yerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]], color=colors[x], marker='o', markersize=5, capsize=3, elinewidth=1.15, linewidth=0)
        ax.set_xticks(range(4), order)
        ax.set_xlim(-0.45, 3.45)
        ax.set_ylabel(axis_label)
        ax.grid(axis='y', color='#e7eaec', linewidth=0.5)
        ax.set_axisbelow(True)
        ax = axes[1, column]
        d = comparisons[comparisons.variable.eq(var)].set_index('contrast').loc[contrast_order].reset_index()
        f.record('Framework_environment', 'cd'[column], d, 'Existing controlled D/E merging and F-to-R replacement, paired SID contrasts')
        f.title(ax, 'cd'[column], 'Effect of lifecycle representation')
        ax.axvline(0, color='#9aa6aa', linewidth=0.75, zorder=0)
        for y, row in d.iterrows():
            ax.errorbar(row.estimate, y, xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]], color=colors[y], marker='o', markersize=4.5, capsize=3, elinewidth=1.1, linewidth=0)
        ax.set_yticks(range(4), labels)
        ax.set_ylim(3.5, -0.5)
        ax.set_xlabel(contrast_label)
        ax.grid(axis='x', color='#e7eaec', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.16)
    emit(fig)
def build(dpi):main()
