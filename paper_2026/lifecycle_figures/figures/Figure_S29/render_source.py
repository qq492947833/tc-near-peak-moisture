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



def zero(ax, vertical=False):
    if vertical:
        ax.axvline(0, lw=0.7, color=GREY, ls='--', zorder=0)
        ax.grid(axis='x', color=GRID, lw=0.45, zorder=0)
    else:
        ax.axhline(0, lw=0.7, color=GREY, zorder=0)
        ax.grid(axis='y', color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)

def fig7():
    effects = rd('thematic/fig7_effects.csv')
    diag = rd('thematic/fig7_diagnostics.csv')
    sens = rd('thematic/fig7_sensitivity.csv')
    policies = ['shear200_850', 'midRH']
    methods = ['gcomp', 'onestep', 'ipw']
    colors, markers = ([INK, GREY, PURPLE], ['o', 's', '^'])
    fig, ax = plt.subplots(2, 2, figsize=(WIDTH_IN, 145 / 25.4))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.88, wspace=0.38, hspace=0.52)
    for j, outcome in enumerate(['recovery48_pp', 'delta24_kt']):
        a = ax[0, j]
        for k, p in enumerate(policies):
            for m, method in enumerate(methods):
                z = effects[effects.policy.eq(p) & effects.outcome.eq(outcome) & effects.method.eq(method)]
                assert len(z) == 1
                e, lo, hi = nums(z, 1, 'ci_low', 'ci_high')
                yy = 1 - k + (1 - m) * 0.2
                a.errorbar(e, [yy], xerr=[e - lo, hi - e], fmt=markers[m], color=colors[m], ms=4, capsize=2.2, elinewidth=0.9)
                record('Fig7', chr(97 + j), 'thematic/fig7_effects.csv', z, 'Prespecified policy-minus-natural-course estimate and 1000-refit interval')
        a.set(yticks=[1, 0], yticklabels=['Shear\n−2 m s⁻¹', 'RH\n+5 pp'], ylim=(-0.5, 1.5), xlabel='Recovery difference (pp)' if j == 0 else 'Wind-change difference (kt)')
        zero(a, vertical=True)
        title(a, chr(97 + j), 'Recovery within 48 h' if j == 0 else 'Wind change over 24 h')
    a = ax[1, 0]
    for j, p in enumerate(policies):
        z = diag[diag.policy.eq(p) & diag.outcome.eq(48)]
        r = z.iloc[0]
        a.bar(j - 0.16, r.coverage * 100, 0.28, color=INK, edgecolor='none')
        a.bar(j + 0.16, r.ratio_ESS / 2231 * 100, 0.28, color='#C8D1D8', edgecolor='none')
        record('Fig7', 'c', 'thematic/fig7_diagnostics.csv', z, 'Exposure-change fraction and density-ratio ESS fraction', 'coverage * 100; ratio_ESS / 2231 * 100')
    a.set(xticks=[0, 1], xticklabels=['Shear −2 m s⁻¹', 'RH +5 pp'], ylim=(0, 100), ylabel='Cohort fraction (%)')
    zero(a)
    title(a, 'c', 'Policy coverage and density support')
    a = ax[1, 1]
    for j, p in enumerate(policies):
        z = sens[sens.policy.eq(p) & sens.outcome.eq(48)].sort_values('offset')
        a.plot(z.offset, z.estimate, ['o-', 's-'][j], color=[BLUE, ORANGE][j], ms=3.5, lw=1.3)
        record('Fig7', 'd', 'thematic/fig7_sensitivity.csv', z, 'Fixed sensitivity scenarios; no confidence intervals calculated')
    a.axvline(0, color=GRID, lw=0.7, zorder=0)
    zero(a)
    a.set(xlabel='Missing-outcome log-odds offset', ylabel='Recovery difference (pp)')
    title(a, 'd', 'Missing-outcome response')
    # Cohort definition and size are retained in the caption.
    fig.legend([Line2D([0], [0], color=co, marker=mk, lw=0.9) for co, mk in zip(colors, markers)], ['G-formula', 'One-step', 'Density weighting'], ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.55, 0.99))
    fig.legend([Rectangle((0, 0), 1, 1, facecolor=co, edgecolor='none') for co in [INK, '#C8D1D8']], ['Changed exposure', 'Density-ratio ESS / n'], frameon=False, loc='lower left', bbox_to_anchor=(0.1, 0.025), handlelength=1.2, fontsize=8)
    fig.legend([Line2D([0], [0], color=co, marker=mk, lw=1.2) for co, mk in [(BLUE, 'o'), (ORANGE, 's')]], ['Shear −2 m s⁻¹', 'RH +5 pp'], frameon=False, loc='lower right', bbox_to_anchor=(0.97, 0.025), handlelength=1.2, fontsize=8)
    export(fig, 'Fig7')

def title(ax, letter, text):
    ax.set_gid(letter)
    ax.set_title(text, loc='left', pad=14, color=INK)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-16, 13), textcoords='offset points', fontsize=10, fontweight='bold', va='bottom', annotation_clip=False)

def nums(rows, div=1000000000000.0, low='low', high='high'):
    e = rows['estimate'].to_numpy(float) / div
    lo = rows[low].to_numpy(float) / div
    hi = rows[high].to_numpy(float) / div
    assert np.all(np.isfinite(e)) and np.all(lo <= e) and np.all(e <= hi)
    return (e, lo, hi)
GREY="#7F9099"
mpl.rcParams.update({"font.size":8,"axes.labelsize":8,"axes.titlesize":9,"legend.frameon":False})
