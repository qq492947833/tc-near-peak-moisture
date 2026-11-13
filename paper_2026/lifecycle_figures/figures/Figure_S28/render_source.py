from display_format import format_number,format_group
from pathlib import Path
import os, sys, json, hashlib, re, zipfile, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, LogNorm
from matplotlib.patches import Circle, Wedge, Rectangle, FancyArrowPatch, Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PIL import Image
BLUE = TEMP = '#167B9E'
ORANGE = FINAL = '#CC7445'
INK = '#233A46'
MUTED = '#677B86'
TEAL, PURPLE, GREY = ('#2C8078', '#9069A3', '#7F9099')
GREEN = '#418D79'
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
CMAP = LinearSegmentedColormap.from_list('signed_native', [ORANGE, '#FAFAF7', BLUE])
WIDTH_MM = 180
WIDTH_IN = 180 / 25.4
GRID = '#DDE3E7'
LEDGER = []
FILES = {}
CAPTIONS = {}
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'savefig.facecolor': 'white'})

def build():
    wide = load('states')
    states = wide.melt(id_vars=['sid', 'radial'], value_vars=['matched_rise', 'matched_Resolution'], var_name='estimand', value_name='value')
    states['metric'] = 'q700'
    summary = load('integrals')
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'npj-rise-moisture-20261023', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65})
    fig, axs = plt.subplots(2, 2, figsize=(7.086614173228347, 6.141732283464567))
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.12, top=0.935, wspace=0.37, hspace=0.46)
    records = []
    scatter_records = []
    spec = [('inner', '100–300 km', '#167B9E'), ('outer', '300–800 km', '#428C79')]
    metrics = ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency']
    labels = ['Horizontal\nadvection', 'Vertical\nadvection', 'Residual', 'Moving\ntendency']
    colors = ['#7C8C95', '#428C79', '#AB685E', 'white']
    pairs = {}
    for radial, _, _ in spec:
        d = states[states.metric.eq('q700') & states.radial.eq(radial) & states.estimand.isin(['matched_rise', 'matched_Resolution'])]
        pair = d.pivot(index='sid', columns='estimand', values='value')
        assert len(pair) == 108 and np.isfinite(pair.to_numpy()).all()
        pairs[radial] = pair
    qlo = np.floor(min((x.to_numpy().min() for x in pairs.values())))
    qhi = np.ceil(max((x.to_numpy().max() for x in pairs.values())))
    for col, (radial, title, color) in enumerate(spec):
        ax = axs[0, col]
        pair = pairs[radial]
        ax.plot([qlo, qhi], [qlo, qhi], color='#ADB6BA', lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.scatter(pair.matched_Resolution, pair.matched_rise, s=12, c=color, alpha=0.55, linewidths=0)
        ax.set(xlim=(qlo, qhi), ylim=(qlo, qhi), xlabel='Resolution specific humidity (g kg$^{-1}$)', ylabel='Rise specific humidity (g kg$^{-1}$)', title='Matched winds · ' + title)
        scatter_records.extend(pair.reset_index().assign(radial=radial, panel=chr(97 + col)).to_dict('records'))
        ax = axs[1, col]
        rows = []
        for metric in metrics:
            row = summary[summary.metric.eq(metric) & summary.radial.eq(radial) & summary.estimand.eq('rise_rate_integral') & summary['sample'].eq('all_events')]
            assert len(row) == 1
            row = row.iloc[0]
            assert row.units == 'g kg-1' and row.n_sid >= 2
            rows.append(row)
        d = pd.DataFrame(rows)
        assert d.n_sid.nunique() == 1 and d.n_events.nunique() == 1
        assert abs(d.estimate.iloc[:3].sum() - d.estimate.iloc[3]) < 1e-12
        x = np.arange(4)
        y = d.estimate.to_numpy()
        ax.axhline(0, color='#ADB6BA', lw=0.75, zorder=0)
        ax.bar(x, y, color=colors, edgecolor=['none'] * 3 + ['#233A46'], linewidth=0.9, width=0.62, zorder=2)
        ax.errorbar(x, y, yerr=np.vstack([y - d.lower.to_numpy(), d.upper.to_numpy() - y]), fmt='none', ecolor='#233A46', elinewidth=0.85, capsize=2.5, zorder=3)
        ax.set_xticks(x, labels)
        ax.set_ylabel('Integrated diagnosed rate (g kg$^{-1}$)')
        ax.set_title('Rise integrals · ' + title)
        ax.set_ylim(-5.1, 5.7)
        records.extend(d.assign(panel=chr(99 + col)).to_dict('records'))
    for i, ax in enumerate(axs.flat):
        ax.text(-0.19, 1.075, chr(97 + i), transform=ax.transAxes, fontsize=10.5, fontweight='bold', va='bottom')
    table('moisture', 'a_b', pd.DataFrame(scatter_records), 'Actual same-SID matched-wind states')
    table('moisture', 'c_d', pd.DataFrame(records), 'Saved actual-ascent diagnosed integrals and percentile intervals')
    export(fig, 'moisture')
