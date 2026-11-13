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

def plot(time_summary, output_name='RISE_ENVIRONMENT_HISTORY', dpi=900):
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'npj-mwt-rise-20261023', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65})
    spec = [('wind', 'track', 'Wind intensity', 'Wind (kt)', '#233A46'), ('midRH', '200-800', 'Mid-level humidity', 'Relative humidity (%)', '#428C79'), ('shear200_850', '200-800', 'Deep-layer shear', 'Shear (m s$^{-1}$)', '#82649A'), ('SST', '200-800', 'Sea-surface temperature', 'SST (°C)', '#CC7445'), ('PI', '200-800', 'Potential intensity', 'PI (m s$^{-1}$)', '#167B9E'), ('PI_headroom', '200-800', 'PI headroom', 'Headroom (m s$^{-1}$)', '#437F9C')]
    fig, axes = plt.subplots(3, 2, figsize=(7.086614173228347, 7.322834645669292))
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.09, top=0.93, wspace=0.39, hspace=0.39)
    records = []
    for i, (metric, radial, title, ylabel, color) in enumerate(spec):
        ax = axes.flat[i]
        d = time_summary[time_summary.metric.eq(metric) & time_summary.radial.eq(radial)].sort_values('relative_hour')
        assert d.relative_hour.tolist() == [-24, -18, -12, -6, 0, 6, 12, 18, 24] and d.n_sid.ge(2).all()
        x = d.relative_hour.to_numpy(float)
        y = d.estimate.to_numpy(float)
        ax.axvline(0, color='#9EAAAF', ls=(0, (3, 3)), lw=0.75, zorder=0)
        ax.fill_between(x, d.lower.to_numpy(float), d.upper.to_numpy(float), color=color, alpha=0.15, lw=0)
        ax.plot(x, y, color=color, marker='o', ms=3, lw=1.5)
        ax.set_xlim(-25, 25)
        ax.set_xticks([-24, -12, 0, 12, 24])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', pad=7)
        ax.text(-0.19, 1.095, chr(97 + i), transform=ax.transAxes, fontsize=10.5, fontweight='bold', va='bottom')
        if i >= 4:
            ax.set_xlabel('Time from trough (h)')
        records.extend(d.assign(panel=chr(97 + i)).to_dict('records'))
    table('rise', 'a_f', pd.DataFrame(records), 'Saved per-SID pointwise histories')
    export(fig, 'rise')

def build():
    plot(load('time'))
