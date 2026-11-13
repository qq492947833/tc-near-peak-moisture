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

SPEC = [{'metric': 'wind', 'radial': 'track', 'title': 'Observed wind recovery', 'ylabel': 'Wind (kt)', 'units': 'kt', 'color': '#233A46', 'source': 'environment'}, {'metric': 'midRH', 'radial': '200-800', 'title': 'Environmental humidity', 'ylabel': 'Relative humidity (%)', 'units': '%', 'color': '#428C79', 'source': 'environment'}, {'metric': 'KE', 'level': '700', 'title': 'Kinetic energy', 'ylabel': 'KE (J kg$^{-1}$)', 'units': 'J kg-1', 'color': '#167B9E', 'source': 'energy'}, {'metric': 'APE', 'level': '700', 'title': 'Dry available potential energy', 'ylabel': 'APE (J kg$^{-1}$)', 'units': 'J kg-1', 'color': '#82649A', 'source': 'energy'}, {'metric': 'C_A_to_K', 'level': '700', 'title': 'APE-to-KE conversion', 'ylabel': 'Conversion (W kg$^{-1}$)', 'units': 'W kg-1', 'color': '#CC7445', 'source': 'energy'}, {'metric': 'vertical_input_K_Wkg', 'level': '500-850', 'title': 'Layer vertical KE input', 'ylabel': 'Net input (W kg$^{-1}$)', 'units': 'W kg-1', 'color': '#437F9C', 'source': 'energy'}]

def build_frozen(panels, output_dir, report_dir, preview_dir, dpi=900):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.2, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'npj-mwt-rise-core-20261023', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65})
    fig, axes = plt.subplots(3, 2, figsize=(180 / 25.4, 168 / 25.4))
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.14, top=0.95, wspace=0.36, hspace=0.43)
    for i, (ax, s, d) in enumerate(zip(axes.flat, SPEC, panels)):
        x = d.relative_hour.to_numpy(float)
        y = d.estimate.to_numpy(float)
        eligible = d.n_sid.to_numpy() >= 2
        lo = np.where(eligible, d.lower.to_numpy(float), np.nan)
        hi = np.where(eligible, d.upper.to_numpy(float), np.nan)
        ax.axvline(0, color='#9EAAAF', ls=(0, (3, 3)), lw=0.75, zorder=0)
        if i >= 4:
            ax.axhline(0, color='#A9B3B7', lw=0.65, zorder=0)
        ax.fill_between(x, lo, hi, color=s['color'], alpha=0.16, lw=0)
        ax.plot(x, y, color=s['color'], marker='o', ms=3, lw=1.5)
        ax.set_xlim(-25, 25)
        ax.set_xticks([-24, -12, 0, 12, 24])
        ax.set_ylabel(s['ylabel'])
        ax.set_title(s['title'], loc='left', pad=7)
        ax.text(-0.25, 1.095, chr(97 + i), transform=ax.transAxes, fontsize=10.5, fontweight='bold', va='bottom')
        if i >= 4:
            ax.set_xlabel('Time from trough (h)')
    coverage = panels[0][panels[0].relative_hour.ge(0)]
    counts = '/'.join((str(int(v)) for v in coverage.n_sid_within_rise))
    # Actual time-specific support is retained in the caption transfer ledger.
    fig.canvas.draw()
    return (fig, axes)

def build():
    curves = load('curves')
    panels = [curves[curves.panel.eq(chr(97 + i))].copy().sort_values('relative_hour') for i in range(6)]
    g = load('g').sort_values('window')
    assert g.window.tolist() == [0, 1, 2] and set(g.n_sid) == {108} and (set(g.n_events) == {115})
    fig, axes = build_frozen(panels, None, None, None)
    axes = list(axes.flat)
    pos_before = [list(a.get_position().bounds) for a in axes]
    OLD_HEIGHT = 168
    HEIGHT = 202
    EXTRA = HEIGHT - OLD_HEIGHT
    fig.set_size_inches(180 / 25.4, HEIGHT / 25.4, forward=True)
    for i, ax in enumerate(axes):
        x, y, w, h = pos_before[i]
        ax.set_position([x, (y * OLD_HEIGHT + EXTRA) / HEIGHT, w, h * OLD_HEIGHT / HEIGHT])
        ax.set_gid(chr(97 + i))
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, (y * OLD_HEIGHT + EXTRA) / HEIGHT))
    ga = fig.add_axes([0.12, 12 / HEIGHT, 0.855, 22 / HEIGHT])
    ga.set_gid('g')
    y = np.array([2.0, 1.0, 0.0])
    est = g.estimate.to_numpy(float)
    lo = g.lower.to_numpy(float)
    hi = g.upper.to_numpy(float)
    span = max(hi.max(), 0) - min(lo.min(), 0)
    limits = (min(lo.min(), 0) - 0.09 * span, max(hi.max(), 0) + 0.09 * span)
    ga.axvline(0, color='#9EAAAF', lw=0.75, ls=(0, (3, 3)), zorder=0)
    ga.errorbar(est, y, xerr=np.vstack([est - lo, hi - est]), fmt='o', color='#82649A', ecolor='#82649A', markersize=4.2, elinewidth=1.3, capsize=3, capthick=1.1)
    ga.set_xlim(*limits)
    ga.set_ylim(-0.55, 2.55)
    ga.set_yticks(y, labels=['>16 d', '2–16 d', '<2 d'])
    ga.set_xticks([-2, 0, 2, 4, 6])
    ga.set_xlabel('Ascent − Resolution APE (J kg⁻¹)', labelpad=3)
    ga.set_title('700 hPa matched-intensity dry APE', loc='left', pad=7)
    ga.annotate('g', (0, 1), xycoords='axes fraction', xytext=(-36.6, 7.5), textcoords='offset points', fontsize=10.5, fontweight='bold', va='bottom')
    # The original 115 events / 108 SIDs are stated in the caption.
    fig.canvas.draw()
    for i, d in enumerate(panels):
        table('rise', chr(97 + i), d, 'Saved equal-SID pointwise histories and percentile intervals')
    table('rise', 'g', g, 'Matched ascent-minus-Resolution APE; original 115 events / 108 SID; three saved windows')
    export(fig, 'rise')
