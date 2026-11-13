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

def matrix(fig, ax, values, rows, cols, unit):
    z = np.asarray(values, float)
    lim = nice_symmetric_limit(max(0.01, float(abs(z).max())))
    if unit=='Integrated tendency (g kg⁻¹)':lim=2.0
    if unit=='Change (m s⁻¹)':lim=20.0
    assert np.nanmax(abs(z))<=lim
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    mesh = ax.pcolormesh(np.arange(z.shape[1] + 1) - 0.5, np.arange(z.shape[0] + 1) - 0.5, z, cmap=CMAP, norm=norm, edgecolors='white', lw=1.5)
    ax.set(xlim=(-0.5, z.shape[1] - 0.5), ylim=(z.shape[0] - 0.5, -0.5), xticks=range(z.shape[1]), xticklabels=cols, yticks=range(z.shape[0]), yticklabels=rows)
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, format_number(z[i, j], decimals=2), ha='center', va='center', fontsize=8, color='black')
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.060, pos.width, 0.012], label='<colorbar>')
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal', ticks=[-lim, 0, lim])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, pad=1, length=2)
    cb.set_label(unit, fontsize=7.5, labelpad=1)
    return cax

def fig3(dpi):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 165 / 25.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
    fig.subplots_adjust(left=0.18, right=0.97, top=0.94, bottom=0.09, wspace=0.58, hspace=0.84)
    a, b, c, d, e = [fig.add_subplot(s) for s in [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1], gs[2, :]]]
    groups = ['D_to_E', 'E_to_F']
    cols = ['D → E', 'E → F']
    bars = []
    ledger = []
    z = read('fig3_background.csv')
    z = z[z.field.eq('q700') & z.window.eq('-24_0')]
    cats = ['raw_change', 'climate_change', 'anomaly_change']
    vals = [[float(z[z.boundary.eq(g) & z.component.eq(k)].estimate.iloc[0]) for g in groups] for k in cats]
    bars.append(matrix(fig, a, vals, ['Observed', 'Background', 'Anomaly'], cols, 'Humidity change (g kg⁻¹)'))
    title(a, 'a', 'Humidity along the track')
    ledger.append({'panel': 'a', 'values': vals})
    z = read('fig3_rh.csv')
    z = z[z.scope.eq('primary25')]
    cats = ['time_q_contribution', 'time_T_contribution', 'domain_symmetric', 'fixed_support_endpoint_change']
    assert z[z.variable.eq('support_change')][['estimate', 'ci_low', 'ci_high']].eq(0).all().all()
    vals = [[float(z[z.boundary.eq(g) & z.level_hpa.eq(lev) & z.variable.eq(k)].estimate.iloc[0]) for lev in [700, 500] for g in groups] for k in cats]
    bars.append(matrix(fig, b, vals, ['Humidity', 'Temperature', 'Geography', 'Total'], ['700 hPa\nD → E', '700 hPa\nE → F', '500 hPa\nD → E', '500 hPa\nE → F'], 'RH change (pp)'))
    title(b, 'b', 'RH change: time and location contributions')
    # Keep the longer, same-size title on one line inside the unchanged canvas.
    b._left_title.set_transform(b._left_title.get_transform() + mpl.transforms.ScaledTranslation(-1.8 / 25.4, 0, fig.dpi_scale_trans))
    ledger.append({'panel': 'b', 'values': vals})
    tr = read('fig3_transport.csv')
    cats = ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency']
    z = tr[tr.estimand.eq('integral_minus24_0') & tr.field.isin(cats)]
    vals = [[float(z[z.boundary.eq(g) & z.field.eq(k)].estimate.iloc[0]) for g in groups] for k in cats]
    bars.append(matrix(fig, c, vals, ['Horizontal', 'Vertical', 'Residual', 'Net tendency'], cols, 'Integrated tendency (g kg⁻¹)'))
    title(c, 'c', 'Moisture-budget contributions')
    ledger.append({'panel': 'c', 'values': vals})
    z = read('fig3_PI.csv')
    cats = ['PI', 'minus_V', 'headroom']
    vals = [[float(z[z.boundary.eq(g) & z.component.eq(k)].estimate.iloc[0]) for g in groups] for k in cats]
    bars.append(matrix(fig, d, vals, ['ΔPI', '−ΔV', 'Δ(PI − V)'], cols, 'Change (m s⁻¹)'))
    title(d, 'd', 'Potential intensity and wind')
    ledger.append({'panel': 'd', 'values': vals})
    for g, col, lab in zip(groups, [BLUE, ORANGE], cols):
        z = tr[tr.boundary.eq(g) & tr.field.eq('q850_MFC') & tr.estimand.str.startswith('lag_')].copy()
        z['lag'] = z.estimand.str.replace('lag_', '', regex=False).astype(int)
        z = z.sort_values('lag')
        e.plot(z.lag, z.estimate, color=col, lw=1.3, label=lab)
        e.fill_between(z.lag, z.lo, z.hi, color=col, alpha=0.16, lw=0)
    e.axvline(0, color=GREY, lw=0.7, ls='--')
    e.axhline(0, color=GREY, lw=0.65)
    grid(e)
    e.set(xlabel='Time from transition (h)', ylabel='850 hPa MFC\n(g kg⁻¹ day⁻¹)', xticks=[-24, -12, 0, 12, 24])
    e.set_ylim(0.9, 2.1)
    e.legend(loc='lower left', ncol=2)
    title(e, 'e', 'Lower-level moisture supply')
    (OUT / 'Figure_03_matrix_values.json').write_text(json.dumps(ledger, indent=2), encoding='utf8')
    save(fig, 'Figure_03_transition_matrices_candidate', dpi, exclude=bars)

def build(dpi): fig3(dpi)
