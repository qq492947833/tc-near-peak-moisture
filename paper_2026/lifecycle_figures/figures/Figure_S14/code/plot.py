from display_format import format_number,format_group
MASTER_SHA = '0b4c4308543fb4e7f5f32911d6b654a5fc4e720b8617fc53ce46a59b2c117b37'
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
TEMP = BLUE = '#167B9E'
FINAL = ORANGE = '#CC7445'
INK = '#233A46'
MUTED = '#677B86'
GREEN = '#418D79'
RINGS = ['100_300km', '300_800km']
RINGLABELS = ['100–300 km', '300–800 km']
CMAP = LinearSegmentedColormap.from_list('signed_native', [ORANGE, '#FAFAF7', BLUE])
VAR = ['X1_sst_c', 'X2_shear_ms', 'X3_mid_rh_pct', 'X4_d200_1e5s', 'X5_minus_omega500_pa_s', 'X6_signed_vo850_1e5s']
VL = ['SST', 'Deep shear', 'Mid-level RH', 'Divergence', 'Ascent', 'Vorticity']
BS = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']
BC = dict(zip(BS, ['#437F9C', '#BD9145', '#428C79', '#9D82AB', '#AB685E', '#67A4AC']))
BASINS = BS
WIDTH_MM = 180
WIDTH_IN = 180 / 25.4
GRID = '#DDE3E7'
SCOPES = ['main_all', 'exclude_single_6h', 'strict_bridge_tropical', 'without_Freddy2023', 'without_basin_WP', 'without_basin_EP', 'without_basin_SI', 'without_basin_NA', 'without_basin_SP', 'without_basin_NI']

def title(ax, label, text):
    ax.set_gid(label)
    ax.set_title(text, loc='left', pad=12)
    ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(-18, 12), textcoords='offset points', weight='bold', fontsize=10, va='baseline')

def clean(ax, axis='y'):
    ax.grid(axis=axis, color='#E7ECEE', lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)

def grid(rows, cols, height=175, left=0.14, bottom=0.1, top=0.92, hspace=0.72, wspace=0.42):
    fig, axs = plt.subplots(rows, cols, figsize=(180 / 25.4, height / 25.4), squeeze=False)
    fig.subplots_adjust(left=left, right=0.975, bottom=bottom, top=top, hspace=hspace, wspace=wspace)
    return (fig, axs)

def ecdf(ax, values, **kw):
    a = np.sort(np.asarray(values, dtype=float))
    assert np.isfinite(a).all()
    ax.step(a, np.arange(1, len(a) + 1) / len(a), where='post', **kw)

def bars(ax, est, lo, hi, pos, color, width=0.6, horizontal=False, label=None):
    e = np.asarray(est, float)
    l = np.asarray(lo, float)
    h = np.asarray(hi, float)
    if horizontal:
        ax.barh(pos, e, height=width, color=color, label=label, zorder=2)
        ax.errorbar(e, pos, xerr=[e - l, h - e], fmt='none', ecolor=INK, lw=0.75, capsize=2, zorder=3)
    else:
        ax.bar(pos, e, width=width, color=color, label=label, zorder=2)
        ax.errorbar(pos, e, yerr=[e - l, h - e], fmt='none', ecolor=INK, lw=0.75, capsize=2, zorder=3)

def matrix(ax, values, xlabels, ylabels, limit, zero_mark=None, fmt='.3f'):
    v = np.asarray(values, float)
    masked = np.ma.masked_invalid(v)
    cm = mpl.colormaps['RdBu_r'].copy()
    cm.set_bad('#E5E8EA')
    im = ax.imshow(masked, cmap=cm, norm=TwoSlopeNorm(0, -limit, limit), aspect='auto')
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0, pad=6)
    ax.set_xticks(np.arange(-0.5, len(xlabels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels)), minor=True)
    ax.grid(which='minor', color='white', lw=3)
    ax.tick_params(which='minor', length=0)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            text = 'NA' if not np.isfinite(v[i, j]) else format_number(v[i, j])
            if zero_mark is not None and bool(zero_mark[i, j]):
                text += '†'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if np.isfinite(v[i, j]) and abs(v[i, j]) > 0.65 * limit else INK)
    return im

def s11():
    name = 'FigS11'
    d = source('climate/P4_raw_climate_anomaly_CI_n.csv')
    fig, ax = grid(2, 2, 160, left=0.13, top=0.88, hspace=0.55, wspace=0.45, bottom=0.13)
    for row, (field, tl, unit) in enumerate([('q700', 'Specific humidity', 'g kg⁻¹'), ('r700', 'Relative humidity', '%')]):
        for col, quantities in enumerate([['raw_common', 'climate_common'], ['anomaly_common']]):
            a = ax[row, col]
            q = table(name, chr(97 + row * 2 + col), d[d.field.eq(field) & d.quantity.isin(quantities)], 'time_series')
            for j, (bound, color) in enumerate([('D_to_E', TEMP), ('E_to_F', FINAL)]):
                for quantity in quantities:
                    z = q[q.boundary.eq(bound) & q.quantity.eq(quantity)].sort_values('lag_h')
                    a.fill_between(z.lag_h, z.ci_low, z.ci_high, color=color, alpha=0.1, lw=0)
                    a.plot(z.lag_h, z.estimate, color=color, ls='--' if quantity == 'climate_common' else '-', marker=['o', 's'][j], ms=2.5, lw=1)
            a.axvline(0, color='#91A1A9', lw=0.7, ls=':')
            a.set_xticks([-24, -12, 0, 12, 24])
            a.set_xlabel('Time from transition (h)')
            a.set_ylabel(('Mean ' if col == 0 else 'Anomaly ') + ('(' + unit + ')' if not (row and col) else '(RH percentage points)'))
            title(a, chr(97 + row * 2 + col), tl + ('\nObserved and background' if col == 0 else '\nObserved − background'))
            clean(a)
    fig.legend(handles=[Line2D([0], [0], color=TEMP, marker='o', label='D → E'), Line2D([0], [0], color=FINAL, marker='s', label='E → F'), Line2D([0], [0], color=INK, label='Observed'), Line2D([0], [0], color=INK, ls='--', label='Climatology')], ncol=4, frameon=False, fontsize=7.7, loc='upper center', bbox_to_anchor=(0.55, 0.995))
    finish(fig, name)
