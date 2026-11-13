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
            text = 'NA' if not np.isfinite(v[i, j]) else format(v[i, j], fmt)
            if zero_mark is not None and bool(zero_mark[i, j]):
                text += '†'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if np.isfinite(v[i, j]) and abs(v[i, j]) > 0.65 * limit else INK)
    return im

def s9():
    name = 'FigS9'
    d = source('SF-F-H4-30')
    de = source('SF-F-EXACT-DE-6')
    table(name, 'a', d, 'complete_H4_family')
    table(name, 'b', de, 'exact_DE_allocation')
    fig = plt.figure(figsize=(180 / 25.4, 190 / 25.4))
    a = fig.add_axes([0.18, 0.48, 0.78, 0.43])
    b = fig.add_axes([0.08, 0.09, 0.88, 0.25])
    cols = ['D', 'E', 'F', 'Res', 'D-E']
    v = d.pivot(index='variable', columns='stage', values='percent_per_1sd').loc[VAR, cols]
    limit = float(np.max(np.abs(v.to_numpy())))
    a.set_xlim(-0.6, 4.6)
    a.set_ylim(5.6, -0.6)
    a.set_xticks(range(5), ['D', 'E', 'F', 'Res', 'D − E'])
    a.set_yticks(range(6), VL)
    a.grid(axis='y', color='#E4EBEE', lw=0.65)
    a.set_axisbelow(True)
    norm = TwoSlopeNorm(0, -limit, limit)
    for i, var in enumerate(VAR):
        for j, st in enumerate(cols):
            z = d[d.variable.eq(var) & d.stage.eq(st)].iloc[0]
            val = z.percent_per_1sd
            a.scatter(j, i, s=90 + 360 * abs(val) / limit, c=[val], cmap='RdBu_r', norm=norm, edgecolors=INK if z.q < 0.05 else 'white', linewidths=1.1, zorder=2)
            a.text(j, i + 0.31, f'{val:.1f}' + ('†' if z.ci_low_percent <= 0 <= z.ci_high_percent else ''), ha='center', va='top', fontsize=8)
    title(a, 'a', 'Stage-duration response per fixed environmental SD (%)')
    b.axis('off')
    b.set_xlim(0, 1)
    b.set_ylim(6.7, -1.3)
    title(b, 'b', 'Exact D + E allocation · native-unit coefficients')
    for x, t in [(0, 'Factor'), (0.22, 'Per unit'), (0.39, 'Estimate'), (0.56, '95% analytic interval'), (0.87, 'P')]:
        b.text(x, -0.45, t, weight='bold', fontsize=8)
    units = ['°C', 'm s⁻¹', 'RH pp', '10⁻⁵ s⁻¹', 'Pa s⁻¹', '10⁻⁵ s⁻¹']
    for i, var in enumerate(VAR):
        z = de[de.variable.eq(var)].iloc[0]
        for x, t in [(0, VL[i]), (0.22, units[i]), (0.39, f'{z.coef:.4g}'), (0.56, f'[{z.ci_low:.4g}, {z.ci_high:.4g}]'), (0.87, f'{z.p:.3g}')]:
            b.text(x, i + 0.45, t, fontsize=8)
    fig.text(0.18, 0.408, 'Outline: BH q < 0.05     † 95% analytic interval includes zero', fontsize=7.5)
    finish(fig, name, exclude_axes=[b])
