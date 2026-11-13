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

def s5():
    name = 'FigS5'
    native = source('FP0071')
    dur = source('FP0223')
    env = source('FP0230')
    fig, ax = grid(2, 2, 190, left=0.16, hspace=0.75, wspace=0.48, bottom=0.15)
    a = ax[0, 0]
    table(name, 'a', native, 'native_thresholds')
    rows = [0.2, 0.25, 0.3]
    cols = [(15, 0.85), (15, 0.9), (15, 0.95), (20, 0.85), (20, 0.9), (20, 0.95)]
    vals = np.array([[int(native[native.escalation_progress.eq(e) & native.min_peak_gain.eq(g) & native.maturity_progress.eq(m)].n_valid.iloc[0]) for g, m in cols] for e in rows])
    a.imshow(vals, aspect='auto', cmap='Blues')
    a.set_yticks(range(3), ['0.20', '0.25', '0.30'])
    a.set_xticks(range(6), [f'{m:.2f}\n{g}' for g, m in cols], fontsize=7.3)
    a.set(xlabel='Near-peak threshold / minimum gain (kt)', ylabel='Intensification threshold')
    for i in range(3):
        for j in range(6):
            a.text(j, i, str(vals[i, j]), ha='center', va='center', fontsize=7.3, color='white' if vals[i, j] > vals.mean() else INK)
    title(a, 'a', 'Valid cyclones · all 18 definitions')
    a = ax[0, 1]
    scopes = ['beta085_native_multi', 'beta085_fixed90_multi_main', 'beta085_fixed90_multi_alternate', 'beta095_native_multi', 'beta095_fixed90_multi_main', 'beta095_fixed90_multi_alternate']
    q = table(name, 'b', dur[dur.scope.isin(scopes) & dur.metric.isin(['L', 'B']) & dur.statistic.eq('mean')], 'duration')
    for j, key in enumerate(['L', 'B']):
        z = q[q.metric.eq(key)].set_index('scope').loc[scopes]
        bars(a, z.estimate, z.ci_low, z.ci_high, np.arange(6) + (j - 0.5) * 0.32, [TEMP, FINAL][j], 0.29, label=['Later L', 'Bridge B'][j])
    a.set_xticks(range(6), ['Native', 'Main', 'Alt.', 'Native', 'Main', 'Alt.'], fontsize=7.3)
    a.set_ylabel('Mean duration (h)')
    a.set_ylim(0, 52)
    a.text(1, -0.34, '0.85', transform=a.get_xaxis_transform(), ha='center')
    a.text(4, -0.34, '0.95', transform=a.get_xaxis_transform(), ha='center')
    a.legend(frameon=False, ncol=2, fontsize=7.3)
    title(a, 'b', 'Native and fixed recurrence')
    clean(a)
    assert len(env) == 36
    lim = float(np.nanmax(np.abs(env.estimate_sd)))
    for k, contrast in enumerate(['F_minus_U', 'U_minus_R']):
        a = ax[1, k]
        q = table(name, chr(99 + k), env[env.contrast.eq(contrast)], 'all_18_threshold_contrasts')
        q = q.assign(cross=q.ci_low_sd.le(0) & q.ci_high_sd.ge(0))
        vals = q.pivot(index='variable', columns='alternate', values='estimate_sd').loc[VAR, ['beta085', 'beta095', 'C5']]
        mark = q.pivot(index='variable', columns='alternate', values='cross').loc[VAR, ['beta085', 'beta095', 'C5']]
        im = matrix(a, vals, ['0.85', '0.95', 'C5'], VL, lim, mark.to_numpy(), '.4f')
        title(a, chr(99 + k), ['Definition effect on F − U', 'Definition effect on U − R'][k])
    cb = fig.add_axes([0.3, 0.065, 0.53, 0.014])
    fig.colorbar(im, cax=cb, orientation='horizontal').set_label('Alternative − main / original reference SD')
    fig.text(0.565, 0.103, '† 95% interval includes zero', ha='center', fontsize=7.6)
    finish(fig, name, exclude_axes=[cb])
