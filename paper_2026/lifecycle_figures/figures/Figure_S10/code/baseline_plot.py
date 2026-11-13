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

def s6():
    name = 'FigS6'
    contr = source('framework/A3_paired_contrasts.csv')
    loss = external('03_Work/v62/framework/A3_SID_loss_bank.csv')
    agg = external('03_Work/v62/framework/A3_aggregate_losses.csv')
    table(name, 'b_c', loss, 'saved_sid_loss')
    table(name, 'd_e', contr, 'frozen_rmse_contrasts')
    table(name, 'b_e', agg, 'saved_aggregate_losses')
    assert len(loss) == 22900 and loss.sid.nunique() == 2290
    methods = ['actual_DE', 'coarse', 'time_midpoint', 'continuous_progress', 'continuous_age']
    labels = ['Actual D/E', 'Pooled D/E', 'Midpoint', 'Progress', 'Age']
    cs = [INK, '#A8B3B8', TEMP, FINAL, '#817297']
    fig = plt.figure(figsize=(180 / 25.4, 200 / 25.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.72, 1, 1])
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1])
    dd = fig.add_subplot(gs[2, 0])
    e = fig.add_subplot(gs[2, 1])
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.105, top=0.94, hspace=0.8, wspace=0.4)
    a.set_xlim(0, 2)
    a.set_ylim(0, 2)
    a.axis('off')
    title(a, 'a', 'Separate subdivision from recurrence retention')
    for x, y, t, st in [(0, 1, 'A4-F', 'D / E / F / Res'), (1, 1, 'A4-R', 'D / E / R / Tail'), (0, 0, 'A3-F', 'DE / F / Res'), (1, 0, 'A3-R', 'DE / R / Tail')]:
        a.add_patch(Rectangle((x + 0.03, y + 0.05), 0.93, 0.82, facecolor='#E4EDF0' if x == 0 else '#EFF1F1', edgecolor='none'))
        a.text(x + 0.5, y + 0.6, t, ha='center', weight='bold')
        a.text(x + 0.5, y + 0.27, st, ha='center')
    for idx, (response, label, unit) in enumerate([('X2_shear_ms', 'Deep-layer shear', '(m s⁻¹)²'), ('X3_mid_rh_pct', 'Mid-level RH', '(percentage points)²')]):
        a = [b, c][idx]
        for method, lab, col in zip(methods, labels, cs):
            v = loss.loc[loss.response.eq(response) & loss.method.eq(method), 'mse']
            target = agg[agg.response.eq(response) & agg.method.eq(method)].iloc[0]
            assert len(v) == 2290 and np.isclose(v.mean(), target.mse, rtol=1e-12)
            ecdf(a, v, color=col, lw=1.35 if method == 'actual_DE' else 0.9, label=lab)
        a.set_xscale('symlog', linthresh=1)
        a.set(xlabel=f'SID mean squared error\n{unit}', ylabel='Cumulative fraction', ylim=(0, 1))
        title(a, chr(98 + idx), label + ' · held-out errors')
        clean(a)
        a = [dd, e][idx]
        q = contr[contr.response.eq(response)].set_index('contrast').loc[['actual_DE-' + x for x in methods[1:]]]
        bars(a, q.delta_rmse, q.rmse_ci_low, q.rmse_ci_high, range(4), [TEMP, FINAL][idx], 0.65)
        a.axhline(0, color='#738690', lw=0.8)
        a.set_xticks(range(4), ['Pooled', 'Midpoint', 'Progress', 'Age'], fontsize=7.3)
        a.set_ylabel('Actual − reference RMSE\n' + ('m s⁻¹' if idx == 0 else 'RH percentage points'))
        title(a, chr(100 + idx), 'Paired RMSE comparisons')
        clean(a)
    fig.legend(handles=[Line2D([0], [0], color=c, label=l, lw=1.2) for c, l in zip(cs, labels)], ncol=5, frameon=False, fontsize=7.5, loc='center', bbox_to_anchor=(0.55, 0.355))
    finish(fig, name, exclude_axes=[a for a in fig.axes if not a.axison])
