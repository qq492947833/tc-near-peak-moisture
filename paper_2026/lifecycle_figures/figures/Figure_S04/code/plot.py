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

def s3():
    name = 'FigS3'
    dur = external('04_Data_and_Code/github_ready/v67_review/baseline/M2_inputs/durations.csv')
    summaries = source('thematic/fig2_duration.csv')
    w = source('thematic/fig2_window.csv')
    colors = ['#354F5E', '#8B9EA8', TEMP, FINAL, '#629786', '#9574A5']
    keys = ['R', 'R_long', 'U', 'F', 'L', 'B']
    labs = ['R', 'Longest', 'U', 'F', 'Later L', 'Bridge B']
    fig = plt.figure(figsize=(180 / 25.4, 175 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.12])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.91, wspace=0.35, hspace=0.6)
    for idx, scope in enumerate(['main_all', 'main_multi']):
        a = axs[idx]
        q = dur[dur.main_complete & (dur.multi if idx else True)]
        table(name, chr(97 + idx), q, 'individual_durations')
        z = table(name, chr(97 + idx), summaries[summaries.scope.eq(scope) & summaries.metric.isin(keys)], 'original_summaries')
        assert len(q) == [2340, 123][idx]
        for key, lab, c in zip(keys, labs, colors):
            values = q['T_' + key + '_h'].to_numpy(float)
            old = z[z.metric.eq(key) & z.statistic.eq('mean')].iloc[0]
            assert np.isclose(values.mean(), old.estimate, rtol=1e-13)
            ecdf(a, values, color=c, lw=1.25, label=lab)
        a.set(xlabel='Covered duration (h)', ylabel='Cumulative fraction', ylim=(0, 1))
        title(a, chr(97 + idx), ['All common histories', 'Recurrent subset'][idx])
        clean(a)
    fig.legend(handles=[Line2D([0], [0], color=c, label=l) for c, l in zip(colors, labs)], ncol=6, frameon=False, loc='center', bbox_to_anchor=(0.55, 0.545))
    a = axs[2]
    q = table(name, 'c', w[w.scope.eq('main_multi')], 'recurrent_windows')
    for j, (contrast, col, lab) in enumerate([('F_minus_U', '#555B77', 'Mean F − mean U'), ('U_minus_R', '#B45B7B', 'Mean U − mean R')]):
        z = q[q.contrast.eq(contrast)].set_index('variable').loc[VAR]
        bars(a, z.estimate_sd, z.ci_low_sd, z.ci_high_sd, np.arange(6) + (j - 0.5) * 0.32, col, 0.29, label=lab)
    a.axhline(0, color='#7B8C94', lw=0.7)
    a.set_xticks(range(6), VL)
    a.set_ylabel('Paired difference / reference SD')
    a.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.13))
    title(a, 'c', 'Recurrent-window environmental differences')
    clean(a)
    finish(fig, name)
