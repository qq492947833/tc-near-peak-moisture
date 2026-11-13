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

def s4():
    name = 'FigS4'
    groups = source('FP0059')
    counts = source('FP0065')
    rcounts = source('FP0066')
    dur = source('FP0238')
    tr = source('FP0240')
    loo = source('FP0242')
    fig, ax = grid(3, 2, 238, left=0.16, hspace=0.83, wspace=0.5, bottom=0.13)
    a = ax[0, 0]
    cats = ['No return', 'One return', 'Two or more returns']
    cs = ['#C3CDD2', TEMP, FINAL]
    left = np.zeros(6)
    table(name, 'a', groups, 'recurrence')
    for cat, c in zip(cats, cs):
        q = groups[groups.group.eq(cat) & groups.basin.ne('Global')].set_index('basin').loc[BS]
        a.barh(range(6), q.fraction, left=left, height=0.65, color=c)
        left += q.fraction.to_numpy(float)
    a.set_yticks(range(6), BS)
    a.invert_yaxis()
    a.set(xlim=(0, 1), xlabel='Fraction within basin')
    title(a, 'a', 'Anchored return classes')
    a = ax[0, 1]
    table(name, 'b', counts, 'episodes')
    table(name, 'b', rcounts, 'reintensifications')
    a.bar(counts.near_lmi_episode_count - 0.18, counts['count'], width=0.36, color=TEMP, label='Anchored near-peak episodes')
    a.bar(rcounts.reintensification_event_count + 0.18, rcounts['count'], width=0.36, color=FINAL, label='Near-peak returns')
    a.set(xlabel='Number per cyclone', ylabel='Cyclones')
    a.legend(frameon=False, fontsize=7.1)
    a.set_xticks([0, 2, 4, 6, 8])
    title(a, 'b', 'Complete count distributions')
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(cs, ['No return', 'One return', 'Two or more returns'])], ncol=3, frameon=False, fontsize=7.1, loc='center', bbox_to_anchor=(0.54, 0.676))
    a = ax[1, 0]
    q = table(name, 'c', dur[dur.scope.isin(SCOPES) & dur.metric.eq('B') & dur.statistic.eq('mean')].set_index('scope').loc[SCOPES].reset_index(), 'scope_duration')
    bars(a, q.estimate, q.ci_low, q.ci_high, np.arange(len(SCOPES)), TEMP, 0.58, horizontal=True)
    a.set_yticks(range(len(SCOPES)), ['S' + str(i) for i in range(len(SCOPES))], fontsize=7.4)
    a.invert_yaxis()
    a.set_xlabel('Mean bridge duration (h)')
    title(a, 'c', 'Duration sensitivity')
    clean(a, 'x')
    a = ax[1, 1]
    q = table(name, 'd', tr[tr.scope.isin(SCOPES) & tr.metric.isin(['L', 'B'])], 'scope_trends')
    vals = q.pivot(index='scope', columns='metric', values='estimate').loc[SCOPES, ['L', 'B']].to_numpy(float)
    lim = max(abs(vals.min()), abs(vals.max()))
    mark = q.assign(cross=q.ci_low.le(0) & q.ci_high.ge(0)).pivot(index='scope', columns='metric', values='cross').loc[SCOPES, ['L', 'B']].to_numpy()
    im = matrix(a, vals, ['Later L', 'Bridge B'], ['S' + str(i) for i in range(len(SCOPES))], lim, mark, '.2f')
    title(a, 'd', 'Trends (h per decade)')
    a.grid(which='minor', color='white', lw=1.0)
    table(name, 'e_f', loo, 'all_deletion_values')
    for k, contrast in enumerate(['F_minus_U', 'U_minus_R']):
        a = ax[2, k]
        q = loo[loo.contrast.eq(contrast)]
        vals = [q.loc[q.variable.eq(v), 'change_from_main_sd'].to_numpy(float) * 10000.0 for v in VAR]
        edges = np.linspace(min(map(np.min, vals)), max(map(np.max, vals)), 62)
        hist = np.array([np.histogram(v, bins=edges)[0] for v in vals])
        assert np.array_equal(hist.sum(axis=1), np.array([len(v) for v in vals]))
        im = a.pcolormesh(edges, np.arange(7) - 0.5, np.ma.masked_equal(hist, 0), cmap='Blues', norm=LogNorm(1, 2340), shading='flat')
        hist_rows = [{'variable': var, 'bin_left': edges[j], 'bin_right': edges[j + 1], 'count': int(hist[i, j]), 'original_n': len(vals[i])} for i, var in enumerate(VAR) for j in range(61)]
        table(name, chr(101 + k), pd.DataFrame(hist_rows), 'descriptive_histogram_counts')
        a.set_yticks(range(6), VL)
        a.set_ylim(5.5, -0.5)
        a.axvline(0, color='#778A94', lw=0.7)
        a.set_xlabel('Deletion change (10⁻⁴ reference SD)')
        title(a, chr(101 + k), ['Influence on F − U', 'Influence on U − R'][k])
    cb = fig.add_axes([0.34, 0.045, 0.48, 0.009])
    fig.colorbar(im, cax=cb, orientation='horizontal', ticks=[1, 10, 100, 1000]).set_label('Deleted-SID results per bin (log colour scale)', fontsize=7.5)
    finish(fig, name, exclude_axes=[cb])
