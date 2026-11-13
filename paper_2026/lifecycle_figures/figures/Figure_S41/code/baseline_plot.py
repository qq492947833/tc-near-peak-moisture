MASTER_SHA = '0b4c4308543fb4e7f5f32911d6b654a5fc4e720b8617fc53ce46a59b2c117b37'
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
TEMP = BLUE = '#167B9E'
FINAL = ORANGE = '#CC7445'
INK = '#233A46'
MUTED = '#677B86'
GREY = '#8A969D'  # Original si14_26_rebuild.py palette constant
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

def pick(df, **conditions):
    d = df
    for k, v in conditions.items():
        d = d[d[k].eq(v)]
    assert len(d) == 1, (conditions, len(d))
    return d.iloc[0]

def decorate(ax, letter, title, pad=16):
    ax.set_gid(letter)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-16, pad - 4), textcoords='offset points', weight='bold', fontsize=10.5, ha='left', va='bottom')
    ax.set_title(title, loc='left', pad=pad, fontweight='normal')

def grid(rows=2, cols=2, height=170, left=0.13, hspace=0.72, wspace=0.45):
    fig, axs = plt.subplots(rows, cols, figsize=(180 / 25.4, height / 25.4), squeeze=False)
    fig.subplots_adjust(left=left, right=0.965, bottom=0.11, top=0.91, wspace=wspace, hspace=hspace)
    return (fig, axs)

def bars(ax, data, keys, groups, labelmap=None, colors=None, xkey='field', groupkey='policy', est='estimate', low='lo', high='hi', scale=1, uncertainty=True):
    xx = np.arange(len(keys))
    width = 0.72 / len(groups)
    for i, g in enumerate(groups):
        sub = data[data[groupkey].eq(g)].set_index(xkey).loc[keys]
        e = sub[est].to_numpy(float) / scale
        color = (colors or {}).get(g, [BLUE, ORANGE, GREY][i % 3])
        pos = xx + (i - (len(groups) - 1) / 2) * width
        ax.bar(pos, e, width * 0.86, color=color, label=(labelmap or {}).get(g, g), zorder=2)
        if uncertainty:
            lo = sub[low].to_numpy(float) / scale
            hi = sub[high].to_numpy(float) / scale
            valid = np.isfinite(lo) & np.isfinite(hi)
            ax.errorbar(pos[valid], e[valid], yerr=[e[valid] - lo[valid], hi[valid] - e[valid]], fmt='none', ecolor=INK, elinewidth=0.75, capsize=2, zorder=3)
    ax.axhline(0, color='#9BA5AA', lw=0.65, zorder=1)
    ax.set_xticks(xx)
    ax.yaxis.set_major_locator(MaxNLocator(4))

def heat(ax, values, xlabels, ylabels, vmax=None, fmt='.2f', cmap='RdBu_r', mark=None):
    values = np.asarray(values, float)
    vmax = max(np.nanmax(np.abs(values)), 1e-10) if vmax is None else vmax
    im = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0, pad=6)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                text = format(values[i, j], fmt)
                ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if abs(values[i, j]) > 0.63 * vmax else INK)
                if mark is not None and mark[i, j]:
                    ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, lw=1.2, zorder=5))
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im

def waterfall(ax, vals, labels, colors=None, errors=None):
    """Show one exact decomposition; final item is a saved total, not another addend."""
    vals = np.asarray(vals, float)
    assert np.isclose(vals[:-1].sum(), vals[-1], rtol=1e-09, atol=1e-10), (vals, vals[:-1].sum())
    pos = np.arange(len(vals))
    start = 0.0
    for i, v in enumerate(vals):
        b = 0 if i == len(vals) - 1 else start
        color = (colors or [BLUE, ORANGE, GREY, INK])[i]
        ax.bar(i, v, 0.65, bottom=b, color=color, zorder=2)
        if i < len(vals) - 1:
            start += v
            ax.plot([i + 0.33, i + 0.67], [start, start], color='#ABB4B9', lw=0.7)
        if errors is not None:
            lo, hi = errors[i]
            ax.errorbar(i, b + v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor=INK, capsize=2, lw=0.7)
    ax.axhline(0, color='#9BA5AA', lw=0.65)
    ax.set_xticks(pos, labels)
    edges = np.r_[0.0, np.cumsum(vals[:-1]), vals[-1]]
    lo, hi = (float(edges.min()), float(edges.max()))
    pad = max((hi - lo) * 0.12, 1e-08)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(MaxNLocator(4))

def fig25():
    name = 'FigS25'
    ms = 'geography/environment_model_paired_comparisons.csv'
    cs = 'geography/post_process_curves.csv'
    models, curves = map(read, [ms, cs])
    fig, ax = grid(1, 2, 125, left=0.12, wspace=0.48)
    fig.subplots_adjust(bottom=0.24)
    names = ['M0+dynamics', 'M0+moisture', 'M0+thermodynamics', 'M0+all']
    z = models[models.metric.eq('interval_log_likelihood') & models.model.isin(names)].set_index('model').loc[names]
    ax[0, 0].bar(range(4), z.paired_difference, color=[GREY, GREY, BLUE, INK], width=0.6)
    ax[0, 0].errorbar(range(4), z.paired_difference, yerr=[z.paired_difference - z.ci_low, z.ci_high - z.paired_difference], fmt='none', ecolor=INK, capsize=3)
    ax[0, 0].set(xticks=range(4), xticklabels=['Dynamics', 'Moisture', 'Thermo-\ndynamics', 'All'], ylabel='Held-out log-likelihood gain per SID')
    ax[0, 0].tick_params(axis='x', labelrotation=0)
    ax[0, 0].axhline(0, color=GREY, ls=':', lw=0.7)
    decorate(ax[0, 0], 'a', 'Global information: 1,281 cyclones')
    log(name, 'a', ms, z.reset_index(), 'Separate additions and joint addition to the same baseline; not additive causal shares')
    z = curves[curves.radial.eq('outer') & curves.field.eq('q700')]
    for group, col, ls, label in [('known_return', BLUE, '-', 'Return (15)'), ('complete_nonreturn', ORANGE, '-', 'Nonreturn (30)'), ('unknown', GREY, '--', 'Unknown (42)')]:
        q = z[z.outcome_group.eq(group)].sort_values('lag_h')
        ax[0, 1].plot(q.lag_h, q.estimate, 'o', ls=ls, color=col, lw=1.4, ms=3, label=label)
        ax[0, 1].fill_between(q.lag_h, q.ci_low, q.ci_high, color=col, alpha=0.14, lw=0)
    ax[0, 1].set(xticks=[0, 6, 12, 18, 24], xlabel='Hours after departure', ylabel='700-hPa specific humidity (g kg⁻¹)')
    fig.legend(*ax[0, 1].get_legend_handles_labels(), ncol=3, frameon=False, fontsize=7.5, loc='lower center', bbox_to_anchor=(.55, .06))
    decorate(ax[0, 1], 'b', 'SCS outer-annulus trajectories')
    log(name, 'b', cs, z)
    finish(fig, name)
