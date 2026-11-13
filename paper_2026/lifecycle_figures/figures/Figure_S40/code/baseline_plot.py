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

def fig24():
    name = 'FigS24'
    sources = ['geography/pre_environment_group_contrasts.csv', 'geography/C_q700_SCS_return_contrasts.csv', 'geography/post_process_return_contrasts.csv']
    pre, climate, budget = map(read, sources)
    fig, ax = grid(3, 2, 210, left=0.16, hspace=0.85, wspace=0.65)
    fields = ['shear200_850', 'shear500_850', 'shear200_500', 'cyclonic_vo850', 'midRH', 'q700_inner', 'q700_outer', 'SST', 'PI_headroom']
    labels = ['Deep shear', 'Lower shear', 'Upper shear', 'Vorticity', 'Mid-level RH', 'Inner q', 'Outer q', 'SST', 'PI − V']
    z = pre[pre.region.eq('SCS_box') & pre.comparison.eq('known_return minus complete_nonreturn') & pre.variable.isin(fields)].set_index('variable').loc[fields]
    e = z.standardized_difference
    ax[0, 0].barh(range(9), e, color=[BLUE] * 4 + [GREEN] * 3 + [ORANGE] * 2, height=0.62)
    lo = z.standardized_ci_low
    hi = z.standardized_ci_high
    valid = np.isfinite(lo) & np.isfinite(hi)
    ax[0, 0].errorbar(e[valid], np.arange(9)[valid], xerr=[e[valid] - lo[valid], hi[valid] - e[valid]], fmt='none', ecolor=INK, capsize=2, lw=0.7)
    ax[0, 0].set(yticks=range(9), yticklabels=labels, ylim=(8.6, -0.6), xlabel='Pre-event contrast / reference SD')
    ax[0, 0].axvline(0, color=GREY, ls=':', lw=0.7)
    decorate(ax[0, 0], 'a', 'Measured pre-event environment')
    log(name, 'a', sources[0], z.reset_index(), 'Separate outcome groups; diagnostic-specific observed support')
    z = climate.copy()
    bars(ax[0, 1], z, ['q700_raw_common', 'q700_background', 'q700_anomaly'], RINGS, dict(zip(RINGS, RINGLABELS)), dict(zip(RINGS, [BLUE, ORANGE])), xkey='component', groupkey='radial', low='ci_low', high='ci_high')
    ax[0, 1].set_xticklabels(['Observed', 'Background', 'Anomaly'], rotation=20, ha='right')
    ax[0, 1].set_ylabel('Humidity contrast (g kg⁻¹)')
    decorate(ax[0, 1], 'b', 'Humidity and its background')
    log(name, 'b', sources[1], z)
    fields = ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency']
    for j, radial in enumerate(['inner', 'outer']):
        z = budget[budget.radial.eq(radial) & budget.field.isin(fields)].set_index('field').loc[fields]
        waterfall(ax[1, j], z.estimate, ['Horizontal', 'Vertical', 'Residual', 'Net'])
        ax[1, j].tick_params(axis='x', labelrotation=20)
        ax[1, j].set_ylabel('Integrated contrast (g kg⁻¹)')
        decorate(ax[1, j], chr(99 + j), RINGLABELS[j] + ' budget')
        log(name, chr(99 + j), sources[2], z.reset_index(), 'Group difference; complete original marginal CI/n supplied in table')
    fields = ['fixed_tendency', 'motion_term', 'moving_tendency']
    z = budget[budget.field.isin(fields) & budget.radial.isin(['inner', 'outer'])]
    bars(ax[2, 0], z, fields, ['inner', 'outer'], dict(zip(['inner', 'outer'], RINGLABELS)), dict(zip(['inner', 'outer'], [BLUE, ORANGE])), groupkey='radial', low='ci_low', high='ci_high')
    ax[2, 0].set_xticklabels(['Fixed-frame', 'Translation', 'Moving'], rotation=20, ha='right')
    ax[2, 0].set_ylabel('Integrated contrast (g kg⁻¹)')
    decorate(ax[2, 0], 'e', 'Frame and motion terms')
    log(name, 'e', sources[2], z)
    z = budget[budget.field.eq('q850_absolute_MFC')].set_index('radial').loc[['inner', 'outer']]
    ax[2, 1].bar([0, 1], z.estimate, color=[BLUE, ORANGE], width=0.6)
    ax[2, 1].errorbar([0, 1], z.estimate, yerr=[z.estimate - z.ci_low, z.ci_high - z.estimate], fmt='none', ecolor=INK, capsize=3)
    ax[2, 1].set(xticks=[0, 1], xticklabels=RINGLABELS, ylabel='Integrated MFC contrast (g kg⁻¹)')
    ax[2, 1].axhline(0, color=GREY, lw=0.7)
    decorate(ax[2, 1], 'f', '850-hPa moisture convergence')
    log(name, 'f', sources[2], z.reset_index(), 'Own pressure-level support')
    fig.legend([Patch(color=BLUE), Patch(color=ORANGE)], RINGLABELS, loc='upper center', bbox_to_anchor=(0.56, 0.995), ncol=2, frameon=False)
    finish(fig, name)
