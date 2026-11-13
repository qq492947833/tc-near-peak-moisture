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

def fig22():
    name = 'FigS22'
    sources = ['geography/G1_genesis_basin_recurrence.csv', 'geography/general_reintensification_genesis_basin_rates.csv', 'geography/G3_region_followup_counts.csv', 'geography/G3_region_risk_estimates.csv', 'geography/post_process_return_contrasts.csv']
    near, general, counts, risk, budget = map(read, sources)
    fig = plt.figure(figsize=(180 / 25.4, 205 / 25.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.85, 1.1, 0.8])
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    a, b, c, d, e = axes
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.09, top=0.92, wspace=0.65, hspace=0.95)
    z1 = near[near.config.eq('main') & near.sensitivity.eq('all') & near.episode_category.eq('any_recurrence') & near.region.isin(BASINS)].copy()
    z2 = general[general.scope.eq('master2340') & general.region.isin(BASINS)].copy()
    for j, (z, col, label) in enumerate([(z1, BLUE, 'Near-peak recurrence'), (z2, ORANGE, 'Post-LMI reintensification')]):
        q = z.set_index('region').loc[BASINS]
        a.bar(np.arange(6) + (j - 0.5) * 0.34, q.fraction * 100, 0.3, color=col, label=label)
        a.errorbar(np.arange(6) + (j - 0.5) * 0.34, q.fraction * 100, yerr=[(q.fraction - q.ci_low) * 100, (q.ci_high - q.fraction) * 100], fmt='none', ecolor=INK, capsize=2, lw=0.7)
    a.set(xticks=range(6), xticklabels=BASINS, ylabel='Genesis-basin fraction (%)')
    fig.legend(*a.get_legend_handles_labels(), frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.55, 0.995))
    decorate(a, 'a', 'Different definitions on the same 2,340 histories')
    log(name, 'a', sources[0], z1)
    log(name, 'a', sources[1], z2)
    regs = BASINS + ['SCS_box', 'Japan_Sea_box']
    labels = BASINS + ['SCS', 'Japan Sea']
    q = counts.set_index('region').loc[regs]
    left = np.zeros(len(regs))
    for field, col, label in [('known_return', BLUE, 'Return'), ('unknown_outcome', '#C6D0D6', 'Unknown'), ('complete_nonreturn', '#ECEFF1', 'Complete nonreturn')]:
        frac = np.divide(q[field], q.n_risk, out=np.zeros(len(q), float), where=q.n_risk.to_numpy() > 0) * 100
        b.barh(np.arange(len(regs)), frac, left=left, color=col, height=0.65, label=label)
        left += frac
    b.set(yticks=range(len(regs)), yticklabels=[f'{lab} ({int(n)})' for lab, n in zip(labels, q.n_risk)], ylim=(7.6, -0.6), xlim=(0, 100), xlabel='Observed outcome composition (%)')
    b.text(0, 7, 'No eligible departures', va='center', fontsize=7.5, color=MUTED)
    b.legend([Patch(color=BLUE), Patch(color='#C6D0D6'), Patch(color='#ECEFF1')], ['Return', 'Unknown', 'Nonreturn'], ncol=3, frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.45, -0.28), columnspacing=0.6, handlelength=0.7)
    decorate(b, 'b', 'Departure-origin denominators')
    log(name, 'b', sources[2], q.reset_index(), 'Known-return to known-return+unknown is the observed risk bound, not CI')
    z = risk[risk.branch.eq('shear200_850') & risk.method.eq('natural_gformula') & risk.region.isin(regs)].set_index('region').loc[regs]
    c.axis('off')
    cells = []
    for region in regs:
        r = z.loc[region]
        cells.append(['—' if not np.isfinite(r.risk_percent) else f'{r.risk_percent:.1f}', '—' if not np.isfinite(r.ci_low) else f'{r.ci_low:.1f}–{r.ci_high:.1f}'])
    tab = c.table(cellText=cells, rowLabels=labels, colLabels=['Predicted (%)', '95% CI'], cellLoc='center', loc='center', bbox=[0.13, 0.0, 0.92, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#EDF3F6' if i == 0 else 'white')
        if i == 0:
            cell.set_text_props(weight='bold')
    decorate(c, 'c', 'Natural-course predictions')
    log(name, 'c', sources[3], z.reset_index(), 'Model-conditional risk intervals distinct from observed unknown-outcome bounds')
    fields = ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency']
    for j, (axis, radial) in enumerate([(d, 'inner'), (e, 'outer')]):
        z = budget[budget.radial.eq(radial) & budget.field.isin(fields)].set_index('field').loc[fields]
        waterfall(axis, z.estimate, ['Horizontal', 'Vertical', 'Residual', 'Net'])
        axis.tick_params(axis='x', labelrotation=20)
        axis.set_ylabel('Integrated contrast (g kg⁻¹)')
        decorate(axis, chr(100 + j), 'SCS: ' + RINGLABELS[j])
        log(name, chr(100 + j), sources[4], z.reset_index(), 'Return minus complete nonreturn; no cross-annulus pairing; CI/n in table')
    finish(fig, name, exclude=[c])
