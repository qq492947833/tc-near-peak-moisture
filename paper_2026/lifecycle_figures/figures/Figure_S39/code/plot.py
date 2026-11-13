from display_format import format_number,format_group
MASTER_SHA = '0b4c4308543fb4e7f5f32911d6b654a5fc4e720b8617fc53ce46a59b2c117b37'
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
TEMP = BLUE = '#167B9E'
FINAL = ORANGE = '#CC7445'
INK = '#233A46'
MUTED = '#677B86'
GREY = '#8A969D'
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
                text = format_number(values[i, j])
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

def coastline(ax):
    import shapefile
    for shape in shapefile.Reader(str(DATA / 'coastline/ne_110m_coastline.shp')).shapes():
        pts = np.asarray(shape.points)
        cuts = list(shape.parts) + [len(pts)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            p = pts[a:b]
            xx = p[:, 0] % 360
            yy = p[:, 1]
            yy = np.where((yy >= -60) & (yy <= 65), yy, np.nan)
            for ix in np.split(np.arange(len(p)), np.flatnonzero(abs(np.diff(xx)) > 180) + 1):
                if len(ix) > 1:
                    ax.plot(xx[ix], yy[ix], color='#C5CDD2', lw=0.4, zorder=0)
    ax.set(xlim=(0, 360), ylim=(-60, 65), xticks=[0, 90, 180, 270, 360], xticklabels=['0°', '90°E', '180°', '90°W', '0°'], yticks=[-60, 0, 60], yticklabels=['60°S', '0°', '60°N'])
    for x, y, w, h in [(105, 5, 15, 20), (130, 35, 10, 10)]:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='#9E8552', lw=0.8))

def fig23():
    name = 'FigS23'
    sources = ['geography/G1_genesis_basin_recurrence.csv', 'geography/general_reintensification_main_episode_sources.csv', 'geography/geographic_grid_counts_5_and_10deg.csv', 'geography/G3_departure_risk_SID_source.csv', 'geography/G3_region_policy_estimates.csv', 'geography/G3_region_policy_support.csv']
    near, events, geo, risk, effects, support = map(read, sources)
    fig, ax = grid(3,2,185,left=.13,hspace=.48,wspace=.5)
    d = near[near.config.eq('main') & near.sensitivity.eq('all') & near.region.isin(BASINS) & near.episode_category.isin(['exactly_two', 'three_or_more'])]
    bottom = np.zeros(6)
    for k, (category, col, label) in enumerate([('exactly_two', BLUE, 'Exactly two'), ('three_or_more', ORANGE, 'Three or more')]):
        z = d[d.episode_category.eq(category)].set_index('region').loc[BASINS]
        ax[0, 0].bar(range(6), z.fraction * 100, bottom=bottom, color=col, width=0.65, label=label)
        bottom += z.fraction.to_numpy() * 100
    ax[0, 0].set(xticks=range(6), xticklabels=BASINS, ylabel='Genesis-basin fraction (%)')
    ax[0,0].legend(*ax[0,0].get_legend_handles_labels(),frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.26),fontsize=7)
    decorate(ax[0, 0], 'a', 'Number of near-peak segments')
    log(name, 'a', sources[0], d, 'Original category fractions; each category CI/n retained in table')
    states = ['tropical_confirmed', 'ET_or_other_or_mixed', 'unknown']
    event_counts = events.groupby('status_class', dropna=False).size().reindex(states, fill_value=0)
    ax[0, 1].bar(range(3), event_counts, color=['#64866A','#AE8AAB','#CBD1D5'], width=0.55)
    for i, n in enumerate(event_counts):
        ax[0, 1].text(i, n + 12, str(n), ha='center', fontsize=8)
    ax[0, 1].set(xticks=range(3), xticklabels=['Tropical', 'Other /\nmixed', 'Unknown'], ylabel='Reintensification events', ylim=(0, max(event_counts) * 1.16))
    decorate(ax[0, 1], 'b', 'Post-LMI onset status')
    log(name, 'b', sources[1], events, 'All actual events retained; locations remain source data, not duplicated from main Fig. 1')
    z = geo[geo.quantity.eq('G3_risk') & geo.grid_degrees.eq(5)]
    coastline(ax[1, 0])
    ax[1, 0].scatter((z.lon_left + 2.5) % 360, z.lat_bottom + 2.5, s=3 + 1.4 * z.n_sid, c='#617D93', alpha=0.65, edgecolors='none')
    decorate(ax[1, 0], 'c', 'First-departure origins')
    log(name, 'c', sources[2], z, 'Saved 5-degree lattice; marker area linear above a fixed minimum')
    z = risk[risk.region.eq('SCS_box')]
    x = np.arange(1, 13)
    bottom = np.zeros(12)
    derived = []
    for group, col, label in [('return','#875C9B','Return'),('nonreturn','#B39B4D','Complete nonreturn'),('unknown','#CDD0D3','Unknown')]:
        mask = z.recovery48.eq(1) if group == 'return' else z.recovery48.eq(0) if group == 'nonreturn' else z.recovery48.isna()
        cnt = z[mask].groupby('month').size().reindex(x, fill_value=0).to_numpy()
        ax[1, 1].bar(x, cnt, bottom=bottom, color=col, width=0.75, label=label)
        bottom += cnt
        derived.extend(({'month': m, 'outcome_group': group, 'count': n} for m, n in zip(x, cnt)))
    ax[1, 1].set(xticks=[1, 3, 5, 7, 9, 11], xlabel='Departure month', ylabel='SCS departure cyclones')
    ax[1, 1].legend(frameon=False, fontsize=7)
    decorate(ax[1, 1], 'd', 'Season and incomplete follow-up')
    log(name, 'd', sources[3], pd.DataFrame(derived), 'Exact month/outcome counts from all 94 SCS departures')
    regs = BASINS + ['SCS_box']
    labs = BASINS + ['SCS']
    policies = ['shear200_850', 'midRH']
    z = effects[effects.method.eq('onestep') & effects.outcome.eq('recovery48_pp') & effects.region.isin(regs)]
    mat = z.pivot(index='region', columns='policy', values='estimate').loc[regs, policies]
    effect_image=heat(ax[2,0],mat,['Shear −2','RH +5'],labs,fmt='.1f',cmap='PuOr')
    ax[2, 0].set_xlabel('One-step recovery contrast (pp)')
    decorate(ax[2, 0], 'e', 'Supported-policy contrasts')
    log(name, 'e', sources[4], z, 'Original regional estimates; complete CI/n and conditional inference in table')
    z = support[pd.to_numeric(support.outcome).eq(48) & support.region.isin(regs)]
    for k, policy in enumerate(policies):
        q = z[z.policy.eq(policy)].set_index('region').loc[regs]
        ax[2, 1].barh(np.arange(7) + (k - 0.5) * 0.32, q.coverage * 100, height=0.28, color=['#375D91','#B96186'][k])
    ax[2, 1].set(yticks=range(7), yticklabels=labs, ylim=(6.6, -0.6), xlim=(0, 100), xlabel='Exposure changed (%)')
    decorate(ax[2, 1], 'f', 'Within-region policy coverage')
    log(name, 'f', sources[5], z)
    ax[2,1].legend([Patch(color='#375D91'),Patch(color='#B96186')],['Shear −2 m s⁻¹','RH +5 pp'],frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.27),fontsize=7,columnspacing=.9,handlelength=1.2)
    effect_cb=fig.add_axes([.16,.043,.28,.012]);effect_cb.set_gid('e_colourbar')
    cb=fig.colorbar(effect_image,cax=effect_cb,orientation='horizontal');cb.ax.tick_params(labelsize=7,length=2)
    finish(fig,name,exclude=[effect_cb])
