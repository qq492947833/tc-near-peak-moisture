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

def fig15():
    name = 'FigS15'
    ms, main = legacy('FP0266')
    ss, state = legacy('FP0262')
    fig = plt.figure(figsize=(180 / 25.4, 140 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_axes([0.28,0.02,0.58,0.07])]
    fig.subplots_adjust(left=0.16, right=0.98, top=0.9, bottom=0.13, hspace=0.42, wspace=0.48)
    a, b, c, d, legend = axes
    main = main.set_index('variable').loc[VAR].reset_index()
    e = main.coef.to_numpy()
    a.errorbar(e, np.arange(6), xerr=[e - main.bootstrap_ci_low, main.bootstrap_ci_high - e], fmt='o', color=INK, capsize=2, ms=4)
    a.axvline(0, color=GREY, ls=':', lw=0.7)
    a.set(yticks=range(6), yticklabels=VL, ylim=(5.6, -0.6), xlabel='Slope difference (kt per model SD)')
    decorate(a, 'a', 'Resolution minus bridge')
    log(name, 'a,b', ms, main, 'Primary six-test H2 family; original P/q distinct from percentile CI')
    b.axis('off')
    decorate(b, 'b', 'Primary inference')
    b.set(xlim=(-.5,2.5),ylim=(5.6,-.6))
    pq_cmap=LinearSegmentedColormap.from_list('P_q_probability',['#9A85B8','#F3EFF8'])
    for j,label in enumerate(['n','P','q']):
        b.text(j,-.88,label,ha='center',va='center',fontsize=8,weight='bold')
    for i,r in enumerate(main.itertuples()):
        for j,value in enumerate([r.n_sid,r.p,r.q]):
            face='#F0F2F3' if j==0 else pq_cmap(float(value))
            b.add_patch(Rectangle((j-.48,i-.45),.96,.9,facecolor=face,edgecolor='#B8B1C5',lw=.6))
            text=str(int(value)) if j==0 else format_number(value)
            b.text(j,i,text,ha='center',va='center',fontsize=8,color=INK)
    inference_cb=fig.add_axes([.675,.565,.27,.012]);inference_cb.set_gid('b_colourbar')
    mapper=mpl.cm.ScalarMappable(norm=Normalize(0,1),cmap=pq_cmap)
    cbar=fig.colorbar(mapper,cax=inference_cb,orientation='horizontal',ticks=[0,.5,1])
    cbar.ax.text(-.07,.5,'P / q',transform=cbar.ax.transAxes,fontsize=7,ha='right',va='center');cbar.ax.tick_params(labelsize=7,length=2)
    for ax, sel, letter, title in [(c, 'observed_6h_endpoints_tropical', 'c', 'Tropical endpoints'), (d, 'observed_6h_endpoints_sea_and_tropical', 'd', 'Sea + tropical')]:
        for k, condition in enumerate(['unfiltered', sel]):
            z = state[state.selection.eq(condition)].set_index('variable').loc[VAR].reset_index()
            e = z.coef.to_numpy()
            ax.errorbar(e, np.arange(6) + (k - 0.5) * 0.22, xerr=[e - z.ci_low, z.ci_high - e], fmt=['o', 's'][k], color=[GREY, BLUE][k], capsize=2, ms=3)
            log(name, letter, ss, z, 'Descriptive endpoint eligibility sensitivity; no added P/q family')
        ax.axvline(0, color=GREY, ls=':', lw=0.7)
        ax.set(yticks=range(6), yticklabels=VL if ax is c else [], ylim=(5.6, -0.6), xlim=(-2.3, 2.3), xticks=[-2, 0, 2], xlabel='kt per fixed SD')
        decorate(ax, letter, title)
    legend.axis('off')
    legend.legend([Line2D([0], [0], color=GREY, marker='o'), Line2D([0], [0], color=BLUE, marker='s')], ['Unfiltered', 'Endpoint rule'], loc='center', frameon=False, ncol=2)
    legend.set_position([0.28, 0.02, 0.58, 0.07])
    finish(fig, name, exclude=[b, legend,inference_cb])
