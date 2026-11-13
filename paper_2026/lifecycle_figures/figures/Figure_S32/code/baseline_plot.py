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

def fig8():
    src = ROOT / 'inputs/9b2cabfa2d_paired_source_and_design.csv'
    pairs = load(src)
    k = 'X3_mid_rh_pct'
    pairs[k + '_value_pre'] = pairs.value_before
    pairs[k + '_value_post'] = pairs.value_after
    mask = pairs.process_pair_valid.astype(str).str.lower().eq('true') & pairs.value_before.notna() & pairs.value_after.notna()
    x = pairs.loc[mask].copy()
    x['difference'] = x.difference_after_minus_before
    d = load(ROOT / 'inputs/05d7e9174b_P5_figure_rows.csv')
    table('Fig8', 'a_b', x, 'raw_valid_intensity_pairs')
    table('Fig8', 'c_d', d, 'original_model_estimates')
    raw = d[d.process.eq('midRH') & d.role.eq('raw')].iloc[0]
    assert x.sid.nunique() == raw.n_sid and len(x) == raw.n_pairs
    assert abs(x.groupby('sid').difference.mean().mean() - raw.estimate) < 1e-12
    fig = plt.figure(figsize=(180 / 25.4, 181 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.02], width_ratios=[1, 1])
    a, b = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
    lower = gs[1, :].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.15)
    c, e = (fig.add_subplot(lower[0]), fig.add_subplot(lower[1]))
    fig.subplots_adjust(left=0.245, right=0.97, bottom=0.08, top=0.88, hspace=0.59, wspace=0.4)
    for ax, letter in zip([a, b, c, e], 'abcd'):
        ax.set_gid(letter)
        ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-15, 18 if letter in 'ab' else 29), textcoords='offset points', fontsize=10, weight='bold', va='bottom')
    ha = a.hexbin(x[k + '_value_pre'], x[k + '_value_post'], gridsize=29, mincnt=1, cmap='Blues', bins='log', linewidths=0)
    a.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls='--')
    a.set(xlim=(0, 100), ylim=(0, 100), xlabel='Pre-peak RH (%)', ylabel='Post-peak RH (%)')
    a.set_title('Matched intensity', loc='left', pad=18)
    h = b.hexbin(x.A_signed, x.difference, gridsize=29, mincnt=1, cmap='Blues', bins='log', linewidths=0)
    b.axhline(0, color=MUTED, lw=0.8, ls='--')
    b.set(xlim=(-1, 1), xlabel='Duration asymmetry A', ylabel='Post − pre RH (pp)')
    b.set_title('Observed RH differences', loc='left', pad=18)
    vmax = max(float(ha.get_array().max()), float(h.get_array().max()))
    ha.set_clim(1, vmax)
    h.set_clim(1, vmax)
    cax = fig.add_axes([0.38, 0.946, 0.5, 0.01])
    cb = fig.colorbar(h, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.tick_top()
    cb.set_label('')
    cb.set_ticks([1, 10])
    cb.set_ticklabels(['1', '10'])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=8)
    fig.text(0.245, 0.95, 'Pairs / cell', fontsize=8)
    proc = ['midRH', 'q700_storm_relative_horizontal_advection', 'q850_moisture_flux_convergence', 'shear200_500', 'PI_headroom']
    labs = ['Mid-level RH (pp)', '700 hPa horizontal\nadvection (g kg⁻¹ day⁻¹)', '850 hPa convergence\n(g kg⁻¹ day⁻¹)', '200–500 hPa shear\n(m s⁻¹)', 'PI − V (m s⁻¹)']
    for ax, ncol, title in [(c, 3, 'Post − pre estimates'), (e, 1, 'Slope per 0.1 A')]:
        ax.set(xlim=(-0.5, ncol - 0.5), ylim=(4.5, -0.5), xticks=range(ncol), yticks=[])
        ax.set_xticklabels(['Raw', 'Screened', 'Adjusted'] if ncol == 3 else ['Slope'])
        ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=7, labelsize=8)
        ax.set_title(title, loc='left', pad=29, fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for j, (p, lab) in enumerate(zip(proc, labs)):
        c.text(-0.07, j, lab, transform=c.get_yaxis_transform(), ha='right', va='center', fontsize=8)
        for ax, role, col in [(c, 'raw', 0), (c, 'screened', 1), (c, 'adjusted', 2), (e, 'slope', 0)]:
            r = d[d.process.eq(p) & d.role.eq(role)].iloc[0]
            v = r.estimate
            face = '#f0f2f4' if not np.isfinite(v) else '#D7E7ED' if r.ci_high < 0 else '#F1DFCE' if r.ci_low > 0 else '#EDF0F2'
            ax.add_patch(Rectangle((col - 0.47, j - 0.43), 0.94, 0.86, facecolor=face, edgecolor='#BAC4CB' if not np.isfinite(v) else 'none', linewidth=0.6))
            s = 'Not estimated' if not np.isfinite(v) else f'{v:.3f}' if p.startswith('q') or role == 'slope' else f'{v:.2f}'
            ax.text(col, j, s.replace('-', '−'), ha='center', va='center', fontsize=7.5 if not np.isfinite(v) else 9, weight='normal' if not np.isfinite(v) else 'bold')
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=z) for z in ['#D7E7ED', '#EDF0F2', '#F1DFCE']], ['Interval below zero', 'Interval spans zero', 'Interval above zero'], loc='lower center', ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.55, 0.012))
    export(fig, 'Fig8', exclude_axes=[cb.ax], row_groups=[['a', 'b'], ['c', 'd']], exemptions=[{'checks': ['panel-width'], 'panels': ['c', 'd'], 'reason': 'Three comparison columns versus one slope column.'}])
    d.to_csv(R / 'fig8_complete_estimates.csv', index=False)
    (R / 'fig8_source_validation.json').write_text(json.dumps({'source': str(src), 'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'n_sid': x.sid.nunique(), 'n_pairs': len(x), 'equal_sid_mean': x.groupby('sid').difference.mean().mean(), 'frozen_mean': raw.estimate, 'no_model_refit': True, 'scatter_unit': 'cyclone-intensity-bin pair', 'descriptive_and_model_supports_distinct': True}, indent=2), encoding='utf-8')
