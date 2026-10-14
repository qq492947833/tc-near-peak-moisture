from .common import *
O=DATA/'geography'
REGNAME={**{k:k for k in ['NA','EP','WP','NI','SI','SP']},'SCS_box':'SCS box','Japan_Sea_box':'Japan Sea box'}
REGIONS=['NA','EP','WP','NI','SI','SP','SCS_box','Japan_Sea_box']
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'DejaVu Sans'], 'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
BASINS = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']
COLORS = {'known_return': '#246F9B', 'complete_nonreturn': '#A27B53', 'unknown': '#88939D'}
GROUPS = {'known_return': 'Return', 'complete_nonreturn': 'Complete nonreturn', 'unknown': 'Unknown'}
sources = ['G2_first_confirmed_return.csv', 'G1_genesis_basin_recurrence.csv', 'general_reintensification_genesis_basin_rates.csv', 'G3_region_followup_counts.csv', 'G3_region_risk_estimates.csv', 'environment_model_paired_comparisons.csv', 'post_process_curves.csv', 'post_process_return_contrasts.csv', 'general_reintensification_main_episode_sources.csv', 'G3_departure_risk_SID_source.csv', 'geographic_grid_counts_5_and_10deg.csv', 'G3_region_policy_estimates.csv', 'G3_region_policy_support.csv', 'pre_environment_group_contrasts.csv', 'post_process_group_estimates.csv']

def read(name):
    return pd.read_csv(O / name, keep_default_na=False, na_values=['', 'nan', 'NaN'])

def grid(height=210):
    fig, axs = plt.subplots(3, 2, figsize=(180 / 25.4, height / 25.4))
    fig.subplots_adjust(left=0.17, right=0.975, bottom=0.075, top=0.94, wspace=0.48, hspace=0.8)
    return (fig, axs)

def decorate(ax, letter, title):
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-38, 10), textcoords='offset points', ha='left', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(title, loc='left', pad=12)

def coast(ax, boxes=True):
    for s in shapefile.Reader(str(DATA/'coastline/ne_110m_coastline.shp')).shapes():
        points = np.asarray(s.points)
        cuts = list(s.parts) + [len(points)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            xx = points[a:b, 0] % 360
            yy = points[a:b, 1]
            seams = np.where(abs(np.diff(xx)) > 180)[0] + 1
            for ix in np.split(np.arange(len(xx)), seams):
                if len(ix) > 1:
                    clipped = LineString(np.column_stack([xx[ix], yy[ix]])).intersection(box(0, -60, 360, 65))
                    parts = list(clipped.geoms) if hasattr(clipped, 'geoms') else [clipped]
                    for part in parts:
                        if part.geom_type == 'LineString' and (not part.is_empty):
                            coords = np.asarray(part.coords)
                            ax.plot(coords[:, 0], coords[:, 1], color='#ABB1B5', lw=0.35, zorder=0)
    ax.set_xlim(0, 360)
    ax.set_ylim(-60, 65)
    ax.set_xticks([0, 90, 180, 270, 360], ['0°', '90°E', '180°', '90°W', '0°'])
    ax.set_yticks([-60, -30, 0, 30, 60], ['60°S', '30°S', '0°', '30°N', '60°N'])
    ax.tick_params(length=2)
    if boxes:
        for x, y, w, h in [(105, 5, 15, 20), (130, 35, 10, 10)]:
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='#997E4A', lw=0.7))

def forest_point(ax, point, low, high, y, color, marker='o', ms=3.5):
    if np.isfinite(point):
        if np.isfinite(low + high):
            ax.plot([low, high], [y, y], color=color, lw=0.9)
            ax.plot([low, high], [y, y], '|', color=color, ms=3.6, mew=0.8)
        ax.plot(point, y, marker, color=color, ms=ms)

def basin_rates(ax, full=False):
    d = read('G1_genesis_basin_recurrence.csv')
    d = d[d.config.eq('main') & d.sensitivity.eq('all')]
    general = read('general_reintensification_genesis_basin_rates.csv')
    general = general[general.scope.eq('common2366')]
    for j, basin in enumerate(BASINS):
        a = d[d.region.eq(basin) & d.episode_category.eq('any_recurrence')].iloc[0]
        if full:
            groups = [('exactly_two', '#246F9B', 'o'), ('three_or_more', '#C18A49', 's')]
            for k, (kind, col, mk) in enumerate(groups):
                z = d[d.region.eq(basin) & d.episode_category.eq(kind)].iloc[0]
                forest_point(ax, 100 * z.fraction, 100 * z.ci_low, 100 * z.ci_high, 5 - j + (k - 0.5) * 0.24, col, mk)
        else:
            z = general[general.region.eq(basin)].iloc[0]
            forest_point(ax, 100 * a.fraction, 100 * a.ci_low, 100 * a.ci_high, 5 - j + 0.12, '#246F9B')
            forest_point(ax, 100 * z.fraction, 100 * z.ci_low, 100 * z.ci_high, 5 - j - 0.12, '#A27B53', 's')
    ax.set_yticks(range(6), BASINS[::-1])
    ax.set_ylim(-0.6, 6.8)
    ax.set_xlabel('Fraction of genesis-basin SID (%)')
    ax.set_xlim(left=0)
    labels = ['Exactly two near-peak segments', 'Three or more'] if full else ['Near-peak recurrence', 'Post-LMI reintensification']
    ax.legend([Line2D([0], [0], marker='o', color='#246F9B', lw=0), Line2D([0], [0], marker='s', color='#C18A49' if full else '#A27B53', lw=0)], labels, loc='upper right', frameon=False, fontsize=7)

def regional_risk(ax):
    e = read('G3_region_risk_estimates.csv')
    e = e[e.branch.eq('shear200_850') & e.method.eq('natural_gformula')]
    n = read('G3_region_followup_counts.csv')
    for j, region in enumerate(REGIONS):
        y = len(REGIONS) - 1 - j
        z = e[e.region.eq(region)].iloc[0]
        c = n[n.region.eq(region)].iloc[0]
        if c.n_risk:
            ax.plot([c.lower_risk_bound * 100, c.upper_risk_bound * 100], [y, y], color='#CDD2D6', lw=4, zorder=0, solid_capstyle='butt')
            forest_point(ax, z.risk_percent, z.ci_low, z.ci_high, y, '#246F9B')
        else:
            ax.text(2, y, 'No eligible departures', va='center', fontsize=7, color='#747D85')
    ax.set_yticks(np.arange(len(REGIONS)), [REGNAME[x] for x in REGIONS[::-1]])
    ax.set_ylim(-0.6, len(REGIONS) - 0.4)
    ax.set_xlim(0, 80)
    ax.set_xlabel('Recovery within 48 h (%)')

def model_comparison(ax, metric):
    d = read('environment_model_paired_comparisons.csv')
    d = d[d.metric.eq(metric)]
    names = ['M0+dynamics', 'M0+moisture', 'M0+thermodynamics', 'M0+all']
    labels = ['+ Dynamics', '+ Moisture', '+ Thermodynamics', '+ All families']
    for j, name in enumerate(names):
        z = d[d.model.eq(name)].iloc[0]
        forest_point(ax, z.paired_difference, z.ci_low, z.ci_high, 3 - j, '#246F9B')
    ax.axvline(0, ls='--', lw=0.7, color='#8A9298')
    ax.set_yticks(range(4), labels[::-1])
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel('Held-out log-likelihood gain per SID' if metric == 'interval_log_likelihood' else 'IPCW Brier change from M0')

def qcurve(ax, radial, field='q700'):
    d = read('post_process_curves.csv')
    d = d[d.radial.eq(radial) & d.field.eq(field)]
    for group in GROUPS:
        g = d[d.outcome_group.eq(group)].sort_values('lag_h')
        x = g.lag_h.to_numpy(float)
        y = g.estimate.to_numpy(float)
        lo = g.ci_low.to_numpy(float)
        hi = g.ci_high.to_numpy(float)
        ax.plot(x, y, 'o-' if group != 'unknown' else 's--', color=COLORS[group], lw=1.1, ms=2.6, label=GROUPS[group])
        ax.fill_between(x, lo, hi, color=COLORS[group], alpha=0.12, lw=0)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel('Hours after departure')
    ax.set_ylabel('700 hPa specific humidity (g/kg)' if field == 'q700' else '850 hPa MFC (g/kg/day)')
    ax.set_ylim(top=d.ci_high.max() + 0.85)
    ax.legend(frameon=False, fontsize=7, loc='upper left')

def budget_contrast(ax, allterms=False):
    d = read('post_process_return_contrasts.csv')
    fields = ['horizontal_advection', 'vertical_advection', 'moving_tendency'] if not allterms else ['fixed_tendency', 'motion_term', 'horizontal_advection', 'vertical_advection', 'moving_tendency', 'residual']
    labs = {'fixed_tendency': 'Local tendency', 'motion_term': 'Motion term', 'horizontal_advection': 'Horizontal advection', 'vertical_advection': 'Vertical advection', 'moving_tendency': 'Moving tendency', 'residual': 'Residual'}
    for j, field in enumerate(fields):
        for k, (radial, col, mk) in enumerate([('inner', '#246F9B', 'o'), ('outer', '#A27B53', 's')]):
            z = d[d.field.eq(field) & d.radial.eq(radial)].iloc[0]
            forest_point(ax, z.estimate, z.ci_low, z.ci_high, len(fields) - 1 - j + (k - 0.5) * 0.22, col, mk)
    ax.axvline(0, ls='--', lw=0.7, color='#8A9298')
    ax.set_yticks(range(len(fields)), [labs[x] for x in fields[::-1]])
    ax.set_ylim(-0.55, len(fields) - 0.45)
    ax.set_xlabel('0–24 h integrated contrast (g/kg)')
    ax.legend([Line2D([0], [0], marker='o', color='#246F9B', lw=0), Line2D([0], [0], marker='s', color='#A27B53', lw=0)], ['100–300 km', '300–800 km'], frameon=False, fontsize=7, loc='best')

def supplementary_regions():
    fig, a = grid(210)
    basin_rates(a[0, 0], True)
    decorate(a[0, 0], 'a', 'Two-segment and multi-segment recurrence')
    d = read('general_reintensification_main_episode_sources.csv')
    coast(a[0, 1])
    assert set(d.status_class).issubset({'tropical_confirmed', 'ET_or_other_or_mixed', 'unknown'})
    for state, col, mk, label in [('tropical_confirmed', '#246F9B', 'o', 'Tropical'), ('ET_or_other_or_mixed', '#C18A49', '^', 'Other/mixed'), ('unknown', '#88939D', 'x', 'Unknown')]:
        z = d[d.status_class.eq(state)]
        a[0, 1].scatter(z.start_lon % 360, z.start_lat, s=7, c=col, marker=mk, alpha=0.7, label=label)
    decorate(a[0, 1], 'b', 'General reintensification locations')
    a[0, 1].legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0, -0.22), ncol=3, columnspacing=0.8, handletextpad=0.3, borderaxespad=0)
    g = read('geographic_grid_counts_5_and_10deg.csv')
    g = g[g.quantity.eq('G3_risk') & g.grid_degrees.eq(5)]
    coast(a[1, 0])
    a[1, 0].scatter((g.lon_left + 2.5) % 360, g.lat_bottom + 2.5, s=4 + g.n_sid * 1.4, c='#6D8B9E', alpha=0.7, edgecolors='none')
    decorate(a[1, 0], 'c', 'Departure-origin counts on a 5° lattice')
    risk = read('G3_departure_risk_SID_source.csv')
    risk = risk[risk.region.eq('SCS_box')]
    x = np.arange(1, 13)
    bottom = np.zeros(12)
    for group in ['known_return', 'complete_nonreturn', 'unknown']:
        mask = risk.recovery48.eq(1) if group == 'known_return' else risk.recovery48.eq(0) if group == 'complete_nonreturn' else risk.recovery48.isna()
        counts = risk[mask].groupby('month').size().reindex(x, fill_value=0).to_numpy()
        a[1, 1].bar(x, counts, bottom=bottom, color=COLORS[group], width=0.75, label=GROUPS[group])
        bottom += counts
    a[1, 1].set_xticks([1, 3, 5, 7, 9, 11])
    a[1, 1].set_xlabel('Departure month')
    a[1, 1].set_ylabel('SCS risk SID')
    a[1, 1].legend(frameon=False, fontsize=7, loc='upper left')
    decorate(a[1, 1], 'd', 'Season and follow-up in the SCS box')
    effects = read('G3_region_policy_estimates.csv')
    effects = effects[effects.method.eq('onestep') & effects.outcome.eq('recovery48_pp')]
    regs = BASINS + ['SCS_box']
    for j, region in enumerate(regs):
        for k, (policy, col, mk) in enumerate([('shear200_850', '#246F9B', 'o'), ('midRH', '#C18A49', 's')]):
            z = effects[effects.region.eq(region) & effects.policy.eq(policy)].iloc[0]
            forest_point(a[2, 0], z.estimate, z.ci_low, z.ci_high, len(regs) - 1 - j + (k - 0.5) * 0.24, col, mk)
    a[2, 0].set_yticks(range(len(regs)), [REGNAME[x] for x in regs[::-1]])
    a[2, 0].set_xlabel('48 h one-step policy contrast (pp)')
    a[2, 0].axvline(0, color='#8B9298', ls='--', lw=0.7)
    decorate(a[2, 0], 'e', 'Regional supported-policy contrasts')
    support = read('G3_region_policy_support.csv')
    support = support[pd.to_numeric(support.outcome).eq(48)]
    for k, (policy, col) in enumerate([('shear200_850', '#246F9B'), ('midRH', '#C18A49')]):
        z = support[support.policy.eq(policy)].set_index('region').loc[regs]
        a[2, 1].barh(np.arange(len(regs))[::-1] + (k - 0.5) * 0.28, 100 * z.coverage.to_numpy(), height=0.24, color=col, label='Shear −2 m/s' if policy.startswith('shear') else 'RH +5 pp')
    a[2, 1].set_yticks(np.arange(len(regs))[::-1], [REGNAME[x] for x in regs])
    a[2, 1].set_xlim(0, 100)
    a[2, 1].set_ylim(-0.6, 8)
    a[2, 1].set_xlabel('Exposure changed by policy (%)')
    a[2, 1].legend(frameon=False, fontsize=7, loc='upper right')
    decorate(a[2, 1], 'f', 'Within-region policy coverage')
    save(fig, 'SF-R')

def supplementary_process():
    fig, a = grid(210)
    d = read('pre_environment_group_contrasts.csv')
    d = d[d.region.eq('SCS_box') & d.comparison.eq('known_return minus complete_nonreturn')]
    groups = [(['shear200_850', 'shear500_850', 'shear200_500', 'cyclonic_vo850'], ['Deep shear', 'Lower shear', 'Upper shear', 'Cyclonic vorticity']), (['midRH', 'q700_inner', 'q700_outer', 'SST', 'PI_headroom'], ['Midlevel RH', 'Inner q700', 'Outer q700', 'SST', 'PI headroom'])]
    for col, (fields, labels) in enumerate(groups):
        for j, field in enumerate(fields):
            z = d[d.variable.eq(field)].iloc[0]
            forest_point(a[0, col], z.standardized_difference, z.standardized_ci_low, z.standardized_ci_high, len(fields) - 1 - j, '#246F9B')
        a[0, col].set_yticks(range(len(fields)), labels[::-1])
        a[0, col].set_xlabel('Pre-event difference / reference SD')
        a[0, col].axvline(0, ls='--', lw=0.7, color='#8B9298')
        decorate(a[0, col], chr(97 + col), 'SCS: preceding dynamics' if col == 0 else 'SCS: preceding moisture\nand thermodynamics')
    climate = read('C_q700_SCS_return_contrasts.csv')
    for col, radial in enumerate(['100_300km', '300_800km']):
        for j, (field, label) in enumerate([('q700_raw_common', 'Observed q'), ('q700_background', 'Seasonal background'), ('q700_anomaly', 'Anomaly')]):
            z = climate[climate.radial.eq(radial) & climate.component.eq(field)].iloc[0]
            forest_point(a[1, col], z.estimate, z.ci_low, z.ci_high, 2 - j, '#246F9B')
        a[1, col].set_yticks([2, 1, 0], ['Observed q', 'Seasonal background', 'Anomaly'])
        a[1, col].set_ylim(-0.6, 2.6)
        a[1, col].axvline(0, ls='--', lw=0.7, color='#8B9298')
        a[1, col].set_xlabel('Pre-event return–nonreturn contrast (g/kg)')
        decorate(a[1, col], chr(99 + col), 'Inner moisture components' if col == 0 else 'Outer moisture components')
    budget_contrast(a[2, 0], True)
    decorate(a[2, 0], 'e', '700 hPa tendency decomposition')
    z = read('post_process_return_contrasts.csv')
    z = z[z.field.eq('q850_absolute_MFC')]
    for j, radial in enumerate(['inner', 'outer']):
        r = z[z.radial.eq(radial)].iloc[0]
        forest_point(a[2, 1], r.estimate, r.ci_low, r.ci_high, 1 - j, '#246F9B' if radial == 'inner' else '#A27B53', 'o' if radial == 'inner' else 's')
    a[2, 1].set_yticks([1, 0], ['100–300 km', '300–800 km'])
    a[2, 1].set_ylim(-0.6, 1.6)
    a[2, 1].axvline(0, color='#8B9298', ls='--', lw=0.7)
    a[2, 1].set_xlabel('0–24 h MFC contrast (g/kg)')
    decorate(a[2, 1], 'f', '850 hPa moisture-flux convergence')
    save(fig, 'SF-S')
def save(fig,name):export(fig,{'SF-R':'FigS23','SF-S':'FigS24'}[name])
def render():
 supplementary_regions();supplementary_process()

