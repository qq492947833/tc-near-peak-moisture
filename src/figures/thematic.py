from .common import *
D=DATA/'thematic'
LEDGER=[]
RC={'100_300km':'#247499','300_800km':'#BB7749'}
RL={'100_300km':'100–300 km','300_800km':'300–800 km'}
def save(fig,name,**kwargs):
 export(fig,{'Fig2':'FigS3','FigS7':'FigS8','Fig3':'FigS10','Fig6':'Fig4','FigS14':'FigS16','Fig7':'Fig7','Fig8':'FigS22','FigS19':'FigS25'}[name],**kwargs)
LEDGER = []
META = []

def read(name):
    return pd.read_csv(D / name, keep_default_na=False, na_values=['', 'nan', 'NaN'])

def record(name, panel, source, data):
    LEDGER.extend(data.assign(new_figure=name, new_panel=panel, local_source=source).to_dict('records'))

def style():
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 8.3, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'savefig.dpi': 601})

def decorate(ax, label, title, offset=-25):
    ax.set_gid(label)
    ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(offset, 10), textcoords='offset points', fontsize=10, fontweight='bold', ha='left', va='bottom')
    ax.set_title(title, loc='left', pad=12)

def interval(ax, e, lo, hi, y, color, marker='o', ms=3.5, open_marker=False):
    if np.isfinite(e):
        if np.isfinite(lo + hi):
            ax.plot([lo, hi], [y, y], color=color, lw=0.95)
            ax.plot([lo, hi], [y, y], '|', color=color, ms=3.8, mew=0.8)
        ax.plot(e, y, marker, color=color, mfc='white' if open_marker else color, ms=ms)

def get_functions(filename, names):
    tree = ast.parse((D / filename).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert set((n.name for n in nodes)) == set(names)
    return ast.unparse(ast.Module(body=nodes, type_ignores=[]))
VAR = ['X1_sst_c', 'X2_shear_ms', 'X3_mid_rh_pct', 'X4_d200_1e5s', 'X5_minus_omega500_pa_s', 'X6_signed_vo850_1e5s']
VL = ['SST', 'Shear', 'Mid-RH', 'Div. 200', 'Ascent 500', 'Vort. 850']

def duration_panel(ax, panel, scope):
    source = 'fig2_duration.csv'
    d = read(source)
    metrics = ['R', 'R_long', 'U', 'F', 'L', 'B']
    z = d[d.scope.eq(scope) & d.statistic.eq('mean') & d.metric.isin(metrics)].set_index('metric').loc[metrics].reset_index()
    for j, r in z.iterrows():
        interval(ax, r.estimate, r.ci_low, r.ci_high, j, '#333333')
        ax.text(r.ci_high + 2, j, f'{r.estimate:.1f}', va='center', fontsize=8)
    ax.set_yticks(range(6), ['R', 'Longest', 'U', 'F', 'Later L', 'Bridge B'])
    ax.set_ylim(5.6, -0.6)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Mean covered duration (h)')
    decorate(ax, panel, 'All common-window cyclones (n = 2366)' if scope == 'main_all' else 'Recurrent subset (n = 123)')
    record('Fig2', panel, source, z)

def window_panel(ax, name, panel, scope):
    d = read('fig2_window.csv')
    colors = ['#7867a7', '#333333']
    for k, contrast in enumerate(['F_minus_U', 'U_minus_R']):
        z = d[d.scope.eq(scope) & d.contrast.eq(contrast) & d.variable.isin(VAR)].set_index('variable').loc[VAR].reset_index()
        for j, r in z.iterrows():
            interval(ax, r.estimate_sd, r.ci_low_sd, r.ci_high_sd, j + (k - 0.5) * 0.23, colors[k], ['o', 's'][k])
        record(name, panel, 'fig2_window.csv', z)
    ax.axvline(0, color='.7', lw=0.65)
    ax.set_yticks(range(6), VL)
    ax.set_ylim(5.6, -0.9)
    ax.set_xlim((-0.035, 0.025) if scope == 'main_all' else (-0.6, 0.4))
    ax.set_xticks([-0.03, 0, 0.02] if scope == 'main_all' else [-0.5, 0, 0.3])
    ax.set_xlabel('Paired window-mean difference / fixed reference SD')
    decorate(ax, panel, 'Time-inclusion shifts: recurrent subset' if scope == 'main_multi' else 'Time-inclusion shifts: all common-window cyclones')
    ax.text(0.98, 0.97, 'n = 120–123' if scope == 'main_multi' else 'n = 2263–2366', transform=ax.transAxes, ha='right', va='top', fontsize=8)
    return [Line2D([0], [0], marker=['o', 's'][k], color=colors[k], lw=0.7, label=l) for k, l in enumerate(['Mean environment over F − mean environment over U', 'Mean environment over U − mean environment over R'])]

def fig2():
    style()
    fig = plt.figure(figsize=(180 / 25.4, 155 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    fig.subplots_adjust(left=0.17, right=0.965, bottom=0.19, top=0.925, wspace=0.45, hspace=0.65)
    duration_panel(axes[0], 'a', 'main_all')
    duration_panel(axes[1], 'b', 'main_multi')
    h = window_panel(axes[2], 'Fig2', 'c', 'main_multi')
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.55, 0.025), ncol=1, frameon=False, fontsize=7.5)
    save(fig, 'Fig2')
    fig, ax = plt.subplots(2, 1, figsize=(180 / 25.4, 150 / 25.4))
    fig.subplots_adjust(left=0.19, right=0.965, bottom=0.16, top=0.91, hspace=0.87)
    h = window_panel(ax[0], 'FigS7', 'a', 'main_all')
    fig.legend(handles=h, loc='center', bbox_to_anchor=(0.56, 0.56), frameon=False, fontsize=7.5)
    d = read('fig2_components.csv')
    cols = ['#333333', '#7867a7', '#c58a00']
    labs = ['Mean(R) − mean(L)', 'Mean(R) − mean(B)', 'Mean(L) − mean(B)']
    for k, comparison in enumerate(['R minus U-R', 'R minus F-U', 'U-R minus F-U']):
        z = d[d.comparison.eq(comparison)].set_index('variable').loc[VAR].reset_index()
        for j, r in z.iterrows():
            interval(ax[1], r.estimate_standardized, r.ci_low_standardized, r.ci_high_standardized, j + (k - 1) * 0.23, cols[k], ['o', 's', '^'][k])
        record('FigS7', 'b', 'fig2_components.csv', z)
    ax[1].axvline(0, color='.7', lw=0.65)
    ax[1].set_yticks(range(6), VL)
    ax[1].set_ylim(5.6, -0.9)
    ax[1].set_xlim(-0.85, 1.25)
    ax[1].set_xticks([-0.5, 0, 0.5, 1])
    ax[1].set_xlabel('Paired component-mean difference / fixed reference SD')
    decorate(ax[1], 'b', 'Disjoint component intervals: recurrent subset')
    ax[1].text(0.98, 0.97, 'n = 120–123', transform=ax[1].transAxes, ha='right', va='top')
    fig.legend(handles=[Line2D([0], [0], marker=['o', 's', '^'][k], color=cols[k], lw=0.7, label=l) for k, l in enumerate(labs)], loc='lower center', bbox_to_anchor=(0.56, 0.02), ncol=3, frameon=False)
    save(fig, 'FigS7')

def fig3():
    src = (D / 'fig3_original.py').read_text(encoding='utf-8')
    a = src.index("V=['X1")
    b = src.index('contract=')
    c = src.index('plt.rcParams.update')
    e = src.index('fig.canvas.draw()')
    code = src[a:b] + src[c:e]
    code = code.replace("O/'raw_stage_summary.csv'", "D/'fig3_raw_stage.csv'").replace("O/'raw_adjacent_contrasts.csv'", "D/'fig3_raw_pairs.csv'").replace("O/'adjusted_adjacent_summary.csv'", "D/'fig3_adjusted.csv'")
    code = code.replace("'Stage mean\\n(whole-lifecycle SD)'", "'(Stage mean − reference mean)\\n/ reference SD'")
    code = code.replace("'All paired cyclones'", "'All eligible pairs (raw)'").replace("'Same geography support'", "'Overlap subset (raw)'").replace("'Geography + season standardized'", "'Same subset, geography/season adjusted'")
    code = code.replace("'Paired change\\n(whole-lifecycle SD)'", "'Paired mean difference\\n/ reference SD'")
    env = dict(globals())
    exec(code, env)
    save(env['fig'], 'Fig3')
    LEDGER.extend((dict(new_figure='Fig3', new_panel=r['panel'], local_source='fig3_raw_stage/fig3_raw_pairs/fig3_adjusted.csv', **r) for r in env['ledger']))
RC = {'100_300km': '#35769B', '300_800km': '#AC674A'}
RL = {'100_300km': '100–300 km', '300_800km': '300–800 km'}

def radial(ax, name, panel, sub, key, cats, labs, title, xlabel):
    for k, region in enumerate(RC):
        z = sub[sub.region.eq(region)].set_index(key).loc[cats].reset_index()
        for j, r in z.iterrows():
            interval(ax, r.estimate, r.lo, r.hi, j + (k - 0.5) * 0.22, RC[region], ['o', 's'][k])
        record(name, panel, 'fig6_regions.csv', z)
    ax.axvline(0, color='.65', ls='--', lw=0.7)
    ax.grid(axis='x', color='#e7eaed', lw=0.5)
    ax.set_yticks(range(len(cats)), labs)
    ax.set_ylim(len(cats) - 0.6, -0.6)
    ax.set_xlabel(xlabel)
    decorate(ax, panel, title)

def radial_legend(fig, pos):
    fig.legend([Line2D([0], [0], marker=['o', 's'][k], color=RC[r], lw=1) for k, r in enumerate(RC)], [RL[r] for r in RC], loc='upper center', bbox_to_anchor=pos, ncol=2, frameon=False)

def fig6():
    style()
    d = read('fig6_regions.csv')
    fig, ax = plt.subplots(1, 2, figsize=(180 / 25.4, 100 / 25.4))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.25, top=0.7, wspace=0.85)
    radial(ax[0], 'Fig6', 'a', d[d.sector.eq('whole') & d.estimand.eq('pre_mean')], 'field', ['q700_raw_common', 'q700_climatology', 'q700_anomaly'], ['Observed', 'Background', 'Anomaly'], 'Pre-event state contrast', 'Specific humidity difference (g kg⁻¹)')
    radial(ax[1], 'Fig6', 'b', d[d.sector.eq('whole') & d.estimand.eq('integral_0_24')], 'field', ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency'], ['Horizontal', 'Vertical', 'Residual', 'Net tendency'], '0–24 h integrated\ntendency contrast', 'Integrated tendency difference (g kg⁻¹)')
    radial_legend(fig, (0.57, 0.985))
    fig.text(0.57, 0.85, 'Temporary − final departure', ha='center', fontsize=9)
    save(fig, 'Fig6')
    fig = plt.figure(figsize=(180 / 25.4, 160 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.08, top=0.86, wspace=0.9, hspace=0.72)
    sectors = ['downshear', 'left_of_shear', 'upshear', 'right_of_shear']
    labs = ['Downshear', 'Left of shear', 'Upshear', 'Right of shear']
    radial(a, 'FigS14', 'a', d[d.field.eq('q700_asymmetric') & d.estimand.eq('pre_mean')], 'sector', sectors, labs, 'Moisture azimuthal deviation', 'Specific humidity difference (g kg⁻¹)')
    radial(b, 'FigS14', 'b', d[d.field.eq('cyclonic_tangential_wind700_asymmetric') & d.estimand.eq('pre_mean')], 'sector', sectors, labs, 'Circulation azimuthal deviation', 'Cyclonic tangential wind difference\n(m s⁻¹)')
    decorate(c, 'c', 'One shear vector defines four relative directions')
    c.set_xlim(-2.9, 2.9)
    c.set_ylim(-1.7, 1.7)
    c.axis('off')
    c.plot([-1, 1], [-1, 1], ls=':', color='.65', lw=0.8)
    c.plot([-1, 1], [1, -1], ls=':', color='.65', lw=0.8)
    c.annotate('', xy=(0, 1.1), xytext=(0, -0.6), arrowprops={'arrowstyle': '->', 'lw': 1.8, 'color': '#35769B'})
    c.text(0, 1.38, 'Downshear', ha='center')
    c.text(0, -1.38, 'Upshear', ha='center')
    c.text(-1.5, 0, 'Left of shear', ha='right', va='center')
    c.text(1.5, 0, 'Right of shear', ha='left', va='center')
    c.text(-2.6, 1.3, 'S = v₂₀₀ − v₈₅₀', ha='left', fontsize=9)
    radial_legend(fig, (0.57, 0.995))
    fig.text(0.57, 0.94, 'Pre-event azimuthal deviations: temporary − final', ha='center')
    save(fig, 'FigS14', exclude_axes=[c])

def fig7():
    style()
    effects = read('fig7_effects.csv')
    diag = read('fig7_diagnostics.csv')
    sens = read('fig7_sensitivity.csv')
    fig, ax = plt.subplots(2, 2, figsize=(180 / 25.4, 170 / 25.4))
    fig.subplots_adjust(left=0.15, right=0.975, bottom=0.16, top=0.91, wspace=0.48, hspace=1.0)
    policies = ['shear200_850', 'midRH']
    pl = ['Shear −2 m/s', 'RH +5 pp']
    methods = ['gcomp', 'onestep', 'ipw']
    mc = ['#246F9B', '#B67432', '#747C85']
    mk = ['o', 's', '^']
    mn = ['G-formula', 'One-step', 'Density weighting']
    for col, outcome in enumerate(['recovery48_pp', 'delta24_kt']):
        a = ax[0, col]
        decorate(a, chr(97 + col), 'Recovery within 48 h' if col == 0 else 'Wind change over 24 h')
        for j, p in enumerate(policies):
            for k, m in enumerate(methods):
                z = effects[effects.policy.eq(p) & effects.outcome.eq(outcome) & effects.method.eq(m)]
                assert len(z) == 1
                r = z.iloc[0]
                interval(a, r.estimate, r.ci_low, r.ci_high, 1 - j + (1 - k) * 0.2, mc[k], mk[k], 4)
                record('Fig7', chr(97 + col), 'fig7_effects.csv', z)
        a.axvline(0, color='.6', ls='--', lw=0.7)
        a.set_yticks([1, 0], ['Shear\n−2 m/s', 'RH\n+5 pp'])
        a.set_ylim(-0.5, 1.5)
        a.set_xlabel('Probability difference (pp)' if col == 0 else 'Wind-change difference (kt)')
    fig.legend([Line2D([0], [0], marker=mk[k], color=mc[k], lw=1) for k in range(3)], ['a,b: ' + mn[0], mn[1], mn[2]], loc='center', bbox_to_anchor=(0.55, 0.525), ncol=3, frameon=False)
    a = ax[1, 0]
    decorate(a, 'c', 'Policy coverage and density support')
    for j, p in enumerate(policies):
        z = diag[diag.policy.eq(p) & diag.outcome.eq(48)]
        r = z.iloc[0]
        a.plot(j - 0.12, r.coverage * 100, 'o', color='#555F68', ms=6)
        a.plot(j + 0.12, r.ratio_ESS / 2371 * 100, 'o', mfc='white', mec='#555F68', ms=6, mew=1.1)
        record('Fig7', 'c', 'fig7_diagnostics.csv', z.assign(changed_fraction_percent=100 * r.coverage, ESS_fraction_percent=100 * r.ratio_ESS / 2371))
    a.set_xticks([0, 1], pl)
    a.set_xlim(-0.55, 1.55)
    a.set_ylim(0, 100)
    a.set_ylabel('Cohort fraction (%)')
    a.legend([Line2D([0], [0], marker='o', color='#555F68', ls='', ms=5), Line2D([0], [0], marker='o', mfc='white', mec='#555F68', ls='', ms=5)], ['Changed exposure fraction', 'Density-ratio ESS/n'], frameon=False, loc='upper left', fontsize=7)
    a = ax[1, 1]
    decorate(a, 'd', 'Missing-outcome response assumption')
    for j, p in enumerate(policies):
        z = sens[sens.policy.eq(p) & sens.outcome.eq(48)].sort_values('offset')
        a.plot(z.offset, z.estimate, 'o-', color=['#246F9B', '#C58137'][j], ms=3, label=pl[j])
        record('Fig7', 'd', 'fig7_sensitivity.csv', z)
    a.axhline(0, color='.6', ls='--', lw=0.7)
    a.axvline(0, color='.85', lw=0.7)
    a.set_xlabel('Assumed log-odds shift in policy response\nfor changed cyclones with missing outcomes')
    a.set_ylabel('Recovery contrast (pp)')
    a.legend(frameon=False, loc='upper left', fontsize=7)
    save(fig, 'Fig7')

def fig8():
    style()
    env = dict(globals())
    env.update(read=lambda name: read(name), BASINS=['NA', 'EP', 'WP', 'NI', 'SI', 'SP'])
    env['REGIONS'] = env['BASINS'] + ['SCS_box', 'Japan_Sea_box']
    env['REGNAME'] = {**{b: b for b in env['BASINS']}, 'SCS_box': 'SCS box', 'Japan_Sea_box': 'Japan Sea box'}
    env['COLORS'] = {'known_return': '#246F9B', 'complete_nonreturn': '#A27B53', 'unknown': '#88939D'}
    env['GROUPS'] = {'known_return': 'Return', 'complete_nonreturn': 'Complete nonreturn', 'unknown': 'Unknown'}
    funcs = get_functions('fig8_original.py', ['coast', 'forest_point', 'basin_rates', 'regional_risk', 'model_comparison', 'qcurve', 'budget_contrast'])
    funcs = funcs.replace("['horizontal_advection', 'vertical_advection', 'moving_tendency']", "['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency']")
    funcs = funcs.replace("'Horizontal advection'", "'Horizontal'").replace("'Vertical advection'", "'Vertical'").replace("'Moving tendency'", "'Net tendency'")
    exec(funcs, env)
    fig, a = plt.subplots(2, 2, figsize=(180 / 25.4, 168 / 25.4))
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.115, top=0.91, wspace=0.75, hspace=0.78)
    env['coast'](a[0, 0])
    r = read('G2_first_confirmed_return.csv')
    a[0, 0].scatter(r.return_lon % 360, r.return_lat, s=9, c='#246F9B', alpha=0.72, edgecolors='none')
    decorate(a[0, 0], 'a', 'Actual first near-peak return\n(n = 123)')
    record('Fig8', 'a', 'G2_first_confirmed_return.csv', r)
    env['basin_rates'](a[0, 1])
    decorate(a[0, 1], 'b', 'Distinct post-peak phenomena\n(n = 2366)')
    env['regional_risk'](a[1, 0])
    decorate(a[1, 0], 'c', 'Natural-course risk\nfrom departure origin')
    env['budget_contrast'](a[1, 1])
    decorate(a[1, 1], 'd', 'SCS: return − complete nonreturn')
    a[1, 1].set_xlabel('0–24 h integrated tendency contrast\n(g kg⁻¹)')
    a[1, 0].set_xlabel('Recovery within 48 h (%)')
    fig.text(0.565, 0.018, 'c: OOF natural-course prediction; grey: unknown-outcome bounds.  d: same-support 700 hPa budget.', ha='center', fontsize=7)
    save(fig, 'Fig8')
    for source, panel in [('G1_genesis_basin_recurrence.csv', 'b'), ('general_reintensification_genesis_basin_rates.csv', 'b'), ('G3_region_risk_estimates.csv', 'c'), ('G3_region_followup_counts.csv', 'c')]:
        record('Fig8', panel, source, read(source))
    budget = read('post_process_return_contrasts.csv')
    z = budget[budget.field.isin(['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency'])]
    assert len(z) == 8
    record('Fig8', 'd', 'post_process_return_contrasts.csv', z)
    for radial, g in z.groupby('radial'):
        assert g.n_return.nunique() == g.n_complete_nonreturn.nunique() == 1
        v = g.set_index('field').estimate
        assert abs(v.moving_tendency - v.horizontal_advection - v.vertical_advection - v.residual) < 1e-10
    fig, a = plt.subplots(1, 2, figsize=(180 / 25.4, 100 / 25.4))
    fig.subplots_adjust(left=0.205, right=0.97, bottom=0.22, top=0.78, wspace=0.63)
    env['model_comparison'](a[0], 'interval_log_likelihood')
    decorate(a[0], 'a', 'Global common sample (n = 1344)')
    env['qcurve'](a[1], 'outer')
    decorate(a[1], 'b', 'SCS: outer-annulus moisture')
    a[1].set_ylabel('700 hPa specific humidity (g kg⁻¹)')
    a[0].set_xlabel('Held-out log-likelihood gain per SID')
    save(fig, 'FigS19')
    record('FigS19', 'a', 'environment_model_paired_comparisons.csv', read('environment_model_paired_comparisons.csv').query('metric == "interval_log_likelihood"'))
    record('FigS19', 'b', 'post_process_curves.csv', read('post_process_curves.csv').query('radial == "outer" and field == "q700"'))
def render():
 fig2();fig3();fig6();fig7();fig8()

