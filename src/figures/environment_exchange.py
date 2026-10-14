from .common import *
D=DATA/'direct'
CONF=json.loads((D/'layout_config.json').read_text(encoding='utf-8'))
C={'temporary':CONF['temporary'],'final':CONF['final'],'difference':CONF['difference']}
LEDGER=[]
save=export
LEDGER = []
OUTPUTS = []
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})

def read(n):
    return pd.read_csv(D / n)

def record(target, panel, source, rows, meaning):
    LEDGER.append({'figure': target, 'panel': panel, 'source': source, 'rows': rows.to_dict('records'), 'meaning': meaning})

def title(ax, panel, t):
    ax.set_gid(panel)
    ax.set_title(t, loc='left', pad=13)
    ax.annotate(panel, (0, 1), xycoords='axes fraction', xytext=(-26, 11), textcoords='offset points', fontsize=10, weight='bold')

def point(ax, r, y, col, marker='o', div=1):
    e, lo, hi = [float(r[k]) / div for k in ['estimate', 'low', 'high']]
    ax.plot([lo, hi], [y, y], color=col, lw=1)
    ax.plot([lo, hi], [y, y], '|', color=col, ms=4)
    ax.plot(e, y, marker, color=col, ms=4)

def classlegend(fig, y=0.98):
    fig.legend([Line2D([0], [0], marker='o', color=C[k], lw=1.2) for k in ['temporary', 'final']], ['Temporary departure', 'Final departure'], ncol=2, loc='upper center', bbox_to_anchor=(0.55, y), frameon=False)

def zero(ax):
    ax.axvline(0, color='.6', lw=0.7, ls='--')
    ax.grid(axis='x', color='#e8ecef', lw=0.5)

def schematic(ax, target, panel):
    d = read('fig1_rows.csv')
    d = d[(d.figure == 'MF-A') & (d.panel == 'a')]
    pal = json.loads((D / 'palette.json').read_text())['stages']
    for stage, a, b in [('D', 0, 12), ('E', 12, 36), ('F', 36, 72), ('Res', 72, 78)]:
        ax.axvspan(a, b, facecolor=pal[stage], edgecolor='none', linewidth=0, alpha=0.13, zorder=0)
        ax.text((a + b) / 2, 105, stage, ha='center', fontsize=8, color=pal[stage], weight='bold')
    ax.plot(d.hours, d.wind_kt, 'o-', lw=1.4, ms=3, color='#36515F')
    ax.axhline(93.5, color='.45', ls='--', lw=0.8)
    ax.text(2, 95.5, 'Near-peak threshold', fontsize=7.5)
    for x, l, c in [(48, 'Temporary', C['temporary']), (72, 'Final', C['final'])]:
        y = float(d.loc[d.hours.eq(x), 'wind_kt'].iloc[0])
        ax.plot(x, y, 'o', color=c, ms=5)
        ax.annotate(l + ' departure', xy=(x, y), xytext=(x - 8, 52 if x == 48 else 43), ha='center', fontsize=7.5, color=c, arrowprops={'arrowstyle': '->', 'color': c, 'lw': 0.8})
    ax.set(xlim=(-1, 80), ylim=(30, 111), xlabel='Time from 35-kt start (h)', ylabel='Wind speed (kt)')
    title(ax, panel, 'Same-cyclone pairing (schematic)')
    record(target, panel, 'fig1_rows.csv', d, 'Frozen schematic (not an observed cyclone); illustrative departures at 48 and 72 h. Threshold/stages unchanged.')

def support(ax, target, panel):
    title(ax, panel, 'Analysis-specific paired support')
    ax.axis('off')
    counts=read('support_population_summary.csv').set_index('population').n_sid
    rows=[(str(int(counts[key])),label) for key,label in [('paired event identities','Environmental histories'),('original radial process comparison','Post-event radial processes'),('original 300 km boundary exchange','Complete boundary exchange'),('common pre/post boundary comparison','Complete pre/post windows')]]
    for j, (n, t) in enumerate(rows):
        y = 0.9 - j * 0.225
        ax.text(0.02, y, n, color='#36515F', fontsize=15, weight='bold', va='center')
        ax.text(0.23, y, t, fontsize=8, va='center')
    ax.text(0.02, 0.08, 'Cyclone identities; variable/lag support\nis reported separately.', fontsize=7.2, va='top')
    LEDGER.append({'figure': target, 'panel': panel, 'source': 'support_population_summary.csv', 'meaning': 'Distinct analysis counts; no inferred attrition flow or unverified nesting.'})

def definitions(target, full):
    fig = plt.figure(figsize=(180 / 25.4, (170 if full else 100) / 25.4))
    gs = fig.add_gridspec(2 if full else 1, 2, width_ratios=[1.45, 1])
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.1, top=0.87, wspace=0.32, hspace=0.8)
    schematic(a, target, 'a')
    support(b, target, 'b')
    if full:
        d = read('fig1_rows.csv')
        z = d[(d.figure == 'MF-A') & (d.panel == 'c')]
        c = fig.add_subplot(gs[1, 0])
        title(c, 'c', 'Stage contributions to elapsed time')
        pal = json.loads((D / 'palette.json').read_text())['stages']
        bottom = np.zeros(3)
        for st in ['D', 'E', 'F', 'Res']:
            q = z[z.stage.eq(st)].set_index('segment').reindex(['pre', 'middle', 'post'])
            v = q.mean_covered_h.to_numpy(float)
            gap = q.mean_uncovered_h.to_numpy(float)
            c.barh(range(3), v, left=bottom, color=pal[st], label=st, height=0.55)
            bottom += v
            c.barh(range(3), gap, left=bottom, facecolor='white', edgecolor=pal[st], linewidth=0.7, height=0.55)
            bottom += gap
        c.set(yticks=range(3), yticklabels=['Before\nfirst LMI', 'First–last\nLMI', 'After\nlast LMI'], xlabel='Mean elapsed duration (h)', ylim=(2.6, -0.6))
        c.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=7)
        c.text(0.99, -0.32, 'Outline: uncovered time', transform=c.transAxes, ha='right', fontsize=7)
        record(target, 'c', 'fig1_rows.csv', z, 'Unchanged saved stage/segment mean elapsed hours, 2368 cohort.')
        q = d[(d.figure == 'MF-A') & (d.panel == 'd')].sort_values('A')
        e = fig.add_subplot(gs[1, 1])
        title(e, 'd', 'Lifecycle duration asymmetry')
        e.step(q.A, q.empirical_CDF, where='post', color='#36515F')
        e.axvline(0, color='.6', ls='--', lw=0.7)
        e.text(0.03, 0.92, 'n = 2,368', transform=e.transAxes, fontsize=8)
        e.set(xlim=(-1, 1), ylim=(0, 1), xlabel='Asymmetry A', ylabel='Empirical cumulative fraction')
        record(target, 'd', 'fig1_rows.csv', q, 'Exact saved asymmetry distribution.')
    save(fig, target, exclude=[b])

def trajectory(ax, target, panel, var, contrast=False):
    d = read('fig5_trajectories.csv')
    labs = {'midRH': ('Mid-level relative humidity', 'RH (%)', 'RH difference (pp)'), 'shear200_850': ('Deep-layer shear', 'Shear (m s⁻¹)', 'Shear difference (m s⁻¹)'), 'PI_headroom': ('PI headroom', 'PI − V (m s⁻¹)', 'Headroom difference (m s⁻¹)')}
    t, yl, yd = labs[var]
    for s in ['difference'] if contrast else ['internal', 'final']:
        z = d[d.variable.eq(var) & d.series.eq(s)].sort_values('lag_h')
        col = C['temporary' if s == 'internal' else s]
        x = z.lag_h.to_numpy()
        ax.plot(x, z.estimate.to_numpy(), 'o-', ms=2.5, lw=1.2, color=col)
        ax.fill_between(x, z.lo.to_numpy(), z.hi.to_numpy(), color=col, alpha=0.16, lw=0)
        record(target, panel, 'fig5_trajectories.csv', z, 'Pointwise saved paired-SID intervals; absolute histories' if not contrast else 'Direct temporary-minus-final contrast; not subtraction of separate intervals.')
    ax.axvline(0, color='.6', ls='--', lw=0.7)
    if contrast:
        ax.axhline(0, color='.6', ls='--', lw=0.7)
    ax.set(xticks=[-24, -12, 0, 12, 24], xlabel='Time from departure (h)', ylabel=yd if contrast else yl)
    title(ax, panel, {'midRH': 'RH contrast', 'shear200_850': 'Shear contrast', 'PI_headroom': 'PI headroom contrast'}[var] if contrast else t)

def background(ax, target, panel):
    d = read('environmental_contrasts.csv')
    z = d[d.variable.eq('midRH')]
    g = z[z.model.eq('geographical')].iloc[0]
    s = z[z.model.eq('geographical_state')].iloc[0]
    vals = [(g.original_raw, g.original_lo, g.original_hi), (g.overlap_raw, g.overlap_raw_lo, g.overlap_raw_hi), (g.standardized, g.lo, g.hi), (s.standardized, s.lo, s.hi)]
    labels = ['Original paired sample (117)', 'Overlap subset, raw (105)', 'Geography / season (105)', 'Geography / season + state (105)']
    for y, (e, l, h) in enumerate(vals):
        point(ax, {'estimate': e, 'low': l, 'high': h}, y, '#36515F' if y < 2 else '#805C94', marker='o' if y < 2 else 's')
    zero(ax)
    ax.set(yticks=range(4), yticklabels=labels, ylim=(3.5, -0.5), xlabel='Pre-event RH difference (pp)')
    title(ax, panel, 'RH and measured background')
    record(target, panel, 'environmental_contrasts.csv', z, 'Original raw, restricted-support raw, and same-support balance; no identical-condition claim.')

def environments(target, full):
    if full:
        fig = plt.figure(figsize=(180 / 25.4, 211 / 25.4))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85])
        axes = []
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.085, top=0.9, wspace=0.62, hspace=0.82)
        for j, v in enumerate(['midRH', 'shear200_850', 'PI_headroom']):
            for i in [0, 1]:
                a = fig.add_subplot(gs[i, j])
                trajectory(a, target, chr(97 + i * 3 + j), v, bool(i))
                axes.append(a)
        g = fig.add_subplot(gs[2, 1:])
        background(g, target, 'g')
        g.set_position([0.48, 0.1, 0.48, 0.16])
        classlegend(fig, 0.99)
        save(fig, target, exclude=[g])
    else:
        fig, ax = plt.subplots(2, 2, figsize=(180 / 25.4, 163 / 25.4))
        fig.subplots_adjust(left=0.12, right=0.975, bottom=0.1, top=0.87, wspace=0.72, hspace=0.78)
        trajectory(ax[0, 0], target, 'a', 'midRH')
        trajectory(ax[0, 1], target, 'b', 'shear200_850')
        trajectory(ax[1, 0], target, 'c', 'PI_headroom', True)
        background(ax[1, 1], target, 'd')
        ax[1, 1].set_yticklabels(['Original (117)', 'Overlap raw (105)', 'Geography (105)', '+ State (105)'])
        classlegend(fig, 0.99)
        save(fig, target)

def radial(ax, target, panel, budget=False):
    d = read('fig6_regions.csv')
    fields = ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency'] if budget else ['q700_raw_common', 'q700_climatology', 'q700_anomaly']
    labs = ['Horizontal', 'Vertical', 'Residual', 'Net tendency'] if budget else ['Observed', 'Background', 'Anomaly']
    for k, (region, col) in enumerate([('100_300km', C['temporary']), ('300_800km', C['final'])]):
        z = d[d.sector.eq('whole') & d.estimand.eq('integral_0_24' if budget else 'pre_mean') & d.region.eq(region)].set_index('field').loc[fields].reset_index()
        for y, r in z.iterrows():
            point(ax, {'estimate': r.estimate, 'low': r.lo, 'high': r.hi}, y + (k - 0.5) * 0.22, col, ['o', 's'][k])
        record(target, panel, 'fig6_regions.csv', z, 'Complete 700 hPa integrated tendency including residual' if budget else 'Pre-event 700 hPa radial state difference')
    zero(ax)
    ax.set(yticks=range(len(labs)), yticklabels=labs, ylim=(len(labs) - 0.5, -0.5), xlabel='Integrated tendency difference (g kg⁻¹)' if budget else 'Specific humidity difference (g kg⁻¹)')
    title(ax, panel, '700 hPa budget: 0–24 h' if budget else '700 hPa pre-event state')

def fluxrows(layer=False):
    d = read('pressure_components.csv' if layer else 'paired_flux_components.csv')
    return d[d.cohort.eq('original_post80') & d.window.eq('0_24h') & d.factor.eq(1) & d.sector.eq('all')]

def gross(ax, target, panel, contrast=False):
    d = fluxrows()
    z = d[d.layer.eq('all')]
    kinds = ['in', 'out', 'net']
    for k, q in enumerate(['difference'] if contrast else ['temporary', 'final']):
        for y, kind in enumerate(kinds):
            point(ax, z[z.kind.eq(kind) & z.quantity.eq(q)].iloc[0], y if contrast else y + (k - 0.5) * 0.22, C[q], div=1000000000000.0)
    zero(ax)
    ax.set(yticks=range(3), yticklabels=['Gross inward', 'Gross outward', 'Net inward'], ylim=(2.6, -0.6), xlabel=('Temporary − final' if contrast else 'Exchange') + ' (10¹² kg)')
    title(ax, panel, 'Paired exchange contrast' if contrast else '300 km boundary exchange')
    record(target, panel, 'paired_flux_components.csv', z[z.kind.isin(kinds) & z.quantity.isin(['difference'] if contrast else ['temporary', 'final'])], '0–24 h, 500–925 hPa, same 80 paired cyclones; net inward = gross in − gross out.')

def layerpanel(ax, target, panel, kind, contrast=False):
    d = fluxrows(True)
    layers = ['500-600_hPa', '600-700_hPa', '700-850_hPa', '850-925_hPa']
    z = d[d.layer.isin(layers) & d.kind.eq(kind)]
    for k, q in enumerate(['difference'] if contrast else ['temporary', 'final']):
        for y, layer in enumerate(layers):
            point(ax, z[z.layer.eq(layer) & z.quantity.eq(q)].iloc[0], y if contrast else y + (k - 0.5) * 0.24, C[q], div=1000000000000.0)
    zero(ax)
    ax.set(yticks=range(4), yticklabels=[x.replace('_hPa', '').replace('-', '–') for x in layers], ylim=(3.6, -0.6), ylabel='Pressure interval (hPa)', xlabel=('Temporary − final' if contrast else 'Interval-integrated mass') + ' (10¹² kg)')
    title(ax, panel, {'in': 'Gross inward', 'out': 'Gross outward', 'net': 'Net inward'}[kind] + (' contrast' if contrast else ' by pressure interval'))
    record(target, panel, 'pressure_components.csv', z[z.quantity.isin(['difference'] if contrast else ['temporary', 'final'])], 'Unequal pressure interval widths; plotted mass integrated per interval, not per hPa.')

def fluxfigure(target, full=True):
    fig, ax = plt.subplots(2, 2, figsize=(180 / 25.4, 161 / 25.4))
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.12, top=0.85, wspace=0.75, hspace=0.82)
    if full:
        gross(ax[0, 0], target, 'a')
        gross(ax[0, 1], target, 'b', True)
        layerpanel(ax[1, 0], target, 'c', 'net')
        fluxbackground(ax[1, 1], target, 'd', 'net')
        classlegend(fig, 0.99)
    else:
        radial(ax[0, 0], target, 'a')
        radial(ax[0, 1], target, 'b', True)
        gross(ax[1, 0], target, 'c')
        gross(ax[1, 1], target, 'd', True)
        fig.legend([Line2D([0], [0], marker=m, color=c, lw=1) for m, c in [('o', C['temporary']), ('s', C['final'])]], ['100–300 km', '300–800 km'], ncol=2, loc='upper center', bbox_to_anchor=(0.55, 0.99), frameon=False)
        fig.text(0.55, 0.915, 'a–b: temporary − final, 700 hPa; c–d: 80 pairs, 500–925 hPa', ha='center', fontsize=7.6)
        fig.legend([Line2D([0], [0], color=C[k], lw=1.2) for k in ['temporary', 'final']], ['Temporary departure (c)', 'Final departure (c)'], ncol=2, loc='lower center', bbox_to_anchor=(0.55, 0.005), frameon=False)
    save(fig, target)

def layers(target):
    fig, ax = plt.subplots(3, 2, figsize=(180 / 25.4, 217 / 25.4))
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.085, top=0.91, wspace=0.75, hspace=0.72)
    for i, k in enumerate(['in', 'out', 'net']):
        layerpanel(ax[i, 0], target, chr(97 + 2 * i), k)
        layerpanel(ax[i, 1], target, chr(98 + 2 * i), k, True)
    classlegend(fig, 0.99)
    fig.text(0.55, 0.025, '300 km boundary; 0–24 h; 80 pairs. Pressure intervals have unequal widths.', ha='center', fontsize=7.5)
    save(fig, target)

def composition(ax, target, panel):
    d = fluxrows()
    z = d[d.layer.eq('all') & d.kind.eq('out')]
    qs = ['humidity', 'relative_motion', 'difference']
    labs = ['Specific humidity', 'Relative normal motion', 'Total outward difference']
    for y, q in enumerate(qs):
        point(ax, z[z.quantity.eq(q)].iloc[0], y, ['#388E7D', '#B57544', '#36515F'][y], div=1000000000000.0)
    zero(ax)
    ax.set(yticks=range(3), yticklabels=labs, ylim=(2.6, -0.6), xlabel='Temporary − final (10¹² kg)')
    title(ax, panel, 'Gross outward composition: 80 pairs')
    record(target, panel, 'paired_flux_components.csv', z[z.quantity.isin(qs)], 'Whole-boundary symmetric q/relative normal-motion decomposition, post0–24 h; 80 pairs.')

def windows(ax, target, panel, change=False):
    d = read('common_prepost.csv')
    kinds = ['out', 'net']
    qs = ['Theta'] if change else ['D_pre', 'D_post']
    for k, q in enumerate(qs):
        for y, kind in enumerate(kinds):
            point(ax, d[d.kind.eq(kind) & d.quantity.eq(q)].iloc[0], y if change else y + (k - 0.5) * 0.25, ['#36515F'][0] if change else ['#939DA4', '#36515F'][k], ['o'][0] if change else ['o', 's'][k], div=1000000000000.0)
    zero(ax)
    ax.set(yticks=[0, 1], yticklabels=['Gross outward', 'Net inward'], ylim=(1.5, -0.5), xlabel=('Post − pre contrast change' if change else 'Temporary − final') + ' (10¹² kg)')
    title(ax, panel, 'Direct paired change: 78 pairs' if change else 'Shared pre/post windows: 78 pairs')
    record(target, panel, 'common_prepost.csv', d[d.kind.isin(kinds) & d.quantity.isin(qs)], 'Same78SIDs; pre−24–0h,post0–24h; direct change is paired difference-of-differences.')

def fluxbackground(ax, target, panel, kind):
    d = read('exchange_background80.csv')
    z = d[d.metric.eq('post_r300_' + kind)]
    g = z[z.model.eq('geographical')]
    s = z[z.model.eq('geographical_state')]
    rows = [g[g.target.eq(t)].iloc[0] for t in ['original80_raw', 'restricted_raw', 'balanced']] + [s[s.target.eq('balanced')].iloc[0]]
    for y, r in enumerate(rows):
        point(ax, r, y, '#36515F' if y < 2 else '#805C94', marker='o' if y < 2 else 's', div=1000000000000.0)
    zero(ax)
    ax.set(yticks=range(4), yticklabels=['Original raw (80)', 'Support raw (74)', 'Geography (74)', '+ State (74)'], ylim=(3.5, -0.5), xlabel='Temporary − final (10¹² kg)')
    title(ax, panel, {'in': 'Gross inward', 'out': 'Gross outward', 'net': 'Net inward'}[kind] + ' and background')
    record(target, panel, 'exchange_background80.csv', pd.DataFrame(rows), 'Original80 raw; restricted74 raw; same74 geography/season and geography/season+state standardized. Fixed2000SID refit intervals.')

def allbackground(target):
    fig, ax = plt.subplots(3, 1, figsize=(180 / 25.4, 188 / 25.4))
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.09, top=0.94, hspace=0.8)
    for i, kind in enumerate(['in', 'out', 'net']):
        fluxbackground(ax[i], target, chr(97 + i), kind)
    save(fig, target)

def components(target, background=False):
    fig = plt.figure(figsize=(180 / 25.4, (216 if background else 151) / 25.4))
    gs = fig.add_gridspec(3 if background else 2, 2, height_ratios=[1, 0.95, 0.85] if background else [1, 0.95])
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.09 if background else 0.18, top=0.93 if background else 0.9, wspace=0.75, hspace=1.05 if background else 0.85)
    composition(a, target, 'a')
    windows(b, target, 'b')
    windows(c, target, 'c', True)
    fig.legend([Line2D([0], [0], marker=m, color=c, lw=1) for m, c in [('o', '#939DA4'), ('s', '#36515F')]], ['−24 to 0 h', '0 to 24 h'], ncol=2, loc='lower center', bbox_to_anchor=(0.56, 0.34 if background else 0.035), frameon=False)
    if background:
        fluxbackground(fig.add_subplot(gs[2, :]), target, 'd', 'net')
    save(fig, target)
def render():
 definitions('Fig1',True)
 environments('Fig3',True)
 fluxfigure('Fig5',True)
 components('Fig6',False)
 layers('FigS18')
 allbackground('FigS19')

