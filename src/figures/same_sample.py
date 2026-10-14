from .common import *
D=DATA/'same_sample'
T1=read_csv(D/'T1_summary.csv').set_index('metric')
T3=read_csv(D/'T3_summary.csv').set_index('metric')
LAYERS=read_csv(D/'T1_pressure_layer_summary.csv')
CURVE=read_csv(D/'T1_same80_absolute_time_curves.csv')
COMP=read_csv(D/'paired_flux_components_0_24_summary.csv')
COMMON=read_csv(D/'common_prepost_paired_change_contrasts_summary.csv')
COMP_LAYER=read_csv(D/'component_by_pressure_and_sector_summary.csv')
SOURCE=[]
def export(fig,name):
 from .common import export as save
 save(fig,'FigS17')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'DejaVu Sans'], 'font.size': 7.6, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7.3, 'legend.fontsize': 7.4, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
COL = {'internal': '#2B769B', 'final': '#B86B45', 'difference': '#36515F'}
GRAY = '#939DA4'
LIGHT = '#E4E8EB'
SOURCE = []

def record(figure, panel, table, metric, row, **extra):
    SOURCE.append(dict(figure=figure, panel=panel, source_table=table, metric=metric, **row.to_dict(), **extra))

def decorate(ax, label, title):
    ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(-31, 10), textcoords='offset points', fontsize=9, fontweight='bold', va='bottom', ha='left')
    ax.set_title(title, loc='left', pad=12)

def grid(height_mm=200):
    fig, axs = plt.subplots(3, 2, figsize=(180 / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.145, right=0.965, bottom=0.075, top=0.9, wspace=0.64, hspace=0.6)
    return (fig, axs)

def forest(ax, means, lo, hi, y, color=COL['difference'], marker='o', scale=1):
    means = np.asarray(means) / scale
    lo = np.asarray(lo) / scale
    hi = np.asarray(hi) / scale
    assert np.isfinite(np.r_[means, lo, hi]).all()
    assert (lo <= means).all() and (means <= hi).all()
    ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt=marker, color=color, ms=4, capsize=2, lw=1.1)

def zero(ax):
    ax.axvline(0, color=GRAY, ls='--', lw=0.7, zorder=0)
    ax.grid(axis='x', color=LIGHT, lw=0.5, zorder=0)

def selected(df, **conditions):
    d = df
    for key, value in conditions.items():
        d = d[d[key] == value]
    assert len(d) == 1, (conditions, len(d))
    return d.iloc[0]

def new_rows(df, quantities, **conditions):
    return pd.DataFrame([selected(df, quantity=q, **conditions) for q in quantities])

def numeric_forest(ax, d, y, **kw):
    forest(ax, d['estimate'], d['low'], d['high'], y, scale=1000000000000.0, **kw)

def bind_new(figure, panel, table, d):
    for _, r in d.iterrows():
        record(figure, panel, table, '/'.join((str(r.get(k, '')) for k in ['kind', 'layer', 'quantity'])), r)

def si():
    fig, axs = grid(height_mm=185)
    rings = ['100_300km', '300_800km']
    colors = [COL['internal'], COL['final']]
    markers = ['o', 's']
    fields = ['q700_raw_common', 'q700_climatology', 'q700_anomaly']
    ax = axs[0, 0]
    decorate(ax, 'a', 'Pre-event 700-hPa humidity')
    for i, ring in enumerate(rings):
        metrics = ['old_' + ring + '_pre_mean_' + f for f in fields]
        d = T3.loc[metrics]
        assert (d.n == 80).all()
        forest(ax, d.difference, d.low, d.high, np.array([2, 1, 0]) + (0.11 if i == 0 else -0.11), color=colors[i], marker=markers[i])
        for m, r in d.iterrows():
            record('FigS15', 'a', 'T3_summary.csv', m, r)
    zero(ax)
    ax.set_yticks([2, 1, 0], ['Observed', 'Background', 'Anomaly'])
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel('Humidity contrast (g kg⁻¹)')
    ax.set_xlim(-0.8, 0.82)
    ax = axs[0, 1]
    decorate(ax, 'b', '700-hPa tendency compensation')
    fields = ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency']
    for i, ring in enumerate(rings):
        metrics = ['old_' + ring + '_integral_0_24_' + f for f in fields]
        d = T3.loc[metrics]
        assert (d.n == 80).all()
        forest(ax, d.difference, d.low, d.high, np.array([3, 2, 1, 0]) + (0.11 if i == 0 else -0.11), color=colors[i], marker=markers[i])
        for m, r in d.iterrows():
            record('FigS15', 'b', 'T3_summary.csv', m, r)
    zero(ax)
    ax.set_yticks([3, 2, 1, 0], ['Horizontal', 'Vertical', 'Residual', 'Net tendency'])
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel('Integrated contrast (g kg⁻¹)')
    ax.set_xlim(-1.4, 1.65)
    ax = axs[1, 0]
    decorate(ax, 'c', 'Annular input and stock change')
    metrics = ['post_side_net', 'stock_change_0_24']
    d = T3.loc[metrics]
    assert (d.n == 79).all()
    forest(ax, d.difference, d.low, d.high, [1, 0], scale=1000000000000.0)
    zero(ax)
    ax.set_yticks([1, 0], ['Side input', 'Stock change'])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(-0.18, 1.5)
    ax.set_xlabel('Temporary − final (10¹² kg)')
    for m, r in d.iterrows():
        record('FigS15', 'c', 'T3_summary.csv', m, r)
    ax = axs[1, 1]
    decorate(ax, 'd', 'Net inward transport at 300 km')
    for side, marker in [('internal', 'o'), ('final', 's')]:
        d = CURVE[(CURVE.kind == 'net') & (CURVE.side == side)].sort_values('lag_h')
        assert (d.n == 80).all()
        t = d.lag_h.to_numpy()
        y = d.mean_kg_s.to_numpy() / 100000000.0
        lo = d.low_kg_s.to_numpy() / 100000000.0
        hi = d.high_kg_s.to_numpy() / 100000000.0
        ax.fill_between(t, lo, hi, color=COL[side], alpha=0.16, lw=0)
        ax.plot(t, y, color=COL[side], marker=marker, ms=3, lw=1.3, label='Temporary' if side == 'internal' else 'Final')
        for _, r in d.iterrows():
            record('FigS15', 'd', 'T1_same80_absolute_time_curves.csv', f'{side}_{r.lag_h}h', r)
    ax.set_xlabel('Time from departure (h)')
    ax.set_ylabel('Net inward flux (10⁸ kg s⁻¹)')
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(loc='upper right', ncol=2, columnspacing=0.8, handlelength=1.5, frameon=False, fontsize=7)
    fig.legend([Line2D([0], [0], marker='o', color=colors[0], lw=1), Line2D([0], [0], marker='s', color=colors[1], lw=1)], ['100–300 km', '300–800 km'], title='a,b: annulus', title_fontsize=7, loc='upper center', bbox_to_anchor=(0.55, 1.005), ncol=2, frameon=False)
    ax = axs[2, 0]
    decorate(ax, 'e', 'Net exchange by pressure layer')
    assert (LAYERS.n == 80).all()
    y = np.arange(4)[::-1]
    forest(ax, LAYERS.difference_kg, LAYERS.low_kg, LAYERS.high_kg, y, scale=1000000000000.0)
    zero(ax)
    ax.set_yticks(y, [s.replace('-', '–') for s in LAYERS.layer_hPa])
    ax.set_ylim(-0.5, 3.8)
    ax.set_xlabel('Net contrast (10¹² kg)')
    ax.set_ylabel('Pressure interval (hPa)')
    ax.set_xlim(-0.13, 0.96)
    for _, r in LAYERS.iterrows():
        record('FigS15', 'e', 'T1_pressure_layer_summary.csv', r.layer_hPa, r)
    ax = axs[2, 1]
    decorate(ax, 'f', 'Outward components by layer')
    layers = ['500-600_hPa', '600-700_hPa', '700-850_hPa', '850-925_hPa']
    for q, offset, color, marker in [('difference', 0.2, COL['difference'], 'D'), ('humidity', 0, COL['internal'], 'o'), ('relative_motion', -0.2, COL['final'], 's')]:
        d = pd.DataFrame([selected(COMP_LAYER, cohort='original_post80', window='0_24h', factor=1, kind='out', sector='all', layer=layer, quantity=q) for layer in layers])
        assert (d.n == 80).all()
        numeric_forest(ax, d, y + offset, color=color, marker=marker)
        bind_new('FigS15', 'f', 'component_by_pressure_and_sector_summary.csv', d)
    zero(ax)
    ax.set_yticks(y, [s.replace('_hPa', '').replace('-', '–') for s in layers])
    ax.set_ylim(-0.5, 4.2)
    ax.set_xlim(-1.0, 0.3)
    ax.set_xlabel('Outward contrast (10¹² kg)')
    ax.legend([Line2D([0], [0], color=COL['difference'], marker='D', lw=1), Line2D([0], [0], color=COL['internal'], marker='o', lw=1), Line2D([0], [0], color=COL['final'], marker='s', lw=1)], ['Total', 'q', 'Relative-motion'], loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=6.0, handlelength=1, columnspacing=0.6, frameon=False)
    export(fig, 'FigS15')
render=si

