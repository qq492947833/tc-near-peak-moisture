from .common import *
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600})
VAR = ['X1_sst_c', 'X2_shear_ms', 'X3_mid_rh_pct', 'X4_d200_1e5s', 'X5_minus_omega500_pa_s', 'X6_signed_vo850_1e5s']
VL = ['SST', 'Shear', 'Mid-RH', 'Div. 200', 'Ascent 500', 'Vort. 850']
PAL = {'D': '#0072B2', 'E': '#E69F00', 'F': '#CC79A7', 'Res': '#009E73'}

def axis(fig, x, y, w, h, label):
    W, H = fig.get_size_inches() * 25.4
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.set_gid(label)
    return a

def title(a, label, text):
    a.annotate(label, xy=(0, 1), xycoords='axes fraction', xytext=(-14, 7), textcoords='offset points', fontsize=10, fontweight='bold', ha='left', va='bottom', annotation_clip=False)
    a.set_title(text, loc='left', pad=7, fontsize=9)

def tidy(a):
    for k in ['top', 'right']:
        a.spines[k].set_visible(False)
    a.tick_params(length=3, pad=2)

def forest(a, d, estimate, low, high, offset=0, color='#333333', marker='o', variables=VAR, labels=VL):
    for j, v in enumerate(variables):
        r = d[d.variable.eq(v)].iloc[0]
        e = float(r[estimate])
        lo = float(r[low])
        hi = float(r[high])
        a.errorbar(e, j + offset, xerr=[[e - lo], [hi - e]], fmt=marker, c=color, ms=3.2, elinewidth=0.8, capsize=2)
    a.set_yticks(range(len(variables)), labels)
    a.set_ylim(len(variables) - 0.45, -0.55)
    a.axvline(0, color='.75', lw=0.6, zorder=0)
    tidy(a)

def sf_e():
    j = Job('SF-E')
    native = j.load('FP0071')
    duration = j.load('FP0223')
    env = j.load('FP0230')
    fig = plt.figure(figsize=(7.08661417323, 7.48031496063))
    axes = [axis(fig, x, y, 56, 52, lab) for x, y, lab in [(31, 115, 'a'), (117, 115, 'b'), (31, 26, 'c'), (117, 26, 'd')]]
    a = axes[0]
    rows = [0.2, 0.25, 0.3]
    cols = [(15, 0.85), (15, 0.9), (15, 0.95), (20, 0.85), (20, 0.9), (20, 0.95)]
    values = np.array([[int(native[native.escalation_progress.eq(e) & native.min_peak_gain.eq(g) & native.maturity_progress.eq(m)].n_valid.iloc[0]) for g, m in cols] for e in rows])
    im = a.imshow(values, aspect='auto', cmap='YlGnBu', vmin=values.min(), vmax=values.max())
    a.set_yticks(range(3), [f'{x:.2f}' for x in rows])
    a.set_ylabel('Escalation progress')
    a.set_xticks(range(6), [f'{m:.2f}\n{g}' for g, m in cols])
    a.set_xlabel('Maturity progress / minimum gain (kt)')
    title(a, 'a', 'All 18 native-cohort thresholds')
    for i in range(3):
        for k in range(6):
            a.text(k, i, str(values[i, k]), ha='center', va='center', color='white' if values[i, k] > (values.min() + values.max()) / 2 else 'black')
    a = axes[1]
    scopes = ['beta085_native_multi', 'beta085_fixed90_multi_main', 'beta085_fixed90_multi_alternate', 'beta095_native_multi', 'beta095_fixed90_multi_main', 'beta095_fixed90_multi_alternate']
    labels = ['0.85: native', '0.85: fixed, main', '0.85: fixed, alt.', '0.95: native', '0.95: fixed, main', '0.95: fixed, alt.']
    q = duration[duration.scope.isin(scopes) & duration.metric.isin(['L', 'B'])]
    for i, s in enumerate(scopes):
        for k, m in enumerate(['L', 'B']):
            r = q[q.scope.eq(s) & q.metric.eq(m)].iloc[0]
            e = float(r.estimate)
            a.errorbar(e, i + (k - 0.5) * 0.22, xerr=[[e - r.ci_low], [r.ci_high - e]], fmt=['o', 's'][k], c=['#0072B2', '#333'][k], ms=3, capsize=2, elinewidth=0.8)
    a.set_yticks(range(6), labels)
    a.set_ylim(5.6, -0.6)
    a.set_xlabel('Mean covered duration (h)')
    title(a, 'b', 'Native versus fixed recurrence')
    tidy(a)
    fig.legend(handles=[Line2D([0], [0], c='#0072B2', marker='o', lw=0.8, label='Later L'), Line2D([0], [0], c='#333', marker='s', lw=0.8, label='Bridge B')], loc='center', bbox_to_anchor=(0.8, 96 / 190), ncol=2, frameon=False)
    for k, contrast in enumerate(['F_minus_U', 'U_minus_R']):
        a = axes[2 + k]
        q = env[env.contrast.eq(contrast)]
        for t, alt in enumerate(['beta085', 'beta095', 'C5']):
            forest(a, q[q.alternate.eq(alt)], 'estimate_sd', 'ci_low_sd', 'ci_high_sd', offset=(t - 1) * 0.22, color=['#0072B2', '#D55E00', '#333'][t], marker=['o', 's', '^'][t])
        a.set_xlabel('Alternate − main / fixed reference SD')
        title(a, ['c', 'd'][k], ['Threshold effect: mean F − mean U', 'Threshold effect: mean U − mean R'][k])
    fig.legend(handles=[Line2D([0], [0], c=c, marker=m, lw=0.8, label=l) for c, m, l in zip(['#0072B2', '#D55E00', '#333'], ['o', 's', '^'], ['Progress 0.85', 'Progress 0.95', 'C5 benchmark'])], loc='center', bbox_to_anchor=(0.55, 9 / 190), ncol=3, frameon=False)
    j.save(fig, axes, [['a', 'b'], ['c', 'd']], [['a', 'c'], ['b', 'd']], '')
def render():
 sf_e()

