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
BC = {'NA': '#0072B2', 'EP': '#E69F00', 'WP': '#009E73', 'NI': '#CC79A7', 'SI': '#D55E00', 'SP': '#56B4E9'}
STCOLS = ['development_duration_hours', 'escalation_duration_hours', 'maturity_duration_hours', 'resolution_duration_hours']

def finish_note(fig, H, text):
    fig.text(0.52, 3.5 / H, text, ha='center', fontsize=8)

def sf_b():
    j = Job('SF-B')
    duration = j.load('FP0047')
    fraction = j.load('FP0048')
    base = j.load('FP0050')
    j.load('FP0049')
    j.load('FP0110')
    fig = plt.figure(figsize=(7.08661417323, 7.08661417323))
    axes = [axis(fig, x, y, 56, 48, lab) for x, y, lab in [(31, 110, 'a'), (117, 110, 'b'), (31, 27, 'c'), (117, 27, 'd')]]
    a = axes[0]
    for st, c in zip(PAL, STCOLS):
        x = np.sort(duration[c].to_numpy(float))
        a.step(x, np.arange(1, len(x) + 1) / len(x), where='post', c=PAL[st], lw=1, label=st)
    a.set_xlabel('Stage duration (h)')
    a.set_ylabel('Empirical cumulative fraction')
    a.set_ylim(0, 1)
    a.set_xlim(left=0)
    a.legend(frameon=False, ncol=2, loc='lower right')
    title(a, 'a', 'All four stage distributions')
    tidy(a)
    a = axes[1]
    q = fraction.set_index('basin_genesis').loc[BS]
    left = np.zeros(6)
    for st, c in zip(PAL, STCOLS):
        a.barh(range(6), q[c], left=left, color=PAL[st], height=0.65)
        left += q[c].to_numpy(float)
    a.set_yticks(range(6), BS)
    a.invert_yaxis()
    a.set_xlim(0, 1)
    a.set_xlabel('Saved mean fraction of lifetime')
    title(a, 'b', 'Basin stage allocation')
    tidy(a)
    for ki in range(2):
        a = axes[2 + ki]
        for basin in BS:
            q = base[base.basin_genesis.eq(basin)]
            x = q.lifecycle_asymmetry_time.to_numpy(float)
            x = np.sort(np.abs(x) if ki else x)
            a.step(x, np.arange(1, len(x) + 1) / len(x), where='post', c=BC[basin], lw=1)
        a.set_xlim((0, 1) if ki else (-1, 1))
        a.set_ylim(0, 1)
        a.set_xlabel('Absolute asymmetry |A|' if ki else 'Signed asymmetry A')
        a.set_ylabel('Empirical cumulative fraction')
        title(a, ['c', 'd'][ki], ['Signed temporal asymmetry', 'Absolute temporal asymmetry'][ki])
        tidy(a)
    fig.legend(handles=[Line2D([0], [0], c=BC[b], lw=1, label=b) for b in BS], loc='center', bbox_to_anchor=(0.55, 92 / 180), ncol=6, frameon=False, handlelength=1.1, columnspacing=1)
    finish_note(fig, 180, 'Historical descriptive population: n = 2442. Every basin and stage retained; no inferential tests.')
    j.save(fig, axes, [['a', 'b'], ['c', 'd']], [['a', 'c'], ['b', 'd']], '')
approved_labels = {'Saved mean fraction of lifetime': 'Mean fraction of lifetime'}
BS=list(BC)

def render():
 sf_b()

