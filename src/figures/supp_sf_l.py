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

def h2():
    j = Job('SF-L')
    main = j.load('FP0266')
    state = j.load('FP0262')
    assert len(main) == 6 and main.variable.nunique() == 6 and (main.bootstrap_success == 2000).all()
    fig = plt.figure(figsize=(7.08661417323, 6.88976377953))
    axs = []
    a = axis(fig, 29, 106, 56, 48, 'a')
    axs.append(a)
    forest(a, main, 'coef', 'bootstrap_ci_low', 'bootstrap_ci_high')
    a.set_xlim(-2.2, 2.2)
    a.set_xticks([-2, 0, 2])
    a.set_xlabel('Slope difference (kt per model SD)')
    title(a, 'a', 'Six primary state interactions')
    a = axis(fig, 115, 106, 56, 48, 'b')
    axs.append(a)
    a.set_xlim(0, 1)
    a.set_ylim(5.6, -1.0)
    a.axis('off')
    title(a, 'b', 'Primary inference: six-test family')
    for x, txt in [(0, 'Factor'), (0.46, 'n'), (0.69, 'P'), (0.93, 'q')]:
        a.text(x, -0.7, txt, ha='right' if x else 'left', va='center', fontweight='bold')
    for i, v in enumerate(VAR):
        r = main[main.variable.eq(v)].iloc[0]
        for x, txt in [(0, VL[i]), (0.46, str(int(r.n_sid))), (0.69, f'{r.p:.3f}'), (0.93, f'{r.q:.3f}')]:
            a.text(x, i, txt, ha='right' if x else 'left', va='center')
    conditions = [('observed_6h_endpoints_tropical', 'Tropical endpoints', 'FP0263'), ('observed_6h_endpoints_sea_and_tropical', 'Sea + tropical endpoints', 'FP0265')]
    for k, (selection, label, binding) in enumerate(conditions):
        a = axis(fig, [29, 115][k], 24, 56, 48, ['c', 'd'][k])
        axs.append(a)
        base = state[state.selection.eq('unfiltered')]
        d = state[state.selection.eq(selection)]
        forest(a, base, 'coef', 'ci_low', 'ci_high', offset=-0.12, color='#777777', marker='o')
        forest(a, d, 'coef', 'ci_low', 'ci_high', offset=0.12, color='#0072B2', marker='s')
        a.set_xlim(-2.2, 2.2)
        a.set_xticks([-2, 0, 2])
        a.set_xlabel('Slope difference (kt per fixed SD)')
        title(a, ['c', 'd'][k], label)
    fig.legend(handles=[Line2D([0], [0], color='#777777', marker='o', lw=0.8, label='Unfiltered'), Line2D([0], [0], color='#0072B2', marker='s', lw=0.8, label='Specified endpoint selection')], loc='center', bbox_to_anchor=(0.55, 90 / 175), ncol=2, frameon=False)
    fig.text(0.52, 5 / 175, 'Intervals: final 95% SID percentile bootstrap. P/q: original model inference; all q > 0.05.', ha='center')
    j.save(fig, axs, [['a', 'b'], ['c', 'd']], [['a', 'c'], ['b', 'd']], '')
def render():
 h2()

