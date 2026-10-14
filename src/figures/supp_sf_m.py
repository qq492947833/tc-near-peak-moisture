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

def n2():
    j = Job('SF-M')
    d = j.load('FP0029')
    assert len(d) == 12 and (d.n_sid == 2254).all() and (d.df == 2233).all()
    fig = plt.figure(figsize=(7.08661417323, 3.93700787402))
    axs = []
    for k, contrast in enumerate(['GB', 'GL']):
        a = axis(fig, [29, 115][k], 25, 56, 52, ['a', 'b'][k])
        axs.append(a)
        q = d[d.contrast.eq(contrast)]
        forest(a, q, 'beta', 'ci_low', 'ci_high', offset=-0.1, color='#333333', marker='o')
        forest(a, q, 'beta', 'bootstrap_low', 'bootstrap_high', offset=0.1, color='#0072B2', marker='s')
        a.set_xlim(-0.035, 0.035)
        a.set_xticks([-0.03, 0, 0.03])
        a.set_xlabel('Coefficient contrast (log duration / SD)')
        title(a, ['a', 'b'][k], ['F minus U coefficients', 'U minus R coefficients'][k])
    fig.legend(handles=[Line2D([0], [0], color='#333333', marker='o', lw=0.8, label='Primary: HC3 / Student t'), Line2D([0], [0], color='#0072B2', marker='s', lw=0.8, label='Auxiliary: SID bootstrap')], loc='center', bbox_to_anchor=(0.54, 12 / 100), ncol=2, frameon=False)
    fig.text(0.52, 2.8 / 100, 'n = 2254; df = 2233. All 12 tests retained; all BH q > 0.05.', ha='center')
    j.save(fig, axs, [['a', 'b']], [], '')
def render():
 n2()

