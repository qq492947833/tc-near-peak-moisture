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

def sf_n():
    j = Job('SF-N')
    d = j.load('FP0177')
    assert len(d) == 72
    fig = plt.figure(figsize=(7.08661417323, 8.46456692913))
    axes = []
    labs = ['12 h: A', '24 h: A', 'Full: A', '12 h: |A|', '24 h: |A|', 'Full: |A|']
    for i, v in enumerate(VAR):
        a = axis(fig, [31, 117][i % 2], [155, 85, 15][i // 2], 56, 43, chr(97 + i))
        axes.append(a)
        q = d[d.variable.eq(v)]
        for oi, outcome in enumerate(['A_signed', 'A_absolute']):
            for wi, window in enumerate(['12h', '24h', 'full']):
                for si, support in enumerate(['window_specific', 'common_SID_all_windows']):
                    z = q[q.outcome.eq(outcome) & q.window.eq(window) & q.support.eq(support)].iloc[0]
                    e = float(z.estimate)
                    a.errorbar(e, oi * 3 + wi + (si - 0.5) * 0.22, xerr=[[e - z.ci_low], [z.ci_high - e]], fmt=['o', 's'][si], c=['#333333', '#0072B2'][si], ms=2.8, capsize=1.6, elinewidth=0.7)
        a.axvline(0, color='.75', lw=0.6)
        a.set_yticks(range(6), labs)
        a.set_ylim(5.6, -0.6)
        a.set_xlabel('A or |A| per fixed contrast SD')
        title(a, chr(97 + i), VL[i])
        tidy(a)
    fig.legend(handles=[Line2D([0], [0], c='#333', marker='o', lw=0.8, label='Window-specific support'), Line2D([0], [0], c='#0072B2', marker='s', lw=0.8, label='Common-SID support')], loc='upper center', bbox_to_anchor=(0.54, 0.997), ncol=2, frameon=False)
    j.save(fig, axes, [['a', 'b'], ['c', 'd'], ['e', 'f']], [['a', 'c', 'e'], ['b', 'd', 'f']], '')
def render():
 sf_n()

