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

def sf_f():
    j = Job('SF-F')
    d = j.load('SF-F-H4-30')
    de = j.load('SF-F-EXACT-DE-6')
    assert len(d) == 30 and len(de) == 6
    fig = plt.figure(figsize=(7.08661417323, 8.46456692913))
    axes = []
    lo = float(d.ci_low_percent.min())
    hi = float(d.ci_high_percent.max())
    pad = (hi - lo) * 0.1
    for i, v in enumerate(VAR):
        a = axis(fig, [24, 81, 138][i % 3], [154, 85][i // 3], 36, 43, chr(97 + i))
        axes.append(a)
        q = d[d.variable.eq(v)]
        for y, s in enumerate(['D', 'E', 'F', 'Res', 'D-E']):
            z = q[q.stage.eq(s)].iloc[0]
            e = float(z.percent_per_1sd)
            a.errorbar(e, y, xerr=[[e - z.ci_low_percent], [z.ci_high_percent - e]], fmt='o' if s != 'D-E' else 's', ms=3, c=PAL.get(s, '#333'), elinewidth=0.8, capsize=2)
        a.set_yticks(range(5), ['D', 'E', 'F', 'Res', 'D−E'])
        a.set_ylim(4.6, -0.6)
        a.axvline(0, color='.75', lw=0.6)
        a.set_xlim(lo - pad, hi + pad)
        a.set_xlabel('Duration response (%)')
        title(a, chr(97 + i), VL[i])
        tidy(a)
    a = axis(fig, 24, 15, 150, 43, 'g')
    axes.append(a)
    a.set_xlim(0, 1)
    a.set_ylim(6, -1.3)
    a.axis('off')
    title(a, 'g', 'Exact D + E allocation: all six native-unit effects (n = 2238)')
    for x, t in [(0, 'Factor'), (0.2, 'Per unit'), (0.39, 'Estimate'), (0.6, '95% analytic CI'), (0.88, 'P')]:
        a.text(x, -0.65, t, ha='left', va='center', fontweight='bold')
    units = ['°C', 'm/s', '% RH', '$10^{-5}\,\mathrm{s}^{-1}$', 'Pa/s', '$10^{-5}\,\mathrm{s}^{-1}$']
    for i, v in enumerate(VAR):
        z = de[de.variable.eq(v)].iloc[0]
        for x, t in [(0, VL[i]), (0.2, units[i]), (0.39, f'{z.coef:.4g}'), (0.6, f'[{z.ci_low:.4g}, {z.ci_high:.4g}]'), (0.88, f'{z.p:.3g}')]:
            a.text(x, i + 0.35, t, ha='left', va='center')
    fig.text(0.52, 3.5 / 215, 'H4: all 30 tests in one BH family. D−E is a direct log-slope contrast, then exponentiated.', ha='center')
    j.save(fig, axes, [['a', 'b', 'c'], ['d', 'e', 'f']], [['a', 'd'], ['b', 'e'], ['c', 'f']], '')
def render():
 sf_f()

