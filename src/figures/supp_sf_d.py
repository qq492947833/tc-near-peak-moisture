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

def sf_d():
    j = Job('SF-D')
    groups = j.load('FP0059')
    counts = j.load('FP0065')
    recounts = j.load('FP0066')
    dur = j.load('FP0238')
    tr = j.load('FP0240')
    loo = j.load('FP0242')
    scopes = ['main_all', 'exclude_single_6h', 'strict_bridge_tropical', 'without_Freddy2023', 'without_basin_WP', 'without_basin_EP', 'without_basin_SI', 'without_basin_NA', 'without_basin_SP', 'without_basin_NI', 'expanded_estimable', 'expanded_added_estimable']
    scopemap = pd.DataFrame({'display_code': ['S' + str(i) for i in range(len(scopes))], 'source_scope': scopes})
    scopemap.to_csv(j.p / 'sensitivity_scope_codes.csv', index=False)
    fig = plt.figure(figsize=(7.08661417323, 8.46456692913))
    axes = [axis(fig, x, y, 56, 43, chr(97 + r * 2 + c)) for r, y in enumerate([155, 85, 15]) for c, x in enumerate([31, 117])]
    a = axes[0]
    bs = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']
    categories = ['Single near-LMI without reintensification', 'Multiple near-LMI episodes', 'Single near-LMI with reintensification']
    colors = ['.65', '#0072B2', '#E69F00']
    left = np.zeros(6)
    for k, key in enumerate(categories):
        q = groups[groups.group.eq(key) & groups.basin.ne('Global')].set_index('basin').loc[bs]
        a.barh(range(6), q.fraction, left=left, color=colors[k], height=0.65)
        left += q.fraction.to_numpy(float)
    a.set_yticks(range(6), bs)
    a.invert_yaxis()
    a.set_xlim(0, 1)
    a.set_xlabel('Fraction within basin')
    title(a, 'a', 'Recurrence classes')
    tidy(a)
    a = axes[1]
    a.bar(counts.near_lmi_episode_count - 0.18, counts['count'], width=0.34, color='#0072B2', label='Near-peak episodes')
    a.bar(recounts.reintensification_event_count + 0.18, recounts['count'], width=0.34, color='#E69F00', label='Reintensification events')
    a.set_xticks([0, 2, 4, 6, 8])
    a.set_xlabel('Number per cyclone')
    a.set_ylabel('Cyclones')
    title(a, 'b', 'Complete event-count distributions')
    a.legend(frameon=False, loc='upper right')
    tidy(a)
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, ['a: Single, no reintens.', 'a: Multiple near-peak', 'a: Single, reintens.'])], loc='center', bbox_to_anchor=(0.53, 140 / 215), ncol=3, frameon=False, handlelength=0.9, columnspacing=1)
    a = axes[2]
    q = dur[dur.scope.isin(scopes) & dur.metric.eq('B') & dur.statistic.eq('mean')]
    assert set(q.scope) == set(scopes)
    for i, s in enumerate(scopes):
        r = q[q.scope.eq(s)].iloc[0]
        e = float(r.estimate)
        a.errorbar(e, i, xerr=[[e - r.ci_low], [r.ci_high - e]], fmt='o', ms=2.8, c='#333', capsize=1.5, elinewidth=0.7)
    a.set_yticks(range(len(scopes)), scopemap.display_code)
    a.invert_yaxis()
    a.set_xlabel('Mean bridge duration (h)')
    title(a, 'c', 'Prespecified duration sensitivities')
    tidy(a)
    a = axes[3]
    q = tr[tr.scope.isin(scopes) & tr.metric.isin(['L', 'B'])]
    for i, s in enumerate(scopes):
        for k, m in enumerate(['L', 'B']):
            r = q[q.scope.eq(s) & q.metric.eq(m)].iloc[0]
            e = float(r.estimate)
            a.errorbar(e, i + (k - 0.5) * 0.22, xerr=[[e - r.ci_low], [r.ci_high - e]], fmt=['o', 's'][k], ms=2.5, c=['#0072B2', '#333'][k], capsize=1.4, elinewidth=0.7)
    a.axvline(0, c='.75', lw=0.6)
    a.set_yticks(range(len(scopes)), scopemap.display_code)
    a.set_ylim(len(scopes) - 0.4, -0.6)
    a.set_xlabel('Covered-duration trend (h/decade)')
    title(a, 'd', 'Later and bridge trend sensitivities')
    tidy(a)
    fig.legend(handles=[Line2D([0], [0], c='#0072B2', marker='o', lw=0.7, label='Later L'), Line2D([0], [0], c='#333', marker='s', lw=0.7, label='Bridge B')], loc='center', bbox_to_anchor=(0.8, 70 / 215), ncol=2, frameon=False)
    for k, contrast in enumerate(['F_minus_U', 'U_minus_R']):
        a = axes[4 + k]
        q = loo[loo.contrast.eq(contrast)]
        for i, v in enumerate(VAR):
            vals = q[q.variable.eq(v)].change_from_main_sd.to_numpy(float)
            a.scatter(vals, np.full(len(vals), i), s=2, c='#0072B2', alpha=0.12, edgecolors='none', rasterized=True)
        a.axvline(0, c='.65', lw=0.6)
        a.set_yticks(range(6), VL)
        a.set_ylim(5.6, -0.6)
        a.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value / 0.0001:g}'))
        a.xaxis.set_major_locator(plt.MaxNLocator(4))
        a.set_xlabel('Deletion change ($10^{-4}$ SD)')
        title(a, ['e', 'f'][k], ['Influence: mean F − mean U', 'Influence: mean U − mean R'][k])
        tidy(a)
    j.save(fig, axes, [['a', 'b'], ['c', 'd'], ['e', 'f']], [['a', 'c', 'e'], ['b', 'd', 'f']], '')
def render():
 sf_d()

