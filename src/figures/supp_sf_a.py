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

def sf_a():
    j = Job('SF-A')
    qc = j.load('FP0044')
    annual = j.load('FP0120')
    season = j.load('FP0046')
    basin = j.load('FP0122')
    support = j.load('FP0152')
    j.load('FP0043')
    j.load('FP0045')
    fig = plt.figure(figsize=(7.08661417323, 8.46456692913))
    axes = []
    for row in range(3):
        for col in range(2):
            axes.append(axis(fig, [31, 117][col], [155, 85, 15][row], 56, 43, chr(97 + row * 2 + col)))
    a = axes[0]
    a.plot(annual.year, annual.candidate_n, c='.6', lw=1, label='Candidates')
    a.plot(annual.year, annual.retained_n, c='#0072B2', lw=1, label='Retained')
    a.set_ylabel('Cyclones per year')
    a.set_xlabel('Year')
    a.legend(frameon=False, fontsize=8, loc='lower left')
    title(a, 'a', 'Archive sample coverage')
    tidy(a)
    a = axes[1]
    order = ['no_tropical_storm_genesis', 'peak_gain_too_small_for_four_stage_framework', 'low_wind_valid_fraction', 'stage_duration_too_short', 'too_few_lifecycle_points', 'genesis_outside_latitude_limit', 'wind_gap_too_large', 'OK']
    labels = ['No TS genesis', 'Insufficient gain', 'Low wind coverage', 'Short stage', 'Few observations', 'Genesis latitude', 'Wind gap', 'Retained']
    for i, k in enumerate(order):
        a.barh(i, int(qc[qc.qc_reason.eq(k)]['count'].iloc[0]), color='#0072B2' if k == 'OK' else '.6', height=0.6)
    a.set_yticks(range(8), labels)
    a.invert_yaxis()
    a.set_xlabel('Cyclones')
    title(a, 'b', 'Frozen quality accounting')
    tidy(a)
    a = axes[2]
    q = season.set_index('basin_genesis').loc[BS]
    im = a.imshow(q[[str(i) for i in range(1, 13)]].to_numpy(float), aspect='auto', origin='upper', cmap='YlGnBu', vmin=0, vmax=40)
    a.set_xticks(range(12), range(1, 13))
    a.set_yticks(range(6), BS)
    a.set_xlabel('Genesis month')
    title(a, 'c', 'Saved within-basin seasonality')
    a.tick_params(length=2, pad=2)
    cb = fig.add_axes([31 / 180, 72 / 215, 56 / 180, 2.2 / 215])
    c = fig.colorbar(im, cax=cb, orientation='horizontal', ticks=[0, 20, 40])
    c.set_label('Cyclones within basin (%)', labelpad=1)
    a = axes[3]
    pivot = basin.pivot(index='year', columns='basin', values='fraction').reindex(columns=BS)
    a.stackplot(pivot.index, pivot.to_numpy().T, colors=[BC[b] for b in BS])
    a.set_ylim(0, 1)
    a.set_ylabel('Fraction of retained cyclones')
    a.set_xlabel('Year')
    title(a, 'd', 'Annual basin composition')
    tidy(a)
    fig.legend(handles=[Patch(color=BC[b], label=b) for b in BS], loc='center', bbox_to_anchor=(0.81, 71 / 215), ncol=3, frameon=False, handlelength=0.8, columnspacing=0.8)
    a = axes[4]
    a.plot(annual.year, annual.valid_wind_fraction, c='#009E73', lw=1, label='Valid wind')
    a.plot(annual.year, annual.E_fallback_fraction, c='#D55E00', lw=1, label='E fallback')
    a.set_ylim(-0.04, 1.1)
    a.set_ylabel('Fraction')
    a.set_xlabel('Year')
    a.legend(frameon=False, loc='center right')
    title(a, 'e', 'Wind coverage and E rule')
    tidy(a)
    a = axes[5]
    q = support[support.grouping.eq('batch')]
    pivot = q.pivot(index='variable', columns='source_batch', values='availability_rate').reindex(VAR)
    im = a.imshow(pivot.to_numpy(float), aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    a.set_yticks(range(6), VL)
    a.set_xticks(range(2), ['Legacy', 'Repair track'])
    title(a, 'f', 'Environmental availability')
    a.tick_params(length=2, pad=2)
    for i in range(6):
        for k in range(2):
            a.text(k, i, f'{pivot.iloc[i, k] * 100:.1f}%', ha='center', va='center', color='white' if pivot.iloc[i, k] > 0.7 else 'black')
    j.save(fig, axes, [['a', 'b'], ['c', 'd'], ['e', 'f']], [['a', 'c', 'e'], ['b', 'd', 'f']], '')
BS=list(BC)

def render():
 sf_a()

