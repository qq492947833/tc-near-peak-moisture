from .common import *
P=DATA/'recovery'
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'xtick.major.width': 0.65, 'ytick.major.width': 0.65, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
C = {'shear200_850': '#246F9B', 'midRH': '#C58137'}
LABEL = {'shear200_850': 'Shear −2 m/s', 'midRH': 'RH +5 pp'}
METHODS = ['gcomp', 'onestep', 'ipw']
MC = ['#246F9B', '#B67432', '#747C85']
MN = ['G-formula', 'One-step', 'Density weighting']
MK = ['o', 's', '^']
sources = ['effects_with_CI.csv', 'point_predictions.csv', 'cohort_recovered.csv', 'weight_and_bias_diagnostics.csv', 'calibration_deciles.csv', 'MNAR_sensitivity.csv', 'MNAR_policy_response_sensitivity.csv', 'physical_confounder_summary.csv', 'nonsynoptic_sensitivity/point_estimates.csv']

def rd(p):
    return pd.read_csv(P / p, keep_default_na=False, na_values=['', 'nan', 'NaN'])

def decorate(ax, label, title):
    ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(-29, 11), textcoords='offset points', weight='bold', fontsize=9, ha='left', va='bottom')
    ax.set_title(title, loc='left', pad=12, fontweight='normal')

def grid(nrows, ncols, height):
    fig, axs = plt.subplots(nrows, ncols, figsize=(7.2047244094, height / 25.4))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.09, top=0.93, wspace=0.44, hspace=0.72 if nrows == 3 else 0.62)
    return (fig, axs)

def curves(ax, df, outcome, kind):
    for policy in C:
        g = df[(df.policy == policy) & df.outcome.astype(str).eq(str(outcome))].sort_values('offset' if kind == 'policy' else 'MNAR_offset')
        x = g['offset' if kind == 'policy' else 'MNAR_offset']
        ax.plot(x, g.estimate, 'o-', color=C[policy], ms=3, lw=1.2, label=LABEL[policy])
    ax.axhline(0, color='#858C93', lw=0.7, ls='--')
    ax.axvline(0, color='#D6DADF', lw=0.7)
    ax.set_xlabel('Missing-outcome log-odds offset' if str(outcome) == '48' else 'Missing-outcome offset (kt)')
    ax.set_ylabel('Recovery contrast (pp)' if str(outcome) == '48' else '24 h wind contrast (kt)')

def si10():
    fig, axs = grid(3, 2, 212)
    ax = axs[0, 0]
    decorate(ax, 'a', 'Follow-up information at 48 h')
    counts = [int(cohort.recovery48.eq(1).sum()), int(cohort.recovery48.eq(0).sum()), int(cohort.recovery48.isna().sum())]
    ax.barh([2, 1, 0], counts, color=['#246F9B', '#8596A6', '#D2D8DE'])
    ax.set_yticks([2, 1, 0], ['Observed\nrecovery', 'Complete\nno recovery', 'Unknown'])
    ax.set_xlabel('Cyclones')
    ax.set_xlim(0, 1600)
    for y, v in zip([2, 1, 0], counts):
        ax.text(v + 25, y, str(v), va='center', fontsize=7)
    ax = axs[0, 1]
    decorate(ax, 'b', 'Probability of complete follow-up')
    g = pred[pred.policy.eq('shear200_850')]
    complete = g.R48.astype(str).str.lower().eq('true')
    for sel, col, lab in [(complete, '#246F9B', 'Complete'), (~complete, '#8D969E', 'Incomplete')]:
        ax.hist(g.loc[sel, 'g48'], bins=np.linspace(0, 1, 16), density=True, histtype='step', lw=1.1, color=col, label=lab)
    ax.set_xlabel('Out-of-fold completeness probability')
    ax.set_ylabel('Density')
    ax.legend(frameon=False, loc='upper left', fontsize=7)
    for out, ax, label, title in [(48, axs[1, 0], 'c', 'Recovery calibration'), (24, axs[1, 1], 'd', '24 h wind-change calibration')]:
        decorate(ax, label, title)
        for policy in C:
            g = cal[cal.policy.eq(policy) & cal.outcome.eq(out)]
            ax.plot(g.predicted * (100 if out == 48 else 1), g.observed * (100 if out == 48 else 1), 'o-', color=C[policy], ms=3, lw=1, label=LABEL[policy])
        lim = (0, 60) if out == 48 else (-45, 5)
        ax.plot(lim, lim, '--', color='#8C9298', lw=0.7)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('Predicted probability (%)' if out == 48 else 'Predicted wind change (kt)')
        ax.set_ylabel('Weighted observed (%)' if out == 48 else 'Weighted observed change (kt)')
    ax = axs[2, 0]
    decorate(ax, 'e', 'Observation-weight distribution')
    for policy in C:
        g = pred[pred.policy.eq(policy)]
        w = g.loc[g.R48.astype(str).str.lower().eq('true'), 'ipcw48_uncapped'].to_numpy()
        assert np.all(w > 0)
        z = np.sort(w)
        ax.step(z, np.arange(len(z), 0, -1) / len(z), where='post', color=C[policy], lw=1, label=LABEL[policy])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvline(20, color='#888F96', ls='--', lw=0.7)
    ax.set_xlabel('Uncapped inverse-observation weight')
    ax.set_ylabel('Fraction at or above weight')
    ax.set_ylim(0.0001, 1.2)
    ax.tick_params(axis='both', labelsize=8)
    ax = axs[2, 1]
    decorate(ax, 'f', 'Original nonsynoptic wind reports')
    for j, policy in enumerate(C):
        for k, method in enumerate(['gcomp', 'onestep']):
            a = effects[(effects.policy == policy) & effects.method.eq(method) & effects.outcome.eq('recovery48_pp')].estimate.iloc[0]
            b = non[(non.policy == policy) & non.method.eq(method) & non.outcome.eq('recovery48_pp')].estimate.iloc[0]
            x = j + (k - 0.5) * 0.2
            ax.plot([x, x], [a, b], color=C[policy], lw=1)
            ax.plot(x, a, 'o', mfc='white', mec=C[policy], ms=4)
            ax.plot(x, b, 's', color=C[policy], ms=4)
    ax.set_xticks([0, 1], ['Shear −2 m/s', 'RH +5 pp'])
    ax.set_ylabel('Recovery contrast (pp)')
    ax.set_ylim(-0.15, 4.2)
    ax.axhline(0, color='#888F96', ls='--', lw=0.7)
    ax.legend(handles=[Line2D([0], [0], marker='o', mfc='white', color='#7A838A', ls='', label='6 h records'), Line2D([0], [0], marker='s', color='#7A838A', ls='', label='+ Original nonsynoptic')], frameon=False, loc='upper right', fontsize=6.5)
    fig.legend([Line2D([0], [0], color=C[p], lw=1.4) for p in C], [LABEL[p] for p in C], loc='lower center', bbox_to_anchor=(0.5, 0.006), ncol=2, frameon=False)
    save(fig, 'MF-F')

def si11():
    fig, axs = grid(3, 2, 212)
    for ax, label, title, df, out, kind in [(axs[0, 0], 'a', 'Shared offset: recovery', mnar, 48, 'common'), (axs[0, 1], 'b', 'Policy-specific offset: recovery', pmnar, 48, 'policy'), (axs[1, 0], 'c', 'Shared offset: wind change', mnar, 24, 'common'), (axs[1, 1], 'd', 'Policy-specific offset: wind change', pmnar, 24, 'policy')]:
        curves(ax, df, out, kind)
        decorate(ax, label, title)
    strengths = np.array([0.5, 1, 2, 4])
    for out, ax, label, title in [(48, axs[2, 0], 'e', 'Unmeasured confounding: recovery'), (24, axs[2, 1], 'f', 'Unmeasured confounding: wind change')]:
        decorate(ax, label, title)
        for policy in C:
            benchmark = 'past_shear' if policy == 'shear200_850' else 'past_RH'
            z = conf[conf.policy.eq(policy) & conf.outcome.eq(out) & conf.benchmark.eq(benchmark)].iloc[0]
            y = z.point_effect - abs(z.mean_signed_bias_one_benchmark) * strengths ** 2
            ax.plot(strengths, y, 'o-', color=C[policy], ms=3, lw=1.2, label=LABEL[policy])
        ax.axhline(0, color='#858C93', ls='--', lw=0.7)
        ax.set_xticks(strengths, ['0.5', '1', '2', '4'])
        ax.set_xlabel('Association strength / measured benchmark')
        ax.set_ylabel('Bias-adjusted contrast (pp)' if out == 48 else 'Bias-adjusted contrast (kt)')
    fig.legend([Line2D([0], [0], color=C[p], lw=1.4) for p in C], [LABEL[p] for p in C], loc='lower center', bbox_to_anchor=(0.5, 0.006), ncol=2, frameon=False)
    save(fig, 'SF-K')
effects=rd(sources[0]);pred=rd(sources[1]);cohort=rd(sources[2]);diag=rd(sources[3]);cal=rd(sources[4]);mnar=rd(sources[5]);pmnar=rd(sources[6]);conf=rd(sources[7]);non=rd(sources[8])
def save(fig,name):export(fig,{'MF-F':'FigS20','SF-K':'FigS21'}[name])
def render():
 si10();si11()

