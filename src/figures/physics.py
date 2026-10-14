from .common import *
P=DATA/'physics'
def thermodynamics():return read('FigS8_panel_source.csv')
def budget_extra_sources():return read('FigS9_motion_panel_source.csv'),read('FigS9_850_windows_panel_source.csv')
def save(fig,name):export(fig,{'FigS8':'FigS13','FigS9':'FigS14'}[name])
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'DejaVu Sans'], 'font.size': 7.6, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
COL = {'internal': '#2B769B', 'final': '#B86B45', 'difference': '#36515F'}
PC = {'all_along': '#2B769B', 'all_fixed': '#B86B45', 'historical_mixed': '#7F8790'}
PL = {'all_along': 'Along track', 'all_fixed': 'Fixed centre', 'historical_mixed': 'Historical mixed'}

def read(name):
    return pd.read_csv(P / name, keep_default_na=False, na_values=['', 'nan', 'NaN'], low_memory=False)

def decorate(ax, label, title):
    ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(-26, 10), textcoords='offset points', fontsize=9, fontweight='bold', va='bottom', ha='left')
    ax.set_title(title, loc='left', pad=12)

def grid(height=150):
    fig, axs = plt.subplots(2, 3, figsize=(7.086614173, height / 25.4))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.87, wspace=0.5, hspace=0.63)
    return (fig, axs)

def figS8():
    data = thermodynamics()
    fig, axs = grid(147)
    specs = [('midRH', 'Relative humidity', 'RH difference (pp)'), ('PI_headroom', 'PI headroom', 'Headroom difference (m s⁻¹)'), ('PI_vmax', 'Potential intensity', 'PI difference (m s⁻¹)'), ('signed_chi', 'Signed entropy ratio', 'χ difference'), ('signed_entropy_deficit', 'Signed entropy deficit', 'Entropy difference (J kg⁻¹ K⁻¹)'), ('Tang_Emanuel_ventilation', 'Signed ventilation', 'Signed-index difference')]
    for i, (var, title, xlabel) in enumerate(specs):
        ax = axs.flat[i]
        decorate(ax, chr(97 + i), title)
        for j, policy in enumerate(PC):
            g = data[data.variable.eq(var) & data.policy.eq(policy)].set_index('estimand').loc[['pre_mean', 'change_0_24']]
            e = g.estimate.to_numpy()
            l = g.lo.to_numpy()
            h = g.hi.to_numpy()
            ax.errorbar(e, np.array([1, 0]) + (1 - j) * 0.2, xerr=[e - l, h - e], fmt=['o', 's', 'D'][j], color=PC[policy], ms=3.3, capsize=2, lw=1)
        ax.set_yticks([1, 0], ['Pre', 'Change'])
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel(xlabel)
        ax.axvline(0, ls='--', color='#959DA3', lw=0.7, zorder=0)
        ax.grid(axis='x', color='#E4E8EB', lw=0.5, zorder=0)
        if var == 'Tang_Emanuel_ventilation':
            ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2), useMathText=True)
    fig.legend([Line2D([0], [0], marker=['o', 's', 'D'][j], color=PC[p], lw=1) for j, p in enumerate(PC)], [PL[p] for p in PC], ncol=3, loc='upper center', bbox_to_anchor=(0.54, 0.995), frameon=False)
    save(fig, 'FigS8')

def figS9():
    budget = read('centre_budget_estimates.csv')
    native = read('centre_native_estimates.csv')
    motion, mfc = budget_extra_sources()
    ledger = []
    fig, axs = plt.subplots(3, 2, figsize=(7.086614173, 8.464566929))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.075, top=0.92, wspace=0.64, hspace=0.7)

    def forest(ax, data, key, cats, labels, title, xlabel, label, policies=('all_along', 'all_fixed')):
        decorate(ax, label, title)
        for j, policy in enumerate(policies):
            s = data[data.policy.eq(policy)].set_index(key).loc[cats]
            e = s.estimate.to_numpy()
            l = s.lo.to_numpy()
            h = s.hi.to_numpy()
            pos = np.arange(len(cats)) + (j - (len(policies) - 1) / 2) * 0.18
            ax.errorbar(e, pos, xerr=[e - l, h - e], fmt=['o', 's', 'D'][j], color=PC[policy], ms=3.3, capsize=2, lw=1)
            ledger.extend(s.reset_index().assign(panel=label).to_dict('records'))
        ax.set_yticks(range(len(cats)), labels)
        ax.set_ylim(len(cats) - 0.5, -0.5)
        ax.set_xlabel(xlabel)
        ax.axvline(0, color='#959DA3', ls='--', lw=0.7, zorder=0)
        ax.grid(axis='x', color='#E4E8EB', lw=0.5, zorder=0)
    fields = ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency']
    labs = ['Horizontal', 'Vertical', 'Residual', 'Net tendency']
    forest(axs[0, 0], budget[budget.estimand.eq('pre_mean')], 'field', fields, labs, '700 hPa: before departure', 'Tendency difference (g kg⁻¹ day⁻¹)', 'a')
    forest(axs[0, 1], budget[budget.estimand.eq('integral_0_24')], 'field', fields, labs, '700 hPa: following 24 h', 'Integrated difference (g kg⁻¹)', 'b')
    forest(axs[1, 0], budget[budget.estimand.eq('integral_0_24')], 'field', ['fixed_tendency', 'motion_term', 'moving_tendency'], ['Local change', 'Translation', 'Moving change'], 'Moving and fixed-frame tendency', 'Integrated difference (g kg⁻¹)', 'c')
    ax = axs[1, 1]
    decorate(ax, 'd', 'Velocity in fixed-centre advection')
    for velocity, color, style in [('contemporaneous', '#B86B45', '-'), ('departure_time', '#7F8790', '--')]:
        s = motion[motion.velocity.eq(velocity)].sort_values('lag_h')
        t = s.lag_h.to_numpy()
        ax.fill_between(t, s.lo.to_numpy(), s.hi.to_numpy(), color=color, alpha=0.12, linewidth=0)
        ax.plot(t, s.estimate.to_numpy(), lw=1.1, ls=style, color=color, label='c(t)' if velocity == 'contemporaneous' else 'c(0)')
        ledger.extend(s.assign(panel='d').to_dict('records'))
    ax.axhline(0, ls='--', color='#959DA3', lw=0.7, zorder=0)
    ax.axvline(0, ls='--', color='#959DA3', lw=0.7, zorder=0)
    ax.set_xticks([-24, -12, 0, 12, 24])
    ax.set_xlabel('Time from departure (h)')
    ax.set_ylabel('Advection difference (g kg⁻¹ day⁻¹)')
    ax.legend(frameon=False, loc='upper left', fontsize=6.5)
    forest(axs[2, 0], mfc, 'window', ['before', 'after'], ['Previous 24 h', 'Following 24 h'], '850 hPa: moisture-flux convergence', 'Integrated MFC difference (g kg⁻¹)', 'e')
    sub = native[native.variable.isin(['shear200_500', 'shear500_850']) & native.estimand.isin(['pre_mean', 'change_0_24'])].copy()
    sub['key'] = sub.variable + '|' + sub.estimand
    cats = ['shear200_500|pre_mean', 'shear500_850|pre_mean', 'shear200_500|change_0_24', 'shear500_850|change_0_24']
    forest(axs[2, 1], sub, 'key', cats, ['Upper: pre', 'Lower: pre', 'Upper: change', 'Lower: change'], 'Shear by atmospheric layer', 'Shear difference (m s⁻¹)', 'f', tuple(PC))
    fig.legend([Line2D([0], [0], marker=['o', 's', 'D'][j], color=PC[p], lw=1) for j, p in enumerate(PC)], [PL[p] for p in PC], ncol=3, loc='upper center', bbox_to_anchor=(0.54, 0.995), frameon=False)
    save(fig, 'FigS9')

def render():
 figS8();figS9()

