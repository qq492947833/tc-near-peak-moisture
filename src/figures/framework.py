from .common import *
BASE=DATA/'framework'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 7, 'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})

def main():
    pass
    d = pd.read_csv(BASE / 'A3_paired_contrasts.csv')
    fig, axs = plt.subplots(1, 3, figsize=(7.2047244094, 3.5826771654), gridspec_kw={'width_ratios': [1, 1, 1]})
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.27, top=0.8, wspace=0.52)
    a = axs[0]
    a.set_xlim(0, 2)
    a.set_ylim(0, 2)
    a.set_axis_off()
    entries = [(0, 1, 'A4-F', 'D / E / F / Res'), (1, 1, 'A4-R', 'D / E / R / Tail'), (0, 0, 'A3-F', 'DE / F / Res'), (1, 0, 'A3-R', 'DE / R / Tail')]
    for x, y, name, stages in entries:
        a.add_patch(Rectangle((x + 0.025, y + 0.05), 0.95, 0.82, facecolor='#EAF1F4' if x == 0 else '#F1F1EF', edgecolor='#B8C3C8', linewidth=0.6))
        a.text(x + 0.5, y + 0.59, name, ha='center', va='center', fontsize=7.5, fontweight='bold')
        a.text(x + 0.5, y + 0.29, stages, ha='center', va='center', fontsize=6)
    a.text(0.5, 1.98, 'Retain returns', ha='center', va='bottom', fontsize=6.5)
    a.text(1.5, 1.98, 'LMI episode', ha='center', va='bottom', fontsize=6.5)
    a.text(1, 0.0, 'Two independent factors', ha='center', va='top', fontsize=7)
    a.text(1, -0.36, '2,366 common-window SIDs\n123 different spans; 2,243 unchanged', ha='center', va='top', fontsize=6.5, linespacing=1.6)
    refs = ['coarse', 'time_midpoint', 'continuous_progress', 'continuous_age']
    labels = ['Pooled D/E', 'Midpoint', 'Progress (u, u²)', 'Age (a, a²)']
    for ax, var, title, unit, color in [(axs[1], 'X2_shear_ms', 'Deep-layer shear', 'm/s', '#366984'), (axs[2], 'X3_mid_rh_pct', 'Midlevel relative humidity', 'percentage points', '#AA6E3B')]:
        z = d[d.response == var].set_index('contrast').loc[['actual_DE-' + r for r in refs]]
        y = np.arange(4)
        x = z.delta_rmse.to_numpy()
        lo = z.rmse_ci_low.to_numpy()
        hi = z.rmse_ci_high.to_numpy()
        ax.axvline(0, color='#777777', linewidth=0.65, linestyle=(0, (2, 2)), zorder=0)
        ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt='o', markersize=4, color=color, elinewidth=1.1, capsize=2.8, capthick=0.8)
        ax.set_yticks(y, labels)
        ax.set_ylim(3.55, -0.55)
        ax.set_title(title, pad=18, fontweight='normal')
        ax.set_xlabel('ΔRMSE (' + unit + ')', labelpad=9)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.spines['left'].set_visible(False)
        if var == 'X2_shear_ms':
            ax.set_xlim(-0.0035, 0.0035)
            ax.set_xticks([-0.003, 0, 0.003], ['−0.003', '0', '0.003'])
        else:
            ax.set_xlim(-0.062, 0.182)
            ax.set_xticks([-0.05, 0, 0.05, 0.1, 0.15], ['−0.05', '0', '0.05', '0.10', '0.15'])
    for lab, ax in zip('abc', axs):
        ax.annotate(lab, xy=(0, 1), xycoords='axes fraction', xytext=(-5, 29), textcoords='offset points', ha='left', va='bottom', fontsize=9, fontweight='bold', annotation_clip=False)
    a.set_title('Controlled representation', pad=18, fontweight='normal')
    fig.text(0.666, 0.072, 'Actual D/E minus reference; negative favours actual D/E', ha='center', fontsize=6.5)
    fig.text(0.666, 0.026, '2,314 SIDs · 26,156 rows · 5 SID folds · 2,000 fixed-OOF resamples', ha='center', fontsize=6.5)
    export(fig,'FigS6')

render=main

