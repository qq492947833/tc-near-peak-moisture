from .common import *
def render():
 d=read_csv(DATA/'climate/P4_raw_climate_anomaly_CI_n.csv')
 plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6})
 fig, axs = plt.subplots(2, 2, figsize=(183 / 25.4, 165 / 25.4))
 fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.74, wspace=0.4, hspace=0.68)
 colors = {'D_to_E': '#2473a6', 'E_to_F': '#c35e17'}
 labels = {'D_to_E': 'D → E', 'E_to_F': 'E → F'}
 mapping = []
 for row, (field, title, unit, au) in enumerate([('q700', '700-hPa specific humidity', 'g kg⁻¹', 'g kg⁻¹'), ('r700', '700-hPa relative humidity', '%', 'pp')]):
     for col, quantities in enumerate([['raw_common', 'climate_common'], ['anomaly_common']]):
         ax = axs[row, col]
         panel = chr(97 + 2 * row + col)
         ax.axvline(0, c='.6', ls=':', lw=0.7)
         for bi, (b, c) in enumerate(colors.items()):
             for quantity in quantities:
                 g = d[(d.field == field) & (d.boundary == b) & (d.quantity == quantity)].sort_values('lag_h')
                 assert len(g) == 9
                 x = g.lag_h.to_numpy(float)
                 y = g.estimate.to_numpy(float)
                 lo = g.ci_low.to_numpy(float)
                 hi = g.ci_high.to_numpy(float)
                 assert np.all(lo <= y) and np.all(y <= hi)
                 ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
                 ax.plot(x, y, color=c, ls='--' if quantity == 'climate_common' else '-', marker='o' if bi == 0 else 's', ms=2.8, lw=1)
                 for z in g.to_dict('records'):
                     z['panel'] = panel
                     mapping.append(z)
             ax.text(0.01, 1.095 - bi * 0.095, labels[b] + ': n=' + str(int(g.n_sid.min())) + '–' + str(int(g.n_sid.max())), transform=ax.transAxes, color=c, fontsize=8, va='bottom')
         ax.set_title(title + '\n' + ('Observed and local climatology' if col == 0 else 'Observed − local climatology'), pad=30)
         ax.text(-0.2, 1.41, panel, transform=ax.transAxes, fontsize=10, weight='bold')
         ax.set_ylabel('Mean (' + unit + ')' if col == 0 else 'Mean anomaly (' + au + ')')
         ax.set_xlim(-25, 25)
         ax.set_xticks([-24, -12, 0, 12, 24])
         ax.set_xlabel('Lag from boundary (h)')
 fig.text(0.12, 0.98, 'Moisture trajectories and their climatological background', ha='left', va='top', fontsize=10.5, weight='bold')
 fig.text(0.12, 0.945, 'Common cells and SIDs within each lag; 1991–2020 month × UTC climatology', va='top', fontsize=8)
 fig.legend(handles=[Line2D([0], [0], color='.3', ls='-', label='Observed'), Line2D([0], [0], color='.3', ls='--', label='Local climatology')], loc='upper center', bbox_to_anchor=(0.53, 0.92), ncol=2, frameon=False, fontsize=8)
 export(fig,'FigS11')

