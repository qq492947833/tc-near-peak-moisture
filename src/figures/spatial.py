from .common import *
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import string
def axis_mm(fig, x, y, w, h, label):
    W, H = fig.get_size_inches() * 25.4
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.set_gid(label)
    return a

def finite_n(a):
    q = a[mask]
    return str(int(q.min())) if q.min() == q.max() else f'{int(q.min())}–{int(q.max())}'

def letter(a, label):
    a.annotate(label, xy=(0, 1), xycoords='axes fraction', xytext=(0, 3), textcoords='offset points', fontsize=9, fontweight='bold', ha='left', va='bottom', annotation_clip=False)

def map_axis(a, letter_id, val, lim, n, yticks=False, xticks=False):
    im = a.pcolormesh(X, Y, val, shading='nearest', cmap='RdBu_r', norm=Normalize(-lim, lim), rasterized=True)
    a.set_xlim(-1500, 1500)
    a.set_ylim(-1500, 1500)
    a.set_aspect('equal')
    a.set_xticks([-1000, 0, 1000])
    a.set_yticks([-1000, 0, 1000])
    a.tick_params(labelleft=yticks, labelbottom=xticks, pad=1.5)
    letter(a, letter_id)
    a.annotate('n=' + finite_n(n), xy=(1, 1), xycoords='axes fraction', xytext=(0, 3), textcoords='offset points', fontsize=8, ha='right', va='bottom', annotation_clip=False)
    return im

def cbar(fig, alias, x, y, w, h, limit, label, ticks):
    a = axis_mm(fig, x, y, w, h, alias)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap='RdBu_r'), cax=a, orientation='horizontal', ticks=ticks, extend='neither')
    cb.ax.tick_params(labelsize=8, pad=1, length=2)
    cb.ax.set_title(label, fontsize=8, pad=3)
    return a
def collect(*args,**kwargs):return None
def render():
 global A,B,mask,X,Y
 with np.load(DATA/'spatial/current117_fixed_pre_shear_anomaly_common_paired_composites.npz',allow_pickle=False) as z:A={k:z[k] for k in z.files}
 with np.load(DATA/'spatial/current117_fixed_pre_shear_raw_common_paired_composites.npz',allow_pickle=False) as z:B={k:z[k] for k in z.files}
 with np.load(DATA/'spatial/geometry.npz',allow_pickle=False) as z:mask=z['display_mask'];X=z['x_m']/1000;Y=z['y_m']/1000
 fig = plt.figure(figsize=(7.08661417323, 7.87401574803))
 ax = []
 starts = [17, 60, 103, 146]
 bottoms = [150, 110, 70, 30]
 titles = ['−24 h\nFixed t0 centre', '0 h\nActual centre', '+24 h\nActual centre', 'Premean\nRaw vs anomaly']
 for ci, title in enumerate(titles):
     fig.text((starts[ci] + 14) / 180, 188 / 200, title, ha='center', va='center', fontsize=8)
 scatter_rows = []
 for ri, (fi, hi) in enumerate([(0, 0), (0, 1), (3, 0), (3, 1)]):
     field = 'q700' if fi == 0 else 'z200'
     hemi = ['NH', 'SH'][hi]
     fig.text(2 / 180, (bottoms[ri] + 14) / 200, field + ' · ' + hemi, rotation=90, rotation_mode='anchor', ha='center', va='center', fontsize=8)
     for ci in range(4):
         lab = string.ascii_lowercase[ri * 4 + ci]
         a = axis_mm(fig, starts[ci], bottoms[ri], 28, 28, lab)
         ax.append(a)
         if ci < 3:
             si = ci + 1
             limit = 2 if fi == 0 else 40
             map_axis(a, lab, A['difference'][hi, si, fi], limit, A['n_sid'][hi, si, fi], yticks=ci == 0, xticks=ri == 3)
             collect('SF-I', lab, hi, si, fi, 'difference', A['difference'])
         else:
             raw = B['difference'][hi, 0, fi][mask]
             anom = A['difference'][hi, 0, fi][mask]
             limit = 2 if fi == 0 else 100
             assert np.nanmax(np.abs(np.r_[raw, anom])) <= limit
             a.scatter(raw, anom, s=2.3, c='#2e5266', alpha=0.35, edgecolors='none', rasterized=True)
             a.plot([-limit, limit], [-limit, limit], c='.55', lw=0.6, ls='--', zorder=1)
             a.axhline(0, color='.8', lw=0.5)
             a.axvline(0, color='.8', lw=0.5)
             a.set_xlim(-limit, limit)
             a.set_ylim(-limit, limit)
             a.set_aspect('equal')
             a.set_xticks([-limit, 0, limit])
             a.set_yticks([-limit, 0, limit])
             a.tick_params(pad=1.5)
             a.tick_params(axis='x', pad=4)
             letter(a, lab)
             for (iy, ix), r, t in zip(zip(*np.where(mask)), raw, anom):
                 scatter_rows.append(dict(figure='SF-I', panel=lab, hemisphere=hemi, field=str(A['fields'][fi]), x_km=X[iy, ix], y_km=Y[iy, ix], raw_common_difference=r, anomaly_common_difference=t, n_raw=B['n_sid'][hi, 0, fi, iy, ix], n_anomaly=A['n_sid'][hi, 0, fi, iy, ix]))
 fig.text(82 / 180, 23 / 200, 'Along fixed pre-shear (km)', ha='center', va='center', fontsize=8)
 fig.text(160 / 180, 23 / 200, 'Raw contrast', ha='center', va='center', fontsize=8)
 fig.text(136 / 180, 104 / 200, 'Anomaly contrast', rotation=90, rotation_mode='anchor', ha='center', va='center', fontsize=8)
 for x, lim, label, ticks in [(22, 2, 'q700 anomaly contrast\n(g/kg)', [-2, 0, 2]), (91, 40, 'z200 anomaly contrast\n(m)', [-40, 0, 40])]:
     cbar(fig, 'bar' + str(x), x, 10, 55, 2.5, lim, label, ticks)
 export(fig, 'FigS12')

