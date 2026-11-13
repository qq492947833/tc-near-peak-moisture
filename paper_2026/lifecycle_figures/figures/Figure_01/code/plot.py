from pathlib import Path
from display_format import format_number, nice_symmetric_limit
import json,hashlib
from decimal import Decimal,ROUND_HALF_UP
import numpy as np,pandas as pd,matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection
from matplotlib.text import Annotation,Text
SRC=Path(__file__).resolve().parents[1]/'input'
SOURCE=SRC;OLD=SRC;MASTER=True;COLLECT=[];RECORDS=[]
C=json.loads((SRC/'manuscript_figures.json').read_text(encoding='utf8'))
# Shared explicit illustrative input; original observations and internal Res code unchanged.
SCHEMATIC = pd.read_csv(SRC / 'lifecycle_schematic.csv')
SMETA = json.loads((SRC / 'lifecycle_schematic.json').read_text(encoding='utf8'))
C['figure1']['schematic_hours'] = SCHEMATIC.hours.tolist()
C['figure1']['schematic_wind_kt'] = SCHEMATIC.wind_kt.tolist()
C['literature_schematic']['hours'] = SCHEMATIC.hours.tolist()
C['literature_schematic']['wind_kt'] = SCHEMATIC.wind_kt.tolist()
assert SMETA['genesis_kt'] + .25 * (SMETA['lmi_kt'] - SMETA['genesis_kt']) == 51.25
assert SMETA['genesis_kt'] + .90 * (SMETA['lmi_kt'] - SMETA['genesis_kt']) == 93.5
P=C['palette'];INK=P['ink'];BLUE=P['return1'];ORANGE=P['return2'];PURPLE=P['return3'];GREY='#809198'
BASINS=C['figure1']['basin_order']
MASTER_SIDS=set((SRC/'main_sid.txt').read_text(encoding='utf8').splitlines()) if (SRC/'main_sid.txt').exists() else set()
def tracked(p):return Path(p)
def read_table(p):
    d=pd.read_csv(p,keep_default_na=False,na_values=['','nan','NaN']);d.attrs['source']=str(p);return d
def record(*args):pass
def save(fig,name,final):COLLECT.append((fig,name))
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'mwt-addendum-manuscript-figures', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'legend.frameon': False, 'savefig.facecolor': 'white', 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
def title(ax, letter, label, pad=12):
    ax.set_gid(letter)
    ax.set_title(label, loc='left', pad=pad)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-21, pad), textcoords='offset points', fontsize=10, weight='bold', va='baseline')

def stage_band(ax, parts, y0=112, height=4, label_y=121, fs=7.6):
    for key, x0, x1, label in parts:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, height, facecolor=P[key], edgecolor='none', zorder=2))
        ax.text((x0 + x1) / 2, label_y, label, ha='center', va='center', fontsize=fs, color=P[key])

def episode_boxes(ax, ranges, threshold=93.5, top=103):
    for x0, x1 in ranges:
        ax.add_patch(Rectangle((x0, threshold), x1 - x0, top - threshold, facecolor='none', edgecolor='black', lw=0.9, ls=(0, (3, 2)), zorder=4))

def fig1(final):
    d = C['figure1']
    fig = plt.figure(figsize=(C['width_mm'] / 25.4, d['height_mm'] / 25.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.63, 0.68])
    fig.subplots_adjust(left=0.115, right=0.975, top=0.935, bottom=0.066, hspace=0.64)
    a, b, c = [fig.add_subplot(gs[i]) for i in range(3)]
    x = np.asarray(d['schematic_hours'])
    v = np.asarray(d['schematic_wind_kt'])
    vg = d['genesis_kt']
    peak = d['lmi_kt']
    ve = vg + d['progress_E'] * (peak - vg)
    vm = vg + d['progress_F'] * (peak - vg)
    pre = x <= d['F_onset_h']
    crossing_idx = np.flatnonzero(pre & ((v - vg) / (peak - vg) < 0.25))[-1] + 1
    assert d['E_onset_h'] == x[crossing_idx]
    for lo, hi in d['episode_ranges_h']:
        assert (v[(x >= lo) & (x <= hi)] >= vm).all()
    parts = [('D', 0, 6, 'Development'), ('E', 6, 36, 'Intensification'), ('F', 36, 72, 'Mature'), ('Res', 72, 78, 'Resolution')]
    stage_band(a, parts, y0=119, height=3.5, label_y=130)
    a.plot(x, v, color=INK, lw=1.45, marker='o', ms=2.2, zorder=3)
    for value, lab, tx in [(ve, 'I = 0.25', 18), (vm, 'I = 0.90', 18)]:
        a.hlines(value, -2, 80, color=GREY, ls=(0, (3, 2)), lw=0.7, zorder=0)
        a.text(tx, value + (2.4 if value > 80 else -8), lab, fontsize=7.5, ha='center', va='bottom', color=GREY)
    episode_boxes(a, d['episode_ranges_h'], vm)
    a.annotate('Temporary weakening', xy=(51, 89), xytext=(44, 64), fontsize=7.3, ha='center', va='center', color=BLUE, arrowprops={'arrowstyle': '-', 'color': BLUE, 'lw': 0.7})
    a.annotate('', xy=(60, 91), xytext=(54, 81), arrowprops={'arrowstyle': '-|>', 'color': BLUE, 'lw': 1.2})
    a.text(58, 76, 'Reintensification', fontsize=7.3, ha='center', color=BLUE)
    a.scatter([60], [96], s=20, color=BLUE, edgecolors='white', lw=0.5, zorder=6)
    a.annotate('Return to near-peak\nintensity', xy=(60, 96), xytext=(69, 104), fontsize=7.3, ha='center', va='bottom', color=BLUE, arrowprops={'arrowstyle': '-', 'color': BLUE, 'lw': 0.7})
    a.scatter([72], [80], s=20, color=ORANGE, edgecolors='white', lw=0.5, zorder=6)
    a.annotate('Resolution onset', xy=(72, 80), xytext=(73, 41), fontsize=7.3, ha='center', color=ORANGE, arrowprops={'arrowstyle': '-', 'color': ORANGE, 'lw': 0.7})
    a.set(xlim=(-8, 86), ylim=(30, 137), xticks=[0, 24, 48, 72], yticks=[35, 60, 85, 100], xlabel='Time from 35-kt start (h)', ylabel='Wind speed (kt)')
    title(a, 'a', 'Framework for identifying the full tropical cyclone life cycle')
    record('Fig1', 'a', pd.DataFrame({'hours': x, 'wind_kt': v}), 'Illustrative curve; operational progress thresholds; not an observed cyclone')
    import shapefile
    from shapely.geometry import LineString, box
    for sh in shapefile.Reader(str(OLD / 'coastline/ne_110m_coastline.shp')).shapes():
        points = np.asarray(sh.points)
        cuts = list(sh.parts) + [len(points)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            xx, yy = (points[start:end, 0] % 360, points[start:end, 1])
            for ids in np.split(np.arange(len(xx)), np.flatnonzero(abs(np.diff(xx)) > 180) + 1):
                if len(ids) < 2:
                    continue
                g = LineString(np.c_[xx[ids], yy[ids]]).intersection(box(0, -60, 360, 65))
                for seg in list(g.geoms) if hasattr(g, 'geoms') else [g]:
                    if seg.geom_type == 'LineString' and (not seg.is_empty):
                        b.plot(*np.asarray(seg.coords).T, color='#ACBCC3', lw=0.48, zorder=1)
    for lon in [30, 100, 180]:
        b.plot([lon, lon], [0, 65], color='#7D949E', lw=0.7, ls=(0, (3, 3)), zorder=0)
    for lon in [10, 135, 290]:
        b.plot([lon, lon], [-60, 0], color='#7D949E', lw=0.7, ls=(0, (3, 3)), zorder=0)
    b.axhline(0, color='#7D949E', lw=0.7, ls=(0, (3, 3)), zorder=0)
    events = read_table(SOURCE / ('all_return_events.csv' if MASTER else 'new_return_events.csv'))
    events = events[events.ledger_eligible].copy()
    if MASTER:
        assert set(events.sid).issubset(MASTER_SIDS)
    assert len(events) > 0 and events.return_order.ge(1).all()
    assert events.groupby('sid').return_order.apply(lambda x: sorted(x.tolist()) == list(range(1, len(x) + 1))).all()
    assert events[['return_lat', 'return_lon']].notna().all().all()
    for order, color, marker, size, label in [(1, BLUE, 'o', 22, '1st reintensification'), (2, ORANGE, 'D', 32, '2nd reintensification'), (3, PURPLE, '^', 48, '3rd and later')]:
        z = events[events.return_order.eq(order) if order < 3 else events.return_order.ge(3)]
        b.scatter(z.return_lon % 360, z.return_lat, s=size, c=color, marker=marker, edgecolors='white', lw=0.5, zorder=2 + order, label=label, alpha=0.9)
    for key, xx, yy in [('NA', 320, 40), ('EP', 235, 4), ('WP', 153, 38), ('NI', 65, 14), ('SI', 64, -43), ('SP', 208, -43)]:
        b.text(xx, yy, key, ha='center', va='center', weight='bold', fontsize=9, color='#68808B', zorder=2)
    b.set(xlim=(0, 360), ylim=(-60, 65), xticks=[0, 90, 180, 270, 360], xticklabels=['0°', '90°E', '180°', '90°W', '0°'], yticks=[-60, -30, 0, 30, 60], yticklabels=['60°S', '30°S', '0°', '30°N', '60°N'], facecolor='#F7FAFB')
    title(b, 'b', 'Global basin distribution of tropical cyclone reintensification')
    b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, handletextpad=0.4, columnspacing=1.0, fontsize=7.2)
    record('Fig1', 'b', events, 'Every qualifying later near-peak return at its true return coordinate; ordinal is within SID')
    z = read_table(SOURCE / 'basin_recurrence_proportions.csv')
    z = z[z['mode'].eq('new')].set_index('basin_genesis').loc[BASINS].reset_index()
    z.attrs['source'] = str(SOURCE / 'basin_recurrence_proportions.csv')
    counts = events.groupby('basin_genesis').sid.nunique()
    assert sum(z.eligible_SID) == (len(MASTER_SIDS) if MASTER else 2366)
    assert sum(z.recurrent_SID) == events.sid.nunique()
    assert all((counts.get(r.basin_genesis, 0) == r.recurrent_SID for r in z.itertuples()))
    assert np.allclose(z.percentage, 100 * z.recurrent_SID / z.eligible_SID, rtol=0, atol=1e-12)
    c.bar(range(6), z.percentage, width=0.56, color=BLUE, zorder=2)
    for i, r in enumerate(z.itertuples()):
        c.text(i, r.percentage + 0.24, format_number(r.percentage, decimals=2) + '%', ha='center', fontsize=8)
    c.set(xticks=range(6), xticklabels=[r.basin_genesis for r in z.itertuples()], ylabel='Recurrent cyclones (%)', ylim=(0, 8))
    c.set_axisbelow(True)
    title(c, 'c', 'Recurrence by genesis basin')
    record('Fig1', 'c', z, 'Unique recurrent SID / frozen main SID in genesis basin; percentages only; uncertainty retained in SI')
    save(fig, 'Fig1_addendum', final)
def array(a):
    a = np.asarray(np.ma.getdata(a))
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}

def line_state(line):
    return {'xy': array(line.get_xydata()), 'color': line.get_color(), 'lw': line.get_linewidth(), 'marker': line.get_marker(), 'ms': line.get_markersize(), 'ls': str(line.get_linestyle())}

def collection_state(item):
    return {'offsets': array(item.get_offsets()), 'sizes': array(item.get_sizes()), 'facecolors': array(item.get_facecolors()), 'edgecolors': array(item.get_edgecolors()), 'linewidths': array(item.get_linewidths()), 'paths': [array(p.vertices) for p in item.get_paths()], 'alpha': item.get_alpha(), 'zorder': item.get_zorder()}

def scientific_state(fig, name):
    result = {'curves': [], 'axes': []}
    for ax in fig.axes:
        result['axes'].append({'gid': ax.get_gid(), 'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim()), 'xticks': ax.get_xticks().tolist(), 'yticks': ax.get_yticks().tolist()})
    if name == 'old_Figure01':
        a, b, c = fig.axes
        result['curves'] = [line_state(a.lines[0])]
        result['map_lines'] = [line_state(line) for line in b.lines]
        result['events'] = [collection_state(item) for item in b.collections]
        result['proportions'] = [{'xy': list(p.get_xy()), 'height': p.get_height(), 'width': p.get_width(), 'color': list(p.get_facecolor())} for p in c.patches]
        result['proportion_labels'] = [(t.get_text(), t.get_fontsize()) for t in c.texts]
    else:
        result['curves'] = [line_state(ax.lines[0]) for ax in fig.axes]
        result['other_lines'] = [[line_state(line) for line in ax.lines[1:] if line.get_gid() != 'event-guide'] for ax in fig.axes]
        result['retained_shading'] = [[{'xy': list(p.get_xy()) if isinstance(p, Rectangle) else array(p.get_xy()), 'facecolor': list(p.get_facecolor())} for p in ax.patches if p.get_facecolor()[-1] > 0] for ax in fig.axes]
    return result

def physical_boxes(fig):
    w, h = fig.get_size_inches() * 25.4
    return [list(np.asarray(ax.get_position().bounds) * [w, h, w, h]) for ax in fig.axes]

def compact(fig, moves_mm, trim_mm):
    fig.canvas.draw()
    w, h = fig.get_size_inches() * 25.4
    before = physical_boxes(fig)
    fig.set_size_inches(w / 25.4, (h - trim_mm) / 25.4, forward=False)
    for ax, box, shift in zip(fig.axes, before, moves_mm):
        ax.set_axes_locator(None)
        ax.set_position([box[0] / w, (box[1] - shift) / (h - trim_mm), box[2] / w, box[3] / (h - trim_mm)])
    fig.canvas.draw()
    after = physical_boxes(fig)
    assert np.allclose(np.asarray(before)[:, 2:], np.asarray(after)[:, 2:], atol=1e-10)
    return {'before_mm': [w, h], 'after_mm': [w, h - trim_mm], 'axes_before_mm': before, 'axes_after_mm': after, 'physical_panel_sizes_preserved': True}

def remove_episode_rectangles(ax):
    selected = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_facecolor()[-1] == 0 and (p.get_edgecolor()[:3] == (0, 0, 0))]
    assert len(selected) == 2, len(selected)
    for p in selected:
        p.remove()

def remove_events(ax, expected):
    found = []
    for t in list(ax.texts):
        if t.get_text() in expected or (isinstance(t, Annotation) and t.get_text() == '' and (t.arrow_patch is not None)):
            found.append(t.get_text())
            t.remove()
    assert set(found) == set(expected) | {''}, found
    return found

def guide(ax, label, x, anchor_y, text_y, color):
    text = ax.text(x, text_y, label, ha='center', va='center', fontsize=7.3, color=color, zorder=5)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = text.get_window_extent(renderer)
    curve_px = ax.transData.transform((x, anchor_y))
    px_per_pt = fig.dpi / 72
    y0 = bounds.y1 + 1.5 * px_per_pt
    y1 = curve_px[1] - 1.5 * px_per_pt
    assert y1 > y0, (label, y0, y1)
    ends = ax.transData.inverted().transform([[curve_px[0], y0], [curve_px[0], y1]])
    line, = ax.plot(ends[:, 0], ends[:, 1], color=color, lw=0.7, ls='-', zorder=4, gid='event-guide')
    assert abs(ends[0, 0] - ends[1, 0]) < 1e-12
    return {'text': label, 'anchor_xy': [x, anchor_y], 'text_xy': [x, text_y], 'line_xy': ends.tolist(), 'line_width_pt': 0.7, 'arrow': False, 'font_pt': 7.3}

def padded_boxes(ax):
    # Display-only boxes enclose each complete piecewise-linear near-peak segment.
    # Frozen discrete episodes, thresholds and all recognition times are unchanged.
    fig = ax.figure
    fig.canvas.draw()
    d = C['figure1']
    x = np.asarray(d['schematic_hours'], dtype=float)
    y = np.asarray(d['schematic_wind_kt'], dtype=float)
    threshold = d['genesis_kt'] + d['progress_F'] * (d['lmi_kt'] - d['genesis_kt'])
    assert np.all(np.diff(x) > 0)
    eligible = y >= threshold
    cuts = np.flatnonzero(np.diff(np.r_[False, eligible, False].astype(int)))
    runs = list(zip(cuts[::2], cuts[1::2] - 1))
    assert [(float(x[i]), float(x[j])) for i, j in runs] == [tuple(v) for v in d['episode_ranges_h']]
    pad_mm = 0.5 + ax.lines[0].get_markersize() / 2 * 25.4 / 72
    pad_px = pad_mm / 25.4 * fig.dpi
    out = []
    for first, last in runs:
        # Intersections bound the displayed curve only, never event classification.
        lo = x[first]
        hi = x[last]
        if first > 0:
            lo = x[first - 1] + (threshold - y[first - 1]) * (x[first] - x[first - 1]) / (y[first] - y[first - 1])
        if last + 1 < len(x):
            hi = x[last] + (threshold - y[last]) * (x[last + 1] - x[last]) / (y[last + 1] - y[last])
        segment = np.vstack(([lo, threshold], np.c_[x[first:last + 1], y[first:last + 1]], [hi, threshold]))
        pts = ax.transData.transform(segment)
        low, high = pts.min(axis=0) - pad_px, pts.max(axis=0) + pad_px
        coords = ax.transData.inverted().transform([low, high])
        box = Rectangle(coords[0], *coords[1] - coords[0], facecolor='none', edgecolor='black', lw=0.9, ls=(0, (3, 2)), zorder=4)
        ax.add_patch(box)
        assert np.all(pts >= low) and np.all(pts <= high)
        assert np.allclose(pts.min(axis=0) - low, pad_px)
        assert np.allclose(high - pts.max(axis=0), pad_px)
        # The weakening trough must lie outside both display rectangles.
        valley = np.array([d['valley_h'], np.interp(d['valley_h'], x, y)])
        assert not np.all((valley >= coords[0]) & (valley <= coords[1]))
        out.append({'displayed_sample_hours': x[first:last + 1].tolist(), 'displayed_wind_kt': y[first:last + 1].tolist(), 'frozen_episode_range_h': [float(x[first]), float(x[last])], 'frozen_episode_ranges_h_unchanged': d['episode_ranges_h'], 'near_peak_threshold_kt': threshold, 'display_only_threshold_crossings_h': [float(lo), float(hi)], 'complete_display_segment': segment.tolist(), 'display_rectangle_data': coords.tolist(), 'four_side_center_padding_mm': pad_mm, 'clear_margin_beyond_marker_mm': 0.5, 'interpolated_analysis_boundaries': False, 'scope': 'One complete near-peak displayed segment per box; subthreshold bridge excluded; frozen classification unchanged'})
    assert len(out) == 2
    assert out[0]['display_rectangle_data'][1][0] < out[1]['display_rectangle_data'][0][0]
    return out

def build(dpi):
    fig1(True); fig,name=COLLECT.pop(); assert name=='Fig1_addendum'
    for text,label in zip(fig.axes[1].get_legend().get_texts(),['1st near-peak return','2nd near-peak return','3rd and later returns']):text.set_text(label)
    before=scientific_state(fig,'old_Figure01')
    layout=compact(fig,[15,7.5,0],15)
    a,b,c=fig.axes
    removed=remove_events(a,{'Temporary weakening','Reintensification','Return to near-peak\nintensity','Resolution onset'})
    remove_episode_rectangles(a)
    overlays=[item for item in a.collections if isinstance(item,PathCollection)];assert len(overlays)==2
    for item in overlays:item.remove()
    box_receipts=padded_boxes(a)
    x=np.asarray(C['figure1']['schematic_hours']);v=np.asarray(C['figure1']['schematic_wind_kt'])
    assert np.all(np.diff(x)>0)
    guides=[guide(a,'Temporary weakening',49,float(np.interp(49,x,v)),57,BLUE),
            guide(a,'Reintensification',57,float(np.interp(57,x,v)),65,BLUE),
            guide(a,'Resolution',72,80,42,ORANGE)]
    b.get_legend().set_bbox_to_anchor((.5,-.105),transform=b.transAxes)
    events=read_table(SOURCE/'all_return_events.csv');events=events[events.ledger_eligible]
    counts=events.return_order.value_counts().sort_index().to_dict()
    assert len(events)==133 and events.sid.nunique()==123 and list(counts.values())==[123,9,1]
    proportions=read_table(SOURCE/'basin_recurrence_proportions.csv');proportions=proportions[proportions['mode'].eq('new')]
    assert proportions.eligible_SID.sum()==2340 and set(proportions.basin_genesis)==set(BASINS)
    assert not events.return_basin.eq('SA').any() and not events.basin_genesis.eq('SA').any()
    main_cohort=read_table(SOURCE/'main_sid.csv')
    assert set(main_cohort.sid)==MASTER_SIDS and len(main_cohort)==2340 and not main_cohort.basin_genesis.eq('SA').any()
    b.text(330,-43,'SA',ha='center',va='center',weight='bold',fontsize=9,color='#68808B',zorder=2)
    for p in (OLD/'coastline').glob('ne_110m_coastline.*'):tracked(p)
    changes={'layout':layout,'removed_event_artists':removed,'restored_ordinary_sampling_points_h':[60,72],
             'near_peak_rectangles':box_receipts,'event_guides':guides,
             'legend_text_unchanged':True,'legend_anchor_y_before':-.14,'legend_anchor_y_after':-.105,
             'SA_label':'No eligible main-cohort SID or return event; explicit caption statement; no0/0bar',
             'return_events':133,'unique_SID':123,'ordinal_counts':counts,
             'BASIN_guides':'Schematic meridians retained; no artificial NA-EP dividing meridian; frozen BASIN field governs events',
             'proportion_data_unchanged':proportions.to_dict(orient='records')}
    
    assert before==scientific_state(fig,'old_Figure01')
    (OUT/'requested_changes.json').write_text(json.dumps(changes,indent=2),encoding='utf8')
    emit(fig)
