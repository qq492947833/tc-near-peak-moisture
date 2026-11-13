from display_format import format_number,format_group
from pathlib import Path
import json, hashlib
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd, matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection
from matplotlib.text import Annotation, Text
SRC = Path(__file__).resolve().parents[1] / 'input'
SOURCE = SRC
OLD = SRC
MASTER = True
COLLECT = []
RECORDS = []
C = json.loads((SRC / 'manuscript_figures.json').read_text(encoding='utf8'))
# Shared explicit illustrative input; original observations and internal Res code unchanged.
SCHEMATIC = pd.read_csv(SRC / 'lifecycle_schematic.csv')
SMETA = json.loads((SRC / 'lifecycle_schematic.json').read_text(encoding='utf8'))
C['figure1']['schematic_hours'] = SCHEMATIC.hours.tolist()
C['figure1']['schematic_wind_kt'] = SCHEMATIC.wind_kt.tolist()
C['literature_schematic']['hours'] = SCHEMATIC.hours.tolist()
C['literature_schematic']['wind_kt'] = SCHEMATIC.wind_kt.tolist()
assert SMETA['genesis_kt'] + .25 * (SMETA['lmi_kt'] - SMETA['genesis_kt']) == 51.25
assert SMETA['genesis_kt'] + .90 * (SMETA['lmi_kt'] - SMETA['genesis_kt']) == 93.5
P = C['palette']
INK = P['ink']
BLUE = P['return1']
ORANGE = P['return2']
PURPLE = P['return3']
GREY = '#809198'
BASINS = C['figure1']['basin_order']
MASTER_SIDS = set((SRC / 'main_sid.txt').read_text(encoding='utf8').splitlines()) if (SRC / 'main_sid.txt').exists() else set()

def tracked(p):
    return Path(p)

def read_table(p):
    d = pd.read_csv(p, keep_default_na=False, na_values=['', 'nan', 'NaN'])
    d.attrs['source'] = str(p)
    return d

def record(*args):
    pass

def save(fig, name, final):
    COLLECT.append((fig, name))
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

def literature(final):
    d = C['literature_schematic']
    x = np.asarray(d['hours'])
    v = np.asarray(d['wind_kt'])
    fig, axes = plt.subplots(4, 1, figsize=(180 / 25.4, d['height_mm'] / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.925, bottom=0.065, hspace=0.63)
    texts = ['Emanuel (2000)', 'Yokoyama & Takayabu (2008)', 'Song et al. (2024)', 'This study']
    for i, a in enumerate(axes):
        a.plot(x, v, lw=1.4, color=INK, marker='o', ms=2.4, zorder=4)
        a.set(xlim=(-8, 86), ylim=(30, 137), yticks=[35, 60, 85, 100], ylabel='Wind speed (kt)')
        a.axvline(42, ymin=0, ymax=0.7, color='#A6B4BA', lw=0.75, ls=(0, (2, 2)), zorder=1)
        title(a, chr(97 + i), texts[i])
        a.tick_params(labelbottom=True)
    a = axes[0]
    a.axvspan(0, 42, ymin=0.82, ymax=0.87, color=P['E'], lw=0)
    a.axvspan(42, 78, ymin=0.82, ymax=0.87, color=P['Res'], lw=0)
    a.text(21, 129, 'Before first LMI', ha='center', fontsize=8, color=P['E'])
    a.text(60, 129, 'After first LMI', ha='center', fontsize=8, color=P['Res'])
    a.text(42, 109, 'First LMI', ha='center', fontsize=7.5, color=GREY)
    a = axes[1]
    a.axhline(80, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    a.text(17, 83, 'V > 0.80 × peak wind', ha='center', fontsize=7.5, color=GREY)
    for lo, hi in [(36, 48), (60, 66)]:
        a.axvspan(lo, hi, ymin=0, ymax=0.76, color=P['F'], alpha=0.15, lw=0, zorder=0)
    for lo, hi, col in [(0, 36, P['E']), (36, 48, P['F']), (48, 54, P['Res']), (54, 60, P['E']), (60, 66, P['F']), (66, 78, P['Res'])]:
        a.add_patch(Rectangle((lo, 113), hi - lo, 4, facecolor=col, edgecolor='none'))
    # Open-up brackets cover all three stage strips in each segment.
    for lo, hi, lab in [(0,54,'Segment 1'),(54,78,'Segment 2')]:
        a.plot([lo,lo,hi,hi],[124,121,121,124],color=INK,lw=.65,clip_on=False)
        a.text((lo+hi)/2,129,lab,ha='center',va='center',fontsize=8)
    a.text(54, 49, 'Restart', ha='center', fontsize=7.5, color=P['E'])
    a = axes[2]
    a.axhline(95, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    a.text(13, 99, 'Within 5 kt of LMI', ha='center', fontsize=7.5, color=GREY)
    a.axvspan(36, 42, ymin=0, ymax=0.76, color=P['F'], alpha=0.2, lw=0, zorder=0)
    a.add_patch(Rectangle((36, 113), 6, 4, facecolor=P['F'], edgecolor='none'))
    a.text(39, 129, 'Interval containing first LMI', ha='center', fontsize=8, color=P['F'])
    a.text(66, 54, 'Later return retained\noutside selected interval', ha='center', fontsize=7.5, color=GREY)
    a = axes[3]
    parts = [('D', 0, 6, 'Development'), ('E', 6, 36, 'Intensification'), ('F', 36, 72, 'Mature'), ('Res', 72, 78, 'Resolution')]
    stage_band(a, parts, y0=113, height=4, label_y=125, fs=7.6)
    for value, lab in [(SMETA['genesis_kt'] + q * (SMETA['lmi_kt'] - SMETA['genesis_kt']), f'I = {q:.2f}') for q in [.25,.90]]:
        a.hlines(value, -2, 80, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
        a.text(17, value + (2 if value > 80 else -4), lab, ha='center', va='bottom' if value > 80 else 'top', fontsize=7.5, color=GREY)
    episode_boxes(a, [(36, 42), (60, 66)])
    a.annotate('', xy=(60, 90), xytext=(54, 69), arrowprops={'arrowstyle': '-|>', 'color': BLUE, 'lw': 1})
    a.text(56, 57, 'Reintensification', ha='center', fontsize=7.3, color=BLUE)
    a.set_xlabel('Time from 35-kt start (h)')
    a.set_xticks([0, 24, 48, 72])
    record('SI_literature_framework', 'a–d', pd.DataFrame({'hours': x, 'wind_kt': v}), 'Same deliberately illustrative wind history in every row; source-specific definitions and declared drawing adaptations')
    save(fig, 'SI_literature_framework', final)

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
    fig = ax.figure
    fig.canvas.draw()
    d = C['figure1']
    x = np.asarray(d['schematic_hours'])
    y = np.asarray(d['schematic_wind_kt'])
    out = []
    pad_mm = 0.5 + ax.lines[0].get_markersize() / 2 * 25.4 / 72
    pad_px = pad_mm / 25.4 * fig.dpi
    for lo, hi in d['episode_ranges_h']:
        take = (x >= lo) & (x <= hi)
        assert (y[take] >= 93.5).all()
        pts = ax.transData.transform(np.c_[x[take], y[take]])
        low = pts.min(axis=0) - pad_px
        high = pts.max(axis=0) + pad_px
        coords = ax.transData.inverted().transform([low, high])
        box = Rectangle(coords[0], *coords[1] - coords[0], facecolor='none', edgecolor='black', lw=0.9, ls=(0, (3, 2)), zorder=4)
        ax.add_patch(box)
        out.append({'qualifying_sample_hours': x[take].tolist(), 'qualifying_wind_kt': y[take].tolist(), 'frozen_episode_bounds_h': [lo, hi], 'display_rectangle_data': coords.tolist(), 'four_side_center_padding_mm': pad_mm, 'clear_margin_beyond_marker_mm': 0.5, 'interpolated_analysis_boundaries': False})
    return out

def build(dpi):
    literature(True)
    fig, name = COLLECT.pop()
    assert name == 'SI_literature_framework'
    before = scientific_state(fig, 'old_FigureS01')
    layout = compact(fig, [9, 6, 3, 0], 9)
    # Match main Fig.1a's data-to-physical mapping, not its scientific values.
    # Only this study's row needs a taller data area; other framework rows remain
    # in their accepted positions relative to one another and retain all marks.
    old_w, old_h = fig.get_size_inches() * 25.4
    old_boxes = physical_boxes(fig)
    target_h = 47.05220256221779
    extra_h = target_h - old_boxes[3][3]
    assert extra_h > 0
    fig.set_size_inches(old_w / 25.4, (old_h + extra_h) / 25.4, forward=False)
    for i, (ax, box) in enumerate(zip(fig.axes, old_boxes)):
        ax.set_position([box[0] / old_w,
            (box[1] + (extra_h if i < 3 else 0)) / (old_h + extra_h),
            box[2] / old_w, (target_h if i == 3 else box[3]) / (old_h + extra_h)])
    fig.canvas.draw()
    (OUT / 'SCHEMATIC_DISPLAY_MAPPING_20261102.json').write_text(json.dumps({
        'reason': 'F1a and S1d already shared arrays and limits; S1d used a shorter data panel.',
        'before_axes_mm': old_boxes, 'after_axes_mm': physical_boxes(fig),
        'before_canvas_mm': [old_w, old_h],
        'after_canvas_mm': [old_w, old_h + extra_h],
        'data_changes': False, 'threshold_changes': False,
        'matching_main_F1a_height_mm': target_h,
        'recommended_docx_width_EMU': 4829995,
        'docx_width_reason': 'Use the current main F1 display width for equal final physical coordinates.'
    }, indent=2), encoding='utf8')
    c, d = fig.axes[2:]
    texts = [t for t in c.texts if t.get_text() == 'Later return retained\noutside selected interval']
    assert len(texts) == 1
    texts[0].remove()
    remove_episode_rectangles(d)
    removed = remove_events(d, {'Reintensification'})
    x = np.asarray(C['literature_schematic']['hours'])
    v = np.asarray(C['literature_schematic']['wind_kt'])
    assert np.all(np.diff(x) > 0)
    assert v[x == 54].item() == 75 and v[x == 54].item() < 80
    guide_receipt = guide(d, 'Reintensification', 57, float(np.interp(57, x, v)), 57, BLUE)
    assert all((np.array_equal(ax.lines[0].get_xydata(), np.c_[x, v]) for ax in fig.axes))
    changes = {'layout': layout, 'removed_c_text': 'Later return retained outside selected interval', 'removed_d_episode_rectangles': 2, 'd_event_guide': guide_receipt, 'all_panels_same_deep_curve': True, 'valley_kt': 75, 'maturity_threshold_kt': 80, 'authors_and_windows_unchanged': True}
    assert before == scientific_state(fig, 'old_FigureS01')
    (OUT / 'requested_changes.json').write_text(json.dumps(changes, indent=2), encoding='utf8')
    emit(fig)
