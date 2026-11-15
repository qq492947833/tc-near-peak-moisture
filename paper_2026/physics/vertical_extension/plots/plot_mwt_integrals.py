"""Plot saved MWT time-integral statistics; no fitting or resampling.

Example: python plot_mwt_integrals.py --output generated/plots
The standalone comparison and the two-panel SI image use native vector marks.
The Fig. 6 replacement tile is rendered directly at the original pixel density.
"""
from pathlib import Path
import argparse
import json
import os


def load_data(path):
    import pandas as pd
    data = pd.read_csv(path, float_precision="round_trip")
    assert len(data) == 9
    assert set(data.variant) == {'joint_with_transfer'}
    assert set(data.comparison) == {'rise_rate_integral'}
    assert set(data.n_sid) == {122} and set(data.n_events) == {132}
    assert set(data.window) == {1} and set(data.region) == {'r0_300'}
    assert set(data.level.astype(str)) == {'500-850'}
    assert (data.ci_lower <= data.estimate).all() and (data.estimate <= data.ci_upper).all()
    return data


def configure_style():
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5,
        'axes.labelsize': 8.5, 'axes.titlesize': 9.2,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': .65, 'xtick.major.width': .65,
        'ytick.major.width': .65, 'xtick.major.size': 3,
        'ytick.major.size': 3, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'mathtext.default': 'regular',
    })


def draw_panel(ax, rows, panel, horizontal=False):
    import numpy as np
    colors = ['#167B9E', '#428C79', '#233A46'] if panel in ('a', 'b') else ['#CC7445', '#437F9C', '#233A46']
    labels = ['Horizontal', 'Vertical', 'Total'] if panel in ('a', 'b') else ['APE-to-KE\nconversion', 'Vertical\nKE input', 'Total KE\ncanonical\ntransfer']
    rows = rows.sort_values('row')
    vals = rows.estimate.to_numpy(float)
    err = np.vstack([vals - rows.ci_lower.to_numpy(float), rows.ci_upper.to_numpy(float) - vals])
    for i, (value, color) in enumerate(zip(vals, colors)):
        if horizontal:
            ax.errorbar(value, 2-i, xerr=err[:, i:i+1], fmt='o', color=color,
                        markersize=4.5, capsize=3, capthick=1, elinewidth=1.2, zorder=3)
        else:
            ax.errorbar(i, value, yerr=err[:, i:i+1], fmt='o', color=color,
                        markersize=4.5, capsize=3, capthick=1, elinewidth=1.2, zorder=3)
    ax.spines[['top', 'right']].set_visible(False)
    ranges = {'a': (-35, 25), 'b': (-3, 8), 'c': (-18, 40)}
    ticks = {'a': [-30, -15, 0, 15], 'b': [-2, 0, 2, 4, 6, 8], 'c': [-15, 0, 15, 30]}
    if horizontal:
        ax.axvline(0, color='#9CAAB0', linestyle=(0, (3, 3)), linewidth=.8, zorder=0)
        ax.set(yticks=[2, 1, 0], yticklabels=labels, ylim=(-.55, 2.55),
               xlim=ranges[panel], xticks=ticks[panel])
        ax.set_xlabel('Time integral (J kg⁻¹)', labelpad=4)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.spines['left'].set_visible(False)
    else:
        ax.axhline(0, color='#9CAAB0', linestyle=(0, (3, 3)), linewidth=.8, zorder=0)
        ax.set(xticks=[0, 1, 2], xticklabels=labels, xlim=(-.55, 2.55),
               ylim=ranges[panel], yticks=ticks[panel])
        ax.set_ylabel('Time integral (J kg⁻¹)', labelpad=4)
        ax.tick_params(axis='x', length=0, pad=6)
    return ax


def export_figure(fig, path, dpi=900, tiff=True):
    from PIL import Image
    import matplotlib.pyplot as plt
    outputs = []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        target = path.with_suffix('.'+extension)
        fig.savefig(target, dpi=dpi)
        outputs.append(str(target))
    if tiff:
        with Image.open(path.with_suffix('.png')) as im:
            im.convert('RGB').save(path.with_suffix('.tiff'), compression='tiff_lzw', dpi=(dpi, dpi))
        outputs.append(str(path.with_suffix('.tiff')))
    geometry = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        geometry.append({'axes': list(ax.get_position().bounds),
                         'tightbbox_pixels': list(ax.get_tightbbox(renderer).bounds),
                         'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel(),
                         'title': ax.get_title()})
    path.with_suffix('.geometry.json').write_text(json.dumps({'size_inches': list(fig.get_size_inches()), 'dpi': dpi, 'panels': geometry}, indent=2), encoding='utf-8')
    plt.close(fig)
    return outputs


def make_figures(data, output, dpi=900):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    titles = {'a': 'KE canonical transfer', 'b': 'Dry APE canonical transfer', 'c': 'Selected KE terms'}
    files = []
    fig = plt.figure(figsize=(180/25.4, 77/25.4))
    for panel, pos in zip('abc', [[.085,.26,.235,.60],[.415,.26,.235,.60],[.745,.26,.235,.60]]):
        ax = fig.add_axes(pos)
        draw_panel(ax, data[data.panel == panel], panel)
        if panel == 'c':
            ax.set_xlim(-.4, 2.4)
        ax.set_title(titles[panel], loc='left', pad=10, fontsize=9)
        fig.text(pos[0]-.07, .91, panel, weight='bold', fontsize=10.5)
    files += export_figure(fig, output/'MWT_vertical_three_panel', dpi)

    # S29 contains the two new supplemental diagnostics, retaining readable type.
    fig = plt.figure(figsize=(180/25.4, 79/25.4))
    for letter, panel, pos in zip('ab', 'bc', [[.09,.25,.365,.61],[.59,.25,.365,.61]]):
        ax = fig.add_axes(pos)
        draw_panel(ax, data[data.panel == panel], panel)
        ax.set_title(titles[panel], loc='left', pad=10)
        fig.text(pos[0]-.065, .92, letter, weight='bold', fontsize=10.5)
    files += export_figure(fig, output/'formal_stage'/'Figure_S29', dpi)

    # Exactly the original Fig6b whole-panel tile: 3157 x 1780 pixels at900dpi.
    # Other old panels are preserved at their original pixels by the composition step.
    fig = plt.figure(figsize=(3157/dpi,1780/dpi), dpi=dpi)
    ax = fig.add_axes([.245,.23,.70,.60])
    draw_panel(ax, data[data.panel == 'a'], 'a', horizontal=True)
    fig.text(.218,.882,'KE canonical transfer',fontsize=9.2)
    fig.text(.034,.918,'b',weight='bold',fontsize=10.5)
    files += export_figure(fig, output/'Fig6b_new_tile', dpi, tiff=False)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent/'plot_source.csv')
    parser.add_argument('--output', type=Path, default=Path.cwd()/'generated'/'plots')
    parser.add_argument('--dpi', type=int, default=900)
    args = parser.parse_args()
    assert args.dpi == 900, 'Main tile preserves original900dpi geometry.'
    for key in ['MPLCONFIGDIR','TMPDIR','TEMP','TMP']:
        os.environ.setdefault(key,str(args.output.resolve().parent/'tmp'))
    (args.output.resolve().parent/'tmp').mkdir(parents=True, exist_ok=True)
    configure_style()
    files = make_figures(load_data(args.data), args.output, args.dpi)
    print(json.dumps({'files':files},indent=2))


if __name__ == '__main__':
    main()
