from pathlib import Path
from display_format import format_number, nice_symmetric_limit
import os,sys,json,hashlib,re,zipfile,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm,Normalize,LogNorm
from matplotlib.patches import Circle,Wedge,Rectangle,FancyArrowPatch,Ellipse,Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PIL import Image
BLUE=TEMP='#167B9E';ORANGE=FINAL='#CC7445';INK='#233A46';MUTED='#677B86'
TEAL,PURPLE,GREY='#2C8078','#9069A3','#7F9099';GREEN='#418D79'
STAGES={'D':'#437F9C','E':'#428C79','F':'#BD9145','Res':'#AB685E'}
CMAP=LinearSegmentedColormap.from_list('signed_native',[ORANGE,'#FAFAF7',BLUE])
WIDTH_MM=180; WIDTH_IN=180/25.4; GRID='#DDE3E7'; LEDGER=[]; FILES={}; CAPTIONS={}
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'savefig.facecolor':'white'})

BLUE, ORANGE, INK = ('#167B9E', '#CC7445', '#233A46')
TEAL, PURPLE, GREY = ('#2C8078', '#9069A3', '#7F9099')
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
CMAP = LinearSegmentedColormap.from_list('signed_native', [ORANGE, '#FAFAF7', BLUE])
WIDTH_MM = 180
FINAL_DPI = 900
PREVIEW_RESOLUTION = 180
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'npj-20261021-main1-6', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'legend.frameon': False, 'savefig.facecolor': 'white'})

def zero(ax):
    ax.axhline(0, color='#9BA5AA', lw=0.65, zorder=0)
    ax.grid(axis='y', color='#E8EDEF', lw=0.5, zorder=0)
    ax.set_axisbelow(True)

def title(ax, letter, text):
    ax.set_gid(letter)
    ax.set_title(text, loc='left', pad=13)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-20, 12), textcoords='offset points', fontsize=10, weight='bold', va='bottom')

def fig6(dpi):
    src = '../individual/common_prepost_paired_change_contrasts.csv'
    source = D / src
    if not source.exists():
        raise FileNotFoundError('Need verified real 78-SID pre/post records: ' + str(source))
    individuals = rd(src)
    required = {'sid', 'kind', 'D_pre', 'D_post'}
    assert required.issubset(individuals.columns)
    support = rd('../individual/common_prepost_support.csv')
    four = rd('../individual/common_prepost_four_cells.csv')
    summary = rd('direct/common_prepost.csv')
    sids = set(support.loc[support.Sstar.astype(str).str.lower().eq('true'), 'sid'])
    assert len(sids) == 78 and set(individuals.sid) == sids and (set(four.sid) == sids)
    assert len(individuals) == 234 and len(four) == 468
    assert set(individuals.units) == {'kg'} and set(four.units) == {'kg'}
    assert set(four.window) == {'pre_minus24_0h', 'post_0_24h'}
    assert set(four.layer) == {'all'} and set(four.sector) == {'all'} and (set(four.factor) == {1})
    checks = []
    for row in summary.itertuples():
        q = individuals[individuals.kind.eq(row.kind)]
        actual = float(q[row.quantity].mean())
        assert np.isclose(actual, row.estimate, rtol=2e-14, atol=0.02), (row.kind, row.quantity, actual, row.estimate)
        checks.append({'kind': row.kind, 'quantity': row.quantity, 'n': len(q), 'source_mean': actual, 'frozen_summary': row.estimate, 'absolute_difference_kg': actual - row.estimate})
    for kind in ['in', 'out', 'net']:
        q = individuals[individuals.kind.eq(kind)].set_index('sid').sort_index()
        assert len(q) == 78 and q.index.nunique() == 78
        for win, suffix in [('pre_minus24_0h', 'pre'), ('post_0_24h', 'post')]:
            f = four[four.kind.eq(kind) & four.window.eq(win)].set_index('sid').loc[q.index]
            for col in ['temporary', 'final']:
                assert np.array_equal(q[col + '_' + suffix].to_numpy(), f[col].to_numpy())
            assert np.array_equal(q['D_' + suffix].to_numpy(), f.difference.to_numpy())
    (REP / 'main1_6_Fig6_individual_verification.json').write_text(json.dumps({'sid_count': 78, 'kinds': ['in', 'out', 'net'], 'units': 'kg', 'summary_means_verified': checks, 'four_cells_exact_equal': True, 'support_78_exact_equal': True, 'windows': ['pre_minus24_0h', 'post_0_24h'], 'no_fitting_or_resampling': True}, indent=2), encoding='utf-8')
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 132 / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1])
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.1, wspace=0.42, hspace=0.42)
    a = fig.add_subplot(gs[0, :])
    d = fluxrows('direct/paired_flux_components.csv')
    names = ['humidity', 'relative_motion', 'difference']
    z = d[d.kind.eq('out') & d.layer.eq('all')].set_index('quantity').loc[names].reset_index()
    e, lo, hi = [z[k].to_numpy(float) / 1000000000000.0 for k in ['estimate', 'low', 'high']]
    a.bar(range(3), e, 0.5, color=[TEAL, PURPLE, INK], zorder=2)
    a.errorbar(range(3), e, yerr=[e - lo, hi - e], fmt='none', color=INK, capsize=3, lw=0.9, zorder=3)
    a.set(xticks=range(3), xticklabels=['Specific humidity', 'Relative normal motion', 'Total outward difference'], ylabel='Temporary − final\n(10¹² kg)')
    zero(a)
    title(a, 'a', 'Composition of the outward-exchange difference')
    record('Fig6', 'a', 'direct/paired_flux_components.csv', d[d.kind.eq('out')], 'All seven saved original and nested components retained in complete estimate table', 'kg / 10^12')
    for j, (kind, col, label) in enumerate([('out', PURPLE, 'Gross outward'), ('net', TEAL, 'Net inward')]):
        ax = fig.add_subplot(gs[1, j])
        q = individuals[individuals.kind.eq(kind)]
        assert len(q) == 78 and q.sid.nunique() == 78 and q[['D_pre', 'D_post']].notna().all().all()
        x, y = (q.D_pre.to_numpy(float) / 1000000000000.0, q.D_post.to_numpy(float) / 1000000000000.0)
        low = min(x.min(), y.min())
        high = max(x.max(), y.max())
        pad = (high - low) * 0.07
        lim = (low - pad, high + pad)
        ax.plot(lim, lim, color='#AEB8BD', lw=0.8, ls='--', zorder=0)
        ax.axhline(0, color='#D9DFE2', lw=0.6, zorder=0)
        ax.axvline(0, color='#D9DFE2', lw=0.6, zorder=0)
        ax.scatter(x, y, s=17, c=col, alpha=0.68, edgecolors='white', linewidths=0.35)
        ax.set(xlim=lim, ylim=lim, xlabel='Pre-event contrast (10¹² kg)', ylabel='Post-event contrast (10¹² kg)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 4, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 4, 5, 10]))
        title(ax, chr(98 + j), label)
        record('Fig6', chr(98 + j), src, q, 'One actual original SID per point; pre/post temporary-minus-final contrasts; no fitting or generated observations', 'kg / 10^12')
    q = rd('direct/common_prepost.csv')
    record('Fig6', 'b,c statistics', 'direct/common_prepost.csv', q, 'Complete common-support original means/components/D_pre/D_post/direct Theta and all saved confidence bounds', 'kg / 10^12')
    export(fig, 'Fig6', dpi)

def fluxrows(src):
    d = rd(src)
    return d[d.cohort.eq('original_post80') & d.window.eq('0_24h') & d.factor.eq(1) & d.sector.eq('all')]