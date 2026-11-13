from display_format import format_number,format_group
from pathlib import Path
import os, sys, json, hashlib, re, zipfile, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, LogNorm
from matplotlib.patches import Circle, Wedge, Rectangle, FancyArrowPatch, Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PIL import Image
BLUE = TEMP = '#167B9E'
ORANGE = FINAL = '#CC7445'
INK = '#233A46'
MUTED = '#677B86'
TEAL, PURPLE, GREY = ('#2C8078', '#9069A3', '#7F9099')
GREEN = '#418D79'
STAGES = {'D': '#437F9C', 'E': '#428C79', 'F': '#BD9145', 'Res': '#AB685E'}
CMAP = LinearSegmentedColormap.from_list('signed_native', [ORANGE, '#FAFAF7', BLUE])
WIDTH_MM = 180
WIDTH_IN = 180 / 25.4
GRID = '#DDE3E7'
LEDGER = []
FILES = {}
CAPTIONS = {}
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'savefig.facecolor': 'white'})
STYLE = {'temporary': '#287FA0', 'final': '#BC693F', 'difference': '#33454F', 'windows': ['#6D84A7', '#337C7D', '#A87846'], 'dpi': 900, 'width_mm': 180}
LABELS = {'KE': 'Kinetic energy', 'APE': 'Dry available potential energy', 'C_A_to_K': 'APE → KE conversion', 'vertical_input_K_Wkg': 'Layer KE input', 'vertical_input_A_Wkg': 'Layer APE input', 'vertical_input_P_Wkg': 'Layer pressure-flux input'}
UNITS = {'KE': 'J kg⁻¹', 'APE': 'J kg⁻¹', 'C_A_to_K': 'W kg⁻¹', 'vertical_input_K_Wkg': 'W kg⁻¹', 'vertical_input_A_Wkg': 'W kg⁻¹', 'vertical_input_P_Wkg': 'W kg⁻¹'}
PRIMARY = ['KE', 'APE', 'C_A_to_K', 'vertical_input_K_Wkg', 'vertical_input_A_Wkg', 'vertical_input_P_Wkg']

def primary_mask(frame):
    mask = frame.metric.isin(PRIMARY) & frame.window.eq(1)
    if 'scheme' in frame:
        mask &= frame.scheme.eq('main')
    if 'region' in frame:
        mask &= frame.region.eq('r0_300')
    return mask & (frame.metric.isin(PRIMARY[:3]) & frame.level.eq('700') | frame.metric.isin(PRIMARY[3:]) & frame.level.eq('500-850'))

def display_intervals(frame, lower='lower', upper='upper', n='n_sid'):
    """Presentation copy only; original bootstrap outputs remain untouched."""
    copy = frame.copy()
    copy['interval_interpretable'] = copy[n].ge(2) & np.isfinite(copy[lower]) & np.isfinite(copy[upper])
    copy['display_lower'] = copy[lower].where(copy.interval_interpretable)
    copy['display_upper'] = copy[upper].where(copy.interval_interpretable)
    copy['evidence_status'] = np.select([copy[n].eq(0), copy[n].eq(1), copy.interval_interpretable], ['unavailable', 'one SID descriptive value; interval not interpretable', 'pointwise SID sampling interval'], default='interval unavailable')
    return copy

def setup():
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.6, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'axes.unicode_minus': True})
    return (plt, None)

def decorate(ax, title, letter=None):
    from matplotlib.ticker import MaxNLocator
    ax.set_title(title, loc='left', pad=7)
    if letter:
        ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-18, 8), textcoords='offset points', fontweight='bold', fontsize=9, va='bottom')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(length=2.5, width=0.6, pad=2)

def interval(ax, estimate, lower, upper, n, y=0, color=None):
    color = color or STYLE['difference']
    if not np.isfinite(estimate):
        return
    if n >= 2 and np.isfinite(lower) and np.isfinite(upper):
        ax.hlines(y, lower, upper, color=color, lw=1.1)
        ax.plot([lower, upper], [y, y], '|', color=color, ms=4)
    ax.plot(estimate, y, 'o', ms=3.8, color=color, mfc=color if n >= 2 else 'white')
import types
inputs = types.SimpleNamespace(PRIMARY=PRIMARY, primary_mask=primary_mask, display_intervals=display_intervals)

def draw_levels(frames):
    import matplotlib.colors as colors
    from matplotlib.ticker import FixedLocator,FuncFormatter
    from display_format import exponent_text
    plt,_=setup()
    fig=plt.figure(figsize=(180/25.4,173/25.4))
    axes=[];ids=[];color_axes=[];pre=frames['pre']
    positions=[.105,.405,.705];width=.245
    for i,metric in enumerate(inputs.PRIMARY[:3]):
        ax=fig.add_axes([positions[i],.655,width,.245]);axes.append(ax);ids.append('abc'[i])
        data=pre[pre.scheme.eq('main')&pre.region.eq('r0_300')&pre.metric.eq(metric)]
        levels=['500','700','850'];array=data.pivot(index='level',columns='window',values='estimate').reindex(index=levels,columns=[0,1,2]).to_numpy(float)
        limit=[20.0,5.0,2e-4][i]
        assert np.nanmax(abs(array))<=limit,'Display norm would clip frozen values'
        cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#D9DDDF')
        im=ax.imshow(np.ma.masked_invalid(array),origin='upper',aspect='auto',cmap=cmap,norm=colors.TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
        exponent=-5 if i==2 else 0
        for y in range(3):
            for x in range(3):
                v=array[y,x];text='NA'if not np.isfinite(v)else format_number(v/(10**exponent),decimals=4 if i==2 else 3).replace('-','−')
                ax.text(x,y,text,ha='center',va='center',fontsize=7,color='black')
        ax.set_xticks([0,1,2],['>16 d','2–16 d','<2 d']);ax.set_yticks([0,1,2],levels)
        ax.set_ylabel('Pressure (hPa)'if i==0 else'')
        title=['Kinetic energy','Dry available\\npotential energy','APE → KE conversion'][i].replace('\\n','\n')
        decorate(ax,title,'abc'[i]);ax.set_yticks([0,1,2],levels)
        ca=fig.add_axes([positions[i],.595,width,.009],label='<colorbar>')
        cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[-limit,0,limit]);cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7,length=2,pad=2)
        cb.set_label('Δ '+UNITS[metric]+(' ('+exponent_text(exponent)+')'if exponent else''),fontsize=7,labelpad=2)
        cb.formatter=FuncFormatter(lambda v,p,e=exponent:format_number(v/10**e,decimals=3));cb.update_ticks()
        if exponent:
            ax.set_title(title+'\nCells '+exponent_text(exponent),loc='left',pad=7)
            cb.formatter=FuncFormatter(lambda v,p,e=exponent:format_number(v/10**e,decimals=3));cb.update_ticks()
        color_axes.append(ca)
    # d is one grouped forest display: each metric row has an explicitly labelled
    # native-unit scale. Window positions are categorical, never a value offset.
    row_positions=[.410,.260,.110]
    exponents=[-4,-5,-4]
    xlimits=[(-2.05e-4,.15e-4),(-1.75e-5,4.75e-5),(-2.5e-4,6.8e-4)]
    ticks=[[-2e-4,-1.5e-4,-1e-4,-.5e-4,0],[-1.5e-5,0,1.5e-5,3e-5,4.5e-5],[-2e-4,0,2e-4,4e-4,6e-4]]
    for i,metric in enumerate(inputs.PRIMARY[3:]):
        ax=fig.add_axes([.36,row_positions[i],.59,.092]);axes.append(ax);ids.append('d'+str(i+1))
        data=pre[pre.scheme.eq('main')&pre.region.eq('r0_300')&pre.metric.eq(metric)&pre.level.eq('500-850')].sort_values('window')
        ax.axvline(0,color='#A6ADB1',lw=.6,zorder=0)
        for j,r in enumerate(data.itertuples()):
            interval(ax,r.estimate,r.lower,r.upper,r.n_sid,y=(1-j)*.30,color=STYLE['windows'][j])
        ax.set(xlim=xlimits[i],ylim=(-.47,.47),yticks=[0],yticklabels=[LABELS[metric].replace('pressure-flux','pressure-flux\n')])
        ax.tick_params(axis='y',length=0,pad=14);ax.tick_params(axis='x',length=2.5,pad=2)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_major_locator(FixedLocator(ticks[i]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p,e=exponents[i]:format_number(v/10**e,decimals=2)))
        ax.set_xlabel('Δ W kg⁻¹ ('+exponent_text(exponents[i])+')',fontsize=7.5,labelpad=2)
    fig.text(.105,.535,'d',fontweight='bold',fontsize=9)
    fig.text(.145,.535,'Layer inputs, 500–850 hPa',fontsize=8)
    handles=[Line2D([0],[0],color=c,marker='o',ms=3,lw=1)for c in STYLE['windows']]
    fig.legend(handles,['>16 d','2–16 d','<2 d'],loc='center right',bbox_to_anchor=(.965,.543),ncol=3,frameon=False,fontsize=7,columnspacing=1.1,handlelength=1.4)
    return fig,dict(axes=axes,panel_ids=ids,exclude_axes=color_axes,row_groups=[['a','b','c']],column_groups=[['d1','d2','d3']])

def build():
    frames = {k: load(k) for k in ['pre']}
    for k, d in frames.items():
        if 'level' in d:
            d['level'] = d['level'].astype(str)
        table('Figure_S17', k, d, 'Frozen precomputed display source; selected below by original plotting statements')
    fig, layout = draw_levels(frames)
    for a, i in zip(layout['axes'], layout['panel_ids']):
        a.set_gid(i)
    export(fig, 'Figure_S17', **layout)


