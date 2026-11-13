from pathlib import Path
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

BLUE, ORANGE, GREY = (TEMP, FINAL, '#8A969D')
GREEN, PURPLE = ('#418D79', '#8471A8')
POLICIES = ['all_along', 'all_fixed', 'historical_mixed']
PC = {'all_along': BLUE, 'all_fixed': ORANGE, 'historical_mixed': GREY}
PL = {'all_along': 'Along track', 'all_fixed': 'Fixed centre', 'historical_mixed': 'Mixed'}
BASINS = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP']
RINGS = ['100_300km', '300_800km']
RINGLABELS = ['100–300 km', '300–800 km']
LAYERS = ['500-600_hPa', '600-700_hPa', '700-850_hPa', '850-925_hPa']
LAYERLABELS = ['500–600', '600–700', '700–850', '850–925']
VAR = ['X1_sst_c', 'X2_shear_ms', 'X3_mid_rh_pct', 'X4_d200_1e5s', 'X5_minus_omega500_pa_s', 'X6_signed_vo850_1e5s']
VL = ['SST', 'Deep shear', 'Mid-level RH', 'Divergence', 'Ascent', 'Vorticity']
PLANS = {'FigS14': {'question': 'How do sampling centre and storm motion affect paired moisture diagnostics?', 'old': 'Six panels dominated by horizontal point/interval comparisons.', 'new': 'Two budget component column panels, frame identity columns, observed velocity time curves, before/after MFC columns and a compact shear matrix.', 'sources': ['physics/centre_budget_estimates.csv', 'physics/centre_native_estimates.csv', 'physics/FigS9_motion_panel_source.csv', 'physics/FigS9_850_windows_panel_source.csv'], 'statistical_destination': 'a–e retain original CI; f full means/CI/n go to editable SI table and CSV. No P/q family exists here.'}, 'FigS15': {'question': 'Do environmental wind-change associations differ between bridges and resolution, and under endpoint eligibility restrictions?', 'old': 'Four equal panels, three forests plus inference table.', 'new': 'Primary six-factor forest paired with original P/q/n, then two smaller endpoint forests.', 'sources': ['legacy/d48acaf71a41_main_h2_estimates.csv', 'legacy/38984d035673_state_h2_estimates.csv'], 'statistical_destination': 'Every displayed interval retained; original six-test P/q/n in panel b and complete table; sensitivity P/q remain NA.'}, 'FigS16': {'question': 'Where relative to deep shear are pre-event azimuthal moisture and circulation contrasts located?', 'old': 'Two sector forests plus orientation schematic.', 'new': 'Two physically oriented four-sector annular maps using only observed 100–300/300–800 km summaries.', 'sources': ['thematic/fig6_regions.csv'], 'statistical_destination': 'All 16 original mean/CI/n rows to editable SI table and CSV. No individual or angular interpolated data created.'}, 'FigS17': {'question': 'Do radial diagnostics and pressure-resolved transport tell a consistent story on matched support?', 'old': 'Five forest panels and one time series.', 'new': 'Humidity component columns, budget matrix, 79-pair annular check, real flux time series and discrete interval columns.', 'sources': ['same_sample/T3_summary.csv', 'same_sample/T1_same80_absolute_time_curves.csv', 'same_sample/T1_pressure_layer_summary.csv', 'same_sample/component_by_pressure_and_sector_summary.csv'], 'statistical_destination': 'a,c,d,e keep uncertainty; b/f all original CI/n retained in editable SI table and CSV. 79/80 support never merged.'}, 'FigS18': {'question': 'Which discrete pressure intervals contribute to inward, outward and net mass exchange?', 'old': 'Six forests showing class means and paired differences.', 'new': 'Three aligned numerical colour matrices: interval × temporary/final/difference, one matrix per exchange.', 'sources': ['direct/pressure_components.csv'], 'statistical_destination': 'All 36 means/CI/n retained in editable SI table and CSV; categorical unequal-width intervals, not continuous or per-hPa profiles.'}, 'FigS19': {'question': 'How much does the exchange contrast change with common support and measured-background weighting?', 'old': 'Three vertically separated forests.', 'new': 'Compact three-column forest on shared four-row support axis; all twelve CIs kept.', 'sources': ['direct/exchange_background80.csv'], 'statistical_destination': 'Original80, restricted74, geography74, state74 for all three exchanges stay plotted and tabulated.'}, 'FigS20': {'question': 'What follow-up, calibration and weight diagnostics support the recovery estimates?', 'old': 'Six equal diagnostics.', 'new': 'Integrated cohort strip, probability distributions, two calibration curves, weight-tail curve, nonsynoptic paired markers.', 'sources': ['recovery/cohort_recovered.csv', 'recovery/point_predictions.csv', 'recovery/calibration_deciles.csv', 'recovery/effects_with_CI.csv', 'recovery/nonsynoptic_sensitivity/point_estimates.csv'], 'statistical_destination': 'All actual observations retained; exact counts and calibration n in CSV; nonsynoptic comparison remains point-only, no invented CI.'}, 'FigS21': {'question': 'How do estimates move under missing-outcome and confounding assumptions?', 'old': 'Six curves in a three-row grid.', 'new': 'Outcome rows × assumption columns, using the original curves and prescribed confounding formula.', 'sources': ['recovery/MNAR_sensitivity.csv', 'recovery/MNAR_policy_response_sensitivity.csv', 'recovery/physical_confounder_summary.csv'], 'statistical_destination': 'Fixed scenario values, no sampling CI or new fit; full source parameters and coordinates in CSV.'}, 'FigS22': {'question': 'How do event definitions and departure-region denominators alter the interpretation of recurrence?', 'old': 'Main-Fig1-like return map, basin forests, risk forest and budget forest.', 'new': 'Event-definition columns, actual follow-up composition, model-risk/CI table and two SCS budget waterfalls.', 'sources': ['geography/G1_genesis_basin_recurrence.csv', 'geography/general_reintensification_genesis_basin_rates.csv', 'geography/G3_region_followup_counts.csv', 'geography/G3_region_risk_estimates.csv', 'geography/post_process_return_contrasts.csv'], 'statistical_destination': 'Definition rates retain CI; risk table retains model CI; observed bounds derived exactly from original known/unknown counts remain explicitly non-CI. Budget marginal CIs and counts in editable table.'}, 'FigS23': {'question': 'How do event complexity, status, departure geography, season and regional policy support differ?', 'old': 'Two maps and repeated interval panels.', 'new': 'Recurrence complexity stacks, actual status counts, unique departure-origin map, monthly follow-up stacks, policy matrix and coverage bars.', 'sources': ['geography/G1_genesis_basin_recurrence.csv', 'geography/general_reintensification_main_episode_sources.csv', 'geography/geographic_grid_counts_5_and_10deg.csv', 'geography/G3_departure_risk_SID_source.csv', 'geography/G3_region_policy_estimates.csv', 'geography/G3_region_policy_support.csv'], 'statistical_destination': 'All recurrence-category and regional-policy CI/n in editable SI tables/CSV; events/status and month counts use exact rows, not bootstrap.'}, 'FigS24': {'question': 'Which measured environments and subsequent moisture processes differ between SCS outcome groups?', 'old': 'Six forests.', 'new': 'Pre-event native-standardized bars, humidity components, two annular budget waterfalls, frame/motion columns and MFC columns.', 'sources': ['geography/pre_environment_group_contrasts.csv', 'geography/C_q700_SCS_return_contrasts.csv', 'geography/post_process_return_contrasts.csv'], 'statistical_destination': 'a,b,e,f retain original CI; c,d original marginal CI/n supplied in editable SI table. Group contrasts and annular populations are not labelled paired radial contrasts.'}, 'FigS25': {'question': 'How does global held-out environmental information relate to regional humidity evolution?', 'old': 'Global score forest and regional trajectory.', 'new': 'Actual held-out score gain columns with CI and all three observed regional trajectories with CI.', 'sources': ['geography/environment_model_paired_comparisons.csv', 'geography/post_process_curves.csv'], 'statistical_destination': 'All CI/n remain displayed or in caption/table; distinct global1344 and regional15/30/46 populations preserved.'}, 'FigS26': {'question': 'How do asymmetry associations vary across categorical windows and support?', 'old': 'Six repetitive 12-interval forests.', 'new': 'Two outcome matrices showing all 72 saved coefficients across six categorical window/support combinations.', 'sources': ['legacy/48fdefb6ab08_S31_effect_source.csv'], 'statistical_destination': 'All 72 estimates/CI/P/q/n and six original test-family IDs in editable SI table/CSV. Correct descriptive n range to1817–2219 from unchanged source.'}}

def fig17():
    name = 'FigS17'
    ts = 'same_sample/T3_summary.csv'
    cs = 'same_sample/T1_same80_absolute_time_curves.csv'
    ls = 'same_sample/T1_pressure_layer_summary.csv'
    ps = 'same_sample/component_by_pressure_and_sector_summary.csv'
    t, curve, layers, comp = map(read, [ts, cs, ls, ps])
    T = t.set_index('metric')
    fig, ax = grid(3, 2, 205, left=0.12, hspace=0.85, wspace=0.53)
    for j, ring in enumerate(RINGS):
        keys = ['old_' + ring + '_pre_mean_' + k for k in ['q700_raw_common', 'q700_climatology', 'q700_anomaly']]
        z = T.loc[keys].reset_index()
        z['field'] = ['Observed', 'Background', 'Anomaly']
        z['ring'] = ring
        bars(ax[0, 0], z, z.field.tolist(), [ring], {ring: RINGLABELS[j]}, {ring: [BLUE, ORANGE][j]}, groupkey='ring', est='difference', low='low', high='high', uncertainty=False)
    ax[0, 0].clear()
    hum = []
    for ring in RINGS:
        for k in ['q700_raw_common', 'q700_climatology', 'q700_anomaly']:
            row = T.loc['old_' + ring + '_pre_mean_' + k].copy()
            row['field'] = k
            row['ring'] = ring
            hum.append(row)
    hum = pd.DataFrame(hum)
    bars(ax[0, 0], hum, ['q700_raw_common', 'q700_climatology', 'q700_anomaly'], RINGS, dict(zip(RINGS, RINGLABELS)), dict(zip(RINGS, [BLUE, ORANGE])), groupkey='ring', est='difference', low='low', high='high')
    ax[0, 0].set_xticklabels(['Observed', 'Background', 'Anomaly'], rotation=15, ha='right')
    ax[0, 0].set_ylabel('Humidity contrast (g kg⁻¹)')
    decorate(ax[0, 0], 'a', 'Pre-event state on 80 pairs')
    log(name, 'a', ts, hum)
    fields = ['horizontal_advection', 'vertical_advection', 'residual', 'moving_tendency']
    rows = []
    for ring in RINGS:
        for f in fields:
            r = T.loc['old_' + ring + '_integral_0_24_' + f].copy()
            r['field'] = f
            r['ring'] = ring
            rows.append(r)
    z = pd.DataFrame(rows)
    mat = z.pivot(index='field', columns='ring', values='difference').loc[fields, RINGS]
    heat(ax[0, 1], mat, ['100–300', '300–800'], ['Horizontal', 'Vertical', 'Residual', 'Net'])
    ax[0, 1].set_xlabel('Annulus (km); integrated contrast (g kg⁻¹)')
    decorate(ax[0, 1], 'b', 'Compensating 700-hPa processes')
    log(name, 'b', ts, z, 'Saved 80-pair estimates; all CI/n in complete statistics')
    z = T.loc[['post_side_net', 'stock_change_0_24']].reset_index()
    assert (z.n == 79).all()
    ax[1, 0].bar([0, 1], z.difference / 1000000000000.0, color=[BLUE, GREY], width=0.55)
    ax[1, 0].errorbar([0, 1], z.difference / 1000000000000.0, yerr=[(z.difference - z.low) / 1000000000000.0, (z.high - z.difference) / 1000000000000.0], fmt='none', ecolor=INK, capsize=3)
    ax[1, 0].axhline(0, color=GREY, lw=0.7)
    ax[1, 0].set(xticks=[0, 1], xticklabels=['Lateral input', 'Stock change'], ylabel='Mass contrast (10¹² kg)')
    decorate(ax[1, 0], 'c', 'Separate 79-pair annular check')
    log(name, 'c', ts, z)
    z = curve[curve.kind.eq('net')]
    for side, col in [('internal', BLUE), ('final', ORANGE)]:
        g = z[z.side.eq(side)].sort_values('lag_h')
        ax[1, 1].plot(g.lag_h, g.mean_kg_s / 100000000.0, 'o-', color=col, ms=3, lw=1.3, label='Temporary' if side == 'internal' else 'Final')
        ax[1, 1].fill_between(g.lag_h, g.low_kg_s / 100000000.0, g.high_kg_s / 100000000.0, color=col, alpha=0.15, lw=0)
    ax[1, 1].set(xlabel='Hours after departure', ylabel='Net inward flux (10⁸ kg s⁻¹)', xticks=[0, 6, 12, 18, 24])
    ax[1, 1].legend(frameon=False, loc='upper right')
    decorate(ax[1, 1], 'd', 'Boundary transport over time')
    log(name, 'd', cs, z)
    ax[2, 0].bar(range(4), layers.difference_kg / 1000000000000.0, color=BLUE, width=0.6)
    ax[2, 0].errorbar(range(4), layers.difference_kg / 1000000000000.0, yerr=[(layers.difference_kg - layers.low_kg) / 1000000000000.0, (layers.high_kg - layers.difference_kg) / 1000000000000.0], fmt='none', ecolor=INK, capsize=2)
    ax[2, 0].set(xticks=range(4), xticklabels=LAYERLABELS, xlabel='Discrete pressure interval (hPa)', ylabel='Net mass contrast (10¹² kg)')
    ax[2, 0].tick_params(axis='x', labelrotation=20)
    ax[2, 0].axhline(0, color=GREY, lw=0.7)
    decorate(ax[2, 0], 'e', 'Layer-integrated net exchange')
    log(name, 'e', ls, layers)
    z = comp[comp.cohort.eq('original_post80') & comp.window.eq('0_24h') & comp.factor.eq(1) & comp.kind.eq('out') & comp.sector.eq('all') & comp.layer.isin(LAYERS) & comp.quantity.isin(['humidity', 'relative_motion', 'difference'])]
    bars(ax[2, 1], z, LAYERS, ['humidity', 'relative_motion', 'difference'], {'humidity': 'Humidity', 'relative_motion': 'Motion', 'difference': 'Total'}, {'humidity': GREEN, 'relative_motion': ORANGE, 'difference': INK}, xkey='layer', groupkey='quantity', low='low', high='high', scale=1000000000000.0, uncertainty=False)
    ax[2, 1].set_xticklabels(LAYERLABELS, rotation=20)
    ax[2, 1].set(xlabel='Discrete pressure interval (hPa)', ylabel='Outward contrast (10¹² kg)')
    fig.legend([Patch(color=GREEN), Patch(color=ORANGE), Patch(color=INK)], ['f: Humidity', 'Motion', 'Total'], loc='lower center', bbox_to_anchor=(0.57, 0.004), ncol=3, frameon=False, fontsize=8)
    decorate(ax[2, 1], 'f', 'Humidity and motion by layer')
    log(name, 'f', ps, z, 'Cq + Cv = total; all saved CIs retained in complete statistics')
    fig.legend([Patch(color=BLUE), Patch(color=ORANGE)], RINGLABELS, loc='upper center', bbox_to_anchor=(0.53, 0.995), ncol=2, frameon=False)
    finish(fig, name)

def decorate(ax, letter, title, pad=16):
    ax.set_gid(letter)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-16, pad - 4), textcoords='offset points', weight='bold', fontsize=10.5, ha='left', va='bottom')
    ax.set_title(title, loc='left', pad=pad, fontweight='normal')

def heat(ax, values, xlabels, ylabels, vmax=None, fmt='.2f', cmap='RdBu_r', mark=None):
    values = np.asarray(values, float)
    vmax = max(np.nanmax(np.abs(values)), 1e-10) if vmax is None else vmax
    im = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0, pad=6)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                text = format(values[i, j], fmt)
                ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if abs(values[i, j]) > 0.63 * vmax else INK)
                if mark is not None and mark[i, j]:
                    ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, lw=1.2, zorder=5))
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im

def bars(ax, data, keys, groups, labelmap=None, colors=None, xkey='field', groupkey='policy', est='estimate', low='lo', high='hi', scale=1, uncertainty=True):
    xx = np.arange(len(keys))
    width = 0.72 / len(groups)
    for i, g in enumerate(groups):
        sub = data[data[groupkey].eq(g)].set_index(xkey).loc[keys]
        e = sub[est].to_numpy(float) / scale
        color = (colors or {}).get(g, [BLUE, ORANGE, GREY][i % 3])
        pos = xx + (i - (len(groups) - 1) / 2) * width
        ax.bar(pos, e, width * 0.86, color=color, label=(labelmap or {}).get(g, g), zorder=2)
        if uncertainty:
            lo = sub[low].to_numpy(float) / scale
            hi = sub[high].to_numpy(float) / scale
            valid = np.isfinite(lo) & np.isfinite(hi)
            ax.errorbar(pos[valid], e[valid], yerr=[e[valid] - lo[valid], hi[valid] - e[valid]], fmt='none', ecolor=INK, elinewidth=0.75, capsize=2, zorder=3)
    ax.axhline(0, color='#9BA5AA', lw=0.65, zorder=1)
    ax.set_xticks(xx)
    ax.yaxis.set_major_locator(MaxNLocator(4))

def grid(rows=2, cols=2, height=170, left=0.13, hspace=0.72, wspace=0.45):
    fig, axs = plt.subplots(rows, cols, figsize=(180 / 25.4, height / 25.4), squeeze=False)
    fig.subplots_adjust(left=left, right=0.965, bottom=0.11, top=0.91, wspace=wspace, hspace=hspace)
    return (fig, axs)