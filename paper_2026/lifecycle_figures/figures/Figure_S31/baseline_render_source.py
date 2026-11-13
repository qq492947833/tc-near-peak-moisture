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

def grid(rows=2, cols=2, height=170, left=0.13, hspace=0.72, wspace=0.45):
    fig, axs = plt.subplots(rows, cols, figsize=(180 / 25.4, height / 25.4), squeeze=False)
    fig.subplots_adjust(left=left, right=0.965, bottom=0.11, top=0.91, wspace=wspace, hspace=hspace)
    return (fig, axs)

def decorate(ax, letter, title, pad=16):
    ax.set_gid(letter)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-16, pad - 4), textcoords='offset points', weight='bold', fontsize=10.5, ha='left', va='bottom')
    ax.set_title(title, loc='left', pad=pad, fontweight='normal')

def pick(df, **conditions):
    d = df
    for k, v in conditions.items():
        d = d[d[k].eq(v)]
    assert len(d) == 1, (conditions, len(d))
    return d.iloc[0]

def fig21():
    name = 'FigS21'
    sources = ['recovery/MNAR_sensitivity.csv', 'recovery/MNAR_policy_response_sensitivity.csv', 'recovery/physical_confounder_summary.csv']
    shared, policy_only, conf = map(read, sources)
    fig, ax = grid(2, 3, 145, left=0.105, hspace=0.8, wspace=0.48)
    fig.subplots_adjust(top=0.84)
    for row, out in enumerate([48, 24]):
        for col, (src, data, xfield, title) in enumerate([(sources[0], shared, 'MNAR_offset', 'Both missing outcomes'), (sources[1], policy_only, 'offset', 'Policy response only')]):
            for policy, colr in [('shear200_850', BLUE), ('midRH', ORANGE)]:
                z = data[data.policy.eq(policy) & data.outcome.astype(str).eq(str(out))].sort_values(xfield)
                ax[row, col].plot(z[xfield], z.estimate, 'o-', color=colr, ms=3, lw=1.4)
            ax[row, col].axhline(0, color=GREY, ls=':', lw=0.7)
            ax[row, col].axvline(0, color='#CBD2D6', lw=0.7)
            ax[row, col].set_xlabel('Log-odds offset' if out == 48 else 'Outcome offset (kt)')
            if col == 0:
                ax[row, col].set_ylabel('Recovery contrast (pp)' if out == 48 else 'Wind-change contrast (kt)')
            decorate(ax[row, col], chr(97 + row * 3 + col), title)
            log(name, chr(97 + row * 3 + col), src, data[data.outcome.astype(str).eq(str(out))])
        strengths = np.array([0.5, 1, 2, 4])
        for policy, colr in [('shear200_850', BLUE), ('midRH', ORANGE)]:
            benchmark = 'past_shear' if policy == 'shear200_850' else 'past_RH'
            z = pick(conf, policy=policy, outcome=out, benchmark=benchmark)
            y = z.point_effect - abs(z.mean_signed_bias_one_benchmark) * strengths ** 2
            ax[row, 2].plot(strengths, y, 'o-', color=colr, ms=3, lw=1.4)
            rows = pd.DataFrame([dict(z, association_multiplier=x, displayed_scenario=y0) for x, y0 in zip(strengths, y)])
            log(name, chr(99 + row * 3), sources[2], rows, 'Original prescribed sensitivity equation; no fitted parameter')
        ax[row, 2].axhline(0, color=GREY, ls=':', lw=0.7)
        ax[row, 2].set(xticks=[0.5, 1, 2, 4], xticklabels=['0.5', '1', '2', '4'], xlabel='Benchmark multiplier')
        decorate(ax[row, 2], chr(99 + row * 3), 'Confounding benchmark')
    fig.legend([Line2D([0], [0], color=BLUE), Line2D([0], [0], color=ORANGE)], ['Shear −2 m/s', 'RH +5 pp'], loc='upper center', bbox_to_anchor=(0.53, 0.995), ncol=2, frameon=False)
    finish(fig, name)
GREY="#7F9099"
mpl.rcParams.update({"font.size":8,"axes.labelsize":8,"axes.titlesize":9,"legend.frameon":False})
