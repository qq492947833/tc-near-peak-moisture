# Basin, historical and latitude analyses

This directory contains the implementations used for the manuscript's basin
comparisons, historical changes, stage-duration standardization and departure-
latitude recovery analysis. The supplied tables retain the frozen 1982–2023
cohort of 2,340 storm identifiers. They distinguish 123 cyclones with an observed
anchored near-peak return, 133 return events, and the separate first-departure
48-hour recovery population. Historical field names such as `Res` remain machine
identifiers for the displayed Decay (`Dec`) stage.

The scripts preserve the original scientific function implementations. Public
adaptations replace workstation-specific paths and dependency bootstraps. The
latitude likelihood's `Basis` and `IntervalHazard` classes are bundled as exact
source extractions from the original recovery implementation. Internal rendering
QA hooks are not needed to generate the figures and have been removed from the
plot entry points; the plotted quantities, geometry and export operations remain.

## Environment and paths

The saved-result checks were run with Python 3.11 and NumPy 1.26.4, pandas 2.1.4,
SciPy 1.11.4, statsmodels 0.14.0, Matplotlib 3.8.0, Pillow 10.2.0,
PyMuPDF 1.28.2 and threadpoolctl 2.2.0. Plotting requires an installed Arial font
for the Arial-based figures; the font is not distributed. No local private
Python package is required.

Run in a working copy of this directory, because the original independent
analysis scripts save their outputs next to their entry points. For example:

```bash
cp -R paper_2026/newanalyses /path/to/work/newanalyses
cd /path/to/work/newanalyses
python validate_saved.py --output /path/to/work/validation.json
```

If the release distributes processed inputs separately, first restore that
asset at its documented relative paths. Default shared inputs are under
`inputs/addendum/`; `release_paths.py` also accepts `FLTCCF_ADDENDUM_DIR` and
`FLTCCF_UNIFIED_DIR` to use an external processed-data directory. The latter
defaults to `<addendum>/unified_main_20261023_1642`. These variables affect file
locations, not the scientific settings. Temporary and plotting caches are
placed beside the working scripts, not in a fixed user directory.

## Analysis map and running order

| Analysis | Actual code entry | Configuration and input | Main saved outputs |
|---|---|---|---|
| Observed basin opportunities and first departures | `history/agents/A/prepare_a.py` | Main SID, return-event and six-hour track/interval/environment tables | `sid_denominators.csv`, `risk_intervals_6h.csv`, `first_departure_reference.csv` |
| Basin rates and adjusted probabilities | `history/agents/A/model_a.py`; `status_sensitivity.py` | Prepared A inputs; `MODEL_CONFIG.json`; within-year SID sampling in circular year blocks | Basin estimates, contrasts, shared-support probabilities and draw bank |
| Annual return trends and basin composition | `history/agents/B/history_counts_trends.py` | Prepared A first-departure table; `DESIGN_BEFORE_COUNTS.json` | Annual and period numerators/denominators, the 11-test trend family, symmetric composition decomposition |
| Historical environmental standardization | `history/agents/B_env/analyze_b_env.py`; `environment_only.py` | A shared-support table, saved model configuration and joint draws | Actual/fixed probability curves, trend contrasts, environmental histories and support diagnostics |
| Duration inputs and recorded wind support | `history/agents/C/audit_duration.py` | Frozen stage entries, duration/track tables and original primary trends | Stage-entry source, coverage, annual stage and F-component tables |
| Stage-duration standardization and sensitivities | `history/agents/C/analyze_duration.py`; `sensitivity_validate.py`; `positive_duration_sensitivity.py` | Prepared C tables, `MODEL_SPEC_AND_AUDIT.json`; common 2,000 indexed draw attempts | 16 standardized trends and BH q values, stage-specific support, complete-case/terminal-wind/Gamma sensitivities |
| Paired interpretation of saved C draws | `spatial/agents/C_audit/audit_saved_C.py` | Existing C outputs and unchanged draw identities | Same-support trend contrasts and intervals, F endpoint identities, terminal-coverage comparisons |
| Locations of actual returns | `spatial/agents/event_latitude/describe_event_latitude.py` | All 133 events, track positions, A event/denominator tables; fixed display bins | `event_latitude_ledger.csv`, latitude support and descriptive counts |
| Departure-latitude 48-hour recovery | `spatial/agents/latitude_risk/v2_fold_supported/fit_latitude_risk.py` | `FROZEN_CONFIG.json`, 1,245 training rows, 322 fixed reference rows, inherited five SID folds and original 1,000 draw indices | Hemisphere point estimates, curves, paired bootstrap results and fit diagnostics |
| Latitude saved-result summaries | `spatial/agents/latitude_risk/v2_fold_supported/summarize_latitude_risk.py` | Saved point/draw results and original recovery draw bank | Risks, contrasts, environment descriptions, support counts and paired intervals |

The input-preparation entry points use the included upstream processed files
listed with byte sizes and SHA-256 hashes in `upstream_inputs.json`. Restore the
processed-data release asset to obtain these files. They are outputs of the
primary lifecycle/environment workflow, not alternative raw observations.
The supplied prepared-model inputs permit later steps to be run without repeating
the raw ERA5 processing. Model fitting and all resampling are optional, explicit
steps; the release checks do not run them.

Representative saved-result commands are:

```bash
python validate_saved.py --output /path/to/work/validation.json
python spatial/agents/latitude_risk/v2_fold_supported/fit_latitude_risk.py --help
python spatial/agents/latitude_risk/v2_fold_supported/summarize_latitude_risk.py
python spatial/agents/C_audit/audit_saved_C.py
```

To deliberately recompute fitted analyses from the prepared inputs, run A before
B/B_env, and C's input preparation before `analyze_duration.py`. The two A/B_env
support sensitivities use `--complete-case-sensitivity` and `--temporal-overlap`,
respectively. Run both the default and temporal-overlap B_env analyses before
`environment_only.py`, which reads both saved configurations. The C sensitivity
entry points use the original C joint-draw bank. The latitude fitter accepts
`--hemisphere NH` or `--hemisphere SH`, `--point-only`, and `--resume`;
omitting `--point-only` performs the original 1,000-draw fit for that hemisphere.
The saved v2 training and reference populations must not be replaced by the
superseded reference-only training population.

## Scientific distinctions retained

- Basin event opportunity rates use the eligible observed six-hour post-departure
  intervals in `risk_intervals_6h.csv`. They are not event counts divided by every
  cyclone-hour, nor a 48-hour recovery probability.
- Annual global return proportions count each returning SID once, use all main
  cohort SID in that genesis year, and retain cross-year histories. The global
  estimate is not an unweighted mean of basin percentages. NI/SP remain
  descriptive and SA's missing main-cohort denominator remains missing.
- The C primary family is exactly four stages by four standardization targets
  (16 tests). The five-test F-component family and other tests remain distinct.
  C's main duration contrasts are on the hours-per-decade scale. Shared support,
  reference weights and paired draw identities are retained.
- `F = R + L + B` is checked on the same 2,340 SID and endpoints. Terminal
  missing-wind sensitivity addresses recorded duration, not physical dissipation.
- Actual-return latitude uses all 133 successful events, including repeats.
  The first-entry 48-hour population instead has 314 observed recoveries, 1,351
  known nonrecoveries and 675 unknown outcomes among 2,340 SID. No complete
  48-hour follow-up condition is added to eligibility.
- Latitude comparisons retain 1,245 training SID and 322 assigned-fold-supported
  reference SID (202 north, 120 south). RH and shear precede entry; PI is measured
  at the prediction time. The environmental model uses their joint distribution,
  without also entering PI-minus-current-wind as a redundant predictor. The
  two models share samples, targets, weights and draw identities. Unknown
  outcomes retain interval/right censoring.

## Figure sources

These entries generate the accepted statistical panels before the final lossless
layout compaction. Use the repository's final figure mapping for the subsequent
panel-composition step and current filenames.

| Current figure | Entry and supplied source tables |
|---|---|
| Main Fig. 7 | `history/figures/SC_DURATION/plot.py` |
| Supplementary Fig. 5 | `history/figures/SA_BASIN/plot.py` |
| Supplementary Fig. 6 | `spatial/figures/LATITUDE/plot_latitude.py` |
| Supplementary Fig. 7 | `history/figures/NEW_HISTORY/plot.py` |
| Supplementary Fig. 8 | `history/figures/SB_HISTORY/plot.py` |

NEW_HISTORY's lower panel is the preserved native-resolution crop of the supplied
`history/backup/Figure_07.png`; it is not recreated from a screenshot. The final
layout-only compaction scripts preserve the original panel pixels. This release
was syntax/input checked without rerendering these accepted figures.

## Verification scope

The portable saved-result check verifies the cohort and event denominators,
six-hour opportunity totals, full 11- and 16-test BH families, the latest global
trend and interval using the delivered HAC implementation, all-SID F identities,
stage-duration saved-draw intervals, latitude populations/folds and paired-draw
intervals. An isolated run additionally executed the latitude summary and C
saved-draw reaggregation and compared ten resulting tables to their frozen
values (maximum permitted absolute difference `1e-11`). No new model fit,
bootstrap draw, ERA5/MWT calculation or figure rendering was performed for this
release. Validation of these selected routes does not claim an end-to-end rerun
of the entire research archive.
