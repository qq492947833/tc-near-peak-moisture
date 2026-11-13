# Moisture, energy and prospective recovery analyses

These are the implemented scientific functions, production sources and frozen processed inputs used for the current 1982–2023, 2,340-SID manuscript. The independent MWT and statistical functions retain their original signatures. Historical internal fields such as `final`, `Res` and `resolution_time` are retained to preserve data identities; their display meanings are **decay**, **Dec**, and the start of decay. `internal` denotes a temporary departure. Neither the six-hour event anchors nor the temporary-minus-decay comparison direction has changed.

## Available routes

Run these commands from the repository root after fetching the release input assets. Use new output directories.

```sh
python paper_2026/physics/direct_mwt/verify.py --output runs/direct_mwt_check
python paper_2026/physics/verify_saved_results.py --output runs/saved_physics_check
```

The first command imports each of five independent MWT functions, performs 55 analytical implementation checks and recomputes 25 diagnostics from a saved ERA5 coefficient stencil. The stencil has three consecutive coefficient times, a 3 × 3 horizontal neighborhood, all 13 wind/temperature pressure levels and the three directly observed omega levels. It is a local diagnostic check, not a population MWT rerun.

The second command verifies the 2,340-SID identity, 117 paired departure SIDs, RH decomposition identities and means, saved moisture-budget means and supports, the 2,231-SID measured-environment recovery subset, confidence intervals from the original 1,000 fitted draws, and the accepted hemisphere correction from the original 2,000 SID multiplicities. It neither fits models nor samples new draws.

For a deliberate recalculation from processed data, the original callable interfaces and drivers remain available:

```sh
python paper_2026/physics/mwt_statistics/reproduce.py --output /absolute/path/to/new_mwt_statistics
python paper_2026/physics/rise_environment/reproduce_environment.py --output /absolute/path/to/new_rise_environment
```

These two drivers recompute the original statistical procedures, including their specified bootstrap attempts. They were not rerun for the present publication. Their frozen inputs, output comparison tables and tolerances are supplied. The MWT driver covers six departure outcomes and 48 fixed return-ascent jobs; it does not claim every diagnostic at every level has been independently replayed in this release. The environment driver covers the saved 23 diagnostic groups.

`recovery/fit_policies_v6.py` is the original five-fold, interval-censored recovery estimator with its original nuisance transformations, support rule, policy contrast and SID-level refitting. `cohort_recovered.csv`, fold membership, point predictions, saved bootstrap estimates and published intervals belong to the same 2,231-SID analysis. Copy `recovery/` to a new working directory before requesting any refit; the manuscript run uses `python fit_policies_v6.py --bootstrap 1000` there (the historical function default is 300, so the explicit argument is required). Use `--resume` only for that same working directory and saved draw bank. The `diagnostics_v6.py`, `mnar_and_balance.py`, `no_pi_sensitivity.py` and `physical_confounder_benchmark.py` companions retain the original observation/support sensitivities. Unknown outcomes are not recoded as failures, and complete 48-hour observation is not an entry requirement.

## Analysis and source map

| Analysis | Actual entry or interface | Inputs and retained outputs |
|---|---|---|
| Common paired environment, fixed/moving centers, radial and vertical moisture budgets | `production/project/03_Work/v6/physics/process_v6.py`, `centre_native_support_v6.py`, `centre_budget_native_v6.py`, `transport_regions_support_v6.py` | `water/paired/` contains the corresponding per-SID values, native/complete supports and interval tables. Diagnostic supports remain separate. |
| RH thermodynamic versus changing-domain decomposition | `RH_matched_decomposition.py`, `rh_native_v6.py`; current-master aggregation in `production/mwt/addendum_20261023/unified_main_20261023_1642/scripts/recompute_boundary_summaries.py` | `water/rh/` contains current-master per-SID components, 192 summary rows and original support fields. |
| Boundary-related moving-frame and climatological companions | `recompute_boundary_summaries.py`, `recompute_boundary_companions.py`, plus `production/mechanism_v5/src/p4_p5_analysis_v5.py` | `water/p4/` contains current-master transport, matched-window, local-background and PI component outputs. Subtracting local climatology and adjusting event-group background are distinct operations. |
| Geographic, seasonal and storm-state adjustment | `production/mechanism_v5/src/p1_p2_analysis_v5.py`, `p1_background_v5.py`; `refit_p4_p5_models.py` for stage-boundary companions | Original model and balance/support routines are supplied. Input schemas and source-based settings remain in the production code and configuration. |
| MWT preparation and direct diagnostics | Five independent files under `direct_mwt/src/`; native driver `production/mwt/scripts/run_mwt_sid.py` | `direct_mwt/config/`, the saved stencil, frozen 234 events and centers; original regional/cache outputs can feed statistics without recalculating fields. |
| Paired MWT effects and sensitivity | `mwt_statistics/src/mwt_paired_statistics.py`, `mwt_motion_intensity_sensitivity.py` | Saved regional values, original support, bootstrap identities and expected output tables. |
| Actual return-ascent environments and matched decay comparison | `rise_environment/src/Rise_Event_Statistics.py`, `production/mwt/addendum_20261023/scripts/build_return_ascent_ledger.py`, `build_rise_comparable_requests.py`, `extract_rise_environment.py` | Frozen 133 events in 123 SIDs, legal intervals, request roles, matched-state pairs and processed environmental values. Support varies by diagnostic and window. |
| Return-ascent energy | `production/mwt/addendum_20261023/scripts/run_mwt_rise_increment.py`, `mwt_rise_statistics_driver.py` | Original coefficient reuse and increment interface; processed replay inputs under `mwt_statistics/`. |
| Prospective 48-hour recovery and observation sensitivities | `recovery/fit_policies_v6.py` and companions | 2,231 measured-environment SIDs; fixed entry thresholds, five folds, original outcome/censoring fields, predictions and 1,000 saved draw results. |
| Accepted Southern Hemisphere shear-sector correction | `water/hemisphere/` and the final block of `verify_saved_results.py` | 103 paired SIDs, including 25 Southern Hemisphere SIDs; swap only left/right sector assignment, retaining already cyclonic-signed scalars and stored draws. `corrected_16cells.csv` is the current directional summary. Historical unreflected sector labels in input rows are not the final pooled display convention. |

## Native-field production inputs

`production/` preserves the actual source tree and scientific interfaces, with private drive paths replaced by aliases in `production/paths.json`. It includes project dependencies, the pinned upstream tcpyPI files and their license. External data are **inputs**, not unprovided private Python dependencies. The production programs write their caches and results beside their working source tree; therefore first create a separate configured copy:

```sh
python paper_2026/physics/configure_production.py \
  --era5-root /data/era5_6hour \
  --ibtracs /data/ibtracs.ALL.list.v04r01.csv \
  --ocean-shapefile /data/natural_earth/ne_10m_ocean.shp \
  --workspace runs/native_production
python runs/native_production/mwt/scripts/theory_build_operators.py
python runs/native_production/mwt/scripts/run_mwt_sid.py \
  --work runs/native_production/mwt --sid 1982005S10105
```

The initializer reads metadata and bounded source probes only. It checks the native variable, shape, unit, coordinate hashes, file size and saved scalar probes before recording the **local** file inode/mtime required by the original integrity checks. It materializes paths and updates relocated code/configuration identity pins; it does not change physical parameters or weaken the source-data checks. `--check-only --limit 1 --report runs/one_raw_file.json` is a bounded preflight. A new downloaded archive must first be organized into the declared native file schema; the initializer rejects mismatches rather than treating a different grid, time index or variable as equivalent.

The saved source catalog describes continuous 6-hourly, 0.25-degree ERA5 fields with `(time, latitude, longitude)` dimensions. `production/mwt/reports/data_era5_inventory.json` records variable names, units, levels, exact native coordinate hashes and bounded original probes. `production/mechanism_v5/config/available_variable_map_v5.json` covers the environmental variables. Full ERA5 fields, native-field caches and fitted-model pickle objects are not distributed. Environment extraction consumes the requested event/center tables and maintains its native-grid cache; its scientific code and configuration are included, but the complete raw-archive-to-manuscript workflow was not executed for this release.

MWT uses u, v, temperature and geopotential at **50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925 and 1,000 hPa**. Omega is read directly only at **500, 700 and 850 hPa**. There are 1,024 six-hourly input samples (256 days), scale parameters `j0=5, j1=8`, symmetric boundaries and physical normalization; the nominal windows are >16 days, 2–16 days and <2 days. The fixed 64-step shift is a 16-day sensitivity. The shared reference state uses area/time averages of valid above-ground observations on the original fixed domain; it is not recomputed separately for the paired events.

Primary energy stores are reported at 700 hPa in the 2–16-day window. Horizontal canonical-transfer contributions, APE-to-KE conversion, horizontal transport and vertical boundary inputs are separate quantities. The 500–850-hPa inputs use `[Q(500) − Q(850)]/g` in W m−2 or division by 35,000 Pa in W kg−1, with positive net input. No extra omega layers, continuity-derived omega, or complete three-dimensional canonical transfer are introduced. `direct_mwt/config/diagnostic_semantics.json` retains the detailed units and signs.

## Software and validation scope

The small MWT/saved-result validation was executed with Python 3.11.7, NumPy 1.26.4 and pandas 2.1.4. The independent files import only their declared NumPy/pandas/SciPy/h5py dependencies. Production raw-field and model sources additionally use pyarrow, netCDF4, scikit-learn, statsmodels, psutil, threadpoolctl, Cartopy, Shapely and pyshp. Only the exact interval functions called by the current-master code are retained from the historical duration preparer; its obsolete standalone Notebook-driven classifier is excluded. Dynamically loaded v4 representation estimators, the current-master wrappers, cache validators and exact-wind sensitivity source are included under the portable `production/legacy_project` and associated directories. Some original production drivers use Windows file locking or process priority; the independent scientific functions and the two small verification commands do not require those drivers. Use the repository environment files for the supported complete dependency set.

For this release, all Python files were compiled for syntax, the local module dependency set was checked, the two small verification commands were run, and one real raw ERA5 file passed the bounded metadata/coordinate/scalar-probe preflight. New MWT caches use the outcome-independent validator with `reuse_queue_hashes=False` (the default), which streams the output hashes; historical pilot-runner and queue-reuse exceptions are not carried to a new workspace. No ERA5 field campaign, population MWT calculation, recovery refit or new bootstrap experiment was run. Public release manifests record source identities and verification results separately from the scientific data.
