# FLTCCF: tropical-cyclone lifecycle and near-peak processes

Code, configurations and processed inputs for the 1982–2023 study of 2,340
tropical-cyclone storm identifiers (SIDs). The current release covers lifecycle
classification and comparisons, environmental evolution, paired temporary
departure and decay conditions, moisture transport, multiscale energy,
reintensification and recovery, spatial contrasts, and historical duration changes.
The reading set contains eight main figures, seventeen supplementary figures and ten core tables. The supplementary information includes the estimates, confidence intervals and support needed for the reported comparisons, including the 700-hPa scale-boundary contrasts in Table S5. The [statistical archive guide](paper_2026/selected_evidence/layout/DATA_DICTIONARY.md) identifies frozen matrices, historical row identifiers and units. The workbook is an optional research archive, not a required separate submission file.

Download the [current figure pairs](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.5/tc-near-peak-moisture-v2.1.5-figures.zip), [full frozen statistics](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.5/Supplementary_Data_1_CSV.zip), [optional archive workbook](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.5/Supplementary_Data_1.xlsx) and [checksums](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.5/SHA256SUMS.txt). See the [panel identity map](paper_2026/selected_evidence/layout/panel_identity.csv), [original-to-current crosswalk](paper_2026/selected_evidence/panel_map.csv) and [figure file checksums](paper_2026/selected_evidence/figure_files.csv).

The ten revised figures use frozen inputs and dedicated axes at their intended reading width. With the project's Python plotting dependencies installed, run from the repository root:

```sh
python paper_2026/selected_evidence/layout/group_a/render_reading_panels.py --inputs paper_2026/selected_evidence/layout/group_a/inputs --output generated/selected
python paper_2026/selected_evidence/layout/group_b/render_selected.py --output-dir generated/selected
```

For the unchanged selected figures, use the [previously published native artwork](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.4/tc-near-peak-moisture-v2.1.4-figure-sources.zip) and the original assembly script with `--numbers 0 1 9 10 12 13 16 17`. All unchanged final images are in the current figure archive. Figure 6e shows nine trough-relative transfer rates; Supplementary Fig. 13b shows integrals over the actual trough-to-return intervals. The inference families and numerical qualifications remain unchanged.

## Historical v2.1.3 figure updates

Figure 6e shows trough-aligned horizontal, pressure-direction and total KE canonical-transfer rates (W kg⁻¹; displayed in units of 10⁻⁴ W kg⁻¹). All nine six-hour observations from −24 to +24 h use the same 126 events in 117 cyclones, 0–300 km, 500–850 hPa and the nominal 2–16-day window. Pointwise 95% intervals use the original 2,000 cyclone-level draws. Some observations at positive lags fall after near-peak return. Supplementary Figure 26a retains the event-specific KE transfer integrals (J kg⁻¹) for 132 events in 122 cyclones.

Download the [v2.1.3 figure archive](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.3/tc-near-peak-moisture-v2.1.3-figures.zip) and [SHA-256 checksum](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.3/SHA256SUMS.txt). From the repository root, with the ZIP in the working directory:

```sh
python -m zipfile -e tc-near-peak-moisture-v2.1.3-figures.zip runs/fig6-assets
python paper_2026/physics/vertical_extension/fig6e_rates/plot_fig6e.py --out runs/fig6-rates
python paper_2026/physics/vertical_extension/fig6e_rates/compose_figure06.py --base runs/fig6-assets/source_assets/v2.1.2/Figure_06.png --panel runs/fig6-rates/Figure_06e_KE_transfer_rate.png --out runs/fig6
```

The last command writes the full Figure 6 PNG and lossless TIFF. For pixel-identical assembly from the released native panel, pass `runs/fig6-assets/source_assets/Figure_06e_KE_transfer_rate.png` to `--panel`. The ZIP includes a SHA-256 manifest for its four images. The [rate package](paper_2026/physics/vertical_extension/fig6e_rates/README.md) documents the scientific inputs and the targeted coefficient recovery and saved-draw calculations.

The [v2.1.2 figure archive](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.2/tc-near-peak-moisture-v2.1.2-figures.zip) contains the v2.1.2 versions of Figures 5, 6, 8, S25 and S26 at the archive root, with native source panels under `source_assets/`. See [the vector-scene reproduction command](paper_2026/lifecycle_figures/figures/Figure_08/REPRODUCE_FIGURE_08.md) and [the supplementary panel assembly command](paper_2026/physics/vertical_extension/plots/README.md). See the [native panel reordering commands](paper_2026/lifecycle_figures/layout/scripts/REPRODUCE_PANEL_ORDER.md). The [figure identity table](figure_updates.csv) lists the revised files separately from the historical catalogue.

## Duration statistics from the original draws

Run the following command from the repository root:

```sh
python paper_2026/newanalyses/history/agents/C/replay_duration.py --output runs/duration
```

This entry point summarizes the saved standardized duration targets and paired
contrasts on their original draw IDs. A rank-deficient fit can still have an
estimable trend; the sample counts in these tables refer to estimable targets.
See [duration definitions and recovery checks](paper_2026/newanalyses/history/agents/C/DURATION_REPLAY.md) for the full and recent periods and the optional original-draw check. The original input archives remain pinned. Their historical figure templates
are separate from the [accepted Figure 7 PNG](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.1/Figure_07.png).
The MWT figure bundle remains available under v2.1.0.

## MWT pressure-direction and layer diagnostics

Version 2.1.0 adds the actual pressure-coordinate canonical-transfer implementation,
horizontal-plus-vertical totals, and consistent 500–850 hPa storage, conversion and
time integrals. The extension retains the original 1982–2023 cohort, fixed MWT
operator, reference state, event definitions and quantity-specific support.

The new [vertical-extension package](paper_2026/physics/vertical_extension/README.md)
contains scientific entry points, configurable input locations, full-precision
summary tables and the figure scripts. In v2.1.0, the integrated Figure 6b and Supplementary Figure 29 used identical spatial/temporal support: 122 SIDs and 132
actual reintensification events, 0–300 km, 500–850 hPa and the nominal 2–16-day window.
Their estimates are equal-SID averages of event integrals, in J kg⁻¹, with saved
2,000-draw percentile intervals. The new high-resolution figures and unchanged
base-panel assets are distributed with the v2.1.0 figure bundle.

The published extraction and statistical drivers are explicit computational
routes; plotting from the released summary tables needs no atmospheric-field
extraction, model fit or new bootstrap. See the package instructions for the
saved-result plotting commands and figure-asset download.

## Start from the versioned processed inputs

Use Python 3.11 for the current scientific validation environment. Work in a
writable copy and choose local paths for inputs, outputs and caches.

```sh
git -c core.autocrlf=false clone --branch v2.1.3 https://github.com/qq492947833/tc-near-peak-moisture.git
cd tc-near-peak-moisture
python -m venv .venv
# Activate .venv using the command appropriate for your operating system.
python -m pip install -r requirements.txt
python fetch_release_data.py
python verify_release.py --output validation/summary.json
```

`fetch_release_data.py` retains the pinned v2.0.0 base inputs and figure sources.
The MWT extension has its own versioned assets and instructions. The base fetcher downloads the processed-input and figure-source
assets, verifies archive/member SHA-256 values, and restores their relative paths.
For an offline run, download the two assets from the [v2.0.0 Release](https://github.com/qq492947833/tc-near-peak-moisture/releases/tag/v2.0.0) and pass each
with `--archive`. Existing matching files are reused; different local files are
not overwritten. The release does not distribute the global ERA5 or IBTrACS raw
archives, manuscript documents, private execution logs, or credentials.

The quick validation checks saved scientific values and small bounded examples.
It does not start model fitting, resampling, full ERA5 extraction or figure
rendering. Subsequent steps can reuse the supplied processed tables, coefficient
stencil and original draw banks rather than repeat expensive extraction.

## Code organisation and analysis map

| Paper analysis | Code entry or source directory | Configuration / input | Outputs |
|---|---|---|---|
| IBTrACS selection, missing winds and lifecycle stages | `src/raw_workflow/ibtracs_prepare.py`, `lifecycle_core.py`, `lifecycle.py` | `config/raw_inputs.json`; current frozen SID and record tables | SID summaries, six-hour records, stages and near-peak intervals |
| Framework comparisons, threshold sensitivities and stage environments | `paper_2026/lifecycle_figures/framework/` | Frozen primary input, 18 threshold configurations and indexed SID draws | Comparison scores, stage contrasts and sensitivity tables |
| Humidity evolution and time/location or q/T decomposition | `src/raw_workflow/humidity.py`; `paper_2026/physics/production/` | Native field requests, fixed masks and saved common-support tables | Time/location and thermodynamic contributions |
| Within-SID temporary departure versus decay, background/state adjustment | `paper_2026/physics/production/` | Actual paired-event ledger, geographical support and original weights | Unadjusted/adjusted contrasts, balance and valid-pair summaries |
| Radial moisture, advection and moving-boundary exchange | `src/full/transport.py`; `paper_2026/physics/production/` and `water/` | Original centres, native fields, boundary nodes and valid-area masks | Signed gross in/out/net terms and constituent decompositions |
| MWT storage, conversion, canonical-transfer contributions and layer input | `paper_2026/physics/direct_mwt/`, `mwt_statistics/`, `production/mwt/`, `vertical_extension/` | `direct_mwt/config/direct_diagnostics.json`; saved coefficients and per-SID values | Energy diagnostics, paired estimates and original statistical families |
| Actual reintensification and matched decay trajectories | `paper_2026/physics/rise_environment/` and production return-ascent scripts | Frozen return-ascent ledger, matched windows and saved inputs | Wind/environment/energy changes and matched contrasts |
| Departure-time environments and 48-hour recovery | `paper_2026/physics/recovery/` | 2,231 eligible model records, fixed folds and original draws | Censored hazards, supported policy contrasts and sensitivities |
| Basin, regional, latitude and historical comparisons | `paper_2026/newanalyses/history/agents/`, `spatial/agents/`; physics regional sources | Frozen cohort, identical support/reference distributions and joint environmental rows | Basin rates, global/basin trends and hemisphere recovery contrasts |
| Temporal asymmetry and stage-duration standardization | `paper_2026/physics/production/`; `paper_2026/newanalyses/history/agents/C/` | Matched peak-relative records; four stage-specific common supports | Asymmetry, F components, 16 standardized trends and q values |
| Statistical tables and current figure sources | Package summary entry points; `figure_map.csv`; `paper_2026/lifecycle_figures/` | Versioned source tables, editable geometry and original accepted panel assets | Current numbered tables and figure exports |

Detailed runnable commands and input/output names are in the three package
READMEs: [lifecycle and figures](paper_2026/lifecycle_figures/README.md),
[physical processes and recovery](paper_2026/physics/README.md), and
[basin, historical and latitude analyses](paper_2026/newanalyses/README.md).
Their source manifests bind the redistributed implementations to the actual
production sources; path adaptations are recorded separately from scientific
function bodies.

## Scientific conventions

* IBTrACS `USA_WIND` is a one-minute sustained wind in knots, sampled at
  00/06/12/18 UTC. The initial tropical-storm observation is selected before the
  study's latitude and minimum lifetime-gain criteria are applied. Whole SID
  histories are retained across calendar-year boundaries.
* Progress is `(V − Vg)/(LMI − Vg)`, using the actual initial wind `Vg`.
  The E boundary uses the accepted 0.25 progress rule. It has **no additional
  24-hour 10-kt intensification gate**. The current implementation replaces the
  early lifecycle implementation while retaining its callable interface.
* Display names use Development, Intensification, Near-peak persistence/Mature,
  and Decay (`Dec`). Historical machine fields such as `Res`, `resolution_time`
  and `final_departure` retain their identities. Temporary departure, the start
  of decay and the neutral first eligible departure are distinct analysis
  anchors; they must not be interchanged.
* The primary cohort is 2,340 SIDs. It contains 123 cyclones with 133 observed
  anchored returns. Diagnostic samples remain quantity-specific: the 117
  within-SID departure pairs, environmental supports, 2,231-record recovery model, and
  latitude training/reference samples are not interchangeable denominators.
* ERA5 inputs are on the native 0.25-degree grid at six-hour UTC sampling.
  Pressure masks, original-area coverage and joint valid-time conditions are
  applied before comparisons; missing values are not filled with zeros.
* Direct MWT uses 1,024 six-hour samples, symmetric boundaries and physical
  normalization. The nominal windows are >16 days, 2–16 days and <2 days;
  these are transform-window boundaries, not sharp Fourier filters. The main
  storage diagnostic is at 700 hPa. Wind, temperature and geopotential use
  13 levels. The base analysis uses direct omega at 500/700/850 hPa; the
  vertical extension additionally uses 400/600/925-hPa derivative support from
  the expanded direct-omega archive. Required fields are never filled with zeros.
  The reference state is shared between paired events and the shifted-window
  sensitivity. Vertical layer input uses 500–850-hPa boundary fluxes.
* Horizontal and pressure-direction canonical-transfer contributions and their
  sum are distinct from same-scale APE-to-KE conversion and boundary input.
  The extension reports all-source transfer for the target window; donor-window
  interactions have not been separated. The selected processes are not a closed
  energy budget. The prespecified numerical comparisons retain the main KE/APE
  signs; dry-APE magnitudes depend on the pressure discretization, and the finite
  transform/derivative combination does not satisfy machine-precision conservation.

## From raw observations

Obtain the frozen IBTrACS snapshot and ERA5 variables listed in
[`config/raw_inputs.json`](config/raw_inputs.json). [RAW_INPUTS.md](RAW_INPUTS.md)
describes the raw-data sequence and configurable directories. A bounded
lifecycle example is:

```sh
python -m src.raw_workflow.lifecycle --ibtracs /data/ibtracs.ALL.list.v04r01.csv --sid 1982005S10105 --out /results/lifecycle_case
```

Raw ERA5 processing requires the documented field catalogue, variable metadata,
native masks and frozen event requests. The production sources and manifest
setup instructions are supplied; a newly downloaded archive must pass its input
identity and coordinate checks. No command silently substitutes a changed input.
Use the prepared-input route for table reproduction without repeating extraction.
For fitting, explicitly invoke the relevant package's model entry points; their
fixed folds, draw counts, support rules and failure bookkeeping are preserved.

## Figures and editable sources

`figure_map.csv` gives the current number, generating entry, required input and
any final assembly step. Original numerical plot code and schematic geometry are
included. Some final compositions reuse accepted high-resolution panel assets.
The earlier layout-only changes to Fig. 7 and S5/S6/S7/S8/S31 (S31 is now S32) crop pure whitespace
and translate complete panels without resampling; their composition sources and
base assets are supplied. Arial is required for the original Arial-based styling
and is not redistributed. Consult each source package for its export command.
The MWT extension validates the new panels from saved statistics and preserves
all other accepted panels. Earlier figure workflows retain their recorded validation scope.

## Verification and citation

[VALIDATION.md](VALIDATION.md) reports the checks actually run, their tolerances,
and the distinction between deterministic reaggregation, local physical checks,
and unexecuted full refits. This is not a claim that the complete global
archive-to-manuscript workflow was rerun for packaging.

Cite the fixed [v2.1.5 Release](https://github.com/qq492947833/tc-near-peak-moisture/releases/tag/v2.1.5)
and its commit shown on that page. [CHANGELOG.md](CHANGELOG.md) describes changes
from the earlier code release. No repository-wide licence has been added or
changed; existing third-party licences and attribution are preserved in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No paper or archival DOI is invented.

## Historical releases

The original v1 notebooks and independent functions remain available for their
explicitly documented historical inputs. Their old figure numbering is not the
current figure catalogue. `fetch_data.py` and `run_notebooks.py` are the **legacy
v1 route**; the old instructions and mapping are preserved in `docs/legacy_v1/`.
The [v1.0.0 source-data asset](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v1.0.0/source_data.zip)
remains unchanged, as do tags v1.0.0 and v1.0.1.
