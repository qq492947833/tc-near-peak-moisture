# FLTCCF: tropical-cyclone lifecycle and near-peak processes

Code, configurations and processed inputs for the 1982–2023 study of 2,340
tropical-cyclone storm identifiers (SIDs). The current release covers lifecycle
classification and comparisons, environmental evolution, paired temporary
departure and decay conditions, moisture transport, multiscale energy,
reintensification and recovery, spatial contrasts, and historical duration changes.
The figure catalogue follows the current **8 main and 42 supplementary figures**.

**Stable version:** [v2.0.0](https://github.com/qq492947833/tc-near-peak-moisture/releases/tag/v2.0.0)
(13 November 2026). Older tags remain unchanged. This repository has no assigned
paper DOI or software-archive DOI.

## Start from the versioned processed inputs

Use Python 3.11 for the current scientific validation environment. Work in a
writable copy and choose local paths for inputs, outputs and caches.

```sh
git clone --branch v2.0.0 https://github.com/qq492947833/tc-near-peak-moisture.git
cd tc-near-peak-moisture
python -m venv .venv
# Activate .venv using the command appropriate for your operating system.
python -m pip install -r requirements.txt
python fetch_release_data.py
python verify_release.py --output validation/summary.json
```

`fetch_release_data.py` downloads the versioned processed-input and figure-source
assets, verifies archive/member SHA-256 values, and restores their relative paths.
For an offline run, download the two assets from the Release page and pass each
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
| MWT storage, conversion, horizontal canonical-transfer contributions and layer input | `paper_2026/physics/direct_mwt/`, `mwt_statistics/`, `production/mwt/` | `direct_mwt/config/direct_diagnostics.json`; saved coefficients and per-SID values | Energy diagnostics, paired estimates and original statistical families |
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
  13 levels, whereas observed omega is supplied only at 500/700/850 hPa.
  The reference state is shared between paired events and the shifted-window
  sensitivity. Vertical layer input uses 500–850-hPa boundary fluxes.
* Horizontal canonical-transfer contributions, same-scale APE-to-KE conversion,
  storage change and boundary input retain distinct definitions, units and
  signs. This release does not imply a complete three-dimensional canonical
  transfer or a closed surface-wind energy budget.

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
The last layout-only changes to Fig. 7 and S5/S6/S7/S8/S31 crop pure whitespace
and translate complete panels without resampling; their composition sources and
base assets are supplied. Arial is required for the original Arial-based styling
and is not redistributed. Consult each source package for its export command.
The release checks did not rerender the accepted manuscript figures.

## Verification and citation

[VALIDATION.md](VALIDATION.md) reports the checks actually run, their tolerances,
and the distinction between deterministic reaggregation, local physical checks,
and unexecuted full refits. This is not a claim that the complete global
archive-to-manuscript workflow was rerun for packaging.

Cite the fixed [v2.0.0 Release](https://github.com/qq492947833/tc-near-peak-moisture/releases/tag/v2.0.0)
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
