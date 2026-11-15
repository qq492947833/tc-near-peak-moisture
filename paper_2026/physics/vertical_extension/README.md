# Vertical canonical transfer and pressure-layer energetics

This package contains the calculation and statistical sources for the direct-omega extension of the lifecycle energy diagnostics. It adds pressure-direction and total canonical transfer of horizontal kinetic energy (KE) and dry available potential energy (APE), together with consistent 500–850-hPa reservoirs, conversion and time integrals. The input population remains the 1982–2023, 2,340-SID cohort. Diagnostic subsets retain their own recorded support.

## Scientific quantities

- `Gamma_K_h`, `Gamma_A_h`: the original all-source horizontal canonical-transfer contributions on the sphere.
- `Gamma_K_p`, `Gamma_A_p`: pressure-direction contributions calculated from the transformed original products omega*u, omega*v and omega*T-prime. Transforming a product is distinct from multiplying two transformed variables.
- `Gamma_K_total`, `Gamma_A_total`: horizontal plus pressure-direction contributions on identical valid support. Positive transfer supplies the target time-scale window; it does not identify a particular donor window.
- `APE_ref_correction`, `APE_compressibility`: separate reference-state terms. They are not included in vertical canonical transfer or labelled latent heating.
- `KE`, `APE`, `C_A_to_K`: horizontal KE, dry APE and same-window APE-to-KE conversion. Positive conversion supplies horizontal KE.
- `vertical_input_K_Wkg`, `vertical_input_A_Wkg`, `vertical_input_P_Wkg`: vertical KE, APE and pressure-flux convergence through the existing 500/850-hPa boundaries, divided by layer mass. These are spatial boundary inputs, distinct from transfer between temporal windows.

The transform keeps 1,024 six-hour samples, symmetric extension, the accepted physical normalization, `j0=5`, `j1=8`, original time anchors and reference state. Nominal windows are >16 d, 2–16 d and <2 d. The four-level pressure mean uses 500, 600, 700 and 850 hPa with trapezoid weights 1/7, 2/7, 5/14 and 3/14. Pressure derivatives also use the required 400/925-hPa support. The available omega archive has 13 levels; this calculation uses only the levels required by its layer and derivative stencils.

Omega is pressure velocity read directly from ERA5 in Pa s-1. It is not geometric vertical velocity and its square is not added to horizontal KE. Reservoirs and rate integrals are in J kg-1; conversion, transfer and layer-input rates are in W kg-1. Column representations multiply the layer mean by 35,000/9.8 kg m-2. A layer input is not combined with a 700-hPa reservoir to claim budget closure.

## Sources and directories

| Analysis | Entry point | Input | Output under `output_root` |
|---|---|---|---|
| Bounded direct-omega extraction | `scripts/extract_vertical.py` | Original coefficient caches, operator and ERA5 files; frozen event requests | `diagnostics/<SID>/request_values.parquet` and completion identities |
| Pure pressure calculations | `scripts/vertical_terms.py` | Explicit arrays and pressure/reference profiles | Returned arrays; no file I/O |
| Direct original spherical diagnostics | `scripts/theory_MWT_Atmosphere_Direct.py` | Explicit coefficient arrays and basic state | Returned diagnostics; no file I/O |
| Paired departures, actual rise and shared-wind comparisons | `scripts/raw_statistics.py` | Verified regional requests and `inputs/legacy_mwt` | `statistics/raw_summary.*`, saved draws and unit-support tables |
| Background adjustment | `scripts/adjusted_statistics.py` | Collected primary requests and `inputs/background` | `statistics/adjusted/` |
| Paired discrete/layer sensitivities | `scripts/derive_sensitivity.py` | Exact request masks and saved unit values | `statistics/sensitivity_*` |
| CSV and support-ledger export | `scripts/curate_diagnostics.py` | Saved statistics | `outputs/` |
| Figure reconstruction | `plots/` | Full-precision saved values | See `plots/README.md` |

`read_era5_fixed_block.py` and `_adjusted_legacy_helpers.py` are included, so these entries do not require private Python modules. `_adjusted_legacy_helpers.py` preserves the fitted-support, weighting and estimability procedures used in the accepted background comparison. `source_manifest.json` records source and distributed-file identities. Small frozen event, interval, request, covariate and comparison inputs are under `inputs/`; large ERA5 fields and coefficient caches are not distributed.

## Environment and configuration

The verified environment is Python 3.11.7 with the package versions in `requirements.txt`:

```sh
python -m pip install -r paper_2026/physics/vertical_extension/requirements.txt
```

Work from this package directory. Copy `config.example.json` to `config.local.json`, edit its paths, and set `FLTCCF_VERTICAL_CONFIG` to that file. Relative paths are resolved against the configuration file's directory. `environment.example` lists optional process-local settings.

- `output_root`: a new writable run directory, separate from accepted inputs.
- `mwt_root`: the prepared original MWT workspace containing the operator, original per-SID `cache/<SID>/diagnostics.h5`, and `addendum_20261023/cache/<SID>/increment.h5`.
- `template_root`: the bundled event/support and old-summary tree; normally leave unchanged.
- `background_root`: bundled covariates, original outcomes and saved background validation inputs.
- `rise_input_root`: the existing repository's `../mwt_statistics/inputs/rise`; it supplies the accepted spatial requests and roles.
- `era5_root`: native six-hour, 0.25-degree ERA5 NetCDF files on the original time/grid convention.

The original inventory is read from `mwt_root/reports/data_era5_inventory.json`. Relative paths and `${RAW_ERA5}` inventory paths are supported. Existing u/v/T/geopotential data use all 13 original pressure levels. New direct-omega files follow `Vertical-velocity-{level}hpa-1980-2025.nc`; only the bounded requested time/spatial blocks at needed levels are read. The ERA5 archive's longer coverage does not extend the analysis past 2023. Missing required observations are not filled or rescaled.

## Running from prepared coefficients

```sh
python -B scripts/extract_vertical.py --workers 4
python -B scripts/raw_statistics.py
python -B scripts/assemble_adjusted_requests.py
python -B scripts/adjusted_statistics.py --request-values generated/statistics/adjusted_requests.parquet
python -B scripts/derive_sensitivity.py
python -B scripts/curate_diagnostics.py
```

The example assumes `output_root` is `generated`; use its configured path for explicit command arguments. The extraction entry can use `--limit 1 --pilot` for a bounded overlapping-coefficient check. A complete population run is required before the full statistical driver accepts its frozen 123-SID input set. Do not present a limited extraction as a population result.

Extraction writes reusable omega-product caches with exact time/grid/reference identities. Current completed outputs are retained. `scripts/preflight_inputs.py` records input metadata without computing fields. `scripts/bind_cache.py` records reusable completed products; `extract_vertical.py --resummarize` supports the matching recorded cache schema and does not regenerate ERA5 fields. The existing complete derivation is already version 2 with mask schema 3.

The three original comparisons remain separate: 117 paired departure SIDs; 133 actual return events in 123 SIDs; and 165 original shared-wind matching units in 115 events/108 SIDs. The adjusted design re-estimates its established support and model within each SID bootstrap draw. New common-support diagnostics can have fewer valid events; support ledgers preserve excluded and partially retained units. The figure's joint process/transfer integral subset has 132 events in 122 SIDs.

## Starting from raw data

Use the existing `paper_2026/physics/configure_production.py` initializer and documented native-production route to prepare the original operator and coefficient caches. The relevant included sources are `production/mwt/scripts/theory_build_operators.py`, `run_mwt_sid.py`, and `production/mwt/addendum_20261023/scripts/run_mwt_rise_increment.py`. The first two produce the original operator/departure caches; the last extends the exact same transform/reference to the frozen rise requests. Point `mwt_root` at that prepared workspace and then run this package. Its HDF5 input schema is the schema produced by those files, not an alternative implementation inferred from manuscript text.

Raw ERA5 pressure-level data are available from the Copernicus Climate Data Store. Organize them into the native inventory schema, including `w` read directly from ERA5 at the required levels and the declared units, six-hour times and matching latitude/longitude grid. IBTrACS v4r01 preparation and the frozen main-SID list remain in the parent analysis package. No new lifecycle classification is performed here.

## Reusing processed results and verification

`results/` contains full-precision accepted summaries; `plots/` uses those values without rerunning experiments. A local processed run with its `diagnostics/<SID>/request_values.parquet` and matching completion identities can enter at `raw_statistics.py`; one with all saved statistics can enter at `curate_diagnostics.py`. The package includes saved small-example requests, event integrals and saved population draws for a bounded readback:

```sh
python -B scripts/verify_saved_example.py --output generated/verification/VALIDATION.json
```

This verifies syntax/imports, 14 predefined analytical checks, exact frozen support counts, one saved SID's time integration, and saved population intervals/P values/additivity. It neither draws a new bootstrap sample nor fits a model. `VALIDATION.json` describes the verification actually performed for this version; a full raw-archive rerun was not part of release verification.

Scientific uncertainty uses the original 2,000 SID draws, event/bin weighting, percentile intervals and centered plus-one P calculation. The primary new test families remain complete; descriptive joint-support and numerical sensitivities have q=NA. Rank-deficient coefficient designs are distinguished from an estimable phase contrast. Layer/pressure-stencil sensitivities preserve exact common grid masks and time support. The finite discrete operator does not imply exact pointwise or pressure-only cross-window cancellation.

The independent function signatures and numeric implementations are retained. Configuration and output routing use explicit paths; the curation entry exports the saved scientific statistics and support ledgers. Historical machine identifiers such as `Resolution` remain where required to bind unchanged source records; their display meaning is Decay. No sample, threshold, comparison sign or frozen value is changed by this packaging.
