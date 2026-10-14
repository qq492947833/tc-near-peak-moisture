# Original inputs and calculation entry points

The ordinary notebooks reproduce the publication from the included frozen
scientific tables. The optional functions below expose the original-input
calculations. Large ERA5 and IBTrACS archives are not included or downloaded by
this repository. Supply their locations explicitly; all output directories must
be new. See `config/raw_inputs.json` for exact variables, units, levels, filenames
and sampling rules.

## Inputs

| Input | Identity and use |
|---|---|
| IBTrACS | v04r01, `ibtracs.ALL.list.v04r01.csv`, 331,329,717 bytes, SHA-256 `85d421e69e6e963ae4e6025d3e270d8c3e5edf81b981fa06bc7b753aa62fc5a3`. Dataset DOI [10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16). This fixed snapshot is required; the current online file can differ. |
| ERA5 pressure levels | [CDS pressure-level product](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels), DOI [10.24381/cds.bd0915c6](https://doi.org/10.24381/cds.bd0915c6). The local source grid is 0.25 degrees and times are 00, 06, 12 and 18 UTC. |
| ERA5 single levels | [CDS single-level product](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels), DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47). Surface pressure, mean sea-level pressure, SST, land-sea mask, sampled accumulations and total column water vapour. |
| Ocean geometry | Natural Earth [1:10m physical land](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/) complement for SST. ERA5 land-sea mask below 0.5 for surface fluxes. These masks are distinct. |

The analysis uses framework genesis years 1982–2023 and retains complete
cross-year tracks. Genesis is the first eligible tropical record reaching
35 kt, subject to the stated lifecycle criteria; it is not storm formation in
the unrestricted physical sense. `USA_WIND` is the 1-minute wind series used
throughout. Other agencies' wind measurements are not spliced into it. The
original stage function and exact-6-hour R/U/F representation are separate
functions because they have different interval rules.

Environmental fields use an original-area denominator. Pressure-level cells
require finite simultaneous surface pressure at least as large as the level
pressure, in Pa. The scalar annulus is 200 <= radius < 800 km, with at least 90%
valid original area. Times are exact source times, with no nearest-time fallback.
The field catalog records coordinate and time-variable names for each file.
Six-hourly sampling of one-hour surface accumulations is not a continuous
six-hour water or energy budget.

## Explicit commands

From the repository directory, use the installed reproducibility environment:

```sh
python -m src.raw_workflow.lifecycle --ibtracs /data/ibtracs.ALL.list.v04r01.csv --sid 2015188N08194 --sid 2015293N13266 --out /results/lifecycle_cases
python -m src.raw_workflow.environment --era5-root /data/era5 --utc 1982-01-13T12:00:00Z --lat -16 --lon 71.8 --field q700 --out /results/annulus_case
python -m src.full.transport --era5-root /data/era5 --event-id EVENT_ID_FROM_EVENTS_FROZEN --lag 0 --out /results/boundary_case
python -m src.raw_workflow.physics rh --era5-root /data/era5 --input /requests/rh_pair.json --out /results/rh_pair
python -m src.raw_workflow.physics budget --era5-root /data/era5 --input /requests/budget_time.json --out /results/budget_time
python -m src.raw_workflow.physics pi --input /requests/pi_scalars.json --profile /requests/augmented_profile.npz --wind-kt 85 --out /results/pi_case
```

`lifecycle --all-frozen` is an explicit full frozen-SID calculation. There is no
implicit full-cohort launch. For the exchange command, take the event identity
and original centre history from `source_data/transport/events_frozen.csv` and
`centres_frozen.csv`. Interpolated boundary nodes describe spatial quadrature;
they do not increase the source's temporal resolution.

The RH request JSON contains `sid`, `boundary`, `UTC0`, `UTC1`, `lat0`, `lon0`,
`lat1`, `lon1`; UTCs are exactly 24 h apart. The function computes 500 and 700 hPa
at all four position/time combinations. Within each domain it fixes both the
valid-cell mask and original cell-area weights across the two times. It reports
the symmetric temporal and domain components, temporal q/T contributions,
and support change separately. The old common-cell remainder is not renamed a
pure geographical effect.

The budget request contains `UTC`, `lat`, `lon`, `lag_h`, `motion_u_ms`,
`motion_v_ms`, `prior_lat`, `prior_lon`. The prior position must be the actual
position at UTC minus 6 h. `budget_nodes.integrate_pair` receives two event
sequences, each containing the five `budget_request` outputs at 0, 6, 12, 18 and
24 h. It fixes support across both events, all times and all fields. The native
700 hPa terms are also exposed in `budget_native.terms`; all signed terms and
the remainder are retained. A residual defined by subtraction is not an
independent closure check. The node endpoint calculation is a distinct
consistent-geometry sensitivity, not a replacement for the original native
105-pair budget estimand.

The PI scalar JSON contains `SST_K`, `msl_Pa` and, when available,
`shear_200_850_ms`, `core600_saturated_entropy_jkgk`,
`environment600_entropy_jkgk`, and boolean `ventilation_support_confirmed`.
The NPZ contains the original arrays `pressure_hpa`, `temperature_K`,
`specific_humidity_kgkg`, `joint_Tq_valid_fraction`, plus `pressure_aug_hpa`,
`T_logp_K`, `q_logp_kgkg`, `T_plinear_K`, `q_plinear_kgkg`, `joint_fraction_aug`.
`pi_profile.augment_from_native` creates the augmented fields from the four
aligned 100/50-hPa grid fields, original area and original annulus mask.

The thirteen required native PI levels are 1000, 925, 850, 700, 600, 500, 400,
300, 250, 200, 150, 100 and 50 hPa. The 70-hPa level is derived by interpolating
T and q values linearly in log pressure on joint native support before averaging;
it is not a native observation or log-q interpolation. The sole alternative is
linear-pressure interpolation. All original levels are retained, including
unsupported ones as missing. The physical gate requires supported actual CAPE
evaluations and interior outflow, and preserves explicit failure reasons. The
inclusive 50-hPa endpoint adaptation is visible in `pi_trace.py`; the vendored
upstream source remains unchanged. PI is calculated from an area-mean profile,
not averaged from gridded PI. The full native profile/SST-mask assembly remains
caller supplied; `environment.py` alone is not that assembly.

## Calculation coverage

| Layer | Executable implementation | Limited verification performed for this release | Remaining full-workflow scope |
|---|---|---|---|
| Raw IBTrACS to lifecycle and R/U/F | `raw_workflow.lifecycle`, original preparation, stage and representation functions | Two fixed SIDs, 77 records; all compared frozen fields agree | Complete raw frozen cohort not rerun |
| Native pressure-level annulus | `raw_workflow.environment`, `native_grid` | One real q700 request, 2,539 valid cells; mean 0.005122457057061452 kg/kg | Cohort request scheduling and full extraction not rerun |
| Moving-boundary gross in/out/net | `full.transport` and original geometry functions | One real time, 16 fields, nine quantities match frozen values | All event/time windows not rerun |
| RH symmetric time/domain and q/T | `raw_workflow.humidity`, `physics.rh_request` | Existing native caches: one SID, D/E and E/F, both levels; all 15 reported numeric fields agree in four rows | Raw four-combination cohort extraction not rerun |
| Native 700-hPa budget | `raw_workflow.budget_native`, original native derivatives | Shared original derivative functions ported; no fresh full native-budget run | Cohort support/aggregation remains separate from primitive fields |
| Moving-node endpoint sensitivity | `raw_workflow.budget_nodes`, `physics.budget_request` | Existing native caches: one SID, two events, both rings, all five times; 16 reported numeric fields agree in four rows | Entire consistent-geometry sensitivity not rerun |
| Environmental-profile PI and signed entropy | `raw_workflow.pi_profile`, `potential_intensity`, `entropy` | Two existing profiles, one supported and one failed; both interpolation variants preserve values and acceptance/failure; native70 augmentation agrees | Full raw native-profile and distinct surface-mask assembly is not packaged as a one-command cohort driver |
| Recovery model and uncertainty | `analysis.recovery_frozen`, `recovery_scenario_functions`, `full_refit` | Quick reproduction uses frozen predictions and saved draws, without loading pickles | Optional full refit is explicit and has not been rerun for repository packaging |

The included scientific calculations therefore do not constitute a newly
executed archive-to-manuscript pipeline. Climatology construction, all extraction
queues, cohort joins and every original fitting job are not silently launched by
these commands. Included frozen tables retain the full published estimates and
countercomparisons. No 925-hPa omega or air-parcel trajectory is required or
claimed by these calculations.

## Model environments

Quick table reproduction does not load model pickle files. Original primary
models were fitted with Python 3.11.7, NumPy 1.26.4, pandas 2.1.4, SciPy 1.11.4
and scikit-learn 1.2.2. The record-scenario models record Python 3.13.7, NumPy
2.2.0, pandas 2.3.3, and pickle metadata identify scikit-learn 1.7.2; historical
SciPy/threadpool versions for those models were not recorded. These are distinct
environments. The current installation is not represented as a universal pickle
compatibility environment. `analysis.full_refit.recovery_refit` requires explicit
`confirm_full=True`, uses the fixed 1,000 SID-index arrays, and saves failed
attempts without drawing replacements. Deserializing the archived models additionally requires restoring or mapping the historical `__main__` (primary) and `fit_policies_v6` (scenario) names for `Transform`, `Basis` and `IntervalHazard` to their corresponding scientific implementations; ordinary `pickle.load` is not a verified loading route.

The PI adapter defaults to tcpyPI's supported pure-Python mode
(`TCPYPI_DISABLE_NUMBA=1`), including the bounded comparisons above. NumPy is
required; Numba is optional. An explicit environment setting before import can
enable Numba when its compatible dependency versions are installed.
