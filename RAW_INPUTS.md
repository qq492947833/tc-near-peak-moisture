# Raw inputs and production order

The current raw lifecycle defaults are in `config/raw_inputs.json`; the
main-SID and six-hour record identities are those of the current 2,340-SID
analysis. The earlier v1 instructions remain under `docs/legacy_v1/`.

## Data acquisition and fields

* **IBTrACS ALL v04r01**: NOAA NCEI, DOI
  [10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16). The configuration
  records the exact CSV SHA-256 and size used in the study. Supply the original
  snapshot for exact cohort verification. Current provider updates are not
  silently treated as the same input. The wind field is `USA_WIND` in knots
  (one-minute sustained), with actual 00/06/12/18 UTC records and whole SID
  histories; preparation and missing-value rules are in `ibtracs_prepare.py`.
* **ERA5 pressure levels**: CDS DOI
  [10.24381/cds.bd0915c6](https://doi.org/10.24381/cds.bd0915c6).
  **Single levels**: [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47).
  Use the documented 0.25-degree native grid, six-hour UTC sampling and exact
  coordinate/time alignment. Variable groups are enumerated in the raw-input
  configuration and the physical production catalogues. Inputs include winds
  (m s−1), temperature (K), specific humidity (kg kg−1), relative humidity (%),
  geopotential (m2 s−2), pressure velocity (Pa s−1), surface/mean-sea-level
  pressure (Pa) and SST (K), with the process-specific surface fields.
* **Pressure levels differ by diagnostic.** PI profiles use native
  1000/925/850/700/600/500/400/300/250/200/150/100/50 hPa and the documented
  log-pressure interpolation of 70 hPa between 100 and 50 hPa. MWT u/v/T/Phi
  use the same 13 native levels, whereas omega is directly supplied at only
  500/700/850 hPa. The 700-hPa moisture vertical derivative uses q at
  600/700/850 hPa. Do not populate missing omega layers or interpolate time.
* **Land/ocean geometry**: Natural Earth physical datasets and the declared
  ERA5 land–sea mask. Use the specified resolution for each process and
  historical sampling route; the production configuration records the choice.
  Natural Earth data are available from https://www.naturalearthdata.com/.

## Raw-data sequence

1. Prepare the exact IBTrACS records, then call the current lifecycle function.
   `python -m src.raw_workflow.lifecycle --help` documents explicit SID selection
   and the separate full-cohort option. The default configuration points to the
   current main SID table and record comparison, not the earlier v1 sample.
2. Construct framework and threshold-comparison inputs with the actual sources
   in `paper_2026/lifecycle_figures/framework/`. Their README specifies the
   18 settings, frozen support, input files and original draw bank. The supplied
   processed inputs allow this step's summaries to be read without repeating
   classifications or environmental extraction.
3. Use `paper_2026/physics/configure_production.py` to create a **new working
   source tree** with explicit ERA5, IBTrACS and geometry paths. It validates
   coordinate hashes, variable names/units, sizes and saved scalar probes before
   recording local file identities. The physical package README gives the
   actual initializer, native MWT entry and required catalogues. This step does
   not generate scientific values or relax data checks.
4. Follow the production environment/request, paired moisture, MWT, return-
   ascent and recovery entry points in the physical package. Intermediate
   native-grid caches, coefficient products and per-SID tables can be reused.
   The distinct common-area masks, paired UTC lags and diagnostic support must
   remain attached to their values. The production files include the actual
   study-specific dependencies; some orchestration drivers retain Windows
   file-lock/priority operations. Independent numerical functions remain
   separately callable.
5. Feed those processed outputs into the newanalyses A/B/B_env/C and supported
   latitude models in their documented order. Those models also have supplied
   prepared inputs. Fitting/refitting and original bootstrap procedures are
   explicit, potentially expensive commands, separate from saved-result checks.
6. Generate numerical tables with the package summary routines and use
   `figure_map.csv` for figure source and composition order. Source figures and
   final panel layouts use their own documented outputs; no manuscript file is
   an input to the scientific calculation.

## What the release verification establishes

The delivered code has syntax/import/entry-point checks and representative
current-lifecycle, saved-statistical and local direct-MWT numerical checks.
One original ERA5 file was also used for the bounded metadata/probe preflight.
The complete global raw-data extraction, model ensemble and figure-rendering
campaign were not rerun for packaging. Full archive computation still requires
the external raw inputs, the documented working-space configuration and the
time/storage needed by the original procedures.
