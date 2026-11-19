# Trough-aligned KE canonical-transfer rates

Figure 6e shows horizontal, pressure-direction and total canonical transfer of horizontal kinetic energy in the 2–16-day window. At each six-hour observation the values are area averages within 300 km of the cyclone centre and pressure-thickness means through 500–850 hPa. Rates are in W kg⁻¹. Each series uses the same 126 return events in 117 cyclones at −24, −18, −12, −6, 0, +6, +12, +18 and +24 h relative to the pre-return wind minimum. Some later observations follow the return to near-peak intensity.

The event integrals in Supplementary Fig. 26a are in J kg⁻¹. They cover each event's own minimum-to-return interval and use 132 events in 122 cyclones.

## Draw the panel

From this directory, use Python 3.11 and the versions in [the vertical-extension environment](../requirements.txt).

```sh
python -B plot_fig6e.py --out fig6e_figure
```

The script selects `nine_time_common` from `data/FIG6E_RATE_ESTIMATES.csv`, checks its source identity and writes `Figure_06e_KE_transfer_rate.png` and a lossless TIFF. Each point is the mean across SIDs at one observed lag. Shading shows the 2.5th and 97.5th percentiles of the saved SID resampling distribution for each rate and lag. The full-figure assembly uses this panel and the previous figure art; see the release instructions in the repository README.

## Reproduce the rates summary from saved event observations

```sh
python -B summarize_curve.py --out fig6e_statistics
```

`data/FINAL_EVENT_HOUR_RATES.csv` holds the 1,188 requested event observations, including missing observations and joint spatial support. `FIXED_NINE_EVENT_MEMBERSHIP.csv` identifies the events with all nine observations. For each lag, the program first averages repeated events within a cyclone, then gives each contributing SID equal weight. It reads the original 2,000 SID index rows from `ORIGINAL_SID_DRAW_INDICES.npz`, retaining the multiplicities of eligible SIDs from each original draw and dividing by their retained total multiplicity. This preserves joint horizontal, pressure-direction and total estimates within each draw. It writes the complete lag table, SID means, event membership and draw-level summaries to the chosen directory. It checks all 54 estimates and interval pairs against the distributed values. The second set of 27 rows (`available_at_time`) describes lag-specific support and is provided for comparison. No random draws are needed.

## Recover the missing pressure-product coefficients from matching prepared inputs

The two entry points below are for researchers with the [vertical-extension inputs and direct-MWT caches](../README.md). The released event observations and rate table are sufficient for the previous two commands.

Configure `FLTCCF_VERTICAL_CONFIG` as in the parent package. Set `era5_root` to the original six-hour u, v and pressure-velocity files and `mwt_root` to the original prepared direct MWT tree. Required files are the physical N=1024, j0=5, j1=8 symmetric operator under `cache/`, per-SID `addendum_20261023/cache/<SID>/increment.h5`, and the original `cache/<SID>/diagnostics.h5` when a requested time comes from that cache. The extension's configured `output_root/diagnostics/<SID>/` supplies the previous overlapping pressure-product NPZ files. Preserve the input filenames, file identities and per-SID coordinate/reference identities recorded in `data/native_input_identity.json`. The native entry checks the recorded byte size and modification time of its nine large NetCDF inputs as well as the operator and overlapping coefficient identities. The raw meteorology and these large prepared caches are external inputs.

```sh
python -B recover_native.py --workers 2
python -B diagnose_rates.py once
python -B diagnose_rates.py final
```

Coefficient and per-SID outputs go under the extension's configured `output_root/fig6e_rates/`; set `FLTCCF_FIG6E_OUTPUT` to use a separate output directory. The missing-request table indexes 376 event observations (363 unique SID/UTC combinations). The extractor processes the 400, 600 and 925 hPa transformed products of pressure velocity with u and v needed by the established six-level derivative support. It preserves the 1,024-sample input, symmetric extension, physical normalization, native grid, 1,088-sample stable mask and frozen reference state. A previously cached observation in each affected SID is used for overlap verification. The diagnosis uses the parent package's sphere and pressure operators. The four pressure levels in the layer mean are 500, 600, 700 and 850 hPa. At least 80% valid area is required on the joint support.

`data/existing_request_values.parquet` and `existing_centres.parquet` are selected rows from the accepted request statistics and centres for this set of requests. Together with the prior event-hour ledger, these preserve the original rates and overlap checks without requiring a population-wide statistical rerun. Supply the recovered ledger and its matched verification record to replay the summary:

```sh
python -B summarize_curve.py --rates PATH_TO_OUTPUT/diagnostics/FINAL_EVENT_HOUR_RATES.csv --qa PATH_TO_OUTPUT/diagnostics/FINAL_EVENT_HOUR_QA.json --out fig6e_statistics
```

The dataset uses Unix seconds for `valley_s`, `return_s` and `sec`. `within_rise` marks the interval from the actual wind minimum through the observed return, including both endpoints. The table contains 1,178 valid event observations among the 1,188 requests. The rate curves retain only events with all nine lags. The input hashes are listed in `data/source_manifest.json`. See the parent package for the method references and existing numerical sensitivity results.
