# Land contact in near-peak return events

This v2.1.6 supplement replays the retrospective land-contact check using the v2.1.5 event identities, original diagnostic samples and estimators. The reference cohort contains 2,340 cyclones, including 133 returns in 123 cyclones. The three public tables in Supplementary Data 1 contain the event-level audit, a support ledger with exclusion reasons, and six full-precision contrasts.

## Run

Use Python 3.11 with NumPy 1.26.4 and pandas 2.1.4. From this directory run the following command. The output directory can be supplied as an absolute path.

```sh
python -B run_land_check.py --out ./replay_output
```

The entry point checks the SHA-256 of each bundled input and reference table, reconstructs centre-contact flags on the native best-track timeline, applies the pre-specified two-sided exclusions, and compares all unrounded effect estimates and percentile endpoints with the archive. The output contains the event windows, paired water records, original and restricted contrasts, and a machine-readable run record. The bundled files suffice for an offline replay. The code reuses the original track-window classifier and production SID bootstrap estimators. The two files under `reference/` are used only for validation. `verified_run.json` records a run of this public entry point.

## Windows and comparisons

- Water uses temporary minus final departure for the 0–24 h integrated exchange across the moving 300 km boundary between 500 and 925 hPa. Both original paired members must have sufficient information and no identified centre contact from −24 through +24 h relative to their own departure. This retains 75 of 80 SID pairs in the unadjusted comparison. In Table S3 the 54 fully adjusted water pairs have a zero land covariate on both sides. Their original covariate used five six-hourly centre records from −24 h through departure and required all five. The code verifies that covariate for all 160 members of the parent sample. The 58 unadjusted water pairs and 76 energy SIDs in Table S3 have separate identities.
- Energy uses wind-matched reintensification minus decay at 700 hPa, 0–300 km, in the >16-day window. Each frozen event × 10-kt wind-bin pair retains its original comparison observations. A pair is eligible when both complete stages have sufficient information and no identified centre contact: the last wind minimum through actual near-peak return on the recovery side, and final departure through the final frozen decay observation on the decay side. Durations can differ. Matching and stage screening retain 71 of 108 SIDs, 74 of 115 return events and 110 of 165 original wind-bin pairs. The slow energy window is a diagnostic scale. The estimators use equal wind-bin weights within events, equal event weights within SIDs and equal SID weights.
- The full interval from temporary departure through actual return has an additional event-level flag, independent of the matched-stage filter and fixed departure windows. The packaged track contains 12,325 published observations with intervals up to three hours. A zero DIST2LAND identifies centre contact at a record. A zero LANDFALL identifies a contact within its forward interval ending at the next record. Only complete intervals ending on or before the window endpoint enter the check. Missing boundaries, invalid fields and gaps longer than three hours leave the status unknown unless positive evidence already establishes contact. A land-to-sea transition requires contact followed by a reported sea position within the window. These indicators inherit the spatial coverage of the IBTrACS coastline, including limitations for small islands.
- Both diagnostic families use 2,000 SID resamples. The water bank restores the original PCG64 state reached under seed 650117 and the original ordering of SID supports. For the restricted water sample the same state generates a bank of the restricted SID size. Energy uses the original seed 650117 directly on sorted SID means. Stored energy conversion rates are in W kg⁻¹. Multiply all three values (estimate and both limits) by 10⁴ for display in 10⁻⁴ W kg⁻¹.

## Data and code

| Path | Contents |
|---|---|
| `inputs/ibtracs_v04r01_tracks.csv.gz` | Seven original IBTrACS fields for the study’s 123 returning SIDs, preserving the source row order and native times. The subset retains MAIN and unlabelled tracks. |
| `inputs/event_windows.csv` | Frozen departure, trough, return and final departure times, IDs and original diagnostic membership; energy members also have a frozen final decay timestamp. |
| `inputs/water_members.csv` | Original water pair member IDs and anchors, historical covariate, geographic and selected short-follow-up exclusion sample flags. |
| `inputs/water_values_kg.csv` | Original within-SID differences and outward-export contributions in kg. The estimator converts to 10¹² kg. |
| `inputs/energy_wind_bin_pairs.csv` | Original matching IDs, winds, times, dry APE (J kg⁻¹), conversion (W kg⁻¹) for the two sides, full-sample weights and Table S3 membership. |
| `inputs/resampling.json` | Water generator state and original index-bank SHA-256 (row-major little-endian int64), seed and number of draws. |
| `reference/event_states.csv` and `reference/core_estimates.csv` | Previously verified event classifications and unrounded estimates/intervals, supports and draw counts. |
| `classify_track.py` and `estimators.py` | Centre-contact classification, paired-SID bootstrap and hierarchical wind-matched estimators. |
| `provenance.json` and `file_hashes.json` | Frozen source identities, baseline commit, versions and SHA-256 digests. |

Event times are UTC. SID is the IBTrACS storm identifier. A blank diagnostic-specific field means that the event lies outside the original diagnostic support; the categorical value `unknown` denotes insufficient evidence within a requested track window. Each sample retains the original event, matching and sign definitions of [v2.1.5](https://github.com/qq492947833/tc-near-peak-moisture/releases/tag/v2.1.5).

The track subset is drawn from NOAA/NCEI IBTrACS v04r01, using the frozen ALL-file SHA-256 recorded in `provenance.json`. Attribution: Gahtan, J., K. R. Knapp, C. J. Schreck, H. J. Diamond, J. P. Kossin and M. C. Kruk (2024), *International Best Track Archive for Climate Stewardship Project, Version 4r01*, NOAA National Centers for Environmental Information, 123 study SIDs (frozen file checked 26 November 2026). [Dataset](https://doi.org/10.25921/82ty-9e16) · [Methods, Knapp et al. (2010)](https://doi.org/10.1175/2009BAMS2755.1) · [Official record definitions](https://www.ncei.noaa.gov/sites/default/files/2025-09/IBTrACS_v04r01_column_documentation.pdf)

