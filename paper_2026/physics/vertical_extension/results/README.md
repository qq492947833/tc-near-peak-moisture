# Saved vertical-transfer and layer diagnostics

These CSV files contain the saved estimates and uncertainty summaries for the
vertical extension of the MWT analysis. Values retain their source precision.
Empty fields represent unavailable or inapplicable quantities, including q
values for descriptive comparisons. They do not represent zero.

| File | Contents |
|---|---|
| `MWT_ALL_STATISTICS.csv` | All 5,076 raw statistical summaries across comparisons, scale windows, regions and discretizations. |
| `MWT_ADJUSTED_WIND_MATCHED.csv` | The six selected layer diagnostics and their raw, common-support, background-adjusted and adjustment-change summaries. |
| `REINTENSIFICATION_RATE_INTEGRALS.csv` | All saved time integrals over the observed valley-to-return intervals. |
| `NUMERICAL_SENSITIVITY.csv` | Paired comparisons of pressure discretizations, derivative templates and single-level versus layer diagnostics. P values are descriptive; no new formal BH family is assigned. |
| `MWT_COLUMN_INTEGRALS.csv` | Pressure-column representation of the same layer estimands, using layer mass 35,000/9.8 kg m⁻². These are not independent additional tests. |
| `OLD_METRICS_ON_NEW_SUPPORT.csv` | Original 700-hPa APE and conversion estimates on original, new common and new conditional supports. |
| `METRIC_DEFINITIONS.csv` | Metric definitions and the layer-mass conversion factor. |

`variant`, `comparison`, `window`, `level`, `region` and `metric` jointly identify
a diagnostic. Scale windows 0, 1 and 2 denote nominal >16 d, 2–16 d and <2 d.
Regions `r0_300`, `r200_800` and `r0_500` are radial distances in kilometres.
Pressure levels are in hPa. Reservoirs and time integrals are J kg⁻¹; conversion,
canonical transfer and layer-mass-normalized input rates are W kg⁻¹. The column
table uses J m⁻² or W m⁻².

Pre-event contrasts are temporary departure minus the start of decay. Wind-
matched contrasts are reintensification minus decay. Time integrals use the
actual legal intervals in seconds. Support and valid bootstrap counts belong
to each specific target; they must not be replaced by the event-domain totals.

The `source_path` and `source_file` references are relative to the
`vertical_extension` directory. `generated/diagnostics` and
`generated/statistics/...` denote outputs of a configured analysis run, not
additional undisclosed input files. The original background request values are
provided under `inputs/background/results/request_values.parquet` through the
release input bundle. The nine-row figure table uses `results/MWT_ALL_STATISTICS.csv`
as its public lookup source and retains the original `draw_key` and statistical
identity fields.

The finite discrete transform and derivative operators retain the numerical
marginalization residuals documented with the analysis; exact local energy
closure is not assumed.
