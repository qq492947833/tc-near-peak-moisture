# Supplementary figures

Run from the repository root using its documented Python environment (Matplotlib, NumPy, pandas and Pillow).

```sh
python paper_2026/near_peak_event_figures/supplementary_figures.py --repo . --out output/supplementary
```

To render only the threshold sensitivity figure, add `--figures 1`. The script exports 900 dpi RGB PNG and losslessly compressed TIFF at a width of 180 mm. Use `--diagnostics PATH` to also save previews, vector PDFs and physical axes bounds.

The threshold comparison uses frozen mean later near-peak occupancy (L) and weakening–recovery bridge duration (B), in hours. The threshold-specific subsets contain 150 cyclones at 0.85 and 109 at 0.95. The main and alternative definitions use the same 123 cyclones for the paired comparison at each threshold. The reported 95% intervals are percentiles of 2,000 original SID resamples. The full-precision rows and the SID memberships are in `inputs/threshold_duration_summary.csv` and `inputs/threshold_duration_membership.csv`. All other panels use the original physical units, supports and statistical intervals listed in their sources.

`supplementary_figure_inputs.csv` identifies the source files and their checksums. `supplementary_figure_mapping.csv` relates the previous 11-figure sequence to the current ten figures and records the checksums of the published image pairs.
