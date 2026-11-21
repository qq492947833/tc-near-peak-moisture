# Near-peak cyclone events: figure reproduction

The presentation version figures-20261121-r03 uses the frozen v2.1.5 analyses of near-peak departures and returns. The reading set contains six main figures, ten supplementary figures and seven supplementary tables.

## Rendering

Use Python with the plotting dependencies from the repository, including NumPy, pandas, matplotlib, Pillow and pyshp. Figures use Arial at the exported display width. Run from the repository root in a checkout of this figure tag:

```sh
python paper_2026/near_peak_event_figures/main_figures.py --repo . --out generated/event_figures
python paper_2026/near_peak_event_figures/concept.py paper_2026/lifecycle_figures/figures generated/event_figures
python paper_2026/near_peak_event_figures/supplementary_figures.py --repo . --out generated/event_figures
```

The first command draws main figures 1–5, the second draws the synthesis (figure 6), and the third draws supplementary figures 1–10. Use `--figures 1` on the first or third command to render the event definitions or the L and B duration sensitivity alone. Outputs are PNG and lossless RGB/LZW TIFF at 180 mm and 900 dpi. File checksums identify the deposited images; local fonts, encoders and library versions can affect rendered bytes.

The source trees `paper_2026/lifecycle_figures`, `paper_2026/selected_evidence` and `paper_2026/physics/vertical_extension/fig6e_rates` retain their historical identifiers. The small `inputs/plot_assets.zip` carries the v2.1.5-bound pre-departure time series and Natural Earth 1:110m public-domain coastline components from the earlier figure input bundle. The inputs directory includes frozen occupancy and bridge-duration summaries, SID memberships, and sampling-sensitivity records. Each renderer selects the original support and units for each panel.

Figure 5e shows KE canonical-transfer rates at nine six-hour lags relative to the wind trough on a fixed set of 117 SIDs and 126 events (10⁻⁴ W kg⁻¹). Supplementary Figure 8b shows event-specific integrals over the trough-to-return interval in 122 SIDs and 132 events (J kg⁻¹). The supporting data use the pre-existing SID-level summaries and uncertainty intervals.

[Display map](display_map.csv) · [Image checksums](figure_files.csv) · [Frozen statistical archive index](statistical_archive.json) · [Release](https://github.com/qq492947833/tc-near-peak-moisture/releases/tag/figures-20261121-r03)

[Supplementary figure sources and duration definitions](SUPPLEMENTARY_FIGURES.md) · [Supplementary figure crosswalk](supplementary_figure_mapping.csv) · [Complete plotting input index](figure_sources.csv)
