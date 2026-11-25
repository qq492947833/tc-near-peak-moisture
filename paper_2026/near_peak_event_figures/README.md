# Near-peak cyclone events: figure reproduction

The presentation version figures-20261125-fig6-r02 uses the frozen v2.1.5 analyses of near-peak departures and returns. The reading set contains six main figures, ten supplementary figures and seven supplementary tables.

Figure 1 uses a six-hourly schematic to distinguish the reference near-peak episode, the subsequent weakening and reintensification, and final decay. Figure 3 uses matched panel bounds to compare moving-boundary moisture exchange.

Figure 6 uses steel blue (#2F6B9A) for temporary departure in panels a–c. Event labels are centred on their shaded intervals in panel a. The top row marks temporary departure, trough, return and final departure at the original six-hourly observations. The upper-left cylinders compare pre-departure shear and outer moisture; inner-ring vertical advection refers to 700 hPa over the subsequent 0–24 h. The lower-left panel shows gross inward, gross outward and net inward exchange across the 300 km boundary (500–925 hPa, 0–24 h), using the unadjusted, within-cyclone comparison of 80 pairs. Both outward-transport decomposition bars use one linear scale. The right column separates pre-departure pressure-surface and layer diagnostics, trough-to-return energy integrals (132 events, 122 cyclones) and wind-matched 700 hPa dry APE differences (115 events, 108 cyclones). The remaining fifteen PNG/TIFF pairs are unchanged.

For the boundary-exchange panel, the frozen input is `paper_2026/lifecycle_figures/figures/Figure_08/input/paired_flux_components.csv`. Select `cohort=original_post80`, `window=0_24h`, `factor=1`, `layer=all`, and `sector=all`. Class means use `quantity=temporary/final` for `kind=in/out/net`. The signed decomposition uses `kind=out` and `quantity=humidity/relative_motion`; all estimates are in kg and use the same unadjusted support. Signs refer to temporary minus final departure. Adjusted estimates are reported separately in Figure 3d and Table S2.

Render Figure 6 only (from the tagged repository root):

```sh
python paper_2026/near_peak_event_figures/concept.py paper_2026/lifecycle_figures/figures generated/figure6 --qa generated/figure6/qa
```

Figure 6 was rendered with Python 3.11.7, Matplotlib 3.8.0, NumPy 1.26.4, pandas 2.1.4, Pillow 10.2.0 and Arial. Rendering uses the schematic CSV/JSON in `paper_2026/lifecycle_figures/figures/Figure_01/input`, the vector cylinder and cloud functions in `Figure_08/code/scene_20261026_preserved.py` under the same figures directory, and `inputs/wind_matched_dry_APE.csv` alongside this renderer. All are included in this tag.

The script writes a 180 × 112 mm PNG, an RGB/LZW TIFF and a vector PDF with embedded fonts. The source datasets, presentation mapping and delivered checksums are listed in `figure_sources.csv`, `display_map.csv` and `figure_files.csv`. The release ZIP and the separate vector PDF are verified by the downloadable SHA256SUMS.txt.

The earlier figure revision (figures-20261123-final-r01) changed four images: Figure 2g (bar spacing), Figure 5g (three pressure levels), Supplementary Figure 2 (aligned colour bars), and Supplementary Figure 7b (an independent vertical axis). The other twelve image pairs are unchanged. Panels a–f of Figures 2 and 5 reuse original-resolution crops of their r04 images; `inputs/upper_panels_provenance.json` identifies the originals and crop bounds. Both g panels are rendered from estimates, not cropped from a final image.

Figure 5g uses the original wind-matched ascent-minus-decay dry APE contrasts at 0–300 km: 500, 700 and 850 hPa for each of >16 d, 2–16 d and <2 d. Bars show means in J kg⁻¹ and whiskers the saved 95% cyclone-bootstrap intervals for 115 events in 108 cyclones. `inputs/wind_matched_dry_APE.csv` preserves the nine source records from `native/ENERGY_ADJ.csv` in the v2.1.5 statistical archive. These are the original matched estimates, not the common-support or adjusted estimates.

Render just the four revised images from the repository root:

```sh
python paper_2026/near_peak_event_figures/main_figures.py --repo . --figures 2 5 --out generated/final
python paper_2026/near_peak_event_figures/supplementary_figures.py --repo . --figures 2 7 --out generated/final
```

[Main figure checks](main_reproduction.json) · [Supplementary figure checks](supplementary_reproduction.json). The current release ZIP contains all 16 PNG/TIFF pairs, the vector Figure_06.pdf, the figure manifest and the internal checksum list.

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

[Display map](display_map.csv) · [Image checksums](figure_files.csv) · [Frozen statistical archive index](statistical_archive.json) · [Release](https://github.com/qq492947833/tc-near-peak-moisture/releases/tag/figures-20261125-fig6-r02)

[Supplementary figure sources and duration definitions](SUPPLEMENTARY_FIGURES.md) · [Supplementary figure crosswalk](supplementary_figure_mapping.csv) · [Complete plotting input index](figure_sources.csv)
