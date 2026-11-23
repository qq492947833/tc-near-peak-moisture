# Figures 2 and 5: final lower panels

From the repository root, run:

```bash
python paper_2026/near_peak_event_figures/main_figures.py --repo . --out out --figures 2 5 --qa out/qa
```

This entry point renders panel g and joins it to the unchanged native pixels of panels a–f without interpolation. `inputs/Figure_02_af.png` and `inputs/Figure_05_af.png` contain only panels a–f. `inputs/upper_panels_provenance.json` specifies the original PNG hashes from `figures-20261121-r04` and the exact pixel crop (left/top inclusive, right/bottom exclusive). The r04 renderers remain available as `draw2_r04` and `draw5_r04`; the current entry point calls the new lower-panel renderers. No final g panel is used as input.

Figure 2g reads `environmental_contrasts.csv` in `paper_2026/lifecycle_figures/figures/Figure_03/inputs/direct/` (117 original pairs; 105 pairs at the other three adjustment levels). Figure 5g reads `inputs/wind_matched_dry_APE.csv`, an unrounded nine-row extract from `Supplementary_Data_1_CSV.zip`, `native/ENERGY_ADJ.csv` (native CSV SHA-256 `1bd3a254540841365db646df0c9d46e237bac881af068715415c70e03c6e89cd`; rows `ENERGY_ADJ:00101` through `ENERGY_ADJ:00109`). Windows 0, 1 and 2 mean >16, 2–16 and <2 days. The contrast is reintensification minus decay at matched winds, 0–300 km, in J kg⁻¹. Matched 10-kt wind bins are averaged within events, events within cyclones, then cyclones equally. Whiskers use the frozen 95% cyclone-bootstrap bounds for 115 events in 108 cyclones. The 700-hPa slow-window original matched estimate is 3.898006021438687. Neither entry point resamples or fits data.

Bitwise reproduction was verified at 180 mm and 900 dpi with Arial on Windows, Python 3.11.7, Matplotlib 3.8.0, NumPy 1.26.4, pandas 2.1.4 and Pillow 10.2.0. Output is RGB PNG and pixel-identical LZW TIFF. The QA option writes the unrounded plotted estimates and panel geometry. The scientific analysis baseline is v2.1.5; this drawing revision is separate.
