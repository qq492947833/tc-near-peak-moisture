# Figures from saved MWT integral estimates

`plot_source.csv` supplies the nine plotted rows on the common support of
122 SIDs and 132 return events. The values describe 2–16-day, pressure-weighted
500–850-hPa diagnostics within 0–300 km. They are time integrals in J kg⁻¹,
with the saved 95% SID-bootstrap percentile intervals. Plotting does not fit a
model, resample events or read ERA5.

From the repository root:

```sh
python paper_2026/physics/vertical_extension/plots/plot_mwt_integrals.py \
  --output runs/vertical_figures
```

This generates the standalone three-panel comparison, Supplementary Figure 29
and the Figure 6b tile, each directly at 900 dpi. Editable SVG and PDF exports
are also produced. The three-panel comparison presents KE canonical transfer,
dry-APE canonical transfer and selected KE terms. Supplementary Figure 29
contains the latter two panels; Figure 6b contains the KE-transfer panel.

Download the [v2.1.0 figure asset bundle](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.0/tc-near-peak-moisture-v2.1.0-mwt-figures.zip) and extract it into
`downloaded_figures/` (or another directory of your choice). The archive root
contains `figures/`, `source_assets/` and `MANIFEST.json`. To reconstruct the full Figure 6 while retaining the other panels:

```sh
python paper_2026/physics/vertical_extension/plots/compose_figure6.py \
  --base downloaded_figures/source_assets/Figure_06_before_vertical.png \
  --tile runs/vertical_figures/Fig6b_new_tile.png \
  --output runs/vertical_figures/formal_stage
```

The base is a 6,377 × 7,157-pixel image. Composition replaces only the 3,157 ×
1,780-pixel panel at `(3220, 0, 6377, 1780)` and verifies all pixels outside that
rectangle are unchanged. No interpolation or redrawing of the retained panels
is performed. The high-resolution base and final PNG/TIFF files are release
assets, rather than repository-tracked binary files.

To compare a local replay with the downloaded reference rasters:

```sh
python paper_2026/physics/vertical_extension/plots/verify_figures.py \
  --generated runs/vertical_figures \
  --reference downloaded_figures \
  --report runs/vertical_figures/figure_comparison.json
```

Dependencies are NumPy, pandas, Matplotlib and Pillow. DejaVu Sans is the
specified plotting font. Plotting geometry, colours, marker sizes, limits and
tick locations are encoded in `plot_mwt_integrals.py`. Asset hashes and the
public source-to-figure relationships are recorded in `SOURCE_MANIFEST.json`.
