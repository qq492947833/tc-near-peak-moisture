# Figure rendering

`python -m src.figures.render` writes the nine main and twenty-six supplementary figures to `outputs/figures` as editable SVG/PDF plus 601 dpi PNG/TIFF. Notebook 07 invokes the same entry point.

The inputs are frozen scientific estimates, interval tables, empirical SID distributions, precomputed spatial arrays and explicit schematic geometry. The figure layer redraws numerical artists from these inputs; it never copies existing figure images. It does not rerun ERA5 extraction, fit the underlying scientific models, perform bootstrap draws or rerun spatial permutation tests. The other analytical notebooks expose the corresponding computational layers where packaged inputs are available.

`figure_map.csv` identifies each panel, plotting module and source table. Selectors and display transformations remain in the plotting functions. Metadata, support, failed-draw counts and complete comparison groups in the scientific input tables retain their meaning. Pointwise intervals, conditional adjusted-draw percentiles and primary analytic intervals are distinct estimands.

Fonts use the DejaVu Sans family supplied by the installed Matplotlib distribution. No system font is redistributed. Natural Earth coastline geometry is public domain. Source data and plotting geometry determine scientific content; PDF creation metadata and font-renderer differences need not yield identical file hashes.

The environment variables `FLTCCF_FIGURE_DPI` and `FLTCCF_FIGURE_FORMATS` can request a lightweight preview; the default uses all four formats at publication resolution. `FLTCCF_REPO_ROOT` is an optional explicit repository path. Figures use no generative image tools or remote data downloads.
