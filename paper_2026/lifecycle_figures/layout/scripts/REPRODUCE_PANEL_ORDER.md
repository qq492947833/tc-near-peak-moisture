# Reproduce the current panel order

Extract the v2.1.2 figure archive in `assets/`. From the repository root:

    python paper_2026/lifecycle_figures/layout/scripts/compose_panel_order.py --number 5 --source assets/source_assets/Figure_05.png --out runs/figures
    python paper_2026/lifecycle_figures/layout/scripts/compose_panel_order.py --number 6 --source assets/source_assets/Figure_06.png --out runs/figures
    python paper_2026/lifecycle_figures/layout/scripts/compose_S25_order.py --source assets/source_assets/Figure_S25.png --out runs/figures

The scientific marks, axis ranges, units and confidence intervals come from the accepted native input panels. Figure 5 uses the original panels a,b,c,f,d,e. Figure 6 uses a,c,d,e,b,f,g; the time-integrated KE transfer is panel e. Supplementary Figure 25 uses c,d,a,b; actual reintensification integrals precede the matched-wind specific-humidity scatterplots. The scripts verify input identities and return PNG and TIFF.
