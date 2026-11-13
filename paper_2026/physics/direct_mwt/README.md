# Direct MWT diagnostics

Run `python verify.py --output runs/check_01` with Python 3.11 and the NumPy version in `requirements.txt`. The output directory must not exist. The five files in `src/` can each be imported independently; their computations do not read project paths or import another custom module.

The command executes small analytical-input checks and recomputes 25 local diagnostics from one saved ERA5-derived coefficient stencil. It records numerical differences and finite-value masks. The real example contains three consecutive coefficient times and a 3 × 3 grid stencil, retaining all 13 wind/temperature levels and the three observed omega levels. Only the central time and grid cell is compared with the original saved diagnostics.

The analytical fixture tests implementation, not atmospheric inference. The saved-coefficient replay tests local diagnostic reproducibility, not physical budget closure or a repeat of the 117-SID study. No group effect, uncertainty interval or new statistical estimate is generated here. Frozen event and centre tables retain unavailable locations and preserve their UTC identities.

For new calculations, use `MWT_Prepare_Direct` on a sufficiently long fixed geographic grid, transform the returned original fields and products with `MWT`, then pass physical-normalization coefficients to `MWT_Atmosphere_Direct`. `MWT_Apply_Operator` applies the exact operator built by the supplied transform; `MWT_Apply_Increment` copies existing coefficients and evaluates only missing rows or grid columns. These are computational interfaces, not a bundled raw-data extraction service.

`config/direct_diagnostics.json` records the actual 1,024-step study settings. Complete reproduction from raw inputs additionally requires continuous 6-hourly u, v, T and geopotential at 13 pressure levels, omega at 500/700/850 hPa, surface pressure, and the frozen event/centre definitions. The raw ERA5 files, full operator/field caches and population statistical replay are not included in this small package. Supplying omega at additional levels is not part of the calculation; continuity-equation reconstruction remains disabled.

KE/APE are energy stores. APE-to-KE conversion, horizontal transport, horizontal canonical-transfer contributions and layer-boundary input are distinct diagnostics. Boundary input is `[Q(500 hPa) − Q(850 hPa)]/g` or the same difference divided by 35,000 Pa. Pressure work must not be added to a separate conversion term as if both were independent contributions to the same budget. See `config/diagnostic_semantics.json` for signs and units.

Source attribution and scientific inputs are recorded in `config/source_provenance.json`. No new license or redistribution permission is assigned. This local package performs no download or upload.
