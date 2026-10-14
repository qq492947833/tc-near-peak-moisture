# Third-party notices and data attribution

## tcpyPI

The numerical modules in `src/raw_workflow/_vendor/tcpyPI` are retained from
tcpyPI v1.4.2, upstream commit
`872dcf42667818fe889bd3fb5a7952076be666e5`, with the original MIT license at
`src/raw_workflow/_vendor/tcpyPI/LICENSE`.

Copyright (c) 2020 Daniel Gilford. Upstream project:
[dgilford/tcpyPI](https://github.com/dgilford/tcpyPI). The copied `pi.py`,
`constants.py`, `utilities.py`, `numba.py` and license are unchanged. The local
package initializer only exposes the required numerical submodules. The
inclusive-endpoint and tracing adaptations are separate in `pi_trace.py` and
`potential_intensity.py`. Upstream attribution and the license are retained;
this notice does not relicense upstream code under any project-specific terms.

## Original scientific datasets

IBTrACS v04r01 is provided by NOAA NCEI. See its
[official product and citation page](https://www.ncei.noaa.gov/products/international-best-track-archive)
and dataset DOI [10.25921/82ty-9e16](https://doi.org/10.25921/82ty-9e16).
The raw archive is not redistributed here. Users should cite both the dataset
and the methodological paper as requested by the provider.

ERA5 is provided through the Copernicus Climate Data Store. The official
[pressure-level](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels)
and [single-level](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
catalogues identify the applicable CC-BY licence and attribution requirements.
Large raw ERA5 archives are not redistributed here. The DOI and variable
inventory are recorded in `RAW_INPUTS.md` and `config/raw_inputs.json`.

Natural Earth physical land data are public domain under the provider's
[terms of use](https://www.naturalearthdata.com/about/terms-of-use/).

Installed scientific Python dependencies retain their own licences. This file
records third-party attribution; it does not assign ownership or a new licence
to the authors' scientific code or to externally supplied datasets.
