# Tropical cyclone near-peak moisture exchange

Analysis notebooks and source data for stage-resolved tropical cyclone environments, temporary and final departures from near-peak intensity, and lifecycle timing. The figure mapping follows the npj Climate and Atmospheric Science manuscript: Figures 1–9 and Supplementary Figures 1–26.

## Run

Use Python 3.12 and a short local checkout path on Windows.

```sh
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -r requirements.txt
python fetch_data.py
python run_notebooks.py
```

The versioned `source_data.zip` asset contains the processed inputs. `fetch_data.py` verifies its checksum and every member against `data_manifest.json` before extraction; `--archive source_data.zip` uses a local download. Outputs go to `outputs/`. Each notebook starts a fresh kernel. Notebook 07 draws all 35 figures at 600 dpi from numerical inputs and the editable synthesis geometry. `figure_map.csv` links each panel to its function and inputs.

| Notebook | Calculation |
|---|---|
| 01 | Lifecycle identity, legal intervals and anchored/full-history near-peak counts |
| 02 | Stage environments, adjacent contrasts and saved adjusted intervals |
| 03 | Departure identities, valid paired windows and background balance |
| 04 | Boundary-node integration, moisture components and common windows |
| 05 | Recovery estimates and observation scenarios from saved predictions and resampling draws |
| 06 | Intensity-matched environmental paths and estimability/support summaries |
| 07 | Main and supplementary figures |

## Calculation scope

The quick workflow starts from processed inputs. It recalculates event counts, numerical summaries, boundary integrals and saved-draw intervals; model predictions and fitted coefficients are retained inputs where specified. It does not refit the historical recovery ensemble. Both the original and background-adjusted contrasts, all observation scenarios, unsuccessful-draw records and complete statistical families are included.

`RAW_INPUTS.md` describes the separate raw-data entry points, required variables and exact limitations. Full global ERA5 and IBTrACS archives are not distributed. Native-grid and lifecycle functions have bounded raw-input checks; a complete archive-to-manuscript run has not been repeated for this release. Historical fitted-model environments are recorded separately from the quick-workflow environment.

## Sources and use

Data sources, units, physical masks and pinned input identities are in `config/raw_inputs.json`. Third-party sources and licenses are listed in `THIRD_PARTY_NOTICES.md`. No repository-wide reuse license has yet been specified for the study's original code and derived data; third-party licenses remain in force. A fixed release can be cited by its repository URL, tag and commit. Author metadata and an archival DOI are not assigned here.
