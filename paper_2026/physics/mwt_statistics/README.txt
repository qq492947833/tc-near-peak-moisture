Processed MWT diagnostics to statistics
======================================

After the input manifest has been assembled, run:
python -B reproduce.py --output runs/check

The entry uses the unchanged statistical functions and saved regional values.
It reproduces six declared departure outcomes in the 2-16-day window, their
input-window/radius and land-free comparisons, and the original motion/intensity
sensitivity. It also reproduces 48 fixed rise-analysis jobs: six native-support
outcomes, the same six on joint request support, and 36 reference/alternative/
direct-difference jobs. Every job retains all events, SID, time points, intervals,
and 2,000 bootstrap attempts. Missing support remains missing. The two event
populations and the two estimands are kept separate.

The reference tables are used only after calculation, for comparison. The check
covers identity, support, units, sample counts, estimates, intervals, and stored
bootstrap values. Numerical tolerances are fixed in config/replay.json; they are
not fitted to discrepancies. Output goes to a new directory.

This is a selected processed-input statistical replay, not a rerun of all
25 diagnostics at all pressure levels, ERA5 extraction, MWT, or raw tracks.
The copied production drivers document the original orchestration; use
reproduce.py as the portable entry. Rate integrals, observed state changes,
conversion, and transport remain distinct. One-SID bootstrap quantiles are
descriptive; successful-refit intervals retain their stated conditional scope.

Requires Python, NumPy, pandas, SciPy, h5py and pyarrow. No network is used.
Until manifest.json reports READY_PROCESSED_STATISTICAL_REPLAY, this directory
contains prepared code only and does not claim an executed reproduction.
