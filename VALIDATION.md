# Validation scope for v2.0.0

The validation uses the released code, frozen inputs and preserved scientific
settings. It distinguishes checks of implementation and saved outputs from a
new execution of the full research campaign.

| Component | Checks actually performed |
|---|---|
| Current lifecycle | Frozen 2,340-SID identity and 63,769 records; three real small trajectories with 45 exact time/value comparisons; corresponding record masks; original positional and keyword interface equivalence |
| Framework and sensitivity | Presence and syntax of the actual calculation drivers and extracted functions; all 18 frozen threshold input sets; saved seven-test primary-trend BH family |
| Figures | All 50 current figure source mappings; 45 standalone plot entries checked with `--help`, syntax and input hashes; five new-analysis sources and six final lossless layout sources checked. No accepted figure was rerendered |
| Direct MWT | 55 analytical implementation checks and 25 local diagnostics recomputed from a real saved coefficient stencil; five independent numerical modules remain independently callable |
| Moisture and recovery | 238 saved-result checks, including 192 RH summary rows, additive/support identities, transport means, temporary-minus-decay signs, 2,231-SID recovery support, five folds, original 1,000-draw intervals and 16 corrected hemisphere cells |
| Basin, historical, duration and latitude | 55 deterministic checks: 2,340/123/133 identities; valid six-hour opportunity counts; global trend and HAC interval; complete 11-test and 16-test BH families; all-SID F=R+L+B; duration and paired-latitude intervals; 1,245 training and 322 reference records |
| Saved summary execution | Independent working copies executed latitude summaries and C draw reaggregation. Ten resulting tables matched their frozen numerical columns within absolute tolerance 1e-11 |
| Native raw-data configuration | One real ERA5 file checked for variable, dimensions, units, coordinate hashes, size and three stored scalar probes; no field extraction or MWT campaign |
| Source dependencies | Actual local imports and dynamic source loads resolved; selected extracted scientific functions/classes compared with their production ASTs; source and adaptation provenance retained |

The RH mean comparison's maximum absolute difference was approximately
1.78e-15. Confidence intervals are recomputed from the same saved draws, not
newly generated draws. The 16 standardized-duration tests, five F-component
tests and other declared families remain separate. A change in display names
does not change historical array/field identities.

The final source tree was copied to an independent working directory without
the asset members. The public asset fetcher restored all 270 members from the
two release archives and verified their checksums. The public verification
entry then passed 284 checks, including seven separate process routes. A
separate raw-IBTrACS run for SID `1982005S10105` produced 53 records and matched
all 23 frozen comparison fields. These are bounded validation runs, not a new
classification of the full cohort. The figure-source validator passed 938
checks after source-path metadata were made portable; scientific table cells
were unchanged by that metadata edit.

Run `python verify_release.py --output validation/summary.json` after restoring
the input assets. This checks source syntax, all asset-member hashes and bounded
scientific routes in separate processes. It records the exact runtime and
subprocess results. Use a new output path/work directory for each run. The
asset fetcher also verifies archive/member checksums and rejects conflicting
local inputs.

The release validation did **not** refit models, generate new bootstrap samples,
repeat global ERA5/MWT processing, or rerender the manuscript's accepted figures.
The original fitting and raw-field programs remain available as explicit steps.
Exact raw-data replay requires the specified archived file schema and identity;
an equivalent newly encoded NetCDF file can require a newly audited inventory.
No claim is made that arbitrary contemporary CDS downloads reproduce the full
study with one unconfigured command. Font/platform changes can affect figure
text metrics even when the supplied numerical inputs are identical.
