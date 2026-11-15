# v2.1.0 — MWT vertical transfer and layer energetics

This release adds the pressure-direction contributions to KE and dry-APE
canonical transfer, their sums with the horizontal contributions, and consistent
500–850 hPa storage, conversion and time integrals. It retains the 1982–2023,
2,340-SID framework and the original event definitions, reference state and
nominal >16-day, 2–16-day and <2-day windows.

The scientific package contains the actual extraction, diagnostic, statistical
and sensitivity implementations with portable input/output configuration.
Full-precision summaries retain all three established comparisons, scale
windows, regions and diagnostic supports. Existing numerical functions remain
independently callable.

Figure 6b and Supplementary Figure 29 use the same 122 SIDs and 132 actual
reintensification events, 0–300 km region and 500–850 hPa layer. Horizontal,
vertical and total canonical terms use coupled saved SID draws. Layer
conversion and boundary input use the same common support. The figure archive
contains the delivered 900 dpi PNG/TIFF files, editable vector exports for the
new panels and the unchanged base image needed to compose Figure 6.

The KE total integral is negative and the dry-APE total positive in the
prespecified discrete comparisons. Dry-APE magnitude remains sensitive to
pressure derivatives and discrete form. Strict machine-precision conservation
is not claimed, and no donor-window interaction decomposition or closed energy
budget is supplied.

The extension archive contains the versioned package at its repository-relative
path and is used with the v2.1.0 checkout for base production dependencies. The existing v2.0.0 processed-input
assets remain available for the base workflows. Validation uses source and
dependency checks, saved-value comparisons and bounded figure reproduction;
it does not rerun the atmospheric calculations, fit models or generate new
bootstrap samples. See the package README and VALIDATION.md for exact commands
and scope. Earlier tags and their assets remain unchanged.
