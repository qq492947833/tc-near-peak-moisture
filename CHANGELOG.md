# Changelog

## v2.1.3 — Trough-aligned KE canonical-transfer rates

Figure 6e shows the horizontal and pressure-direction contributions to kinetic-energy canonical transfer and their sum. The rates (W kg⁻¹) are averaged over 0–300 km and 500–850 hPa in the nominal 2–16-day window. Every six-hour point from −24 to +24 h uses the same 126 events in 117 cyclones. Events are averaged within cyclone, with equal weight per cyclone and pointwise percentile 95% intervals from the original 2,000 cyclone-level draws. Some positive-lag observations occur after the cyclone has returned to near-peak intensity.

Supplementary Figure 26a retains the event-specific trough-to-return KE transfer integrals in J kg⁻¹ for 132 events in 122 cyclones. The other six Figure 6 panels retain their original pixels.

The source package contains the targeted coefficient-recovery and saved-draw calculation entries, the fixed support and full-precision plotting data. The figure archive contains the full PNG and lossless TIFF, the native rate panel and the v2.1.2 source canvas for assembly. See the [rate and recovery instructions](https://github.com/qq492947833/tc-near-peak-moisture/blob/v2.1.3/paper_2026/physics/vertical_extension/fig6e_rates/README.md).

[Figure archive](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.3/tc-near-peak-moisture-v2.1.3-figures.zip) · [SHA-256 checksum](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.3/SHA256SUMS.txt).

## v2.1.2 — Mechanism artwork and saved MWT integral panels

* Distinguish the dry APE and horizontal KE nodes and conversion arrows by colour in the mechanism figure. The vector scene and eleven frozen CSV inputs reproduce the revised artwork.
* Provide the three saved MWT integral panels in current supplementary order: KE canonical transfer, dry APE canonical transfer, and selected KE terms. The nine estimates, confidence intervals, units and support (122 SIDs, 132 events) are unchanged. The supplementary panels retain their native pixels.
* Reorder the native panels of Figures 5, 6 and S25 to follow the narrative. Figure 6 retains the valley-to-return KE transfer integral; it does not contain a new transfer-rate trajectory.
* The figure archive contains Figures 5, 6, 8, S25 and S26 (PNG and TIFF), editable SVG for Figure 8, and the native source panels needed for reordering. The duration correction, transport compatibility fix and saved-example validation from v2.1.1 are unchanged.

[Figure archive](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.2/tc-near-peak-moisture-v2.1.2-figures.zip) · [SHA256 checksums](https://github.com/qq492947833/tc-near-peak-moisture/releases/download/v2.1.2/SHA256SUMS.txt).

## v2.1.1 — Duration inference and saved-example checks

* Publish the previously accepted duration estimability correction, original draw identities, saved target values and deterministic replay. The four full-period E-stage standardized trends have 2,000 estimable targets rather than 1,915 full-rank fits. Environmental trends, duration associations and paired standardization differences retain their own supports and multiple-testing families.
* Use a compatible trapezoidal integration interface for the affected moisture-transport routines.
* Validate expected rate metrics, event keys, matching coverage and finite values before comparing the 12 saved MWT integrals.
* Distribute the accepted duration figure without redrawing it. The v2.1.0 MWT figure bundle and the v2.0.0 processed-input bundles remain available at their original releases.

This update reuses existing draws and accepted results. No new atmospheric-field extraction, model fit or bootstrap was run for publication.

## v2.1.0 — 2026-11-16

* Add the implemented pressure-direction KE and dry-APE canonical-transfer terms, horizontal-plus-vertical totals, and consistent 500–850hPa layer diagnostics.
* Provide the three established comparison routes, support-aware full-precision summaries, joint-support figure data and reproducible high-resolution figure composition.
* Integrate the KE decomposition as Figure 6b and the dry-APE/selected-KE panels as Supplementary Figure 29; map earlier S29–S42 to S30–S43 without redrawing retained panels.
* Document direct-omega input requirements, preserved statistical families and the finite-discretization qualifications.
* Preserve callable scientific interfaces, the frozen cohort, prior input archives, historical tags, authorship and third-party notices.

## v2.0.0 — 2026-11-13

* Bind the current analysis entry points to the frozen 1982–2023, 2,340-SID cohort.
* Replace the superseded E-stage rate-gated raw lifecycle implementation with the actual progress-threshold production source; retain the independent function interface and historical machine-field names.
* Add current framework/threshold comparisons, paired environmental and moisture diagnostics, direct MWT and reintensification code, recovery models and saved-result checks.
* Add basin and historical analyses, supported hemisphere/latitude recovery comparisons and stage-duration standardization with their actual configurations and statistical families.
* Supply the current 8-main/42-supplementary figure source map, schematic sources and final lossless layout composition steps.
* Publish versioned processed-input and figure-source assets with per-member checksums and an offline-capable verifier.
* Preserve existing tags, Git history, independent computational interfaces and third-party licences. Retain the earlier notebooks as an explicitly labelled historical route.

## v1.0.1 — 2026-10-14

Earlier notebook release with unit-aware transport numerical validation.

## v1.0.0 — 2026-10-14

Initial lifecycle and moisture-analysis notebooks with processed source-data asset.
