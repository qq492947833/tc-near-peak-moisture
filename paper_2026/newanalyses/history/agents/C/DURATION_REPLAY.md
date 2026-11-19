# Stage-duration trend targets

Run from the repository root with the current Python environment:

```sh
python paper_2026/newanalyses/history/agents/C/replay_duration.py --output results/duration --check-recovery
```

Omit `--check-recovery` to summarize the saved target values directly. The standalone `analyze_duration.py` command accepts the same arguments. The original preparation, bootstrap and full-rank fit functions remain available for historical analyses. Run the command in a working directory where you can write the chosen output. The replay does not require the ERA5 files or an input download.

The full period is 1982–2023 and the recent check is 2000–2023. Each period has its own frozen three-year circular-block/within-year SID draw bank. Row index is the original zero-based draw ID. Each stage uses its own common-support SID list. The bank counts, SID order and year multiplicities are paired with that stage’s design, reference distribution and outcomes (elapsed and covered hours). The design sets the year coordinate to decades. The release uses `Dec` for the stage stored as `Res` in the historical inputs. Inputs in `duration_targets/` include original and packaged SHA256 identities in `source_manifest.json`; string SID arrays are stored without requiring pickle.

For an original rank-guard failure, `duration_linear_targets.py` forms the original weighted annual contrasts, tests each for estimability on the scaled weighted design and evaluates only estimable contrasts. The relative singular-value threshold is 1e-10; the relative contrast residual tolerance is 1e-9. Stored successful trend values are retained unchanged. The optional check repeats the accepted target evaluation on the original failed draws, including the auxiliary targets. There are no replacements or new random draws. The point estimates use the frozen point contrasts or original auxiliary point estimates.

`replayed_statistics.csv` records the four trend targets for each stage and period (32 rows); full-period environment histories and duration associations (24); successive differences and basin/season minus fully fixed (16); and descriptive environment products (12). Intervals use the 2.5th and 97.5th percentiles. P values use the original two-sided sign-tail counts with one added to each tail numerator and denominator. BH uses 16 tests for each period's primary family and 12 each for environment trends, duration coefficients and adjacent differences. Basic-minus-fully-fixed has no P/q; descriptive products have no BH family. Each difference is paired by draw ID. Counts refer to estimable targets, not the number of full-rank fits. Gamma/log-link and separately defined sensitivity analyses retain their original support and inference.

The compatibility tables in this directory and the stage-duration plotting inputs retain their existing column names and stage IDs. `replay_checks.json` identifies which rank-guard failures were checked per period and stage. No plotting or model selection is part of this entry point.
