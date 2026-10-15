# Stage 5 equivalence-grid marking

This file records how the Stage 5 equivalence grids were marked, so the
decisions behind the hand-grid scores reported in the manuscript can be
inspected.

## What the grid is

Stage 5 emits a matrix with reference labels as rows and discovered topic
names as columns. Every cell starts as `No`. A cell is set to `Yes` when the
discovered topic and the reference label denote the same investigative topic.
Stage 6 counts a window as a hit only through a `Yes` cell, so this grid
determines every hand-grid score in the paper.

## The equivalence rule

A discovered topic is marked `Yes` for a reference label when its name and
its cluster text are semantically equivalent to that label. Equivalence is
judged on meaning, not on string form: a discovered name need not match the
label literally, and a literal match is not sufficient if the cluster text is
about something else.

Topics whose cluster text carries no substantive subject matter are not
marked `Yes` for any label.

Reference labels are used exactly as supplied. In particular `Finance`,
`Cryptocurrency`, and `Finance_Cryptocurreny` (sic) are three distinct
classes and were not merged, since merging them would change the 30-way
problem the grid was built to test.

## Who marked each grid

| Grid | Marked by | Reviewed by |
|------|-----------|-------------|
| Sample, hand | the authors, using the synonymy rule shipped with that file | — |
| CrystalClear, hand | a single large language model evaluator, following the rule above | the authors |
| strict | deterministic string rule after case and underscore normalization | — |
| auto p90 | deterministic cosine threshold at the 90th percentile of pairwise message cosines | — |

The Sample and CrystalClear rules are not the same rule, and the two grids are
not pooled.

Dataset 3 has no hand grid. `auto p90` is the only matcher reported for it,
and every Dataset 3 ranking in the manuscript is provisional.

## Author review of the CrystalClear grid

After the machine pass, the authors reviewed the marking and changed it in two
places:

1. **A contentless-topic clause was added to the rule while the pass was under
   way**, so that topics with no substantive subject matter are not marked
   `Yes` for any label.
2. **Four reference label categories were re-marked** — Police, Personal,
   Code, and Place — after a sample of ground-truth rows showed the initial
   marking was wrong for them.

Both changes are reported in Section 5.1 of the manuscript.

## Limitations

This is a single-rater procedure. The CrystalClear review was carried out by
the authors, who also wrote the rule, so it is not independent dual annotation
and no inter-rater agreement statistic is available. The CrystalClear hand
grid is machine-marked at base and has not been independently validated.

Independent human validation of these grids is named as future work in the
manuscript.
