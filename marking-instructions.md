# Stage 5 equivalence-grid marking instructions

This file records the written rule used to mark the Stage 5 equivalence grids
reported in the manuscript, so that the marking decisions behind every
reported hand-grid F1 can be audited.

> **TEMPLATE — FILL BEFORE COMMITTING.**
> Every `[FILL]` below must be replaced with the actual rule text you used.
> Reproduce the rule verbatim as it was supplied to the evaluator; do not
> paraphrase or tidy it. If the rule was given to the model as a prompt,
> paste the prompt exactly, including any system message.

## Who marked what

| Grid | File | Marked by | Rater count |
|------|------|-----------|-------------|
| Sample, hand | `grids/sample_hand.csv` | the authors | 1 |
| CrystalClear, hand | `grids/crystalclear_hand.csv` | a single large language model evaluator | 1 |
| Sample / CrystalClear, strict | `grids/*_strict.csv` | deterministic string rule (no rater) | — |
| Sample / CrystalClear, auto p90 | `grids/*_autop90.csv` | deterministic cosine threshold (no rater) | — |

No grid was marked by a second rater, so no inter-rater agreement statistic
is reported in the manuscript.

**Model used for the CrystalClear hand grid:** `[FILL: model name and version]`
**Date marked:** `[FILL]`
**Decoding settings:** `[FILL: temperature, max tokens, or "defaults"]`

## The written equivalence rule, as supplied

```
[FILL: paste the rule verbatim]
```

## Amendments made during marking

The manuscript (Section 5.1) records that the rule was amended once during
the pass and that four label categories were re-marked afterwards. Both are
recorded here so the final grid is not mistaken for the product of a single
unchanged rule.

### Amendment 1 — contentless-topic clause

- **When:** `[FILL: at what point in the pass]`
- **Why:** `[FILL]`
- **Clause added, verbatim:**

```
[FILL]
```

- **Were cells marked before this amendment re-marked under the amended
  rule?** `[FILL: yes / no — and if no, how many cells were marked under the
  original rule]`

### Amendment 2 — revision of four label categories

After a later sample of ground-truth rows showed the initial marking was
wrong for four categories, the marking for these was revised:

- Police
- Personal
- Code
- Place

- **What was wrong initially:** `[FILL]`
- **Revised treatment:** `[FILL]`
- **Number of cells changed:** `[FILL]`

## Grid file format

Each grid is a CSV with reference labels as rows and discovered topic names
as columns. Cells contain `Yes` or `No`. A `Yes` means the discovered name
and the reference label were judged to denote the same investigative topic;
Stage 6 counts a window as a hit only through a `Yes` cell.

## Known limitations of this marking

- Single rater on every hand grid; no inter-rater agreement is available.
- The CrystalClear hand grid is machine-marked and has not been
  independently validated by a human.
- Dataset 3 has no hand grid at all; `auto p90` is the only matcher reported
  for it, and its rankings are provisional throughout the manuscript.
- `Finance`, `Cryptocurrency`, and `Finance_Cryptocurreny` (sic) are treated
  as three distinct reference classes and were not merged.
