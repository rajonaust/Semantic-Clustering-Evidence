# How the Stage 5 equivalence grids were marked

This file records how the hand grids reported in the manuscript were produced,
so the decisions behind every hand-grid score can be inspected.

## What a grid is

Each scored cell has its own grid, because each cell discovers its own topic
names. A grid is a matrix with reference labels as rows and that cell's
discovered topic names as columns. Every cell starts as `No`. A cell is set to
`Yes` when the discovered topic and the reference label denote the same
investigative topic. Stage 6 counts a message as matching when its reference
label and its window's topic are equal or marked `Yes`.

## Coverage

| Corpus | Scored cells | Grids present |
|--------|-------------:|---------------|
| CrystalClear | 48 (12 same-model, 12 mix, 24 `s4x`) | hand, strict, auto p90 |
| Dataset 3 coreset | 44 | auto p90 only |
| Sample (walkthrough) | 42 | hand, strict, auto p90 |

Dataset 3 has **no hand grid and no strict grid**. Auto p90 is the only Stage 6
matcher available for it, and every Dataset 3 ranking in the manuscript is
provisional. Its reference labels were produced separately, by LLM judges
auditing a keyword rule; that process is recorded in `JUDGE_RUBRIC.md`, not here.

The `s4x` cells reuse their base cell's topic set, so they reuse its marks.

## The three matchers

**hand** — semantic equivalence under a written rule. A discovered topic is
marked `Yes` for a reference label when its name and its cluster text are
semantically equivalent to that label. Equivalence is judged on meaning, not
string form.

One sub-rule has visible effect on the scores: a crypto-named discovered topic
is treated as equivalent to all four finance labels — `Bitcoin`,
`Cryptocurrency`, `Finance`, and `Finance_Cryptocurreny`.

**strict** — literal name synonymy after normalizing case and underscores. On
CrystalClear this is `No` in every cell of every scored grid: no discovered
topic name literally equals any of the 30 reference labels, so strict
overall-F1 is 0.000 throughout.

**auto p90** — a pair is marked `Yes` when the cosine between the two name
strings is at least the 90th percentile of pairwise message cosines in the
same encoder. In the source outputs this file is `topic_equivalence_grid_auto.csv`;
it is named `topic_equivalence_grid_auto_p90.csv` here to make the threshold
explicit. The p80 and p95 variants exist in the source outputs and are not
included.

## How the CrystalClear hand grid was marked

**No human checked individual marks.** The grid was produced in three steps by
two language models.

1. **Marking pass.** One Claude Sonnet agent marked all 112 discovered topics
   across 20 topic sets under the written rule. It had the list of reference
   label names but none of the labelled messages. It raised seven open
   questions.
2. **Rulings.** A second model — the session coordinating the work — answered
   those seven questions after reading sample messages carrying each reference
   label.
3. **Revision.** The marking agent applied those answers once.

Five reference labels changed as a result:

| Label | `Yes` marks before | After |
|---|---:|---:|
| Police | 0 | 13 |
| Personal | 0 | 35 |
| Place | 8 | 0 |
| Advertisement | 23 | 28 |
| Code | 16 | 21 |

The Sample hand grid was marked by the authors, using the synonymy rule shipped
with that file rather than the CrystalClear rule. The two rules are not the
same and the two grids are not pooled.

> **To complete:** paste the verbatim written rule, the seven open questions,
> and the per-topic-set rationale from the project's original marking log. This
> file reproduces the sequence and the outcome; the original log holds the exact
> wording supplied to the marking agent.

## Reference label strings

Reference labels are used exactly as supplied. `Finance`, `Cryptocurrency`, and
`Finance_Cryptocurreny` are three distinct classes and were not merged; merging
them would change the 30-way problem the grid was built to test.

## Limitations

- **No human verification.** The CrystalClear grid was marked, ruled on, and
  revised entirely by language models. No person checked individual marks.
- **No second rater, so no agreement statistic.** Cohen's κ is not reported on
  any Stage 5 grid.
- **The rulings pass saw the reference labels.** Step 2 resolved its questions
  by reading messages carrying each reference label, so the grid was adjusted
  with sight of the labels it is used to score against. This is reported rather
  than presented as independent validation.
- **Dataset 3 has no hand grid at all.**

Independent human validation of these grids is named as future work in the
manuscript.
