# SCOPE Extension — Unsupervised Topic Discovery for Multi-Platform Chat

A six-stage unsupervised pipeline that takes a raw multi-user chat export and
produces topic segments, data-derived topic names, and per-user topic
timelines — with no seed topic list and no labeled training data.

This repository accompanies the manuscript *"Encoder Mixing for Unsupervised
Topic Discovery in Multi-Platform Chat: An Exploratory Study on Proxy
Corpora"* (submitted to *Machine Learning and Knowledge Extraction*).

## Pipeline

| Stage | What it does | Script |
|-------|--------------|--------|
| 1 | Cleaning: drops unnamed columns, writes missing `Text` as an empty string, keeps all rows | `stage1_clean_dataset.py` |
| 2 | Segmentation: cuts the message stream at accelerations of semantic dispersion, then places each cut by a local divergence criterion | `stage2_topic_shifting.py` |
| 3 | Topic discovery: HDBSCAN over segment embeddings, KeyBERT+MMR names, per-topic Kneedle threshold | `stage3_topic_naming.py` |
| 4 | Timelines: per-user contiguous runs on each topic, merged while the full-segment text clears the topic threshold | `stage4_topic_timeline.py` |
| 5 | Equivalence grid (evaluation): builds the reference-label × discovered-name grid that decides what counts as a hit | `stage5_topic_equivalence.py` |
| 6 | Window score (evaluation): scores each Stage 4 window against the messages in its sender-and-timestamp hull | `stage6_window_score.py` |

Stages 1–4 are operational and use no reference labels. Stages 5–6 are the
evaluation harness and are the only stages that see reference labels.

## Requirements

Required columns: `Text`. Stage 4 additionally requires `Sender`,
`Timestamp`, and `Chatroom`. Stages 5–6 require a reference label column.
Column names are configurable in `config.py`.

```
pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` prompts for the dataset path and the Stage 4 model. Stage 4
may use a different sentence encoder from Stages 1–3; when it does, it reuses
the Stage 3 Kneedle thresholds rather than re-estimating them.

## Worked example

`data/sample/` contains a 67-message, two-chatroom conversation with all
intermediate and final outputs precomputed. In the manuscript this file is
the **Sample walkthrough** — an illustrative example, not one of the two
evidential corpora.

## How the Stage 5 equivalence grid is marked

Stage 5 emits `topic_equivalence_grid.csv`, a reference-label × discovered-name
matrix in which every cell starts as "No". A cell becomes "Yes" when the
discovered name and the reference label are judged to denote the same topic.

The manuscript reports three distinct matchers over this grid:

- **hand** — semantic equivalence. On the Sample walkthrough this was marked
  by the authors. **On CrystalClear it was marked by a single large language
  model evaluator following a written equivalence rule**, with no second rater
  and no independent validation.
- **strict** — literal name synonymy after case and underscore normalization.
- **auto p90** — an automated proxy: a pair is equivalent when the cosine of
  the two name strings meets or exceeds the 90th percentile of pairwise
  message cosines in the same encoder.

A fourth matcher reported in the manuscript, many-to-one definition matching
(`def-many`), maps each discovered topic's definition text to its single
closest reference label.

## Evaluation corpora

The manuscript evaluates two corpora and three settings. **None of them is
seized evidence**; all are public or coreset chat used as a proxy for
multi-platform seized-device material.

| Setting | Messages | Rooms | Reference topics | Source |
|---------|---------:|------:|-----------------:|--------|
| Sample (walkthrough) | 66 | 2 | 4 | ChitChat slice |
| CrystalClear | 917 | 7 | 30 | Netherlands Forensic Institute, Apache-2.0 |
| Dataset 3 coreset | 2,999 | 280 | 23 | ChitChat coreset |

CrystalClear is third-party data released by the Netherlands Forensic
Institute at
<https://github.com/NetherlandsForensicInstitute/local-llm-chat-report-benchmark>
and introduced in Dehing et al. It is **not redistributed here**.

## Scope: what this repository does and does not contain

This repository contains the six-stage pipeline run with **one encoder
assignment at a time**. The manuscript additionally reports a 49-cell encoder
grid built by running this pipeline repeatedly under different encoder
assignments, plus several evaluation components that are not in this
repository:

- the 49-cell grid driver (same-model, mixed segmentation/naming, and
  second-stage `s4x` cells)
- the many-to-one definition matcher (`def-many`)
- BERTopic and FASTopic as Stage 3 substitutes
- the *P*<sub>k</sub> and WindowDiff segmentation proxy scorers
- the per-chatroom Stage 2 variant

<!--
  REPRODUCIBILITY ARTIFACTS — uncomment this section only once the files are
  actually committed. Do not promise files that are not present.

## Reproducibility artifacts

- `grids/` — the marked hand, strict, and auto p90 equivalence grids for
  Sample and CrystalClear
- `marking-instructions.md` — the verbatim written rule supplied to the large
  language model evaluator that marked the CrystalClear hand grid
- `data/crystalclear-subset.md` — the 917-message subset definition and its
  reference label column
- `data/dataset3-coreset/` — the 2,999-message coreset and its audited labels
-->

## Data handling

`data/raw/` is excluded by `.gitignore`. Do not commit case data of any kind
to this repository.

## License

MIT — see `LICENSE`.

## Citation

<!-- Replace with the final article citation and Zenodo DOI on acceptance. -->
