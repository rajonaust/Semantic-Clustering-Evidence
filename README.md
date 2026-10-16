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

All six stage modules, `run_pipeline.py`, and `config.py` live under
`src/semantic_clustering_evidence/` (standard "src layout" — keeps the
installable package separate from the README/data/tests clutter at the repo
root, and avoids accidental same-directory import shadowing).

## Installation

```
pip install -e ".[dev]"   # editable install + pytest/pytest-cov for the test suite
```

(`pip install -e .` alone, without `[dev]`, is enough just to run the
pipeline.) This reads dependencies and the console-script entry point from
`pyproject.toml` rather than a loose `requirements.txt`.

## Running the pipeline

```
scope-pipeline
```

(installed as a console script by `pyproject.toml`; equivalently
`python -m semantic_clustering_evidence.run_pipeline`.)

Required column: `Text`. Stage 4 additionally requires `Sender`,
`Timestamp`, and `Chatroom`. Stages 5–6 require a reference label column.
Column names are configurable in `config.py`.

`run_pipeline.py` prompts for the dataset path and the Stage 4 model. Stage 4
may use a different sentence encoder from Stages 1–3; when it does, it reuses
the Stage 3 Kneedle thresholds rather than re-estimating them.

## Tests and coverage

```
pytest --cov=semantic_clustering_evidence --cov-report=term-missing
```

The suite (`tests/`) unit-tests each stage's logic — cleaning, the Kneedle
threshold edge cases, HDBSCAN cluster separation, per-user run merging, the
equivalence grid builder, and window scoring (the last is also a regression
test locked to the exact numbers in `data/sample/`) — without needing a real
sentence-transformer model downloaded: a deterministic `FakeEncoder` fixture
(`tests/conftest.py`) is injected wherever stage code accepts a `model=`
parameter, and `sentence_transformers`/`keybert` are stubbed in `sys.modules`
so importing the stage modules doesn't require installing those (optional,
multi-GB) dependencies just to run unit tests. Current coverage is ~68%
line/branch combined; the main gaps are `run_pipeline.py`'s interactive
`input()` prompts and a few verbose-print branches, both low-value to test.

## Worked example

`data/sample/` contains a 67-message, two-chatroom conversation with all
intermediate and final outputs precomputed. In the manuscript this file is
the **Sample walkthrough** — an illustrative example, not one of the two
evidential corpora.

## The Stage 5 equivalence grids

Stage 5 emits a reference-label × discovered-name matrix in which every cell
starts as "No". A cell becomes "Yes" when the discovered topic and the
reference label denote the same investigative topic. Stage 6 counts a window
as a hit only through a "Yes" cell, so this grid determines every hand-grid
score reported in the paper.

Three matchers are reported:

- **hand** — semantic equivalence under a written rule.
- **strict** — literal name synonymy after case and underscore normalization.
  On CrystalClear this is "No" in every cell of every scored grid, so strict
  overall-F1 is 0.000 throughout.
- **auto p90** — a pair is equivalent when the cosine of the two name strings
  meets or exceeds the 90th percentile of pairwise message cosines in the same
  encoder.

A fourth matcher reported in the manuscript, many-to-one definition matching
(`def-many`), maps each discovered topic's definition text to its single
closest reference label.

**The CrystalClear hand grid was produced by two language models and reviewed
by no human.** `HANDGRID_MARKING_NOTES.md` records the full sequence, the five
labels whose marks changed, and the limitations that follow.

## Evaluation corpora

The manuscript evaluates two corpora and three settings. **None of them is
seized evidence**; all are public or coreset chat used as a proxy for
multi-platform seized-device material.

| Setting | Messages | Rooms | Reference topics | Hand grid |
|---------|---------:|------:|-----------------:|-----------|
| Sample (walkthrough) | 66 | 2 | 4 | yes |
| CrystalClear | 917 | 7 | 30 | yes |
| Dataset 3 coreset | 2,999 | 280 | 23 | no (auto p90 only) |

CrystalClear is third-party data released by the Netherlands Forensic
Institute under the Apache-2.0 licence at
<https://github.com/NetherlandsForensicInstitute/local-llm-chat-report-benchmark>
and introduced in Dehing et al.

## Scope: what this repository does and does not contain

This repository contains the six-stage pipeline run with **one encoder
assignment at a time**. The manuscript additionally reports a 49-cell encoder
grid built by running this pipeline repeatedly under different encoder
assignments, plus several evaluation components that are **not** in this
repository:

- the 49-cell grid driver (same-model, mixed segmentation/naming, and
  second-stage `s4x` cells)
- the many-to-one definition matcher (`def-many`)
- BERTopic and FASTopic as Stage 3 substitutes
- the *P*<sub>k</sub> and WindowDiff segmentation proxy scorers
- the per-chatroom Stage 2 variant

## Reproducibility artifacts

- `grids/` — the marked hand, strict and auto p90 equivalence grids per scored
  cell: CrystalClear (48 cells, all three matchers), Dataset 3 (44 cells, auto
  p90 only), Sample (42 cells, all three matchers)
- `HANDGRID_MARKING_NOTES.md` — how the hand grids were marked, what changed
  during marking, and the limitations

## Data handling

`data/raw/` is excluded by `.gitignore`. Do not commit case data of any kind
to this repository.

## License

MIT — see `LICENSE`.

## Citation

Archived on Zenodo. Cite the version DOI matching the release used, not the
GitHub URL alone.

<!-- Replace with the final release DOI and the article citation on acceptance. -->
