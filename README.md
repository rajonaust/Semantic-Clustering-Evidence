# Encoder Mixing for Unsupervised Topic Discovery in Multi-Platform Chat: An Exploratory Study on Proxy Corpora

A six-stage unsupervised pipeline that takes a raw multi-user chat export and
produces topic segments, data-derived topic names, and per-user topic
timelines — with no seed topic list and no labeled training data.

This repository accompanies the manuscript *"Encoder Mixing for Unsupervised
Topic Discovery in Multi-Platform Chat: An Exploratory Study on Proxy
Corpora"* (*Machine Learning and Knowledge Extraction*).

## Install

Requires Python 3.10 or newer.

```bash
pip install -e .
```

That installs the `semantic-clustering-evidence` package and the
`scope-pipeline` command. Alternatively, `pip install -r requirements.txt`
installs the dependencies without the package.

## Run

```bash
scope-pipeline
```

The command prompts for the dataset path and the Stage 4 model. Stage 4 may
use a different sentence encoder from Stages 1–3; when it does, it reuses the
Stage 3 Kneedle thresholds rather than re-estimating them.

Required column: `Text`. Stage 4 additionally requires `Sender`, `Timestamp`,
and `Chatroom`. Stages 5–6 require a reference label column. Column names are
configurable in the package's `config` module.

## Pipeline

| Stage | What it does |
|-------|--------------|
| 1 | Cleaning: drops unnamed columns, writes missing `Text` as an empty string, keeps all rows |
| 2 | Segmentation: cuts the message stream at accelerations of semantic dispersion, then places each cut by a local divergence criterion |
| 3 | Topic discovery: HDBSCAN over segment embeddings, KeyBERT+MMR names, per-topic Kneedle threshold |
| 4 | Timelines: per-user contiguous runs on each topic, merged while the full-segment text clears the topic threshold |
| 5 | Equivalence grid (evaluation): builds the reference-label × discovered-name grid that decides what counts as a hit |
| 6 | Window score (evaluation): scores each Stage 4 window against the messages in its sender-and-timestamp hull |

Stages 1–4 are operational and use no reference labels. Stages 5–6 are the
evaluation harness and are the only stages that see reference labels.

## Repository layout

```
src/semantic_clustering_evidence/   pipeline modules (stages 1-6, config, runner)
grids/                              marked Stage 5 equivalence grids
data/sample/                        worked example with all outputs precomputed
tests/                              test suite
HANDGRID_MARKING_NOTES.md           how the hand grids were marked
```

Run the tests with `pytest`.

## Worked example

`data/sample/` contains a 66-message, two-chatroom conversation with all
intermediate and final outputs precomputed. In the manuscript this file is the
**Sample walkthrough** — an illustrative example, not one of the two evidential
corpora.

## The Stage 5 equivalence grids

Stage 5 emits a reference-label × discovered-name matrix in which every cell
starts as "No". A cell becomes "Yes" when the discovered topic and the
reference label denote the same investigative topic. Stage 6 counts a window
as a hit only through a "Yes" cell, so these grids determine every hand-grid
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
by no human.** See `HANDGRID_MARKING_NOTES.md` for the full sequence, the five
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
and introduced in Dehing et al. It is not redistributed here.

## Scope: what this repository does and does not contain

This repository contains the six-stage pipeline run with **one encoder
assignment at a time**, together with the marked equivalence grids and the
marking record. The manuscript additionally reports a 49-cell encoder grid
built by running this pipeline repeatedly under different encoder
assignments, plus several evaluation components that are **not** here:

- the 49-cell grid driver (same-model, mixed segmentation/naming, and
  second-stage `s4x` cells)
- the many-to-one definition matcher (`def-many`)
- BERTopic and FASTopic as Stage 3 substitutes
- the *P*<sub>k</sub> and WindowDiff segmentation proxy scorers
- the per-chatroom Stage 2 variant

## Data handling

Do not commit case data of any kind to this repository.

## License

MIT — see `LICENSE`.

## Citation

Archived on Zenodo. Cite the version DOI matching the release used, not the
GitHub URL alone.

<!-- Add the article citation and the final release DOI on publication. -->
