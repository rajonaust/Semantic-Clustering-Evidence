# SCOPE Extension — Unsupervised Topic Assignment for Multi-Platform Chat

A fully unsupervised pipeline that takes a raw, multi-user chat export and turns it into:

1. **Semantically coherent segments** of the conversation (no timestamps or speaker-turn
   assumptions required),
2. **Data-derived topics** for those segments (no fixed topic list, no labeled training data),
3. **Per-user, per-topic timelines** — contiguous stretches of time each user spent on each
   topic, and
4. An **evaluation report** (precision / recall / F1) against a human-labeled reference column,
   if one is available.

This repo is the runnable pipeline behind the topic-discovery stage of the SCOPE line of work
(digital-forensic triage of seized multi-platform chat — see `SCOPE Extension.pdf` in the
project for the full methodology writeup and the encoder-selection experiments). It implements
the **segmentation → topic discovery → dynamic-threshold assignment** part of that pipeline
end to end, using a single sentence-transformer encoder rather than the multi-encoder /
SLM-disambiguation comparison described in the paper. See [How this relates to the
paper](#how-this-relates-to-the-paper) below.

## How it works, in one picture

```
raw dataset.csv                                                            (Stage 1)
   │  drop bad rows, coerce Text to plain strings
   ▼
dataset_fresh.csv                                                          (Stage 2)
   │  drift-rate boundary detection over sentence embeddings
   │  (no fixed window size; a topic-shift point is detected, then a
   │   reverse scan + local search pin down its exact boundary)
   ▼
conversation_with_index.csv  (+ Segment column)                            (Stage 3)
   │  embed each segment → HDBSCAN cluster (searches min_cluster_size /
   │   min_samples, keeps the best silhouette score) → KeyBERT names each
   │   cluster → softmax(cosine-to-centroid) scores every segment against
   │   every topic → per-topic Kneedle threshold on that probability column
   ▼
topic_assignments.csv (topic names + thresholds)
topic_definitions.json (+ each topic's defining text, for Stage 4 to reuse)
   │
   ▼                                                                       (Stage 4)
   walk each user's active segments in order; a segment extends a topic's
   run while the run's combined probability still clears that topic's
   threshold, otherwise the run closes
   ▼
topic_timeline.csv  (one row per user × topic × contiguous run)
   │
   ▼                                                                       (Stage 5 — evaluation, optional)
   build an all-"No" Manual_Topic × Predicted_Topic grid for you to
   hand-mark "Yes" wherever a manual label and a discovered topic agree
   ▼
topic_equivalence_grid.csv  (you hand-edit this)
   │
   ▼                                                                       (Stage 6 — evaluation, optional)
   score each timeline window against the raw dataset's manual labels
   ▼
window_matches.csv + window_score_report.txt  (precision / recall / F1)
```

Stages 1–4 are the actual topic-discovery pipeline and need nothing but the raw chat export.
Stages 5–6 are an **evaluation harness** — they only run if your dataset happens to carry a
manual/reference topic label per message, and stage 6 additionally needs a short round of
hand-editing from you (see [Evaluating against manual labels](#evaluating-against-manual-labels-stages-5-6)).

## Repository layout

```
config.py                     All tunable settings (model names, file paths, clustering
                               search ranges, column names). Edit this, not the stage scripts.
run_pipeline.py                Orchestrator — run this. Prompts for the dataset path and
                               stage 4's model; everything else comes from config.py.
stage1_clean_dataset.py        Stage 1: sanitize the Text column.
stage2_topic_shifting.py       Stage 2: segmentation (drift-rate boundary detection).
stage3_topic_naming.py         Stage 3: cluster segments into topics, name them, score them,
                               derive per-topic Kneedle thresholds.
stage4_topic_timeline.py       Stage 4: per-user topic timelines, reusing stage 3's topics.
stage5_topic_equivalence.py    Stage 5 (evaluation): builds the hand-editable equivalence grid.
stage6_window_score.py         Stage 6 (evaluation): precision/recall/F1 from that grid.
requirements.txt               Python dependencies.
data/sample/                   A small (~67-message, 2-chatroom) worked example: the raw
                               dataset plus every intermediate and final file the pipeline
                               produces from it, so you can see the expected shape of each
                               file without running anything first.
outputs/                       Empty — where your own run's outputs will land if you point
                               config.py's output paths here (see below).
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the sentence-transformer model(s) named in `config.py` /
`data/sample` from Hugging Face, so it needs internet access once; after that they're cached
locally.

## Running the pipeline

```bash
python run_pipeline.py
```

It's interactive and asks two questions at most:

1. **Raw dataset CSV path** — press Enter to accept `config.DATASET_PATH` (`dataset.csv` by
   default). The file must have a `Text` column; see [Input format](#input-format) below.
2. **Sentence-transformer model name for Stage 4** — press Enter to accept the default
   (`config.STAGE4_MODEL_NAME`). Stage 4 can use a *different* embedding model than stages
   1–3 without breaking anything (it re-embeds each topic's defining text with whichever
   model it's given — see the docstring in `stage4_topic_timeline.py`).

Stages 1–3's model is **not** prompted for — it's always `config.MODEL_NAME`. Change that
constant directly in `config.py` if you want a different model for cleaning/segmentation/
topic-naming; it's treated as a one-time pipeline setting, not a per-run choice.

If `conversation_with_index.csv` (stage 2's output) already exists, `run_pipeline.py` offers
to reuse it and jump straight to stage 4 — handy for re-running stage 4 with a different model
without redoing segmentation and clustering. Delete that file (or change
`config.SEGMENTED_CSV`) to force a full re-run.

### What you get back

| File | Produced by | Contents |
|---|---|---|
| `dataset_fresh.csv` | Stage 1 | Raw dataset with `Text` guaranteed to be a plain string |
| `conversation_with_index.csv` | Stage 2 | Every row, plus a `Segment` column |
| `topic_assignments.csv` | Stage 3 | One row per discovered topic: name + dynamic threshold |
| `topic_definitions.json` | Stage 3 | Same, plus each topic's defining text (Stage 4 reads this) |
| `topic_timeline.csv` | Stage 4 | One row per (user, topic, contiguous run): start/end, duration, probability, chat text |
| `topic_equivalence_grid.csv` | Stage 5 | Manual-topic × predicted-topic grid, for you to hand-mark |
| `window_matches.csv` | Stage 6 | Per-window message counts and match ratios |
| `window_score_report.txt` | Stage 6 | Per-topic precision/recall/F1 (macro, weighted, and overall/micro) |

By default these land in the working directory (matching `config.py`'s bare filenames). Point
the corresponding `config.py` constants at `outputs/...` if you'd rather they land there.

## Input format

The raw CSV needs, at minimum:

- **`Text`** — the message content (required for stages 1–4).

For per-user timelines (stage 4) it additionally needs:

- **`Sender`**, **`Timestamp`**, **`Chatroom`** — these pass through stages 1–2 untouched. If
  your columns are named differently, change `SENDER_COLUMN` / `TIMESTAMP_COLUMN` /
  `CHATROOM_COLUMN` in `config.py` rather than renaming your data. If any is missing, stage 4
  is skipped with a clear message and stages 1–3's outputs are kept.

For evaluation (stage 5) it additionally needs:

- A manual/reference topic label column, named by `config.MANUAL_TOPIC_COLUMN`
  (`"Predicted Topic"` in the sample data — an unfortunate name inherited from the source
  dataset; it's the ground-truth column, not the model's output). If missing, stage 5 is
  skipped with a clear message.

See `data/sample/dataset.csv` for a minimal worked example of all of the above.

## The worked example (`data/sample/`)

A tiny two-chatroom conversation (67 messages, 2 users) with every stage's output already
computed, so you can inspect the pipeline's behavior without installing anything:

- It discovers **2 topics** — "Nap" and "Practicing Sketching" — out of message content that
  also touches sketching-app tools, photography/photoshoots, and religion.
- `topic_timeline.csv` shows both users independently drifting into and out of the "Nap" and
  "Practicing Sketching" topics at different points in the conversation.
- `topic_equivalence_grid.csv` shows the hand-editing step: the manual labels "Photoshoot" and
  "Skteching" [sic] were both marked equivalent to the discovered "Practicing Sketching" topic,
  and "Sleep" to "Nap"; "Religious" was left unmatched (the pipeline never discovered a topic
  for it, since the demo dataset has only 2 discovered topics).
- `window_score_report.txt` shows the resulting per-manual-topic precision/recall/F1 — including
  the expected zero score for "Religious", the one manual topic the 2-topic pipeline run on
  this tiny sample couldn't discover.

This is deliberately a toy-scale run (a real run typically discovers more topics from a larger,
more diverse corpus). It exists to make the file formats and the evaluation workflow concrete,
not to demonstrate accuracy.

## Evaluating against manual labels (stages 5–6)

This is optional and only relevant if your dataset already carries a human-assigned topic
label per message.

1. Run the pipeline normally. If stage 4 produces a non-empty timeline **and** your dataset has
   `config.MANUAL_TOPIC_COLUMN`, stage 5 runs automatically and writes
   `topic_equivalence_grid.csv` — a table of every manual topic name (rows) against every
   discovered topic name (columns), all initialized to `"No"`.
2. Open that CSV and change cells to `"Yes"` wherever a manual topic and a discovered topic
   mean the same thing. One topic can be marked equivalent to more than one row/column (e.g.
   one discovered topic absorbing several manual topics).
3. `run_pipeline.py` pauses right after stage 5 with a prompt asking whether to run stage 6
   now (in the same terminal session, no need to re-invoke the script) — answer `y` once you're
   done editing. Answering `n` (or Enter) is fine too; run `python stage6_window_score.py`
   yourself whenever you're ready.
4. Stage 6 writes `window_matches.csv` (per-window detail: how many of a window's messages
   actually match its assigned topic) and `window_score_report.txt` (precision, recall, and F1
   per manual topic, plus macro/weighted/overall summaries).

If `window_matches.csv` shows every window with 0 member messages, stage 6 prints a diagnostic
comparing the `Sender`/`User` value lists and the timestamp ranges between the raw dataset and
the timeline — that combination is the signature of a `Sender` naming mismatch or a date-parsing
disagreement between the two files.

## Configuration reference (`config.py`)

| Setting | Purpose |
|---|---|
| `MODEL_NAME` | Sentence-transformer used by stages 1–3. Not prompted for — edit directly. |
| `DATASET_PATH` | Default raw dataset path (offered at the one prompt `run_pipeline.py` asks). |
| `CLEANED_CSV`, `SEGMENTED_CSV`, `RESULTS_CSV`, `TOPIC_DEFINITIONS_PATH`, `TIMELINE_CSV`, `EQUIVALENCE_GRID_CSV`, `WINDOW_MATCHES_CSV`, `WINDOW_SCORE_REPORT_TXT` | Output filenames for each stage. |
| `MIN_CLUSTER_SIZE_RANGE`, `MIN_SAMPLES_RANGE` | HDBSCAN search ranges for stage 3 — widen for more expected topics, narrow to speed up the search. |
| `STAGE4_MODEL_NAME` | Default offered at stage 4's model prompt. |
| `STAGE4_MODEL_KWARGS` | Extra kwargs passed to `SentenceTransformer(...)` for stage 4 — e.g. `{"trust_remote_code": True}` for some third-party HF models (Jina, etc.). |
| `SENDER_COLUMN`, `TIMESTAMP_COLUMN`, `CHATROOM_COLUMN` | Raw-dataset column names stage 4 needs, in case yours differ. |
| `MANUAL_TOPIC_COLUMN` | Raw-dataset column holding the ground-truth topic label, for stage 5/6 evaluation. |

## Methodology notes

- **Segmentation (stage 2)** detects topic-shift points from the *rate of change* in a
  running segment's mean cosine distance to its own constituent messages (no fixed window
  size), then uses a reverse scan and local search to pin the exact boundary — a port of the
  drift-rate boundary detection method described in the SCOPE paper's Algorithm 1/2.
- **Topic discovery (stage 3)** clusters segment embeddings with HDBSCAN, searching over
  `(min_cluster_size, min_samples)` and keeping whichever configuration gets the best
  silhouette score — so the number of topics comes from the data, not a fixed constant. Each
  cluster is named with KeyBERT (MMR-diversified 1–2 word keyphrase extraction).
- **Dynamic thresholds (stage 3)** use the Kneedle knee-of-the-curve method (Satopaa et al.,
  2011, "Finding a Kneedle in a Haystack") on each topic's softmax probability column, rather
  than one fixed global cutoff — see the extensive docstring on `kneedle_threshold()` in
  `stage3_topic_naming.py` for the exact derivation.
- **Per-user timelines (stage 4)** score eligibility and probability on each segment's *full*
  text (every participant's messages), not just one user's lines — so a one-word reply gets
  the surrounding context it needs to be judged accurately. Chat Summary in the output still
  reports only that user's own lines.

## How this relates to the paper

`SCOPE Extension.pdf` (in the project) describes a larger study: ~1.2M messages, 488 users,
seven sentence encoders benchmarked head-to-head, and a confidence-routed hybrid that escalates
only low-margin (geometrically ambiguous) topic assignments to a local Qwen2.5-3B-Instruct
model for disambiguation — raising human-verified precision on the high-confidence subset from
57% (best single encoder alone) to 100%.

This repository implements the **shared foundation** of that pipeline — segmentation, topic
discovery, dynamic per-topic thresholding, and per-user timeline extraction — with a single
configurable encoder. It does **not** yet include:

- Running/comparing multiple encoders side by side,
- The cosine-margin routing signal between top-1/top-2 topic candidates, or
- The Qwen2.5-3B-Instruct disambiguation step for low-margin assignments.

If you're extending this repo toward the full paper pipeline, the natural next step is a stage
between today's stage 3 and stage 4 that computes the top-1/top-2 cosine margin per group and
routes low-margin groups to a local SLM for constrained selection among the encoder's proposed
candidates.

## A note on data

The evaluation dataset used in the paper is real seized chat data and is not shareable (privacy
and chain-of-custody restrictions). `data/sample/` is a small, non-sensitive, publicly
shareable conversation used only to make the file formats concrete. Do not commit real
case/investigative data to this repository — `.gitignore` excludes `data/raw/` for that reason;
keep any real corpus outside of version control entirely.
