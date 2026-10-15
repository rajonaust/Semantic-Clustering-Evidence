# dataset3-coreset -- audited-label coreset of dataset 3 (E10-S1)

## 1. Provenance chain
- **Google Sheet:** Google Sheet `1yzfLk640FvqrluI8DAPqQH21yuSug8lqci6L-wy97II`, tab `gid=582793278` (workbook tabs: `ChitChat`, `TopicRules`), supplied by the user 2026-09-11 with the goal "a model or strategy of mixing models that will ensure consistency in all 3 datasets".
- **Unedited export:** `Semantic-Clustering-Evidence/data/dataset3-gid582793278/dataset.csv` (sha256 `bb9857731ef9392addbc81e5c2ac608fa546e5d05f971c126422772264e0254a`)
- **`coreset_manifest.json`:** `Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/label_audit/coreset_manifest.json` (sha256 `6ff7750da89bb118347cf21200b7b3bc110c20ed87071309ef1ec9934f5450cd`), seed=20260911, algorithm (first 300 chars): "Deterministic greedy whole-chatroom selection over the file's chatroom-contiguous original order. Step 1: include every chatroom that carries >=1 of the 14 free-form labels (F4 set: Personal, Country, Skteching, Sports, Greetings, Ambitious, Education, Friendship, Craziest Thing, Mozart, Religious, "...
- **`label_audit/`:** `Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/label_audit`
- **`JUDGE_RUBRIC.md`:** `Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/label_audit/JUDGE_RUBRIC.md` (sha256 `4e1c9d8d8ad6f3fcebc6e5a1ac55a0eacc811a188779f3fbde256d62b184ec8a`)
- **Judge identity:** Two independent Sonnet/Opus-tier judges, working from the identical rubric and packets, agree on only 78.2% [R:interjudge-raw-agreement] of exact message-level labels (kappa 0.747 [R:interjudge-cohens-kappa]) and 82.1% [R:interjudge-boundary-tolerant] even after excluding segment-boundary ambiguity. That is the ceiling on any single-judge-vs-single-judge comparison this coreset can support: a ...
- **Primary-judge assignment (packet -> token):** {'packet_01': 'judgeA', 'packet_02': 'judgeA', 'packet_03': 'judgeA', 'packet_04': 'judgeA', 'packet_05': 'judgeA', 'packet_06': 'judgeA', 'packet_07': 'judgeA', 'packet_08': 'judgeA', 'packet_09': 'judgeA'}
- **Secondary-judge assignment (double-judged packets only):** {'packet_01': 'judgeB', 'packet_06': 'judgeB'}
- **This script:** `/home/venkat/Documents/Scope_Project/Scope_Project_Extension/.claude/scripts/build_dataset3_coreset.py` (sha256 `e5cbe0f22993bad276a0027fe7c1f9dc6338bf6e87b0f5ec43312be74eb67f76`)
- **Exact `assemble` command line:** `uv run python .claude/scripts/build_dataset3_coreset.py assemble --manifest Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/label_audit/coreset_manifest.json --source Semantic-Clustering-Evidence/data/dataset3-gid582793278/dataset.csv --judgements-dir Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/label_audit/judgements --rubric Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/label_audit/JUDGE_RUBRIC.md --expect Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/label_audit/coreset_rows.csv --out Semantic-Clustering-Evidence/data/dataset3-coreset/dataset.csv --source-md Semantic-Clustering-Evidence/data/dataset3-coreset/SOURCE.md`

## 2. What `Predicted Topic` is
**These are LLM-judge AUDITED labels produced by E10-S0, not the supplied Gemini labels and not colleague hand labels.** `config.MANUAL_TOPIC_COLUMN` (`"Predicted Topic"`) is what Stages 5-6 score against; it is the primary judge's label for each coreset row, with the `Skteching` -> `Sketching` misspelling folded (Amendment B/D2) and nothing else changed. Inter-judge noise floor (from `Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/label_audit/interjudge_agreement.json`, keys `raw_agreement`/`cohens_kappa`): raw agreement = 0.7824 (Wilson 95% CI [0.7497984507855647, 0.8117352132631462]), Cohen's kappa = 0.7473 (bootstrap 95% CI [0.7112610212084773, 0.7818480893003911]), n=680 double-judged rows. Boundary-tolerant core agreement = 0.8208661417322834. Pk=0.12063839372975792, WindowDiff=0.12561839009273304. Cross-check against `Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/label_audit/S0_partAB_note.md`: MATCHES. Every dataset-3 accuracy number is bounded by this noise floor (board standing rule 7).

## 3. What `Gemini Topic` is
`Gemini Topic` is the source row's original supplied `Predicted Topic`, preserved verbatim, **never scored**. E10-S0's audit established that ~92% of the supplied labels are a chatroom-level default derived by substring-matching a `TopicRules` keyword against the chatroom's `Prompt` text (not a per-segment topic judgement), and that `Illustration` is largely a keyword-substring artifact (`art` inside part/start/heart/smart) rather than a genuine topic hit -- this is the whole reason the coreset carries audited labels.

## 4. Profile
- Rows: **2999**; chatrooms: **280**; senders: **251**
- Distinct non-null `Prompt` values: **34**; null-`Prompt` rows: **47**
- Per-label counts (audited `Predicted Topic`, V=23 labels): General: 918; Travel: 330; Food: 296; Music: 284; Education: 228; Movie: 160; Sleep: 149; Technology: 144; Money: 70; Mozart: 65; Sports: 62; Country: 54; Ambitious: 51; Personal: 49; Greetings: 30; Craziest Thing: 21; Relationship: 21; Hero: 18; Religious: 16; Illustration: 15; Fossil: 12; Friendship: 3; Sketching: 3
- Per-label counts (supplied `Gemini Topic`): General: 1041; Money: 322; Technology: 308; Illustration: 243; Music: 157; Movie: 153; Food: 152; Travel: 150; Sleep: 138; Personal: 68; Country: 57; Skteching: 39; Sports: 38; Greetings: 25; Ambitious: 24; Education: 22; Friendship: 16; Craziest Thing: 13; Mozart: 12; Religious: 11; Hero: 4; Sketching: 4; Fossil: 1; Relationship: 1
- `Text` length: mean 57.0 chars, max 231 chars
- Embedded-newline rows: **0** (Gate 7 -- a non-zero count would have stopped this run)
- Null/empty-`Text` rows carried into the coreset: **1** (Gate 8, report-only; Stage 1 drops no rows -- Amendment C)
- Timestamp range: **2018-04-10 18:37:27 .. 2018-11-15 0:21:32**
- Duplicate `(Sender, Timestamp)` pairs: **1**

## 5. Selection design
The coreset is a **stratified cluster sample, not a random row sample**: whole chatrooms, taken in the source file's original contiguous, time-ascending order, never sorted/shuffled/re-grouped -- whole chatrooms are what preserve segmentation continuity (designing-ml-systems sampling taxonomy: per-label quotas over whole chatrooms, cluster = sampling unit; and leakage-cause 1, splitting time-correlated data at random, is exactly what this design avoids).
- Seed: `20260911`. Target size: `'approx. 2,800 messages (soft target), 3,000 hard cap'`.
- Quotas: `{'rule_labels_quota_each': 150, 'prompt_coverage_required': 'all 34 real Prompt values + 1 null-prompt bucket'}`.
- Prompt coverage achieved: `36` / `36` (see E10-S1 Amendment E for a routed reporting-defect finding on this pair -- not corrected here).
- Achieved: **2999** messages across **280** chatrooms.
- Algorithm (verbatim from the manifest): "Deterministic greedy whole-chatroom selection over the file's chatroom-contiguous original order. Step 1: include every chatroom that carries >=1 of the 14 free-form labels (F4 set: Personal, Country, Skteching, Sports, Greetings, Ambitious, Education, Friendship, Craziest Thing, Mozart, Religious, Hero, Fossil, Relationship), whole, unconditionally (14 chatrooms, 374 rows). Step 2: for each of the 34 real Prompt values plus the null-prompt bucket ('None'/empty) not yet covered, add the smallest not-yet-selected chatroom carrying that prompt, preferring chatrooms sized 20-120 messages (falls back to the smallest of any size if none is in-band), ties among equal-sized in-band candidates broken by ascending chatroom number (510 rows). Step 3: for the 9 rule labels, processed in the order ['Travel', 'Sleep', 'Music', 'Food', 'Movie', 'Illustration', 'Money', 'Technology', 'Sketching'] (reachable labels first by ascending corpus scarcity; Sketching last, corpus total 95 rows < the 150 quota so it is mathematically unreachable, given a smaller per-label spend cap of 250 rows so it cannot starve the reachable quotas), greedily add not-yet-selected chatrooms carrying that label until its running count reaches 150 or a per-label message-spend cap (600 rows; 250 for Sketching) is hit or the 3000-row hard cap is hit (2115 rows). Candidates are ranked by (a) label-row density (label_count/chatroom_size) descending, (b) whether chatroom size falls in the preferred 20-120 band, (c) smaller chatroom size, (d) chatroom number -- all ties broken deterministically; every sort key is unique per candidate so the seed is recorded for reproducibility but not exercised as an RNG draw."

## 6. Column-tolerance ruling (Sec 4)
- **Ruling: `inline`** -- D3 columns are carried inline in `dataset.csv` (no `dataset_provenance.csv` sidecar was created) when `inline`.
- `conversation_with_index.csv` sha256 (A, six source columns only): `3f38e11e670aa8af5ae553d709f4b1b75a7c7dfe468d1c5aa37f8403ba72ed26`
- `conversation_with_index.csv` sha256 (B, + four D3 provenance columns): `825516431fa6dbe5cc4072dc6162dd54c9eb6dbc256fc165895f77878c32a32d`
- Evidence recorded in `.claude/logs/tasks/E10-S1.log.md` (X tier, Sec 4/AC-P1-6).

## 7. Rule line
Data only; pipeline = `../../stage*.py` unchanged (checksums in `.claude/logs/tasks/E8-colleague-code-checksums-before.txt`). Outputs for this dataset live under `Output/eval_v2.2/revision-r1/ablation/FULL-STAGES-dataset3/`, never here.
