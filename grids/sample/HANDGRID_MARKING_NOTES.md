# E8-S4 Phase 1 -- hand/strict topic-equivalence marking notes

Generated: 2026-09-10T20:20:15+00:00 | owner: E8-S4 Phase 1 (Claude pass (b), judgment only -- no driver code, no GPU)

Data file (the machine-readable output Phase 2 consumes): `.claude/scripts/data/E8_handgrids.json`.
This document is the human-readable companion: every distinct discovered-topic set, the cells that share it,
the marks under both rules, and the justification for every `Yes` and every non-obvious `No`.

**Evaluation unit.** The colleague pipeline's Stage-6 window score on
`Semantic-Clustering-Evidence/data/sample/` (66 annotated messages, 4 manual labels). Message indices below are
0-based row indices into `Semantic-Clustering-Evidence/data/sample/conversation_with_index.csv`.

**Row order (never reordered, never renamed):** `Photoshoot, Religious, Skteching, Sleep`. `Skteching` is the dataset's own spelling.

## The two rules (verbatim from `run_r1_full_stages_handgrid.py`'s pre-P2 module docstring)

> 'hand' = colleague precedent (shipped grid: Sleep=Nap; Skteching AND Photoshoot = Practicing Sketching, because the Photoshoot-labelled span itself says 'now all I am really doing is just practicing some sketching'), applied uniformly: sleep-subject names (Nap/Tired/Head Ache) = Sleep; religion names = Religious; sketching names = Skteching + Photoshoot; photoshoot names = Photoshoot; 'Personal Projects' (cluster = the sketching/photoshoot projects exchange) = both; 'Good Day' (greetings) = none.

> 'strict' = literal name synonymy only (Head Ache != Sleep; sketching names = Skteching only; Personal Projects = none).

## Resolution order

`by_cell[<slug>]` -> `by_topic_set[<topic-set key>]` -> unmarked. The topic-set key is the cell's discovered
`Topic_Name` values in `topic_definitions.json` array order joined with `|`. Marking is a property of the
**discovered names**, not of the cell, so one mark serves every cell that discovers the same topic set -- which
is what makes the `s4x__base-*` cells (they copy their base's `topic_definitions.json`) and the M1 mix cells
(they reuse `minilm-l6-v2`'s Stage-3 topics) resolve for free.

## What each manual label's span is actually about

- **Photoshoot** -- msgs 39-50 -- 'was this a paid photoshoot?' / 'engagement shoots or weddings are goooood money' / 'but i hear in utah its super competitive' / 'do you think if you wanted to pursue your photography you wuold have to move out of utah?' / 'too hard to make a living on'. 12 rows.
- **Religious** -- msgs 51-65 -- 'if you werent Christian what religion would you be in if at all?' / 'Maybe Buddhist' / 'I probably would be buddhist culturally because IM korean' / 'egnostic? is that the term?', then the closing ('I need to go. Thanks for chatting!' / 'Hope you have a good day!') and the Chatroom-2 greetings ('hi' / 'anyone here' / 'hey whats up'). 15 rows.
- **Skteching** -- msgs 14-38 -- 'im refining my skills on Sketch App' / 'its a vector graphic editor like illustrator' / 'Im doing a free lance job for a professor for his app idea' / 'do you have any personal projects you are working on?' / 'Not really projects. I finished editing a bunch of engagement pictures. So now all I am really doing is just practicing some sketching'. 25 rows.
- **Sleep** -- msgs 0-13 -- 'Hello' / 'How are you doing today?' / 'whats up MD' / 'im doing good' / 'Im alright, I just took a nap. But it was one of those naps that doesnt help anything' / 'so you still feel tired huh?' / 'did you go to bed late?' / 'I have more of a head ache than anything else' / 'Its just adjusting to a time change'. 14 rows.

## Marking method (how the two rules were applied)

The colleague precedent is **name-driven**: the mark follows the topical category of the discovered *name*, and
the cluster span is consulted only to resolve a name that is not topically self-evident. Three precedent
observations pin this down, and every new marking below is derived from them:

1. `Sketch App` on `uae-large-v1` and `bge-large-en-v1.5`, and `Sketch` on `minilm-l12-v2`, have clusters that
   swallow the Sleep-labelled msgs 0-13, and the colleague grid still gives them **no** Sleep. So a span that
   spills into another label does not earn that label when the name is self-evident.
2. `Engagement Shoots` (`mxbai-large-v1`) and `Pursue Photography` (`uae-large-v1`) have clusters that span the
   Skteching tail (msgs 28-38 / msg 36, which contains 'practicing some sketching'), and both are marked
   **Photoshoot only**. Same conclusion, opposite direction.
3. `Personal Projects` (`e5-large-v2`) and `Good Day` (`bge`, `mxbai`) are **not** self-evident names, and the
   docstring resolves both from their cluster content -- 'cluster = the sketching/photoshoot projects exchange'
   -> both labels; greetings -> none. That is the licensed method for a name outside the four precedent
   categories, and it is the method used below for `Utah` and `Hello`.

Consistency was preferred over generosity: each discovered name carries exactly one `hand` mark and one
`strict` mark across every topic set in which it appears (verified programmatically -- 0 inconsistencies over
16 distinct names).

## Per-name mark register

| discovered name | hand | strict | precedent category | source |
|---|---|---|---|---|
| `Buddhist` | -Y--  (Religious) | -Y--  (Religious) | religion name | E8-S4-P1, extends `Maybe Buddhist` |
| `Engagement Shoots` | Y---  (Photoshoot) | Y---  (Photoshoot) | photoshoot name | colleague precedent (mxbai-large-v1) |
| `Good Day` | ----  (none) | ----  (none) | greetings -> none | colleague precedent (bge, mxbai) |
| `Head Ache` | ---Y  (Sleep) | ----  (none) | sleep-subject name (hand only) | colleague precedent (e5-large-v2) |
| `Hello` | ---Y  (Sleep) | ----  (none) | **not self-evident** -> resolved from cluster (Sleep) | E8-S4-P1, **ESCALATED** |
| `Maybe Buddhist` | -Y--  (Religious) | -Y--  (Religious) | religion name | colleague precedent (minilm-l12-v2) |
| `Nap` | ---Y  (Sleep) | ---Y  (Sleep) | sleep-subject name | colleague precedent (shipped grid) |
| `Paid Photoshoot` | Y---  (Photoshoot) | Y---  (Photoshoot) | photoshoot name | E8-S4-P1, extends the photoshoot category |
| `Personal Projects` | Y-Y-  (Photoshoot, Skteching) | ----  (none) | **not self-evident** -> both (projects exchange) | colleague precedent (e5-large-v2) |
| `Practicing Sketching` | Y-Y-  (Photoshoot, Skteching) | --Y-  (Skteching) | sketching name | colleague precedent (shipped grid) |
| `Pursue Photography` | Y---  (Photoshoot) | Y---  (Photoshoot) | photoshoot name | colleague precedent (uae-large-v1) |
| `Sketch` | Y-Y-  (Photoshoot, Skteching) | --Y-  (Skteching) | sketching name | colleague precedent (minilm-l12-v2) |
| `Sketch App` | Y-Y-  (Photoshoot, Skteching) | --Y-  (Skteching) | sketching name | colleague precedent (bge, e5, mxbai, uae) |
| `Tired` | ---Y  (Sleep) | ---Y  (Sleep) | sleep-subject name | colleague precedent (minilm-l12-v2) |
| `Tired Huh` | ---Y  (Sleep) | ---Y  (Sleep) | sleep-subject name | E8-S4-P1, extends `Tired` |
| `Utah` | Y---  (Photoshoot) | ----  (none) | **not self-evident** -> resolved from cluster (Photoshoot) | E8-S4-P1 |

Mark strings are `Y`/`N`, one character per manual topic, in `Photoshoot, Religious, Skteching, Sleep` order.

## Topic sets

14 distinct discovered-topic sets cover all 26 `status=ok` cells on disk at generation time; 8 of them are new Phase-1 markings and 6 mirror an existing colleague-precedent `by_cell` entry.

### 1. `Buddhist|Paid Photoshoot|Nap|Practicing Sketching`  --  NEW (E8-S4-P1)

Cells sharing this topic set (1):

- `mix__seg-qwen3-emb-0.6b__name-minilm-l6-v2__timeline-minilm-l6-v2` (status=ok)

| Manual_Topic \ Predicted_Topic | Buddhist (hand) | Paid Photoshoot (hand) | Nap (hand) | Practicing Sketching (hand) | Buddhist (strict) | Paid Photoshoot (strict) | Nap (strict) | Practicing Sketching (strict) |
|---|---|---|---|---|---|---|---|---|
| Photoshoot | No | **Yes** | No | **Yes** | No | **Yes** | No | No |
| Religious | **Yes** | No | No | No | **Yes** | No | No | No |
| Skteching | No | No | No | **Yes** | No | No | No | **Yes** |
| Sleep | No | No | **Yes** | No | No | No | **Yes** | No |

Justifications:

- **`Buddhist`** -- religion name -> Religious; the same name as the precedent 'Maybe Buddhist' with the hedge dropped, so marked identically. Cluster = the 'if you werent Christian what religion would you be in if at all? / Maybe Buddhist / I probably would be buddhist culturally because IM korean' exchange = Religious-labelled msgs 51-65.
- **`Paid Photoshoot`** -- photoshoot name, literally the manual label's own word, from Photoshoot-labelled msg 39 'was this a paid photoshoot?'; cluster = the Photoshoot span (msgs 39-50, 'engagement shoots or weddings are goooood money ... too hard to make a living on'). Hand: 'photoshoot names = Photoshoot'. Strict Y on the same basis the precedent gives strict Y to 'Engagement Shoots' and 'Pursue Photography' -- literal name synonymy with 'Photoshoot' holds, here even word-identically.
- **`Nap`** -- sleep-subject name and the shipped colleague grid's own Sleep=Nap row (hand rule: 'Sleep=Nap'); cluster is the 'I just took a nap ... so you still feel tired huh ... I have more of a head ache' exchange = Sleep-labelled msgs 0-13. Strict Y too: 'Nap' is a literal sleep synonym (the docstring excludes only 'Head Ache' from strict).
- **`Practicing Sketching`** -- colleague precedent verbatim: sketching name -> Skteching, plus Photoshoot under hand because the Photoshoot-labelled span itself says 'now all I am really doing is just practicing some sketching' (msg 36). Strict drops Photoshoot (sketching names = Skteching only).

### 2. `Engagement Shoots|Good Day|Sketch App`  --  mirrors colleague precedent

Cells sharing this topic set (1):

- `mxbai-large-v1` (status=ok)

| Manual_Topic \ Predicted_Topic | Engagement Shoots (hand) | Good Day (hand) | Sketch App (hand) | Engagement Shoots (strict) | Good Day (strict) | Sketch App (strict) |
|---|---|---|---|---|---|---|
| Photoshoot | **Yes** | No | **Yes** | **Yes** | No | No |
| Religious | No | No | No | No | No | No |
| Skteching | No | No | **Yes** | No | No | **Yes** |
| Sleep | No | No | No | No | No | No |

Justifications:

- **`Engagement Shoots`** -- photoshoot name from Photoshoot-labelled msg 41 'engagement shoots or weddings are goooood money'; colleague precedent on mxbai-large-v1, Y under both rules. Non-obvious No on Skteching: the mxbai cluster also spans Skteching-labelled msgs 28-38 and the precedent still marks Photoshoot only -- photoshoot names get Photoshoot, the span does not add Skteching.
- **`Good Day`** -- greetings -> none; colleague precedent on bge-large-en-v1.5 and mxbai-large-v1. Non-obvious No justified: the cluster is greetings ONLY -- msgs 0-3 ('Hello', 'How are you doing today?', 'whats up MD', 'im doing good', which are Sleep-labelled) plus the closing msgs 59-65 ('I need to go. Thanks for chatting!', 'Hope you have a good day!', 'hi', 'anyone here', 'hey whats up') -- and carries none of the nap / tired / head-ache content, which is why it earns no Sleep despite covering Sleep-labelled rows.
- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.

### 3. `Good Day|Sketch App`  --  mirrors colleague precedent

Cells sharing this topic set (1):

- `bge-large-en-v1.5` (status=ok)

| Manual_Topic \ Predicted_Topic | Good Day (hand) | Sketch App (hand) | Good Day (strict) | Sketch App (strict) |
|---|---|---|---|---|
| Photoshoot | No | **Yes** | No | No |
| Religious | No | No | No | No |
| Skteching | No | **Yes** | No | **Yes** |
| Sleep | No | No | No | No |

Justifications:

- **`Good Day`** -- greetings -> none; colleague precedent on bge-large-en-v1.5 and mxbai-large-v1. Non-obvious No justified: the cluster is greetings ONLY -- msgs 0-3 ('Hello', 'How are you doing today?', 'whats up MD', 'im doing good', which are Sleep-labelled) plus the closing msgs 59-65 ('I need to go. Thanks for chatting!', 'Hope you have a good day!', 'hi', 'anyone here', 'hey whats up') -- and carries none of the nap / tired / head-ache content, which is why it earns no Sleep despite covering Sleep-labelled rows.
- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.

### 4. `Head Ache|Sketch App|Personal Projects`  --  mirrors colleague precedent

Cells sharing this topic set (1):

- `e5-large-v2` (status=ok)

| Manual_Topic \ Predicted_Topic | Head Ache (hand) | Sketch App (hand) | Personal Projects (hand) | Head Ache (strict) | Sketch App (strict) | Personal Projects (strict) |
|---|---|---|---|---|---|---|
| Photoshoot | No | **Yes** | **Yes** | No | No | No |
| Religious | No | No | No | No | No | No |
| Skteching | No | **Yes** | **Yes** | No | **Yes** | No |
| Sleep | **Yes** | No | No | No | No | No |

Justifications:

- **`Head Ache`** -- hand: sleep-subject name, named explicitly by the hand rule. Strict: the docstring's own exclusion 'Head Ache != Sleep' -> no Yes anywhere.
- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.
- **`Personal Projects`** -- colleague precedent verbatim (e5-large-v2): the cluster is the sketching/photoshoot projects exchange -- 'do you have any personal projects you are working on?' (msg 35) answered 'Not really projects. I finished editing a bunch of engagement pictures. So now all I am really doing is just practicing some sketching' (msg 36) -- so hand = both Photoshoot and Skteching. Strict = none: 'Personal Projects' is no manual label's literal synonym (the docstring says so explicitly).

### 5. `Hello|Sketch App|Paid Photoshoot`  --  NEW (E8-S4-P1)

Cells sharing this topic set (1):

- `mix__seg-qwen3-emb-0.6b__name-uae-large-v1__timeline-uae-large-v1` (status=ok)

| Manual_Topic \ Predicted_Topic | Hello (hand) | Sketch App (hand) | Paid Photoshoot (hand) | Hello (strict) | Sketch App (strict) | Paid Photoshoot (strict) |
|---|---|---|---|---|---|---|
| Photoshoot | No | **Yes** | **Yes** | No | No | **Yes** |
| Religious | No | No | No | No | No | No |
| Skteching | No | **Yes** | No | No | **Yes** | No |
| Sleep | **Yes** | No | No | No | No | No |

> **OPEN QUESTION for the user:** 'Hello' hand mark: NNNY (Sleep) on cluster-content grounds vs NNNN by analogy to the greeting-named 'Good Day' precedent. Marked NNNY under hand, NNNN under strict, pending user adjudication. Only cell affected: mix__seg-qwen3-emb-0.6b__name-uae-large-v1__timeline-uae-large-v1.

Justifications:

- **`Hello`** -- ESCALATED / open question -- see HANDGRID_MARKING_NOTES.md and the E8-S4 log. Best hand reading = Sleep: 'Hello' is not topically self-evident, so it is resolved from its cluster (the docstring's method for 'Personal Projects'), and that cluster is the full sleep exchange -- Sleep-labelled msgs 0-10, containing all three sleep-subject markers the hand rule names ('I just took a nap', 'so you still feel tired huh?', 'I have more of a head ache') -- plus the closing greetings msgs 59-65; its timeline windows 0/4 are the sleep-span windows. This is the same message set that qwen3-emb-0.6b names 'Tired Huh' (marked Sleep), so marking it No would penalise a naming choice rather than the clustering. The competing reading is NNNN by analogy to the greeting-named precedent 'Good Day'; the two are distinguished by content -- 'Good Day' is greetings only, 'Hello' contains the nap/tired/head-ache exchange. Strict = NNNN either way (no literal synonymy, and the escalation rule requires strict No). Affects one cell only: mix__seg-qwen3-emb-0.6b__name-uae-large-v1__timeline-uae-large-v1.
- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.
- **`Paid Photoshoot`** -- photoshoot name, literally the manual label's own word, from Photoshoot-labelled msg 39 'was this a paid photoshoot?'; cluster = the Photoshoot span (msgs 39-50, 'engagement shoots or weddings are goooood money ... too hard to make a living on'). Hand: 'photoshoot names = Photoshoot'. Strict Y on the same basis the precedent gives strict Y to 'Engagement Shoots' and 'Pursue Photography' -- literal name synonymy with 'Photoshoot' holds, here even word-identically.

### 6. `Maybe Buddhist|Sketch App|Paid Photoshoot`  --  NEW (E8-S4-P1)

Cells sharing this topic set (1):

- `qwen3-emb-4b` (status=ok)

| Manual_Topic \ Predicted_Topic | Maybe Buddhist (hand) | Sketch App (hand) | Paid Photoshoot (hand) | Maybe Buddhist (strict) | Sketch App (strict) | Paid Photoshoot (strict) |
|---|---|---|---|---|---|---|
| Photoshoot | No | **Yes** | **Yes** | No | No | **Yes** |
| Religious | **Yes** | No | No | **Yes** | No | No |
| Skteching | No | **Yes** | No | No | **Yes** | No |
| Sleep | No | No | No | No | No | No |

Justifications:

- **`Maybe Buddhist`** -- religion name; colleague precedent on minilm-l12-v2 (Y under both rules).
- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.
- **`Paid Photoshoot`** -- photoshoot name, literally the manual label's own word, from Photoshoot-labelled msg 39 'was this a paid photoshoot?'; cluster = the Photoshoot span (msgs 39-50, 'engagement shoots or weddings are goooood money ... too hard to make a living on'). Hand: 'photoshoot names = Photoshoot'. Strict Y on the same basis the precedent gives strict Y to 'Engagement Shoots' and 'Pursue Photography' -- literal name synonymy with 'Photoshoot' holds, here even word-identically.

### 7. `Maybe Buddhist|Tired|Sketch`  --  mirrors colleague precedent

Cells sharing this topic set (1):

- `minilm-l12-v2` (status=ok)

| Manual_Topic \ Predicted_Topic | Maybe Buddhist (hand) | Tired (hand) | Sketch (hand) | Maybe Buddhist (strict) | Tired (strict) | Sketch (strict) |
|---|---|---|---|---|---|---|
| Photoshoot | No | No | **Yes** | No | No | No |
| Religious | **Yes** | No | No | **Yes** | No | No |
| Skteching | No | No | **Yes** | No | No | **Yes** |
| Sleep | No | **Yes** | No | No | **Yes** | No |

Justifications:

- **`Maybe Buddhist`** -- religion name; colleague precedent on minilm-l12-v2 (Y under both rules).
- **`Tired`** -- sleep-subject name; colleague precedent on minilm-l12-v2 (Y under both rules).
- **`Sketch`** -- sketching name; colleague precedent on minilm-l12-v2.

### 8. `Nap|Engagement Shoots|Sketch App`  --  NEW (E8-S4-P1)

Cells sharing this topic set (1):

- `mix__seg-e5-large-v2__name-minilm-l6-v2__timeline-minilm-l6-v2` (status=ok)

| Manual_Topic \ Predicted_Topic | Nap (hand) | Engagement Shoots (hand) | Sketch App (hand) | Nap (strict) | Engagement Shoots (strict) | Sketch App (strict) |
|---|---|---|---|---|---|---|
| Photoshoot | No | **Yes** | **Yes** | No | **Yes** | No |
| Religious | No | No | No | No | No | No |
| Skteching | No | No | **Yes** | No | No | **Yes** |
| Sleep | **Yes** | No | No | **Yes** | No | No |

Justifications:

- **`Nap`** -- sleep-subject name and the shipped colleague grid's own Sleep=Nap row (hand rule: 'Sleep=Nap'); cluster is the 'I just took a nap ... so you still feel tired huh ... I have more of a head ache' exchange = Sleep-labelled msgs 0-13. Strict Y too: 'Nap' is a literal sleep synonym (the docstring excludes only 'Head Ache' from strict).
- **`Engagement Shoots`** -- photoshoot name from Photoshoot-labelled msg 41 'engagement shoots or weddings are goooood money'; colleague precedent on mxbai-large-v1, Y under both rules. Non-obvious No on Skteching: the mxbai cluster also spans Skteching-labelled msgs 28-38 and the precedent still marks Photoshoot only -- photoshoot names get Photoshoot, the span does not add Skteching.
- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.

### 9. `Nap|Practicing Sketching`  --  mirrors colleague precedent

Cells sharing this topic set (9):

- `minilm-l6-v2` (status=ok)
- `mix__seg-minilm-l6-v2__name-minilm-l6-v2__timeline-arctic-l-v2.0` (status=ok)
- `mix__seg-minilm-l6-v2__name-minilm-l6-v2__timeline-e5-large-v2` (status=ok)
- `mix__seg-minilm-l6-v2__name-minilm-l6-v2__timeline-qwen3-emb-0.6b` (status=ok)
- `mix__seg-minilm-l6-v2__name-minilm-l6-v2__timeline-uae-large-v1` (status=ok)
- `s4x__base-minilm-l6-v2__bge-reranker-v2-m3__q0.5` (status=ok)
- `s4x__base-minilm-l6-v2__bge-reranker-v2-m3__q1.0` (status=ok)
- `s4x__base-minilm-l6-v2__qwen2.5-3b-instruct__q0.5` (status=ok)
- `s4x__base-minilm-l6-v2__qwen2.5-3b-instruct__q1.0` (status=ok)

| Manual_Topic \ Predicted_Topic | Nap (hand) | Practicing Sketching (hand) | Nap (strict) | Practicing Sketching (strict) |
|---|---|---|---|---|
| Photoshoot | No | **Yes** | No | No |
| Religious | No | No | No | No |
| Skteching | No | **Yes** | No | **Yes** |
| Sleep | **Yes** | No | **Yes** | No |

Justifications:

- **`Nap`** -- sleep-subject name and the shipped colleague grid's own Sleep=Nap row (hand rule: 'Sleep=Nap'); cluster is the 'I just took a nap ... so you still feel tired huh ... I have more of a head ache' exchange = Sleep-labelled msgs 0-13. Strict Y too: 'Nap' is a literal sleep synonym (the docstring excludes only 'Head Ache' from strict).
- **`Practicing Sketching`** -- colleague precedent verbatim: sketching name -> Skteching, plus Photoshoot under hand because the Photoshoot-labelled span itself says 'now all I am really doing is just practicing some sketching' (msg 36). Strict drops Photoshoot (sketching names = Skteching only).

### 10. `Paid Photoshoot|Tired Huh|Sketch App`  --  NEW (E8-S4-P1)

Cells sharing this topic set (5):

- `qwen3-emb-0.6b` (status=ok)
- `s4x__base-qwen3-emb-0.6b__bge-reranker-v2-m3__q0.5` (status=ok)
- `s4x__base-qwen3-emb-0.6b__bge-reranker-v2-m3__q1.0` (status=ok)
- `s4x__base-qwen3-emb-0.6b__qwen2.5-3b-instruct__q0.5` (status=ok)
- `s4x__base-qwen3-emb-0.6b__qwen2.5-3b-instruct__q1.0` (status=ok)

| Manual_Topic \ Predicted_Topic | Paid Photoshoot (hand) | Tired Huh (hand) | Sketch App (hand) | Paid Photoshoot (strict) | Tired Huh (strict) | Sketch App (strict) |
|---|---|---|---|---|---|---|
| Photoshoot | **Yes** | No | **Yes** | **Yes** | No | No |
| Religious | No | No | No | No | No | No |
| Skteching | No | No | **Yes** | No | No | **Yes** |
| Sleep | No | **Yes** | No | No | **Yes** | No |

Justifications:

- **`Paid Photoshoot`** -- photoshoot name, literally the manual label's own word, from Photoshoot-labelled msg 39 'was this a paid photoshoot?'; cluster = the Photoshoot span (msgs 39-50, 'engagement shoots or weddings are goooood money ... too hard to make a living on'). Hand: 'photoshoot names = Photoshoot'. Strict Y on the same basis the precedent gives strict Y to 'Engagement Shoots' and 'Pursue Photography' -- literal name synonymy with 'Photoshoot' holds, here even word-identically.
- **`Tired Huh`** -- the KeyBERT bigram of Sleep-labelled msg 7 'so you still feel tired huh?', i.e. the same sleep-subject name as the precedent 'Tired' on minilm-l12-v2 with the interjection attached; cluster = Sleep-labelled msgs 0-10 ('Hello ... I just took a nap ... head ache'). Marked identically to 'Tired' under both rules; strict Y because the literal sleep synonym 'tired' is present in the name.
- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.

### 11. `Personal Projects|Maybe Buddhist`  --  NEW (E8-S4-P1)

Cells sharing this topic set (1):

- `mix__seg-minilm-l6-v2__name-e5-large-v2__timeline-e5-large-v2` (status=ok)

| Manual_Topic \ Predicted_Topic | Personal Projects (hand) | Maybe Buddhist (hand) | Personal Projects (strict) | Maybe Buddhist (strict) |
|---|---|---|---|---|
| Photoshoot | **Yes** | No | No | No |
| Religious | No | **Yes** | No | **Yes** |
| Skteching | **Yes** | No | No | No |
| Sleep | No | No | No | No |

Justifications:

- **`Personal Projects`** -- colleague precedent verbatim (e5-large-v2): the cluster is the sketching/photoshoot projects exchange -- 'do you have any personal projects you are working on?' (msg 35) answered 'Not really projects. I finished editing a bunch of engagement pictures. So now all I am really doing is just practicing some sketching' (msg 36) -- so hand = both Photoshoot and Skteching. Strict = none: 'Personal Projects' is no manual label's literal synonym (the docstring says so explicitly).
- **`Maybe Buddhist`** -- religion name; colleague precedent on minilm-l12-v2 (Y under both rules).

### 12. `Sketch App|Paid Photoshoot`  --  NEW (E8-S4-P1)

Cells sharing this topic set (1):

- `mix__seg-uae-large-v1__name-qwen3-emb-0.6b__timeline-qwen3-emb-0.6b` (status=ok)

| Manual_Topic \ Predicted_Topic | Sketch App (hand) | Paid Photoshoot (hand) | Sketch App (strict) | Paid Photoshoot (strict) |
|---|---|---|---|---|
| Photoshoot | **Yes** | **Yes** | No | **Yes** |
| Religious | No | No | No | No |
| Skteching | **Yes** | No | **Yes** | No |
| Sleep | No | No | No | No |

Justifications:

- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.
- **`Paid Photoshoot`** -- photoshoot name, literally the manual label's own word, from Photoshoot-labelled msg 39 'was this a paid photoshoot?'; cluster = the Photoshoot span (msgs 39-50, 'engagement shoots or weddings are goooood money ... too hard to make a living on'). Hand: 'photoshoot names = Photoshoot'. Strict Y on the same basis the precedent gives strict Y to 'Engagement Shoots' and 'Pursue Photography' -- literal name synonymy with 'Photoshoot' holds, here even word-identically.

### 13. `Sketch App|Pursue Photography`  --  mirrors colleague precedent

Cells sharing this topic set (1):

- `uae-large-v1` (status=ok)

| Manual_Topic \ Predicted_Topic | Sketch App (hand) | Pursue Photography (hand) | Sketch App (strict) | Pursue Photography (strict) |
|---|---|---|---|---|
| Photoshoot | **Yes** | **Yes** | No | **Yes** |
| Religious | No | No | No | No |
| Skteching | **Yes** | No | **Yes** | No |
| Sleep | No | No | No | No |

Justifications:

- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.
- **`Pursue Photography`** -- photoshoot name from Photoshoot-labelled msg 46 'do you think if you wanted to pursue your photography you wuold have to move out of utah?'; colleague precedent on uae-large-v1, Y under both rules. Non-obvious No on Skteching: its cluster starts at msg 36, which contains 'practicing some sketching', and the precedent still marks Photoshoot only.

### 14. `Sketch App|Utah`  --  NEW (E8-S4-P1)

Cells sharing this topic set (1):

- `mix__seg-minilm-l6-v2__name-uae-large-v1__timeline-uae-large-v1` (status=ok)

| Manual_Topic \ Predicted_Topic | Sketch App (hand) | Utah (hand) | Sketch App (strict) | Utah (strict) |
|---|---|---|---|---|
| Photoshoot | **Yes** | **Yes** | No | No |
| Religious | No | No | No | No |
| Skteching | **Yes** | No | **Yes** | No |
| Sleep | No | No | No | No |

Justifications:

- **`Sketch App`** -- sketching name, same category as 'Practicing Sketching'; already precedent on bge-large-en-v1.5, e5-large-v2, mxbai-large-v1 and uae-large-v1. Name governs, not span: uae-large-v1's and bge-large-en-v1.5's 'Sketch App' clusters also swallow the Sleep-labelled msgs 0-13 and the colleague grid still gives them no Sleep, so cells whose 'Sketch App' cluster spans the sleep exchange (qwen3-emb-4b, the uae/qwen3 mixes) get the same YNYN/NNYN.
- **`Utah`** -- not a topically self-evident name, so resolved from its cluster -- the docstring's own method for 'Personal Projects'. The cluster is msgs 33-51 and is majority Photoshoot (12 of 19 rows: msgs 39-50), and 'Utah' is uttered only inside that Photoshoot span ('but i hear in utah its super competitive', msg 42; 'move out of utah', msg 46) -- so hand = Photoshoot. Non-obvious No on Skteching: the cluster's msgs 33-38 tail is Skteching-labelled, but the precedent pair 'Engagement Shoots' (mxbai) and 'Pursue Photography' (uae) marks Photoshoot only on this same sketching+photoshoot span, so adding Skteching would break that precedent. Strict = none: 'Utah' is a place name with no literal synonymy to any manual label.

## Cells left unmarked (nothing to mark)

| cell | status | reason |
|---|---|---|
| `arctic-l-v2.0` | failed | RuntimeError: No clusters were found -- try more segments, or widen the min_cluster_size_range/min_samples_range passed to cluster_segments(). |
| `bge-m3` | failed | RuntimeError: No clusters were found -- try more segments, or widen the min_cluster_size_range/min_samples_range passed to cluster_segments(). |
| `mix__seg-minilm-l6-v2__name-qwen3-emb-0.6b__timeline-qwen3-emb-0.6b` | failed | RuntimeError: No clusters were found -- try more segments, or widen the min_cluster_size_range/min_samples_range passed to cluster_segments(). |
| `mix__seg-uae-large-v1__name-minilm-l6-v2__timeline-minilm-l6-v2` | failed | RuntimeError: No clusters were found -- try more segments, or widen the min_cluster_size_range/min_samples_range passed to cluster_segments(). |
| `mpnet-base-v2` | failed | RuntimeError: No clusters were found -- try more segments, or widen the min_cluster_size_range/min_samples_range passed to cluster_segments(). |
| `nomic-embed-v1.5` | failed | RuntimeError: No clusters were found -- try more segments, or widen the min_cluster_size_range/min_samples_range passed to cluster_segments(). |
| `s4x__base-minilm-l6-v2__phi-4-mini-instruct__q0.5` | failed | load_second_stage('phi-4-mini-instruct') raised -- see traceback / ImportError: cannot import name 'LossKwargs' from 'transformers.utils' (/home/venkat/Documents/Scope_Project/Scope_Project_Extension/.venv/lib/python3.12/site-packages/transformers/utils/__init__.py) |
| `s4x__base-minilm-l6-v2__phi-4-mini-instruct__q1.0` | failed | load_second_stage('phi-4-mini-instruct') raised -- see traceback / ImportError: cannot import name 'LossKwargs' from 'transformers.utils' (/home/venkat/Documents/Scope_Project/Scope_Project_Extension/.venv/lib/python3.12/site-packages/transformers/utils/__init__.py) |
| `s4x__base-minilm-l6-v2__phi-4__q1.0` | skipped | 14B, not cached; ~29 GB fp16 download exceeds the epic's <10 GB NF4 download rule. Recorded as a status=skipped cell, not attempted. |
| `s4x__base-minilm-l6-v2__qwen3-reranker-0.6b__q0.5` | failed | ValueError: Cannot handle batch sizes > 1 if no padding token is defined. |
| `s4x__base-minilm-l6-v2__qwen3-reranker-0.6b__q1.0` | failed | ValueError: Cannot handle batch sizes > 1 if no padding token is defined. |
| `s4x__base-qwen3-emb-0.6b__phi-4-mini-instruct__q0.5` | failed | load_second_stage('phi-4-mini-instruct') raised -- see traceback / ImportError: cannot import name 'LossKwargs' from 'transformers.utils' (/home/venkat/Documents/Scope_Project/Scope_Project_Extension/.venv/lib/python3.12/site-packages/transformers/utils/__init__.py) |
| `s4x__base-qwen3-emb-0.6b__phi-4-mini-instruct__q1.0` | failed | load_second_stage('phi-4-mini-instruct') raised -- see traceback / ImportError: cannot import name 'LossKwargs' from 'transformers.utils' (/home/venkat/Documents/Scope_Project/Scope_Project_Extension/.venv/lib/python3.12/site-packages/transformers/utils/__init__.py) |
| `s4x__base-qwen3-emb-0.6b__qwen3-reranker-0.6b__q0.5` | failed | ValueError: Cannot handle batch sizes > 1 if no padding token is defined. |
| `s4x__base-qwen3-emb-0.6b__qwen3-reranker-0.6b__q1.0` | failed | ValueError: Cannot handle batch sizes > 1 if no padding token is defined. |
| `s4x__base-uae-large-v1__bge-reranker-v2-m3__q0.5` | skipped_base_unusable | base cache status=gate_failed |
| `s4x__base-uae-large-v1__bge-reranker-v2-m3__q1.0` | skipped_base_unusable | base cache status=gate_failed |
| `s4x__base-uae-large-v1__phi-4-mini-instruct__q0.5` | failed | load_second_stage('phi-4-mini-instruct') raised -- see traceback / ImportError: cannot import name 'LossKwargs' from 'transformers.utils' (/home/venkat/Documents/Scope_Project/Scope_Project_Extension/.venv/lib/python3.12/site-packages/transformers/utils/__init__.py) |
| `s4x__base-uae-large-v1__phi-4-mini-instruct__q1.0` | failed | load_second_stage('phi-4-mini-instruct') raised -- see traceback / ImportError: cannot import name 'LossKwargs' from 'transformers.utils' (/home/venkat/Documents/Scope_Project/Scope_Project_Extension/.venv/lib/python3.12/site-packages/transformers/utils/__init__.py) |
| `s4x__base-uae-large-v1__qwen2.5-3b-instruct__q0.5` | skipped_base_unusable | base cache status=gate_failed |
| `s4x__base-uae-large-v1__qwen2.5-3b-instruct__q1.0` | skipped_base_unusable | base cache status=gate_failed |
| `s4x__base-uae-large-v1__qwen3-reranker-0.6b__q0.5` | skipped_base_unusable | base cache status=gate_failed |
| `s4x__base-uae-large-v1__qwen3-reranker-0.6b__q1.0` | skipped_base_unusable | base cache status=gate_failed |

None of these produced a `topic_definitions.json`, so there are no discovered topics to mark; each is a genuine
pipeline outcome, not a marking gap. Every `status=ok` cell resolves to a mark -- zero silent non-coverage.

## Open questions escalated to the user

1. **`Hello` (hand rule) -- one cell affected:** `mix__seg-qwen3-emb-0.6b__name-uae-large-v1__timeline-uae-large-v1`.
   Marked `hand = Sleep`, `strict = none`. The cluster is the whole sleep exchange (Sleep-labelled msgs 0-10,
   carrying all three sleep-subject markers the hand rule names -- 'I just took a nap', 'so you still feel tired
   huh?', 'I have more of a head ache') plus the closing greetings (msgs 59-65); its timeline windows 0 and 4 are
   the sleep-span windows. It is the *same message set* that `qwen3-emb-0.6b` names `Tired Huh`, which is marked
   Sleep -- so `No` would penalise a KeyBERT naming choice rather than the clustering. The competing reading is
   `none`, by analogy to the greeting-named `Good Day` precedent; the two clusters are distinguished by content
   (`Good Day` is greetings **only** -- msgs 0-3 plus the closing -- and carries no nap/tired/head-ache text,
   which is exactly why it earns no Sleep). If the user prefers the strict-greeting reading, change
   `by_topic_set['Hello|Sketch App|Paid Photoshoot'].hand.Hello` from `NNNY` to `NNNN`; nothing else moves.

## Judgment calls worth a second pair of eyes (not escalated -- precedent settles them)

- **`Utah` = Photoshoot under hand.** A place name, not a photoshoot name, so resolved from its cluster
  (msgs 33-51, majority Photoshoot at 12 of 19 rows), and 'utah' is uttered only inside the Photoshoot span.
  Skteching withheld because the precedent pair `Engagement Shoots` / `Pursue Photography` marks Photoshoot
  only on this same sketching+photoshoot span. `strict = none`.
- **`Paid Photoshoot` = Photoshoot under BOTH rules.** Strict `Yes` follows the precedent that gives strict
  `Yes` to `Engagement Shoots` and `Pursue Photography`; here literal synonymy is word-identical.
- **`Tired Huh` = Sleep under BOTH rules,** identical to the `Tired` precedent (the trailing interjection is
  KeyBERT bigram noise, not a change of subject).
- **`Buddhist` = Religious under BOTH rules,** identical to the `Maybe Buddhist` precedent.
- **`Sketch App` / `Personal Projects` marks unchanged in the mix cells even though their clusters are much
  larger there** (in several mix cells `Sketch App` swallows the entire Sleep span, and in
  `mix__seg-minilm-l6-v2__name-e5-large-v2__timeline-e5-large-v2` `Personal Projects` covers msgs 0-51).
  Precedent observation 1 above forbids letting a swollen span add a label; treating span coverage as the
  criterion would turn the grid into a content-coverage instrument rather than the name-equivalence instrument
  `stage6_window_score.is_equivalent` consumes.

## Appendix -- pre-P2 docstring, verbatim

```
Pass (b) of the FULL-STAGES ablation: Claude-hand-marked equivalence grids, Stage 6 re-run.
Two rules per encoder: 'hand' = colleague precedent (shipped grid: Sleep=Nap; Skteching AND Photoshoot = Practicing Sketching,
because the Photoshoot-labelled span itself says 'now all I am really doing is just practicing some sketching'), applied uniformly:
sleep-subject names (Nap/Tired/Head Ache) = Sleep; religion names = Religious; sketching names = Skteching + Photoshoot;
photoshoot names = Photoshoot; 'Personal Projects' (cluster = the sketching/photoshoot projects exchange) = both; 'Good Day' (greetings) = none.
'strict' = literal name synonymy only (Head Ache != Sleep; sketching names = Skteching only; Personal Projects = none).
```
