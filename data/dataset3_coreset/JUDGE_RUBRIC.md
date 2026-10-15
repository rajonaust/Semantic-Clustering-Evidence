# Dataset-3 Coreset Label-Audit — Judge Rubric

You are given ONLY this file and one packet file (`packets/packet_NN.md`). Do not use any other
context, file, or prior knowledge of this project. Judge every message in the packet independently
of any other packet.

## What you are doing

Each packet holds several whole chatrooms (a header line per chatroom, then one line per message,
in original time order). Each chatroom is a real two-(or more-)party chat, or occasionally a
one-sided monologue. Your job: assign every message a **topic label** drawn from the fixed
vocabulary V below, at **segment** granularity (see Segment principle), plus a confidence and a
running segment id.

**The Prompt shown in the chatroom header is context, not a label.** Chatrooms were seeded with a
discussion prompt (e.g. "What's the last film that truly made an impression on you?"), but
real conversations drift — greet, joke, answer the prompt, then wander elsewhere. Label the
**actual subject of the messages**, never the prompt itself. A chatroom seeded with the film
prompt that actually spends half its length discussing pizza should have a Food segment there,
not a copy-the-prompt Movie label.

## Vocabulary V (23 labels)

Glosses are inferred from the TopicRules keyword table (for the first 10) and from how the
labels were actually used in the one small hand/Gemini-labelled slice of this dataset (for the
next 13). `Skteching` (a misspelling in the raw data) is not a separate label — always write
`Sketching`.

| Label | Use when the segment is about... |
|---|---|
| General | small talk, logistics, greetings/sign-offs with no other content, or **any** real subject that is not in this table (pair with `suggested_topic`, see below) |
| Money | money, spending, being rich, buying things, prices, finances |
| Food | food, cooking, restaurants, pizza, eating |
| Technology | computers, tech, apps, software, the internet, computing/AI as a technology |
| Music | music, songs, bands, instruments, playlists (not classical-composer trivia — see Mozart) |
| Movie | movies, films, actors, directors, cinema |
| Sketching | the activity/technique of sketching or drawing |
| Illustration | art, illustration, drawing or design **as a subject genuinely being discussed** (e.g. an iPad Pro for drawing, an illustration program) — most rule-generated "Illustration" rows in the raw data are a keyword-substring artifact (`art` inside "start/smart/heart/part"), so do not let the raw label bias you; judge the actual text |
| Sleep | sleep, napping, being tired, insomnia |
| Travel | travel, vacations, trips |
| Mozart | discussion specifically of Mozart or classical composers (e.g. "Mozart or Beethoven?") |
| Fossil | fossils / paleontology |
| Hero | heroism, heroes, role models |
| Craziest Thing | recounting a wild/crazy personal story or anecdote |
| Ambitious | talk of ambition, big goals, drive, striving |
| Country | a country/nation, homeland, or its governance/politics |
| Greetings | a segment whose **entire** content is a greeting/farewell ritual with no other subject and no substantive chit-chat beyond hello/goodbye — this is a narrow exception; most hello/goodbye stretches are better labelled General (see worked example 2). If genuinely unsure, prefer General and mark confidence=low |
| Relationship | romantic relationships |
| Friendship | friendship, platonic relationships |
| Education | school, studying, classes, learning as a topic |
| Personal | personal reflection, autobiographical sharing, reminiscing about one's own life/family |
| Religious | religion, faith, spirituality |
| Sports | sports, athletics, physical games |

## Segment principle

A **segment** is a contiguous run of messages in one chatroom that shares one real subject.
- Greetings, fillers ("haha", "yeah", "oh nice!"), and short logistics **inside** a topical run
  inherit that run's label — do not break the segment or fall back to General just because one
  message in the middle is a filler.
- When the subject genuinely changes, start a new segment (increment `segment_id`) and re-label.
- A stretch with **no** identifiable V topic — pure small talk, pure logistics, a greeting/farewell
  with nothing else — is `General`.
- When the real subject is outside V entirely (e.g. pets, superpowers, unicorns, a flying car),
  still write `General` for `label`, but you **must** fill `suggested_topic` with a short free-text
  guess (1-3 words) for what the real subject is. Do not invent a new V label.
- `segment_id` is a running integer **per chatroom**, starting at 1, incremented at every subject
  change (whether the new segment is a V label or another General/suggested_topic stretch).

## Confidence

`confidence` ∈ {high, medium, low} — your own certainty that the label is correct for this
message. Use `low` whenever a message could plausibly belong to the segment before or after it,
or when General vs. a V label is a close call.

## Worked examples (from the actual coreset data)

**1. A topical run with fillers inside it** (Chatroom-1004, an iPad-Pro/drawing discussion):
```
149 | User-84 | Awwwww that's so sweet of you!
150 | User-84 | Way to go! I seriously only want the iPad pro... like so bad!
151 | User-84 | I'm rooting for you!
152 | User-2  | That would be awesome for illustration
153 | User-2  | my girlfriend has one for industrial design and loves it
154 | User-84 | Yeah that's exactly what I want if for! My dad has one, and I've drawn on it a couple times, and it's literally the best
```
All six messages get `label=Illustration`, same `segment_id`. 149-151 are affirmations/fillers
that sit inside the drawing/iPad discussion — they inherit Illustration, they are not General.

**2. A General small-talk stretch, followed by a real topic** (Chatroom-1):
```
0 | User-1 | Hello
1 | User-1 | How are you doing today?
2 | User-2 | whats up MD
3 | User-2 | im doing good
4 | User-2 | how are you doing?
5 | User-1 | Im alright, I just took a nap. But it was one of those naps that doesnt help anything...
```
Rows 0-4 are pure greeting/logistics with no other content: `label=General`, `segment_id=1`
(not `Greetings` — there is no other subject here besides hello/how-are-you, but this is the
common case the rubric defaults to General; reserve `Greetings` only for a segment that reads
as a deliberate, self-contained goodbye/greeting ritual, not an ordinary hello). Row 5 starts a
nap/tiredness discussion: `label=Sleep`, new `segment_id=2`.

**3. An off-V subject → General + suggested_topic** (Chatroom-1674, prompted "How would the world
be different if pets could talk?"):
```
24365 | User-54  | Hey, I'm good, how are you?
24366 | User-54  | If pets could talk I would hope dogs would still be as nice and loving.
24367 | User-585 | Haha yeah that would be the worst if they could talk and we found at dogs were just jerks
24368 | User-585 | I'm doing awesome
24369 | User-585 | If animals could talk would different species have different languages?
```
24365 is a plain greeting (`label=General`, `segment_id=1`, no `suggested_topic` needed — it truly
has no topic). 24366-24369 are all about talking pets/animals, a real subject with no V label:
`label=General`, new `segment_id=2`, `suggested_topic=pets` (or `talking animals`) on every row
in that run.

## Output contract

Write `judgements/packet_NN.<judge>.csv` (same `NN` as your packet, `<judge>` = your own short
identifier) with this exact header and nothing else:

```
msg_id,label,segment_id,confidence,suggested_topic
```

- Exactly one row per `msg_id` that appears in your packet — cross-check against
  `packets/packet_NN.expected_ids.txt` before you finish; every id in that file must appear
  exactly once in your CSV, and no extra ids.
- `label` must be spelled exactly as in the V table above (23 values only, "Sketching" never
  "Skteching").
- `segment_id` is an integer, restarting at 1 for each new chatroom in the packet.
- `confidence` is exactly one of `high`, `medium`, `low`.
- `suggested_topic` is empty for every row except where `label=General` **and** the real subject
  is a genuine, nameable topic outside V (leave it empty for ordinary small talk/logistics that
  has no real subject at all).
- No other columns, no header changes, no extra rows, no commentary outside the CSV file.
