"""
Pipeline stage 4: per-user topic timelines.

For each user (Sender), walks their messages in the order of stage 2's
conversation segments and finds contiguous stretches where the user is
talking about a given topic with above-threshold probability. Each stretch
becomes one output row: who, when it started/ended, how long it lasted,
which topic, how confident, and the chat text itself.

This replaces the SCOPE_with_Cosine notebook's approach in three ways:

  1. No fixed list of 21 topics (News, Research, Technology, ... Fantasy).
     Instead, stage 4 reuses stage 3's own topics: it reads
     topic_definitions.json (written by stage3_topic_naming.run_topic_naming(),
     alongside topic_assignments.csv) to get each topic's name, dynamic
     threshold, and defining text, then re-embeds that defining text with
     whichever model stage 4 is using. Stage 4 does NOT re-cluster or
     re-run KeyBERT -- the topic set and thresholds are exactly stage 3's,
     so topic_assignments.csv and topic_timeline.csv always agree on topic
     names. (Re-embedding rather than reusing stage 3's raw vectors is
     what still lets stage 4 use a different sentence-transformer model
     than stages 1-3 -- see run_topic_timeline()'s docstring.)
  2. No fixed `standard_probability = 0.05` cutoff. Each topic's dynamic
     (Kneedle) threshold, as computed by stage 3, is read straight from
     topic_definitions.json.
  3. No fixed 1-hour-of-day window. The windowing unit is stage 2's
     'Segment' id instead of a calendar hour. A user's activity is walked
     in segment order; segments with no message from that user are simply
     absent, the same way the original algorithm skipped empty hours.
     Eligibility and the reported Probability are both scored on each
     segment's FULL text -- every participant's messages in it, not just
     this user's own lines -- since that's the actual text stage 2's
     segmentation and stage 3's clustering were built on, and it gives a
     short message (e.g. a one-word reply) the surrounding context it
     needs to be judged accurately (see _full_segment_texts()). Chat
     Summary still reports only this user's own lines -- what's scored
     and what's printed are deliberately different.

Also dropped from the original notebook: the spellcheck/lemmatize/
stopword word-cleaning pipeline, and the fixed-threshold parameter sweep
("All In a Loop" cell, which ran the whole thing once per candidate
`standard_probability` from 0.01-0.10). Both existed specifically to
support the old word-vs-fixed-topic-label matching approach; scoring text
against topic embeddings directly (the same method stage 3 uses) works on
raw segment text and needs neither -- and a sweep over a threshold that no
longer exists as a free parameter has nothing left to sweep. Say if you
want either brought back for a different purpose.

Expects stage2_topic_shifting.py's output (conversation_with_index.csv),
which must additionally carry the original 'Sender', 'Timestamp', and
'Chatroom' columns from the raw dataset (stage 1/2 only ever touch
'Text', so these pass through untouched), plus stage3_topic_naming.py's
topic_definitions.json (run stage 3 at least once first).

Caveat worth knowing for write-ups: each topic's Dynamic_Threshold was
calibrated in stage 3's own model's similarity/probability distribution.
Reusing it as-is is exact when stage 4 uses the same model as stage 3.
When stage 4 is deliberately given a different model, the threshold still
applies (a probability is still a probability), but it is no longer
guaranteed to sit at that model's own knee point -- it is stage 3's knee,
carried over. If that matters for your methodology, recomputing the
threshold per model is a follow-up worth flagging rather than something
this version does silently.

Output: topic_timeline.csv, one row per (user, topic, contiguous run of
segments): User, Start Date, Start Time, End Date, End Time, Time
Duration, Topic, Probability, Chat Summary -- the same 9 columns the
original notebook produced -- plus one added column, Full Segment
Context: every message from every participant across the run's segments
(not just this user's own lines), in chronological order. That's the
actual text the run was scored against (see build_topic_timeline()), so
it's there to make the Topic/Probability assignment easy to sanity-check
by eye -- it isn't part of the original notebook's schema.

Usage:
    from stage4_topic_timeline import run_topic_timeline
    run_topic_timeline("conversation_with_index.csv")  # -> topic_timeline.csv

From a terminal:
    python stage4_topic_timeline.py

Requires: pandas, numpy, scikit-learn, sentence-transformers, keybert
    pip install pandas numpy scikit-learn sentence-transformers keybert
"""
import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .stage3_topic_naming import DEFAULT_MODEL_NAME


def load_conversation(
    input_csv="conversation_with_index.csv",
    text_column="Text",
    segment_column="Segment",
    sender_column="Sender",
    timestamp_column="Timestamp",
):
    """Load stage 2's output and check it carries everything stage 4 needs
    beyond what stage 2 itself produces (Sender, Timestamp -- passed
    through from the raw dataset).

    Returns the DataFrame sorted by Segment id (chronological order).
    """
    df = pd.read_csv(input_csv)
    required = (text_column, segment_column, sender_column, timestamp_column)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing column(s) {missing} in {input_csv}. Available columns: {list(df.columns)}"
        )

    df[text_column] = df[text_column].fillna("").astype(str)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    return df.sort_values(segment_column).reset_index(drop=True)


def load_stage3_topics(definitions_path, model):
    """Load stage 3's topics from topic_definitions.json (written by
    stage3_topic_naming.run_topic_naming()) instead of discovering topics
    independently -- so stage 4's Topic/Probability columns always agree
    with stage 3's topic_assignments.csv.

    Each topic's name and Dynamic_Threshold are used exactly as stage 3
    computed them. Its defining text is re-embedded with `model` --
    whichever model stage 4 is using -- rather than reusing stage 3's raw
    vectors, so this still works when stage 4 is given a different
    sentence-transformer model than stages 1-3 used.

    Returns (topic_ids, topic_vectors, topic_names, thresholds), in the
    same shapes the rest of this module (build_topic_timeline(), etc.)
    already expects.
    """
    if not os.path.exists(definitions_path):
        raise FileNotFoundError(
            f"'{definitions_path}' not found -- run stage3_topic_naming.run_topic_naming() "
            "at least once first (it writes this file alongside topic_assignments.csv)."
        )

    with open(definitions_path) as f:
        definitions = json.load(f)

    if not definitions:
        raise RuntimeError(
            f"'{definitions_path}' contains no topics -- re-run stage 3 on a dataset "
            "with enough segments to form at least one topic."
        )

    topic_ids = list(range(len(definitions)))
    definition_texts = [d["Definition_Text"] for d in definitions]
    embeddings = np.array(model.encode(definition_texts))

    topic_vectors = {tid: embeddings[tid] for tid in topic_ids}
    topic_names = {tid: d["Topic_Name"] for tid, d in zip(topic_ids, definitions)}
    thresholds = {tid: float(d["Dynamic_Threshold"]) for tid, d in zip(topic_ids, definitions)}

    return topic_ids, topic_vectors, topic_names, thresholds


def _topic_probabilities(text, model, topic_ids, topic_vectors):
    """Same softmax-over-cosine-similarity scoring as
    stage3_topic_naming.score_segments_against_topics(), evaluated for one
    arbitrary piece of text (e.g. several segments' text joined together)
    against every topic's mean embedding.

    Returns {topic_id: probability}.
    """
    topic_matrix = np.stack([topic_vectors[t] for t in topic_ids])
    vec = model.encode(text)
    sims = cosine_similarity(vec.reshape(1, -1), topic_matrix)[0]

    temperature = 1.0 / len(topic_ids)
    shifted = sims - sims.max()
    weights = np.exp(shifted / temperature)
    probabilities = weights / weights.sum()

    return dict(zip(topic_ids, probabilities))


def _user_active_segments(df, user, segment_column, sender_column):
    """The Segment ids where this user said something -- used only to
    decide which segments the user was *present* for (their presence
    gates whether a run can start/continue there at all). It no longer
    supplies the text that gets scored -- see _full_segment_texts() for
    that -- so a segment with no message from this user is simply absent,
    same as the original notebook skipping empty hours.

    Returns a sorted list of segment ids.
    """
    return sorted(df.loc[df[sender_column] == user, segment_column].unique().tolist())


def _full_segment_texts(df, text_column, segment_column):
    """Every row's text (all users, not just one) grouped by stage 2's
    Segment id -- the actual unit scored against each topic, so a user's
    one-line message is judged in the context of the whole exchange it
    was part of (the same full-segment text stage 2's segmentation and
    stage 3's clustering/naming were built on), rather than in isolation.

    Returns {segment_id: joined text}.
    """
    grouped = (
        df.groupby(segment_column)[text_column]
          .apply(lambda texts: " ".join(texts))
          .sort_index()
    )
    return grouped.to_dict()


def extract_user_topic_runs(
    df, user, model, topic_ids, topic_vectors, thresholds, full_segment_texts,
    text_column="Text", segment_column="Segment", sender_column="Sender",
):
    """Port of the original notebook's `my_algorithm_for_user`, with the
    1-hour window replaced by stage 2's Segment id and the fixed
    `standard_probability` replaced by each topic's dynamic threshold.

    For every topic, walks this user's active segments (segments they
    said something in) in order. A segment is eligible to start or extend
    a run if the *full segment's* probability for that topic (all users'
    text in that segment, not just this user's own lines -- see
    _full_segment_texts()) already clears the threshold; an eligible
    segment extends the current run only if the run's *combined* full-segment
    probability still clears the threshold once it's added, otherwise the
    current run is closed and a new one starts there. A segment that is
    active but not eligible closes any run in progress without starting
    one of its own -- it's a break, not a candidate.

    `full_segment_texts`: {segment_id: joined text of every user's
    messages in that segment}, from _full_segment_texts() -- computed
    once and shared across all users, since it doesn't depend on which
    user is being walked.

    Returns {topic_id: [[segment_id, ...], ...]} -- one list of runs per
    topic, each run a list of segment ids merged together.
    """
    active_segments = _user_active_segments(df, user, segment_column, sender_column)

    cache = {}

    def probability_of(run, topic_id):
        key = (tuple(run), topic_id)
        if key not in cache:
            joined = " ".join(full_segment_texts[s] for s in run)
            cache[key] = _topic_probabilities(joined, model, topic_ids, topic_vectors)
        return cache[key][topic_id]

    runs_by_topic = {}
    for topic_id in topic_ids:
        threshold = thresholds[topic_id]
        eligible = {s for s in active_segments if probability_of([s], topic_id) >= threshold}

        runs = []
        current_run = []
        for seg in active_segments:
            if seg in eligible:
                if not current_run:
                    current_run = [seg]
                else:
                    candidate = current_run + [seg]
                    if probability_of(candidate, topic_id) >= threshold:
                        current_run = candidate
                    else:
                        runs.append(current_run)
                        current_run = [seg]
            elif current_run:
                runs.append(current_run)
                current_run = []
        if current_run:
            runs.append(current_run)

        runs_by_topic[topic_id] = runs

    return runs_by_topic


def build_topic_timeline(
    df, model, topic_ids, topic_vectors, topic_names, thresholds,
    text_column="Text", segment_column="Segment", sender_column="Sender",
    timestamp_column="Timestamp", chatroom_column="Chatroom",
):
    """Runs extract_user_topic_runs() for every user and turns the result
    into one report row per (user, topic, run): when it happened, how
    long it lasted, how confident, and the chat text itself.

    Scoring (both which segments are eligible and the reported
    Probability) is done on each run's full segment text -- every user's
    messages in those segments, from _full_segment_texts() -- for
    consistency: the same text that decided a run also produces the
    probability reported for it. Chat Summary still prints only this
    user's own lines.

    Returns the results DataFrame.
    """
    users = df[sender_column].dropna().unique().tolist()
    full_segment_texts = _full_segment_texts(df, text_column, segment_column)
    rows = []

    for user in users:
        runs_by_topic = extract_user_topic_runs(
            df, user, model, topic_ids, topic_vectors, thresholds, full_segment_texts,
            text_column=text_column, segment_column=segment_column, sender_column=sender_column,
        )
        for topic_id, runs in runs_by_topic.items():
            for run in runs:
                run_rows = (
                    df[(df[sender_column] == user) & (df[segment_column].isin(run))]
                    .sort_values(timestamp_column)
                )
                if run_rows.empty:
                    continue

                start_ts = run_rows[timestamp_column].min()
                end_ts = run_rows[timestamp_column].max()
                joined_text = " ".join(full_segment_texts[s] for s in run)
                probability = _topic_probabilities(joined_text, model, topic_ids, topic_vectors)[topic_id]

                chat_summary = "\n".join(
                    f"{user} :: {r[chatroom_column]} [{r[timestamp_column]}]: {r[text_column]}"
                    for _, r in run_rows.iterrows()
                )

                # Every message from every user across the run's segments (not
                # just this user's own lines) -- the same text that was actually
                # scored to produce Probability, laid out message-by-message so
                # it's easy to read what the run's classification was based on.
                context_rows = (
                    df[df[segment_column].isin(run)]
                    .sort_values(timestamp_column)
                )
                full_segment_context = "\n".join(
                    f"{r[sender_column]} :: {r[chatroom_column]} [{r[timestamp_column]}]: {r[text_column]}"
                    for _, r in context_rows.iterrows()
                )

                rows.append({
                    "User": user,
                    "Start Date": start_ts.strftime("%Y-%m-%d"),
                    "Start Time": start_ts.strftime("%H:%M:%S"),
                    "End Date": end_ts.strftime("%Y-%m-%d"),
                    "End Time": end_ts.strftime("%H:%M:%S"),
                    "Time Duration": str(end_ts - start_ts),
                    "Topic": topic_names[topic_id],
                    "Probability": str(probability),
                    "Chat Summary": chat_summary,
                    "Full Segment Context": full_segment_context,
                })

    return pd.DataFrame(rows)


def run_topic_timeline(
    input_csv="conversation_with_index.csv",
    output_csv="topic_timeline.csv",
    definitions_path="topic_definitions.json",
    model=None,
    model_name=DEFAULT_MODEL_NAME,
    text_column="Text",
    segment_column="Segment",
    sender_column="Sender",
    timestamp_column="Timestamp",
    chatroom_column="Chatroom",
    verbose=True,
):
    """
    Runs the full stage-4 pipeline (load -> load stage 3's topics ->
    extract each user's per-topic segment runs -> build the report) and
    writes `output_csv`.

    `model`: an already-loaded SentenceTransformer to reuse (e.g. shared
    across pipeline stages by run_pipeline.py). If omitted, one is loaded
    from `model_name`. This can be a different model than stages 1-3 used
    -- see load_stage3_topics() for how that stays valid.

    `definitions_path`: stage 3's topic_definitions.json (written by
    stage3_topic_naming.run_topic_naming()). Stage 4 reuses those topics
    and thresholds rather than discovering its own -- see this module's
    docstring.

    Returns the results DataFrame.
    """
    model = model or SentenceTransformer(model_name)

    df = load_conversation(input_csv, text_column, segment_column, sender_column, timestamp_column)
    if chatroom_column not in df.columns:
        raise KeyError(
            f"Column '{chatroom_column}' not found in {input_csv}. Available columns: {list(df.columns)}"
        )

    topic_ids, topic_vectors, topic_names, thresholds = load_stage3_topics(definitions_path, model)

    results = build_topic_timeline(
        df, model, topic_ids, topic_vectors, topic_names, thresholds,
        text_column=text_column, segment_column=segment_column, sender_column=sender_column,
        timestamp_column=timestamp_column, chatroom_column=chatroom_column,
    )
    results.to_csv(output_csv, index=False)

    if verbose:
        print(f"Reusing {len(topic_ids)} topic(s) from stage 3: {', '.join(topic_names.values())}")
        print("Dynamic thresholds (from stage 3):")
        for tid in topic_ids:
            print(f"  {topic_names[tid]:<20} {thresholds[tid]:.4f}")
        n_users = results["User"].nunique() if not results.empty else 0
        print(f"Extracted {len(results)} topic segment(s) across {n_users} user(s) -> {output_csv}")

    return results


if __name__ == "__main__":
    run_topic_timeline()
