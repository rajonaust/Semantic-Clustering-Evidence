"""
Pipeline stage 3: topic discovery and naming.

Pipeline:
    1. Load segments from conversation_with_index.csv (the output of
       stage2_topic_shifting.run_topic_shifting() -- see
       stage2_topic_shifting.py). This stage doesn't do any segmentation
       itself; it starts from that output and groups rows by the
       'Segment' column into one text blob per segment.
    2. Embed each segment's full text with a sentence-transformer.
    3. Cluster segment embeddings with HDBSCAN, searching over
       (min_cluster_size, min_samples) and keeping whichever configuration
       gets the best silhouette score, so the number of topics is chosen
       from the data instead of a fixed constant.
    4. Name each cluster with KeyBERT: run keyword extraction (with MMR for
       diversity) over each cluster's combined text and use its single
       best 1-2 word keyphrase as the topic name.
    5. Score every segment against every topic with a softmax over cosine
       similarity to each topic's mean embedding, giving each segment a
       named best-matching topic plus a probability.
    6. For each topic, derive a dynamic confidence threshold from its
       probability column with the Kneedle knee-of-the-curve method
       (Satopaa et al., 2011) -- see kneedle_threshold() for the full
       method description and its citation.

Output: topic_assignments.csv, one row per topic (Topic_Name,
Dynamic_Threshold).

Usage:
    from stage3_topic_naming import run_topic_naming
    run_topic_naming("conversation_with_index.csv")  # -> topic_assignments.csv

From a terminal:
    python stage3_topic_naming.py

Requires: pandas, numpy, scikit-learn, sentence-transformers, keybert
    pip install pandas numpy scikit-learn sentence-transformers keybert
"""
import json

import numpy as np
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_segments(input_csv="conversation_with_index.csv", text_column="Text", segment_column="Segment"):
    """Group an already-segmented conversation into one text blob per segment.

    Expects the CSV produced by stage2_topic_shifting.run_topic_shifting(),
    which has a `text_column` (message text) and a `segment_column`
    (integer segment id) for every row.

    Returns (segment_ids, segment_texts), both ordered by segment id.
    """
    df = pd.read_csv(input_csv)
    for col in (text_column, segment_column):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in {input_csv}. Available columns: {list(df.columns)}"
            )

    df[text_column] = df[text_column].fillna("").astype(str)
    grouped = (
        df.groupby(segment_column)[text_column]
          .apply(lambda texts: " ".join(texts))
          .sort_index()
    )
    return grouped.index.tolist(), grouped.tolist()


def cluster_segments(embeddings, min_cluster_size_range=range(2, 8), min_samples_range=range(1, 4)):
    """Search HDBSCAN's (min_cluster_size, min_samples) for the configuration
    with the best silhouette score, so the number of topics is picked from
    the data rather than hard-coded.

    Falls back to a single lenient configuration if nothing in the search
    range produces at least two clusters (e.g. very few segments).

    Raises RuntimeError up front if there are fewer than 2 segments -- HDBSCAN
    cannot cluster a single point, and the underlying sklearn error for that
    case ("n_samples=1 while HDBSCAN requires more than one sample") doesn't
    say what to actually do about it.
    """
    n_samples = len(embeddings)
    if n_samples < 2:
        raise RuntimeError(
            f"Not enough segments to cluster into topics (got {n_samples}, need at least 2). "
            "Check that stage2_topic_shifting produced more than one segment for this dataset."
        )

    best_score = -1.0
    best_labels = None

    for size in min_cluster_size_range:
        for samples in min_samples_range:
            if samples > size or samples > n_samples:
                # samples > n_samples: HDBSCAN itself rejects this (raises
                # ValueError) on small datasets -- skip rather than crash the
                # whole search over one invalid combination.
                continue
            labels = HDBSCAN(min_cluster_size=size, min_samples=samples, metric="cosine").fit_predict(embeddings)

            clustered_mask = labels != -1
            n_clusters = len(set(labels[clustered_mask]))
            if n_clusters < 2 or clustered_mask.sum() < 2:
                continue

            score = silhouette_score(embeddings[clustered_mask], labels[clustered_mask], metric="cosine")
            if score > best_score:
                best_score, best_labels = score, labels

    if best_labels is None:
        best_labels = HDBSCAN(min_cluster_size=2, min_samples=1, metric="cosine").fit_predict(embeddings)

    return best_labels


def _cluster_documents(segment_texts, labels):
    """Concatenate each cluster's segment texts into one document per
    cluster. Noise points (label -1) are dropped -- they don't form a
    topic of their own.

    This is the same "defining text" used both to name a topic with
    KeyBERT and, in topic_definitions.json, to let another stage
    (stage4_topic_timeline.py) re-embed the topic with its own model
    later without re-clustering.

    Returns {cluster_id: joined_text}.
    """
    cluster_ids = sorted(set(labels) - {-1})
    return {
        cid: " ".join(text for text, label in zip(segment_texts, labels) if label == cid)
        for cid in cluster_ids
    }


def name_topics_keybert(cluster_docs, kw_model, max_ngram=2, diversity=0.25):
    """Name each cluster with KeyBERT: extract keyphrases (1-2 words, MMR
    for diversity so the top pick isn't just the single most repeated
    word) from its defining document (see _cluster_documents()) and use
    the top-ranked keyphrase as the topic name.

    `cluster_docs`: {cluster_id: joined_text}, as returned by
    _cluster_documents().

    Returns {cluster_id: topic_name}.
    """
    if not cluster_docs:
        return {}

    topic_names = {}
    for cid, cluster_doc in cluster_docs.items():
        keywords = kw_model.extract_keywords(
            cluster_doc,
            keyphrase_ngram_range=(1, max_ngram),
            stop_words="english",
            use_mmr=True,
            diversity=diversity,
            top_n=1,
        )
        topic_names[cid] = keywords[0][0].title() if keywords else f"Topic {cid}"

    return topic_names


def score_segments_against_topics(segment_embeddings, topic_vectors):
    """Softmax over cosine similarity to each topic's mean embedding -- a
    dynamic (data-derived) probability of each segment belonging to each
    topic, rather than a fixed hard assignment.

    Returns (topic_ids, probabilities) where probabilities is an
    (n_segments, n_topics) array aligned with topic_ids.
    """
    topic_ids = list(topic_vectors.keys())
    topic_matrix = np.stack([topic_vectors[t] for t in topic_ids])

    sims = cosine_similarity(segment_embeddings, topic_matrix)

    temperature = 1.0 / len(topic_ids)
    shifted = sims - sims.max(axis=1, keepdims=True)
    weights = np.exp(shifted / temperature)
    probabilities = weights / weights.sum(axis=1, keepdims=True)

    return topic_ids, probabilities


def kneedle_threshold(scores, k_step=0.001):
    """Kneedle (knee-of-the-curve) threshold on a 1-D array of scores,
    following Satopaa, V., Albuquerque, D., Banerjee, A. & Gunopulos, D.
    (2011), "Finding a Kneedle in a Haystack: Detecting Knee Points in
    System Behavior," Proc. IEEE ICDCSW, pp. 166-171.

    The candidate cutoff is parameterized as c(k) = mu + k*sigma for
    k in [0, 1], where mu and sigma are the sample mean and standard
    deviation of `scores`. N(k), the count of scores at or above c(k), is
    monotonically non-increasing in k. Both axes are min-max normalized to
    [0, 1] and the knee is the k that maximizes the Kneedle difference
    curve D(k) = y_norm(k) - (1 - x_norm(k)) -- i.e. the point of greatest
    vertical departure from the line connecting the curve's two endpoints,
    which for a monotonic curve is equivalent to the point of maximum
    curvature. Because N(k) here is an exact, noise-free step function of
    k (not a sampled/noisy signal), the knee is taken as the single global
    maximizer of |D(k)|; the local-maxima-plus-sensitivity filtering in
    Satopaa et al. exists specifically to reject noise-induced spurious
    maxima, which cannot arise for a deterministic, monotonic curve.

    Degenerate cases are detected explicitly rather than inferred from the
    shape of D(k), and return mu (the only value mu + k*sigma can take
    when no cutoff in the sweep discriminates any scores):
      - sigma == 0: every score is identical, so c(k) == mu for every k.
      - N(k) constant across the whole sweep despite sigma > 0: no cutoff
        in [mu, mu + sigma] changes which scores clear it.

    Raises ValueError if `scores` is empty.
    Returns the threshold mu + k* * sigma, where k* is the selected knee.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("kneedle_threshold: `scores` must not be empty.")

    mu = scores.mean()
    sigma = scores.std()
    if sigma == 0:
        return float(mu)

    n_steps = int(round(1.0 / k_step))
    k_range = np.linspace(0.0, 1.0, n_steps + 1)  # exact endpoints, unlike float arange
    cutoffs = mu + k_range * sigma
    counts = (scores[None, :] >= cutoffs[:, None]).sum(axis=1)

    count_span = counts.max() - counts.min()
    if count_span == 0:
        return float(mu)

    x_norm = k_range  # already spans [0, 1] by construction
    y_norm = (counts - counts.min()) / count_span

    difference_curve = y_norm - (1 - x_norm)
    knee_idx = int(np.argmax(np.abs(difference_curve)))

    return float(mu + k_range[knee_idx] * sigma)


def compute_topic_thresholds(probabilities, topic_ids, topic_names):
    """Run kneedle_threshold() on each topic's probability column, giving a
    per-topic cutoff above which a segment counts as a confident member of
    that topic (rather than just its single best-matching topic).

    Returns {topic_id: threshold}.
    """
    return {
        tid: kneedle_threshold(probabilities[:, i])
        for i, tid in enumerate(topic_ids)
    }


def run_topic_naming(
    input_csv="conversation_with_index.csv",
    output_csv="topic_assignments.csv",
    definitions_path="topic_definitions.json",
    model=None,
    model_name=DEFAULT_MODEL_NAME,
    min_cluster_size_range=None,
    min_samples_range=None,
    verbose=True,
):
    """
    Runs the full stage-3 pipeline (load segments -> cluster -> name with
    KeyBERT -> score with softmax probabilities -> Kneedle thresholds) and
    writes `output_csv`.

    `model`: an already-loaded SentenceTransformer to reuse (e.g. shared
    across pipeline stages by run_pipeline.py). If omitted, one is loaded
    from `model_name`.

    `min_cluster_size_range` / `min_samples_range`: passed through to
    cluster_segments() if given (e.g. from config.py); otherwise its own
    defaults are used.

    `output_csv` gets exactly one row per topic: its name and its dynamic
    (Kneedle) probability threshold -- nothing per-segment. `definitions_path`
    gets one JSON record per topic (name, threshold, and the cluster's
    defining text) -- this is what stage4_topic_timeline.py reads to reuse
    these exact topics/thresholds instead of discovering its own.

    Returns the results DataFrame.
    """
    model = model or SentenceTransformer(model_name)
    kw_model = KeyBERT(model)

    _segment_ids, segment_texts = load_segments(input_csv)
    embeddings = np.array(model.encode(segment_texts))

    cluster_kwargs = {}
    if min_cluster_size_range is not None:
        cluster_kwargs["min_cluster_size_range"] = min_cluster_size_range
    if min_samples_range is not None:
        cluster_kwargs["min_samples_range"] = min_samples_range
    labels = cluster_segments(embeddings, **cluster_kwargs)
    topic_vectors = {
        cid: embeddings[labels == cid].mean(axis=0)
        for cid in sorted(set(labels) - {-1})
    }
    if not topic_vectors:
        raise RuntimeError(
            "No clusters were found -- try more segments, or widen the "
            "min_cluster_size_range/min_samples_range passed to cluster_segments()."
        )

    cluster_docs = _cluster_documents(segment_texts, labels)
    topic_names = name_topics_keybert(cluster_docs, kw_model)
    topic_ids, probabilities = score_segments_against_topics(embeddings, topic_vectors)
    thresholds = compute_topic_thresholds(probabilities, topic_ids, topic_names)

    results = pd.DataFrame({
        "Topic_Name": [topic_names[tid] for tid in topic_ids],
        "Dynamic_Threshold": [thresholds[tid] for tid in topic_ids],
    })
    results.to_csv(output_csv, index=False)

    definitions = [
        {
            "Topic_Name": topic_names[tid],
            "Dynamic_Threshold": thresholds[tid],
            "Definition_Text": cluster_docs[tid],
        }
        for tid in topic_ids
    ]
    with open(definitions_path, "w") as f:
        json.dump(definitions, f, indent=2)

    if verbose:
        print(f"Found {len(topic_vectors)} topic(s):")
        for _, row in results.iterrows():
            print(f"  {row['Topic_Name']:<20} {row['Dynamic_Threshold']:.4f}")
        print(f"Wrote {len(results)} topic(s) -> {output_csv}")
        print(f"Wrote topic definitions (for stage 4) -> {definitions_path}")

    return results


if __name__ == "__main__":
    run_topic_naming()
