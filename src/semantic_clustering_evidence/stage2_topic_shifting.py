"""
Pipeline stage 2: segmentation.

Detects topic-shift boundaries in a conversation and writes a fully
segment-labeled version of it (every row, no subsampling) to CSV.

Meant to run on the OUTPUT of stage1_clean_dataset.py (a CSV whose 'Text'
column is guaranteed to be plain strings), so it takes the input CSV path
as an argument instead of hard-coding "conversation.csv".

Usage:
    from stage2_topic_shifting import run_topic_shifting
    run_topic_shifting("dataset_fresh.csv")                     # -> conversation_with_index.csv
    run_topic_shifting("dataset_fresh.csv", "my_segments.csv")  # custom output name

From a terminal:
    python stage2_topic_shifting.py dataset_fresh.csv
    python stage2_topic_shifting.py dataset_fresh.csv conversation_with_index.csv
"""
import sys

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Hugging Face model ids are case-sensitive -- the real model is
# "all-MiniLM-L6-v2" (not "all-miniLM-L6-V2").
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_model_cache = {}


def _get_model(model_name=DEFAULT_MODEL_NAME):
    """Load a SentenceTransformer model by name, caching it per name so
    repeated calls (or other stages asking for the same name) don't reload
    it from disk."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def cosine_distance(A, B):
    return 1 - cosine_similarity(A, B)[0][0]


def _find_last_point(model, chat_messages, start, middle, finish):
    distance = []
    index = []
    for i in range(middle + 1, finish + 1):
        text1 = " ".join(chat_messages[start:i])
        text2 = " ".join(chat_messages[i:i + 1])
        text3 = " ".join(chat_messages[i:finish + 1])
        vec1 = model.encode(text1)
        vec2 = model.encode(text2)
        vec3 = model.encode(text3)
        dist1 = cosine_distance(vec1.reshape(1, -1), vec2.reshape(1, -1))
        dist2 = cosine_distance(vec1.reshape(1, -1), vec3.reshape(1, -1))
        distance.append([dist1, dist2])
        index.append(i)

    ans = -1
    for i in range(1, len(distance) - 1):
        if (distance[i][0] > distance[i - 1][0] and distance[i][1] > distance[i + 1][1]
                and distance[i][0] > distance[i + 1][0] and distance[i][1] > distance[i - 1][1]):
            ans = index[i]
    return ans


def _find_topic_shift_points(model, chat_messages):
    """Same sliding-window / reverse-scan logic as the original script,
    just moved into a function so it can be called with any message list."""
    vectors = []
    text = ""
    prev_dist = 0
    prev_rate = 10000
    cur_length = 0
    cur_rate = 10000
    answer = []
    i = 0
    while i < len(chat_messages):
        vectors.append(model.encode(chat_messages[i]))
        cur_length += 1
        text = text + "\n" + chat_messages[i]
        cur_vec = model.encode(text)

        dist = 0
        for j in range(len(vectors)):
            dist += cosine_distance(cur_vec.reshape(1, -1), vectors[j].reshape(1, -1))
        dist = dist / len(vectors)

        if cur_length > 1:
            cur_rate = (dist - prev_dist) / dist
            if cur_rate > prev_rate:
                # Reverse scan to pin down the exact shift point.
                k = i - 1
                reverse_vectors = []
                reverse_text = ""
                reverse_prev_dist = 0
                reverse_prev_rate = 10000
                reverse_cur_length = 0
                reverse_cur_point = -1
                while k >= i - (cur_length - 1):
                    reverse_cur_length += 1
                    reverse_vectors.append(model.encode(chat_messages[k]))
                    reverse_text = reverse_text + "\n" + chat_messages[k]
                    reverse_cur_vec = model.encode(reverse_text)
                    reverse_dist = 0
                    for l in range(len(reverse_vectors)):
                        reverse_dist += cosine_distance(
                            reverse_cur_vec.reshape(1, -1), reverse_vectors[l].reshape(1, -1)
                        )
                    reverse_dist = reverse_dist / len(reverse_vectors)
                    if reverse_cur_length > 1:
                        reverse_cur_rate = (reverse_dist - reverse_prev_dist) / reverse_dist
                        if reverse_cur_rate > reverse_prev_rate:
                            reverse_cur_point = k
                        reverse_prev_rate = reverse_cur_rate
                    reverse_prev_dist = reverse_dist
                    k -= 1

                easy = -1
                if reverse_cur_point != -1:
                    easy = _find_last_point(model, chat_messages, i - (cur_length - 1), reverse_cur_point, i)
                if easy != -1:
                    k = easy
                    i = k
                    answer.append(k)
                    vectors.clear()
                    text = ""
                    prev_dist = 0
                    prev_rate = 10000
                    cur_length = 0
                    cur_rate = 10000
                    dist = 10000
            prev_rate = cur_rate
        prev_dist = dist
        i += 1

    answer.append(len(chat_messages) - 1)
    return answer


def run_topic_shifting(
    input_csv,
    output_csv="conversation_with_index.csv",
    text_column="Text",
    verbose=True,
    model=None,
    model_name=DEFAULT_MODEL_NAME,
):
    """
    Reads `input_csv` (expected to already be cleaned, e.g. by
    stage1_clean_dataset.clean_dataset), detects topic-shift boundaries
    across every row of `text_column` (no subsampling), labels all of them
    with a 'Segment' column, and writes the result to `output_csv`.

    `model`: an already-loaded SentenceTransformer to reuse (e.g. shared
    across pipeline stages by run_pipeline.py). If omitted, one is loaded
    (and cached) from `model_name`.

    Returns the path to `output_csv`.
    """
    model = model or _get_model(model_name)

    df_full = pd.read_csv(input_csv)
    if text_column not in df_full.columns:
        raise KeyError(
            f"Column '{text_column}' not found in {input_csv}. "
            f"Available columns: {list(df_full.columns)}"
        )

    chat_messages = df_full[text_column].fillna("").astype(str).tolist()
    chat_messages_db = df_full.reset_index(drop=True)

    if verbose:
        print(f"Running topic shifting on all {len(chat_messages)} messages from {input_csv}...")

    answer = _find_topic_shift_points(model, chat_messages)

    test_answer = []
    point = 0
    cur_segment = 0
    for i in range(len(chat_messages_db)):
        test_answer.append(cur_segment)
        if answer[point] == i:
            cur_segment += 1
            point += 1

    chat_messages_db["Segment"] = test_answer
    chat_messages_db.to_csv(output_csv, index=False)

    if verbose:
        # `answer` always ends with a sentinel (the final row's index), which
        # triggers one last increment to cur_segment that no row ever actually
        # receives -- so cur_segment + 1 overcounts by one. Read the true
        # count back off the assigned column instead of trusting the counter.
        n_segments = chat_messages_db["Segment"].nunique()
        print(f"Wrote {len(chat_messages_db)} rows ({n_segments} segment(s)) -> {output_csv}")

    return output_csv


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stage2_topic_shifting.py <input_csv> [output_csv]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "conversation_with_index.csv"
    run_topic_shifting(input_path, output_path)
