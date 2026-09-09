"""
Central configuration for the pipeline.

Edit the values below, then run:
    python run_pipeline.py

Stages 1-3 always use MODEL_NAME below -- run_pipeline.py never prompts for
it, so change it here directly if you want a different model for those
stages. It only ever prompts for the dataset path and for stage 4's model
(STAGE4_MODEL_NAME further down is just the default offered at that one
prompt).
"""

# --- Model -------------------------------------------------------------------
# Sentence-transformer model used by stages 1-3 (segmentation, clustering,
# and topic scoring for stage 3's "canonical" topic set). Not prompted for
# at runtime -- edit this value directly to change it.
MODEL_NAME = "all-MiniLM-L6-v2"

# --- Input ---------------------------------------------------------------
# Path to the raw dataset. Must have a "Text" column.
DATASET_PATH = "dataset.csv"

# --- Stage outputs -------------------------------------------------------
CLEANED_CSV = "dataset_fresh.csv"              # stage1_clean_dataset output
SEGMENTED_CSV = "conversation_with_index.csv"  # stage2_topic_shifting output
RESULTS_CSV = "topic_assignments.csv"          # stage3_topic_naming output (final)
TOPIC_DEFINITIONS_PATH = "topic_definitions.json"  # stage3_topic_naming output (internal,
                                                    # read by stage 4 -- name + threshold +
                                                    # defining text for each topic)

# --- Stage 3: clustering ---------------------------------------------------
# HDBSCAN (min_cluster_size, min_samples) search range: the pipeline tries
# every combination in these ranges and keeps whichever gets the best
# silhouette score, so the number of topics is chosen from the data instead
# of a fixed constant. Widen these if your dataset has more segments and
# you expect more topics; narrow them to speed up the search.
MIN_CLUSTER_SIZE_RANGE = range(2, 8)
MIN_SAMPLES_RANGE = range(1, 4)

# --- Stage 4: per-user topic timelines --------------------------------------
TIMELINE_CSV = "topic_timeline.csv"  # stage4_topic_timeline output (final)

# Stage 4 reuses stage 3's topics and thresholds (read from
# TOPIC_DEFINITIONS_PATH above) instead of discovering its own -- so
# topic_timeline.csv's Topic column always agrees with topic_assignments.csv.
# It can still use a different embedding model than stages 1-3: stage 4
# re-embeds each topic's defining text with whichever model this is, rather
# than reusing stage 3's raw vectors, so the model doesn't have to match.
# run_pipeline.py always asks for this separately at the point stage 4
# actually runs; this is just the default offered there.
STAGE4_MODEL_NAME = "all-MiniLM-L6-v2"

# Extra keyword arguments passed straight to SentenceTransformer(...) when
# loading stage 4's model. Leave as {} for a standard sentence-transformers
# model. Some third-party embedding models on Hugging Face (e.g. Jina AI's
# jina-embeddings-v2-*) require trust_remote_code=True to load -- if
# STAGE4_MODEL_NAME fails to load with a "requires you to execute custom
# code" error, set this to {"trust_remote_code": True}.
STAGE4_MODEL_KWARGS = {}

# Column names in the raw dataset that stage 4 needs beyond "Text" -- these
# pass through stages 1-2 untouched, so they must already be in your raw
# dataset under these names (change the values here if yours differ).
SENDER_COLUMN = "Sender"
TIMESTAMP_COLUMN = "Timestamp"
CHATROOM_COLUMN = "Chatroom"

# --- Stage 5: topic equivalence grid (evaluation) ---------------------------
# Runs automatically right after stage 4 produces a non-empty timeline.
# Column in the RAW dataset (DATASET_PATH, not CLEANED_CSV) holding the
# manual/reference topic label for each message -- ground truth for scoring,
# never touched by stages 1-4.
MANUAL_TOPIC_COLUMN = "Predicted Topic"
EQUIVALENCE_GRID_CSV = "topic_equivalence_grid.csv"  # stage5_topic_equivalence output --
                                                      # hand-edit "No"->"Yes" before stage 6

# --- Stage 6: window-first scoring (evaluation) -----------------------------
# NOT run automatically -- depends on you having hand-edited
# EQUIVALENCE_GRID_CSV above first. Run it yourself when ready:
#     python stage6_window_score.py
WINDOW_MATCHES_CSV = "window_matches.csv"          # stage6_window_score output (per-window detail)
WINDOW_SCORE_REPORT_TXT = "window_score_report.txt"  # stage6_window_score output (precision/recall/F1)
