"""
Orchestrates the full pipeline, each stage feeding its output into the next.

Stages:
    1. stage1_clean_dataset     -> dataset_fresh.csv
    2. stage2_topic_shifting    -> conversation_with_index.csv
    3. stage3_topic_naming      -> topic_assignments.csv (dynamic topics +
                                    dynamic Kneedle threshold, one row per topic)
    4. stage4_topic_timeline    -> topic_timeline.csv (per-user, per-topic
                                    contiguous activity runs)
    5. stage5_topic_equivalence -> topic_equivalence_grid.csv (evaluation).
                                    Runs AUTOMATICALLY right after stage 4
                                    produces a non-empty timeline -- see
                                    run_stage4_only() below.
    6. stage6_window_score      -> window_matches.csv + window_score_report.txt
                                    (evaluation). Never starts on its own --
                                    it scores against stage 5's grid, which
                                    you're meant to hand-edit ("No" -> "Yes")
                                    first. Right after stage 5 writes the
                                    grid, run_stage4_only() pauses this same
                                    terminal session with a prompt; edit the
                                    grid file, then answer that prompt to
                                    run stage 6 without leaving or re-running
                                    the script. Answering "no" (or running
                                    stage6_window_score.py separately, later)
                                    both remain available -- see
                                    run_stage4_only()'s docstring.

Stages 1-3 always use config.MODEL_NAME -- no prompt for it, since cleaning
and segmentation aren't meant to be a per-run choice, and stage 3's topic
discovery is meant to be the dataset's one "canonical" topic set. Stage 4
reuses that exact topic set and its dynamic thresholds (from
topic_definitions.json) rather than discovering its own, so
topic_timeline.csv's Topic column always matches topic_assignments.csv.
Only stage 4 is ever asked interactively for a model: it re-embeds each
topic's defining text with whichever model it's given, so it can still use
a different embedding model than stages 1-3 without the two stages'
vectors being mixed. Change config.MODEL_NAME directly if you want a
different model for stages 1-3.

If conversation_with_index.csv (stage 2's output) already exists on disk,
running this script skips stages 1-3 entirely and only asks for stage 4's
model -- so re-running stage 4 with a different model is fast and doesn't
redo work stages 1-3 already did. Delete that file (or point
config.SEGMENTED_CSV elsewhere) to force a full run again. (topic_definitions.json
must also already exist from a prior stage 3 run -- run_stage4_only()
raises a clear error if it's missing.)

All the tunable values (model names, dataset path, output filenames,
clustering search ranges, stage-4 column names) live in config.py.

Stage 4 needs 'Sender', 'Timestamp', and 'Chatroom' columns in the raw
dataset (passed through stages 1-2 untouched); if your dataset doesn't have
them, stage 4 is skipped with a clear message and stages 1-3's outputs are
still kept.

Stage 5 needs config.MANUAL_TOPIC_COLUMN in the RAW dataset (a manual/
reference topic label per message, for evaluation only -- never used by
stages 1-4). If it's missing, stage 5 is skipped with a clear message and
stage 4's output is still kept; stage 5 is not needed at all unless you're
scoring the pipeline against manual labels.
"""
import os

from sentence_transformers import SentenceTransformer

from . import config
from .stage1_clean_dataset import clean_dataset
from .stage2_topic_shifting import run_topic_shifting
from .stage3_topic_naming import run_topic_naming
from .stage4_topic_timeline import run_topic_timeline
from .stage5_topic_equivalence import build_topic_equivalence_table
from .stage6_window_score import run_window_score


def get_dataset_input():
    """Ask for the raw dataset path, at the start of a full run. Press
    Enter to accept the default from config.py. Stages 1-3's model is
    never prompted for -- it's always config.MODEL_NAME."""
    return input(f"Raw dataset CSV path [{config.DATASET_PATH}]: ").strip() or config.DATASET_PATH


def get_stage4_model_input():
    """Ask for stage 4's own model name -- independent of whatever model
    stages 1-3 used, and asked every time stage 4 runs (fresh or reused)."""
    return input(
        f"Sentence-transformer model name for stage 4 [{config.STAGE4_MODEL_NAME}]: "
    ).strip() or config.STAGE4_MODEL_NAME


def run_stages_1_to_3(dataset_path=None):
    """Runs cleaning, segmentation, and topic naming (stages 1-3), sharing
    config.MODEL_NAME across all three -- not a per-run choice.

    Returns (segmented_csv, topics).
    """
    dataset_path = dataset_path or config.DATASET_PATH

    print(f"Loading model '{config.MODEL_NAME}' for stages 1-3...")
    model = SentenceTransformer(config.MODEL_NAME)

    cleaned_csv = clean_dataset(dataset_path, config.CLEANED_CSV)
    segmented_csv = run_topic_shifting(cleaned_csv, config.SEGMENTED_CSV, model=model)
    topics = run_topic_naming(
        segmented_csv,
        config.RESULTS_CSV,
        definitions_path=config.TOPIC_DEFINITIONS_PATH,
        model=model,
        min_cluster_size_range=config.MIN_CLUSTER_SIZE_RANGE,
        min_samples_range=config.MIN_SAMPLES_RANGE,
    )
    print(f"Stage 3 finished: {config.RESULTS_CSV} ({len(topics)} topic(s))")

    return segmented_csv, topics


def run_stage4_only(
    model_name=None, segmented_csv=None, definitions_path=None, dataset_path=None,
    ask_before_stage6=True,
):
    """Runs stage 4 alone against an already-segmented CSV, reusing stage
    3's topics/thresholds (from `definitions_path`) but with its own,
    possibly different, embedding model. Doesn't touch stages 1-3 at all
    -- use this to re-run stage 4 with a different embedding model
    without redoing cleaning, segmentation, or topic naming.

    If stage 4 produces a non-empty timeline, automatically runs stage 5
    (the topic equivalence grid) against `dataset_path` (the RAW dataset --
    defaults to config.DATASET_PATH) right after -- see this module's
    docstring.

    Stage 6 (window scoring) never runs on its own initiative -- it needs
    your hand-edits to stage 5's grid first. But if `ask_before_stage6` is
    True (the default -- set it False for a non-interactive/scripted call,
    e.g. from tests), once stage 5 finishes this function PAUSES with an
    input() prompt in the same terminal session: go edit
    config.EQUIVALENCE_GRID_CSV, then answer that prompt to run stage 6
    immediately afterward, in the same run, with no separate script
    invocation needed. Answering no (or Enter) leaves stage 6 for later --
    run `python stage6_window_score.py` yourself whenever you're ready.

    Returns the timeline DataFrame (or None if stage 4 was skipped because
    the dataset is missing the columns it needs).
    """
    model_name = model_name or config.STAGE4_MODEL_NAME
    segmented_csv = segmented_csv or config.SEGMENTED_CSV
    definitions_path = definitions_path or config.TOPIC_DEFINITIONS_PATH
    dataset_path = dataset_path or config.DATASET_PATH

    if not os.path.exists(segmented_csv):
        raise FileNotFoundError(
            f"'{segmented_csv}' not found -- run the full pipeline (stages 1-3) at least once first."
        )
    if not os.path.exists(definitions_path):
        raise FileNotFoundError(
            f"'{definitions_path}' not found -- run the full pipeline (stages 1-3) at least once first "
            "so stage 3 can write it."
        )

    print(f"Loading model '{model_name}' for stage 4...")
    model = SentenceTransformer(model_name, **config.STAGE4_MODEL_KWARGS)

    timeline = None
    try:
        timeline = run_topic_timeline(
            segmented_csv,
            config.TIMELINE_CSV,
            definitions_path=definitions_path,
            model=model,
            sender_column=config.SENDER_COLUMN,
            timestamp_column=config.TIMESTAMP_COLUMN,
            chatroom_column=config.CHATROOM_COLUMN,
        )
        print(f"Stage 4 finished: {config.TIMELINE_CSV} ({len(timeline)} timeline row(s))")
    except (KeyError, RuntimeError) as e:
        # KeyError: dataset is missing a column stage 4 needs (Sender/Timestamp/Chatroom).
        # RuntimeError: topic_definitions.json has no topics to reuse.
        print(f"Stage 4 skipped -- {e}")
        return timeline

    if timeline is None or timeline.empty:
        print("Stage 5 skipped -- stage 4 produced no timeline rows to evaluate.")
        return timeline

    try:
        build_topic_equivalence_table(
            dataset_path=dataset_path,
            timeline_path=config.TIMELINE_CSV,
            out_path=config.EQUIVALENCE_GRID_CSV,
            manual_topic_column=config.MANUAL_TOPIC_COLUMN,
        )
        print(
            f"Stage 5 finished: {config.EQUIVALENCE_GRID_CSV} -- hand-edit \"No\" -> \"Yes\" where a "
            "manual and predicted topic mean the same thing."
        )
    except (FileNotFoundError, KeyError) as e:
        # FileNotFoundError: dataset_path doesn't exist. KeyError: dataset_path is
        # missing config.MANUAL_TOPIC_COLUMN -- expected if you have no manual labels.
        print(f"Stage 5 skipped -- {e}")
        return timeline

    if ask_before_stage6:
        answer = input(
            f"Finished editing {config.EQUIVALENCE_GRID_CSV}? Run stage 6 (window scoring) now? [y/N]: "
        ).strip().lower()
        if answer in ("y", "yes"):
            run_window_score(
                dataset_path=dataset_path,
                timeline_path=config.TIMELINE_CSV,
                grid_path=config.EQUIVALENCE_GRID_CSV,
                manual_topic_column=config.MANUAL_TOPIC_COLUMN,
                window_matches_path=config.WINDOW_MATCHES_CSV,
                report_path=config.WINDOW_SCORE_REPORT_TXT,
            )
        else:
            print(
                "Skipping stage 6 for now -- run `python stage6_window_score.py` "
                "yourself whenever you're ready."
            )

    return timeline


def run_all(dataset_path=None, stage4_model_name=None):
    """Runs the full pipeline: stages 1-3 with config.MODEL_NAME, then
    stage 4 (and, if it succeeds, stage 5) with stage 4's own (possibly
    different) model."""
    dataset_path = dataset_path or config.DATASET_PATH
    segmented_csv, topics = run_stages_1_to_3(dataset_path)
    timeline = run_stage4_only(stage4_model_name, segmented_csv, dataset_path=dataset_path)

    print("Pipeline finished.")
    return topics, timeline


def main():
    """Entry point for the `scope-pipeline` console script (see
    pyproject.toml) as well as `python -m semantic_clustering_evidence.run_pipeline`."""
    if os.path.exists(config.SEGMENTED_CSV):
        reuse = input(
            f"Found existing '{config.SEGMENTED_CSV}'. Reuse it and run stage 4 only "
            "(skip stages 1-3)? [Y/n]: "
        ).strip().lower()
    else:
        reuse = "n"

    if reuse in ("", "y", "yes"):
        run_stage4_only(get_stage4_model_input())
    else:
        dataset_path = get_dataset_input()
        stage4_model_name = get_stage4_model_input()
        run_all(dataset_path, stage4_model_name)


if __name__ == "__main__":
    main()
