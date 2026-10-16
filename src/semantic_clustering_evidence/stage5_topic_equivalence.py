"""
Pipeline stage 5 (evaluation): topic equivalence grid.

Builds a small Manual_Topic x Predicted_Topic grid (one row per manual
topic name found in the raw dataset's manual/reference label column, one
column per predicted topic name found in stage 4's topic_timeline.csv).
Every cell starts "No".

This is the first half of a two-step, human-in-the-loop evaluation:

  1. (this stage) Build the grid, automatically, right after stage 4
     finishes -- so it's ready to edit the moment a run completes.
  2. You edit the grid by hand: put "Yes" in a cell wherever that
     predicted topic name and that manual topic name mean the same thing
     (e.g. row "Sleep", column "Nap" -> Yes). One topic can be marked
     equivalent to more than one row or column if that's genuinely the
     case (e.g. one predicted topic absorbing several manual topics, or
     vice versa).
  3. Run stage6_window_score.py (a separate, deliberate step -- it is
     NOT run automatically, since it depends on your edits to this grid).

Messages the model left unassigned (no topic_timeline.csv window covers
them) are handled automatically by stage6_window_score.py -- they always
score as a miss against whatever the manual label was, with no row/column
needed for them here.

Usage:
    from stage5_topic_equivalence import build_topic_equivalence_table
    build_topic_equivalence_table()  # -> topic_equivalence_grid.csv

From a terminal:
    python stage5_topic_equivalence.py
"""
import pandas as pd

DEFAULT_MANUAL_TOPIC_COLUMN = "Predicted Topic"


def build_topic_equivalence_table(
    dataset_path="dataset.csv",
    timeline_path="topic_timeline.csv",
    out_path="topic_equivalence_grid.csv",
    manual_topic_column=DEFAULT_MANUAL_TOPIC_COLUMN,
    verbose=True,
):
    """
    Reads the raw dataset's manual/reference topic labels and stage 4's
    predicted topic names, and writes an all-"No" Manual_Topic x
    Predicted_Topic grid to `out_path` for hand-editing.

    `dataset_path`: the RAW dataset (not stage 1's cleaned output) --
    `manual_topic_column` is a ground-truth label that stages 1-4 never
    touch, so either file would have it, but the raw path is the more
    obviously "original" source and doesn't depend on stage 1 having run.

    Raises KeyError if `manual_topic_column` isn't in `dataset_path`, or
    if topic_timeline.csv is missing its 'Topic' column (i.e. stage 4
    hasn't produced a real output yet).

    Returns the grid DataFrame (also written to `out_path`).
    """
    ds = pd.read_csv(dataset_path)
    if manual_topic_column not in ds.columns:
        raise KeyError(
            f"Column '{manual_topic_column}' not found in {dataset_path}. "
            f"Available columns: {list(ds.columns)}"
        )

    tl = pd.read_csv(timeline_path)
    if "Topic" not in tl.columns:
        raise KeyError(
            f"Column 'Topic' not found in {timeline_path}. Available columns: {list(tl.columns)}"
        )

    manual_topics = sorted(ds[manual_topic_column].dropna().unique())
    predicted_topics = sorted(tl["Topic"].dropna().unique())

    grid = pd.DataFrame("No", index=manual_topics, columns=predicted_topics)
    grid.index.name = "Manual_Topic \\ Predicted_Topic"
    grid.to_csv(out_path)

    if verbose:
        print(f"Wrote {out_path}")
        print(grid.to_string())
        print(
            "\nEdit cells to 'Yes' where a manual topic and predicted topic mean the same "
            "thing, then run stage6_window_score.py"
        )

    return grid


if __name__ == "__main__":
    build_topic_equivalence_table()
