"""
Pipeline stage 6 (evaluation): window-first scoring.

NOT run automatically by run_pipeline.py -- unlike stages 1-5, this stage
depends on you having hand-edited topic_equivalence_grid.csv (written by
stage5_topic_equivalence.py) first. Run it yourself once that's done:

    python stage6_window_score.py

or from code:

    from stage6_window_score import run_window_score
    run_window_score()

Step 1: for EACH ROW ("window") in topic_timeline.csv -- one contiguous
run of one user talking about one predicted topic -- find that window's
member messages in the raw dataset (same Sender, timestamp inside
[Start, End]), and count how many of them have a manual/reference topic
label (the dataset's manual-topic column) that is the SAME as or marked
EQUIVALENT to (via topic_equivalence_grid.csv) the window's own Topic.

This produces window_matches.csv: one row per window, with
Total_Messages / Matching_Messages / Match_Ratio -- so you can see
window-by-window whether the model's segment was mostly right before
any aggregate score is computed.

Step 2: roll those window-level counts up into precision/recall/F1 per
manual topic:
  - Precision(X): across every window whose predicted Topic is X (or
    equivalent to X), what fraction of their messages actually matched?
    sum(Matching_Messages) / sum(Total_Messages) over those windows.
  - Recall(X): across every message in the raw dataset whose manual topic
    is X, what fraction fell inside a window that matched? A message not
    covered by any window at all is automatically a non-match (the model
    never even attempted it).

Requires: the raw dataset, topic_timeline.csv, and topic_equivalence_grid.csv
(from stage5_topic_equivalence.py) -- see run_window_score()'s defaults.
"""
import pandas as pd

DEFAULT_MANUAL_TOPIC_COLUMN = "Predicted Topic"


def run_window_score(
    dataset_path="dataset.csv",
    timeline_path="topic_timeline.csv",
    grid_path="topic_equivalence_grid.csv",
    manual_topic_column=DEFAULT_MANUAL_TOPIC_COLUMN,
    window_matches_path="window_matches.csv",
    report_path="window_score_report.txt",
    verbose=True,
):
    """
    Runs the full window-first scoring pipeline (see module docstring) and
    writes `window_matches_path` and `report_path`.

    Returns (window_table, report) -- the per-window DataFrame and the
    per-manual-topic precision/recall/F1 DataFrame.
    """
    ds = pd.read_csv(dataset_path).rename(columns={manual_topic_column: "Manual_Topic"})
    tl = pd.read_csv(timeline_path).reset_index(drop=True)
    tl.index.name = "Window_Index"

    grid = pd.read_csv(grid_path, index_col=0)
    grid = grid.apply(lambda col: col.str.strip().str.lower().isin(["yes", "y", "true", "1"]))

    def is_equivalent(manual_topic, window_topic):
        if manual_topic == window_topic:
            return True
        return manual_topic in grid.index and window_topic in grid.columns and grid.loc[manual_topic, window_topic]

    ds["ts"] = pd.to_datetime(ds["Timestamp"])
    tl["seg_start"] = pd.to_datetime(tl["Start Date"] + " " + tl["Start Time"])
    tl["seg_end"] = pd.to_datetime(tl["End Date"] + " " + tl["End Time"])

    # ---- Step 1: per-window message counts ----
    window_rows = []
    # also track, per raw-dataset message, which window (if any) covers it
    # and whether that window's topic matched -- needed for recall in step 2.
    ds["Covering_Window"] = -1
    ds["Window_Match"] = False

    for widx, w in tl.iterrows():
        member_mask = (
            (ds["Sender"] == w["User"])
            & (ds["ts"] >= w["seg_start"])
            & (ds["ts"] <= w["seg_end"])
        )
        members = ds[member_mask]
        total = len(members)
        # Guard the empty-window case explicitly: on some pandas string-dtype
        # configurations, Series.sum() over an empty object/str Series
        # returns '' instead of 0, which would crash int(matches) below.
        matches = (
            members["Manual_Topic"].apply(lambda mt: is_equivalent(mt, w["Topic"])).sum()
            if total else 0
        )

        window_rows.append({
            "Window_Index": widx,
            "User": w["User"],
            "Start": f"{w['Start Date']} {w['Start Time']}",
            "End": f"{w['End Date']} {w['End Time']}",
            "Topic": w["Topic"],
            "Total_Messages": total,
            "Matching_Messages": int(matches),
            "Match_Ratio": round(matches / total, 3) if total else 0.0,
        })

        # record coverage on the message table (a message covered by more than
        # one window -- shouldn't normally happen -- keeps the last one checked)
        if total:
            match_flags = members["Manual_Topic"].apply(lambda mt: is_equivalent(mt, w["Topic"]))
            ds.loc[member_mask, "Covering_Window"] = widx
            ds.loc[member_mask, "Window_Match"] = match_flags.values

    window_table = pd.DataFrame(window_rows)
    window_table.to_csv(window_matches_path, index=False)

    if verbose:
        print("=== Per-window match counts ===")
        print(window_table.to_string(index=False))
        print()

    # ---- Diagnostics: catch the two most common causes of an all-zero result ----
    if verbose and window_table["Total_Messages"].sum() == 0:
        print("WARNING: every window has 0 member messages -- the Sender/timestamp")
        print(f"join between {dataset_path} and {timeline_path} found nothing at all.")
        print(f"  {dataset_path} Sender values:      {sorted(ds['Sender'].dropna().unique())}")
        print(f"  {timeline_path} User values: {sorted(tl['User'].dropna().unique())}")
        print(f"  {dataset_path} timestamp range:      {ds['ts'].min()}  to  {ds['ts'].max()}")
        print(f"  {timeline_path} window range:  {tl['seg_start'].min()}  to  {tl['seg_end'].max()}")
        print("  -> if the Sender/User lists don't overlap, or the timestamp ranges")
        print("     don't overlap, that's the bug (name mismatch or a date-parsing")
        print("     issue -- e.g. pandas reading '5/2/18' as May 2 in one file and")
        print("     as a different date in the other).")
        print()

    n_yes_cells = int(grid.values.sum())
    if verbose:
        print(f"{grid_path} has {n_yes_cells} cell(s) marked Yes "
              f"(grid shape {grid.shape[0]} manual topics x {grid.shape[1]} predicted topics).")
        if n_yes_cells == 0:
            print("WARNING: no cells are marked Yes, and no manual topic name is spelled")
            print("exactly the same as a predicted topic name either -- every message will")
            print("score as a non-match. Open the grid and mark the obvious equivalences.")
        print()

    # ---- Step 2: roll up into precision/recall/F1 per manual topic ----
    manual_topics = sorted(ds["Manual_Topic"].dropna().unique())
    rows = []
    for topic in manual_topics:
        # Precision: over every window whose Topic is equivalent to this manual
        # topic, what fraction of their messages matched?
        topic_windows = window_table[window_table["Topic"].apply(lambda wt: is_equivalent(topic, wt))]
        tp_precision = topic_windows["Matching_Messages"].sum()
        total_precision = topic_windows["Total_Messages"].sum()
        precision = tp_precision / total_precision if total_precision else 0.0

        # Recall: over every raw-dataset message truly labeled this manual topic,
        # what fraction fell inside a matching window?
        topic_messages = ds[ds["Manual_Topic"] == topic]
        tp_recall = int(topic_messages["Window_Match"].sum())
        total_recall = len(topic_messages)
        recall = tp_recall / total_recall if total_recall else 0.0

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        rows.append({
            "Manual_Topic": topic,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
            "Support": total_recall,
        })

    report = pd.DataFrame(rows)
    report_lines = [report.to_string(index=False), ""]

    macro_p, macro_r, macro_f1 = report["Precision"].mean(), report["Recall"].mean(), report["F1"].mean()
    weighted_p = (report["Precision"] * report["Support"]).sum() / report["Support"].sum()
    weighted_r = (report["Recall"] * report["Support"]).sum() / report["Support"].sum()
    weighted_f1 = (report["F1"] * report["Support"]).sum() / report["Support"].sum()

    # Overall (micro-averaged): pool every message together FIRST, then compute
    # one precision and one recall -- as opposed to macro/weighted, which average
    # the already-computed PER-TOPIC precision/recall values. A topic with very
    # few messages barely moves this number, unlike macro (where it counts the
    # same as every other topic) or weighted (where its own support scales it).
    overall_precision = (
        window_table["Matching_Messages"].sum() / window_table["Total_Messages"].sum()
        if window_table["Total_Messages"].sum() else 0.0
    )
    overall_recall = ds["Window_Match"].sum() / len(ds) if len(ds) else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) else 0.0
    )

    report_lines.append(f"macro    -> precision={macro_p:.3f}  recall={macro_r:.3f}  F1={macro_f1:.3f}")
    report_lines.append(f"weighted -> precision={weighted_p:.3f}  recall={weighted_r:.3f}  F1={weighted_f1:.3f}")
    report_lines.append(
        f"overall  -> precision={overall_precision:.3f}  recall={overall_recall:.3f}  F1={overall_f1:.3f}"
        f"  (all messages pooled together, not averaged per topic)"
    )

    report_text = "\n".join(report_lines)

    with open(report_path, "w") as f:
        f.write("=== Per-window match counts ===\n")
        f.write(window_table.to_string(index=False))
        f.write("\n\n=== Per-topic precision/recall/F1, rolled up from window counts ===\n")
        f.write(report_text + "\n")

    if verbose:
        print("=== Per-topic precision/recall/F1, rolled up from window counts ===")
        print(report_text)
        print(f"\nWindow-level detail written to {window_matches_path}")
        print(f"Full report written to {report_path}")

    return window_table, report


if __name__ == "__main__":
    run_window_score()
