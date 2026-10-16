from pathlib import Path

import pandas as pd
import pytest

from semantic_clustering_evidence.stage6_window_score import run_window_score

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


class TestRunWindowScoreOnSampleData:
    """Regression test: locks in the exact scores this pipeline produced on
    the checked-in worked example, so a refactor that silently changes the
    scoring logic gets caught."""

    def test_matches_the_checked_in_sample_output(self, tmp_path):
        window_matches_out = tmp_path / "window_matches.csv"
        report_out = tmp_path / "window_score_report.txt"

        window_table, report = run_window_score(
            dataset_path=str(SAMPLE_DIR / "dataset.csv"),
            timeline_path=str(SAMPLE_DIR / "topic_timeline.csv"),
            grid_path=str(SAMPLE_DIR / "topic_equivalence_grid.csv"),
            window_matches_path=str(window_matches_out),
            report_path=str(report_out),
            verbose=False,
        )

        expected = pd.read_csv(SAMPLE_DIR / "window_matches.csv")
        pd.testing.assert_frame_equal(
            window_table[expected.columns].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

        # Spot-check the per-topic rollup too (see window_score_report.txt).
        by_topic = report.set_index("Manual_Topic")
        assert by_topic.loc["Sleep", "Recall"] == pytest.approx(1.0)
        assert by_topic.loc["Religious", "Support"] == 15
        assert by_topic.loc["Religious", "Precision"] == pytest.approx(0.0)


class TestRunWindowScoreEdgeCases:
    def test_window_with_zero_member_messages_scores_zero_not_crash(self, tmp_path):
        dataset = tmp_path / "dataset.csv"
        pd.DataFrame({
            "Sender": ["U1"],
            "Timestamp": ["2018-05-02 19:38:00"],
            "Predicted Topic": ["Sleep"],
        }).to_csv(dataset, index=False)

        timeline = tmp_path / "timeline.csv"
        pd.DataFrame({
            "User": ["U-nonexistent"],
            "Start Date": ["2018-05-02"], "Start Time": ["23:00:00"],
            "End Date": ["2018-05-02"], "End Time": ["23:05:00"],
            "Topic": ["Nap"],
        }).to_csv(timeline, index=False)

        grid = tmp_path / "grid.csv"
        pd.DataFrame({"Nap": ["No"]}, index=["Sleep"]).to_csv(grid, index_label="x")

        window_table, report = run_window_score(
            dataset_path=str(dataset), timeline_path=str(timeline), grid_path=str(grid),
            window_matches_path=str(tmp_path / "wm.csv"), report_path=str(tmp_path / "r.txt"),
            verbose=False,
        )

        assert window_table.loc[0, "Total_Messages"] == 0
        assert window_table.loc[0, "Match_Ratio"] == 0.0

    def test_exact_topic_name_match_counts_without_grid_entry(self, tmp_path):
        # is_equivalent() short-circuits on manual_topic == window_topic even
        # if the grid has no row/column for it at all.
        dataset = tmp_path / "dataset.csv"
        pd.DataFrame({
            "Sender": ["U1"],
            "Timestamp": ["2018-05-02 19:38:00"],
            "Predicted Topic": ["SameName"],
        }).to_csv(dataset, index=False)

        timeline = tmp_path / "timeline.csv"
        pd.DataFrame({
            "User": ["U1"],
            "Start Date": ["2018-05-02"], "Start Time": ["19:37:00"],
            "End Date": ["2018-05-02"], "End Time": ["19:39:00"],
            "Topic": ["SameName"],
        }).to_csv(timeline, index=False)

        grid = tmp_path / "grid.csv"
        pd.DataFrame({"Unrelated": ["No"]}, index=["AlsoUnrelated"]).to_csv(grid, index_label="x")

        window_table, _ = run_window_score(
            dataset_path=str(dataset), timeline_path=str(timeline), grid_path=str(grid),
            window_matches_path=str(tmp_path / "wm.csv"), report_path=str(tmp_path / "r.txt"),
            verbose=False,
        )

        assert window_table.loc[0, "Matching_Messages"] == 1
