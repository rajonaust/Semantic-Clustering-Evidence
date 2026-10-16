import pandas as pd
import pytest

from semantic_clustering_evidence.stage5_topic_equivalence import build_topic_equivalence_table


class TestBuildTopicEquivalenceTable:
    def test_grid_shape_and_default_no(self, tmp_path):
        dataset = tmp_path / "dataset.csv"
        pd.DataFrame({"Predicted Topic": ["Sleep", "Sleep", "Photoshoot"]}).to_csv(dataset, index=False)
        timeline = tmp_path / "timeline.csv"
        pd.DataFrame({"Topic": ["Nap", "Nap", "Sketch"]}).to_csv(timeline, index=False)
        out = tmp_path / "grid.csv"

        grid = build_topic_equivalence_table(
            dataset_path=str(dataset), timeline_path=str(timeline), out_path=str(out), verbose=False,
        )

        assert sorted(grid.index) == ["Photoshoot", "Sleep"]
        assert sorted(grid.columns) == ["Nap", "Sketch"]
        assert (grid == "No").all().all()
        assert out.exists()

    def test_missing_manual_topic_column_raises_keyerror(self, tmp_path):
        dataset = tmp_path / "dataset.csv"
        pd.DataFrame({"SomeOtherColumn": ["x"]}).to_csv(dataset, index=False)
        timeline = tmp_path / "timeline.csv"
        pd.DataFrame({"Topic": ["Nap"]}).to_csv(timeline, index=False)

        with pytest.raises(KeyError, match="Predicted Topic"):
            build_topic_equivalence_table(
                dataset_path=str(dataset), timeline_path=str(timeline),
                out_path=str(tmp_path / "grid.csv"), verbose=False,
            )

    def test_missing_topic_column_in_timeline_raises_keyerror(self, tmp_path):
        dataset = tmp_path / "dataset.csv"
        pd.DataFrame({"Predicted Topic": ["Sleep"]}).to_csv(dataset, index=False)
        timeline = tmp_path / "timeline.csv"
        pd.DataFrame({"NotTopic": ["Nap"]}).to_csv(timeline, index=False)

        with pytest.raises(KeyError, match="Topic"):
            build_topic_equivalence_table(
                dataset_path=str(dataset), timeline_path=str(timeline),
                out_path=str(tmp_path / "grid.csv"), verbose=False,
            )
