import numpy as np
import pandas as pd
import pytest

from semantic_clustering_evidence.stage2_topic_shifting import cosine_distance, run_topic_shifting


class TestCosineDistance:
    def test_identical_vectors_have_zero_distance(self):
        v = np.array([[1.0, 2.0, 3.0]])
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_vectors_have_distance_one(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-9)


class TestRunTopicShifting:
    def test_labels_every_row_and_covers_all_rows(self, tmp_path, fake_encoder):
        messages = (
            ["nap sleep", "tired bed", "sleep nap", "bed tired"]
            + ["sketch design", "deadline freelance", "design sketch", "freelance deadline"]
        )
        src = tmp_path / "in.csv"
        pd.DataFrame({"Text": messages}).to_csv(src, index=False)
        out = tmp_path / "out.csv"

        run_topic_shifting(str(src), str(out), model=fake_encoder, verbose=False)

        result = pd.read_csv(out)
        assert "Segment" in result.columns
        assert len(result) == len(messages)
        # Segment ids must be non-decreasing and cover every row.
        assert result["Segment"].is_monotonic_increasing
        assert result["Segment"].min() == 0

    def test_missing_text_column_raises_keyerror(self, tmp_path, fake_encoder):
        src = tmp_path / "in.csv"
        pd.DataFrame({"NotText": ["a", "b"]}).to_csv(src, index=False)

        with pytest.raises(KeyError, match="Text"):
            run_topic_shifting(str(src), str(tmp_path / "out.csv"), model=fake_encoder, verbose=False)
