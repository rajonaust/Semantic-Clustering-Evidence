import pandas as pd
import pytest

from semantic_clustering_evidence.stage1_clean_dataset import clean_dataset, to_clean_text


class TestToCleanText:
    def test_none_becomes_empty_string(self):
        assert to_clean_text(None) == ""

    def test_nan_becomes_empty_string(self):
        assert to_clean_text(float("nan")) == ""

    def test_plain_string_passes_through(self):
        assert to_clean_text("hello") == "hello"

    def test_non_string_is_stringified(self):
        assert to_clean_text(42) == "42"
        assert to_clean_text(3.14) == "3.14"


class TestCleanDataset:
    def test_drops_unnamed_index_column(self, tmp_path):
        src = tmp_path / "in.csv"
        pd.DataFrame({"Unnamed: 0": [0, 1], "Text": ["a", "b"]}).to_csv(src, index=False)
        out = tmp_path / "out.csv"

        clean_dataset(str(src), str(out), verbose=False)

        result = pd.read_csv(out)
        assert "Unnamed: 0" not in result.columns
        assert list(result["Text"]) == ["a", "b"]

    def test_coerces_non_string_text_to_string(self, tmp_path):
        src = tmp_path / "in.csv"
        # A numeric value in the Text column, mixed with a real NaN.
        pd.DataFrame({"Text": ["hi", 123, None]}).to_csv(src, index=False)
        out = tmp_path / "out.csv"

        clean_dataset(str(src), str(out), verbose=False)

        result = pd.read_csv(out, keep_default_na=False)
        assert list(result["Text"]) == ["hi", "123", ""]
        assert result["Text"].apply(lambda x: isinstance(x, str)).all()

    def test_missing_text_column_raises_keyerror(self, tmp_path):
        src = tmp_path / "in.csv"
        pd.DataFrame({"NotText": ["a"]}).to_csv(src, index=False)

        with pytest.raises(KeyError, match="Text"):
            clean_dataset(str(src), str(tmp_path / "out.csv"), verbose=False)

    def test_returns_output_path(self, tmp_path):
        src = tmp_path / "in.csv"
        pd.DataFrame({"Text": ["a"]}).to_csv(src, index=False)
        out = tmp_path / "out.csv"

        returned = clean_dataset(str(src), str(out), verbose=False)

        assert returned == str(out)
        assert out.exists()
