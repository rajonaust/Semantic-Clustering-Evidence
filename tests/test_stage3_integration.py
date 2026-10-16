import json

import pandas as pd

from semantic_clustering_evidence import stage3_topic_naming
from semantic_clustering_evidence.stage3_topic_naming import run_topic_naming


class FakeKeyBERT:
    """Deterministic stand-in for KeyBERT.extract_keywords: picks whichever
    known keyword appears first in the cluster's text, so topic names stay
    predictable in the test below."""

    _known_keywords = ["nap", "sketch"]

    def __init__(self, *args, **kwargs):
        pass

    def extract_keywords(self, doc, **kwargs):
        for kw in self._known_keywords:
            if kw in doc.lower():
                return [(kw, 0.9)]
        return [("misc", 0.1)]


class TestRunTopicNamingEndToEnd:
    def test_discovers_and_writes_two_topics(self, tmp_path, fake_encoder, monkeypatch):
        monkeypatch.setattr(stage3_topic_naming, "KeyBERT", FakeKeyBERT)

        # Two clearly separated groups of segments (by the fake encoder's
        # bag-of-words vocabulary): "nap" segments and "sketch" segments.
        rows = []
        for seg_id, text in enumerate(
            ["nap sleep", "nap tired", "nap bed", "nap sleep tired"]
            + ["sketch design", "sketch deadline", "sketch freelance", "sketch design deadline"]
        ):
            rows.append({"Text": text, "Segment": seg_id})
        input_csv = tmp_path / "segmented.csv"
        pd.DataFrame(rows).to_csv(input_csv, index=False)

        output_csv = tmp_path / "topic_assignments.csv"
        definitions_path = tmp_path / "topic_definitions.json"

        results = run_topic_naming(
            input_csv=str(input_csv),
            output_csv=str(output_csv),
            definitions_path=str(definitions_path),
            model=fake_encoder,
            min_cluster_size_range=range(2, 4),
            min_samples_range=range(1, 3),
            verbose=False,
        )

        assert len(results) == 2
        assert set(results["Topic_Name"]) == {"Nap", "Sketch"}
        assert output_csv.exists()

        with open(definitions_path) as f:
            definitions = json.load(f)
        assert len(definitions) == 2
        assert {d["Topic_Name"] for d in definitions} == {"Nap", "Sketch"}
        assert all("Definition_Text" in d and "Dynamic_Threshold" in d for d in definitions)

    def test_raises_runtime_error_when_no_clusters_form(self, tmp_path, fake_encoder, monkeypatch):
        monkeypatch.setattr(stage3_topic_naming, "KeyBERT", FakeKeyBERT)

        # A single segment can never form two clusters.
        input_csv = tmp_path / "segmented.csv"
        pd.DataFrame({"Text": ["only one segment"], "Segment": [0]}).to_csv(input_csv, index=False)

        import pytest

        with pytest.raises(RuntimeError):
            run_topic_naming(
                input_csv=str(input_csv),
                output_csv=str(tmp_path / "out.csv"),
                definitions_path=str(tmp_path / "defs.json"),
                model=fake_encoder,
                verbose=False,
            )
