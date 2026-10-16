import builtins

from semantic_clustering_evidence import run_pipeline, config


class TestGetDatasetInput:
    def test_blank_answer_falls_back_to_config_default(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda prompt: "")
        assert run_pipeline.get_dataset_input() == config.DATASET_PATH

    def test_typed_answer_is_used_and_stripped(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda prompt: "  my_data.csv  ")
        assert run_pipeline.get_dataset_input() == "my_data.csv"


class TestGetStage4ModelInput:
    def test_blank_answer_falls_back_to_config_default(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda prompt: "")
        assert run_pipeline.get_stage4_model_input() == config.STAGE4_MODEL_NAME

    def test_typed_answer_overrides_default(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda prompt: "a-different-model")
        assert run_pipeline.get_stage4_model_input() == "a-different-model"


class TestRunAllWiring:
    def test_calls_stage_1_to_3_then_stage_4_with_its_own_model(self, monkeypatch):
        calls = []

        def fake_run_stages_1_to_3(dataset_path=None):
            calls.append(("stages_1_to_3", dataset_path))
            return "segmented.csv", ["topic-a"]

        def fake_run_stage4_only(model_name, segmented_csv, dataset_path=None):
            calls.append(("stage4", model_name, segmented_csv, dataset_path))
            return "fake-timeline"

        monkeypatch.setattr(run_pipeline, "run_stages_1_to_3", fake_run_stages_1_to_3)
        monkeypatch.setattr(run_pipeline, "run_stage4_only", fake_run_stage4_only)

        topics, timeline = run_pipeline.run_all("raw.csv", "stage4-model")

        assert calls == [
            ("stages_1_to_3", "raw.csv"),
            ("stage4", "stage4-model", "segmented.csv", "raw.csv"),
        ]
        assert topics == ["topic-a"]
        assert timeline == "fake-timeline"
