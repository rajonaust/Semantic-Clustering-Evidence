import pandas as pd
import pytest

from semantic_clustering_evidence.stage4_topic_timeline import (
    _full_segment_texts,
    _user_active_segments,
    build_topic_timeline,
    extract_user_topic_runs,
    load_conversation,
)


def _sample_df():
    # Two users, four segments: User-1 only ever talks about "nap", User-2
    # drifts from "nap" (segment 0) to "sketch" (segments 1-2) and is silent
    # in segment 3.
    rows = [
        {"Text": "nap sleep", "Segment": 0, "Sender": "User-1", "Timestamp": "2018-05-02 19:38:00", "Chatroom": "C1"},
        {"Text": "nap tired", "Segment": 0, "Sender": "User-2", "Timestamp": "2018-05-02 19:38:30", "Chatroom": "C1"},
        {"Text": "sketch design", "Segment": 1, "Sender": "User-2", "Timestamp": "2018-05-02 19:39:00", "Chatroom": "C1"},
        {"Text": "sketch deadline", "Segment": 2, "Sender": "User-2", "Timestamp": "2018-05-02 19:40:00", "Chatroom": "C1"},
        {"Text": "nap bed", "Segment": 3, "Sender": "User-1", "Timestamp": "2018-05-02 19:41:00", "Chatroom": "C1"},
    ]
    df = pd.DataFrame(rows)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


class TestUserActiveSegments:
    def test_returns_sorted_unique_segments_for_that_user(self):
        df = _sample_df()
        assert _user_active_segments(df, "User-2", "Segment", "Sender") == [0, 1, 2]
        assert _user_active_segments(df, "User-1", "Segment", "Sender") == [0, 3]


class TestFullSegmentTexts:
    def test_joins_every_participants_text_per_segment(self):
        df = _sample_df()
        texts = _full_segment_texts(df, "Text", "Segment")
        assert texts[0] == "nap sleep nap tired"
        assert texts[1] == "sketch design"


class TestExtractUserTopicRuns:
    def test_run_closes_when_topic_no_longer_eligible(self, fake_encoder):
        df = _sample_df()
        full_segment_texts = _full_segment_texts(df, "Text", "Segment")
        topic_ids = ["nap", "sketch"]
        topic_vectors = {
            "nap": fake_encoder.encode("nap sleep tired bed"),
            "sketch": fake_encoder.encode("sketch design deadline freelance"),
        }
        thresholds = {"nap": 0.3, "sketch": 0.3}

        runs = extract_user_topic_runs(
            df, "User-2", fake_encoder, topic_ids, topic_vectors, thresholds, full_segment_texts,
        )

        # User-2 is active in segments [0, 1, 2]; segment 0 is nap-flavored,
        # 1-2 are sketch-flavored, so "sketch" should produce one run
        # covering [1, 2] and should NOT include segment 0.
        sketch_runs = runs["sketch"]
        assert any(0 not in run and set(run) <= {1, 2} for run in sketch_runs)

    def test_user_with_no_active_segments_gets_no_runs(self, fake_encoder):
        df = _sample_df()
        full_segment_texts = _full_segment_texts(df, "Text", "Segment")
        topic_ids = ["nap"]
        topic_vectors = {"nap": fake_encoder.encode("nap sleep")}
        thresholds = {"nap": 0.3}

        runs = extract_user_topic_runs(
            df, "Nonexistent-User", fake_encoder, topic_ids, topic_vectors, thresholds, full_segment_texts,
        )

        assert runs == {"nap": []}


class TestBuildTopicTimeline:
    def test_produces_one_row_per_user_topic_run(self, fake_encoder):
        df = _sample_df()
        topic_ids = ["nap", "sketch"]
        topic_vectors = {
            "nap": fake_encoder.encode("nap sleep tired bed"),
            "sketch": fake_encoder.encode("sketch design deadline freelance"),
        }
        topic_names = {"nap": "Nap", "sketch": "Sketch"}
        thresholds = {"nap": 0.3, "sketch": 0.3}

        result = build_topic_timeline(df, fake_encoder, topic_ids, topic_vectors, topic_names, thresholds)

        assert set(result.columns) >= {
            "User", "Start Date", "Start Time", "End Date", "End Time",
            "Time Duration", "Topic", "Probability", "Chat Summary", "Full Segment Context",
        }
        assert not result.empty
        # User-1 only ever appears in "nap" segments.
        user1_rows = result[result["User"] == "User-1"]
        assert set(user1_rows["Topic"]) <= {"Nap"}


class TestLoadConversation:
    def test_missing_required_column_raises_keyerror(self, tmp_path):
        src = tmp_path / "in.csv"
        pd.DataFrame({"Text": ["a"], "Segment": [0]}).to_csv(src, index=False)

        with pytest.raises(KeyError, match="Sender"):
            load_conversation(str(src))

    def test_sorts_by_segment(self, tmp_path):
        src = tmp_path / "in.csv"
        pd.DataFrame({
            "Text": ["b", "a"],
            "Segment": [1, 0],
            "Sender": ["U1", "U1"],
            "Timestamp": ["2018-05-02 19:39:00", "2018-05-02 19:38:00"],
        }).to_csv(src, index=False)

        result = load_conversation(str(src))

        assert list(result["Text"]) == ["a", "b"]
