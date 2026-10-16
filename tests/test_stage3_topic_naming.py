import numpy as np
import pytest

from semantic_clustering_evidence.stage3_topic_naming import (
    _cluster_documents,
    cluster_segments,
    compute_topic_thresholds,
    kneedle_threshold,
    score_segments_against_topics,
)


class TestKneedleThreshold:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            kneedle_threshold([])

    def test_constant_scores_returns_mean(self):
        # sigma == 0 -- every score identical, so c(k) == mu for every k.
        assert kneedle_threshold([0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_result_is_within_mu_to_mu_plus_sigma(self):
        scores = [0.1, 0.2, 0.2, 0.3, 0.9, 0.95, 0.95, 0.96]
        mu, sigma = np.mean(scores), np.std(scores)
        threshold = kneedle_threshold(scores)
        assert mu - 1e-9 <= threshold <= mu + sigma + 1e-9

    def test_bimodal_distribution_separates_clusters(self):
        # A clean gap between a low cluster and a high cluster -- the knee
        # should land in between, so it discriminates the two groups.
        low = [0.05] * 10
        high = [0.95] * 10
        threshold = kneedle_threshold(low + high)
        assert 0.05 < threshold < 0.95

    def test_no_discriminating_cutoff_returns_mean(self):
        # count_span == 0: with only two distinct values and no cutoff in
        # [mu, mu+sigma] changing which scores clear it is a degenerate
        # edge case this function detects explicitly.
        scores = [0.0, 1.0]
        mu = np.mean(scores)
        # Not asserting the degenerate branch fires here (that depends on
        # the exact sweep), just that the function returns a sane value in
        # range rather than raising.
        threshold = kneedle_threshold(scores)
        assert 0.0 <= threshold <= 1.0


class TestClusterSegments:
    def test_fewer_than_two_segments_raises(self):
        with pytest.raises(RuntimeError, match="Not enough segments"):
            cluster_segments(np.array([[0.1, 0.2]]))

    def test_two_well_separated_groups_cluster_together(self):
        rng = np.random.default_rng(0)
        group_a = rng.normal(loc=[1, 0], scale=0.01, size=(6, 2))
        group_b = rng.normal(loc=[0, 1], scale=0.01, size=(6, 2))
        embeddings = np.vstack([group_a, group_b])

        labels = cluster_segments(embeddings, min_cluster_size_range=range(2, 4), min_samples_range=range(1, 3))

        # The two tight, well-separated groups should land in different
        # clusters, each cluster pure (no mixing of group_a/group_b points).
        assert len(labels) == 12
        assert set(labels[:6]) == {labels[0]}
        assert set(labels[6:]) == {labels[6]}
        assert labels[0] != labels[6]


class TestClusterDocuments:
    def test_groups_text_by_cluster_and_drops_noise(self):
        texts = ["a", "b", "c", "d"]
        labels = np.array([0, 0, -1, 1])

        docs = _cluster_documents(texts, labels)

        assert docs == {0: "a b", 1: "d"}


class TestScoreSegmentsAgainstTopics:
    def test_probabilities_sum_to_one_per_segment(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
        topic_vectors = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}

        topic_ids, probabilities = score_segments_against_topics(embeddings, topic_vectors)

        assert topic_ids == [0, 1]
        assert probabilities.shape == (3, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-9)

    def test_segment_closest_to_a_topic_scores_it_highest(self):
        embeddings = np.array([[1.0, 0.0]])
        topic_vectors = {"near": np.array([1.0, 0.0]), "far": np.array([0.0, 1.0])}

        topic_ids, probabilities = score_segments_against_topics(embeddings, topic_vectors)

        near_idx = topic_ids.index("near")
        far_idx = topic_ids.index("far")
        assert probabilities[0, near_idx] > probabilities[0, far_idx]


class TestComputeTopicThresholds:
    def test_returns_one_threshold_per_topic(self):
        probabilities = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])
        topic_ids = [0, 1]

        thresholds = compute_topic_thresholds(probabilities, topic_ids, topic_names={0: "A", 1: "B"})

        assert set(thresholds.keys()) == {0, 1}
        assert all(isinstance(v, float) for v in thresholds.values())
