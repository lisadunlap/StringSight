import unittest
from unittest.mock import patch


class TestGenerateClusterSummaries(unittest.TestCase):
    def test_generate_cluster_summaries_handles_none(self) -> None:
        """
        `parallel_completions()` can return None values for failed/empty LLM calls.
        `generate_cluster_summaries()` should not crash and should fall back to a safe label.
        """
        from stringsight.clusterers.config import ClusterConfig
        from stringsight.clusterers import hierarchical_clustering as hc

        def fake_parallel_completions(messages, **_kwargs):
            # Preserve length/index alignment while simulating failures.
            return [None for _ in messages]

        with patch.object(hc, "parallel_completions", side_effect=fake_parallel_completions):
            cluster_values = {
                0: ["a", "b"],
                1: ["c"],
                -1: ["outlier"],
            }
            config = ClusterConfig(verbose=False)

            labels = hc.generate_cluster_summaries(cluster_values, config, column_name="property_description")

            self.assertEqual(labels[-1], "Outliers")
            self.assertEqual(labels[0], "Unlabeled cluster 0")
            self.assertEqual(labels[1], "Unlabeled cluster 1")

    def test_generate_cluster_summaries_handles_empty_string(self) -> None:
        from stringsight.clusterers.config import ClusterConfig
        from stringsight.clusterers import hierarchical_clustering as hc

        def fake_parallel_completions(messages, **_kwargs):
            return ["" for _ in messages]

        with patch.object(hc, "parallel_completions", side_effect=fake_parallel_completions):
            cluster_values = {0: ["a", "b"]}
            config = ClusterConfig(verbose=False)

            labels = hc.generate_cluster_summaries(cluster_values, config, column_name="property_description")

            self.assertEqual(labels[0], "Unlabeled cluster 0")



