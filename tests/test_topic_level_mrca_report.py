import unittest

import pandas as pd

from scripts.render_topic_level_mrca_report import (
    correspondence_analysis,
    normalized,
    product_topic_alignment,
    spearman_for_modalities,
)


class TopicLevelMrcaReportTests(unittest.TestCase):
    def test_correspondence_analysis_returns_two_dimensions(self):
        matrix = pd.DataFrame(
            {
                "visual": [5.0, 2.0, 3.0],
                "texture": [2.0, 5.0, 3.0],
                "flavor": [3.0, 2.0, 6.0],
            },
            index=["A", "B", "C"],
        )

        result = correspondence_analysis(matrix)

        self.assertEqual(result["row_coords"].shape, (3, 2))
        self.assertEqual(result["col_coords"].shape, (3, 2))
        self.assertAlmostEqual(float(result["inertia_share"].sum()), 1.0)

    def test_product_topic_alignment_uses_product_means(self):
        data = pd.DataFrame(
            {
                "Product": ["A", "A", "B"],
                "overall_visual_match": [6, 4, 2],
                "color_pinkness_match": [5, None, 1],
            }
        )

        aligned = product_topic_alignment(data)

        self.assertEqual(float(aligned.loc["A", "overall_visual_match"]), 5.0)
        self.assertEqual(float(aligned.loc["A", "color_pinkness_match"]), 5.0)

    def test_normalized_and_spearman(self):
        left = {"Visual": 0.1, "Texture": 0.2, "Flavor": 0.3}
        right = {"Visual": 2.0, "Texture": 4.0, "Flavor": 6.0}

        self.assertAlmostEqual(sum(normalized(left).values()), 1.0)
        self.assertAlmostEqual(spearman_for_modalities(left, right), 1.0)


if __name__ == "__main__":
    unittest.main()
