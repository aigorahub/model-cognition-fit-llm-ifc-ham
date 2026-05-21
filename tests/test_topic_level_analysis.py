from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_topic_level_analysis.py"


def load_module():
    spec = importlib.util.spec_from_file_location("topic_level_analysis", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TopicLevelAnalysisTests(unittest.TestCase):
    def test_parse_keypool_env_expands_combined_and_numbered_keys(self):
        mod = load_module()
        text = """
OTHER=value
GEMINI_API_KEY=alpha
GEMINI_API_KEYS=beta,gamma beta
GEMINI_API_KEY_2=delta
GEMINI_API_KEY_1=gamma
"""

        keys = mod.parse_keypool_env(text)

        self.assertEqual(keys, ["alpha", "beta", "gamma", "delta"])


    def test_parse_topic_response_accepts_markdown_json_and_nulls(self):
        mod = load_module()
        payload = {topic: None for topic in mod.TOPICS}
        payload["overall_visual_match"] = 6
        payload["saltiness_match"] = 1
        text = "```json\n" + mod.json.dumps(payload) + "\n```"

        parsed = mod.parse_topic_response(text, scale_points=6)

        self.assertEqual(parsed["overall_visual_match"], 6)
        self.assertEqual(parsed["saltiness_match"], 1)
        self.assertIsNone(parsed["color_pinkness_match"])


    def test_parse_topic_response_rejects_out_of_range_values(self):
        mod = load_module()
        payload = {topic: None for topic in mod.TOPICS}
        payload["overall_visual_match"] = 7

        with self.assertRaisesRegex(ValueError, "overall_visual_match"):
            mod.parse_topic_response(mod.json.dumps(payload), scale_points=6)


    def test_prepare_modeling_frame_adds_missingness_flags_and_drops_errors(self):
        mod = load_module()
        rows = [
            {
                "row_id": 0,
                "Consumer": "H001",
                "Product": "J01",
                "Liking": 8.0,
                "parse_error": "",
                "overall_visual_match": 6,
                "saltiness_match": None,
            },
            {
                "row_id": 1,
                "Consumer": "H002",
                "Product": "J02",
                "Liking": 4.0,
                "parse_error": "bad json",
                "overall_visual_match": 2,
                "saltiness_match": 1,
            },
        ]
        for row in rows:
            for topic in mod.TOPICS:
                row.setdefault(topic, None)

        frame, features = mod.prepare_modeling_frame(pd.DataFrame(rows), scale_points=6)

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "saltiness_match"], 3.5)
        self.assertEqual(frame.loc[0, "saltiness_match_missing"], 1)
        self.assertEqual(frame.loc[0, "overall_visual_match_missing"], 0)
        self.assertIn("saltiness_match", features)
        self.assertIn("saltiness_match_missing", features)


if __name__ == "__main__":
    unittest.main()
