import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from scripts.run_agent_type_simulation import (
    ANALYSIS_MODELS,
    GENERATOR_MODELS,
    build_generation_config,
    call_gemini_json,
    resolve_product_ids,
    summarize_matrix,
    similarity_to_liking,
    write_run_outputs,
)


class AgentTypeSimulationTests(unittest.TestCase):
    def test_models_are_pinned_to_exact_ids(self):
        self.assertEqual(GENERATOR_MODELS["system1"]["model"], "gemini-2.5-flash-lite")
        self.assertEqual(GENERATOR_MODELS["system2"]["model"], "gemini-3-flash-preview")
        self.assertEqual(ANALYSIS_MODELS["flash_lite_25"]["model"], "gemini-2.5-flash-lite")
        self.assertEqual(ANALYSIS_MODELS["g3_flash_low"]["thinking_config"], {"thinkingLevel": "low"})

    def test_generation_config_adds_low_thinking_only_when_requested(self):
        flash = build_generation_config(temperature=0.7, thinking_config=None)
        g3 = build_generation_config(temperature=0.7, thinking_config={"thinkingLevel": "low"})

        self.assertNotIn("thinkingConfig", flash)
        self.assertEqual(g3["thinkingConfig"], {"thinkingLevel": "low"})

    def test_similarity_to_liking_maps_six_point_scale_to_zero_ten(self):
        self.assertEqual(similarity_to_liking(1, 1, 1), 0.0)
        self.assertEqual(similarity_to_liking(6, 6, 6), 10.0)
        self.assertEqual(similarity_to_liking(3.5, 3.5, 3.5), 5.0)

    def test_summarize_matrix_groups_by_agent_type_and_analysis_model(self):
        rows = pd.DataFrame(
            [
                {"agent_type": "system1", "analysis_model": "flash_lite_25", "liking": 8, "predicted_liking": 7},
                {"agent_type": "system1", "analysis_model": "g3_flash_low", "liking": 8, "predicted_liking": 5},
                {"agent_type": "system2", "analysis_model": "flash_lite_25", "liking": 4, "predicted_liking": 6},
                {"agent_type": "system2", "analysis_model": "g3_flash_low", "liking": 4, "predicted_liking": 4},
            ]
        )

        matrix = summarize_matrix(rows)

        self.assertEqual(len(matrix), 4)
        s1_flash = matrix[
            (matrix["agent_type"] == "system1") & (matrix["analysis_model"] == "flash_lite_25")
        ].iloc[0]
        self.assertEqual(float(s1_flash["mae"]), 1.0)

    def test_call_gemini_json_retries_transport_errors(self):
        calls = {"count": 0}

        def fake_post(**_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise urllib.error.URLError(OSError("network down"))
            body = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"ok": true}'
                                }
                            ]
                        }
                    }
                ]
            }
            return 200, __import__("json").dumps(body)

        with patch("scripts.run_agent_type_simulation.post_gemini_json", side_effect=fake_post):
            result = call_gemini_json(
                model="gemini-2.5-flash-lite",
                thinking_config=None,
                prompt="Return JSON.",
                api_key="fake",
                temperature=0.7,
                timeout_seconds=1,
                attempts=2,
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls["count"], 2)

    def test_write_run_outputs_checkpoints_partial_files(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_run_outputs(
                output_dir=output_dir,
                agents=[{"agent_id": "system1_01", "agent_type": "system1"}],
                evaluations=[
                    {
                        "agent_id": "system1_01",
                        "agent_type": "system1",
                        "Product": "J01",
                        "liking": 7.0,
                    }
                ],
                scores=[
                    {
                        "agent_id": "system1_01",
                        "agent_type": "system1",
                        "Product": "J01",
                        "analysis_model": "flash_lite_25",
                        "liking": 7.0,
                        "predicted_liking": 6.0,
                    }
                ],
                metadata={"product_ids": ["J01"], "complete": False},
            )

            self.assertTrue((output_dir / "synthetic_agents.csv").exists())
            self.assertTrue((output_dir / "synthetic_evaluations.csv").exists())
            self.assertTrue((output_dir / "similarity_scores.csv").exists())
            self.assertTrue((output_dir / "matrix_summary.csv").exists())
            self.assertTrue((output_dir / "simulation_metadata.json").exists())

    def test_resolve_product_ids_reuses_metadata_on_resume(self):
        source = pd.DataFrame({"Product": ["J01", "J02", "J03", "J04"]})
        with TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "simulation_metadata.json"
            metadata_path.write_text('{"product_ids": ["J03", "J02"]}')

            product_ids = resolve_product_ids(
                source,
                count=3,
                seed=42,
                output_dir=Path(tmp),
                resume=True,
            )

        self.assertEqual(product_ids, ["J03", "J02"])


if __name__ == "__main__":
    unittest.main()
