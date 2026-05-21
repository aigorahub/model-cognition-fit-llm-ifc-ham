import unittest

import pandas as pd

from scripts.render_agent_type_simulation_report import build_findings, render_report_html


class AgentTypeReportTests(unittest.TestCase):
    def test_build_findings_identifies_winners_and_bias(self):
        matrix = pd.DataFrame(
            [
                {
                    "agent_type": "system1",
                    "analysis_model": "flash_lite_25",
                    "n": 60,
                    "mae": 1.1389,
                    "rmse": 1.362,
                    "mean_liking": 5.35,
                    "mean_predicted_liking": 4.3222,
                },
                {
                    "agent_type": "system1",
                    "analysis_model": "g3_flash_low",
                    "n": 60,
                    "mae": 0.7334,
                    "rmse": 0.925,
                    "mean_liking": 5.35,
                    "mean_predicted_liking": 5.2889,
                },
                {
                    "agent_type": "system2",
                    "analysis_model": "flash_lite_25",
                    "n": 60,
                    "mae": 2.9145,
                    "rmse": 3.115,
                    "mean_liking": 4.1033,
                    "mean_predicted_liking": 1.1889,
                },
                {
                    "agent_type": "system2",
                    "analysis_model": "g3_flash_low",
                    "n": 60,
                    "mae": 1.7089,
                    "rmse": 1.890,
                    "mean_liking": 4.1033,
                    "mean_predicted_liking": 2.4222,
                },
            ]
        )

        findings = build_findings(matrix)

        self.assertEqual(findings["winner_by_agent_type"]["system1"]["analysis_model"], "g3_flash_low")
        self.assertEqual(findings["winner_by_agent_type"]["system2"]["analysis_model"], "g3_flash_low")
        self.assertAlmostEqual(findings["mae_advantage_by_agent_type"]["system1"], 0.4055, places=4)
        self.assertAlmostEqual(findings["bias_by_cell"][("system2", "flash_lite_25")], -2.9144, places=4)

    def test_render_report_html_contains_experiment_claims(self):
        matrix = pd.DataFrame(
            [
                {
                    "agent_type": "system1",
                    "analysis_model": "flash_lite_25",
                    "n": 60,
                    "mae": 1.1389,
                    "rmse": 1.362,
                    "mean_liking": 5.35,
                    "mean_predicted_liking": 4.3222,
                },
                {
                    "agent_type": "system1",
                    "analysis_model": "g3_flash_low",
                    "n": 60,
                    "mae": 0.7334,
                    "rmse": 0.925,
                    "mean_liking": 5.35,
                    "mean_predicted_liking": 5.2889,
                },
            ]
        )
        metadata = {
            "rows": {"agents": 20, "evaluations": 120, "scores": 240},
            "product_ids": ["J22", "J17"],
            "generator_models": {
                "system1": {"model": "gemini-2.5-flash-lite", "thinking_config": None},
                "system2": {"model": "gemini-3-flash-preview", "thinking_config": {"thinkingLevel": "low"}},
            },
            "analysis_models": {
                "flash_lite_25": {"model": "gemini-2.5-flash-lite", "thinking_config": None},
                "g3_flash_low": {"model": "gemini-3-flash-preview", "thinking_config": {"thinkingLevel": "low"}},
            },
        }
        agents = pd.DataFrame({"agent_id": ["system1_01"], "agent_type": ["system1"]})

        html = render_report_html(matrix=matrix, agents=agents, metadata=metadata)

        self.assertIn("Model-person fit smoke test", html)
        self.assertIn("gemini-2.5-flash-lite", html)
        self.assertIn("gemini-3-flash-preview", html)
        self.assertIn("The matching effect did not appear in this first smoke test.", html)


if __name__ == "__main__":
    unittest.main()
