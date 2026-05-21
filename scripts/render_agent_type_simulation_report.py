#!/usr/bin/env python3
"""Render a polished temporary HTML report for the agent-type simulation."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "data" / "agent_type_simulation_smoke_10x6").exists():
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "agent_type_simulation_smoke_10x6"
else:
    DEFAULT_OUTPUT_DIR = Path("Embeddings and Liking/agent_type_simulation_smoke_10x6")

AGENT_LABELS = {
    "system1": "System 1 synthetic agents",
    "system2": "System 2 synthetic agents",
}

ANALYSIS_LABELS = {
    "flash_lite_25": "Gemini 2.5 Flash Lite",
    "g3_flash_low": "Gemini 3 Flash low thinking",
}


def esc(value: Any) -> str:
    return html_lib.escape(str(value))


def fmt(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def build_findings(matrix: pd.DataFrame) -> dict[str, Any]:
    winners: dict[str, dict[str, Any]] = {}
    mae_advantage: dict[str, float] = {}
    bias: dict[tuple[str, str], float] = {}

    for _, row in matrix.iterrows():
        cell = (str(row["agent_type"]), str(row["analysis_model"]))
        bias[cell] = float(row["mean_predicted_liking"]) - float(row["mean_liking"])

    for agent_type, group in matrix.groupby("agent_type"):
        ordered = group.sort_values("mae")
        winners[str(agent_type)] = ordered.iloc[0].to_dict()

        flash = group[group["analysis_model"].eq("flash_lite_25")]
        g3 = group[group["analysis_model"].eq("g3_flash_low")]
        if not flash.empty and not g3.empty:
            mae_advantage[str(agent_type)] = float(flash.iloc[0]["mae"]) - float(g3.iloc[0]["mae"])

    return {
        "winner_by_agent_type": winners,
        "mae_advantage_by_agent_type": mae_advantage,
        "bias_by_cell": bias,
    }


def model_config_line(config: dict[str, Any]) -> str:
    thinking = config.get("thinking_config")
    if thinking:
        return f"{config.get('model')} | thinkingLevel={thinking.get('thinkingLevel')}"
    return str(config.get("model"))


def stat_card(label: str, value: Any, note: str) -> str:
    return f"""
<div class="stat-card">
  <div class="stat-value">{esc(value)}</div>
  <div class="stat-label">{esc(label)}</div>
  <div class="stat-note">{esc(note)}</div>
</div>
""".strip()


def render_matrix(matrix: pd.DataFrame) -> str:
    max_mae = max(float(matrix["mae"].max()), 0.01)
    cards = []
    for agent_type in ["system1", "system2"]:
        group = matrix[matrix["agent_type"].eq(agent_type)].copy().sort_values("mae")
        if group.empty:
            continue
        rows = []
        best_mae = float(group["mae"].min())
        for _, row in group.iterrows():
            analysis_model = str(row["analysis_model"])
            width = max(4.0, 100.0 * float(row["mae"]) / max_mae)
            is_winner = abs(float(row["mae"]) - best_mae) < 0.000001
            class_name = "matrix-row winner" if is_winner else "matrix-row"
            rows.append(
                f"""
<div class="{class_name}">
  <div class="matrix-row-head">
    <span>{esc(ANALYSIS_LABELS.get(analysis_model, analysis_model))}</span>
    <strong>MAE {fmt(row['mae'], 2)}</strong>
  </div>
  <div class="mae-track"><span style="width:{width:.1f}%"></span></div>
  <div class="matrix-note">
    predicted mean {fmt(row['mean_predicted_liking'], 2)} vs actual mean {fmt(row['mean_liking'], 2)}
  </div>
</div>
""".strip()
            )
        cards.append(
            f"""
<section class="matrix-card">
  <div class="eyebrow">{esc(AGENT_LABELS.get(agent_type, agent_type))}</div>
  <div class="matrix-rows">{''.join(rows)}</div>
</section>
""".strip()
        )
    return f'<div class="matrix-grid">{"".join(cards)}</div>'


def render_calibration(matrix: pd.DataFrame) -> str:
    strips = []
    for _, row in matrix.sort_values(["agent_type", "analysis_model"]).iterrows():
        actual_left = max(0.0, min(100.0, 10.0 * float(row["mean_liking"])))
        predicted_left = max(0.0, min(100.0, 10.0 * float(row["mean_predicted_liking"])))
        analysis_model = str(row["analysis_model"])
        agent_type = str(row["agent_type"])
        strips.append(
            f"""
<div class="cal-row">
  <div>
    <strong>{esc(AGENT_LABELS.get(agent_type, agent_type))}</strong>
    <span>{esc(ANALYSIS_LABELS.get(analysis_model, analysis_model))}</span>
  </div>
  <div class="cal-track" aria-label="Mean actual and predicted liking on a zero to ten scale">
    <i class="actual-marker" style="left:{actual_left:.1f}%"></i>
    <i class="pred-marker" style="left:{predicted_left:.1f}%"></i>
  </div>
  <div class="cal-values">actual {fmt(row['mean_liking'], 2)} | predicted {fmt(row['mean_predicted_liking'], 2)}</div>
</div>
""".strip()
        )
    return "".join(strips)


def render_model_table(metadata: dict[str, Any]) -> str:
    rows = []
    for agent_type in ["system1", "system2"]:
        config = metadata.get("generator_models", {}).get(agent_type, {})
        rows.append(
            f"""
<tr>
  <td>{esc(AGENT_LABELS.get(agent_type, agent_type))}</td>
  <td>agent profile and ham response generation</td>
  <td><code>{esc(model_config_line(config))}</code></td>
</tr>
""".strip()
        )
    for analysis_model in ["flash_lite_25", "g3_flash_low"]:
        config = metadata.get("analysis_models", {}).get(analysis_model, {})
        rows.append(
            f"""
<tr>
  <td>{esc(ANALYSIS_LABELS.get(analysis_model, analysis_model))}</td>
  <td>similarity scoring for every synthetic row</td>
  <td><code>{esc(model_config_line(config))}</code></td>
</tr>
""".strip()
        )
    return f"""
<table class="method-table">
  <thead><tr><th>Component</th><th>Role</th><th>Exact model route</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
""".strip()


def render_agent_mix(agents: pd.DataFrame) -> str:
    if agents.empty:
        return ""
    columns = ["agent_type", "salt_preference", "texture_priority", "flavor_priority"]
    available = [column for column in columns if column in agents.columns]
    rows = []
    for agent_type, group in agents.groupby("agent_type"):
        bits = [f"{len(group)} agents"]
        for column in available:
            if column == "agent_type":
                continue
            counts = group[column].value_counts().to_dict()
            if counts:
                top_value = max(counts, key=counts.get)
                bits.append(f"{column.replace('_', ' ')}: mostly {top_value}")
        rows.append(
            f"""
<li>
  <strong>{esc(AGENT_LABELS.get(str(agent_type), agent_type))}</strong>
  <span>{esc('; '.join(bits))}</span>
</li>
""".strip()
        )
    return f'<ul class="agent-mix">{"".join(rows)}</ul>'


def render_report_html(*, matrix: pd.DataFrame, agents: pd.DataFrame, metadata: dict[str, Any]) -> str:
    findings = build_findings(matrix)
    winners = findings["winner_by_agent_type"]
    all_winners_are_g3 = bool(winners) and all(
        str(winner.get("analysis_model")) == "g3_flash_low" for winner in winners.values()
    )
    headline = (
        "The matching effect did not appear in this first smoke test."
        if all_winners_are_g3
        else "The smoke test showed mixed model-agent fit."
    )
    rows = metadata.get("rows", {})
    product_ids = metadata.get("product_ids", [])

    agent_count = rows.get("agents", len(agents))
    evaluation_count = rows.get("evaluations", "")
    score_count = rows.get("scores", "")
    product_count = len(product_ids)
    agent_types = len(agents["agent_type"].unique()) if "agent_type" in agents.columns and not agents.empty else 0

    advantage_lines = []
    for agent_type in ["system1", "system2"]:
        if agent_type in findings["mae_advantage_by_agent_type"]:
            advantage = findings["mae_advantage_by_agent_type"][agent_type]
            advantage_lines.append(
                f"""
<li>
  <strong>{esc(AGENT_LABELS.get(agent_type, agent_type))}:</strong>
  G3 low reduced MAE by {fmt(advantage, 2)} points versus 2.5 Flash Lite.
</li>
""".strip()
            )

    product_list = ", ".join(str(product) for product in product_ids)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model-person fit smoke test</title>
<style>
:root {{
  --paper: #f7f6f2;
  --ink: #1a2721;
  --sage: #4a5d53;
  --terracotta: #c05a45;
  --sand: #eae6db;
  --line: #d9d1c0;
  --white: #fffdf8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: radial-gradient(circle at top left, rgba(192, 90, 69, 0.16), transparent 32rem), var(--paper);
  color: var(--ink);
  font-family: Manrope, Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}}
main {{ width: min(1180px, calc(100vw - 48px)); margin: 0 auto; padding: 48px 0 72px; }}
.hero {{
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.55fr);
  gap: 32px;
  align-items: end;
  padding-bottom: 28px;
  border-bottom: 1px solid var(--line);
}}
h1 {{
  margin: 0 0 14px;
  font-family: Georgia, "Times New Roman", serif;
  font-style: italic;
  font-size: clamp(46px, 7vw, 92px);
  line-height: 0.94;
  font-weight: 500;
  letter-spacing: 0;
}}
.kicker {{ color: var(--terracotta); font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; font-size: 12px; }}
.deckline {{ margin: 0; max-width: 760px; color: var(--sage); font-size: 20px; }}
.verdict {{
  background: var(--ink);
  color: var(--paper);
  border-radius: 8px;
  padding: 22px;
  box-shadow: 0 18px 36px rgba(26, 39, 33, 0.12);
}}
.verdict h2 {{ margin: 0 0 10px; font-size: 22px; }}
.verdict p {{ margin: 0; color: #e9e3d4; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 24px 0 36px; }}
.stat-card {{
  background: rgba(255, 253, 248, 0.72);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  min-height: 122px;
}}
.stat-value {{ font-size: 38px; line-height: 1; color: var(--terracotta); font-weight: 800; }}
.stat-label {{ margin-top: 10px; font-weight: 800; }}
.stat-note {{ margin-top: 4px; color: var(--sage); font-size: 13px; }}
.section {{
  margin-top: 34px;
  background: rgba(255, 253, 248, 0.62);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 24px;
}}
.section-head {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 18px; }}
.section h2 {{ margin: 0; font-size: 26px; letter-spacing: 0; }}
.section p {{ margin: 4px 0 0; color: var(--sage); max-width: 760px; }}
.matrix-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
.matrix-card {{ background: var(--sand); border-radius: 8px; padding: 18px; border: 1px solid #d8cfbd; }}
.eyebrow {{ color: var(--sage); text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; font-weight: 800; }}
.matrix-rows {{ display: grid; gap: 12px; margin-top: 14px; }}
.matrix-row {{
  background: rgba(247, 246, 242, 0.68);
  border: 1px solid transparent;
  border-radius: 8px;
  padding: 14px;
}}
.matrix-row.winner {{ border-color: var(--terracotta); background: #fff8f4; }}
.matrix-row-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; font-size: 16px; }}
.matrix-row-head strong {{ color: var(--terracotta); white-space: nowrap; }}
.matrix-note {{ margin-top: 7px; color: var(--sage); font-size: 13px; }}
.mae-track {{ width: 100%; height: 12px; border-radius: 999px; background: rgba(74, 93, 83, 0.18); overflow: hidden; margin-top: 10px; }}
.mae-track span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--terracotta), #cf806f); border-radius: 999px; }}
.callout-list {{ margin: 0; padding: 0; list-style: none; display: grid; gap: 10px; }}
.callout-list li {{ padding: 12px 14px; border-left: 4px solid var(--terracotta); background: rgba(234, 230, 219, 0.58); border-radius: 0 8px 8px 0; }}
.calibration {{ display: grid; gap: 14px; }}
.cal-row {{ display: grid; grid-template-columns: 260px 1fr 190px; gap: 14px; align-items: center; }}
.cal-row strong {{ display: block; }}
.cal-row span, .cal-values {{ color: var(--sage); font-size: 13px; }}
.cal-track {{
  position: relative;
  height: 16px;
  border-radius: 999px;
  background: linear-gradient(90deg, #efe9dc, #dacdb6);
  border: 1px solid var(--line);
}}
.cal-track i {{ position: absolute; top: 50%; transform: translate(-50%, -50%); display: block; }}
.actual-marker {{ width: 16px; height: 16px; border-radius: 50%; background: var(--ink); box-shadow: 0 0 0 3px var(--paper); }}
.pred-marker {{ width: 6px; height: 28px; background: var(--terracotta); border-radius: 999px; }}
.legend {{ display: flex; gap: 16px; color: var(--sage); font-size: 13px; margin-top: 10px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.dot {{ width: 12px; height: 12px; border-radius: 50%; background: var(--ink); display: inline-block; }}
.tick {{ width: 5px; height: 18px; border-radius: 999px; background: var(--terracotta); display: inline-block; }}
.two-col {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 0.9fr); gap: 22px; }}
.method-table {{ width: 100%; border-collapse: collapse; background: var(--white); border-radius: 8px; overflow: hidden; }}
.method-table th, .method-table td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }}
.method-table th {{ color: var(--sage); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
code {{ color: var(--ink); background: var(--sand); padding: 2px 5px; border-radius: 4px; }}
.agent-mix {{ margin: 0; padding: 0; list-style: none; display: grid; gap: 12px; }}
.agent-mix li {{ background: var(--sand); border: 1px solid #d8cfbd; border-radius: 8px; padding: 14px; }}
.agent-mix strong {{ display: block; margin-bottom: 3px; }}
.agent-mix span {{ color: var(--sage); }}
.products {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
.products span {{ background: var(--ink); color: var(--paper); padding: 7px 10px; border-radius: 999px; font-weight: 800; font-size: 13px; }}
.next-step {{ font-size: 18px; color: var(--ink); max-width: 900px; }}
footer {{ margin-top: 26px; color: var(--sage); font-size: 13px; }}
@media (max-width: 900px) {{
  main {{ width: min(100vw - 28px, 760px); padding-top: 28px; }}
  .hero, .stat-grid, .matrix-grid, .two-col {{ grid-template-columns: 1fr; }}
  .stat-grid {{ gap: 10px; }}
  .cal-row {{ grid-template-columns: 1fr; gap: 8px; }}
}}
</style>
</head>
<body>
<main>
  <header class="hero">
    <div>
      <div class="kicker">Synthetic consumer simulation</div>
      <h1>Model-person fit smoke test</h1>
      <p class="deckline">A small 10 plus 10 agent run asked whether a System 1 analysis model works better for System 1 synthetic consumers, and whether a low-thinking Gemini 3 analysis model works better for System 2 synthetic consumers.</p>
    </div>
    <aside class="verdict">
      <h2>{esc(headline)}</h2>
      <p>Gemini 3 Flash low thinking produced lower MAE for both synthetic agent groups. The current signal looks more like scoring calibration than model-person matching.</p>
    </aside>
  </header>

  <section class="stat-grid" aria-label="Run size">
    {stat_card("Synthetic agents", agent_count, f"{agent_types} decision styles")}
    {stat_card("Ham evaluations", evaluation_count, "same fixed products for each agent")}
    {stat_card("Similarity scores", score_count, "two analysis models per row")}
    {stat_card("Products", product_count, product_list)}
    {stat_card("Winner", "G3 low", "lower MAE in both groups")}
  </section>

  <section class="section">
    <div class="section-head">
      <div>
        <h2>The 2 x 2 result</h2>
        <p>Rows are synthetic consumer type. Within each row, both analysis models scored the same actual-vs-ideal comments. Lower MAE is better.</p>
      </div>
    </div>
    {render_matrix(matrix)}
  </section>

  <section class="section">
    <div class="section-head">
      <div>
        <h2>What this says</h2>
        <p>The first pass is useful because it rules out the simple version of the matching story. We did not see 2.5 Flash Lite win on System 1 synthetic consumers.</p>
      </div>
    </div>
    <ul class="callout-list">
      {''.join(advantage_lines)}
      <li><strong>Likely issue:</strong> the direct similarity-to-liking mapping is not calibrated. Flash Lite's predicted liking means are lower than the generated liking scores, especially for System 2 agents.</li>
      <li><strong>Next cleaner test:</strong> use the model scores as features in a small calibrated downstream model, then compare held-out performance by synthetic agent type.</li>
    </ul>
  </section>

  <section class="section">
    <div class="section-head">
      <div>
        <h2>Calibration check</h2>
        <p>Black dots show mean generated liking. Terracotta ticks show mean predicted liking after the current direct 1-to-6 similarity mapping.</p>
      </div>
    </div>
    <div class="calibration">{render_calibration(matrix)}</div>
    <div class="legend"><span><i class="dot"></i>actual mean liking</span><span><i class="tick"></i>predicted mean liking</span></div>
  </section>

  <section class="section">
    <div class="section-head">
      <div>
        <h2>Model routing</h2>
        <p>The smoke test pinned Flash Lite to the 2.5 model ID and used low thinking for Gemini 3 Flash.</p>
      </div>
    </div>
    {render_model_table(metadata)}
  </section>

  <section class="section">
    <div class="two-col">
      <div>
        <h2>Run details</h2>
        <p>All agents evaluated the same six hams, so product choice does not explain differences across cells.</p>
        <div class="products">{''.join(f'<span>{esc(product)}</span>' for product in product_ids)}</div>
      </div>
      <div>
        <h2>Agent mix</h2>
        {render_agent_mix(agents)}
      </div>
    </div>
  </section>

  <section class="section">
    <h2>How to read it</h2>
    <p class="next-step">This is a pipeline and hypothesis-generation result, not a paper-ready finding. The useful broad conclusion is that the naive direct-score setup did not produce a person-model matching effect. The next version should keep the synthetic design, but train or calibrate the liking prediction step before judging whether different analysis models fit different consumer types.</p>
  </section>

  <footer>
    Source folder: <code>{esc(metadata.get("output_dir", DEFAULT_OUTPUT_DIR))}</code>. Report reads saved CSV and JSON outputs only.
  </footer>
</main>
</body>
</html>
"""


def load_inputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matrix = pd.read_csv(output_dir / "matrix_summary.csv")
    agents = pd.read_csv(output_dir / "synthetic_agents.csv")
    metadata = json.loads((output_dir / "simulation_metadata.json").read_text())
    return matrix, agents, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--html", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix, agents, metadata = load_inputs(args.output_dir)
    html_path = args.html or args.output_dir / "agent_type_smoke_summary.html"
    html_path.write_text(render_report_html(matrix=matrix, agents=agents, metadata=metadata))
    print(html_path)


if __name__ == "__main__":
    main()
