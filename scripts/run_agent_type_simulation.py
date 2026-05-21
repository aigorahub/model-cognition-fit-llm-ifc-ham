#!/usr/bin/env python3
"""Small-scale System 1/System 2 agent-type simulation for the ham study."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from run_topic_level_analysis import (
        DEFAULT_DATA_XLSX,
        DEFAULT_KEYPOOL_ENV,
        extract_response_text,
        load_api_keys,
        load_ham_home_use_data,
        post_gemini_json,
    )
except ModuleNotFoundError:
    from scripts.run_topic_level_analysis import (
        DEFAULT_DATA_XLSX,
        DEFAULT_KEYPOOL_ENV,
        extract_response_text,
        load_api_keys,
        load_ham_home_use_data,
        post_gemini_json,
    )


GENERATOR_MODELS = {
    "system1": {
        "label": "System 1",
        "model": "gemini-2.5-flash-lite",
        "thinking_config": None,
        "description": "fast, intuitive, affective consumer response",
    },
    "system2": {
        "label": "System 2",
        "model": "gemini-3-flash-preview",
        "thinking_config": {"thinkingLevel": "low"},
        "description": "deliberate, analytical consumer response",
    },
}

ANALYSIS_MODELS = {
    "flash_lite_25": {
        "label": "Gemini 2.5 Flash Lite analysis",
        "model": "gemini-2.5-flash-lite",
        "thinking_config": None,
    },
    "g3_flash_low": {
        "label": "Gemini 3 Flash low-thinking analysis",
        "model": "gemini-3-flash-preview",
        "thinking_config": {"thinkingLevel": "low"},
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "data" / "agent_type_simulation_smoke_10x6").exists():
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "agent_type_simulation_smoke_10x6"
else:
    DEFAULT_OUTPUT_DIR = Path("Embeddings and Liking/agent_type_simulation_smoke")


def log(message: str) -> None:
    print(message, flush=True)


def build_generation_config(
    *,
    temperature: float,
    thinking_config: dict[str, Any] | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "temperature": temperature,
    }
    if thinking_config:
        config["thinkingConfig"] = thinking_config
    return config


def similarity_to_liking(
    visual_similarity: float,
    texture_similarity: float,
    flavor_similarity: float,
    *,
    scale_points: int = 6,
) -> float:
    mean_similarity = (visual_similarity + texture_similarity + flavor_similarity) / 3
    predicted = ((mean_similarity - 1) / (scale_points - 1)) * 10
    return round(max(0.0, min(10.0, predicted)), 3)


def summarize_matrix(scored_rows: pd.DataFrame) -> pd.DataFrame:
    frame = scored_rows.copy()
    frame["abs_error"] = (frame["predicted_liking"] - frame["liking"]).abs()
    frame["squared_error"] = (frame["predicted_liking"] - frame["liking"]) ** 2
    summary = (
        frame.groupby(["agent_type", "analysis_model"], as_index=False)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: math.sqrt(float(values.mean()))),
            mean_liking=("liking", "mean"),
            mean_predicted_liking=("predicted_liking", "mean"),
        )
        .sort_values(["agent_type", "analysis_model"])
    )
    return summary


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def call_gemini_json(
    *,
    model: str,
    thinking_config: dict[str, Any] | None,
    prompt: str,
    api_key: str,
    temperature: float,
    timeout_seconds: float,
    attempts: int,
) -> dict[str, Any]:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": build_generation_config(
            temperature=temperature,
            thinking_config=thinking_config,
        ),
    }
    last_error = ""
    for attempt in range(attempts):
        try:
            status, body = post_gemini_json(
                model=model,
                api_key=api_key,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
        except (OSError, urllib.error.URLError) as exc:
            last_error = f"transport error: {exc}"
            time.sleep(min(8, 2**attempt))
            continue
        if status == 429:
            last_error = "rate limited"
            time.sleep(2**attempt)
            continue
        if status != 200:
            last_error = f"HTTP {status}: {body[:400]}"
            time.sleep(1)
            continue
        response_text = extract_response_text(json.loads(body))
        return parse_json_text(response_text)
    raise RuntimeError(last_error or "Gemini request failed")


def build_product_profiles(source: pd.DataFrame, products: list[str], *, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    profiles: list[dict[str, str]] = []
    for product in products:
        rows = source[source["Product"].astype(str).eq(product)].copy()
        if rows.empty:
            continue
        profile = {"Product": product}
        for source_column, target_key in [
            ("DescriptionVisual", "visual_evidence"),
            ("DescriptionTexture", "texture_evidence"),
            ("DescriptionFlavor", "flavor_evidence"),
        ]:
            values = [str(value).strip() for value in rows[source_column].dropna().tolist() if str(value).strip()]
            rng.shuffle(values)
            profile[target_key] = " | ".join(values[:4])
        profiles.append(profile)
    return profiles


def choose_products(source: pd.DataFrame, *, count: int, seed: int) -> list[str]:
    products = sorted(source["Product"].astype(str).unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(products)
    return products[:count]


def resolve_product_ids(
    source: pd.DataFrame,
    *,
    count: int,
    seed: int,
    output_dir: Path,
    resume: bool,
) -> list[str]:
    metadata_path = output_dir / "simulation_metadata.json"
    if resume and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        product_ids = metadata.get("product_ids")
        if isinstance(product_ids, list) and product_ids:
            return [str(product) for product in product_ids]
    return choose_products(source, count=count, seed=seed)


def load_existing_records(output_dir: Path, *, resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not resume:
        return [], [], []

    def load_csv(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return pd.read_csv(path).to_dict("records")

    return (
        load_csv(output_dir / "synthetic_agents.csv"),
        load_csv(output_dir / "synthetic_evaluations.csv"),
        load_csv(output_dir / "similarity_scores.csv"),
    )


def find_record(records: list[dict[str, Any]], **matches: Any) -> dict[str, Any] | None:
    for record in records:
        if all(str(record.get(key)) == str(value) for key, value in matches.items()):
            return record
    return None


def write_run_outputs(
    *,
    output_dir: Path,
    agents: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    agents_df = pd.DataFrame(agents)
    evaluations_df = pd.DataFrame(evaluations)
    scores_df = pd.DataFrame(scores)

    agents_df.to_csv(output_dir / "synthetic_agents.csv", index=False)
    evaluations_df.to_csv(output_dir / "synthetic_evaluations.csv", index=False)
    scores_df.to_csv(output_dir / "similarity_scores.csv", index=False)

    matrix = pd.DataFrame(
        columns=[
            "agent_type",
            "analysis_model",
            "n",
            "mae",
            "rmse",
            "mean_liking",
            "mean_predicted_liking",
        ]
    )
    if not scores_df.empty:
        matrix = summarize_matrix(scores_df)
    matrix.to_csv(output_dir / "matrix_summary.csv", index=False)

    html_path = output_dir / "simulation_smoke_report.html"
    if not agents_df.empty and not evaluations_df.empty and not scores_df.empty:
        write_html_report(
            output_path=html_path,
            matrix=matrix,
            agents=agents_df,
            evaluations=evaluations_df,
            scores=scores_df,
        )

    metadata = {
        **metadata,
        "html_report": str(html_path),
        "rows": {
            "agents": int(len(agents_df)),
            "evaluations": int(len(evaluations_df)),
            "scores": int(len(scores_df)),
            "matrix": int(len(matrix)),
        },
    }
    (output_dir / "simulation_metadata.json").write_text(json.dumps(metadata, indent=2))


def agent_profile_prompt(agent_id: str, agent_type: str) -> str:
    config = GENERATOR_MODELS[agent_type]
    return f"""
Create one synthetic French cooked-ham consumer for a methodological simulation.

Agent id: {agent_id}
Agent type: {config["label"]}
Decision style: {config["description"]}

System 1 agents should sound intuitive, sensory, affective, and fast.
System 2 agents should sound deliberate, comparative, analytical, and explicit about trade-offs.

Return JSON only:
{{
  "agent_id": "{agent_id}",
  "agent_type": "{agent_type}",
  "decision_style_en": "one sentence",
  "preference_summary_en": "two short sentences",
  "ideal_visual_fr": "French free comment about ideal visual appearance",
  "ideal_texture_fr": "French free comment about ideal texture in mouth",
  "ideal_flavor_fr": "French free comment about ideal flavor",
  "salt_preference": "low, moderate, or high",
  "texture_priority": "low, moderate, or high",
  "flavor_priority": "low, moderate, or high"
}}
""".strip()


def reaction_prompt(agent: dict[str, Any], product_profile: dict[str, str]) -> str:
    return f"""
Simulate this consumer evaluating a cooked ham at home.

Consumer profile:
{json.dumps(agent, ensure_ascii=False)}

Product evidence from real consumer comments:
Product: {product_profile["Product"]}
Visual evidence: {product_profile["visual_evidence"]}
Texture evidence: {product_profile["texture_evidence"]}
Flavor evidence: {product_profile["flavor_evidence"]}

Write the consumer's actual free comments in French. Make the comments plausible and sensory-specific.
Then assign a liking score from 0 to 10.

Return JSON only:
{{
  "actual_visual_fr": "French free comment",
  "actual_texture_fr": "French free comment",
  "actual_flavor_fr": "French free comment",
  "liking": 0.0,
  "liking_rationale_en": "one sentence"
}}
""".strip()


def similarity_prompt(agent: dict[str, Any], evaluation: dict[str, Any]) -> str:
    return f"""
You are scoring sensory similarity for a cooked-ham consumer simulation.

Compare the ACTUAL ham experience with the same consumer's IDEAL ham.
Use a 1 to 6 scale:
1 = extremely different
2 = very different
3 = somewhat different
4 = somewhat similar
5 = very similar
6 = extremely similar

IDEAL:
Visual: {agent["ideal_visual_fr"]}
Texture: {agent["ideal_texture_fr"]}
Flavor: {agent["ideal_flavor_fr"]}

ACTUAL:
Visual: {evaluation["actual_visual_fr"]}
Texture: {evaluation["actual_texture_fr"]}
Flavor: {evaluation["actual_flavor_fr"]}

Return JSON only:
{{
  "visual_similarity": 1,
  "texture_similarity": 1,
  "flavor_similarity": 1
}}
""".strip()


def coerce_similarity_scores(data: dict[str, Any]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for key in ["visual_similarity", "texture_similarity", "flavor_similarity"]:
        value = int(data[key])
        if not 1 <= value <= 6:
            raise ValueError(f"{key} out of range: {value}")
        parsed[key] = value
    return parsed


def write_html_report(
    *,
    output_path: Path,
    matrix: pd.DataFrame,
    agents: pd.DataFrame,
    evaluations: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        frame = frame.copy()
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body_rows = []
        for _, row in frame[columns].iterrows():
            cells = "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in columns)
            body_rows.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agent Type Simulation Smoke Test</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; background: #f7f6f2; color: #1a2721; }}
h1 {{ font-family: Georgia, serif; font-style: italic; font-size: 42px; font-weight: 500; }}
h2 {{ margin-top: 34px; }}
p {{ max-width: 920px; color: #4a5d53; }}
table {{ border-collapse: collapse; margin: 14px 0 24px; min-width: 720px; background: #fffdf7; }}
th, td {{ border-bottom: 1px solid #d8d2c2; padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ color: #4a5d53; font-size: 12px; text-transform: uppercase; }}
.note {{ border-left: 4px solid #c05a45; padding-left: 12px; color: #1a2721; }}
</style>
</head>
<body>
<h1>Agent type simulation smoke test</h1>
<p class="note">This is a tiny pipeline smoke test. It validates model routing and output shape. It is not evidence for the scientific hypothesis yet.</p>
<h2>2 x 2 matrix</h2>
{table(matrix.round(3), ["agent_type", "analysis_model", "n", "mae", "rmse", "mean_liking", "mean_predicted_liking"])}
<h2>Agents</h2>
{table(agents, ["agent_id", "agent_type", "generator_model", "salt_preference", "texture_priority", "flavor_priority"])}
<h2>Evaluations</h2>
{table(evaluations.round(2), ["agent_id", "agent_type", "Product", "liking", "generator_model"])}
<h2>Similarity scores</h2>
{table(scores.round(3), ["agent_id", "agent_type", "Product", "analysis_model", "visual_similarity", "texture_similarity", "flavor_similarity", "predicted_liking"])}
</body>
</html>
"""
    output_path.write_text(html_text)


def run_simulation(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = load_api_keys(args.keypool_env)
    if not keys:
        raise RuntimeError("No Gemini API keys found.")

    source = load_ham_home_use_data(args.data_xlsx)
    product_ids = resolve_product_ids(
        source,
        count=args.products_per_agent,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    product_profiles = build_product_profiles(source, product_ids, seed=args.seed)

    agents, evaluations, scores = load_existing_records(args.output_dir, resume=args.resume)
    key_index = 0
    metadata = {
        "agents_per_type": args.agents_per_type,
        "products_per_agent": len(product_ids),
        "seed": args.seed,
        "generator_models": GENERATOR_MODELS,
        "analysis_models": ANALYSIS_MODELS,
        "product_ids": product_ids,
        "output_dir": str(args.output_dir),
        "complete": False,
    }
    write_run_outputs(
        output_dir=args.output_dir,
        agents=agents,
        evaluations=evaluations,
        scores=scores,
        metadata=metadata,
    )
    log(f"Using fixed products for all agents: {', '.join(product_ids)}")
    if args.resume:
        log(
            "Resuming with "
            f"{len(agents)} agents, {len(evaluations)} evaluations, {len(scores)} scores already on disk."
        )

    for agent_type in ["system1", "system2"]:
        model_config = GENERATOR_MODELS[agent_type]
        for number in range(args.agents_per_type):
            agent_id = f"{agent_type}_{number + 1:02d}"
            agent = find_record(agents, agent_id=agent_id)
            if agent is None:
                api_key = keys[key_index % len(keys)]
                key_index += 1
                agent = call_gemini_json(
                    model=model_config["model"],
                    thinking_config=model_config["thinking_config"],
                    prompt=agent_profile_prompt(agent_id, agent_type),
                    api_key=api_key,
                    temperature=args.temperature,
                    timeout_seconds=args.request_timeout,
                    attempts=args.attempts,
                )
                agent["generator_model"] = model_config["model"]
                agent["generator_thinking_config"] = json.dumps(model_config["thinking_config"])
                agents.append(agent)
                write_run_outputs(
                    output_dir=args.output_dir,
                    agents=agents,
                    evaluations=evaluations,
                    scores=scores,
                    metadata=metadata,
                )
                log(f"Generated {agent_id} with {model_config['model']}")
            else:
                log(f"Reusing {agent_id} from checkpoint")

            for product_profile in product_profiles:
                evaluation = find_record(
                    evaluations,
                    agent_id=agent_id,
                    Product=product_profile["Product"],
                )
                if evaluation is None:
                    api_key = keys[key_index % len(keys)]
                    key_index += 1
                    reaction = call_gemini_json(
                        model=model_config["model"],
                        thinking_config=model_config["thinking_config"],
                        prompt=reaction_prompt(agent, product_profile),
                        api_key=api_key,
                        temperature=args.temperature,
                        timeout_seconds=args.request_timeout,
                        attempts=args.attempts,
                    )
                    liking = float(reaction["liking"])
                    liking = max(0.0, min(10.0, liking))
                    evaluation = {
                        "agent_id": agent_id,
                        "agent_type": agent_type,
                        "Product": product_profile["Product"],
                        "liking": liking,
                        "generator_model": model_config["model"],
                        "actual_visual_fr": reaction["actual_visual_fr"],
                        "actual_texture_fr": reaction["actual_texture_fr"],
                        "actual_flavor_fr": reaction["actual_flavor_fr"],
                        "liking_rationale_en": reaction.get("liking_rationale_en", ""),
                    }
                    evaluations.append(evaluation)
                    write_run_outputs(
                        output_dir=args.output_dir,
                        agents=agents,
                        evaluations=evaluations,
                        scores=scores,
                        metadata=metadata,
                    )
                    log(f"  Simulated {agent_id} on {product_profile['Product']}: liking={liking:.2f}")
                else:
                    log(f"  Reusing {agent_id} on {product_profile['Product']} from checkpoint")

                for analysis_name, analysis_config in ANALYSIS_MODELS.items():
                    existing_score = find_record(
                        scores,
                        agent_id=agent_id,
                        Product=product_profile["Product"],
                        analysis_model=analysis_name,
                    )
                    if existing_score is not None:
                        log(f"    Reusing {analysis_name} score from checkpoint")
                        continue
                    api_key = keys[key_index % len(keys)]
                    key_index += 1
                    raw_score = call_gemini_json(
                        model=analysis_config["model"],
                        thinking_config=analysis_config["thinking_config"],
                        prompt=similarity_prompt(agent, evaluation),
                        api_key=api_key,
                        temperature=args.temperature,
                        timeout_seconds=args.request_timeout,
                        attempts=args.attempts,
                    )
                    similarity = coerce_similarity_scores(raw_score)
                    predicted = similarity_to_liking(
                        similarity["visual_similarity"],
                        similarity["texture_similarity"],
                        similarity["flavor_similarity"],
                    )
                    scores.append(
                        {
                            "agent_id": agent_id,
                            "agent_type": agent_type,
                            "Product": product_profile["Product"],
                            "analysis_model": analysis_name,
                            "analysis_model_id": analysis_config["model"],
                            "analysis_thinking_config": json.dumps(analysis_config["thinking_config"]),
                            "liking": liking,
                            "predicted_liking": predicted,
                            **similarity,
                        }
                    )
                    write_run_outputs(
                        output_dir=args.output_dir,
                        agents=agents,
                        evaluations=evaluations,
                        scores=scores,
                        metadata=metadata,
                    )
                    log(f"    Scored with {analysis_name}: predicted={predicted:.2f}")

    metadata["complete"] = True
    write_run_outputs(
        output_dir=args.output_dir,
        agents=agents,
        evaluations=evaluations,
        scores=scores,
        metadata=metadata,
    )
    return json.loads((args.output_dir / "simulation_metadata.json").read_text())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-xlsx", default=DEFAULT_DATA_XLSX, type=Path)
    parser.add_argument("--keypool-env", default=DEFAULT_KEYPOOL_ENV, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--agents-per-type", default=1, type=int)
    parser.add_argument("--products-per-agent", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--request-timeout", default=120.0, type=float)
    parser.add_argument("--attempts", default=3, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    args = parse_args(argv or sys.argv[1:])
    metadata = run_simulation(args)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
