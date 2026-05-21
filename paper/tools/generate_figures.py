#!/usr/bin/env python3
"""Generate publication-style figures for the Sensometrics manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
if (ROOT / "data" / "grid" / "feature_importance_results.csv").exists():
    GRID_DATA = ROOT / "data" / "grid"
    TOPIC_DATA = ROOT / "data" / "topic_level"
else:
    GRID_DATA = ROOT / "Embeddings and Liking"
    TOPIC_DATA = ROOT / "Embeddings and Liking" / "topic_level_analysis"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1A2721"
MUTED = "#4A5D53"
ACCENT = "#C05A45"
SAND = "#EAE6DB"
GREY = "#8A938D"
LIGHT = "#F7F6F2"


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


GRID_RESULTS = GRID_DATA / "grid_comparison_results.csv"


def load_grid_results() -> pd.DataFrame:
    df = pd.read_csv(GRID_RESULTS)
    required = {"config_id", "model", "scale", "temperature", "r2", "mae", "llm_time", "errors", "n_valid"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{GRID_RESULTS} is missing columns: {sorted(missing)}")
    if "error_rate" not in df.columns:
        df["error_rate"] = df["errors"] / df["n_valid"]
    return df


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )


def draw_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.45))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)

    boxes = [
        (0.35, 0.95, 1.85, 1.05, "Actual and\nideal\ncomments"),
        (2.75, 0.95, 1.85, 1.05, "LLM\nalignment\nscoring"),
        (5.05, 0.95, 1.55, 1.05, "Visual\nTexture\nFlavor"),
        (7.1, 0.95, 1.75, 1.05, "TabPFN\nregression"),
        (9.35, 0.95, 1.85, 1.05, "Held-out\nliking"),
    ]
    for x, y, w, h, label in boxes:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=LIGHT,
            edgecolor=INK,
            linewidth=0.9,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            linespacing=1.1,
            color=INK,
        )

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.08, y),
                (x2 - 0.08, y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color=MUTED,
            )
        )

    ax.text(
        6,
        2.55,
        "Direct LLM alignment scoring converts paired Free-Comment text into compact predictors.",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "fig1_workflow")


def draw_model_grid() -> None:
    df = load_grid_results()
    order = ["Gemini 2.5 Flash Lite", "Gemini 3 Flash low", "Gemini 3 Flash minimal"]
    colors = {
        "Gemini 2.5 Flash Lite": ACCENT,
        "Gemini 3 Flash low": MUTED,
        "Gemini 3 Flash minimal": GREY,
    }
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for ax, metric, ylabel in [(axes[0], "r2", "$R^2$"), (axes[1], "mae", "MAE")]:
        for i, model in enumerate(order):
            vals = df.loc[df["model"] == model, metric].to_numpy()
            xs = np.full(len(vals), i) + rng.normal(0, 0.04, len(vals))
            ax.scatter(xs, vals, s=32, color=colors[model], alpha=0.85, edgecolor="white", linewidth=0.4)
            ax.plot([i - 0.22, i + 0.22], [vals.mean(), vals.mean()], color=INK, linewidth=1.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["2.5 Flash\nLite", "3 Flash\nlow", "3 Flash\nminimal"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_title("Prediction accuracy")
    axes[1].set_title("Prediction error")
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    save(fig, "fig2_model_grid")


def draw_operations() -> None:
    df = load_grid_results().set_index("config_id")
    configs = ["Flash Lite_6pt_t07", "G3 Flash (low)_6pt_t07", "G3 Flash (minimal)_6pt_t07"]
    labels = ["2.5 Flash Lite", "3 Flash low", "3 Flash minimal"]
    seconds = [df.loc[config, "llm_time"] for config in configs]
    errors = [100 * df.loc[config, "error_rate"] for config in configs]
    colors = [ACCENT, MUTED, GREY]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].bar(labels, seconds, color=colors, width=0.62)
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Runtime")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(labels, errors, color=colors, width=0.62)
    axes[1].set_ylabel("Rows with scoring errors (%)")
    axes[1].set_title("Scoring reliability")
    axes[1].grid(axis="y", alpha=0.2)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    save(fig, "fig3_operations")


def draw_modality_importance() -> None:
    imp = pd.read_csv(GRID_DATA / "feature_importance_results.csv")
    ours = imp.groupby("feature", as_index=False)["importance"].mean()
    ours["feature"] = ours["feature"].str.capitalize()
    ours_map = dict(zip(ours["feature"], ours["importance"]))
    mahieu = {"Flavor": 0.261, "Texture": 0.201, "Visual": 0.139}
    labels = ["Flavor", "Texture", "Visual"]
    ours_raw = np.array([ours_map[x] for x in labels])
    mahieu_raw = np.array([mahieu[x] for x in labels])
    ours_vals = 100 * ours_raw / ours_raw.sum()
    mahieu_vals = 100 * mahieu_raw / mahieu_raw.sum()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, mahieu_vals, width, label="Mahieu et al. modality signal", color=GREY)
    ax.bar(x + width / 2, ours_vals, width, label="LLM feature importance", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Within-method share (%)")
    ax.set_title("Both analyses put flavor first")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "fig4_modality_importance")


def draw_topic_rank() -> None:
    path = TOPIC_DATA / "topic_level_contrast_points.csv"
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7.4, 5.9))
    color_map = {"Flavor": ACCENT, "Texture": MUTED, "Visual": GREY}

    label_map = {
        "Off-notes and aftertaste": "Off-notes",
        "Blandness and intensity": "Blandness",
        "Fibrous pieces": "Fibrous",
        "Overall texture match": "Overall texture",
        "Firmness and rubberiness": "Firmness",
        "Overall flavor match": "Overall flavor",
        "Dryness and juiciness": "Dryness",
        "Smoky and spiced notes": "Smoky/spiced",
        "Ham taste": "Ham taste",
        "Moisture and shine": "Moisture",
        "Slice structure": "Slice structure",
        "Color and pinkness": "Color",
        "Saltiness": "Saltiness",
        "Tenderness and softness": "Tenderness",
        "Overall visual match": "Overall visual",
        "Fat and lean appearance": "Fat/lean",
        "Thickness and chew": "Thickness",
    }

    for _, row in df.iterrows():
        color = color_map.get(row["modality"], INK)
        label = label_map.get(row["label"], row["label"])
        ax.text(
            row["mahieu_rank"],
            row["our_rank"],
            label,
            ha="center",
            va="center",
            fontsize=6.6,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.18,rounding_size=0.04",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.6,
                "alpha": 0.95,
            },
        )
    lim = [0.0, 18.0]
    ax.plot(lim, lim, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    rank_ticks = list(range(1, 18, 2))
    ax.set_xticks(rank_ticks)
    ax.set_yticks(rank_ticks)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Mahieu et al. topic rank")
    ax.set_ylabel("LLM topic rank")
    ax.set_title("Rank-order agreement across mapped topic families", fontsize=12, pad=16)
    legend_handles = [
        plt.Line2D([0], [0], color=color_map["Flavor"], lw=2, label="Flavor"),
        plt.Line2D([0], [0], color=color_map["Texture"], lw=2, label="Texture"),
        plt.Line2D([0], [0], color=color_map["Visual"], lw=2, label="Visual"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=8)
    ax.grid(alpha=0.18)
    save(fig, "fig5_topic_rank")


def write_summary() -> None:
    summary = {
        "figures": [
            "fig1_workflow",
            "fig2_model_grid",
            "fig3_operations",
            "fig4_modality_importance",
            "fig5_topic_rank",
        ],
        "source_data": [
            repo_path(GRID_DATA / "bootstrap_family_win_probabilities.csv"),
            repo_path(GRID_DATA / "bootstrap_pairwise_probabilities.csv"),
            repo_path(GRID_DATA / "bootstrap_top_config_results.csv"),
            repo_path(GRID_DATA / "downstream_model_comparison_results.csv"),
            repo_path(GRID_RESULTS),
            repo_path(GRID_DATA / "feature_importance_results.csv"),
            repo_path(TOPIC_DATA / "topic_level_contrast_points.csv"),
        ],
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    draw_workflow()
    draw_model_grid()
    draw_operations()
    draw_modality_importance()
    draw_topic_rank()
    write_summary()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
