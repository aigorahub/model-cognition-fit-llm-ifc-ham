# Topic-Level Comparison With Mahieu et al. (2022)

Generated on 2026-05-21 from the Sensometrics 2026 cooked-ham analysis workspace.

## Purpose

This note documents the exploratory topic-level analysis used to compare our LLM-based sensory scoring pipeline with the original Mahieu, Worch, and Schlich cooked-ham analysis. It is intended as source material for a future paper section, not as final manuscript text.

The main question was whether direct LLM scoring of actual-versus-ideal sensory alignment recovers the broad sensory structure reported by Mahieu et al. (2022).

## Data

Source workbook:

`Embeddings and Liking/data/dataset.xlsx`

Rows used:

| Quantity | Value |
|---|---:|
| Source evaluations | 2,758 |
| Valid topic-scored evaluations | 2,757 |
| Products | 30 |
| Consumers | 483 |
| Gemini parse errors | 1 |

The single parse-error row was excluded from downstream topic-level summaries.

## LLM Topic Scoring

Script:

`scripts/run_topic_level_analysis.py`

LLM configuration:

| Setting | Value |
|---|---|
| Model | `gemini-flash-lite-latest` |
| Temperature | 0.7 |
| Scale | 6-point Likert alignment scale |
| Prompt unit | One consumer-product evaluation row |
| Task | Compare actual ham comments to that consumer's ideal ham comments |

Each row was scored on 17 sensory topic-alignment variables:

| Modality | Topics |
|---|---|
| Visual | overall visual match; color and pinkness; fat and lean appearance; surface moisture and shine; slice structure and homogeneity |
| Texture | overall texture match; tenderness and softness; firmness and rubberiness; dryness and juiciness; fibrous/stringy pieces; thickness and chew |
| Flavor | overall flavor match; saltiness; ham taste; aromatic/smoky/spiced notes; blandness/insipid intensity; off-notes and aftertaste |

Each topic score estimates how closely the actual product matched the consumer's ideal product for that topic. This is not a descriptor citation count.

## Mahieu et al. Reference Metrics

Mahieu et al. processed Free-Comment and Ideal-Free-Comment data by cleaning and lemmatizing French text, grouping descriptors into latent descriptors, retaining descriptors cited in at least 5 percent of evaluations for at least one product, and cross-tabulating descriptor presence by consumer and product.

They estimated descriptor effects on liking with a mixed linear model:

`liking ~ product + descriptor presence + consumer random effect`

Descriptor loadings from this model were interpreted as drivers of liking. They also ran Multiple-Response Correspondence Analysis (MR-CA) by modality on product-by-descriptor citation proportions, projected the ideal product as supplementary, and projected mean liking as a supplementary variable.

For this comparison, Mahieu driver strengths were coded from the published figure and processed paper notes. They are used for structural comparison of modality and topic order, not as exact coefficient re-estimates.

## Why We Compare Strength Rather Than Sign

The original paper's descriptor model and our LLM score have different meanings.

Mahieu et al. model whether a descriptor is present. A descriptor such as `F_insipid` has a negative loading because citing "insipid" is bad for liking.

Our LLM score measures actual-versus-ideal alignment. If a consumer dislikes insipid ham and the actual product is not insipid, the alignment score can increase with liking.

For that reason, the main topic-level contrast uses absolute driver strength:

| Axis | Metric |
|---|---|
| Mahieu et al. | Average absolute mixed-model loading for the closest descriptor family, normalized by the strongest descriptor |
| Our pipeline | Absolute Spearman correlation between LLM topic-alignment score and liking, normalized by the strongest topic |

This avoids forcing an inappropriate signed comparison.

## Modality-Level Comparison

Mahieu et al.'s reported modality hierarchy:

`Flavor > Texture > Visual`

Our recovered modality hierarchy:

`Flavor > Texture > Visual`

Rank agreement:

| Metric | Value |
|---|---:|
| Spearman rank correlation across modalities | 1.000 |

This is the cleanest high-level result. The automated topic scoring recovered the same broad sensory hierarchy: flavor carries the strongest liking signal, texture is second, and visual appearance is weaker.

## Topic-Level Rank Comparison

We mapped each LLM topic to the closest Mahieu descriptor family and compared rank order. There were 17 matched topic families.

| Metric | Value |
|---|---:|
| Spearman rank correlation | 0.715 |
| Spearman p-value | 0.00125 |
| Kendall tau | 0.519 |
| Kendall p-value | 0.00387 |
| Top-5 overlap | 3 of 5 |
| Top-8 overlap | 6 of 8 |
| Top-10 overlap | 9 of 10 |

Interpretation: the topic-order agreement is substantial, especially considering that Mahieu et al. ranked descriptor effects while our analysis ranked actual-versus-ideal topic alignment.

## Matched Topic Rank Table

| Topic | Modality | Mahieu rank | Our rank | Shift |
|---|---|---:|---:|---:|
| Off-notes and aftertaste | Flavor | 1 | 3 | 2 |
| Blandness and intensity | Flavor | 1 | 4 | 3 |
| Fibrous pieces | Texture | 3 | 6 | 3 |
| Overall texture match | Texture | 4 | 5 | 1 |
| Firmness and rubberiness | Texture | 5 | 10 | 5 |
| Overall flavor match | Flavor | 6 | 2 | -4 |
| Dryness and juiciness | Texture | 7 | 8 | 1 |
| Smoky and spiced notes | Flavor | 8 | 9 | 1 |
| Ham taste | Flavor | 9 | 1 | -8 |
| Moisture and shine | Visual | 10 | 14 | 4 |
| Slice structure | Visual | 11 | 12 | 1 |
| Color and pinkness | Visual | 12 | 17 | 5 |
| Saltiness | Flavor | 12 | 13 | 1 |
| Tenderness and softness | Texture | 14 | 7 | -7 |
| Overall visual match | Visual | 15 | 11 | -4 |
| Fat and lean appearance | Visual | 16 | 16 | 0 |
| Thickness and chew | Texture | 17 | 15 | -2 |

Positive shift means our topic rank is lower priority than Mahieu et al. Negative shift means our topic rank is higher priority than Mahieu et al.

Largest upward shifts in our analysis:

| Topic | Mahieu rank | Our rank | Shift |
|---|---:|---:|---:|
| Ham taste | 9 | 1 | -8 |
| Tenderness and softness | 14 | 7 | -7 |
| Overall flavor match | 6 | 2 | -4 |
| Overall visual match | 15 | 11 | -4 |

Largest downward shifts in our analysis:

| Topic | Mahieu rank | Our rank | Shift |
|---|---:|---:|---:|
| Firmness and rubberiness | 5 | 10 | 5 |
| Color and pinkness | 12 | 17 | 5 |
| Moisture and shine | 10 | 14 | 4 |
| Fibrous pieces | 3 | 6 | 3 |
| Blandness and intensity | 1 | 4 | 3 |

## MR-CA Style Topic Map

Script:

`scripts/render_topic_level_mrca_report.py`

Because the workbook does not contain Mahieu et al.'s curated descriptor citation matrix, we could not rerun the original MR-CA exactly. Instead, we ran a correspondence-analysis analog on a product-by-topic matrix:

1. Average each LLM topic-alignment score by product.
2. Run correspondence analysis on the resulting 30 product by 17 topic matrix.
3. Project product mean liking against the first two CA dimensions using Spearman correlation.

Results:

| Metric | Value |
|---|---:|
| CA Dimension 1 correlation with product mean liking | 0.273 |
| CA Dimension 2 correlation with product mean liking | 0.319 |

This map is useful as a visual diagnostic, but it should not be described as a literal replication of Mahieu et al.'s MR-CA. It is an MR-CA-style topic-alignment map.

## Predictive Fallback Used In The Temporary Report

The full topic-level TabPFN run on the Mini hit memory limits on MPS. A CPU attempt was started with the large-dataset guard enabled, but it ran for more than seven minutes without producing artifacts, so it was stopped.

For the temporary HTML report only, a local Gradient Boosting fallback was used to show a held-out topic-level predictive metric on the same consumer-group split:

| Metric | Value |
|---|---:|
| Train rows | 2,219 |
| Holdout rows | 538 |
| R2 | 0.470 |
| MAE | 1.367 |
| RMSE | 1.767 |

This fallback metric should not replace the main presentation result, which remains the 3-score Flash Lite plus TabPFN result already reported in the deck. It is included here only to inspect the topic-level feature set.

## Artifacts

Temporary HTML report:

`Embeddings and Liking/topic_level_analysis/topic_level_mrca_report.html`

Generated topic-level data and summaries:

| Artifact | Purpose |
|---|---|
| `Embeddings and Liking/topic_level_analysis/topic_level_flash_lite_scores.csv` | Row-level LLM topic scores |
| `Embeddings and Liking/topic_level_analysis/topic_level_mrca_summary.json` | Key numeric summary |
| `Embeddings and Liking/topic_level_analysis/topic_level_contrast_points.csv` | Topic bridge metrics and ranks |
| `Embeddings and Liking/topic_level_analysis/topic_level_row_correlations.csv` | Row-level topic-liking correlations |
| `Embeddings and Liking/topic_level_analysis/topic_level_product_correlations.csv` | Product-level topic-liking correlations |
| `Embeddings and Liking/topic_level_analysis/topic_level_gradient_boosting_importance.csv` | Fallback model importance |

The `topic_level_analysis` folder is ignored by git because it contains generated outputs. The reproducible source scripts are tracked separately.

## Reproduction Commands

Run or resume LLM topic scoring:

```bash
talk/.venv/bin/python scripts/run_topic_level_analysis.py --resume --retry-errors --max-workers 32 --progress-every 100
```

Render the temporary HTML report:

```bash
talk/.venv/bin/python scripts/render_topic_level_mrca_report.py --output-dir "Embeddings and Liking/topic_level_analysis"
```

Run tests:

```bash
talk/.venv/bin/python -m unittest tests/test_topic_level_analysis.py tests/test_topic_level_mrca_report.py
```

## Paper-Ready Claim

A conservative paper-ready statement would be:

> To compare the LLM-derived topic scores with the original cooked-ham analysis, we mapped each LLM topic to the closest descriptor families reported by Mahieu et al. (2022). Because descriptor-presence loadings and actual-versus-ideal alignment scores are not on the same scale, we compared normalized driver strength and rank order rather than raw signed coefficients. The LLM scoring recovered the same modality hierarchy as the original analysis, with flavor strongest, texture second, and visual appearance weakest. Across 17 matched topic families, the rank-order agreement was substantial, Spearman rho = 0.715, Kendall tau = 0.519, with 9 of the top 10 topic families overlapping. This suggests that direct LLM scoring preserved much of the sensory priority structure recovered by the original human-coded descriptor workflow.

## Limitations To Preserve In The Paper

1. The original Mahieu descriptor matrix was not available in this workspace, so the original MR-CA could not be reproduced exactly.
2. Mahieu driver loadings used here are coded from the published figure and processed paper notes. Treat them as structural comparison values rather than exact coefficient re-estimates.
3. Our scores measure alignment to the consumer's ideal, not descriptor presence. Signed effects are therefore not directly comparable.
4. The topic-level analysis is exploratory and was not the primary grid-search result in the Sensometrics deck.
5. The current topic list was hand-specified from the paper's descriptor families. A future version could score a larger descriptor inventory directly.
