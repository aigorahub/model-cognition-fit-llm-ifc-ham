# Data dictionary

## `data/raw/dataset.xlsx`

Original cooked-ham workbook used for the analysis.

Relevant sheets:

- `product sensory properties`: product-level consumer evaluations, Free-Comment descriptions and liking scores.
- `consumer questionnaire (home)`: consumer-level ideal ham descriptions used to pair each actual product evaluation with the same consumer's ideal.

Core columns used by the scripts:

- `Consumer`: consumer identifier.
- `Product`: product identifier.
- `Liking`: 0 to 10 liking score.
- `DescriptionVisual`: actual-product visual Free-Comment.
- `DescriptionTexture`: actual-product texture Free-Comment.
- `DescriptionFlavor`: actual-product flavor Free-Comment.
- `IdealVisual`: ideal-product visual Free-Comment.
- `IdealTexture`: ideal-product texture Free-Comment.
- `IdealFlavor`: ideal-product flavor Free-Comment.

## `data/grid/`

Saved outputs from the model-grid comparison. The primary grid crossed model family, Likert scale granularity and generation temperature.

## `data/topic_level/`

Saved outputs from the topic-level LLM alignment analysis. Topic scores compare actual and ideal comments for 17 sensory topic families.

## `data/agent_type_simulation_smoke_10x6/`

Saved outputs from a small synthetic-agent smoke test. It should be treated as hypothesis-generating rather than confirmatory.
