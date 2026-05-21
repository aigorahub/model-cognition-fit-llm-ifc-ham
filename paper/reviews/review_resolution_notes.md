# Review resolution notes

Two review passes were run before opening the pull request:

- A Codex subagent reviewed the paper package for reproducibility, formatting, and publication readiness.
- `gemini-3.5-flash` reviewed `paper/manuscript.pdf` through the Gemini API with low thinking.

Actions taken from those reviews:

- Replaced the public 9-row grid artifact with the 27-row model x scale x temperature grid used by the manuscript.
- Updated figure generation so Figures 2 and 3 read `data/grid/grid_comparison_results.csv` instead of hard-coded constants.
- Added `data/grid/bootstrap_top_config_results.csv` for the selected bootstrap summaries cited in the manuscript.
- Added `data/grid/bootstrap_pairwise_probabilities.csv` and `data/grid/bootstrap_family_win_probabilities.csv` for the pairwise and family-level bootstrap claims.
- Added `data/grid/downstream_model_comparison_results.csv` for the downstream learner table.
- Changed the primary results table to a controlled comparison at the same six-point scale and temperature 0.7.
- Renamed parse-error columns to row-level scoring errors, matching the saved grid artifact.
- Normalized Figure 4 within method so the modality comparison no longer implies shared raw units.
- Simplified Figure 3 labels and moved configuration details into the caption and table.
- Removed internal draft notes from the limitations section and reframed them as caveats.
- Fixed Word table generation so table headers and first rows are preserved.
- Replaced generic LaTeX equation flattening in the Word build with readable equation text for the equations used in the manuscript.
