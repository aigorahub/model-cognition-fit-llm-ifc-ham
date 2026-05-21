# Food Quality and Preference formatting check

Sources checked:

- Food Quality and Preference guide for authors, Elsevier:
  https://www.elsevier.com/journals/food-quality-and-preference/0950-3293/guide-for-authors
- Mahieu et al. cooked-ham article stored locally at:
  `/Users/johnennis/aigora/dev/sensometrics-2026-posters/Embeddings and Liking/Ham Paper.pdf`

Working choices applied in this manuscript:

- Main manuscript source is `paper/manuscript.tex`.
- Article structure follows the Mahieu Food Quality and Preference article:
  Introduction; Material and methods; Results; Discussion; conclusion and
  declarations.
- Abstract is kept below 250 words.
- Keywords are limited to six.
- Highlights are stored as a separate editable file at `paper/highlights.txt`.
  Each highlight is below 85 characters.
- Tables are written as editable LaTeX tables rather than figure images.
- Figures are generated from source data with reusable code in
  `paper/tools/generate_figures.py`.
- The AI-use declaration is included as a separate declaration section.

Scoring explanation included in the paper:

- For each consumer-product row, the LLM received six Free-Comment fields:
  ideal visual, ideal texture, ideal flavor, actual visual, actual texture,
  and actual flavor.
- The model was asked to act as a sensory and consumer scientist, focus only on
  sensory comments about the ham itself, and account for the fact that the
  comments were written in French.
- In the primary analysis, it returned three integer alignment scores: visual,
  texture, and flavor.
- The six-point scale ran from 1, extremely different, to 6, extremely similar.
- Liking was not shown to the model during scoring.
- The response had to be JSON with exactly `visual`, `texture`, and `flavor`
  fields. Scores outside the valid range or missing fields were treated as
  parse failures.
