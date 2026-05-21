# Is more thinking always better? Think again

Model-cognition fit in AI-supported sensory analysis.

This repository supports a working paper on model-cognition fit in LLM scoring of Ideal-Free-Comment sensory data. The case study uses the cooked-ham consumer dataset from Mahieu, Visalli and Schlich (2022), where French consumers described actual hams and their ideal ham using Free-Comment across visual appearance, texture and flavor.

The main empirical question is whether a fast, direct-response LLM is a better scoring instrument for sensory actual-versus-ideal alignment than a reasoning-oriented model. The current results show that Gemini 2.5 Flash Lite produced stronger liking prediction, lower runtime and fewer row-level scoring errors than tested Gemini 3 Flash reasoning configurations in the controlled six-point comparison.

## Repository contents

- `data/raw/`: cooked-ham source workbook and questionnaire used for the analysis.
- `data/grid/`: saved 27-configuration model-grid results and feature-importance outputs.
- `data/topic_level/`: topic-level LLM scores, MR-CA comparison artifacts and HTML report.
- `data/agent_type_simulation_smoke_10x6/`: small synthetic-agent smoke test artifacts.
- `scripts/`: reusable analysis and report-generation scripts.
- `tests/`: lightweight unit tests for parsing, reporting and simulation helpers.
- `paper/`: LaTeX manuscript, generated figures, journal highlights, PDF and DOCX outputs.

## Key outputs

- Paper PDF: `paper/manuscript.pdf`
- Paper Word file: `paper/manuscript.docx`
- Topic-level report: `data/topic_level/topic_level_mrca_report.html`
- Synthetic-agent smoke test report: `data/agent_type_simulation_smoke_10x6/agent_type_smoke_summary.html`

## Reproducing reports from saved artifacts

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run tests:

```bash
python3 -m unittest discover -s tests
```

Rebuild the topic-level HTML report from saved scores:

```bash
python3 scripts/render_topic_level_mrca_report.py
```

Rebuild the synthetic-agent smoke test report:

```bash
python3 scripts/render_agent_type_simulation_report.py
```

Rebuild the manuscript PDF and DOCX:

```bash
cd paper
bash tools/build.sh
```

The full LLM scoring scripts require Gemini API keys in the environment or in a key-pool file. The saved artifacts are included so that the reports and manuscript figures can be rebuilt without rerunning paid model calls.

## Data citation

The cooked-ham data come from:

Mahieu, B., Visalli, M., and Schlich, P. (2022). Identifying drivers of liking and characterizing the ideal product thanks to Free-Comment. *Food Quality and Preference*, 96, 104389. https://doi.org/10.1016/j.foodqual.2021.104389

Visalli, M., Loiseau, A.-L., Cordelle, S., Mahieu, B., and Schlich, P. (2024). A dataset of perception and preferences of French consumers for commercial cooked hams sampled according to their nutritional values and claims. *Data in Brief*, 54, 110549. https://doi.org/10.1016/j.dib.2024.110549

## Data and code availability statement

This repository contains the data, analysis scripts, saved model-scoring artifacts, generated reports, manuscript figures and manuscript build files for the paper.
