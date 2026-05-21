# Paper build notes

The canonical manuscript source is now `paper/manuscript.tex`, following the
same pattern as the Husk paper in `lsa-web-application/docs/paper`.

Build everything with:

```bash
cd paper
bash tools/build.sh
```

The build script:

1. regenerates manuscript figures in `paper/figures/`;
2. compiles `paper/manuscript.tex` with `pdflatex`;
3. copies the PDF to `paper/build/manuscript.pdf`;
4. creates a Word version at `paper/build/manuscript.docx`.
5. copies the journal highlights file to `paper/build/highlights.txt`.

Outputs:

- `paper/build/manuscript.pdf`
- `paper/build/manuscript.docx`
- `paper/build/highlights.txt`

Supporting scripts:

- `paper/tools/generate_figures.py`: builds the publication-style figures from
  saved CSV/JSON artifacts and the 27-grid constants from the talk.
- `paper/tools/build_docx.py`: creates the Word manuscript from the LaTeX source.
- `paper/tools/build.sh`: one-command build.

Reference verification:

- `paper/source_materials/reference_verification_report.md`
- `paper/source_materials/reference_verification.csv`
- `paper/notes/reference_verification_integration.md`

The earlier Markdown draft is retained at `paper/manuscript_draft.md` as a
development note, not the submission source.
