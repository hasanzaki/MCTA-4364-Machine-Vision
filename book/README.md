# The Book (Quarto project)

This folder holds the textbook manuscript, built with [Quarto](https://quarto.org) from a single set of `.qmd` sources into HTML, PDF and EPUB.

## Structure

```text
book/
├── _quarto.yml            # Book configuration (chapters, formats, bibliography)
├── index.qmd              # Preface
├── references.bib         # Bibliography
├── custom.scss            # Styling
├── chapters/
│   └── 07-segmentation.qmd   # Prototype chapter
└── figures/               # Figure assets (most figures are generated in-code)
```

See `PROPOSAL.md` for the full plan and table of contents.

## Build

1. Install Quarto: <https://quarto.org/docs/get-started/>
2. Install Python dependencies (the chapters execute code):
   ```powershell
   pip install numpy matplotlib opencv-python scikit-image
   ```
3. From this `book/` folder:
   ```powershell
   quarto preview      # live preview at http://localhost:xxxx
   quarto render       # build all formats
   quarto render --to html
   quarto render --to pdf
   ```

The rendered site is written to `book/_book/` (ignored by Git).

## Conventions

- Every figure must be **generated in code** (reproducible) or openly licensed with attribution.
- Use `#| label: fig-...` and `#| fig-cap:` so figures can be cross-referenced with `@fig-...`.
- Add references to `references.bib` and cite with `@key`.
- Follow the chapter template: learning outcomes, motivation, theory, worked example, Colab lab, misconceptions, summary, exercises, further reading.
