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

## Troubleshooting

**Quarto launches the wrong kernel (Windows).** If you have several Jupyter kernels installed (e.g. old conda environments), Quarto may fail with `[WinError 2] The system cannot find the file specified` because it selected a kernel whose interpreter no longer exists. Fix by ensuring a valid kernel is found first:

```powershell
# list kernels and find broken ones
jupyter kernelspec list
# create a valid kernel that sorts first (workaround)
python -m ipykernel install --user --name 000-python --display-name "Python (book)"
```

Then re-render with `quarto render`.

**PDF output fails.** The PDF format needs a LaTeX installation (TinyTeX or MiKTeX). HTML and EPUB do not. Install TinyTeX with `quarto install tinytex` if you need PDF.

## Conventions

- Every figure must be **generated in code** (reproducible) or openly licensed with attribution.
- Use `#| label: fig-...` and `#| fig-cap:` so figures can be cross-referenced with `@fig-...`.
- Add references to `references.bib` and cite with `@key`.
- Follow the chapter template: learning outcomes, motivation, theory, worked example, Colab lab, misconceptions, summary, exercises, further reading.
