# AGENTS.md — Repository Guide and Conventions

This file documents how the **MCTA 4364 / MCTE 4323 Machine Vision** teaching repository is organized, and the rules to follow when editing it. Read this before making changes.

## 1. Purpose

A public teaching repository for a 14-week undergraduate Machine Vision course. It contains lecture notes (Markdown), hands-on Google Colab notebooks, a syllabus, projects, resources, and validation tooling.

## 2. Repository layout

```text
README.md                      Landing page + modernized 14-week syllabus
LICENSE                        MIT
syllabus/                      CLOs, assessment plan, weekly plan (CSV)
lectures/NN_topic/README.md    Concise weekly notes (renders on GitHub)
notebooks/NN_topic/NN_topic.ipynb   Enriched Colab lab
notebooks/README.md            Index with "Open in Colab" badges
projects/                      Mini-project guidelines (+ datasets/, gitignored)
assessments/                   Public sample revision material
assessments/private/           Exams/grades — NEVER committed
resources/scripts/cvhelpers.py Shared notebook helpers
resources/images/ videos/ models/   Shared assets (videos/models gitignored)
setup/requirements.txt         Python dependencies
setup/local_setup.md           Local install guide
legacy/                        Original instructor notebooks (not student-facing)
.github/workflows/             CI
.github/scripts/               CI helper scripts
```

The week folders are numbered `01`–`14` and **must match** between `lectures/` and `notebooks/`.

## 3. Notebook structure (mandatory)

Every `notebooks/NN_topic/NN_topic.ipynb` must follow this order and naming:

1. Markdown: `# Week NN — Title`, `### Learning objectives` bullets, one-line context.
2. `## 1. Setup`:
   - **Code cell 1 (repository detection)** — exact block:
     ```python
     import os

     # Works in Colab (clones the repo) and locally or in CI (runs inside the repo)
     if not os.path.exists("resources/scripts/cvhelpers.py"):
         if not os.path.isdir("MCTA-4364-Machine-Vision"):
             !git clone https://github.com/hasanzaki/MCTA-4364-Machine-Vision.git
         %cd MCTA-4364-Machine-Vision
     ```
   - **Code cell 2 (install)** — must list every third-party package the notebook imports, and **always include `ipywidgets`**.
3. Topic sections: markdown explanation followed by a code cell. Reference existing images in `resources/images/` (do not hardcode absolute paths).
4. `## Visual summary` — a `concept_map([...])` code cell.
5. `## Interactive exploration` — an `ipywidgets` cell with sliders/dropdowns that call the shared `show()` helper.
6. `## Exercise (complete the code)` — with `# TODO` placeholders.
7. `## Challenge (independent)` — an open task.
8. `## Check your understanding (Q&A)` — markdown using collapsible blocks:
   ```html
   <details><summary><b>Q1. ...</b></summary>
   ...answer...
   </details>
   ```
9. `## Further reading & self-exploration` — links to stable sources (OpenCV docs, scikit-image, PyTorch, Ultralytics, Hugging Face, Wikipedia, papers).
10. `## Key takeaways` — short bullet list.

Notebook metadata must use `kernelspec name: python3` and `nbformat 4.5` (with cell `id`s).

## 4. Shared helpers

Use the helpers in `resources/scripts/cvhelpers.py` instead of redefining them:

```python
import sys
sys.path.append("resources/scripts")
from cvhelpers import show, concept_map, image_grid, plot_histogram
```

- `show(*images, titles=[...])` — displays BGR/gray images via Matplotlib (works headless).
- `concept_map([...])` — draws a vertical flow infographic.

## 5. Do NOT (Colab/headless safety)

- No `cv2.imshow`, `cv2.waitKey`, `cv2.destroyAllWindows`, or webcam (`cv2.VideoCapture(0)`) in `notebooks/`. Use `show()` or Matplotlib. Wrap any optional/video/model call in `try/except`.
- No absolute local paths (`D:\...`, `c:\Users\...`).
- No `input()` calls.

## 6. Privacy and large files — never commit

`.gitignore` blocks these; do not override it:

- `assessments/private/`, `*.xlsx`, `*.docx` (grade/admin), `*classlist*`, `*VALIDATION_SHEET*`, `*CAMRELEASE*`.
- Datasets: `projects/datasets/`.
- Model weights: `*.pt`, `*.pth`, `*.onnx`, `*.h5`.
- Video: `*.mp4`, `*.avi`, `*.mov`.
- Oversized lecture decks with embedded video (listed explicitly in `.gitignore`).

Public Markdown/PDF exceptions are whitelisted at the bottom of `.gitignore` (e.g. `!projects/*.pdf`). Add new public PDFs by whitelisting them there.

## 7. Validate before pushing

Run these locally before committing:

```powershell
# 1. Structure validation (fast)
python .github/scripts/validate_notebooks.py

# 2. Execute a notebook end-to-end (repo root as cwd)
python -m papermill notebooks/07_segmentation/07_segmentation.ipynb `
  "$env:TEMP\out.ipynb" --cwd . --kernel python3 --execution-timeout 900
```

CI (`.github/workflows/validate-notebooks.yml`) runs on every push/PR:
- **Notebook structure** — `validate_notebooks.py`.
- **Execute all notebooks** — installs CPU PyTorch, torchvision, Ultralytics, Transformers, then runs every notebook with Papermill.

If you change a notebook, make sure both jobs stay green.

## 8. Adding or changing content

- **New week:** create matching `lectures/NN_topic/README.md` and `notebooks/NN_topic/NN_topic.ipynb` using the structure in Section 3; add a Colab badge row to `notebooks/README.md`; update `syllabus/weekly-plan.csv` if topics change.
- **Lecture notes:** keep the sections used in existing files (official block, objectives, key concepts, key formulas, common misconceptions, lab link, further reading).
- **Helpers:** add reusable plotting/utility functions to `cvhelpers.py`, not inline copies.
- **Images/assets:** add to `resources/images/` (small) and reference by relative path.

## 9. Commit style

- Small, focused commits; imperative mood (e.g. `Fix HoughLinesP unpacking for newer OpenCV`).
- Do not commit generated outputs (`/tmp/executed`, executed notebooks) or downloaded weights (`yolo11n.pt`, `mobile_sam.pt`, `*.pth`).
- Never commit secrets or tokens.

## 10. Quick facts

- Target Python: 3.10–3.12 (CI uses 3.11).
- Dependencies: `setup/requirements.txt`.
- Colab is the primary runtime; GPU recommended for notebooks 09–11 and 14, but all run on CPU.
- The accredited syllabus is fixed; modernize **materials** (not the official topic list).
