# Book Proposal

## Working title
**Machine Vision: From Pixels to Foundation Models — A Modern Engineering Approach**

Alternatives:
- *Applied Machine Vision: Classical and Deep Learning Methods for Engineers*
- *Machine Vision for Mechatronics: Principles, Python, and Practice*

**Author:** Dr. Hasan Zaki, Department of Mechatronics Engineering, Kulliyyah of Engineering, IIUM
**Format:** Open-access textbook, single source (Quarto) → HTML + PDF + EPUB, versioned releases with DOI
**Licence (proposed):** CC BY-NC-SA 4.0 (or CC BY 4.0 for widest reuse)
**Length:** ~300–340 pages, ~140,000 words, ~550 original figures
**Edition model:** living book — numbered releases (v1.0, v1.1, …) archived on Zenodo

---

## 1. Overview

This textbook is a modern, applied introduction to machine vision for undergraduate engineering students. It teaches the complete pipeline — acquisition, classical image processing, learning-based recognition, 2D/3D vision, and modern foundation models — through **reproducible Python and Google Colab laboratories**. It is written to be *used in a 14-week course*, not merely read: every chapter ships with runnable code, interactive demonstrations, worked numerical examples, and graded exercises.

The distinguishing premise: **the classical and the modern belong together.** Students first learn why thresholding, filtering and homography work, then see how deep detectors and promptable segmentation models generalise those ideas. This sequencing builds intuition that a deep-learning-only syllabus cannot.

## 2. Audience and prerequisites

- **Primary:** 3rd/4th-year undergraduate engineering students (mechatronics, electrical, mechanical, robotics).
- **Prerequisites:** introductory programming (Python), basic linear algebra and calculus, elementary probability.
- **Assumed environment:** Python with OpenCV, NumPy, Matplotlib, PyTorch, and Ultralytics; Google Colab as the zero-install default.

## 3. Positioning and market analysis

| Title | Orientation | Gap it leaves |
|---|---|---|
| Sonka, Hlavac & Boyle (2008) | Classical, encyclopaedic, math-heavy | Dated; no deep learning; not code-first |
| Szeliski (2022) | Modern CS, algorithms | Advanced/CS-oriented; not engineering-applied; dense |
| Prince (2012) | Probabilistic/learning | Heavy mathematics; not practical |
| Gonzalez & Woods (2018) | Digital image processing | Processing, not vision systems or deployment |
| Forsyth & Ponce (2012) | CS modern approach | Theory-first; limited labs |
| Nixon & Aguado (2019) | Feature extraction, engineering | Largely classical; limited modern DL/3D/VLM |

**Differentiators of this book**
1. **Engineering-applied, not CS-theoretical** — framed around instrumentation, inspection, robotics and metrology.
2. **Code-first and reproducible** — every figure and lab is generated and validated by CI.
3. **Integrated modern vision** — R-CNN, YOLO, U-Net, Mask R-CNN, SAM, open-vocabulary detection, VLMs, point clouds.
4. **Classical + modern in one narrative**, with explicit bridges between them.
5. **Assessment-aligned** — each chapter maps to course learning outcomes and includes a question bank.
6. **Open access** — free to students worldwide, citable via DOI, continuously updated.

## 4. Learning-outcome alignment

The book is designed backwards from the accredited course outcomes (CLOs):

| CLO | Chapters that develop it |
|---|---|
| CLO1 Develop vision systems with software tools | Every chapter's lab |
| CLO2 Apply modern vision algorithms | 5–7, 9–11 |
| CLO3 Apply 2D and 3D techniques | 3, 12, 13 |
| CLO4 Select appropriate techniques | 8–11, 14, plus comparison case studies |

Each chapter declares 4–6 measurable outcomes (Bloom-tagged) and an explicit CLO mapping.

## 5. Detailed table of contents

**Front matter**
- Preface; How to use this book (student / instructor tracks); Notation and conventions; Setting up Python and Colab.

### Part I — Foundations

**Chapter 1 — What Machine Vision Is** (~14 pp)
1.1 Images as measurements; 1.2 Image processing, computer vision, machine vision; 1.3 The vision pipeline; 1.4 Image geometry and coordinates; 1.5 Sampling and quantisation; 1.6 Levels of computation; 1.7 Road map and engineering case studies.
*Lab:* first image, array anatomy. *Exercises:* 12. *Figures:* 22.

**Chapter 2 — Image Representation and Colour** (~16 pp)
2.1 Digital images as arrays; 2.2 Resolution, contrast, entropy; 2.3 Colour fundamentals; 2.4 BGR, RGB, HSV, Lab; 2.5 Colour-based segmentation; 2.6 Cameras and sensors; 2.7 Histograms.
*Lab:* colour spaces and HSV segmentation. *Exercises:* 14.

**Chapter 3 — Image Formation, Optics and Calibration** (~18 pp)
3.1 Light and radiometry; 3.2 The pinhole model; 3.3 Intrinsics and extrinsics; 3.4 Lens aberration and distortion; 3.5 Camera calibration; 3.6 Reflectance and illumination models; 3.7 Metric measurement.
*Lab:* chessboard calibration, undistortion. *Exercises:* 14.

**Chapter 4 — Data Structures and Multi-scale Images** (~14 pp)
4.1 Matrices, chains and run-length representations; 4.2 Topological and relational structures; 4.3 Contours and hierarchy; 4.4 Pyramids (Gaussian/Laplacian); 4.5 Quadtrees; 4.6 Choosing a representation.
*Lab:* pyramids, contours, quadtree. *Exercises:* 10.

### Part II — Classical Processing

**Chapter 5 — Preprocessing I: Radiometry and Geometry** (~16 pp)
5.1 Point operations; 5.2 Gamma and contrast; 5.3 Histogram equalisation and CLAHE; 5.4 Affine transforms; 5.5 Perspective and homography; 5.6 Interpolation.
*Lab:* enhancement and warping. *Exercises:* 12.

**Chapter 6 — Preprocessing II: Filtering and Morphology** (~18 pp)
6.1 Convolution and correlation; 6.2 Linear filters; 6.3 Non-linear filters (median, bilateral); 6.4 Noise models and restoration; 6.5 Derivative filters; 6.6 Sharpening; 6.7 Morphology.
*Lab:* filtering, denoising, morphology. *Exercises:* 14.

**Chapter 7 — Segmentation** (~18 pp)
7.1 The segmentation problem; 7.2 Global, Otsu, adaptive thresholding; 7.3 Edge-based segmentation and Canny; 7.4 Hough transforms; 7.5 Region growing, connected components, watershed; 7.6 Texture; 7.7 Evaluation; 7.8 Preview of learned segmentation.
*Lab:* thresholding, Hough, watershed. *Exercises:* 14.

**Chapter 8 — Features, Matching and Homography** (~18 pp)
8.1 Why keypoints; 8.2 Harris and Shi–Tomasi; 8.3 SIFT; 8.4 ORB; 8.5 Descriptors and matching; 8.6 The ratio test; 8.7 RANSAC and robust fitting; 8.8 Applications: stitching, pose, AR.
*Lab:* matching and panorama. *Exercises:* 12.

### Part III — Learning-Based Vision

**Chapter 9 — Detection I: Two-Stage Detectors** (~18 pp)
9.1 Detection = localisation + classification; 9.2 IoU and NMS; 9.3 R-CNN; 9.4 Fast R-CNN and ROI pooling; 9.5 Faster R-CNN and the RPN; 9.6 Anchors; 9.7 Evaluation: precision, recall, AP, mAP; 9.8 Data and annotation.
*Lab:* Faster R-CNN inference and IoU/NMS. *Exercises:* 12.

**Chapter 10 — Detection II: One-Stage and Open-Vocabulary** (~18 pp)
10.1 One-stage philosophy; 10.2 YOLO family; 10.3 Training on custom data; 10.4 Augmentation and robustness; 10.5 Hands-on mAP; 10.6 Open-vocabulary detection (YOLO-World, Grounding DINO, OWL-ViT); 10.7 Deployment trade-offs.
*Lab:* YOLO training/inference and prompts. *Exercises:* 12.

**Chapter 11 — Segmentation with Deep Learning** (~18 pp)
11.1 Semantic vs instance vs panoptic; 11.2 Fully convolutional networks; 11.3 U-Net and skip connections; 11.4 DeepLab and atrous convolution; 11.5 Mask R-CNN; 11.6 YOLO-seg; 11.7 SAM and promptable segmentation; 11.8 Metrics (IoU, mIoU, Dice); 11.9 Model-assisted annotation.
*Lab:* U-Net, Mask R-CNN, SAM. *Exercises:* 12.

### Part IV — 3D, Motion, and Frontiers

**Chapter 12 — 3D Vision and Point Clouds** (~18 pp)
12.1 Projective geometry review; 12.2 Epipolar geometry and rectification; 12.3 Stereo matching and depth; 12.4 Depth cameras; 12.5 Point clouds and registration; 12.6 Reconstructing geometry; 12.7 Modern 3D reconstruction (NeRF/Gaussian splatting, conceptual).
*Lab:* stereo depth and point clouds. *Exercises:* 12.

**Chapter 13 — Motion, Optical Flow and Tracking** (~18 pp)
13.1 Change detection; 13.2 Background modelling; 13.3 Sparse flow (Lucas–Kanade); 13.4 Dense flow (Farneback); 13.5 Association and Kalman filtering; 13.6 ByteTrack and DeepSORT; 13.7 Counting and analytics; 13.8 Shape from motion.
*Lab:* flow and multi-object tracking. *Exercises:* 12.

**Chapter 14 — Vision-Language Models and Deployment** (~16 pp)
14.1 Why language for vision; 14.2 Contrastive models (CLIP/SigLIP); 14.3 Captioning and VQA; 14.4 Multimodal LLMs (LLaVA, Qwen-VL); 14.5 Prompt engineering; 14.6 Grounding detection+segmentation; 14.7 Hallucination, bias and privacy; 14.8 Edge deployment and quantisation; 14.9 Ethics and responsible systems.
*Lab:* CLIP, BLIP, prompting. *Exercises:* 10.

**Appendices**
- A. Mathematics refresher (linear algebra, probability, optimisation)
- B. Python and Colab primer
- C. Datasets and annotation guide
- D. Evaluation metrics reference
- E. Glossary and notation
- F. Index

## 6. Pedagogical features (repeated in every chapter)

- Learning outcomes and CLO mapping
- Engineering motivation vignette
- Boxed key formulas and definitions
- **Colab lab** linked to the companion repository
- Worked numerical example
- Common-misconceptions box
- "Classical ↔ modern" bridge
- Summary, exercises (conceptual / computational / programming), further reading
- Instructor-only question bank with solutions

## 7. Figure and media plan (licensing-safe)

- **Placeholder-first drafting:** every figure is inserted as a numbered placeholder with caption, intent, and source note; writing is never blocked.
- **Original, code-generated vector figures by default** (Matplotlib/draw.io/TikZ → SVG).
- **Reuse only openly licensed material:** CC0/public domain (preferred), CC BY, or CC BY-NC (book is non-commercial). Avoid CC BY-SA (share-alike propagation) and all-rights-reserved sources unless written permission is obtained.
- **Photos/data** from Wikimedia Commons, COCO/OpenCV samples, Unsplash, or original captures.
- **Figure register (CSV):** `id, chapter, caption, type, source, licence, permission status, alt text`.
- Colour-blind-safe palette and alt text on all figures.

## 8. Production toolchain

- **Quarto** (single Markdown source → HTML, PDF, EPUB).
- Executable code cells tested in CI; notebooks and the book never diverge.
- Reference manager via `references.bib`; auto-generated bibliography.
- Version control (Git); releases tagged and archived on **Zenodo** for a citable DOI.
- Accessibility: structured headings, alt text, screen-reader-friendly output.

## 9. Companion materials

- Colab labs (from the existing repository)
- Instructor manual and solutions
- Question bank (auto-gradable where possible)
- Slide decks
- Errata workflow via GitHub Issues

## 10. Timeline (solo, ~18 months)

| Phase | Months | Output |
|---|---|---|
| Planning and proposal | 1–2 | Proposal, TOC, style guide, Quarto scaffold |
| Prototype | 3–4 | One complete polished chapter (Chapter 7) |
| Drafting | 5–12 | Chapters 1–14 (~2/month), figures as placeholders |
| Figure and asset production | 8–14 | Replace placeholders, build figure register |
| Review | 13–15 | Technical + student review, revision |
| Production | 16–17 | Copyedit, index, cover, DOI |
| Publish and pilot | 18 | v1.0; teach from it; begin v1.1 errata |

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Rapid obsolescence | Concept-first; pin library versions; living-book releases |
| Scope creep | Strict chapter template and word budget |
| Copyright | Original figures by default; licence checks; figure register |
| Solo burnout | Reuse existing notebooks; recruit co-authors for heavy chapters later |
| Quality | Beta-test in own semester; formal review passes |

## 12. Success metrics

- Adopted in the IIUM course and at least 2 other institutions within 2 years.
- 1,000+ Colab opens from the companion repository per semester.
- Positive student feedback (>4/5) on clarity and labs.
- Citations/DOI usage and community contributions (issues, PRs).

## 13. Author

Dr. Hasan Zaki, Assistant Professor, Department of Mechatronics Engineering, IIUM — teaches Machine Vision and develops the companion open repository. (Expand with publication record and teaching experience for a formal submission.)

---

## Immediate next steps

1. Approve this proposal and the chapter template.
2. Build the Quarto scaffold and draft **Chapter 7 (Segmentation)** as the prototype.
3. Create the figure register and insert placeholders while drafting.
4. Register the project on Zenodo and choose the final licence.
