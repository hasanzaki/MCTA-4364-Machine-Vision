# MCTA 4364 Machine Vision — Course Plan

14 weeks × 2 sessions × 80 minutes. Each teaching session has its own Jupyter notebook in `notebooks/`.
Every notebook runs in Google Colab, on Kaggle, or locally on a laptop with a 4 GB NVIDIA GPU (see [SETUP.md](SETUP.md)).

## Storyline

**Pixels → camera → enhancement → segmentation → hand-crafted features → learned features (CNNs) → midterm →
detection → data-centric engineering → tracking → deep segmentation → 3D → vision-language and generative models → project.**

## Weekly plan

| Week | Session A | Session B | Milestone |
|---|---|---|---|
| 0 | *Setup: installation, GPU and webcam check (self-study before Week 1)* | | |
| 1 | **W01A** Machine vision systems and digital images | **W01B** Colour and histograms | |
| 2 | **W02A** Image formation, optics and lighting | **W02B** Camera calibration in practice | |
| 3 | **W03A** Point operations and geometric transforms | **W03B** Convolution, filtering, Fourier and morphology | Groups formed |
| 4 | **W04A** Classical segmentation: thresholds, edges, Hough | **W04B** Regions and shape analysis | |
| 5 | **W05A** Features and scale space | **W05B** Matching, RANSAC and stitching | Project proposal |
| 6 | **W06A** Deep learning I: from hand-crafted to learned features | **W06B** Deep learning II: training well and transfer learning | |
| 7 | **W07A** Integration lab and revision | **Midterm** (Weeks 1–6) | |
| 8 | **W08A** Object detection I: fundamentals and two-stage detectors | **W08B** Object detection II: YOLO, DETR and open-vocabulary | |
| 9 | **W09A** Data-centric machine vision | **W09B** Training and deploying a custom detector | Dataset and baseline |
| 10 | **W10A** Motion analysis and optical flow | **W10B** Tracking detected objects | |
| 11 | **W11A** Deep semantic and instance segmentation | **W11B** Foundation segmentation (SAM) and anomaly detection | |
| 12 | **W12A** Stereo vision and point clouds | **W12B** Monocular depth and 3D foundation models | Progress check |
| 13 | **W13A** Vision-language models and auto-labelling | **W13B** Generative vision and synthetic data | |
| 14 | **Project demonstrations** | **Project demonstrations** | Final demo |

## How the threads connect

- **Convolution thread:** hand-designed kernels (W03B) → edge and texture features (W04–W05) → learned kernels in CNNs (W06) → detectors and segmenters (W08–W11).
- **Geometry thread:** camera model and calibration (W02) → homography (W03A, W05B) → speed from pixels (W10B) → stereo and 3D (W12).
- **Data thread:** splits and augmentation (W06B) → data-centric machine vision (W09A) → custom datasets (W09B) → auto-labelling and synthetic data (W13).
- **Engineering thread:** lighting and optics (W02) → a 4 GB GPU budget (W06B) → deployment with ONNX (W09B) → real-time tracking (W10B).

## Every notebook contains

- learning outcomes and an 80-minute session plan;
- theory with equations and many explanatory figures;
- guided examples on real, openly licensed images;
- interactive widgets (sliders, pickers, 3D views);
- exercises with automatic self-checks, and an open challenge;
- an auto-marked quiz, reflection questions, a "what's new" box, references (seminal papers, free textbooks, videos, interactive demos) and key takeaways.

## Assessment slots

- **Midterm:** Week 7, Session B (Weeks 1–6).
- **Project:** proposal (Week 5), dataset and baseline (Week 9), progress check (Week 12), demonstration (Week 14). See `notebooks/W14_project/`.

## Core references (free online)

- Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd ed. [szeliski.org/Book](https://szeliski.org/Book/)
- Torralba, A., Isola, P. & Freeman, W. (2024). *Foundations of Computer Vision*. [visionbook.mit.edu](https://visionbook.mit.edu/)
- Prince, S. J. D. (2023). *Understanding Deep Learning*. [udlbook.github.io](https://udlbook.github.io/udlbook/)
- Nayar, S. *First Principles of Computer Vision* (video lectures). [fpcv.cs.columbia.edu](https://fpcv.cs.columbia.edu/)

Recommended for machine-vision engineering: Steger, Ulrich & Wiedemann, *Machine Vision Algorithms and Applications*, 2nd ed. (Wiley, 2018);
Corke, *Robotics, Vision and Control*, 3rd ed. (Springer, 2023).
