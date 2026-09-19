# Mini-Project Guidelines

**MCTE 4323 / MCTA 4364 Machine Vision** - Semester 1, 2026/2027

The mini-project lets you apply the full machine vision pipeline to a real problem and develop all four Course Learning Outcomes, especially **CLO4 (select appropriate techniques)**. It replaces disconnected exercises with one coherent piece of engineering work.

---

## 1. Choose your project

You may choose **one** of the following tracks.

### Track A - Applied system (recommended)
Build a working vision system for an engineering problem, for example:
- Automatic License Plate Recognition (ALPR) for Malaysian plates
- Vehicle counting and speed estimation from traffic video
- Defect detection on a production line (PCB, metal, fabric)
- Currency or product counting using detection
- Football analytics: player tracking and heatmaps, or offside detection
- Lane detection (classical vs deep learning)
- Depth estimation (stereo or monocular) and 3D reconstruction

### Track B - Technique investigation
Take a technique from the course and investigate it rigorously, e.g.:
- Classical vs deep segmentation (Otsu/Canny vs U-Net/SAM)
- R-CNN vs YOLO vs DETR: accuracy-speed trade-offs
- Effect of data augmentation on detection robustness
- Auto-labelling a YOLO dataset with SAM / Grounding DINO

Whatever you choose, you must **implement**, **evaluate**, and **analyse failures** - not only run a pretrained demo.

---

## 2. Required workflow

```text
1. Problem definition   -> What decision does the system make? What is the success metric?
2. Data                 -> Source, size, classes, annotation, limitations and bias
3. Baseline             -> Simplest reasonable method; report its performance
4. Method / innovation  -> At least one justified improvement over the baseline
5. Evaluation           -> Quantitative metrics on a held-out test set
6. Failure analysis     -> Where and why it fails (with visual examples)
7. Deployment note      -> Speed, hardware, and practical constraints
8. Demo + report        -> Live/recorded demo, poster, GitHub repository
```

---

## 3. Deliverables

| Deliverable | Description | CLOs |
|---|---|---|
| **Proposal (Week 3)** | One page: problem, data, baseline, metric, plan | 4 |
| **Progress demo (Week 10)** | Working baseline + initial results | 1, 2 |
| **GitHub repository** | Code, README, instructions, results | 1 |
| **One-page poster** | Problem, method, results, conclusion, link | 3, 4 |
| **Final presentation + live demo (Week 14)** | 10-12 minutes + Q&A | 1, 2, 3, 4 |
| **Report (Week 14)** | 6-10 pages, structured like a short paper | 2, 3, 4 |

---

## 4. Assessment rubric (100 marks)

| Criterion | Weight | Excellent (80-100%) | Satisfactory (50-79%) | Weak (<50%) |
|---|---|---|---|---|
| Problem definition and CLO4 selection | 10 | Clear problem, justified method choice | Problem stated, weak justification | Vague problem |
| Data handling and annotation | 15 | Well described, biases discussed, pipeline reproducible | Basic data handling | Undocumented data |
| Baseline implementation | 15 | Correct, working, measured | Partially working | Missing |
| Method / innovation | 20 | Meaningful, justified, evaluated gain | Small improvement | None beyond a demo |
| Evaluation and metrics | 20 | Correct metrics (IoU/mAP/mIoU/accuracy), held-out test | Some metrics | No quantitative evaluation |
| Failure analysis | 10 | Insightful, visual, links cause to effect | Brief | None |
| Presentation, demo, repo, poster | 10 | Clear, reproducible, professional | Adequate | Disorganised |

---

## 5. Suggested metrics (use the right one)

| Task | Primary metric |
|---|---|
| Detection | mAP50, mAP50-95, precision, recall |
| Segmentation | IoU, mIoU, Dice |
| Classification | accuracy, F1, confusion matrix |
| Counting / tracking | count error, MOTA, IDF1 |
| Depth / 3D | RMSE, absolute relative error |
| Recognition (OCR/plates) | character accuracy, plate accuracy |

---

## 6. Minimum technical requirements
- Python, OpenCV, and either PyTorch or Ultralytics.
- At least one trained or fine-tuned model, **or** a rigorously evaluated pretrained baseline.
- A held-out test set that is **not** used for tuning.
- A reproducible README: environment, data, commands, expected output.

## 7. Academic integrity
- You may use open-source code, but you must **cite** it and state clearly what you changed.
- Datasets must be public or self-collected with permission.
- All results must be reproducible from your repository.

## 8. Suggested timeline

| Week | Milestone |
|---|---|
| 3 | Proposal submitted |
| 5-6 | Dataset prepared and annotation checked |
| 8 | Baseline running and measured |
| 10 | Progress demo + innovation implemented |
| 12 | Full evaluation + failure analysis |
| 14 | Poster, report, repository, live demo |

## 9. Report structure
1. **Introduction** - problem, motivation, objectives
2. **Related work** - 3-5 references
3. **Method** - baseline and your improvement, with a pipeline diagram
4. **Data** - source, annotation, split, augmentation
5. **Results** - tables and figures with metrics
6. **Discussion** - failure cases, limitations, threats to validity
7. **Conclusion and future work**
8. **References** and a link to the GitHub repository

## 10. Submission checklist
- [ ] GitHub repository is public and runs from the README
- [ ] Test set is held out and disclosed
- [ ] Metrics are reported with the exact configuration used
- [ ] At least three failure cases are analysed with images
- [ ] Poster link and video demo included
- [ ] All external code and datasets are cited
