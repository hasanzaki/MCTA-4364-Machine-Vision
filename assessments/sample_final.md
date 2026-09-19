# Sample Final Examination (40 marks)

**MCTE 4323 / MCTA 4364 Machine Vision** - covers the whole course.
Time: 2 hours. Answer all questions. Calculators permitted.

---

## Section A - Short answers (12 marks, 2 each)

**A1.** Explain what makes a **keypoint** repeatable across different viewpoints, and why corners are preferred over edges.

**A2.** State the **ratio test** in feature matching and explain what it removes.

**A3.** Define **Intersection over Union (IoU)** and explain its role in deciding a true positive.

**A4.** Differentiate **semantic**, **instance** and **panoptic** segmentation with one example each.

**A5.** Explain why **mIoU** is preferred over pixel accuracy for segmentation evaluation.

**A6.** Explain the difference between **frame differencing** and **background subtraction** for motion detection.

---

## Section B - Calculations (12 marks, 4 each)

**B1.** A detector produces the predictions below for one class. The IoU threshold is 0.5. Compute precision and recall.
| Prediction | Score | Matches a ground truth? (IoU) |
|---|---|---|
| 1 | 0.95 | yes (0.80) |
| 2 | 0.80 | yes (0.55) |
| 3 | 0.70 | no (0.20) |
| 4 | 0.60 | yes (0.65) |

**B2.** A stereo rig: $f = 600$ px, $B = 0.20$ m. Two points have disparities $d_1 = 30$ px and $d_2 = 6$ px. Compute both depths and comment on which is more uncertain.

**B3.** A U-Net predicts a foreground mask over a 200 x 200 image. The predicted mask has 8,000 foreground pixels; the ground truth has 6,000 foreground pixels; their intersection is 5,000 pixels. Compute the foreground IoU.

---

## Section C - Detection and segmentation design (8 marks)

**C1.** Compare **Faster R-CNN**, **YOLO** and **SAM** for a robotic bin-picking application that must locate and grasp unknown objects at ~15 frames per second. Recommend one system (or combination), justify it, and state one limitation. (8)

---

## Section D - 3D, motion and modern AI (8 marks)

**D1.** A company wants to add a natural-language interface: *"Is there a missing screw on this board?"* using only a camera. Discuss how you would build this with current models, and analyse **two** risks (e.g. hallucination, latency) with a mitigation for each. (8)

---

## Mark scheme

<details><summary>Show mark scheme</summary>

### Section A
- **A1:** A keypoint is a distinct, locally repeatable location. Corners change in two directions, so they are localised precisely and matched reliably; edges only pin location along the normal. 2 marks.
- **A2:** Accept a match only if the nearest descriptor distance $d_1 < \\rho\\, d_2$ (second nearest). It removes ambiguous matches. 2 marks.
- **A3:** $\\mathrm{IoU} = $ overlap area / union area. A prediction is a true positive if IoU with a ground-truth box exceeds the threshold (commonly 0.5). 2 marks.
- **A4:** Semantic = class per pixel (all cars one blob). Instance = separate mask per object (car 1, car 2). Panoptic = stuff + instances together. Examples arbitrary but correct. 2 marks.
- **A5:** Pixel accuracy is dominated by large classes/background; mIoU averages per-class overlap so small classes count equally. 2 marks.
- **A6:** Frame differencing compares consecutive frames (detects recent change only). Background subtraction models the static background and detects anything deviating from it. 2 marks.

### Section B
- **B1:** TP = 3 (predictions 1, 2, 4), FP = 1 (prediction 3), FN = 0. Precision $= 3/4 = 0.75$. Recall $= 3/3 = 1.0$. (4)
- **B2:** $Z_1 = 600\\times0.20/30 = 4.0$ m; $Z_2 = 600\\times0.20/6 = 20.0$ m. The larger depth is more uncertain because depth error grows with $Z^2/(fB)$ and small disparities are noise-sensitive. (4)
- **B3:** Union $= 8000 + 6000 - 5000 = 9000$. IoU $= 5000/9000 = 0.556$. (4)

### Section C (indicative bands)
- Recognize the need for **speed** (15 FPS): YOLO is the real-time choice. (2)
- Recognize **unknown objects**: SAM or open-vocabulary detectors generalise without retraining. (2)
- Recommendation: YOLO/OWL-ViT/YOLO-World for detection + SAM for precise masks, or a two-stage perception pipeline; justify integration. (2)
- Limitation: SAM is slow/not class-aware; open-vocabulary may miss small objects; grasp planning still needed. (2)

### Section D (indicative bands)
- Pipeline: open-vocabulary detector (Grounding DINO / OWL-ViT) or VLM (LLaVA/CLIP) prompted with the question, optionally grounded by a detector/classifier. (3)
- Risk 1: **hallucination** - mitigation: constrain to grounded detections, require confidence, human verification. (2)
- Risk 2: **latency** - mitigation: smaller/distilled models, edge deployment, caching, or run asynchronously. (2)
- Overall coherence and engineering judgement. (1)

</details>
