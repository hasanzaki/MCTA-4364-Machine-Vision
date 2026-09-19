# Week 09 - Object Detection I: R-CNN to Faster R-CNN
> **Official syllabus block:** Object recognition and image understanding foundations (Chapters 9-10).

## Learning objectives
- State the detection problem.
- Compute IoU and apply NMS.
- Explain the R-CNN family.
- Evaluate detection with precision, recall and mAP.

## Key concepts
- Localisation plus classification.
- Region proposals and ROI pooling.
- Two-stage detectors.
- Evaluation metrics.

## Key formulas
- IoU = intersection area / union area.
- Precision = TP / (TP + FP); Recall = TP / (TP + FN).
- mAP averages AP over classes and IoU thresholds.

## Common misconceptions
- High confidence does not guarantee correctness.
- Accuracy is misleading for detection.
- NMS is not training; it is a post-processing step.

## In the lab
Open `notebooks/09_object_detection_rcnn/09_object_detection_rcnn.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- torchvision detection models: https://pytorch.org/vision/stable/models.html
- COCO dataset: https://cocodataset.org/
