# Week 10 - Object Detection II: YOLO and Open-Vocabulary Detection
> **Official syllabus block:** Image understanding: scene labelling and semantic segmentation (Chapter 10).

## Learning objectives
- Explain one-stage detection.
- Run YOLO inference and tune conf/NMS.
- Train on a custom dataset.
- Use open-vocabulary detection.

## Key concepts
- Single network predicts boxes and classes.
- Anchor-free heads in modern YOLO.
- Dataset annotation formats (YOLO, COCO, VOC).
- Open-vocabulary models accept text prompts.

## Key formulas
- mAP50 and mAP50-95.
- NMS overlap threshold controls duplicate removal.

## Common misconceptions
- Speed and accuracy trade off across model sizes (n, s, m, l, x).
- More epochs do not fix bad annotations.
- Open-vocabulary models can miss small or unusual objects.

## In the lab
Open `notebooks/10_object_detection_yolo/10_object_detection_yolo.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- Ultralytics documentation: https://docs.ultralytics.com/
- YOLO-World: https://docs.ultralytics.com/models/yolo-world/
