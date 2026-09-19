# Week 11 - Deep Segmentation
> **Official syllabus block:** Semantic image segmentation and understanding (Chapter 10).

## Learning objectives
- Distinguish semantic, instance and panoptic segmentation.
- Explain U-Net.
- Run YOLO-seg, Mask R-CNN and SAM.
- Evaluate with IoU and mIoU.

## Key concepts
- Encoder-decoder with skip connections.
- Per-instance masks.
- Promptable foundation models.
- Dense prediction metrics.

## Key formulas
- mIoU = mean over classes of intersection over union.
- U-Net output has the same spatial size as the input.

## Common misconceptions
- Pixel accuracy is dominated by background; prefer mIoU.
- SAM segments, it does not classify.
- Instance masks are not the same as semantic masks.

## In the lab
Open `notebooks/11_deep_segmentation/11_deep_segmentation.ipynb` in Colab for guided examples, interactive widgets, exercises and self-check questions.

## Further reading
- U-Net paper (Ronneberger et al., 2015).
- SAM: https://segment-anything.com/
