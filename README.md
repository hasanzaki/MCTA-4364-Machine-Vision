# Machine Vision — Hands-On Labs

A set of **14 Jupyter notebooks** that walk through the machine vision pipeline: image basics, colour, camera calibration, preprocessing, segmentation, feature matching, object detection, deep segmentation, 3D vision, motion tracking, and vision-language models.

Each notebook is **self-contained** — the first cell installs any missing packages, and sample images are loaded from `scikit-image` or generated in code. No setup required.

## Open a notebook

Click any badge to run it instantly in **Google Colab** (free). You can also open the notebooks in **Kaggle**, **Jupyter**, or any other environment — see below.

| Week | Topic | Open in Colab |
|---|---|---|
| 01 | Image basics and setup | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/01_image_basics/01_image_basics.ipynb) |
| 02 | Image representation and colour | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/02_color_spaces/02_color_spaces.ipynb) |
| 03 | Optics and camera calibration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/03_optics_calibration/03_optics_calibration.ipynb) |
| 04 | Data structures and pyramids | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/04_image_structures/04_image_structures.ipynb) |
| 05 | Preprocessing and geometric transforms | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/05_geometric_transforms/05_geometric_transforms.ipynb) |
| 06 | Filtering and morphology | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/06_filtering_morphology/06_filtering_morphology.ipynb) |
| 07 | Segmentation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/07_segmentation/07_segmentation.ipynb) |
| 08 | Keypoints and feature matching | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/08_keypoint_features/08_keypoint_features.ipynb) |
| 09 | Object detection with R-CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/09_object_detection_rcnn/09_object_detection_rcnn.ipynb) |
| 10 | Object detection with YOLO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/10_object_detection_yolo/10_object_detection_yolo.ipynb) |
| 11 | Deep segmentation (U-Net, SAM) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/11_deep_segmentation/11_deep_segmentation.ipynb) |
| 12 | 3D vision and point clouds | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/12_3d_point_clouds/12_3d_point_clouds.ipynb) |
| 13 | Motion, optical flow and tracking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/13_motion_tracking/13_motion_tracking.ipynb) |
| 14 | Vision-language models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/MCTA-4364-Machine-Vision/blob/main/notebooks/14_vision_language_models/14_vision_language_models.ipynb) |

## Running in other environments

- **Kaggle** — create a new notebook, then *File → Import Notebook → URL* and paste the notebook's GitHub link (use the `blob` link above), or download the `.ipynb` and upload it.
- **Jupyter / VS Code / any notebook tool** — clone this repository and open the notebook, or download the single `.ipynb` file you need.
- **Locally** — the first cell of each notebook installs its own dependencies. You can also install everything at once:
  ```bash
  pip install numpy matplotlib opencv-python scikit-image ipywidgets
  pip install torch torchvision ultralytics transformers   # weeks 09-14
  ```

> Weeks 09–11 and 14 download pretrained models on first run; a GPU runtime (free in Colab) is recommended but not required.

## References

Current, freely available resources used and recommended by these labs:

- [OpenCV documentation](https://docs.opencv.org/4.x/)
- [scikit-image user guide](https://scikit-image.org/docs/stable/)
- [PyTorch / torchvision models](https://pytorch.org/vision/stable/models.html)
- [Ultralytics YOLO](https://docs.ultralytics.com/) — detection, segmentation, tracking
- [Meta SAM 2](https://github.com/facebookresearch/sam2) — promptable segmentation
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — open-vocabulary detection
- [OpenAI CLIP](https://github.com/openai/CLIP) — image–text models
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) — vision-language models
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — modern 3D reconstruction

## License

MIT. See [LICENSE](LICENSE).
