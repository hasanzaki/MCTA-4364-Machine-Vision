# MCTE 4323 / MCTA 4364: Machine Vision (2026 Update)

Welcome to the official repository for **MCTE 4323 / MCTA 4364: Machine Vision**, offered by the **Department of Mechatronics Engineering, Kulliyyah of Engineering, International Islamic University Malaysia (IIUM)**. 

This repository has been reorganized and modernized for the upcoming semester to provide a clean, structured, and **2026-ready** learning experience. It combines rigorous classical computer vision foundations with state-of-the-art deep learning, object detection, real-time tracking, 3D point cloud processing, and Vision-Language Models (VLMs).

---

## 🚀 Course Overview & Learning Outcomes

This course equips mechatronics engineering undergraduates with the theoretical foundations and practical engineering skills required to design, develop, and deploy machine vision systems for industrial automation, robotics, autonomous vehicles, and intelligent systems.

### Course Learning Outcomes (CLOs)
Upon successful completion of this course, students will be able to:
1. **Develop** machine vision systems using specific software tools (Python, OpenCV, PyTorch).
2. **Apply** modern machine vision algorithms for engineering applications.
3. **Apply** 2D and 3D machine vision techniques for various engineering applications.
4. **Select** appropriate techniques for object recognition and scene understanding for various industrial applications.

---

## 📅 Modernized 14-Week Syllabus & Mapping

The accredited outline (Sem 1 2015/2016) is entirely classical and contains **no deep learning content**. To keep every official topic while making the course industry-relevant in 2026, each week retains its accredited theme but the teaching materials and labs are modernized. Classical segmentation and RANSAC are taught in context rather than as standalone topics; classical image classification is intentionally omitted because it is covered in the Deep Learning course.

| Week | Official Block (accredited) | Modernized Content (2026) | Lab / Colab Notebook |
|---|---|---|---|
| **1** | Introduction (Ch. 1) | CV pipeline, Python & Colab setup, image geometry, sampling & quantization | [01 Image Basics](notebooks/01_image_basics/) |
| **2** | Image Representation (Ch. 2) | Colour spaces (RGB/HSV/Lab), digitization, image properties, cameras | [02 Colour Spaces](notebooks/02_color_spaces/) |
| **3** | Image Formation Physics (Ch. 3) | Optics, lens aberration, distortion, calibration, radiometry | [03 Optics & Calibration](notebooks/03_optics_calibration/) |
| **4** | Data Structures for Image Analysis (Ch. 4) | Arrays, chains, pyramids, quadtrees, topological/contour structures | [04 Image Structures](notebooks/04_image_structures/) |
| **5** | Image Preprocessing I (Ch. 5) | Brightness transforms, histogram equalization, geometric transforms | [05 Preprocessing & Transforms](notebooks/05_geometric_transforms/) |
| **6** | Image Preprocessing II (Ch. 5) | Convolution, filtering, denoising, restoration, morphology | [06 Filtering & Morphology](notebooks/06_filtering_morphology/) |
| **7** | Segmentation (Ch. 6) | Thresholding (Otsu), edge-based, region-based, texture → preview of learned segmentation | [07 Segmentation](notebooks/07_segmentation/) |
| **8** | Object Recognition (Ch. 9) | Keypoints: Harris, SIFT, ORB, descriptor matching, homography | [08 Keypoint Features](notebooks/08_keypoint_features/) |
| **9** | Object Recognition (Ch. 9) | **Two-stage detection: R-CNN → Fast/Faster R-CNN, anchors, IoU, mAP** | [09 Object Detection (R-CNN)](notebooks/09_object_detection_rcnn/) |
| **10** | Image Understanding (Ch. 10) | **One-stage & open-vocabulary detection: YOLO, YOLO-World, Grounding DINO** | [10 Object Detection (YOLO)](notebooks/10_object_detection_yolo/) |
| **11** | Image Understanding (Ch. 10) | **Segmentation: U-Net, YOLO-seg, Mask R-CNN, SAM** | [11 Deep Segmentation](notebooks/11_deep_segmentation/) |
| **12** | 3D Vision (Ch. 11) | Projective geometry, stereo, depth cameras, point clouds, 3D reconstruction | [12 3D Vision](notebooks/12_3d_point_clouds/) |
| **13** | Dynamic Vision (Ch. 16) | Change detection, optical flow, motion correspondence, tracking (ByteTrack/DeepSORT), counting | [13 Motion & Tracking](notebooks/13_motion_tracking/) |
| **14** | Dynamic Vision (Ch. 16) | VLMs, generative/diffusion vision, edge deployment, ethics, project demos | [14 Vision-Language Models](notebooks/14_vision_language_models/) |

---

## 📂 Repository Structure

The educational materials in this repository are strictly organized into separate directories to make following the course simple and professional for both students and instructors:

```text
mcta-4364-machine-vision/
├── README.md                    # Syllabus, course overview, and landing page
├── LICENSE                      # MIT licence for the teaching code
├── lectures/                    # Slide decks and conceptual notes, one folder per week
│   ├── 01_introduction/
│   ├── 02_representation/
│   ├── 03_optics_calibration/
│   ├── 04_image_structures/
│   ├── 05_geometric_transforms/
│   ├── 06_filtering_morphology/
│   ├── 07_segmentation/
│   ├── 08_keypoint_features/
│   ├── 09_object_detection_rcnn/
│   ├── 10_object_detection_yolo/
│   ├── 11_deep_segmentation/
│   ├── 12_3d_point_clouds/
│   ├── 13_motion_tracking/
│   └── 14_vision_language_models/
├── notebooks/                   # Hands-on Colab notebooks, one folder per week
│   ├── 01_image_basics/
│   ├── 02_color_spaces/
│   ├── ...
│   ├── 11_deep_segmentation/
│   └── 14_vision_language_models/
├── projects/                    # Mini-project guidelines, starter code, and datasets
│   ├── miniproject_guidelines.docx
│   ├── MV_ProjectDescription_SEM12526.pdf
│   └── datasets/                # Custom datasets (not pushed to GitHub)
├── assessments/                 # Student revision materials
│   └── private/                 # Exam papers and grade material (never pushed)
├── resources/                   # Shared media and support files
│   ├── images/
│   ├── videos/
│   ├── models/                  # Weights and configs (not pushed to GitHub)
│   └── scripts/
└── setup/                       # Environment setup
    ├── requirements.txt         # Python dependencies
    └── local_setup.md           # Guide to run the labs locally
```

> **Note on Private Materials:** Internal administrative files (student marks sheets, final grade calculations, draft exam sheets, validation documents) are strictly omitted from this public repository to comply with academic integrity, privacy regulations, and university policies.

---

## 🛠️ Quick Start for Students

### 1. Running on Google Colab (Highly Recommended)
All lab notebooks in the `notebooks/` directory include an **Open in Colab** button. Clicking this button loads the notebook directly in a cloud environment with free GPU access, requiring zero local setup.

### 2. Running Locally (Optional)
If you prefer to run the code locally on your laptop:
1. Clone this repository:
   ```bash
   git clone https://github.com/hasanzaki/MCTA-4364-Machine-Vision.git
   cd MCTA-4364-Machine-Vision
   ```
2. Follow the detailed steps in [setup/local_setup.md](setup/local_setup.md) to set up a clean Python virtual environment and install OpenCV, PyTorch, and other required packages.

---

## 📚 Textbooks & References

### Required Text
* Sonka, M., Hlavac, V., & Boyle, R. (2008). *Image Processing, Analysis, and Machine Vision*. Thomson Learning. (Chapters 1-6, 9-11, 16)

### Recommended References
1. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*. Springer (2nd Edition, free draft available online).
2. Bradski, G., & Kaehler, A. (2017). *Learning OpenCV 3: Computer Vision in C++ with the OpenCV Library*. O'Reilly Media.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Crane, R. (1997). *A Simplified Approach to Image Processing*. Prentice Hall.
5. Petrou, M., & Bosdogianni, P. (2000). *Image Processing: The Fundamentals*. John Wiley & Sons.

---

## 🤝 Contribution & License
This repository is maintained by **Dr. Hasan Zaki** for educational purposes. Feel free to fork, report issues, or suggest improvements to the labs. All code is distributed under the MIT License unless specified otherwise.
