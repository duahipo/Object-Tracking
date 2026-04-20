# YOLOv8 vs EfficientDet — Ship Detection in Autonomous Maritime Vehicles

A comparative study of two state-of-the-art deep learning object detection architectures for real-time ship detection, evaluated on maritime datasets in the context of Autonomous Maritime Vehicle (AMV) collision avoidance.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Research Context & Literature](#3-research-context--literature)
4. [Research Gaps Addressed](#4-research-gaps-addressed)
5. [Objectives](#5-objectives)
6. [Datasets](#6-datasets)
7. [Models](#7-models)
8. [Sample Training Data](#8-sample-training-data)
9. [Results](#9-results)
10. [Project Structure](#10-project-structure)
11. [Installation & Usage](#11-installation--usage)
12. [References](#12-references)
13. [Authors](#13-authors)

---

## 1. Project Overview

Autonomous vehicles are increasingly deployed across industries, but Autonomous Maritime Vehicles (AMVs) remain technically challenging due to the complexity of real-time decision-making in open-water environments. Collision avoidance — which depends on detecting ships and obstacles reliably at speed — is one of the most critical unsolved problems in this domain.

This project benchmarks two leading deep learning object detection architectures:

- **YOLOv8** (You Only Look Once, version 8) — single-stage, real-time oriented
- **EfficientDet** — compound-scaling, accuracy-efficiency balanced

Both models were trained on a ship image dataset and tested on a real-world maritime benchmark. Performance is evaluated across accuracy, computational efficiency, and generalization to challenging maritime conditions.

---

## 2. Motivation

![Maritime environment overview](https://github.com/user-attachments/assets/edd115e6-b3ed-4de5-bf12-e0f479877500)

The maritime sector is undergoing rapid transformation driven by AI, sensor fusion, and autonomous systems. Key drivers for this research:

- Growing demand for offshore exploration, environmental monitoring, and naval surveillance
- AMVs offer measurable advantages in safety, operational cost, and mission endurance over crewed vessels
- Applications span oceanographic research, port logistics, border security, and anti-collision systems
- Object detection in maritime environments is uniquely difficult: dynamic backgrounds, variable lighting, small and distant targets, and radar-optical sensor fusion requirements
- Deep learning architectures (CNNs) have demonstrated significant improvements in detection accuracy and inference speed over classical computer vision approaches

---

## 3. Research Context & Literature

![Dataset sample — maritime scene](https://github.com/user-attachments/assets/37b933d7-951c-4077-81b5-0b94d9fbd1dd)

Research on AMVs covers a broad spectrum: autonomous navigation, obstacle avoidance, multi-sensor integration, and regulatory compliance. Recent work has focused on improving reliability and energy efficiency while advancing sensor technologies and onboard data processing.

Key prior work in maritime object detection includes:

- Video processing from electro-optical sensors for maritime object tracking (Prasad et al., 2017)
- Safety challenges in mixed navigational environments for autonomous ships (Kim et al., 2022)
- Multi-scale object detection for autonomous ship navigation (Shao et al., 2022)
- YOLOv2-based ship detection on SAR imagery (Chang et al., 2019)
- Sea mine detection using YOLO, SSD, and EfficientDet (Munteanu et al., 2022)

---

## 4. Research Gaps Addressed

![Challenging maritime detection scenario](https://github.com/user-attachments/assets/c4a67c56-5b67-4d2e-8888-e4a113769ce0)

This work targets the following open problems identified in the literature:

- Insufficient accuracy in challenging maritime environments (fog, glare, wave motion)
- Real-time operation constraints on embedded AMV hardware
- Limited generalizability across vessel types, sizes, and orientations
- Sensitivity to changes in lighting conditions and cluttered backgrounds
- Absence of sensor fusion integration in most published benchmarks
- Weak performance on small and distant objects
- Gap between laboratory evaluation and real-world maritime scenario testing
- Underexplored regulatory and ethical dimensions of autonomous maritime systems

---

## 5. Objectives

1. Evaluate the performance of YOLOv8 and EfficientDet for object detection in the context of autonomous maritime vehicles
2. Compare both architectures on a real-world maritime dataset across three axes: **detection accuracy**, **computational efficiency**, and **generalizability**
3. Provide actionable guidance on architecture selection for AMV deployment scenarios

---

## 6. Datasets

| Role | Dataset | Description |
|---|---|---|
| Training | Roboflow Ship2 Image Dataset | Annotated ship images with bounding boxes, multiple vessel classes |
| Testing | Singapore Maritime Dataset (SMD) | Real-world maritime footage, electro-optical and near-infrared sensors |

The Singapore Maritime Dataset is a standard benchmark in the field, containing sequences captured from onboard cameras in busy port and open-water environments.

---

## 7. Models

### YOLOv8

YOLOv8 is a single-stage, anchor-free detector from Ultralytics. It processes the full image in a single forward pass, making it well-suited for real-time inference on embedded systems.

```python
from ultralytics import YOLO

# Load and train
model = YOLO('yolov8n.pt')
model.train(data='ship_dataset.yaml', epochs=100, imgsz=640)

# Inference
results = model.predict(source='test_images/', conf=0.25, save=True)
```

### EfficientDet

EfficientDet applies compound scaling to simultaneously optimize backbone depth, width, and input resolution. It achieves state-of-the-art accuracy at controlled computational cost.

```python
import torch
from effdet import create_model

model = create_model('efficientdet_d0', pretrained=True, num_classes=1)
model.train()

# Training loop with maritime dataset
for images, targets in dataloader:
    loss = model(images, targets)
    loss.backward()
    optimizer.step()
```

### Architecture Comparison

| Property | YOLOv8 | EfficientDet |
|---|---|---|
| Detection paradigm | Single-stage, anchor-free | Single-stage, anchor-based |
| Scaling strategy | Fixed architecture variants (n/s/m/l/x) | Compound scaling (D0–D7) |
| Inference speed | Very fast | Moderate |
| Accuracy profile | High, especially on mid-size objects | Very high, strong on small objects |
| Hardware suitability | Edge / embedded | GPU-optimized |
| Training complexity | Low | Moderate |

---

## 8. Sample Training Data

Detection results on training and validation imagery:

![Detection result — sample 1](https://github.com/user-attachments/assets/d709b5eb-9d0c-4efa-972f-c652ec2c8c15)

![Detection result — sample 2](https://github.com/user-attachments/assets/2f08c6c3-98d0-41c9-a8cf-83a78e06881a)

![Bounding box precision — close range](https://github.com/user-attachments/assets/0c41b807-f76a-4f68-a13b-fac4f2b2160b)

![Detection result — sample 3](https://github.com/user-attachments/assets/d431f616-5794-4743-b83f-85be8305613e)

---

## 9. Results

Both models were evaluated using a Python inference script parameterized with confidence thresholds, network weight paths, and input sources (images and video streams).

**Key findings:**

- Both models correctly detect ships in static images under varied maritime conditions
- Real-time video inference was validated on inputted video sequences — not only on still frames
- Detection is robust to scale variation (near and far vessels) and partial occlusion

Detailed output files are available in the `Results/` directory:

```
Results/
├── Ship_Detection_Images/     # Per-image bounding box predictions
└── Ship_Detection_Video/      # Annotated video inference output
```

Performance evaluation covers:

- **mAP@0.5** — mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95** — standard COCO metric
- **Inference time (ms/frame)** — measured on standard GPU hardware
- **FPS** — frames per second for real-time viability assessment

---

## 10. Project Structure

```
Yolov8_vs_Efficientdet_in_ship_detection/
├── datasets/
│   ├── ship2_roboflow/              # Training dataset (annotated ship images)
│   └── singapore_maritime/          # Test dataset (SMD)
├── models/
│   ├── yolov8/
│   │   ├── train.py                 # YOLOv8 training script
│   │   ├── detect.py                # Inference script
│   │   └── weights/                 # Trained model weights (.pt)
│   └── efficientdet/
│       ├── train.py                 # EfficientDet training script
│       ├── detect.py                # Inference script
│       └── weights/                 # Trained model weights (.pth)
├── Results/
│   ├── Ship_Detection_Images/       # Image inference outputs
│   └── Ship_Detection_Video/        # Video inference outputs
├── notebooks/
│   ├── 01_EDA_Dataset.ipynb         # Dataset exploration and statistics
│   ├── 02_YOLOv8_Training.ipynb     # YOLOv8 training and evaluation
│   ├── 03_EfficientDet_Training.ipynb
│   └── 04_Comparison.ipynb          # Side-by-side performance comparison
├── requirements.txt
└── README.md
```

---

## 11. Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

```
ultralytics==8.0.196
effdet==0.4.1
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.0
numpy==1.25.2
matplotlib==3.7.2
Pillow==10.0.0
PyYAML==6.0.1
```

### Training YOLOv8

```bash
python models/yolov8/train.py \
  --data datasets/ship2_roboflow/ship.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

### Training EfficientDet

```bash
python models/efficientdet/train.py \
  --data datasets/ship2_roboflow/ \
  --model efficientdet_d0 \
  --epochs 100 \
  --batch 8
```

### Running inference

```bash
# Image inference — YOLOv8
python models/yolov8/detect.py \
  --weights models/yolov8/weights/best.pt \
  --source datasets/singapore_maritime/images/ \
  --conf 0.25 \
  --save

# Video inference — YOLOv8
python models/yolov8/detect.py \
  --weights models/yolov8/weights/best.pt \
  --source path/to/video.mp4 \
  --conf 0.25 \
  --save
```

---

## 12. References

[1] Prasad, D.K., Rajan, D., Rachmawati, L., Rajabally, E. and Quek, C., 2017. Video processing from electro-optical sensors for object detection and tracking in a maritime environment: A survey. *IEEE Transactions on Intelligent Transportation Systems*, 18(8), pp.1993–2016.

[2] Kim, T.E., Perera, L.P., Sollid, M.P., Batalden, B.M. and Sydnes, A.K., 2022. Safety challenges related to autonomous ships in mixed navigational environments. *WMU Journal of Maritime Affairs*, 21(2), pp.141–159.

[3] Shao, Z., Lyu, H., Yin, Y., et al., 2022. Multi-Scale Object Detection Model for Autonomous Ship Navigation in Maritime Environment. *Journal of Marine Science and Engineering*, 10(11), p.1783.

[4] Chang, Y.L., Anagaw, A., Chang, L., Wang, Y.C., Hsiao, C.Y. and Lee, W.H., 2019. Ship detection based on YOLOv2 for SAR imagery. *Remote Sensing*, 11(7), p.786.

[5] Munteanu, D., Moina, D., Zamfir, C.G., Petrea, S.M., Cristea, D.S. and Munteanu, N., 2022. Sea Mine Detection Framework Using YOLO, SSD and EfficientDet Deep Learning Models. *Sensors*, 22(23), p.9536.

[6] Redmon, J., Divvala, S., Girshick, R. and Farhadi, A., 2016. You Only Look Once: Unified, Real-Time Object Detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp.779–788.

[7] Roboflow Ship2 Dataset — [https://roboflow.com](https://roboflow.com)

[8] Singapore Maritime Dataset — [https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset)

---

## 13. Authors

**Research project — Deep Learning for Maritime Object Detection**

Comparative evaluation of YOLOv8 and EfficientDet architectures applied to ship detection in the context of Autonomous Maritime Vehicle development.

---

*Detailed results and annotated outputs are available in the `Results/` directory. For methodology questions, open an issue or refer to the notebooks.*
