GEO5017 Project Group 9
Waste Detection in the Built Environment

Overview
This project implements a waste detection pipeline for street-view images using transfer learning.
Our final reported workflow used a single-model ResNet-50 setup with a labeled folder-based dataset
split into train, val, and test. The main pipeline is in waste_detection_resnet50_v2.py.

Dataset structure
dataset/
├── train/    <- (2016-2019)
│   ├── waste/
│   └── no_waste/
├── val/      <- (2020-2021)
│   ├── waste/
│   └── no_waste/
└── test/     <- (2022-2023)
    ├── waste/
    └── no_waste/

Environment
- Python 3.11
- PyTorch
- torchvision
- numpy
- matplotlib
- pillow
- scikit-learn

Hardware used
- NVIDIA RTX 1000 Ada Generation GPU
- CUDA 12.8

Install packages
pip install torch torchvision numpy matplotlib pillow scikit-learn
pip install ultralytics opencv-python

Main settings
- Backbone: ResNet-50
- Batch size: 256
- WeightedRandomSampler enabled
- Two-phase training:
  1. Train classification head
  2. Fine-tune last block

The script saves checkpoints, evaluates on the test set, and can generate Top-100 detections. :contentReference[oaicite:1]{index=1}

Run
python waste_detection_resnet50_v2.py

Main outputs
- checkpoints/
- learning_curves.png
- confusion_matrix.png
- roc_curve.png
- top100/
- top100_annotated/

Final workflow used
Our final submission used a single-model test-only workflow.
Images in dataset/test were ranked by predicted waste confidence, and the Top 100 highest-confidence
images were selected as the final result set. This was logically done only on the test set.

Submission contents
The final submission archive should contain:
- Report
- Source code (to be reached through link GitHub in report)
- README / ReadMe.txt
- Top 100 waste detections as images
- A CSV file containing some extra information

According to the project brief, the required results file is the Top 100 detections as images. :contentReference[oaicite:2]{index=2}

Main file
- waste_detection_resnet50_v2.py

Optional helper files
- namestrip.py
- singlemodelscore.py
- ensemble.py

Group
Group: [09]
Members:
- Henryk
- Adriano
- Mathijs
