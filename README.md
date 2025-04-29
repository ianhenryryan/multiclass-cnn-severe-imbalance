# Multi-Class CNN with Severe Class Imbalance from Combined Datasets (Image Classification)
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    </header>
    <section>
        <p><strong>Machine Learning Engineer (aspiring):</strong> Ian H. Ryan</p>
        <p><strong>Project Length:</strong> March 2025 - April 2025</p>
        <p><strong>Description:</strong> A Complete overhaul of my <strong><a href="https://github.com/ianhenryryan/capstone" target="_blank">Capstone Project</a></strong>. For my capstone, I took on the challenge of building a <strong>multi-class Convolutional Neural Network</strong> from scratch using <strong>PyTorch</strong>, with the explicit constraint of <em>not using pre-trained models</em> (such as ResNet, MobileNet, etc). The model was designed to handle <strong>severe class imbalance</strong>, mirroring real-world challenges often seen in production data. Instead of using balanced or curated datasets, I intentionally created an extreme case of class imbalance by merging two different datasets. One of the 18 classes makes up 50% of the images, while the remaining 17 classes split the other half.

Since graduating, I’ve been refining this project and correcting mistakes that were present in the original capstone version. One major issue I uncovered was that I had incorrectly combined the datasets, which resulted in misclassified images and even some completely empty classes — <strong>making all the original training results invalid</strong>. I committed to debugging everything: redundant code, mismatched variable names, inconsistent transforms, and broken visualizations. Once the pipeline was stable and fully functional, I moved on to implementing upgrades across the board — from dataset engineering and augmentation to better loss strategies and metrics. It became a personal challenge — one I used to reinforce core deep learning concepts under tougher, more realistic conditions. 

It’s been a fun dissection that’s helped me grow—improving my skills, decision making, and overall approach to building machine learning pipelines, while continuing my journey of finding employment after graduating in December 2024.

<strong>for in-depth analysis of Capstone VS this version, check out my blog post @ https://ianhryan.com/blog.html</strong></p>
    </section>
<hr>
  <p>
    <strong>The combined dataset includes:</strong><br>
    - <strong><a href="https://universe.roboflow.com/pascal-to-yolo-8yygq/inria-person-detection-dataset" target="_blank">INRIA Person Detection Dataset</a></strong> (Roboflow Universe) by user <strong>Pascal to Yolo</strong>: 902 images of class <code>person</code>, manually split 70/20/10 for train/val/test.<br>
    - <strong><a href="https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/data" target="_blank">Animal Image Dataset</a></strong> (Kaggle) by user <strong>iamsouravbanerjee</strong>: 17 cherry-picked animal classes from their 90 total, totaling 902 images, also split 70/20/10.
  </p>

  <p>
    This resulted in a <strong>long-tail distribution</strong> — where one class dominates, and the rest are minority classes with as few as ~37 images each in training. Notably difficult classes include <strong>badger</strong>, <strong>possum</strong>, <strong>porcupine</strong>, and visually similar primates like <strong>chimpanzee</strong>, <strong>gorilla</strong>, and <strong>orangutan</strong>.
  </p>

  <hr>

  <p>
    <strong>This version of the notebook reflects a full transformation from the original capstone project.</strong><br>
    The primary goals were:
    <ul>
      <li>Improve <strong>training stability</strong> under mixed-precision (AMP)</li>
      <li>Ensure <strong>reproducibility</strong> through structured config and deterministic data processing</li>
      <li>Optimize for <strong>generalization</strong> on a limited hardware setup (NVIDIA 3060 mobile GPU)</li>
    </ul>
  </p>
  
  <h3>Highlights</h3>
  <ul>
    <li>Custom-built CNN architecture (no transfer learning)</li>
    <li><strong>MixUp</strong> augmentation with soft-label-compatible <strong>Focal Loss</strong></li>
    <li><strong>WeightedRandomSampler</strong> to balance class exposure across epochs</li>
    <li><strong>Per-class threshold tuning</strong> based on softmax probabilities</li>
    <li>AMP (mixed precision) training loop with <strong>early stopping</strong> and best model saving</li>
    <li><strong>Lightweight, memory-aware training</strong> compatible with constrained local GPUs</li>
  </ul>
</body>
</html>

### ✅ Key Improvements

- **Fixed dataset corruption bug** due to class label offset after dataset merging  
- **MixUp augmentation + SoftTarget FocalLoss** with per-class weighting  
- **WeightedRandomSampler** to ensure fair sampling across all classes  
- **Dynamic architecture upgrades**:  
  - ReLU → GELU (Conv layers), ReLU → LeakyReLU (FC layers)  
  - BatchNorm1d reordered to follow activation (pretrained model pattern)  
  - Added Dropout and bottleneck layers for regularization  
- **AMP (mixed precision)** with custom `ToFloat32()` patch for stable training  
- **Optimizer improvement:** no weight decay on bias/BatchNorm  
- **Scheduler tuning:** settled on `ReduceLROnPlateau(mode='max')` after testing alternatives  
- **Early stopping** with best F1 + val accuracy model checkpointing  
- **CUDA memory management**, soft-label handling, and cleaner loader logic  

---

### 🧰 Utility Enhancements

- `class_names.json` and `dataset_stats.json` saved for reusability  
- Modular utility functions (verify structure, convert formats, cleanup, etc.)  
- Visualizations for dataset class balance and training metrics  

<h1>Multi-Class CNN with Severe Class Imbalance from Combined Datasets - Computer Vision (Classification)</h1>

## **Table of Contents**
- [Libraries & Imports](#libraries--imports)
  - [Libraries](#libraries)
  - [Imports](#imports)
- [Check GPU Availability for CUDA](#check-gpu-availability-for-cuda)
- [CUDA allocation limiting ~64mb](#cuda-allocation-limiting-64mb)
- [Seed](#seed)
- [Utility Functions & Archived Debugging (Not Used in This Notebook)](#utility-functions--archived-debugging-not-used-in-this-notebook)
- [Download & Load Datasets](#download--load-datasets)
  - [First Dataset Resource](#first-dataset-resource)
  - [Second Dataset Resource](#second-dataset-resource)
- [Path Datasets](#path-datasets)
  - [First Dataset](#first-dataset)
  - [Second Dataset](#second-dataset)
- [Merge & Prepare Datasets](#merge--prepare-datasets)
  - [Build Combined Dataset (Class Merge + Folder Setup)](#build-combined-dataset-class-merge--folder-setup)
  - [Combine The Datasets](#combine-the-datasets)
  - [Save Combo Set](#save-combo-set)
  - [Delete Raw Datasets to Reclaim CUDA Memory](#delete-raw-datasets-to-reclaim-cuda-memory)
- [Resume From Checkpoint: Load Combined Class Info](#resume-from-checkpoint-load-combined-class-info)
- [Global Config: Paths, Splits, and Dataset Setup](#global-config-paths-splits-and-dataset-setup)
  - [Check Combined Folder Structure (Train/Valid/Test)](#check-combined-folder-structure-trainvalidtest)
  - [Stats & Class Distribution](#stats--class-distribution)
- [Mean & Std](#mean--std)
- [Classification Or Object Detection Model?](#classification-or-object-detection-model)
  - [Classification Task (e.g., Image Classification)](#classification-task-eg-image-classification)
  - [Regression Task (e.g., Object Detection or Localization)](#regression-task-eg-object-detection-or-localization)
- [MixUp Dataset Wrapper](#mixup-dataset-wrapper)
- [Transforms & Loaders](#transforms--loaders)
- [CNN Model](#cnn-model)
  - [Define Class Weights](#define-class-weights)
    - [Log-scaled Inverse Frequency Logic - Class Weighting Strategy](#log-scaled-inverse-frequency-logic---class-weighting-strategy)
    - [Visualization Bar Chart - Weights Assigned To Each Class](#visualization-bar-chart---weights-assigned-to-each-class)
  - [Define Lightweight Layer IF cublasLtMatmulAlgoGetHeuristic](#define-lightweight-layer-if-cublasltmatmulalgogetheuristic)
  - [Define CNN Model Architecture](#define-cnn-model-architecture)
  - [Weights & Biases](#weights--biases)
    - [Weight Initialization](#weight-initialization)
  - [Hyperparameters](#hyperparameters)
    - [Epochs, Learning Rate, Accumulation Steps](#epochs-learning-rate-accumulation-steps)
  - [Loss Function](#loss-function)
    - [Custom FocalLoss](#custom-focalloss)
  - [Mixed Precision Training - Autocast & GradScaler](#mixed-precision-training---autocast--gradscaler)
    - [Optimizer & Learning Rate Scheduler](#optimizer--learning-rate-scheduler)
  - [Profile Memory Usage](#profile-memory-usage)
  - [Training CNN Model](#training-cnn-model)
- [Save Metadata Alongside Best Weights](#save-metadata-alongside-best-weights)
- [Save Summary & Classification Report](#save-summary--classification-report)
- [Auto-Create Config](#auto-create-config)
- [Log Training History](#log-training-history)
- [Create ChangeLog](#create-changelog)
- [Create ChangeLog & Config](#create-changelog--config)
- [Save Model Weights or Save Model of Training](#save-model-weights-or-save-model-of-training)
- [Test Accuracy of CNN Training](#test-accuracy-of-cnn-training)
- [Visualizations](#visualizations)
  - [Metrics](#metrics)
    - [Summary](#summary)
    - [Evaluate Per Class Metrics](#evaluate-per-class-metrics)
    - [Model Performance](#model-performance)
    - [Check class representation in the dataset](#check-class-representation-in-the-dataset)
  - [Visuals](#visuals)
    - [Learning Rate Visual](#learning-rate-visual)
    - [F1-Score over Epochs Validation](#f1-score-over-epochs-validation)
    - [Training/Validation Curves](#trainingvalidation-curves)
    - [Training & Validation Loss Plots](#training--validation-loss-plots)
    - [Confusion Matrices](#confusion-matrices)
    - [Feature Maps](#feature-maps)
    - [Kernel/Visualizations](#kernelvisualizations)
    - [Gradient Visualization](#gradient-visualization)
    - [CAM/Grad-CAM](#camgrad-cam)
    - [Explainability Tools](#explainability-tools)
    - [Side-by-Side: Original, IG, and Grad-CAM](#side-by-side-original-ig-and-grad-cam)
    - [Compare with Target Class (Grad-CAM)](#compare-with-target-class-grad-cam)
- [Acknowledgements](#acknowledgements)
- [Environment](#environment)
- [Recommended Resources](#recommended-resources)
- [Permission](#permission)

## 🚀 Overview

This project evolves into a robust, memory-optimized, and modular CNN pipeline. After identifying and fixing earlier dataset merge issues, it features:

- Merged dataset of **1,804 images** across **18 classes**
- Class `person` makes up **~50%** of all samples
- Built entirely from scratch with **no pretrained models**
- Tuned using **MixUp**, **log-scaled class weights**, and **SoftTarget Focal Loss**
- Trained and validated on local hardware (Alienware RTX 3060 laptop)

---

## 📦 Dataset

- **INRIA Person Detection** (902 images, class: `person`)
- **Animal Image Dataset** (17 selected animal classes, 902 images)
- Structured into `train/valid/test` splits (70/20/10) for each class
- Final merged dataset lives in `data/comboset/` with:
  - `class_names.json`
  - `dataset_stats.json`

---

## 🏗️ Model Architecture

- Custom CNN using `nn.Sequential`
- Stack: `Conv → BatchNorm → GELU → Pooling`
- Custom FC head with `LeakyReLU`, dropout, and `ToFloat32()` fix for AMP
- Dynamic output class sizing
- Mixed precision-compatible

---

## 🧪 Key Features & Fixes

- ✅ Manual label offset to avoid class index collisions after merging datasets
- ✅ Dataset validation scripts to detect empty/missing folders
- ✅ Mean/std normalization with pixel-accurate two-pass calculation
- ✅ Caching of stats and transforms for restart efficiency
- ✅ Use of `WeightedRandomSampler` for balanced training batches
- ✅ Log-scaled inverse frequency class weights
- ✅ `MixUpDataset` wrapper for soft-label data augmentation
- ✅ Custom `SoftTargetFocalLoss` (supports `alpha`, `gamma`, label smoothing)
- ✅ Optimizer param grouping (exclude bias/batchnorm from decay)
- ✅ Early stopping + checkpointing by accuracy and F1
- ✅ Robust training loop with AMP, accumulation, and NaN protection

## ⚙️ Core Training Parameters

- **Epochs:** `200`
- **Optimizer:** `AdamW` with parameter grouping
- **LR Scheduler:** `ReduceLROnPlateau(mode='max')`
- **Loss Function:** `SoftTargetFocalLoss`
- **Augmentation Techniques:** `MixUp`, horizontal flips, normalization

---

## 📊 Results Comparison

| Metric                  | Capstone Model | Final Model   |
|-------------------------|----------------|---------------|
| **Test Accuracy**       | 39.6%          | **71%+**      |
| **Macro F1-Score**      | 0.23           | **0.52–0.60** |
| **Dominant Class Bias** | High           | Controlled    |

---

## 🖥️ Environment Comparison

| Feature       | Capstone (AWS SageMaker)  | Final Version (Local Laptop) |
|---------------|---------------------------|------------------------------|
| **GPU**       | NVIDIA T4 (16 GB)         | RTX 3060 Mobile (6 GB)       |
| **OS**        | Amazon Linux 2            | Pop!_OS 22.04 LTS            |
| **CUDA Ver**  | 11.x                      | 12.8                         |
| **RAM**       | 32 GB                     | 16 GB                        |

---

# **Environment**
- **Local Environment: My Laptop**
  - **Make & Model** Alienware m15 R7
  - **GPU:** NVIDIA GeForce RTX 3060 Mobile (6 GB VRAM)
  - **Secondary GPU:** Integrated AMD Radeon Graphics
    **CPU:** AMD Ryzen 7 6800H (16 threads @ 4.78 GHz)
  - **RAM:** 16 GB DDR5
  - **CUDA Version:** CUDA Version: 12.8
  - **Operating System:** Pop!_OS 22.04 LTS
  - **Kernel:** 6.12.10-76061203-generic

# **Recommended Resources**
<p>Resources that I found useful while working on a Computer Vision project and learning about Machine Learning & AI.</p>

<ul>
  <li>
    <a href="https://greenteapress.com/thinkpython2/thinkpython2.pdf" target="_blank">
      Think Python How to Think Like a Computer Scientist 2nd Edition, Version 2.4.0 by Allen Downey
    </a>
  </li>
    <li>
    <a href="https://github.com/dvgodoy/PyTorchStepByStep/blob/master/README.md" target="_blank">
      Deep Learning with PyTorch Step-by-Step: A Beginner's Guide: Volume I: Fundamentals by Daniel Voigt Godoy
    </a>
  </li>

  <li>
    <a href="https://link.springer.com/book/10.1007/978-1-4614-5323-9" target="_blank">
      Pattern Recognition and Classification An Introduction (Springer, 2012) by Geoff Dougherty
    </a>
  </li>

  <li>
    <a href="https://www.amazon.com/author/msoltys" target="_blank">
      An Introduction to the Analysis of Algorithms (3rd Ed, 2018) by Michael Soltys
    </a>
  </li>

  <li>
    <a href="https://docs.ultralytics.com/models/yolov8/" target="_blank">
      Ultralytics YOLOv8 Documentation
    </a>
  </li>

  <li>
    <a href="https://shisrrj.com/paper/SHISRRJ247267.pdf" target="_blank">
      Object Detection and Localization with YOLOv3 by B. Rupadevi and J. Pallavi
    </a>
  </li>

  <li>
    <a href="https://www.amazon.com/Digital-Image-Processing-Medical-Applications/dp/0521860857" target="_blank">
      Digital Image Processing for Medical Applications by Geoff Dougherty
    </a>
  </li>
</ul>

# **Creator Information**
No Guarantee that I will regularly upload things but if I do they would be at one of these locations.

- **GitHub:** 
  - https://github.com/ianhenryryan
- **LinkedIn:**
  - https://linkedin.com/in/ianhenryryan/
- **Websites:**
  - http://ianhryan.com/
- **Kaggle:**
  - https://kaggle.com/ianryan
- **Hugging Face:**
  - https://huggingface.co/Ianryan
 
# **Permission**
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<p>Students, educators, and anyone else keen on learning about Convolutional Neural Networks (CNNs), Computer Vision, Supervised Learning Algorithms, Machine Learning, Deep Learning, AI, or related fields are welcome to use this notebook in any capacity as a learning resource.</p>

<p>The first iteration was part of my Capstone Project for my Bachelors in Computer Science at California State University Channel Islands. I structured it in a digestible way for myself while getting comfortable with this area of AI/ML.</p>

<p>I hope this notebook can be of use to those exploring similar topics. This specific Jupyter Notebook focuses on Image Classification and demonstrates combining two datasets to create a class imbalance for training purposes.</p>

<h3><Strong>Important Note on Datasets:</Strong></h3>
<p>The datasets used in this project are not my property. Credit is given to the original dataset creators, and their links are provided within the notebook in the dataset section at the beginning.</p>

<p>I understand that grasping the fundamentals of CNNs and related AI concepts can be overwhelming at first. My goal is to make these topics more accessible through these notebooks.</p>

<h3>Permissions:</h3>
<p>You are free to download, use, edit, and reference the notebooks, Python code, and Markdown content. I aim for accuracy in the explanations provided, though I acknowledge that scientific understanding is always evolving. I welcome constructive feedback and corrections.</p>

<p>This project is intended as a learning resource.</p>

<p><strong>&mdash; Ian Ryan</strong></p>
</body>
</html>

## 🧾 Acknowledgments & Third-Party Libraries

This project uses the following open-source libraries under their respective licenses:

- [PyTorch](https://pytorch.org/) — BSD-3 License
- [Torchvision](https://github.com/pytorch/vision) — BSD-3 License
- [Scikit-learn](https://scikit-learn.org/) — BSD License
- [Matplotlib](https://matplotlib.org/) — PSF-based License
- [Seaborn](https://seaborn.pydata.org/) — BSD License
- [NumPy](https://numpy.org/) — BSD License
- [Pandas](https://pandas.pydata.org/) — BSD License
- [OpenCV](https://opencv.org/) — Apache License 2.0
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — AGPLv3 License
- [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) — MIT License
- [Captum](https://github.com/pytorch/captum) — BSD License
- [torchviz](https://github.com/szagoruyko/pytorchviz) — MIT License
- [torchsummary](https://github.com/sksq96/pytorch-summary) — MIT License

