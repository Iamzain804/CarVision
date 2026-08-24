<div align="center">

# 🚗 CarVision: AI Vehicle Damage Inspection & Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11m](https://img.shields.io/badge/Model-YOLO11m-orange.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An intelligent end-to-end Computer Vision system powered by custom fine-tuned YOLO11m for automated vehicle body damage inspection, real-time GUI detection, and batch video annotation.**

</div>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Detectable Damage Classes](#-detectable-damage-classes)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [1. Real-Time Interactive GUI](#1-real-time-interactive-gui-with-slider--auto-screenshot)
  - [2. Image Damage Detection](#2-single--batch-image-damage-detection)
  - [3. Full Video Annotation](#3-full-video-annotation-pipeline)
  - [4. Direct Low-Latency Frame Seek](#4-low-latency-5th-frame-inference)
  - [5. Jupyter Notebook](#5-jupyter-notebook)
- [License](#-license)

---

## 🔍 Overview

**CarVision** is designed to assist automotive service centers, insurance pre-claim imaging, rental car return auditing, and fleet condition inspections. The model automates the identification and classification of visible exterior vehicle damages to prevent disputes and accelerate intake workflows.

---

## ✨ Key Features

- 🎯 **Fine-Tuned YOLO11m:** ~20M parameters optimized for high-precision vehicle defect classification.
- 🎛️ **Interactive Real-Time GUI:** Live OpenCV playback with an on-the-fly confidence score slider (0%–100%).
- 📸 **Automated Periodic Logging:** Automatically captures and timestamps damage screenshots every 30 seconds into an isolated `screenshots/` directory.
- 🎥 **HD Video Annotation:** Batch process and render fully annotated MP4 inspection videos with bounding boxes and confidence levels.
- ⚡ **Zero-Latency Direct Seek:** Low-overhead frame extraction without CPU decoding bottlenecks.
- 📦 **Modern Dependency Management:** Fast and isolated virtual environment setup using `uv`.

---

## 🏷️ Detectable Damage Classes

| Class | Description |
| :--- | :--- |
| **`dent`** | Body panel surface depression and sheet metal deformation |
| **`scratch`** | Paint layer scratches, abrasions, and scuffs |
| **`crack`** | Bumper, grill, or windshield fracture lines |
| **`shattered_glass`** | Window, windshield, or sunroof breakage |
| **`broken_lamp`** | Headlight, taillight, or turn indicator damage |
| **`flat_tire`** | Deflated, punctured, or damaged wheel condition |

---

## 📊 Model Performance

| Class | Box Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **shattered_glass** | **0.979** | **0.978** | **0.994** | **0.963** |
| **flat_tire** | **0.943** | **0.919** | **0.959** | **0.932** |
| **broken_lamp** | **0.826** | **0.821** | **0.895** | **0.796** |
| **scratch** | **0.737** | **0.800** | **0.905** | **0.610** |
| **dent** | **0.832** | **0.520** | **0.692** | **0.568** |
| **crack** | **0.699** | **0.586** | **0.620** | **0.424** |

---

## 📂 Project Structure

```text
CarVision/
├── trained.pt                   # Custom fine-tuned YOLO11m weights (~40MB)
├── gui_detector.py              # Interactive GUI with live threshold slider & auto-screenshot
├── create_annotated_video.py    # Full video annotation pipeline to MP4
├── predict.py                   # Image prediction script
├── video_inference.py           # Low-latency direct 5th frame inference
├── stream_every_5th_frame.py    # Fast video stream frame skipping
├── YOLO11m_trained.ipynb        # Jupyter Notebook for experiments & Colab
├── requirements.txt             # Python dependencies
├── public/                      # Sample inspection images
├── screenshots/                 # Auto-saved inspection screenshots (gitignored)
└── annotated_output/            # Exported annotated videos (gitignored)
```

---

## 🚀 Installation & Environment Setup (using `uv`)

This project uses [**`uv`**](https://github.com/astral-sh/uv) (an extremely fast Python package manager) to create the isolated virtual environment (`.venv`) and install dependencies in seconds.

### Step 1: Clone the Repository
```bash
git clone git@github.com:Iamzain804/CarVision.git
cd CarVision
```

### Step 2: Install `uv` (if not already installed)
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3: Create `.venv` Virtual Environment
Create the dedicated virtual environment inside the project directory:
```bash
uv venv .venv
```

### Step 4: Activate the Virtual Environment
```bash
# On Windows (PowerShell / CMD)
.\.venv\Scripts\activate

# On macOS / Linux
source .venv/bin/activate
```

### Step 5: Install Requirements with `uv`
Install all dependencies (PyTorch, Ultralytics, OpenCV, etc.) at lightning speed:
```bash
uv pip install -r requirements.txt
```

> **Note:** Once activated, `(.venv)` will appear in your terminal prompt, indicating all scripts will run inside this isolated environment.

---

## 💻 Usage Guide

### 1. Real-Time Interactive GUI (with Slider & Auto-Screenshot)
Opens an interactive window with live detection, a confidence slider, and auto-screenshot capture every 30 seconds:
```bash
python gui_detector.py
```
* **Slider:** Adjust confidence threshold in real-time.
* **`S` key:** Instant manual screenshot.
* **`Q` / `ESC`:** Exit.

---

### 2. Single / Batch Image Damage Detection
Run inference on any vehicle photo:
```bash
# Test sample image
python predict.py public/1.png

# Test your custom image
python predict.py "path/to/car_image.jpg"
```

---

### 3. Full Video Annotation Pipeline
Process an entire inspection video and generate an annotated MP4 video with bounding boxes and labels:
```bash
python create_annotated_video.py
```
Output saved to: `annotated_output/annotated_car.mp4`

---

### 4. Low-Latency 5th Frame Inference
Extract and detect damage on the 5th frame without decoding preceding frames:
```bash
python video_inference.py
```

---

### 5. Jupyter Notebook
Launch the notebook for interactive step-by-step execution:
```bash
jupyter notebook YOLO11m_trained.ipynb
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
