<div align="center">

# 🚗 CarVision AI & FireGuard
### Intelligent Vehicle Damage Inspection & Fire Hazard Segmentation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11m](https://img.shields.io/badge/Model-YOLO11m-orange.svg)](https://github.com/ultralytics/ultralytics)
[![YOLO11n-seg](https://img.shields.io/badge/Model-YOLO11n--seg-red.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)

**An end-to-end, unified Computer Vision platform combining high-precision exterior vehicle defect detection with real-time fire and hazard instance segmentation.**

</div>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Key Features](#-key-features)
- [Detectable Damage Classes](#-detectable-damage-classes)
- [Installation & Quickstart](#-installation--quickstart)
- [Running the Project](#-running-the-project)
  - [1. Interactive Master Launcher](#1-interactive-master-launcher-easiest)
  - [2. Real-Time Video Stream with Slider & Auto-Screenshot](#2-real-time-video-stream-with-slider--auto-screenshot)
  - [3. Single or Batch Image Inspection (Neon Polygon Contours)](#3-single-or-batch-image-inspection-neon-polygon-contours)
  - [4. Fire & Smoke Hazard Segmentation](#4-fire--smoke-hazard-segmentation)
  - [5. Full Video MP4 Annotation & Export](#5-full-video-mp4-annotation--export)
  - [6. Modern Streamlit Web Dashboard](#6-modern-streamlit-web-dashboard)
  - [7. Desktop App (Tkinter Image Browser)](#7-desktop-app-tkinter-image-browser)
- [CLI Reference](#-cli-reference)

---

## 🔍 Overview

**CarVision AI & FireGuard** solves the challenge of automated vehicle intake, insurance claims appraisal, fleet return auditing, and accident hazard assessment:
1. **Vehicle Damage Detection:** Powered by fine-tuned **YOLO11m** (~20M parameters), identifying sheet metal deformities, scratches, cracks, broken lamps, shattered glass, and flat tires.
2. **Neon Polygon Contours:** Picture-perfect defect boundary extraction using edge detection and semi-transparent alpha overlays.
3. **Hazard & Fire Segmentation:** Powered by fine-tuned **YOLO11n-seg** for instant fire hazard detection with pixel-level instance masks.
4. **Unified Multi-Interface Support:** Run via Console Menu, CLI, OpenCV Live GUI, Desktop App, or Web Browser.

---

## 📁 Project Architecture

```text
d:\car demage\
│
├── models/                           # 🧠 Centralized AI model weights
│   ├── car_damage_yolo11m.pt        # 40.5 MB fine-tuned YOLO11m vehicle damage model
│   ├── fire_seg_yolo11n.pt          # 6.0 MB fine-tuned YOLO11n fire segmentation model
│   └── mobile_model.h5              # 105.7 MB vehicle classification weights
│
├── data/                             # 📥 Input media & test assets
│   ├── images/                      # Sample vehicle & fire images
│   ├── videos/                      # Sample inspection videos (car.mp4, etc.)
│   └── public/                      # Sample reference assets
│
├── src/                              # ⚙️ Reusable core AI package
│   ├── __init__.py                  # High-level package exports
│   ├── config.py                    # Central paths, thresholds, classes, neon colors
│   ├── damage_detector.py           # CarDamageDetector class (YOLO11m)
│   ├── fire_segmenter.py            # FireSegmenter class (YOLO11n-seg)
│   ├── video_pipeline.py            # Hardware frame seek, auto-screenshot, batch video export
│   └── utils/
│       ├── __init__.py
│       ├── visualizer.py            # Neon polygon contours, bounding boxes, telemetry HUD
│       └── media_loader.py          # Intelligent path resolution and media I/O
│
├── apps/                             # 🖥️ User interfaces & entry points
│   ├── __init__.py
│   ├── cli.py                       # Unified Command Line Interface
│   ├── live_gui.py                  # Realtime OpenCV GUI with dynamic trackbar slider
│   ├── desktop_app.py               # Tkinter desktop folder/image damage inspector
│   └── web_dashboard.py             # Streamlit interactive web application
│
├── outputs/                          # 📤 Centralized export directory
│   ├── images/                      # Annotated result images
│   ├── videos/                      # Rendered annotated MP4 videos
│   └── screenshots/                 # Auto & manual captured inspection screenshots
│
├── notebooks/                        # 📓 Research and training notebooks
│   ├── YOLO11m_trained.ipynb
│   ├── YOLOv8_training.ipynb
│   └── Faster_RCNN.ipynb
│
├── run.py                            # 🚀 Master runner (Interactive Console Menu + CLI)
├── requirements.txt                  # 📦 Clean Python dependencies
└── README.md                         # 📖 Complete project guide
```

---

## ✨ Key Features

- 🎯 **Dual AI Models:** YOLO11m for damage detection + YOLO11n for fire segmentation.
- 🎛️ **Live Trackbar Slider:** Adjust detection confidence threshold on the fly during video playback.
- ⚡ **Zero-Latency Direct Seek:** Hardware-level 5th-frame seeking eliminates CPU decoding overhead.
- 📸 **Periodic Auto-Screenshot:** Automatically captures and timestamps damage detections every 30s into `outputs/screenshots/`.
- 🌐 **Modern Web Dashboard:** Drag-and-drop web UI powered by Streamlit with side-by-side inspection views.

---

## 🏷️ Detectable Damage Classes

| Class | Color | Description |
| :--- | :--- | :--- |
| **`dent`** | Bright Yellow | Sheet metal surface depressions and panel impact |
| **`scratch`** | Neon Green | Paint layer scratches, scuffs, and abrasions |
| **`crack`** | Neon Purple | Bumper, fender, or body panel fracture lines |
| **`broken_lamp`** | Sky Blue / Cyan | Damaged headlights, taillights, or indicators |
| **`shattered_glass`**| Vibrant Orange | Broken windshield, side windows, or sunroof |
| **`flat_tire`** | Pinkish Red | Deflated, punctured, or damaged wheels |

---

## 🚀 Installation & Quickstart

```bash
# 1. Clone or navigate to the project directory
cd "d:\car demage"

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Running the Project

### 1. Interactive Master Launcher (Easiest)
Aap simple command run karein:
```bash
python run.py
```
Yeh command chalate hi console mein interactive menu khul jayega:
```text
============================================================
🚗  CarVision AI: Vehicle Damage & Hazard Inspection System
============================================================
 [1] Real-Time Video Damage Inspection (Live GUI + Slider)
 [2] Image Damage Inspection (Neon Polygon Contours)
 [3] Fire & Smoke Hazard Instance Segmentation
 [4] Full Video Batch Annotation (Export MP4)
 [5] Launch Web Dashboard (Streamlit UI in Browser)
 [6] Launch Desktop App (Tkinter Image Browser)
 [0] Exit
============================================================
```

---

### 2. Real-Time Video Stream with Slider & Auto-Screenshot
Video par live damage detection chalayein:
```bash
python run.py --mode damage-video --source data/videos/car.mp4
```
**Controls:**
- **Slider:** Drag slider to change Confidence Threshold (0% to 100%).
- **S key:** Manual instant screenshot lena.
- **Q / ESC key:** Window close karna.

---

### 3. Single or Batch Image Inspection (Neon Polygon Contours)
Kisi bhi image par damage detect karein:
```bash
python run.py --mode damage-image --source data/images/download_1.jpg
```
*Output image automatically `outputs/images/` mein save ho jayegi.*

---

### 4. Fire & Smoke Hazard Segmentation
Fire hazard segmentation chalayein:
```bash
python run.py --mode fire-image --source data/images/download_2.jpg
```

---

### 5. Full Video MP4 Annotation & Export
Puri video ko process karke annotated MP4 save karein:
```bash
python run.py --mode annotate-video --source data/videos/car.mp4
```
*Rendered video `outputs/videos/car_annotated.mp4` mein save hogi.*

---

### 6. Modern Streamlit Web Dashboard
Browser mein drag-and-drop web dashboard chalane ke liye:
```bash
streamlit run apps/web_dashboard.py
```
Ya menu se option `[5]` select karein. Browser automatically `http://localhost:8501` par open ho jayega.

---

### 7. Desktop App (Tkinter Image Browser)
Standalone desktop GUI chalane ke liye:
```bash
python apps/desktop_app.py
```

---

## 🛠️ CLI Reference

| Flag | Description | Example |
| :--- | :--- | :--- |
| `--mode` | `damage-video`, `damage-image`, `fire-image`, `annotate-video` | `--mode damage-image` |
| `--source` | Image/Video path ya camera index `0` | `--source data/videos/car.mp4` |
| `--conf` | Confidence threshold float (0.05 to 1.0) | `--conf 0.35` |
| `--polygon` | Precise neon polygon contours toggle (default: True) | `--polygon` |
| `--bbox-only`| Only show bounding boxes | `--bbox-only` |
| `--step` | Video frame skip seek step (default: 5) | `--step 5` |
| `--no-show` | Headless mode (bina GUI window ke process karna) | `--no-show` |
