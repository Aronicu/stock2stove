<p align="center">
  <img src="images/logo.png" alt="Team Kaggleing Logo" width="180" />
</p>

<h1 align="center">Stock-to-Stove: Automated Pantry Inventory Detection & Recipe Suggestion</h1>

<p align="center">
  <strong>CSST 106 — Computer Vision Final Project</strong><br/>
  <strong>Milestone 4 — 80% Progress Check (<code>cp4-80</code>)</strong> · <strong>Team Kaggleing</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-black.svg" alt="Python 3.10+" /></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.8+-gray.svg" alt="OpenCV" /></a>
  <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/Ultralytics-YOLOv8n-lightgray.svg" alt="YOLOv8n" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-Server%20%2B%20Web%20UI-009688.svg" alt="FastAPI" /></a>
  <a href="android/"><img src="https://img.shields.io/badge/Android-CameraX%20%2B%20WebSocket-3DDC84.svg" alt="Android" /></a>
</p>

**Members:**
- **Ogayon, Aron Gabriel L.** — Team Lead
- **Kumar, Allain A.** — Documentation
- **Sarmiento, Janvher Lucas J.** — Programmer

---

## 📌 Project Overview
**Stock-to-Stove** solves the common kitchen dilemma: *"A cupboard full of food, but you don't know what to cook."* Scanning kitchen pantry and refrigerator shelves via a live smartphone camera, uploaded video, or web dashboard isolates, classifies, and tallies in-stock grocery items across dark cupboard shadows (`<100 lux`), clutter, and camera motion. Confirmed inventory counts are passed to a downstream LLM recipe generator to suggest 3 actionable, zero-waste recipes.

- **Unified 8-Class Taxonomy:** Pantry (`pasta_noodles`, `canned_tomato`, `canned_tuna_meat`, `cereal_grains`, `coffee_condiment`) and Refrigerator (`milk_dairy`, `eggs`, `fresh_produce`).
- **Luminance-Gated Preprocessing (`src/cv/preprocess.py`):** Aspect-preserving `640x640` letterboxing (`pad=114`) with inverse gamma + LAB L-channel CLAHE triggered when mean lightness `L < 45.0`.
- **Detectors:** Fine-tuned **YOLOv8n** (`models/best.pt`, `6.2 MB` PT / `12.19 MB` ONNX), classical **ORB / SIFT + RANSAC** baselines (`src/cv/baseline_detector.py`), plus **Gemini Vision** and **Hybrid Gemini** server modes (`src/cv/gemini_detector.py`).
- **Temporal Confirmation (`src/cv/temporal_tracker.py`):** Class-wise NMS (`IoU=0.45`), meat-safety confidence gate (`canned_tuna_meat >= 0.76`), and a 5-frame sliding window (`K >= 3` hits).
- **Applications:** FastAPI inference server & browser dashboard (`src/server/app.py`), native Android CameraX WebSocket app (`android/`, prebuilt `app-debug.apk`), CLI pipeline (`src/run.py`), and LLM recipe module (`src/llm_recipe.py`).

---

## 📂 Repository Structure
```text
stock2stove/
├── README.md                        # Project overview, results, and run guide
├── requirements.txt                 # Python dependencies
├── run_server.ps1                   # FastAPI server launcher (with optional ADB USB reverse)
├── app-debug.apk                    # Prebuilt Android debug APK
├── images/
│   └── logo.png                     # Team Kaggleing logo
├── android/                         # Native Kotlin + Jetpack Compose + CameraX Android app
├── data/
│   ├── dataset_yolo/                # Unified 8-class YOLO dataset & pantry_negatives/
│   ├── raw_home/                    # Demo pan clip (demo_pantry_pan.mp4)
│   ├── samples/                     # Benchmark pantry sample images
│   └── templates/                   # ORB/SIFT reference templates
├── models/
│   ├── best.pt                      # Evaluated fine-tuned YOLOv8n weights (6.2 MB)
│   ├── yolov8n_stock2stove.pt       # Fine-tuned YOLOv8n PyTorch checkpoint
│   └── yolov8n_stock2stove.onnx     # Exported ONNX weights (12.19 MB)
├── src/
│   ├── run.py                       # Unified CLI pipeline (preprocess -> infer -> postprocess -> render)
│   ├── evaluate.py                  # Held-out benchmark & Milestone 4 artifact generator
│   ├── llm_recipe.py                # Zero-waste recipe generator from confirmed inventory
│   ├── cv/                          # Preprocessing, YOLOv8, ORB/SIFT baselines, Gemini, tracker
│   ├── server/                      # FastAPI app, Web UI (/), REST & /ws/scan WebSocket endpoints
│   └── training/                    # Dataset merging, auto-labeling, YOLOv8 training & eval scripts
├── tests/
│   ├── test_phase1.py               # Backend & CV pipeline unit/integration tests
│   ├── test_phase2.py               # Phase 2 integration tests
│   └── run_e2e_tests.ps1            # Android Gradle unit & E2E test runner
├── outputs/                         # Evaluation metrics, confusion matrix, spike & run outputs
└── docs/
    ├── milestone3/                  # Milestone 3 (cp3-core) report & evaluation artifacts
    ├── milestone4/                  # Milestone 4 (cp4-80) report, slides, metrics & failure gallery
    ├── CSST106_Milestone1_Proposal_Stock2Stove.pdf
    └── CSST106_Milestone2_ProgressCheck_Stock2Stove.docx
```

---

## 🚀 Quickstart

### 1. Installation
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the FastAPI Server & Web Dashboard
```powershell
.\run_server.ps1
```
Or directly via Python:
```powershell
python src/server/app.py
```
- **Web UI:** `http://127.0.0.1:8000/`
- **API Endpoints:** `GET /api/v1/health`, `POST /api/v1/scan/image`, `POST /api/v1/scan/video`, `POST /api/v1/recipes/generate`, `POST /api/v1/session/reset`, and `WS /ws/scan`.

### 3. Run the CLI Pipeline (`src/run.py`)
```powershell
python src/run.py --input data/test/ --output outputs/run_latest/
```
Supports image folders, single images, video files (e.g. `--input demo.mp4`), or live webcam (`--webcam 0`), with `--mode` (`yolov8`, `orb_baseline`, `sift_baseline`, `yolo_world`, `gemini_vision`, `hybrid_gemini`) and `--generate-recipes`.

### 4. Reproduce Evaluation & Run Tests
```powershell
python src/evaluate.py --split test --weights models/best.pt
.\tests\run_e2e_tests.ps1
```

---

## 📊 Milestone 4 Evaluation Summary (`cp4-80`)
Evaluated on the locked held-out test set (`104` images, `261` GT boxes, `conf=0.45`, `eval IoU=0.50`) using `models/best.pt`:

| Condition | ORB mAP@0.50 | SIFT mAP@0.50 | YOLOv8n mAP@0.50 | YOLOv8n F1 | YOLOv8n Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Normal | 0.0109 | 0.0246 | **0.8190** | 0.8544 | 42.28 ms (23.65 FPS) |
| Shadow (`<100 lux`) | 0.0182 | 0.0259 | **0.6750** | 0.7906 | 56.55 ms (17.68 FPS) |
| Motion Blur (`k=11`) | 0.0000 | 0.0000 | **0.5334** | 0.6856 | 42.69 ms (23.42 FPS) |
| Angled (`±16°`) | 0.0115 | 0.0259 | **0.5878** | 0.7489 | 50.14 ms (19.95 FPS) |
| Occluded (`35%`) | 0.0000 | 0.0000 | **0.3919** | 0.5398 | 48.38 ms (20.67 FPS) |
| Cluttered (`>=4 items`) | 0.0000 | 0.0000 | **0.8665** | 0.8975 | 45.20 ms (22.12 FPS) |

- **Video Pan (`demo_pantry_pan.mp4`):** `7/9` confirmed items at `19.63 FPS` (`50.94 ms/frame`).
- **False-Meat Safety (`pantry_negatives`):** `1/115` (`0.0087`).
- Full details, failure gallery, report, and slides are in [`docs/milestone4/`](docs/milestone4/README.md).
