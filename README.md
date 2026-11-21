<p align="center">
  <img src="images/logo.png" alt="Team Kaggleing Logo" width="180" />
</p>

<h1 align="center">Stock-to-Stove: Automated Pantry Inventory Detection & Recipe Suggestion</h1>

<p align="center">
  <strong>CSST 106 — Computer Vision Final Project</strong><br/>
  <strong>React + Vite Web App SPA + Gemini Vision</strong> · <strong>Team Kaggleing</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-black.svg" alt="Python 3.10+" /></a>
  <a href="frontend/"><img src="https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite%20PWA-61DAFB.svg" alt="React + Vite PWA" /></a>
  <a href="src/cv/gemini_detector.py"><img src="https://img.shields.io/badge/Vision-Gemini%20Interactions%20API-4285F4.svg" alt="Gemini Vision" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688.svg" alt="FastAPI" /></a>
  <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/Fallback-YOLOv8n-lightgray.svg" alt="YOLOv8n" /></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.8+-gray.svg" alt="OpenCV" /></a>
</p>

**Members:**
- **Ogayon, Aron Gabriel L.** — Team Lead
- **Kumar, Allain A.** — Documentation
- **Sarmiento, Janvher Lucas J.** — Programmer

---

## 📌 Project Overview
**Stock-to-Stove** solves the common kitchen dilemma: *"A cupboard full of food, but you don't know what to cook."* Scanning kitchen pantry and refrigerator shelves via a mobile or desktop browser camera, uploaded photo, or video stream in the **React + Vite Web App SPA (installable PWA)** isolates, classifies, tallies, and estimates calories for in-stock grocery items across dark cupboard shadows (`<100 lux`), clutter, and camera motion. Confirmed inventory counts are passed to a **Gemini Interactions API** recipe generator to suggest 3 actionable, zero-waste recipes.

- **Primary Application — React + Vite PWA SPA (`frontend/`):** Mobile- and desktop-responsive Progressive Web App served directly by FastAPI from `frontend/dist/`. Features live WebRTC camera streaming (`getUserMedia` with rear/front camera toggle), photo/video upload scanning, quick detector switching (`gemini_vision` default), real-time bounding-box HUD overlays, live API vs. offline fallback status badges, `localStorage` basket persistence, system diagnostics, and interactive zero-waste recipe cards.
- **Unified 8-Class Taxonomy:** Pantry (`pasta_noodles`, `canned_tomato`, `canned_tuna_meat`, `cereal_grains`, `coffee_condiment`) and Refrigerator (`milk_dairy`, `eggs`, `fresh_produce`), with explicit prompt/schema filtering against non-food objects and session-excluded items.
- **Luminance-Gated Preprocessing (`src/cv/preprocess.py`):** Aspect-preserving `640x640` letterboxing (`pad=114`) with inverse gamma + LAB L-channel CLAHE triggered when mean lightness `L < 45.0`.
- **Detectors:** **Gemini Vision** (`src/cv/gemini_detector.py`, default `gemini-4-hydrogen` via Google GenAI Interactions API with `thinking_level`, structured `class_id` enum, and per-item `estimated_calories`), **Hybrid Gemini** (YOLOv8n + Gemini Vision fusion), fine-tuned **YOLOv8n** (`models/best.pt`, `6.2 MB` PT / `12.19 MB` ONNX, also serving as offline fallback when `GEMINI_API_KEY` is unavailable), and classical **ORB / SIFT + RANSAC** baselines (`src/cv/baseline_detector.py`).
- **Temporal Confirmation (`src/cv/temporal_tracker.py`):** Class-wise NMS (`IoU=0.45`), meat-safety confidence gating (`canned_tuna_meat`), and sliding-window consensus with per-item calorie aggregation.
- **Backend & CLI (`src/server/app.py`, `src/run.py`, `src/llm_recipe.py`):** FastAPI server hosting the SPA plus REST and WebSocket endpoints, unified CLI pipeline, and structured recipe generation (`gemini-4-hydrogen`).

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------+
|  Browser / Mobile PWA Client (React 19 + Vite SPA in frontend/)       |
|  - Live Camera (getUserMedia rear/front) · Photo & Video Upload       |
|  - Mode Switcher (gemini_vision default, hybrid_gemini, yolov8, ...)  |
|  - Live HUD Overlay · API Status Badge · localStorage Basket Sync     |
+-------------------------------+---------------------------------------+
                                |  WS /ws/scan  &  HTTP /api/v1/*
                                v
+-----------------------------------------------------------------------+
|  FastAPI Server & SPA Host (src/server/app.py)                        |
|  - Serves frontend/dist/ (/, /assets, /icons, /manifest.webmanifest)  |
|  - Session Manager (confirmed inventory, exclusions, calories)        |
+-------------------------------+---------------------------------------+
                                |
          +---------------------+---------------------+
          v                                           v
+------------------------------------+     +------------------------------+
|  CV Detection & Tracking           |     |  Zero-Waste Recipe LLM       |
|  - preprocess.py (CLAHE+640px)     |     |  - llm_recipe.py             |
|  - gemini_detector.py              |     |  - Gemini Interactions API   |
|    (Interactions API + fallback)   |     |    (gemini-4-hydrogen)       |
|  - yolo_detector.py (YOLOv8n)      |     |  - Offline template fallback |
|  - baseline_detector.py (ORB/SIFT) |     +------------------------------+
|  - temporal_tracker.py (NMS+vote)  |
+------------------------------------+
```

---

## 📂 Repository Structure
```text
stock2stove/
├── README.md                        # Project overview, architecture, quickstart, and benchmarks
├── requirements.txt                 # Python dependencies
├── run_server.ps1                   # Windows PowerShell launcher (builds SPA + starts FastAPI)
├── run_server.sh                    # Bash launcher (venv setup, SPA build + FastAPI server)
├── images/
│   └── logo.png                     # Team Kaggleing logo & PWA icon source
├── frontend/
│   ├── index.html                   # SPA entry HTML with PWA & mobile viewport metadata
│   ├── package.json                 # React 19 + Vite scripts and dependencies
│   ├── vite.config.js               # Vite build & /api /ws dev proxy configuration
│   ├── public/                      # PWA manifest.webmanifest, favicon.png, icons/
│   ├── src/
│   │   ├── App.jsx                  # Root SPA shell, basket persistence & session sync
│   │   ├── api/client.js            # REST & WebSocket client for FastAPI backend
│   │   ├── components/              # ScannerView, BasketView, RecipesView, DiagnosticsModal, HeaderBar
│   │   └── constants/catalog.js     # Detector modes & 8-class catalog metadata
│   └── dist/                        # Production Vite bundle served by FastAPI
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
│   ├── llm_recipe.py                # Zero-waste recipe generator via Gemini Interactions API
│   ├── cv/                          # Preprocessing, Gemini Vision, YOLOv8, ORB/SIFT, temporal tracker
│   ├── server/                      # FastAPI app (SPA + REST + /ws/scan), config, schemas, sessions
│   └── training/                    # Dataset merging, auto-labeling, YOLOv8 training & eval scripts
├── tests/
│   ├── test_phase1.py               # Backend & CV pipeline unit/integration tests
│   ├── test_phase2.py               # Phase 2 integration tests
│   ├── test_reeval.py               # Gemini Vision, calorie & SPA backend tests
│   └── run_e2e_tests.ps1            # Backend pytest + frontend Vite production build runner
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
Install Node.js (`npm`) to build or develop `frontend/`. Both `run_server.ps1` and `./run_server.sh` install `frontend/node_modules` and rebuild `frontend/dist` automatically when needed.

### 2. Configure Gemini API Key (Recommended)
Set `GEMINI_API_KEY` to enable live **Gemini Vision** detection (`gemini-4-hydrogen`) and **Gemini Interactions API** recipe generation. Without a key, the server runs in explicit `Offline Fallback` mode using local YOLOv8n and offline recipe templates.
```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key"
```
Optional overrides:
- `STOCK2STOVE_GEMINI_VISION_MODEL` (default `gemini-4-hydrogen`; supports e.g. `gemini-3.5-flash-lite` for ultra-low latency)
- `STOCK2STOVE_GEMINI_RECIPE_MODEL` (default `gemini-4-hydrogen`)
- `STOCK2STOVE_GEMINI_THINKING_LEVEL` (`minimal`, `low`, `medium`, `high`)

### 3. Launch the Web App SPA & FastAPI Server
Windows (PowerShell):
```powershell
.\run_server.ps1
```
Linux / macOS (Bash):
```bash
./run_server.sh
```
Hot-reload development mode (FastAPI on `:8000` + Vite dev server on `:5173`):
```powershell
.\run_server.ps1 -Dev
```
Or launch FastAPI directly with a prebuilt `frontend/dist`:
```powershell
python src/server/app.py
```
- **Web App SPA (PWA):** `http://127.0.0.1:8000/`
- **API & WebSocket Endpoints:** `GET /api/v1/health`, `POST /api/v1/ui/detect`, `POST /api/v1/scan/image`, `POST /api/v1/scan/video`, `POST /api/v1/recipes/generate`, `POST /api/v1/session/reset`, `POST /api/v1/session/items/delete`, `POST /api/v1/session/items/restore`, `DELETE /api/v1/session/items/{class_id}`, and `WS /ws/scan`.

### 4. Run the CLI Pipeline (`src/run.py`)
```powershell
python src/run.py --input data/test/ --output outputs/run_latest/
```
Supports image folders, single images, video files (e.g. `--input demo.mp4`), or live webcam (`--webcam 0`), with `--mode` (`gemini_vision`, `hybrid_gemini`, `yolov8`, `orb_baseline`, `sift_baseline`, `yolo_world`) and `--generate-recipes`.

### 5. Reproduce Evaluation & Run Tests
```powershell
python src/evaluate.py --split test --weights models/best.pt
.\tests\run_e2e_tests.ps1
```
To run checks individually:
```powershell
python -m pytest -v --basetemp=.pytest_tmp
cd frontend && npm run build
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
