# Stock-to-Stove: Draft Technical Report (Milestone 4 — `cp4-80`)

**Course:** CSST 106 — Computer Vision Final Project  
**Team Kaggleing:** Aron Gabriel L. Ogayon (Lead), Allain A. Kumar (Documentation), Janvher Lucas J. Sarmiento (Developer)  
**Evaluated Weights:** `models/best.pt` (`6.2 MB` PyTorch, `12.19 MB` ONNX)  
**Demo Hardware:** AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (16 CPU threads, Windows 11)  

---

## 1. Introduction & Problem Statement
Managing home pantry and refrigerator inventory is manual, error-prone, and visually cluttered. Forgotten rear-shelf items lead to duplicate grocery purchases and preventable food waste, while decision fatigue drives unnecessary takeout orders. Barcode-only apps fail on loose produce, repackaged grains, and occluded containers.

**Objectives:**
1. Build a unified 8-class pantry and refrigerator object detection pipeline robust to dark cupboard shadows (`<100 lux`), shelf clutter, and camera pan motion.
2. Benchmark classical planar feature matching (`ORB` and `SIFT` + `RANSAC`) against a fine-tuned single-stage deep detector (`YOLOv8n`) on a locked held-out test set.
3. Achieve real-time CPU throughput (`>=15 FPS`) using a single cached model instance inside an interactive browser web dashboard (`GET /`), native Android CameraX app, and CLI (`src/run.py`).
4. Connect confirmed ingredient counts to a downstream recipe generator producing three zero-waste cooking suggestions.

---

## 2. Dataset & Preprocessing
The dataset (`data/dataset_yolo/data.yaml`) unifies 8 domestic grocery categories across Pantry (`pasta_noodles`, `canned_tomato`, `canned_tuna_meat`, `cereal_grains`, `coffee_condiment`) and Refrigerator (`milk_dairy`, `eggs`, `fresh_produce`) zones.

| Partition | Images | GT Boxes | Negatives | Composition |
| :--- | :---: | :---: | :---: | :--- |
| Train | 605 | 1,937 | 74 empty | Freiburg (225), Kitchen-YOLO (146), Fridge (76), Composite (65), Negatives (74), Pantry (19) |
| Validation | 104 | 282 | 13 empty | Freiburg (57), Kitchen-YOLO (18), Fridge (12), Pantry (4), Negatives (13) |
| Held-Out Test (Locked) | 104 | 261 | 13 empty | Freiburg (57), Kitchen-YOLO (18), Fridge (12), Pantry (4), Negatives (13) |
| Pantry Negatives | 115 | 0 | 115 | Non-meat pantry safety benchmark |
| Video Pan Clip | 60 frames | 9 shelf items | Synthetic | `data/raw_home/demo_pantry_pan.mp4` (`640x480` @ 15 FPS) |

**Preprocessing (`src/cv/preprocess.py`):** Every frame passes through luminance-gated low-light restoration and aspect-preserving `640x640` letterboxing (padding value `114`). When mean LAB lightness `L < 45.0`, inverse gamma restoration `(img / 0.26)^(1/2.4)` and L-channel CLAHE (`clipLimit=1.8`, `8x8` grid) recover shadow detail; well-lit frames (`L >= 45.0`) bypass CLAHE to prevent contrast drift against training distributions.

---

## 3. Methodology & Integrated Application
The unified procedural pipeline (`src/run.py`) executes `preprocess() -> infer() -> postprocess() -> render()`:
- **Workflow Diagram:** ![Workflow Diagram](workflow_diagram.png)
- **Classical Baselines (`src/cv/baseline_detector.py`):** Extracts 1,000 ORB or SIFT keypoints, matches multi-angle templates with Lowe's ratio test (`0.75`), and estimates 4-point homography via RANSAC (`min_inliers=6`).
- **Main Detector (`src/cv/yolo_detector.py`):** Fine-tuned YOLOv8n (`models/best.pt`) cached once in `MODEL_REGISTRY`. Also supports `hybrid_gemini` and `gemini_vision` modes in the server.
- **Post-Processing (`src/cv/temporal_tracker.py`):** Class-wise NMS (`IoU=0.45`), meat-safety confidence gate (`canned_tuna_meat >= 0.76`), and a 5-frame sliding window (`K >= 3` hits for video confirmation).
- **Application Interfaces:** Browser Dashboard (`GET /` in `src/server/app.py`), Native Android CameraX WebSocket App (`android/`), and CLI (`src/run.py`).

![Application Interface](application_interface_overview.png)

---

## 4. Experiments & Reproducible Results (`src/evaluate.py`)
Reproduced via `python src/evaluate.py --split test --weights models/best.pt` (`seed=42`, `conf=0.45`, `NMS IoU=0.45`, `eval IoU=0.50`).

### 4.1 Held-Out Test Set Metrics Across Conditions
| Condition | Method | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 | Speed (ms / FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Normal | ORB Baseline | 0.0308 | 0.0077 | 0.0123 | 0.0109 | 0.0083 | 2083.70 ms / 0.48 FPS |
| Normal | SIFT Baseline | 0.3636 | 0.0153 | 0.0294 | 0.0246 | 0.0112 | 522.85 ms / 1.91 FPS |
| Normal | YOLOv8n (Main) | 0.8661 | 0.8429 | 0.8544 | 0.8190 | 0.6454 | 42.28 ms / 23.65 FPS |
| Shadow (<100 lux) | ORB Baseline | 0.0678 | 0.0153 | 0.0250 | 0.0182 | 0.0089 | 2168.24 ms / 0.46 FPS |
| Shadow (<100 lux) | SIFT Baseline | 0.5556 | 0.0192 | 0.0370 | 0.0259 | 0.0131 | 600.97 ms / 1.66 FPS |
| Shadow (<100 lux) | YOLOv8n (Main) | 0.8080 | 0.7739 | 0.7906 | 0.6750 | 0.5217 | 56.55 ms / 17.68 FPS |
| Motion Blur (k=11) | ORB Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2055.85 ms / 0.49 FPS |
| Motion Blur (k=11) | SIFT Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 352.21 ms / 2.84 FPS |
| Motion Blur (k=11) | YOLOv8n (Main) | 0.7970 | 0.6015 | 0.6856 | 0.5334 | 0.4092 | 42.69 ms / 23.42 FPS |
| Angled (±16°) | ORB Baseline | 0.0476 | 0.0115 | 0.0185 | 0.0115 | 0.0074 | 2065.72 ms / 0.48 FPS |
| Angled (±16°) | SIFT Baseline | 0.6250 | 0.0192 | 0.0372 | 0.0259 | 0.0113 | 537.15 ms / 1.86 FPS |
| Angled (±16°) | YOLOv8n (Main) | 0.8607 | 0.6628 | 0.7489 | 0.5878 | 0.2567 | 50.14 ms / 19.95 FPS |
| Occluded (35%) | ORB Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2104.78 ms / 0.48 FPS |
| Occluded (35%) | SIFT Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 435.74 ms / 2.29 FPS |
| Occluded (35%) | YOLOv8n (Main) | 0.7273 | 0.4291 | 0.5398 | 0.3919 | 0.3245 | 48.38 ms / 20.67 FPS |
| Cluttered (>=4 items) | ORB Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2056.25 ms / 0.49 FPS |
| Cluttered (>=4 items) | SIFT Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 605.31 ms / 1.65 FPS |
| Cluttered (>=4 items) | YOLOv8n (Main) | 0.9407 | 0.8581 | 0.8975 | 0.8665 | 0.6424 | 45.20 ms / 22.12 FPS |
| Normal | YOLOv8n (--no-clahe) | 0.8661 | 0.8429 | 0.8544 | 0.8190 | 0.6454 | 48.99 ms / 20.41 FPS |
| Shadow (<100 lux) | YOLOv8n (--no-clahe) | 0.7328 | 0.3678 | 0.4898 | 0.3010 | 0.2386 | 48.08 ms / 20.80 FPS |

### 4.2 Per-Class Normal Held-Out Test Metrics (YOLOv8n)
| Class ID | AP@0.50 | Precision | Recall |
| :--- | :---: | :---: | :---: |
| `pasta_noodles` | 0.6457 | 0.8947 | 0.6538 |
| `canned_tomato` | 0.9478 | 0.8000 | 0.9524 |
| `canned_tuna_meat` | 0.9218 | 0.8824 | 0.9375 |
| `cereal_grains` | 0.6241 | 0.7500 | 0.7895 |
| `coffee_condiment` | 0.6390 | 0.7500 | 0.6562 |
| `milk_dairy` | 0.8987 | 0.8409 | 0.9250 |
| `eggs` | 1.0000 | 1.0000 | 1.0000 |
| `fresh_produce` | 0.8752 | 0.9375 | 0.8824 |

### 4.3 Checkpoint 3 (`cp3-core`) vs. Checkpoint 4 (`cp4-80`) Comparison
| Condition | CP3 mAP@0.50 | CP4 mAP@0.50 | Delta mAP | CP3 F1 | CP4 F1 | Delta F1 | CP4 FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Normal | 0.7494 | 0.8190 | +0.0696 | 0.8359 | 0.8544 | +0.0185 | 23.65 |
| Shadow (<100 lux) | 0.4427 | 0.6750 | +0.2323 | 0.6345 | 0.7906 | +0.1561 | 17.68 |
| Motion Blur (k=11) | 0.4362 | 0.5334 | +0.0972 | 0.6150 | 0.6856 | +0.0706 | 23.42 |
| Angled (+/-16 deg) | 0.4829 | 0.5878 | +0.1049 | 0.6915 | 0.7489 | +0.0574 | 19.95 |
| Occluded (35%) | 0.3678 | 0.3919 | +0.0241 | 0.5172 | 0.5398 | +0.0226 | 20.67 |
| Cluttered (>=4 items) | 0.8190 | 0.8665 | +0.0475 | 0.8936 | 0.8975 | +0.0039 | 22.12 |

**Cause of Change Since Checkpoint 3:** Unconditional LAB CLAHE in `cp3-core` degraded normal well-lit mAP@0.50 from `0.8190` to `0.7494` and only achieved `0.4427` under `<100 lux` shadows. Switching to luminance-gated Gamma + CLAHE (`mean L < 45.0`) restored normal contrast (`0.8190`) while elevating `<100 lux` shadow mAP@0.50 to `0.6750` (`+0.2323`) and boosting motion blur (`+0.0972`) and angled (`+0.1049`) mAP@0.50.

**Video Pan & False-Meat Checks:** `demo_pantry_pan.mp4` achieved `7/9` confirmed items at `19.63 FPS` (`50.94 ms/frame`). False-meat triggers on 115 negative scenes: `1/115` (`0.0087`).

---

## 5. Performance Analysis, Operating Conditions & Limitations
- **Where It Works:** High reliability on well-lit and cluttered real pantry/fridge scenes (`Normal mAP@0.50 = 0.8190`, `Cluttered >=4 items = 0.8665`), especially `eggs` (`1.0000`), `canned_tomato` (`0.9478`), `canned_tuna_meat` (`0.9218`), `milk_dairy` (`0.8987`), and `fresh_produce` (`0.8752`).
- **Operating Conditions:** Handheld distance `0.3–1.2 m`, ambient lighting `>=30 lux` (or shadowed scenes corrected by luminance-gated CLAHE), frontal to `+-15 deg` angle, and `<=20%` item overlap.
- **Not Designed For:** Synthetic/cartoon imagery (`demo_pantry_pan.mp4` missed 2/9 blocks with 78 per-frame overcounts), severe `>=35%` central foreground occlusion, heavy `k>=11` motion blur, or unlabeled opaque containers.
- **Measured Error Causes (`n=295` events across test runs):** Occlusion and clutter (`123`), Image quality/blur (`70`), Lighting/shadows (`18`), Viewpoint/scale (`52`), Label/domain gap (`20`), Look-alike classes (`12`). See `failure_gallery.png` (`F01`–`F12`).

---

## 6. Pre-Demo Test Cases (`TC01`–`TC16`)
Summary: **12/16 PASS** (9 hard/edge cases).

| ID | Input | Condition | Type | Expected | Actual | Pass |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| TC01 | `rf_raw_rf_frieburg_0000_CEREAL0014_png.rf.862d1662f6d7a3f829dd42f5311a9fbd.jpg` | Normal daylight shelf/fridge | Standard | 2x cereal_grains | 2 box(es): cereal_grains 0.87, cereal_grains 0.69 | 46.0 ms | PASS |
| TC02 | `rf_raw_rf_frieburg_0006_COFFEE0003_png.rf.fa8e65b81fee5f7268f71b416a99f0d5.jpg` | Normal daylight shelf/fridge | Standard | 1x coffee_condiment | 1 box(es): coffee_condiment 0.94 | 43.0 ms | PASS |
| TC03 | `rf_raw_rf_frieburg_0010_FISH0003_png.rf.3b9f093fda33d27fd36411cdf93b94f4.jpg` | Normal daylight shelf/fridge | Standard | 1x canned_tuna_meat | 1 box(es): canned_tuna_meat 0.94 | 36.4 ms | PASS |
| TC04 | `rf_raw_rf_frieburg_0022_MILK0004_png.rf.e5bed198fb3ffe8d1f119ec05fee39d0.jpg` | Normal daylight shelf/fridge | Standard | 1x milk_dairy | 1 box(es): milk_dairy 0.91 | 35.5 ms | PASS |
| TC05 | `rf_raw_rf_frieburg_0034_PASTA0020_png.rf.8688af670e18d467ddb94d8ef06f53c3.jpg` | Normal daylight shelf/fridge | Standard | 2x pasta_noodles | 2 box(es): pasta_noodles 0.89, pasta_noodles 0.81 | 40.1 ms | PASS |
| TC06 | `neg_test_0000.jpg` | Empty kitchen background | Standard | 0 items (empty background) | 0 items | 48.9 ms | PASS |
| TC07 | `neg_test_0001.jpg` | Empty kitchen background | Standard | 0 items (empty background) | 0 items | 47.9 ms | PASS |
| TC08 | `rf_raw_rf_frieburg_0055_VINEGAR0011_png.rf.95d2b75b3de8139f0c441bececcf2914.jpg` | Look-alike negative (vinegar bottle) | Hard (Look-Alike) | 3x coffee_condiment | 2 box(es): canned_tuna_meat 0.88, milk_dairy 0.82 | 36.6 ms | FAIL |
| TC09 | `rf_raw_rf_frieburg_0043_SUGAR0012_png.rf.1d247785c79ab0835b8d7ed5cc60a46e.jpg` | Cluttered shelf (>=4 items overlapping) | Hard (Cluttered) | 4x coffee_condiment | 4 box(es): coffee_condiment 0.94, coffee_condiment 0.92, coffee_condiment 0.89, coffee_condiment 0.65 | 45.3 ms | PASS |
| TC10 | `rf_raw_rf_frieburg_0004_CEREAL0045_png.rf.a1392188acb57a43ecdd3108e33a5e69.jpg` | Shadowed (<100 lux) + Gated CLAHE | Hard (Low-Light) | 1x cereal_grains | 1 box(es): cereal_grains 0.80 | 44.7 ms | PASS |
| TC11 | `rf_raw_rf_frieburg_0000_CEREAL0014_png.rf.862d1662f6d7a3f829dd42f5311a9fbd.jpg` | Shadowed (<100 lux) + Gated CLAHE | Hard (Low-Light) | 2x cereal_grains | 2 box(es): cereal_grains 0.83, coffee_condiment 0.74 | 55.2 ms | FAIL |
| TC12 | `rf_raw_rf_frieburg_0002_CEREAL0037_png.rf.35f1a7b364114f50166d8998639b4f1c.jpg` | Partial occlusion (35% foreground box) | Hard (Occlusion) | 1x cereal_grains | 1 box(es): cereal_grains 0.69 | 47.3 ms | PASS |
| TC13 | `rf_raw_rf_frieburg_0000_CEREAL0014_png.rf.862d1662f6d7a3f829dd42f5311a9fbd.jpg` | Partial occlusion (35% foreground box) | Hard (Occlusion) | 2x cereal_grains | 1 box(es): cereal_grains 0.49 | 48.3 ms | FAIL |
| TC14 | `rf_raw_rf_frieburg_0000_CEREAL0014_png.rf.862d1662f6d7a3f829dd42f5311a9fbd.jpg` | Handheld motion blur (k=11) | Hard (Motion Blur) | 2x cereal_grains | 0 items | 39.0 ms | FAIL |
| TC15 | `rf_raw_rf_frieburg_0000_CEREAL0014_png.rf.862d1662f6d7a3f829dd42f5311a9fbd.jpg` | Angled viewpoint (+/-16 deg tilt) | Hard (Viewpoint) | 2x cereal_grains | 2 box(es): cereal_grains 0.76, cereal_grains 0.66 | 50.8 ms | PASS |
| TC16 | `demo_pantry_pan.mp4` | Video pan (60 frames, W=5, K=3) | Hard (Video Domain Gap) | 9 synthetic shelf boxes (pan) | 7 confirmed (2x cereal_grains, 4x coffee_condiment, 1x milk_dairy) | 19.63 FPS | PASS |
