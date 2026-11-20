# Stock-to-Stove: Milestone 4 Draft Presentation Slides (10 Slides)

*(Companion Markdown version of `CSST106_Milestone4_Slides_Stock2Stove.pptx` and `.pdf`)*

---
## Slide 1 — Title & Project Overview
- **Stock-to-Stove:** Automated Pantry Inventory Detection & Zero-Waste Recipe Application
- **Team Kaggleing:** Ogayon, Kumar, Sarmiento | **Tag:** `cp4-80`
- **Weights:** `models/best.pt` (`6.2 MB` PT / `12.19 MB` ONNX)

---
## Slide 2 — Problem Statement & Objectives
- Household grocery waste caused by hidden rear-shelf items and decision fatigue.
- Barcodes fail on fresh produce, loose bags, and occluded shelves.
- Objectives: 8-class detection, baseline vs. YOLOv8n evaluation, >=15 FPS CPU throughput, Web/Android UI + LLM recipes.

---
## Slide 3 — Dataset & 8-Class Schema
- 813 annotated images (2,480 boxes) + 115 pantry negatives + 60-frame video clip.
- Train: 605 | Valid: 104 | Held-Out Test (frozen): 104 (261 GT boxes).
- 5 Pantry classes (`pasta_noodles`, `canned_tomato`, `canned_tuna_meat`, `cereal_grains`, `coffee_condiment`) + 3 Fridge classes (`milk_dairy`, `eggs`, `fresh_produce`).

---
## Slide 4 — Technical Approach (Baselines vs. Main Model)
- Classical Baselines: ORB / SIFT (1,000 keypoints) + Lowe's ratio (0.75) + RANSAC homography -> brittle under curvature, shadows, and blur.
- Main Model: Fine-tuned YOLOv8n + Luminance-Gated Gamma/CLAHE (`mean L < 45.0`) + Class NMS (0.45) + 5-Frame Temporal Voter (`K>=3`).

---
## Slide 5 — End-to-End Pipeline & Application Interface
- Single cached model instance in `src/run.py` & `src/server/app.py`.
- Browser Dashboard (`GET /`), Native Android CameraX WebSocket client, CLI (`src/run.py`).
- Visual diagrams: `workflow_diagram.png` & `application_interface_overview.png`.

---
## Slide 6 — Reproducible Evaluation Results (`evaluate.py`)
- Command: `python src/evaluate.py --split test --weights models/best.pt`.
- Normal Held-Out: YOLOv8n mAP@0.50 = `0.8190`, F1 = `0.8544` vs. ORB (`0.0109`) & SIFT (`0.0246`).
- Cluttered (>=4 items): `0.8665` mAP@0.50 | False-Meat Safety: `1/115` (`0.0087`).

---
## Slide 7 — Progress Since Checkpoint 3 (`cp3-core` -> `cp4-80`)
| Condition | CP3 mAP@0.50 | CP4 mAP@0.50 | Delta | CP4 FPS |
| :--- | :---: | :---: | :---: | :---: |
| Normal | 0.7494 | 0.8190 | +0.0696 | 23.65 |
| Shadow (<100 lux) | 0.4427 | 0.6750 | +0.2323 | 17.68 |
| Motion Blur (k=11) | 0.4362 | 0.5334 | +0.0972 | 23.42 |
| Angled (+/-16 deg) | 0.4829 | 0.5878 | +0.1049 | 19.95 |
| Occluded (35%) | 0.3678 | 0.3919 | +0.0241 | 20.67 |
| Cluttered (>=4 items) | 0.8190 | 0.8665 | +0.0475 | 22.12 |

---
## Slide 8 — Sample Outputs & Pre-Demo Test Cases
- 16 pre-demo test cases (9 hard/edge): 12 PASS, 4 FAIL (`test_cases.csv`).
- Includes normal daylight, empty kitchen negatives, look-alike vinegar bottle, cluttered shelves, <100 lux shadows, 35% occlusion, motion blur, +/-16 deg tilt, and video pan.

---
## Slide 9 — Performance Analysis, Operating Conditions & Limitations
- Works reliably on well-lit/cluttered real shelves (`0.3-1.2 m`, `>=30 lux`, `<=20%` overlap).
- Fails on synthetic video graphics (`demo_pantry_pan.mp4` 7/9 items), >=35% central occlusion (`F01-F03`), `k=11` motion blur (`F11`), and cereal/pasta/coffee/vinegar look-alikes (`F07-F08`, `TC08`).

---
## Slide 10 — Live Demo Plan & To-100% Roadmap
- Live Demo: Start FastAPI server (`http://127.0.0.1:8000/`), test `TC01`–`TC16`, switch detectors live, show Android CameraX overlay, run unseen teacher images, verify `evaluate.py`.
- Remaining Tasks (`docs/remaining.md`): R1 Real Video Clips (11/24), R2 Stage-2 Hardening (11/26), R3 Blur/Angle Aug (11/27), R4 App Polish (11/29), R5 Final Report & Deck (12/03).
