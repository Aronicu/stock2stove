# Milestone 4 — 80% Progress Check (`cp4-80`)

## 1. Launching the Integrated Application
```powershell
python src/server/app.py
```
- Browser UI: `http://127.0.0.1:8000/`
- CLI pipeline: `python src/run.py --input data/test/ --output outputs/run_latest/`

## 2. Reproducing All Evaluation Numbers (`src/evaluate.py`)
```powershell
python src/evaluate.py --split test --weights models/best.pt
```
- **Evaluated Weights:** `models/best.pt` (`6.2 MB` PT / `12.19 MB` ONNX)
- **Outputs Written:** `outputs/eval/metrics.json`, `outputs/eval/confusion_matrix.png`, and `docs/milestone4/*`.

## 3. Held-Out Test Results Summary (`docs/milestone4/preliminary_results.csv`)
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

## 4. Progress Since Checkpoint 3 (`docs/milestone4/cp3_vs_cp4_comparison.csv`)
| Condition | CP3 mAP@0.50 | CP4 mAP@0.50 | Delta mAP@0.50 | CP3 F1 | CP4 F1 | Delta F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Normal | 0.7494 | 0.8190 | +0.0696 | 0.8359 | 0.8544 | +0.0185 |
| Shadow (<100 lux) | 0.4427 | 0.6750 | +0.2323 | 0.6345 | 0.7906 | +0.1561 |
| Motion Blur (k=11) | 0.4362 | 0.5334 | +0.0972 | 0.6150 | 0.6856 | +0.0706 |
| Angled (+/-16 deg) | 0.4829 | 0.5878 | +0.1049 | 0.6915 | 0.7489 | +0.0574 |
| Occluded (35%) | 0.3678 | 0.3919 | +0.0241 | 0.5172 | 0.5398 | +0.0226 |
| Cluttered (>=4 items) | 0.8190 | 0.8665 | +0.0475 | 0.8936 | 0.8975 | +0.0039 |

## 5. Pre-Demo Test Cases (`docs/milestone4/test_cases.csv`): 12/16 PASS (9 Hard/Edge)
- **Video Clip (`demo_pantry_pan.mp4`):** 7/9 confirmed @ 19.63 FPS (50.94 ms/frame)
- **False-Meat Safety (`pantry_negatives`):** 1/115 (0.0087)
- **Failure Taxonomy:** Occlusion and clutter: 123, Image quality (blur/low-res): 70, Viewpoint and scale: 52, Label errors / domain gap: 20, Lighting (shadows/glare): 18, Look-alike classes: 12

## 6. Deliverables in `docs/milestone4/`
- `CSST106_Milestone4_80PercentCheck_Stock2Stove.docx` & `.pdf`
- `CSST106_Milestone4_Slides_Stock2Stove.pptx` & `.pdf` (10 slides)
- `DRAFT_TECHNICAL_REPORT.md` & `DRAFT_PRESENTATION_SLIDES.md`
- `remaining.md` (mirrored at `docs/remaining.md`)
- `preliminary_results.csv`, `preliminary_results_comparison.png`, `cp3_vs_cp4_comparison.csv`, `cp3_vs_cp4_comparison.png`
- `test_cases.csv`, `test_log.csv`, `classification_report.txt`, `test_confusion_matrix.png`
- `failure_gallery.png`, `failures/`, `pipeline_stages_strip.png`, `workflow_diagram.png`, `application_interface_overview.png`
- `milestone4_evaluation_summary.json`
