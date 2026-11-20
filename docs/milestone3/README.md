# Milestone 3 — Core Implementation, Testing & Error Analysis (`cp3-core`)

## 1. End-to-End Pipeline Commands (`src/run.py`)
```powershell
python src/run.py --input data/test/ --output outputs/run_2026-10-29/
python src/run.py --input demo.mp4 --output outputs/demo_annotated.mp4
python src/run.py --webcam 0
```

Pipeline stages in `src/run.py`: `preprocess()` -> `infer()` -> `postprocess()` -> `render()`.

## 2. Reproducing All Evaluation Tables & Plots (`src/evaluate.py`)
```powershell
python src/evaluate.py
```

## 3. Preliminary Results Summary (`docs/milestone3/preliminary_results.csv`)
| Condition | Method | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 | Speed (ms / FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Normal | ORB Baseline | 0.0426 | 0.0077 | 0.0130 | 0.0109 | 0.0064 | 1837.76 ms / 0.54 FPS |
| Normal | SIFT Baseline | 0.6250 | 0.0192 | 0.0372 | 0.0212 | 0.0112 | 579.63 ms / 1.73 FPS |
| Normal | YOLOv8n (Main) | 0.8526 | 0.8199 | 0.8359 | 0.7494 | 0.5814 | 53.00 ms / 18.87 FPS |
| Shadow (<100 lux) | ORB Baseline | 0.0746 | 0.0192 | 0.0305 | 0.0142 | 0.0088 | 2049.68 ms / 0.49 FPS |
| Shadow (<100 lux) | SIFT Baseline | 0.5000 | 0.0115 | 0.0225 | 0.0187 | 0.0128 | 430.39 ms / 2.32 FPS |
| Shadow (<100 lux) | YOLOv8n (Main) | 0.7931 | 0.5287 | 0.6345 | 0.4427 | 0.3461 | 47.58 ms / 21.02 FPS |
| Motion Blur (k=11) | ORB Baseline | 0.0164 | 0.0038 | 0.0062 | 0.0031 | 0.0006 | 2233.28 ms / 0.45 FPS |
| Motion Blur (k=11) | SIFT Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 431.81 ms / 2.32 FPS |
| Motion Blur (k=11) | YOLOv8n (Main) | 0.7584 | 0.5172 | 0.6150 | 0.4362 | 0.3329 | 49.21 ms / 20.32 FPS |
| Angled (±16°) | ORB Baseline | 0.0909 | 0.0230 | 0.0367 | 0.0268 | 0.0136 | 2151.05 ms / 0.46 FPS |
| Angled (±16°) | SIFT Baseline | 0.7000 | 0.0268 | 0.0517 | 0.0360 | 0.0128 | 608.63 ms / 1.64 FPS |
| Angled (±16°) | YOLOv8n (Main) | 0.8061 | 0.6054 | 0.6915 | 0.4829 | 0.2116 | 59.31 ms / 16.86 FPS |
| Occluded (35%) | ORB Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2260.40 ms / 0.44 FPS |
| Occluded (35%) | SIFT Baseline | 1.0000 | 0.0038 | 0.0076 | 0.0078 | 0.0039 | 495.26 ms / 2.02 FPS |
| Occluded (35%) | YOLOv8n (Main) | 0.7241 | 0.4023 | 0.5172 | 0.3678 | 0.3010 | 50.26 ms / 19.90 FPS |
| Cluttered (>=4 items) | ORB Baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1826.62 ms / 0.55 FPS |
| Cluttered (>=4 items) | SIFT Baseline | 1.0000 | 0.0135 | 0.0267 | 0.0031 | 0.0012 | 626.93 ms / 1.60 FPS |
| Cluttered (>=4 items) | YOLOv8n (Main) | 0.9403 | 0.8514 | 0.8936 | 0.8190 | 0.5989 | 53.17 ms / 18.81 FPS |
| Normal | YOLOv8n (--no-clahe) | 0.8661 | 0.8429 | 0.8544 | 0.8190 | 0.6454 | 48.27 ms / 20.72 FPS |
| Shadow (<100 lux) | YOLOv8n (--no-clahe) | 0.7328 | 0.3678 | 0.4898 | 0.3010 | 0.2386 | 44.41 ms / 22.52 FPS |

## 4. Video Clip Tracking Evaluation (`demo_pantry_pan.mp4`)
- Frames: 60 (640x480 @ 15.0 FPS) | Inference: 52.95 ms/frame (18.89 FPS)
- Expected synthetic shelf blocks: 9 | Confirmed items: 5 (Error: -4, MAE/frame: 1.37)

## 5. Error Analysis & Failure Counts (`docs/milestone3/failure_gallery.png`)
- **Occlusion and clutter:** 127
- **Image quality (blur/low-res):** 89
- **Lighting (shadows/glare):** 76
- **Viewpoint and scale:** 62
- **Label errors / domain gap:** 25
- **Look-alike classes:** 13

Representative Test Log (`docs/milestone3/test_log.csv`): 10/24 passed.

## 6. Generated Assets in `docs/milestone3/`
- `CSST106_Milestone3_CoreImplementation_Stock2Stove.docx` & `.pdf`
- `preliminary_results.csv` & `preliminary_results_comparison.png`
- `test_log.csv`
- `classification_report.txt` & `test_confusion_matrix.png`
- `failure_gallery.png` & `failures/`
- `pipeline_stages_strip.png`
- `milestone3_evaluation_summary.json`
