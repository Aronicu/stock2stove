# Re-Evaluation Tasks (Target: Tuesday)

- [x] **Fine-Tune Datasets:** Clean and expand `data/dataset_yolo` with more shelf/pantry images (varied lighting, angles, occlusion), re-merge via `src/training/merge_datasets.py`, and retrain via `src/training/train_yolo.py`.
- [x] **Fix Accuracy Issues:** Benchmark precision/recall/mAP using `src/training/evaluate_all.py` and tune confidence/IoU thresholds & temporal voting in `src/cv/yolo_detector.py` and `src/cv/temporal_tracker.py`.
- [x] **Deletion of Unwanted Output:** Filter out non-food/irrelevant classes and low-confidence false positives in `src/cv/yolo_detector.py` / `src/server/app.py`, and add UI/API support to delete unwanted detected items.
- [x] **Kotlin → React SPA Migration (Vite):** Scaffold a React + Vite SPA (`npm create vite@latest`) and port camera/upload, ingredient detection list (with delete), and recipe display flows connected to `src/server/app.py`.
- [x] **Optional — Calorie Counter:** Add estimated calorie calculation per detected ingredient / generated recipe in `src/llm_recipe.py` and `src/server/schemas.py`, and display it in the React SPA.
