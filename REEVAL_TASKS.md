# Gemini Vision & Full Web App SPA Plan

Audit result:
1. **Gemini Vision:** Current usage (`src/cv/gemini_detector.py`, `src/server/config.py`, `src/server/app.py`, `src/training/auto_label.py`, `frontend/`) only partially follows the prior re-evaluation tasks and violates current Gemini 3.5 / Gemini 4 Interactions API requirements (passes deprecated `temperature`, lacks `thinking_level`, swallows API errors into silent OpenCV fallback, omits schema class `enum` & non-food prompt guards, is excluded from `evaluate_all.py`, and hides live/fallback status in the React SPA).
2. **Android → Full Web App SPA:** The React + Vite SPA (`frontend/`) is now the sole client application, fully replacing the native Android app (`android/`, `app-debug.apk`, `app-release.apk`) and legacy inline HTML UI in `src/server/app.py`.

## Plan Tasks

### A. Gemini Vision Model & API Upgrade

- [x] **1. Fix Gemini API Compliance & Upgrade Vision Model (`src/server/config.py`, `src/cv/gemini_detector.py`, `src/llm_recipe.py`)**
  - Update default `GEMINI_VISION_MODEL` to `gemini-4-hydrogen` (keeping `gemini-3.5-flash-lite` configurable via `STOCK2STOVE_GEMINI_VISION_MODEL` for ultra-low-latency mode) and add `STOCK2STOVE_GEMINI_THINKING_LEVEL` (default `"low"` for `gemini-4-hydrogen`, `"minimal"` for `gemini-3.5-flash-lite`).
  - Remove deprecated `temperature` from `generation_config` in `query_gemini_vision_api()` and replace with `{"thinking_level": resolved_thinking_level}`.
  - Reuse a cached `genai.Client` instead of creating a client and `ThreadPoolExecutor` per frame, raise default vision timeout (`timeout_sec`), and record `last_error` instead of silently swallowing exceptions.
  - Remove legacy `client.models.generate_content()` fallback branches in `src/llm_recipe.py` so all Gemini calls use `client.interactions.create()`.

- [x] **2. Tighten Structured Output Schema & Unwanted-Output Filtering (`src/cv/gemini_detector.py`, `src/server/app.py`)**
  - Constrain `class_id` in `build_gemini_detection_schema()` using `"enum": sorted(VALID_CLASS_IDS)` and add field descriptions (`box_2d`, `class_id`, `display_name`, `confidence`, `estimated_calories`).
  - Update `build_gemini_detection_prompt()` to explicitly instruct the model to ignore non-food items (books, cups, bowls, utensils, appliances, people, shelves) and any session `excluded_classes`.
  - Stop `run_gemini_detection()` offline fallback from injecting noisy OpenCV HSV heuristic boxes (`run_opencv_region_fallback`) when YOLOv8n weights are available.

- [x] **3. Align Accuracy, Temporal Voting, Auto-Labeling & Benchmarking (`src/server/app.py`, `src/training/auto_label.py`, `src/training/evaluate_all.py`)**
  - Tune temporal voting for `gemini_vision` in `execute_single_frame_pipeline()` (avoid unconditional single-frame confirmation on noisy live webcam streams unless confidence exceeds a tuned fast-confirm threshold).
  - Update `src/training/auto_label.py` to pass an appropriate timeout (e.g. `15.0s`) for `GEMINI_AUTOLABEL_MODEL` (`gemini-4-hydrogen`) and skip/log frames where `meta["live_api"]` is `False` so fallback boxes do not pollute training labels.
  - Add an optional `--include-gemini` benchmark path in `src/training/evaluate_all.py` to measure `gemini_vision` and `hybrid_gemini` precision, recall, mAP@50, and false-meat rate when `GEMINI_API_KEY` is configured.

- [x] **4. Upgrade Calorie Estimation with Gemini Vision (`src/cv/gemini_detector.py`, `src/server/schemas.py`, `src/cv/temporal_tracker.py`)**
  - Request per-item `estimated_calories` in `build_gemini_detection_schema()`, validate/clamp returned values in `parse_gemini_boxes()`, and fall back to `get_class_calories(cls_id, 1)` when missing or out of range.
  - Preserve Gemini per-item `estimated_calories` in `format_detection_item()` and `update_temporal_tracker()` instead of always overwriting with static class averages.

### B. Full Web App SPA Replacing Android

- [x] **5. Backend & SPA Gemini Vision Integration + Full Mobile/Desktop Feature Parity (`src/server/app.py`, `src/server/schemas.py`, `frontend/`)**
  - Stop discarding `gemini_meta` in `execute_single_frame_pipeline()` and `handle_ui_detect()`; expose `detector_backend`, `live_api`, and `cached` in scan/detect responses and `last_error` / `recipe_model` / `thinking_level` in `/api/v1/health`.
  - Align SPA default detector mode with Gemini Vision (with quick mode switching to `hybrid_gemini`, `yolov8`, `orb_baseline`, `sift_baseline`), remove hardcoded `'gemini-3.5-flash-lite'` fallback text in `DiagnosticsModal.jsx`, and display live status (`Live API` vs `Offline Fallback`, model ID, last error).
  - Upgrade `ScannerView.jsx` for full mobile/desktop replacement of Android CameraX: rear/front camera switcher, `"Gemini Vision analyzing..."` live HUD indicator, increased Gemini WebSocket `maxWaitMs`, and mobile-friendly touch controls.
  - Persist basket/session preferences in `localStorage` and add PWA metadata (`manifest.webmanifest`, app icons from `images/logo.png`, mobile viewport/theme meta tags in `frontend/index.html`) so the SPA is installable on mobile and desktop browsers.

- [x] **6. Retire Native Android App, APKs & Legacy Inline HTML UI (`android/`, `app-debug.apk`, `app-release.apk`, `src/server/app.py`, `run_server.ps1`, `run_server.sh`, `README.md`)**
  - Remove the `android/` directory and prebuilt `app-debug.apk` / `app-release.apk` artifacts.
  - Remove the legacy inline `handle_web_ui()` HTML fallback in `src/server/app.py` and serve `frontend/dist` (assets, manifest, icons, and SPA fallback route) as the sole web frontend.
  - Remove ADB USB reverse logic (`adb.exe` / `-SkipAdbReverse`) from `run_server.ps1` and `run_server.sh`.
  - Update `README.md` (badges, repository structure, quickstart, and architecture) to document the React + Vite Web App SPA as the primary application.

### C. Verification & Regression Tests

- [x] **7. Update Test Suites & Verify End-to-End (`tests/test_phase1.py`, `tests/test_reeval.py`, `tests/run_e2e_tests.ps1`, `frontend/`)**
  - Add unit tests confirming `query_gemini_vision_api()` omits `temperature`, sends `thinking_level`, enforces `class_id` enum in schema, parses/clamps `estimated_calories`, propagates `live_api`/`detector_backend` metadata, and rejects non-food & excluded classes.
  - Replace the Android Gradle runner in `tests/run_e2e_tests.ps1` with backend `pytest` + frontend `npm run build` verification.
  - Run full `pytest` suite and `npm run build` in `frontend/` to verify the complete SPA + Gemini Vision pipeline.
