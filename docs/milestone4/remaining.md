# Remaining Tasks to 100% (Milestone 5 Roadmap)

| ID | Task | Details & Verification Target | Owner | Target Date | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | Real Kitchen Video Test Suite | Capture and annotate 3 real smartphone pantry/fridge MP4 pans (daylight, dim cupboard, cluttered shelf) to replace the synthetic `demo_pantry_pan.mp4` cartoon clip and raise video tracking count accuracy within `+/-1` item. | Ogayon | 2026-11-24 | Planned |
| **R2** | Occlusion & Look-Alike Hardening | Run Stage-2 fine-tuning (`50 epochs`, `close_mosaic=10`, stronger Cutout/Mixup) plus 60 domestic hard-negative crops for `cereal_grains`, `pasta_noodles`, and `coffee_condiment` to push `occluded_20_50pct` mAP@0.50 `>= 0.55` and normal mAP@0.50 `>= 0.85`. | Sarmiento | 2026-11-26 | Planned |
| **R3** | Motion-Blur & Extreme-Angle Augmentation | Add directional motion blur (`k=7..13`) and `+/-20 deg` perspective warps during training and evaluate `models/best.pt` via `src/evaluate.py` to lift `motion_blur` mAP@0.50 `>= 0.65`. | Sarmiento | 2026-11-27 | Planned |
| **R4** | Android + Web UI Polish & E2E Stress Test | Finalize APK release build, verify WebSocket reconnect recovery, verify Web UI video upload/webcam live stream FPS display, and run `tests/run_e2e_tests.ps1`. | Ogayon | 2026-11-29 | Planned |
| **R5** | Final Milestone 5 Report & Defense Deck | Expand Draft Technical Report into final Results, Discussion, Conclusion, and polished 12-slide final defense presentation; freeze `cp5-final` tag. | Kumar & All | 2026-12-03 | Planned |
