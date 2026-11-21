import csv
import json
import os
import cv2
import yaml

from src.server.config import (
    DATASET_YOLO_DIR,
    FINE_TUNED_ONNX_PATH,
    FINE_TUNED_PT_PATH,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    SAMPLES_DIR,
)


def test_task_2_1_extract_frames_and_auto_label(tmp_path, monkeypatch):
    import src.training.auto_label as auto_label_mod
    from src.server.config import GEMINI_AUTOLABEL_MODEL
    from src.training.auto_label import auto_label_directory, auto_label_image
    from src.training.extract_frames import extract_frames_from_directory

    raw_dir = tmp_path / "raw_home"
    frames_dir = tmp_path / "frames"
    auto_dir = tmp_path / "auto_labeled"
    raw_dir.mkdir(parents=True, exist_ok=True)

    extract_frames_from_directory(
        str(raw_dir),
        str(frames_dir),
        target_fps=3.0,
        synthesize_demo=True,
    )

    extracted_frames = sorted(frames_dir.glob("*.jpg"))
    assert len(extracted_frames) >= 1

    sample_path = os.path.join(SAMPLES_DIR, "pantry_images_cv_01.jpg")
    sample_bgr = cv2.imread(sample_path)
    assert sample_bgr is not None

    preds = auto_label_image(sample_bgr, conf_threshold=0.45)
    assert isinstance(preds, list)
    assert len(preds) >= 1
    for item in preds:
        assert "class_idx" in item
        assert "class_id" in item
        assert "confidence" in item
        assert 0 <= int(item["class_idx"]) <= 7
        assert 0.0 <= float(item["xc"]) <= 1.0
        assert 0.0 <= float(item["yc"]) <= 1.0
        assert 0.0 <= float(item["w"]) <= 1.0
        assert 0.0 <= float(item["h"]) <= 1.0

    subset_dir = tmp_path / "subset_frames"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame_path in enumerate(extracted_frames[:2]):
        frame_img = cv2.imread(str(frame_path))
        assert frame_img is not None
        cv2.imwrite(str(subset_dir / f"frame_{idx:03d}.jpg"), frame_img)

    def fake_offline_gemini(bgr_image, **kwargs):
        return (
            [
                {
                    "class_id": "pasta_noodles",
                    "display_name": "Fallback Pasta",
                    "confidence": 0.90,
                    "bbox_norm": [0.1, 0.1, 0.5, 0.5],
                }
            ],
            {
                "detector_mode": "gemini_vision",
                "backend": "yolov8_offline_fallback",
                "live_api": False,
                "cached": False,
                "last_error": "offline",
            },
        )

    monkeypatch.setattr(auto_label_mod, "run_gemini_detection", fake_offline_gemini)
    offline_preds = auto_label_image(sample_bgr, conf_threshold=0.45, backend="gemini")
    assert offline_preds == []

    offline_auto_dir = tmp_path / "auto_labeled_offline_gem"
    offline_summary = auto_label_directory(
        str(subset_dir),
        str(offline_auto_dir),
        conf_threshold=0.45,
        backend="gemini",
    )
    assert offline_summary["skipped_fallback_frames"] == 2
    assert offline_summary["images_processed"] == 0
    assert len(list((offline_auto_dir / "images").glob("*.*"))) == 0
    assert len(list((offline_auto_dir / "labels").glob("*.txt"))) == 0

    captured_calls = []

    def fake_live_gemini(bgr_image, **kwargs):
        captured_calls.append(dict(kwargs))
        return (
            [
                {
                    "class_id": "canned_tomato",
                    "display_name": "San Marzano Tomatoes",
                    "confidence": 0.92,
                    "bbox_norm": [0.15, 0.20, 0.55, 0.70],
                }
            ],
            {
                "detector_mode": "gemini_vision",
                "backend": GEMINI_AUTOLABEL_MODEL,
                "live_api": True,
                "cached": False,
                "last_error": None,
            },
        )

    monkeypatch.setattr(auto_label_mod, "run_gemini_detection", fake_live_gemini)
    gem_preds = auto_label_image(sample_bgr, conf_threshold=0.45, backend="gemini")
    assert isinstance(gem_preds, list)
    assert len(gem_preds) == 1
    assert len(captured_calls) == 1
    assert captured_calls[0]["timeout_sec"] == 15.0
    assert captured_calls[0]["model_name"] == GEMINI_AUTOLABEL_MODEL
    assert captured_calls[0]["use_cache"] is False
    for item in gem_preds:
        assert 0 <= int(item["class_idx"]) <= 7
        assert 0.0 <= float(item["xc"]) <= 1.0
        assert 0.0 <= float(item["yc"]) <= 1.0
        assert 0.0 <= float(item["w"]) <= 1.0
        assert 0.0 <= float(item["h"]) <= 1.0

    live_auto_dir = tmp_path / "auto_labeled_live_gem"
    live_summary = auto_label_directory(
        str(subset_dir),
        str(live_auto_dir),
        conf_threshold=0.45,
        backend="gemini",
    )
    assert live_summary["skipped_fallback_frames"] == 0
    assert live_summary["images_processed"] == 2

    summary = auto_label_directory(str(subset_dir), str(auto_dir), conf_threshold=0.45)
    assert summary is not None
    assert (auto_dir / "data.yaml").exists()
    assert (auto_dir / "images").exists()
    assert (auto_dir / "labels").exists()

    saved_images = list((auto_dir / "images").glob("*.*"))
    saved_labels = list((auto_dir / "labels").glob("*.txt"))
    assert len(saved_images) >= 1
    assert len(saved_labels) >= 1

    for lbl_file in saved_labels:
        content = lbl_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        for line in content.splitlines():
            parts = line.strip().split()
            assert len(parts) == 5
            cls_idx = int(parts[0])
            assert 0 <= cls_idx <= 7
            for coord_str in parts[1:]:
                val = float(coord_str)
                assert 0.0 <= val <= 1.0


def test_task_2_2_dataset_and_trained_weights():
    data_yaml_path = os.path.join(DATASET_YOLO_DIR, "data.yaml")
    pantry_fridge_yaml_path = os.path.join(DATASET_YOLO_DIR, "pantry_fridge.yaml")
    assert os.path.exists(data_yaml_path)
    assert os.path.exists(pantry_fridge_yaml_path)

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    assert data_cfg["nc"] == 8
    assert len(data_cfg["names"]) == 8

    with open(pantry_fridge_yaml_path, "r", encoding="utf-8") as f:
        pf_cfg = yaml.safe_load(f)
    assert pf_cfg["nc"] == 8
    assert len(pf_cfg["names"]) == 8

    found_classes = set()
    for split in ["train", "valid", "test"]:
        images_dir = os.path.join(DATASET_YOLO_DIR, split, "images")
        labels_dir = os.path.join(DATASET_YOLO_DIR, split, "labels")
        assert os.path.isdir(images_dir)
        assert os.path.isdir(labels_dir)
        image_files = [fn for fn in os.listdir(images_dir) if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
        assert len(image_files) >= 1
        for fn in os.listdir(labels_dir):
            if not fn.endswith(".txt"):
                continue
            lbl_path = os.path.join(labels_dir, fn)
            with open(lbl_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        found_classes.add(int(parts[0]))

    for cls_idx in range(8):
        assert cls_idx in found_classes

    assert os.path.exists(FINE_TUNED_PT_PATH)
    assert os.path.getsize(FINE_TUNED_PT_PATH) > 0
    assert os.path.exists(FINE_TUNED_ONNX_PATH)
    assert os.path.getsize(FINE_TUNED_ONNX_PATH) > 0

    results_png = os.path.join(PROJECT_ROOT, "runs", "detect", "stock2stove_yolov8n", "results.png")
    assert os.path.exists(results_png)
    assert os.path.getsize(results_png) > 0


def test_task_2_3_evaluate_all_benchmarks():
    csv_path = os.path.join(OUTPUTS_DIR, "benchmark_results.csv")
    json_path = os.path.join(OUTPUTS_DIR, "benchmark_summary.json")
    assert os.path.exists(csv_path)
    assert os.path.exists(json_path)

    expected_splits = ["normal", "shadowed_sub100lux", "occluded_20_50pct"]
    expected_detectors = ["orb_baseline", "sift_baseline", "yolov8n_finetuned"]

    rows_by_key = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = None
            row_detector = None
            for val in row.values():
                v_clean = str(val).strip()
                if v_clean in expected_splits:
                    row_split = v_clean
                if v_clean in expected_detectors:
                    row_detector = v_clean
            if row_split is not None and row_detector is not None:
                rows_by_key[(row_split, row_detector)] = row

    for split_name in expected_splits:
        for det_name in expected_detectors:
            assert (split_name, det_name) in rows_by_key

        yolo_row = rows_by_key[(split_name, "yolov8n_finetuned")]
        orb_row = rows_by_key[(split_name, "orb_baseline")]
        sift_row = rows_by_key[(split_name, "sift_baseline")]

        assert float(yolo_row["mAP50"]) > float(orb_row["mAP50"])
        assert float(yolo_row["mAP50"]) > float(sift_row["mAP50"])
        assert float(yolo_row["recall"]) > float(orb_row["recall"])
        assert float(yolo_row["recall"]) > float(sift_row["recall"])

    with open(json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    summary_text = json.dumps(summary_data)
    for split_name in expected_splits:
        assert split_name in summary_text
    for det_name in expected_detectors:
        assert det_name in summary_text
