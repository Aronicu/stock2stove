import os
import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.cv.baseline_detector import ensure_reference_templates, run_baseline_detection
from src.cv.preprocess import (
    apply_clahe_lab,
    encode_image_base64_jpeg,
    letterbox_image,
    preprocess_frame,
    unletterbox_bbox_norm,
    unletterbox_detections,
)
from src.cv.gemini_detector import (
    parse_gemini_boxes,
    run_gemini_detection,
    run_hybrid_gemini_detection,
)
from src.cv.temporal_tracker import create_tracker_state, update_temporal_tracker
from src.cv.yolo_detector import run_yolo_detection
from src.llm_recipe import generate_recipes_from_inventory, generate_structured_recipes
from src.server.app import app
from src.server.config import SAMPLES_DIR, TEMPLATES_DIR, UNIFIED_CLASSES


def load_sample_bgr(index=1):
    sample_path = os.path.join(SAMPLES_DIR, f"pantry_images_cv_{index:02d}.jpg")
    img = cv2.imread(sample_path)
    assert img is not None
    return img


def test_task_1_1_preprocess():
    sample = load_sample_bgr(1)
    clahe_img = apply_clahe_lab(sample)
    assert clahe_img.shape == sample.shape
    assert clahe_img.dtype == np.uint8

    padded, meta = letterbox_image(sample, target_size=640)
    assert padded.shape == (640, 640, 3)
    assert meta["target_size"] == 640

    norm_box = unletterbox_bbox_norm([100, 120, 420, 500], meta)
    assert len(norm_box) == 4
    assert 0.0 <= norm_box[0] <= norm_box[2] <= 1.0
    assert 0.0 <= norm_box[1] <= norm_box[3] <= 1.0

    prep_out = preprocess_frame(sample, enable_clahe=True, target_size=640)
    assert prep_out["letterboxed_bgr"].shape == (640, 640, 3)
    assert "resized_bgr" in prep_out
    assert prep_out["resized_bgr"].shape == (prep_out["meta"]["new_h"], prep_out["meta"]["new_w"], 3)
    assert prep_out["diagnostics"]["clahe_applied"] is True
    assert prep_out["diagnostics"]["edge_density_pct"] > 0.0
    assert prep_out["diagnostics"]["orb_keypoints"] > 0

    mapped_dets = unletterbox_detections(
        [
            {
                "class_id": "pasta_noodles",
                "bbox_norm": [0.25, 0.25, 0.75, 0.75],
                "polygon_norm": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
            }
        ],
        meta,
    )
    assert len(mapped_dets) == 1
    assert mapped_dets[0]["bbox_norm"] == unletterbox_bbox_norm([160.0, 160.0, 480.0, 480.0], meta)
    assert mapped_dets[0]["polygon_norm"] is not None
    assert len(mapped_dets[0]["polygon_norm"]) == 4


def test_task_1_2_baseline_detector():
    ensure_reference_templates(templates_dir=TEMPLATES_DIR, samples_dir=SAMPLES_DIR)
    sample = load_sample_bgr(1)
    prep_out = preprocess_frame(sample, enable_clahe=True, target_size=640)
    letterboxed = prep_out["letterboxed_bgr"]

    orb_dets, orb_diag = run_baseline_detection(letterboxed, mode="orb_baseline", conf_threshold=0.50)
    assert orb_diag["scene_keypoints"] > 0
    assert orb_diag["templates_checked"] >= 8
    assert len(orb_dets) >= 1
    first_orb = orb_dets[0]
    assert "polygon_norm" in first_orb
    assert len(first_orb["polygon_norm"]) == 4

    sift_dets, sift_diag = run_baseline_detection(letterboxed, mode="sift_baseline", conf_threshold=0.50)
    assert sift_diag["scene_keypoints"] > 0
    assert len(sift_dets) >= 1


def test_task_1_3_yolo_and_temporal_tracker():
    sample = load_sample_bgr(1)
    prep_out = preprocess_frame(sample, enable_clahe=True, target_size=640)
    dets, meta = run_yolo_detection(prep_out["letterboxed_bgr"], mode="yolov8", conf_threshold=0.50)
    assert len(dets) >= 1
    assert "detector_mode" in meta

    tracker = create_tracker_state(window_size=5, min_hits=3, conf_threshold=0.50, iou_threshold=0.45)
    mock_frame_dets = [
        {
            "class_id": "pasta_noodles",
            "display_name": "Pasta & Noodles",
            "confidence": 0.91,
            "bbox_norm": [0.10, 0.20, 0.35, 0.80],
            "polygon_norm": None,
        },
        {
            "class_id": "pasta_noodles",
            "display_name": "Pasta & Noodles",
            "confidence": 0.85,
            "bbox_norm": [0.12, 0.22, 0.36, 0.79],
            "polygon_norm": None,
        },
        {
            "class_id": "canned_tomato",
            "display_name": "Canned Tomatoes",
            "confidence": 0.88,
            "bbox_norm": [0.45, 0.30, 0.65, 0.75],
            "polygon_norm": None,
        },
        {
            "class_id": "canned_tomato",
            "display_name": "Canned Tomatoes",
            "confidence": 0.86,
            "bbox_norm": [0.70, 0.30, 0.90, 0.75],
            "polygon_norm": None,
        },
    ]

    _, confirmed_1 = update_temporal_tracker(tracker, mock_frame_dets)
    assert len(confirmed_1) == 0

    _, confirmed_2 = update_temporal_tracker(tracker, mock_frame_dets)
    assert len(confirmed_2) == 0

    annotated_3, confirmed_3 = update_temporal_tracker(tracker, mock_frame_dets)
    assert len(confirmed_3) == 2

    by_class = {}
    for item in confirmed_3:
        by_class[item["class_id"]] = item

    assert by_class["pasta_noodles"]["count"] == 1
    assert by_class["canned_tomato"]["count"] == 2

    for det in annotated_3:
        assert det["temporally_confirmed"] is True


def test_task_1_4_llm_recipe_engine():
    inventory = ["pasta_noodles", "canned_tomato", "canned_tuna_meat", "eggs", "milk_dairy"]
    res = generate_structured_recipes(inventory, api_key=None, timeout_sec=2.5)
    assert res["latency_ms"] < 2500.0
    assert len(res["recipes"]) == 3
    for card in res["recipes"]:
        assert "id" in card
        assert "title" in card
        assert "prep_time_mins" in card
        assert "matched_ingredients" in card
        assert "steps" in card
        assert len(card["steps"]) >= 3

    legacy_str = generate_recipes_from_inventory(["pasta", "canned tomatoes"])
    assert isinstance(legacy_str, str)
    assert len(legacy_str) > 20


def test_task_1_5_fastapi_server():
    client = TestClient(app)

    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "ok"
    assert len(health_data["supported_classes"]) == 8

    img_resp = client.post("/api/v1/scan/image", json={"sample_name": "pantry_images_cv_01.jpg"})
    assert img_resp.status_code == 200
    img_data = img_resp.json()
    assert "yolov8" in img_data
    assert "gemini_vision" in img_data
    assert "hybrid_gemini" in img_data
    assert "orb_baseline" in img_data
    assert "sift_baseline" in img_data

    vid_resp = client.post("/api/v1/scan/video", json={"detector_mode": "yolov8"})
    assert vid_resp.status_code == 200
    vid_data = vid_resp.json()
    assert vid_data["frames_processed"] >= 3
    assert "confirmed_inventory" in vid_data
    assert len(vid_data["keyframes"]) >= 1

    rec_resp = client.post(
        "/api/v1/recipes/generate",
        json={
            "confirmed_inventory": [
                {"class_id": "pasta_noodles", "count": 1},
                {"class_id": "canned_tomato", "count": 2},
                {"class_id": "eggs", "count": 6},
            ]
        },
    )
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert len(rec_data["recipes"]) == 3

    sample_bgr = load_sample_bgr(1)
    sample_b64 = encode_image_base64_jpeg(sample_bgr, quality=80)
    ok, jpeg_buf = cv2.imencode(".jpg", sample_bgr)
    assert ok

    with client.websocket_connect("/ws/scan") as ws:
        for fid in range(1, 4):
            ws.send_json({
                "frame_id": fid,
                "detector_mode": "yolov8",
                "enable_clahe": True,
                "conf_threshold": 0.50,
                "image_b64": sample_b64,
            })
            msg = ws.receive_json()
            assert msg["frame_id"] == fid
            assert msg["detector_mode"] == "yolov8"
            assert "inference_ms" in msg
            assert "fps" in msg
            assert "diagnostics" in msg
            assert "detections" in msg
            assert "confirmed_inventory" in msg
            if fid >= 3:
                assert len(msg["confirmed_inventory"]) >= 1

        ws.send_bytes(jpeg_buf.tobytes())
        bin_msg = ws.receive_json()
        assert bin_msg["frame_id"] >= 4
        assert len(bin_msg["confirmed_inventory"]) >= 1

        ws.send_json({
            "frame_id": 5,
            "detector_mode": "gemini_vision",
            "enable_clahe": True,
            "conf_threshold": 0.45,
            "image_b64": sample_b64,
            "reset_session": True,
        })
        gem_msg = ws.receive_json()
        assert gem_msg["frame_id"] == 5
        assert gem_msg["detector_mode"] == "gemini_vision"
        assert len(gem_msg["detections"]) >= 1

        ws.send_json({
            "frame_id": 6,
            "detector_mode": "hybrid_gemini",
            "enable_clahe": True,
            "conf_threshold": 0.45,
            "image_b64": sample_b64,
        })
        hyb_msg = ws.receive_json()
        assert hyb_msg["frame_id"] == 6
        assert hyb_msg["detector_mode"] == "hybrid_gemini"
        assert len(hyb_msg["detections"]) >= 1


def test_task_1_6_gemini_detector():
    raw_json = (
        '{"boxes": ['
        '{"box_2d": [100, 200, 450, 600], "class_id": "canned_tomato", "display_name": "Del Monte Tomato Sauce", "confidence": 0.94},'
        '{"box_2d": [500, 50, 900, 350], "class_id": "pasta_noodles", "display_name": "Spaghetti Pack", "confidence": 0.88}'
        ']}'
    )
    parsed = parse_gemini_boxes(raw_json, conf_threshold=0.50)
    assert len(parsed) == 2
    assert parsed[0]["class_id"] == "canned_tomato"
    assert parsed[0]["bbox_norm"] == [0.2, 0.1, 0.6, 0.45]
    assert parsed[1]["class_id"] == "pasta_noodles"
    assert parsed[1]["bbox_norm"] == [0.05, 0.5, 0.35, 0.9]

    valid_ids = set()
    for entry in UNIFIED_CLASSES:
        valid_ids.add(entry["class_id"])

    sample = load_sample_bgr(1)
    prep_out = preprocess_frame(sample, enable_clahe=True, target_size=640)
    gem_dets, gem_meta = run_gemini_detection(prep_out["letterboxed_bgr"], conf_threshold=0.45)
    assert gem_meta["detector_mode"] == "gemini_vision"
    assert len(gem_dets) >= 1
    for det in gem_dets:
        assert det["class_id"] in valid_ids
        box = det["bbox_norm"]
        assert 0.0 <= box[0] < box[2] <= 1.0
        assert 0.0 <= box[1] < box[3] <= 1.0

    hyb_dets, hyb_meta = run_hybrid_gemini_detection(prep_out["letterboxed_bgr"], conf_threshold=0.45)
    assert hyb_meta["detector_mode"] == "hybrid_gemini"
    assert len(hyb_dets) >= 1
    for det in hyb_dets:
        assert det["class_id"] in valid_ids
        box = det["bbox_norm"]
        assert 0.0 <= box[0] < box[2] <= 1.0
        assert 0.0 <= box[1] < box[3] <= 1.0
