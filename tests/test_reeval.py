import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.cv.gemini_detector import parse_gemini_boxes
from src.cv.preprocess import encode_image_base64_jpeg
from src.cv.temporal_tracker import (
    apply_cross_class_nms,
    create_tracker_state,
    purge_class_from_tracker,
    update_temporal_tracker,
)
from src.cv.yolo_detector import COCO_TO_UNIFIED_MAP, MODEL_REGISTRY, run_yolo_detection
from src.server.app import app
from src.server.config import (
    CLASS_CALORIE_ESTIMATES,
    CROSS_CLASS_NMS_IOU,
    FRONTEND_DIST_DIR,
    NON_FOOD_LABELS,
    SAMPLES_DIR,
    VALID_CLASS_IDS,
    get_class_calories,
    get_class_conf_threshold,
    is_valid_food_class,
    normalize_class_id,
)
from src.server.session_manager import record_session_frame, remove_session, reset_session


def test_reeval_non_food_rejection():
    for bad_label in ["book", "person", "chair", "wine glass", "cup", "bowl", "dining table", "", None, "unknown_xyz"]:
        assert is_valid_food_class(bad_label) is False
        assert normalize_class_id(bad_label) != "pasta_noodles"

    assert normalize_class_id("") == ""
    assert normalize_class_id(None) == ""
    for non_food_key in ["book", "wine glass", "cup", "bowl", "person", "chair"]:
        assert non_food_key not in COCO_TO_UNIFIED_MAP
        assert non_food_key in NON_FOOD_LABELS

    for good_label in ["pasta_noodles", "canned_tomato", "eggs", "Spaghetti", "Banana"]:
        assert is_valid_food_class(good_label) is True
        assert normalize_class_id(good_label) in VALID_CLASS_IDS

    assert get_class_conf_threshold("pasta_noodles", 0.45) == 0.45
    assert get_class_conf_threshold("canned_tuna_meat", 0.45) == 0.72
    assert get_class_conf_threshold("eggs", 0.60) == 0.60

    prev_model = MODEL_REGISTRY["yolov8_model"]
    prev_path = MODEL_REGISTRY["yolov8_path"]
    prev_avail = MODEL_REGISTRY["ultralytics_available"]
    try:
        names_map = {
            0: "person",
            1: "book",
            2: "chair",
            3: "wine glass",
            4: "cup",
            5: "bowl",
            6: "pasta_noodles",
            7: "canned_tomato",
        }
        boxes_xyxy = torch.tensor(
            [
                [20.0, 20.0, 200.0, 300.0],
                [210.0, 40.0, 340.0, 260.0],
                [350.0, 50.0, 500.0, 300.0],
                [510.0, 50.0, 600.0, 200.0],
                [30.0, 320.0, 140.0, 440.0],
                [160.0, 320.0, 280.0, 440.0],
                [300.0, 320.0, 450.0, 520.0],
                [470.0, 320.0, 610.0, 520.0],
                [10.0, 10.0, 14.0, 14.0],
                [10.0, 550.0, 590.0, 565.0],
            ],
            dtype=torch.float32,
        )
        boxes_conf = torch.tensor(
            [0.95, 0.93, 0.91, 0.89, 0.88, 0.87, 0.86, 0.84, 0.92, 0.90],
            dtype=torch.float32,
        )
        boxes_cls = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 6, 7],
            dtype=torch.float32,
        )
        fake_res = SimpleNamespace(
            names=names_map,
            boxes=SimpleNamespace(xyxy=boxes_xyxy, conf=boxes_conf, cls=boxes_cls),
        )
        MODEL_REGISTRY["ultralytics_available"] = True
        MODEL_REGISTRY["yolov8_model"] = SimpleNamespace(predict=lambda **kwargs: [fake_res])
        MODEL_REGISTRY["yolov8_path"] = "mock_yolov8n.pt"

        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        yolo_dets, yolo_meta = run_yolo_detection(dummy_img, mode="yolov8", conf_threshold=0.45)
        assert yolo_meta["backend"] == "ultralytics"
        yolo_classes = [d["class_id"] for d in yolo_dets]
        assert sorted(yolo_classes) == ["canned_tomato", "pasta_noodles"]
        for d in yolo_dets:
            assert d["estimated_calories"] == get_class_calories(d["class_id"], 1)
    finally:
        MODEL_REGISTRY["yolov8_model"] = prev_model
        MODEL_REGISTRY["yolov8_path"] = prev_path
        MODEL_REGISTRY["ultralytics_available"] = prev_avail

    raw_gemini_json = json.dumps({
        "boxes": [
            {"box_2d": [100, 100, 400, 350], "class_id": "book", "display_name": "Cookbook", "confidence": 0.95},
            {"box_2d": [120, 380, 500, 620], "class_id": "person", "display_name": "Person", "confidence": 0.94},
            {"box_2d": [520, 100, 850, 350], "class_id": "chair", "display_name": "Chair", "confidence": 0.92},
            {"box_2d": [520, 380, 850, 620], "class_id": "unknown_item", "display_name": "Mystery Object", "confidence": 0.91},
            {"box_2d": [100, 100, 108, 108], "class_id": "eggs", "display_name": "Tiny Egg", "confidence": 0.90},
            {"box_2d": [150, 650, 450, 920], "class_id": "canned_tomato", "display_name": "Tomato Sauce", "confidence": 0.89},
            {"box_2d": [500, 650, 850, 920], "class_id": "pasta_noodles", "display_name": "Spaghetti", "confidence": 0.87},
        ]
    })
    gem_dets = parse_gemini_boxes(raw_gemini_json, conf_threshold=0.45)
    assert [d["class_id"] for d in gem_dets] == ["canned_tomato", "pasta_noodles"]
    assert gem_dets[0]["estimated_calories"] == CLASS_CALORIE_ESTIMATES["canned_tomato"]
    assert gem_dets[1]["estimated_calories"] == CLASS_CALORIE_ESTIMATES["pasta_noodles"]

    tracker = create_tracker_state(window_size=5, min_hits=2, conf_threshold=0.45, iou_threshold=0.45)
    noisy_dets = [
        {"class_id": "book", "display_name": "Book", "confidence": 0.99, "bbox_norm": [0.1, 0.1, 0.3, 0.4]},
        {"class_id": "person", "display_name": "Person", "confidence": 0.98, "bbox_norm": [0.35, 0.1, 0.55, 0.5]},
        {"class_id": "chair", "display_name": "Chair", "confidence": 0.97, "bbox_norm": [0.6, 0.1, 0.8, 0.5]},
        {"class_id": "canned_tomato", "display_name": "Canned Tomatoes", "confidence": 0.85, "bbox_norm": [0.2, 0.5, 0.4, 0.8]},
    ]
    update_temporal_tracker(tracker, noisy_dets)
    ann_2, conf_2 = update_temporal_tracker(tracker, noisy_dets)
    assert [d["class_id"] for d in ann_2] == ["canned_tomato"]
    assert len(conf_2) == 1
    assert conf_2[0]["class_id"] == "canned_tomato"


def test_reeval_cross_class_nms_and_consensus_counting():
    assert CROSS_CLASS_NMS_IOU == 0.65
    overlapping = [
        {"class_id": "cereal_grains", "confidence": 0.91, "bbox_norm": [0.10, 0.10, 0.50, 0.50]},
        {"class_id": "pasta_noodles", "confidence": 0.78, "bbox_norm": [0.11, 0.11, 0.50, 0.50]},
        {"class_id": "eggs", "confidence": 0.85, "bbox_norm": [0.60, 0.60, 0.90, 0.90]},
    ]
    deduped = apply_cross_class_nms(overlapping, iou_threshold=CROSS_CLASS_NMS_IOU)
    assert [d["class_id"] for d in deduped] == ["cereal_grains", "eggs"]

    tracker = create_tracker_state(window_size=5, min_hits=3, conf_threshold=0.45, iou_threshold=0.45)

    def make_tomato_list(n):
        out = []
        for idx in range(n):
            x1 = 0.05 + idx * 0.22
            out.append({
                "class_id": "canned_tomato",
                "display_name": "Canned Tomatoes",
                "confidence": 0.88,
                "bbox_norm": [x1, 0.20, x1 + 0.16, 0.60],
            })
        return out

    frame_1 = make_tomato_list(1) + [
        {"class_id": "pasta_noodles", "display_name": "Pasta & Noodles", "confidence": 0.61, "bbox_norm": [0.05, 0.20, 0.21, 0.60]}
    ]
    frame_2 = make_tomato_list(1)
    frame_3 = make_tomato_list(3)

    ann_1, conf_1 = update_temporal_tracker(tracker, frame_1)
    assert len(ann_1) == 1
    assert ann_1[0]["class_id"] == "canned_tomato"
    assert len(conf_1) == 0

    update_temporal_tracker(tracker, frame_2)
    _, conf_3 = update_temporal_tracker(tracker, frame_3)
    assert len(conf_3) == 1
    assert conf_3[0]["class_id"] == "canned_tomato"
    assert conf_3[0]["count"] == 1
    assert conf_3[0]["calories_per_unit"] == 70
    assert conf_3[0]["estimated_calories"] == 70

    update_temporal_tracker(tracker, make_tomato_list(2))
    _, conf_5 = update_temporal_tracker(tracker, make_tomato_list(2))
    assert conf_5[0]["count"] == 2
    assert conf_5[0]["estimated_calories"] == 140

    purge_class_from_tracker(tracker, "canned_tomato")
    _, conf_after_purge = update_temporal_tracker(tracker, [])
    assert conf_after_purge == []


def test_reeval_delete_restore_rest_and_ws():
    client = TestClient(app)
    sid = "reeval_rest_session"
    remove_session(sid)
    reset_session(sid)

    try:
        record_session_frame(
            sid,
            [
                {"class_id": "pasta_noodles", "display_name": "Pasta & Noodles", "confidence": 0.90, "bbox_norm": [0.1, 0.1, 0.3, 0.4]},
                {"class_id": "canned_tomato", "display_name": "Canned Tomatoes", "confidence": 0.88, "bbox_norm": [0.4, 0.1, 0.6, 0.4]},
            ],
            min_hits_override=1,
        )

        del_param_resp = client.delete(f"/api/v1/session/items/pasta_noodles?session_id={sid}")
        assert del_param_resp.status_code == 200
        del_param_data = del_param_resp.json()
        assert del_param_data["status"] == "item_deleted"
        assert del_param_data["class_id"] == "pasta_noodles"
        assert "pasta_noodles" in del_param_data["excluded_classes"]
        remaining_ids = [item["class_id"] for item in del_param_data["confirmed_inventory"]]
        assert remaining_ids == ["canned_tomato"]
        assert del_param_data["total_estimated_calories"] == 70

        ui_excluded_resp = client.post(
            "/api/v1/ui/detect",
            json={
                "sample_name": "pantry_images_cv_01.jpg",
                "detector_mode": "yolov8",
                "session_id": sid,
                "excluded_classes": ["canned_tomato"],
            },
        )
        assert ui_excluded_resp.status_code == 200
        ui_excluded_data = ui_excluded_resp.json()
        assert "pasta_noodles" in ui_excluded_data["excluded_classes"]
        assert "canned_tomato" in ui_excluded_data["excluded_classes"]
        for det in ui_excluded_data["detections"]:
            assert det["class_id"] not in ("pasta_noodles", "canned_tomato")
        for inv in ui_excluded_data["confirmed_inventory"]:
            assert inv["class_id"] not in ("pasta_noodles", "canned_tomato")

        restore_resp = client.post(
            "/api/v1/session/items/restore",
            json={"session_id": sid, "class_id": "pasta_noodles"},
        )
        assert restore_resp.status_code == 200
        restore_data = restore_resp.json()
        assert restore_data["status"] == "item_restored"
        assert restore_data["class_id"] == "pasta_noodles"
        assert "pasta_noodles" not in restore_data["excluded_classes"]

        del_post_resp = client.post(
            "/api/v1/session/items/delete",
            json={"session_id": sid, "class_id": "canned_tomato"},
        )
        assert del_post_resp.status_code == 200
        del_post_data = del_post_resp.json()
        assert del_post_data["status"] == "item_deleted"
        assert "canned_tomato" in del_post_data["excluded_classes"]
        assert del_post_data["confirmed_inventory"] == []
        assert del_post_data["total_estimated_calories"] == 0
    finally:
        client.post("/api/v1/session/reset", json={"session_id": sid})
        remove_session(sid)

    sample_path = os.path.join(SAMPLES_DIR, "pantry_images_cv_01.jpg")
    sample_bgr = cv2.imread(sample_path)
    assert sample_bgr is not None
    sample_b64 = encode_image_base64_jpeg(sample_bgr, quality=80)

    with client.websocket_connect("/ws/scan") as ws:
        confirmed_msg = None
        for fid in range(1, 4):
            ws.send_json({
                "frame_id": fid,
                "detector_mode": "yolov8",
                "enable_clahe": True,
                "conf_threshold": 0.45,
                "image_b64": sample_b64,
            })
            confirmed_msg = ws.receive_json()

        assert confirmed_msg is not None
        assert len(confirmed_msg["confirmed_inventory"]) >= 1
        target_cls = confirmed_msg["confirmed_inventory"][0]["class_id"]

        ws.send_json({"action": "delete_item", "class_id": target_cls})
        ws_del = ws.receive_json()
        assert ws_del["status"] == "item_deleted"
        assert ws_del["class_id"] == target_cls
        assert target_cls in ws_del["excluded_classes"]
        assert target_cls not in [item["class_id"] for item in ws_del["confirmed_inventory"]]

        ws.send_json({
            "frame_id": 4,
            "detector_mode": "yolov8",
            "enable_clahe": True,
            "conf_threshold": 0.45,
            "image_b64": sample_b64,
        })
        ws_after = ws.receive_json()
        assert target_cls in ws_after["excluded_classes"]
        assert target_cls not in [d["class_id"] for d in ws_after["detections"]]
        assert target_cls not in [item["class_id"] for item in ws_after["confirmed_inventory"]]

        ws.send_json({"action": "restore_item", "class_id": target_cls})
        ws_res = ws.receive_json()
        assert ws_res["status"] == "item_restored"
        assert target_cls not in ws_res["excluded_classes"]


def test_reeval_calorie_estimates_and_recipes():
    client = TestClient(app)

    assert get_class_calories("pasta_noodles", 1) == 220
    assert get_class_calories("canned_tomato", 2) == 140
    assert get_class_calories("eggs", 3) == 234

    ui_resp = client.post(
        "/api/v1/ui/detect",
        json={"sample_name": "pantry_images_cv_01.jpg", "detector_mode": "yolov8"},
    )
    assert ui_resp.status_code == 200
    ui_data = ui_resp.json()
    assert len(ui_data["detections"]) >= 1
    for det in ui_data["detections"]:
        assert isinstance(det["estimated_calories"], int)
        assert det["estimated_calories"] > 0

    expected_total = 0
    assert len(ui_data["confirmed_inventory"]) >= 1
    for inv in ui_data["confirmed_inventory"]:
        assert isinstance(inv["calories_per_unit"], int)
        assert inv["calories_per_unit"] > 0
        assert inv["estimated_calories"] == inv["calories_per_unit"] * inv["count"]
        expected_total += inv["estimated_calories"]
    assert ui_data["total_estimated_calories"] == expected_total
    assert ui_data["total_inventory_calories"] == expected_total

    vid_resp = client.post("/api/v1/scan/video", json={"detector_mode": "yolov8"})
    assert vid_resp.status_code == 200
    vid_data = vid_resp.json()
    assert "total_estimated_calories" in vid_data
    vid_total = 0
    for inv in vid_data["confirmed_inventory"]:
        assert inv["estimated_calories"] == inv["calories_per_unit"] * inv["count"]
        vid_total += inv["estimated_calories"]
    assert vid_data["total_estimated_calories"] == vid_total

    rec_resp = client.post(
        "/api/v1/recipes/generate",
        json={
            "confirmed_inventory": [
                {"class_id": "pasta_noodles", "count": 1},
                {"class_id": "canned_tomato", "count": 2},
                {"class_id": "eggs", "count": 3},
            ]
        },
    )
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert rec_data["total_inventory_calories"] == 594
    assert len(rec_data["recipes"]) == 3
    for card in rec_data["recipes"]:
        assert isinstance(card["estimated_calories"], int)
        assert card["estimated_calories"] > 0
        assert isinstance(card["ingredient_calories"], dict)
        assert len(card["ingredient_calories"]) >= 1
        for kcal_val in card["ingredient_calories"].values():
            assert isinstance(kcal_val, int)
            assert kcal_val >= 0


def test_reeval_react_spa_build_and_serving():
    index_path = os.path.join(FRONTEND_DIST_DIR, "index.html")
    assets_dir = os.path.join(FRONTEND_DIST_DIR, "assets")
    assert os.path.isfile(index_path)
    assert os.path.isdir(assets_dir)

    with open(index_path, "r", encoding="utf-8") as f:
        index_html = f.read()
    assert '<div id="root"></div>' in index_html
    assert "/assets/" in index_html

    asset_files = [fn for fn in os.listdir(assets_dir) if fn.endswith((".js", ".css"))]
    assert len(asset_files) >= 2

    client = TestClient(app)
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers.get("content-type", "")
    assert '<div id="root"></div>' in root_resp.text
    assert "/assets/" in root_resp.text

    asset_resp = client.get(f"/assets/{asset_files[0]}")
    assert asset_resp.status_code == 200
