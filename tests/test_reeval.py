import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
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


def test_reeval_gemini_api_config_client_and_errors(monkeypatch):
    import threading
    import google.genai as genai_mod
    import src.cv.gemini_detector as gd_mod
    from src.server.config import get_gemini_thinking_level

    prev_cache_key = gd_mod.GEMINI_CLIENT_CACHE.get("api_key")
    prev_cache_client = gd_mod.GEMINI_CLIENT_CACHE.get("client")
    prev_state = dict(gd_mod.GEMINI_RUNTIME_STATE)

    try:
        monkeypatch.delenv("STOCK2STOVE_GEMINI_THINKING_LEVEL", raising=False)
        assert get_gemini_thinking_level("gemini-4-hydrogen") == "medium"
        assert get_gemini_thinking_level("gemini-3.5-flash-lite") == "minimal"

        monkeypatch.setenv("STOCK2STOVE_GEMINI_THINKING_LEVEL", "high")
        assert get_gemini_thinking_level("gemini-4-hydrogen") == "high"
        monkeypatch.delenv("STOCK2STOVE_GEMINI_THINKING_LEVEL", raising=False)

        dummy_img = np.zeros((120, 120, 3), dtype=np.uint8)

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        gd_mod.GEMINI_CLIENT_CACHE["api_key"] = None
        gd_mod.GEMINI_CLIENT_CACHE["client"] = None
        res_missing = gd_mod.query_gemini_vision_api(dummy_img, api_key=None)
        assert res_missing is None
        assert gd_mod.GEMINI_RUNTIME_STATE["last_error"]
        assert gd_mod.GEMINI_RUNTIME_STATE["last_live_api"] is False

        recorded_calls = []
        client_creations = []
        behavior = {"mode": "error", "wait_event": None}

        def fake_create(**kwargs):
            recorded_calls.append(dict(kwargs))
            if behavior["mode"] == "error":
                raise RuntimeError("boom")
            if behavior["mode"] == "timeout":
                behavior["wait_event"].wait(0.25)
            return SimpleNamespace(
                errors=None,
                output_text='{"boxes": []}',
            )

        def fake_client_factory(api_key=None):
            client_creations.append(api_key)
            return SimpleNamespace(interactions=SimpleNamespace(create=fake_create))

        monkeypatch.setattr(genai_mod, "Client", fake_client_factory)

        gd_mod.GEMINI_CLIENT_CACHE["api_key"] = None
        gd_mod.GEMINI_CLIENT_CACHE["client"] = None
        behavior["mode"] = "error"
        res_err = gd_mod.query_gemini_vision_api(dummy_img, api_key="key-1", model_name="gemini-4-hydrogen")
        assert res_err is None
        assert "boom" in str(gd_mod.GEMINI_RUNTIME_STATE["last_error"])
        assert gd_mod.GEMINI_RUNTIME_STATE["last_live_api"] is False

        behavior["mode"] = "timeout"
        behavior["wait_event"] = threading.Event()
        res_to = gd_mod.query_gemini_vision_api(dummy_img, api_key="key-1", model_name="gemini-4-hydrogen", timeout_sec=0.02)
        behavior["wait_event"].set()
        assert res_to is None
        assert "timed out" in str(gd_mod.GEMINI_RUNTIME_STATE["last_error"]).lower()
        assert gd_mod.GEMINI_RUNTIME_STATE["last_live_api"] is False

        behavior["mode"] = "ok"
        res_ok_1 = gd_mod.query_gemini_vision_api(dummy_img, api_key="key-1", model_name="gemini-4-hydrogen")
        assert res_ok_1 == '{"boxes": []}'
        assert gd_mod.GEMINI_RUNTIME_STATE["last_error"] is None
        assert gd_mod.GEMINI_RUNTIME_STATE["last_live_api"] is True

        res_ok_2 = gd_mod.query_gemini_vision_api(dummy_img, api_key="key-1", model_name="gemini-3.5-flash-lite")
        assert res_ok_2 == '{"boxes": []}'

        monkeypatch.setenv("STOCK2STOVE_GEMINI_THINKING_LEVEL", "high")
        res_ok_3 = gd_mod.query_gemini_vision_api(dummy_img, api_key="key-1", model_name="gemini-4-hydrogen")
        assert res_ok_3 == '{"boxes": []}'

        assert len(client_creations) == 1
        assert client_creations[0] == "key-1"

        call_h = recorded_calls[-3]
        call_fl = recorded_calls[-2]
        call_env = recorded_calls[-1]
        for c in (call_h, call_fl, call_env):
            assert c["store"] is False
            assert c["response_format"]["type"] == "text"
            assert c["response_format"]["mime_type"] == "application/json"
            assert "schema" in c["response_format"]
            assert "temperature" not in c["generation_config"]
        assert call_h["generation_config"]["thinking_level"] == "medium"
        assert call_fl["generation_config"]["thinking_level"] == "minimal"
        assert call_env["generation_config"]["thinking_level"] == "high"
    finally:
        gd_mod.GEMINI_CLIENT_CACHE["api_key"] = prev_cache_key
        gd_mod.GEMINI_CLIENT_CACHE["client"] = prev_cache_client
        gd_mod.GEMINI_RUNTIME_STATE.clear()
        gd_mod.GEMINI_RUNTIME_STATE.update(prev_state)


def test_reeval_gemini_schema_prompt_and_parser():
    from src.cv.gemini_detector import (
        build_gemini_detection_prompt,
        build_gemini_detection_schema,
        clamp_estimated_calories,
    )

    schema = build_gemini_detection_schema()
    item_schema = schema["properties"]["boxes"]["items"]
    props = item_schema["properties"]
    assert props["class_id"]["enum"] == sorted(VALID_CLASS_IDS)
    assert props["estimated_calories"]["type"] == "integer"
    assert "estimated_calories" in item_schema["required"]
    for field_name in ("box_2d", "class_id", "display_name", "confidence", "estimated_calories"):
        assert props[field_name].get("description")

    prompt = build_gemini_detection_prompt(excluded_classes=["canned_tomato", "eggs"])
    prompt_lower = prompt.lower()
    for non_food_kw in ("books", "cups", "bowls", "plates", "utensils", "appliances"):
        assert non_food_kw in prompt_lower
    assert "excluded_classes" in prompt
    assert "- canned_tomato:" not in prompt
    assert "- eggs:" not in prompt
    assert "- pasta_noodles:" in prompt

    assert clamp_estimated_calories(320, "pasta_noodles") == 320
    assert clamp_estimated_calories(5, "pasta_noodles") == 15
    assert clamp_estimated_calories(2900, "pasta_noodles") == 2500
    assert clamp_estimated_calories(3500, "pasta_noodles") == get_class_calories("pasta_noodles", 1)
    assert clamp_estimated_calories(0, "pasta_noodles") == get_class_calories("pasta_noodles", 1)
    assert clamp_estimated_calories("bad", "pasta_noodles") == get_class_calories("pasta_noodles", 1)

    raw_json = json.dumps({
        "boxes": [
            {"box_2d": [100, 100, 400, 350], "class_id": "book", "display_name": "Cookbook", "confidence": 0.95, "estimated_calories": 100},
            {"box_2d": [100, 100, 400, 350], "class_id": "canned_tomato", "display_name": "Mutti", "confidence": 0.94, "estimated_calories": 95},
            {"box_2d": [100, 400, 400, 700], "class_id": "pasta_noodles", "display_name": "Barilla", "confidence": 0.91, "estimated_calories": 320},
            {"box_2d": [450, 100, 750, 350], "class_id": "fresh_produce", "display_name": "Mini Berry", "confidence": 0.88, "estimated_calories": 4},
            {"box_2d": [450, 400, 750, 700], "class_id": "cereal_grains", "display_name": "Bulk Oats", "confidence": 0.87, "estimated_calories": 3500},
        ]
    })
    parsed = parse_gemini_boxes(raw_json, conf_threshold=0.45, excluded_classes=["canned_tomato"])
    assert [d["class_id"] for d in parsed] == ["pasta_noodles", "fresh_produce", "cereal_grains"]
    assert parsed[0]["estimated_calories"] == 320
    assert parsed[1]["estimated_calories"] == 15
    assert parsed[2]["estimated_calories"] == get_class_calories("cereal_grains", 1)


def test_reeval_per_item_calories_end_to_end(monkeypatch):
    import src.run as run_mod
    import src.server.app as app_mod
    from src.llm_recipe import compute_inventory_total_calories
    from src.server.schemas import format_confirmed_inventory_item, format_detection_item

    fmt_det = format_detection_item({
        "class_id": "pasta_noodles",
        "display_name": "Barilla",
        "confidence": 0.92,
        "bbox_norm": [0.1, 0.1, 0.4, 0.4],
        "estimated_calories": 380,
    })
    assert fmt_det["estimated_calories"] == 380

    tracker = create_tracker_state(window_size=5, min_hits=2, conf_threshold=0.45, iou_threshold=0.45)
    frame_dets = [
        {"class_id": "pasta_noodles", "display_name": "Barilla", "confidence": 0.91, "bbox_norm": [0.05, 0.1, 0.35, 0.6], "estimated_calories": 380},
        {"class_id": "pasta_noodles", "display_name": "Barilla", "confidence": 0.89, "bbox_norm": [0.55, 0.1, 0.85, 0.6], "estimated_calories": 380},
    ]
    update_temporal_tracker(tracker, frame_dets)
    ann_2, conf_2 = update_temporal_tracker(tracker, frame_dets)
    assert ann_2[0]["estimated_calories"] == 380
    assert conf_2[0]["calories_per_unit"] == 380
    assert conf_2[0]["estimated_calories"] == 760

    fmt_inv_1 = format_confirmed_inventory_item({"class_id": "pasta_noodles", "count": 2, "calories_per_unit": 380})
    assert fmt_inv_1["calories_per_unit"] == 380
    assert fmt_inv_1["estimated_calories"] == 760

    fmt_inv_2 = format_confirmed_inventory_item({"class_id": "pasta_noodles", "count": 2, "estimated_calories": 760})
    assert fmt_inv_2["calories_per_unit"] == 380
    assert fmt_inv_2["estimated_calories"] == 760

    total_cals = compute_inventory_total_calories([
        {"class_id": "pasta_noodles", "count": 2, "calories_per_unit": 380},
        {"class_id": "canned_tomato", "count": 1, "estimated_calories": 95},
    ])
    assert total_cals == 855

    fake_gem_dets = [
        {"class_id": "pasta_noodles", "display_name": "Barilla", "confidence": 0.92, "bbox_norm": [0.1, 0.1, 0.4, 0.5], "polygon_norm": None, "temporally_confirmed": False, "estimated_calories": 380},
        {"class_id": "canned_tomato", "display_name": "Mutti", "confidence": 0.89, "bbox_norm": [0.5, 0.1, 0.8, 0.5], "polygon_norm": None, "temporally_confirmed": False, "estimated_calories": 95},
    ]
    fake_gem_meta = {
        "detector_mode": "gemini_vision",
        "backend": "gemini-4-hydrogen",
        "live_api": True,
        "cached": False,
        "last_error": None,
    }
    monkeypatch.setattr(run_mod, "run_gemini_detection", lambda *args, **kwargs: (list(fake_gem_dets), dict(fake_gem_meta)))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_mod.app)
    ui_resp = client.post("/api/v1/ui/detect", json={"sample_name": "pantry_images_cv_01.jpg"})
    assert ui_resp.status_code == 200
    ui_data = ui_resp.json()
    assert [d["estimated_calories"] for d in ui_data["detections"]] == [380, 95]
    inv_map = {item["class_id"]: item for item in ui_data["confirmed_inventory"]}
    assert inv_map["pasta_noodles"]["calories_per_unit"] == 380
    assert inv_map["pasta_noodles"]["estimated_calories"] == 380
    assert inv_map["canned_tomato"]["calories_per_unit"] == 95
    assert inv_map["canned_tomato"]["estimated_calories"] == 95
    assert ui_data["total_estimated_calories"] == 475
    assert ui_data["total_inventory_calories"] == 475


def test_reeval_gemini_offline_fallback_skips_opencv(monkeypatch):
    import src.cv.gemini_detector as gd_mod

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)

    opencv_calls = []

    def fake_opencv(*args, **kwargs):
        opencv_calls.append(True)
        return [
            {
                "class_id": "fresh_produce",
                "display_name": "Fresh Produce",
                "confidence": 0.70,
                "bbox_norm": [0.2, 0.2, 0.5, 0.5],
                "polygon_norm": None,
                "temporally_confirmed": False,
                "estimated_calories": 55,
            }
        ]

    monkeypatch.setattr(gd_mod, "run_opencv_region_fallback", fake_opencv)
    monkeypatch.setattr(
        gd_mod,
        "run_yolo_detection",
        lambda *args, **kwargs: (
            [
                {
                    "class_id": "pasta_noodles",
                    "display_name": "Pasta & Noodles",
                    "confidence": 0.88,
                    "bbox_norm": [0.1, 0.1, 0.4, 0.4],
                    "polygon_norm": None,
                    "temporally_confirmed": False,
                    "estimated_calories": 220,
                }
            ],
            {"detector_mode": "yolov8", "backend": "ultralytics", "weights_path": "mock.pt"},
        ),
    )

    dets_yolo, meta_yolo = gd_mod.run_gemini_detection(dummy_img, conf_threshold=0.45, use_cache=False)
    assert len(opencv_calls) == 0
    assert meta_yolo["backend"] == "yolov8_offline_fallback"
    assert meta_yolo["live_api"] is False
    assert len(dets_yolo) == 1

    monkeypatch.setattr(
        gd_mod,
        "run_yolo_detection",
        lambda *args, **kwargs: ([], {"detector_mode": "yolov8", "backend": "no_model_empty", "weights_path": "none"}),
    )
    dets_cv, meta_cv = gd_mod.run_gemini_detection(dummy_img, conf_threshold=0.45, use_cache=False)
    assert len(opencv_calls) == 1
    assert meta_cv["backend"] == "opencv_offline_fallback"
    assert meta_cv["live_api"] is False
    assert len(dets_cv) == 1


def test_reeval_gemini_temporal_voting(monkeypatch):
    import src.server.app as app_mod

    img = cv2.imread(os.path.join(SAMPLES_DIR, "pantry_images_cv_01.jpg"))
    assert img is not None

    current_conf = {"value": 0.75}

    def fake_gemini(bgr, **kwargs):
        return (
            [
                {
                    "class_id": "pasta_noodles",
                    "display_name": "Spaghetti",
                    "confidence": current_conf["value"],
                    "bbox_norm": [0.1, 0.1, 0.4, 0.5],
                    "polygon_norm": None,
                    "temporally_confirmed": False,
                    "estimated_calories": 280,
                }
            ],
            {
                "detector_mode": "gemini_vision",
                "backend": "gemini-4-hydrogen",
                "live_api": True,
                "cached": False,
                "last_error": None,
            },
        )

    monkeypatch.setattr(app_mod, "run_gemini_detection", fake_gemini)

    sid_low = "reeval_gem_vote_low"
    sid_high = "reeval_gem_vote_high"
    remove_session(sid_low)
    remove_session(sid_high)
    reset_session(sid_low)
    reset_session(sid_high)

    try:
        current_conf["value"] = 0.75
        res_1 = app_mod.execute_single_frame_pipeline(img, detector_mode="gemini_vision", session_id=sid_low, frame_id=1)
        assert res_1["confirmed_inventory"] == []
        assert res_1["detections"][0]["temporally_confirmed"] is False

        res_2 = app_mod.execute_single_frame_pipeline(img, detector_mode="gemini_vision", session_id=sid_low, frame_id=2)
        assert len(res_2["confirmed_inventory"]) == 1
        assert res_2["confirmed_inventory"][0]["class_id"] == "pasta_noodles"
        assert res_2["detections"][0]["temporally_confirmed"] is True

        current_conf["value"] = 0.86
        res_fast = app_mod.execute_single_frame_pipeline(img, detector_mode="gemini_vision", session_id=sid_high, frame_id=1)
        assert len(res_fast["confirmed_inventory"]) == 1
        assert res_fast["confirmed_inventory"][0]["class_id"] == "pasta_noodles"
        assert res_fast["detections"][0]["temporally_confirmed"] is True
    finally:
        remove_session(sid_low)
        remove_session(sid_high)


def test_reeval_metadata_propagation_health_and_interactions(monkeypatch):
    import inspect
    import google.genai as genai_mod
    import src.cv.gemini_detector as gd_mod
    import src.llm_recipe as llm_mod
    import src.run as run_mod
    import src.server.app as app_mod

    llm_source = inspect.getsource(llm_mod)
    assert "interactions.create" in llm_source
    assert "generate_content" not in llm_source

    prev_cache_key = gd_mod.GEMINI_CLIENT_CACHE.get("api_key")
    prev_cache_client = gd_mod.GEMINI_CLIENT_CACHE.get("client")
    try:
        llm_calls = []

        def fake_llm_create(**kwargs):
            llm_calls.append(dict(kwargs))
            return SimpleNamespace(output_text='{"recipes": []}')

        def fake_llm_client_factory(api_key=None):
            return SimpleNamespace(interactions=SimpleNamespace(create=fake_llm_create))

        gd_mod.GEMINI_CLIENT_CACHE["api_key"] = None
        gd_mod.GEMINI_CLIENT_CACHE["client"] = None
        monkeypatch.setattr(genai_mod, "Client", fake_llm_client_factory)
        out_txt, _ = llm_mod.query_live_llm_worker("test prompt", "recipe-key", None)
        assert out_txt == '{"recipes": []}'
        assert len(llm_calls) == 1
        assert llm_calls[0]["store"] is False
    finally:
        gd_mod.GEMINI_CLIENT_CACHE["api_key"] = prev_cache_key
        gd_mod.GEMINI_CLIENT_CACHE["client"] = prev_cache_client

    gem_det = [
        {
            "class_id": "pasta_noodles",
            "display_name": "De Cecco",
            "confidence": 0.91,
            "bbox_norm": [0.1, 0.1, 0.4, 0.5],
            "polygon_norm": None,
            "temporally_confirmed": False,
            "estimated_calories": 310,
        }
    ]
    gem_meta = {
        "detector_mode": "gemini_vision",
        "backend": "gemini-4-hydrogen",
        "live_api": True,
        "cached": True,
        "last_error": None,
    }
    hyb_meta = {
        "detector_mode": "hybrid_gemini",
        "backend": "ultralytics+gemini-4-hydrogen",
        "yolo_backend": "ultralytics",
        "gemini_backend": "gemini-4-hydrogen",
        "live_api": True,
        "cached": False,
        "last_error": None,
    }

    monkeypatch.setattr(run_mod, "run_gemini_detection", lambda *args, **kwargs: (list(gem_det), dict(gem_meta)))
    monkeypatch.setattr(app_mod, "run_gemini_detection", lambda *args, **kwargs: (list(gem_det), dict(gem_meta)))
    monkeypatch.setattr(app_mod, "run_hybrid_gemini_detection", lambda *args, **kwargs: (list(gem_det), dict(hyb_meta)))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = TestClient(app_mod.app)

    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    ms = health_resp.json()["model_status"]
    for key in ("yolov8n_loaded", "weights_path", "orb_ready", "sift_ready", "gemini_status"):
        assert key in ms
    gs = ms["gemini_status"]
    for key in ("api_key_configured", "vision_model", "recipe_model", "thinking_level", "live_api", "last_error", "cache_entries"):
        assert key in gs

    ui_resp = client.post("/api/v1/ui/detect", json={"sample_name": "pantry_images_cv_01.jpg"})
    assert ui_resp.status_code == 200
    ui_data = ui_resp.json()
    assert ui_data["detector_backend"] == "gemini-4-hydrogen"
    assert ui_data["live_api"] is True
    assert ui_data["cached"] is True
    assert ui_data["last_error"] is None

    scan_resp = client.post("/api/v1/scan/image", json={"sample_name": "pantry_images_cv_01.jpg"})
    assert scan_resp.status_code == 200
    scan_data = scan_resp.json()
    assert scan_data["gemini_vision"]["detector_backend"] == "gemini-4-hydrogen"
    assert scan_data["gemini_vision"]["live_api"] is True
    assert scan_data["gemini_vision"]["cached"] is True
    assert scan_data["hybrid_gemini"]["detector_backend"] == "ultralytics+gemini-4-hydrogen"
    assert scan_data["hybrid_gemini"]["live_api"] is True
    assert scan_data["hybrid_gemini"]["cached"] is False

    sample_bgr = cv2.imread(os.path.join(SAMPLES_DIR, "pantry_images_cv_01.jpg"))
    sample_b64 = encode_image_base64_jpeg(sample_bgr, quality=75)
    with client.websocket_connect("/ws/scan") as ws:
        ws.send_json({
            "frame_id": 1,
            "detector_mode": "gemini_vision",
            "enable_clahe": True,
            "conf_threshold": 0.45,
            "image_b64": sample_b64,
        })
        ws_msg = ws.receive_json()
        assert ws_msg["detector_backend"] == "gemini-4-hydrogen"
        assert ws_msg["live_api"] is True
        assert ws_msg["cached"] is True
        assert ws_msg["last_error"] is None


def test_reeval_evaluate_all_include_gemini(tmp_path, monkeypatch):
    import sys
    import src.training.evaluate_all as eval_mod

    test_img_dir = tmp_path / "test" / "images"
    test_lbl_dir = tmp_path / "test" / "labels"
    neg_dir = tmp_path / "pantry_negatives"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    dummy_img = np.full((128, 128, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(test_img_dir / "sample_01.jpg"), dummy_img)
    with open(test_lbl_dir / "sample_01.txt", "w", encoding="utf-8") as f:
        f.write("0 0.500000 0.500000 0.400000 0.400000\n")
    cv2.imwrite(str(neg_dir / "neg_01.jpg"), dummy_img)

    data_yaml_path = tmp_path / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "test": "test/images",
                "nc": 8,
                "names": {0: "pasta_noodles", 1: "canned_tomato", 2: "canned_tuna_meat"},
            },
            f,
        )

    mock_box = [
        {
            "class_id": "pasta_noodles",
            "display_name": "Pasta & Noodles",
            "confidence": 0.90,
            "bbox_norm": [0.30, 0.30, 0.70, 0.70],
        }
    ]
    monkeypatch.setattr(eval_mod, "load_ultralytics_detector", lambda mode="yolov8": (None, "mock.pt"))
    monkeypatch.setattr(eval_mod, "load_templates_for_mode", lambda mode="orb_baseline": [])
    monkeypatch.setattr(eval_mod, "run_baseline_detection", lambda *args, **kwargs: (list(mock_box), {}))
    monkeypatch.setattr(eval_mod, "run_yolo_detection", lambda *args, **kwargs: (list(mock_box), {"backend": "ultralytics"}))
    monkeypatch.setattr(eval_mod, "run_gemini_detection", lambda *args, **kwargs: (list(mock_box), {"backend": "gemini-4-hydrogen", "live_api": True}))
    monkeypatch.setattr(eval_mod, "run_hybrid_gemini_detection", lambda *args, **kwargs: (list(mock_box), {"backend": "ultralytics+gemini-4-hydrogen", "live_api": True}))

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    skip_summary = eval_mod.run_benchmark_suite(
        str(data_yaml_path),
        str(tmp_path / "skip.csv"),
        str(tmp_path / "skip.json"),
        str(tmp_path / "skip.png"),
        include_gemini=True,
    )
    assert skip_summary["include_gemini"] is False
    assert skip_summary["detectors"] == ["orb_baseline", "sift_baseline", "yolov8n_finetuned"]
    assert "gemini_vision" not in skip_summary["false_meat_by_detector"]

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    plot_path = tmp_path / "all5.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_all.py",
            "--data",
            str(data_yaml_path),
            "--output-csv",
            str(tmp_path / "all5.csv"),
            "--output-json",
            str(tmp_path / "all5.json"),
            "--output-plot",
            str(plot_path),
            "--include-gemini",
        ],
    )
    eval_mod.main()
    with open(tmp_path / "all5.json", "r", encoding="utf-8") as f:
        full_summary = json.load(f)
    assert full_summary["include_gemini"] is True
    assert full_summary["detectors"] == [
        "orb_baseline",
        "sift_baseline",
        "yolov8n_finetuned",
        "gemini_vision",
        "hybrid_gemini",
    ]
    assert len(full_summary["results"]) == 15
    assert "gemini_vision" in full_summary["false_meat_by_detector"]
    assert "hybrid_gemini" in full_summary["false_meat_by_detector"]
    assert os.path.isfile(plot_path)
    assert os.path.getsize(plot_path) > 0


def test_reeval_spa_pwa_and_android_retirement(tmp_path, monkeypatch):
    import src.server.app as app_mod
    from src.server.config import PROJECT_ROOT

    assert not hasattr(app_mod, "handle_web_ui")
    assert not hasattr(app_mod, "HTMLResponse")
    assert not os.path.exists(os.path.join(PROJECT_ROOT, "android"))
    assert not os.path.exists(os.path.join(PROJECT_ROOT, "app-debug.apk"))
    assert not os.path.exists(os.path.join(PROJECT_ROOT, "app-release.apk"))

    client = TestClient(app_mod.app)

    manifest_resp = client.get("/manifest.webmanifest")
    assert manifest_resp.status_code == 200
    assert "application/manifest+json" in manifest_resp.headers.get("content-type", "")
    manifest_data = manifest_resp.json()
    assert "icons" in manifest_data

    icon_192 = client.get("/icons/icon-192.png")
    assert icon_192.status_code == 200
    icon_512 = client.get("/icons/icon-512.png")
    assert icon_512.status_code == 200
    fav_resp = client.get("/favicon.png")
    assert fav_resp.status_code == 200

    spa_resp = client.get("/basket/recipes")
    assert spa_resp.status_code == 200
    assert '<div id="root"></div>' in spa_resp.text

    api_404 = client.get("/api/nonexistent")
    assert api_404.status_code == 404
    assert "error" in api_404.json()

    ws_404 = client.get("/ws/nonexistent")
    assert ws_404.status_code == 404
    assert "error" in ws_404.json()

    monkeypatch.setattr(app_mod, "FRONTEND_DIST_DIR", str(tmp_path))
    missing_root = client.get("/")
    assert missing_root.status_code == 503
    assert "error" in missing_root.json()
    missing_spa = client.get("/basket")
    assert missing_spa.status_code == 503
    assert "error" in missing_spa.json()
