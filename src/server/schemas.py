from src.cv.temporal_tracker import extract_positive_calories
from src.server.config import (
    DEFAULT_CONF_THRESHOLD,
    LLM_TIMEOUT_SECONDS,
    SUPPORTED_DETECTOR_MODES,
    get_class_calories,
    get_display_name,
    normalize_class_id,
)


def validate_detector_mode(mode_value):
    if not mode_value:
        return "gemini_vision"
    cleaned = str(mode_value).strip().lower()
    if cleaned in SUPPORTED_DETECTOR_MODES:
        return cleaned
    if cleaned in ("orb", "baseline", "baseline_orb"):
        return "orb_baseline"
    if cleaned in ("sift", "baseline_sift"):
        return "sift_baseline"
    if cleaned in ("world", "yolov8s-worldv2"):
        return "yolo_world"
    if cleaned in ("gemini", "gemini_vision", "gemini-4-hydrogen", "gemini-3.5-flash-lite"):
        return "gemini_vision"
    if cleaned in ("hybrid", "hybrid_gemini", "yolo_gemini"):
        return "hybrid_gemini"
    return "gemini_vision"


def parse_scan_frame_request(payload, session_defaults=None):
    defaults = session_defaults or {}
    if not isinstance(payload, dict):
        payload = {}

    frame_id = int(payload.get("frame_id", defaults.get("frame_id", 0)))
    raw_mode = payload.get("detector_mode", defaults.get("detector_mode", "gemini_vision"))
    detector_mode = validate_detector_mode(raw_mode)

    if "enable_clahe" in payload:
        enable_clahe = bool(payload["enable_clahe"])
    else:
        enable_clahe = bool(defaults.get("enable_clahe", True))

    raw_conf = payload.get("conf_threshold", defaults.get("conf_threshold", DEFAULT_CONF_THRESHOLD))
    conf_threshold = max(0.05, min(0.99, float(raw_conf)))

    image_b64 = payload.get("image_b64") or payload.get("image") or None
    session_id = str(payload.get("session_id", defaults.get("session_id", "default")))
    reset_session = bool(payload.get("reset_session", False))

    return {
        "frame_id": frame_id,
        "detector_mode": detector_mode,
        "enable_clahe": enable_clahe,
        "conf_threshold": conf_threshold,
        "image_b64": image_b64,
        "session_id": session_id,
        "reset_session": reset_session,
    }


def format_detection_item(det):
    raw_cls = det.get("class_id") or "pasta_noodles"
    cls_id = normalize_class_id(raw_cls)
    disp = det.get("display_name") or get_display_name(cls_id)
    conf = round(float(det.get("confidence", 0.0)), 2)

    raw_bbox = det.get("bbox_norm") or [0.0, 0.0, 1.0, 1.0]
    bbox_norm = []
    for coord in raw_bbox[:4]:
        bbox_norm.append(round(float(coord), 4))
    while len(bbox_norm) < 4:
        bbox_norm.append(0.0)

    raw_poly = det.get("polygon_norm")
    polygon_norm = None
    if isinstance(raw_poly, list) and len(raw_poly) > 0:
        polygon_norm = []
        for pt in raw_poly:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                polygon_norm.append([round(float(pt[0]), 4), round(float(pt[1]), 4)])

    est_calories = extract_positive_calories(det.get("estimated_calories"))
    if est_calories is None:
        est_calories = get_class_calories(cls_id, count=1)

    return {
        "class_id": cls_id,
        "display_name": str(disp),
        "confidence": conf,
        "bbox_norm": bbox_norm,
        "polygon_norm": polygon_norm,
        "temporally_confirmed": bool(det.get("temporally_confirmed", False)),
        "crop_jpeg_b64": det.get("crop_jpeg_b64"),
        "estimated_calories": int(est_calories),
    }


def format_confirmed_inventory_item(item):
    raw_cls = item.get("class_id") or "pasta_noodles"
    cls_id = normalize_class_id(raw_cls)
    disp = item.get("display_name") or get_display_name(cls_id)
    count = max(1, int(item.get("count", 1)))
    max_conf = round(float(item.get("max_conf", 0.5)), 2)

    calories_per_unit = extract_positive_calories(item.get("calories_per_unit"))
    if calories_per_unit is None:
        raw_est = extract_positive_calories(item.get("estimated_calories"))
        if raw_est is not None:
            calories_per_unit = max(1, int(round(float(raw_est) / float(count))))
    if calories_per_unit is None:
        calories_per_unit = get_class_calories(cls_id, count=1)

    estimated_calories = int(calories_per_unit * count)
    item_key = str(item.get("item_key") or f"{cls_id}::{' '.join(str(disp).strip().lower().split())}")
    return {
        "class_id": cls_id,
        "item_key": item_key,
        "display_name": str(disp),
        "count": count,
        "max_conf": max_conf,
        "crop_jpeg_b64": item.get("crop_jpeg_b64"),
        "calories_per_unit": int(calories_per_unit),
        "estimated_calories": estimated_calories,
    }


def format_scan_response(
    frame_id,
    detector_mode,
    inference_ms,
    diagnostics,
    detections,
    confirmed_inventory,
    excluded_classes=None,
    detector_meta=None,
):
    inf_ms = round(max(0.01, float(inference_ms)), 2)
    fps = round(1000.0 / inf_ms, 1)

    formatted_dets = []
    for d in detections:
        formatted_dets.append(format_detection_item(d))

    formatted_inv = []
    total_estimated_calories = 0
    for item in confirmed_inventory:
        formatted_item = format_confirmed_inventory_item(item)
        formatted_inv.append(formatted_item)
        total_estimated_calories += int(formatted_item["estimated_calories"])

    excluded_list = []
    if excluded_classes:
        for cls_val in excluded_classes:
            excluded_list.append(str(cls_val))

    diag_out = {
        "edge_density_pct": round(float(diagnostics.get("edge_density_pct", 0.0)), 2),
        "orb_keypoints": int(diagnostics.get("orb_keypoints", 0)),
        "clahe_applied": bool(diagnostics.get("clahe_applied", True)),
    }
    if "good_matches" in diagnostics:
        diag_out["good_matches"] = int(diagnostics["good_matches"])
    if "ransac_inliers" in diagnostics:
        diag_out["ransac_inliers"] = int(diagnostics["ransac_inliers"])

    meta = detector_meta if isinstance(detector_meta, dict) else {}
    detector_backend = meta.get("backend") or meta.get("detector_backend") or str(detector_mode)
    live_api = bool(meta.get("live_api", False))
    cached = bool(meta.get("cached", False))
    last_error = meta.get("last_error")

    return {
        "frame_id": int(frame_id),
        "detector_mode": str(detector_mode),
        "detector_backend": str(detector_backend),
        "live_api": live_api,
        "cached": cached,
        "last_error": last_error,
        "inference_ms": inf_ms,
        "fps": fps,
        "diagnostics": diag_out,
        "detections": formatted_dets,
        "confirmed_inventory": formatted_inv,
        "total_estimated_calories": int(total_estimated_calories),
        "excluded_classes": excluded_list,
    }


def parse_recipe_request(payload):
    if not isinstance(payload, dict):
        payload = {}

    raw_items = (
        payload.get("confirmed_inventory")
        or payload.get("inventory")
        or payload.get("items")
        or payload.get("detected_items")
        or []
    )

    timeout_sec = float(payload.get("timeout_sec", LLM_TIMEOUT_SECONDS))
    timeout_sec = max(0.2, min(60.0, timeout_sec))
    api_key = payload.get("api_key") or None
    force_refresh = bool(payload.get("force_refresh", False))

    return {
        "items": raw_items,
        "timeout_sec": timeout_sec,
        "api_key": api_key,
        "force_refresh": force_refresh,
    }
