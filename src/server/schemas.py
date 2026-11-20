from src.server.config import (
    DEFAULT_CONF_THRESHOLD,
    LLM_TIMEOUT_SECONDS,
    SUPPORTED_DETECTOR_MODES,
    get_display_name,
    normalize_class_id,
)


def validate_detector_mode(mode_value):
    if not mode_value:
        return "yolov8"
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
    return "yolov8"


def parse_scan_frame_request(payload, session_defaults=None):
    defaults = session_defaults or {}
    if not isinstance(payload, dict):
        payload = {}

    frame_id = int(payload.get("frame_id", defaults.get("frame_id", 0)))
    raw_mode = payload.get("detector_mode", defaults.get("detector_mode", "yolov8"))
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
    raw_cls = det.get("class_id", "pasta_noodles")
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

    return {
        "class_id": cls_id,
        "display_name": str(disp),
        "confidence": conf,
        "bbox_norm": bbox_norm,
        "polygon_norm": polygon_norm,
        "temporally_confirmed": bool(det.get("temporally_confirmed", False)),
        "crop_jpeg_b64": det.get("crop_jpeg_b64"),
    }


def format_confirmed_inventory_item(item):
    raw_cls = item.get("class_id", "pasta_noodles")
    cls_id = normalize_class_id(raw_cls)
    disp = item.get("display_name") or get_display_name(cls_id)
    count = max(1, int(item.get("count", 1)))
    max_conf = round(float(item.get("max_conf", 0.5)), 2)
    return {
        "class_id": cls_id,
        "display_name": str(disp),
        "count": count,
        "max_conf": max_conf,
        "crop_jpeg_b64": item.get("crop_jpeg_b64"),
    }


def format_scan_response(
    frame_id,
    detector_mode,
    inference_ms,
    diagnostics,
    detections,
    confirmed_inventory,
):
    inf_ms = round(max(0.01, float(inference_ms)), 2)
    fps = round(1000.0 / inf_ms, 1)

    formatted_dets = []
    for d in detections:
        formatted_dets.append(format_detection_item(d))

    formatted_inv = []
    for item in confirmed_inventory:
        formatted_inv.append(format_confirmed_inventory_item(item))

    diag_out = {
        "edge_density_pct": round(float(diagnostics.get("edge_density_pct", 0.0)), 2),
        "orb_keypoints": int(diagnostics.get("orb_keypoints", 0)),
        "clahe_applied": bool(diagnostics.get("clahe_applied", True)),
    }
    if "good_matches" in diagnostics:
        diag_out["good_matches"] = int(diagnostics["good_matches"])
    if "ransac_inliers" in diagnostics:
        diag_out["ransac_inliers"] = int(diagnostics["ransac_inliers"])

    return {
        "frame_id": int(frame_id),
        "detector_mode": str(detector_mode),
        "inference_ms": inf_ms,
        "fps": fps,
        "diagnostics": diag_out,
        "detections": formatted_dets,
        "confirmed_inventory": formatted_inv,
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
    timeout_sec = max(0.2, min(10.0, timeout_sec))
    api_key = payload.get("api_key") or None

    return {
        "items": raw_items,
        "timeout_sec": timeout_sec,
        "api_key": api_key,
    }
