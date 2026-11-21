import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import cv2
import numpy as np

from src.cv.preprocess import encode_image_base64_jpeg
from src.cv.temporal_tracker import (
    apply_class_nms,
    apply_cross_class_nms,
    compute_iou,
    extract_positive_calories,
)
from src.cv.yolo_detector import run_opencv_region_fallback, run_yolo_detection
from src.server.config import (
    CROSS_CLASS_NMS_IOU,
    DEFAULT_CONF_THRESHOLD,
    GEMINI_RECIPE_MODEL,
    GEMINI_VISION_MODEL,
    GEMINI_VISION_TIMEOUT_SECONDS,
    NMS_IOU_THRESHOLD,
    NON_FOOD_LABELS,
    UNIFIED_CLASSES,
    VALID_CLASS_IDS,
    get_class_calories,
    get_display_name,
    get_gemini_thinking_level,
    is_valid_food_class,
    normalize_class_id,
)

GEMINI_DETECTION_CACHE = {}
GEMINI_CACHE_TTL_SEC = 15.0
GEMINI_CACHE_MAX_ENTRIES = 64
GEMINI_CLIENT_CACHE = {"api_key": None, "client": None}
GEMINI_CLIENT_LOCK = threading.Lock()
GEMINI_EXECUTOR = ThreadPoolExecutor(max_workers=2)
GEMINI_RUNTIME_STATE = {
    "last_error": None,
    "last_model": GEMINI_VISION_MODEL,
    "last_thinking_level": get_gemini_thinking_level(GEMINI_VISION_MODEL),
    "last_live_api": False,
}


def get_cached_gemini_client(api_key):
    resolved_key = str(api_key or "").strip()
    if not resolved_key:
        return None
    with GEMINI_CLIENT_LOCK:
        if GEMINI_CLIENT_CACHE.get("api_key") == resolved_key and GEMINI_CLIENT_CACHE.get("client") is not None:
            return GEMINI_CLIENT_CACHE["client"]
        from google import genai
        client = genai.Client(api_key=resolved_key)
        GEMINI_CLIENT_CACHE["api_key"] = resolved_key
        GEMINI_CLIENT_CACHE["client"] = client
        return client


def normalize_excluded_classes(excluded_classes):
    exc_set = set()
    if excluded_classes:
        for exc in excluded_classes:
            norm_exc = normalize_class_id(exc)
            if norm_exc:
                exc_set.add(norm_exc)
    return sorted(exc_set)


def compute_frame_signature(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    quantized = (thumb // 8).astype(np.uint8)
    return hashlib.sha1(quantized.tobytes()).hexdigest()


def get_valid_unified_class_ids():
    return set(VALID_CLASS_IDS)


def build_gemini_detection_schema():
    return {
        "type": "object",
        "properties": {
            "boxes": {
                "type": "array",
                "description": "Detected pantry and refrigerator food item bounding boxes.",
                "items": {
                    "type": "object",
                    "properties": {
                        "box_2d": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Bounding box [ymin, xmin, ymax, xmax] normalized to integer coordinates 0-1000.",
                        },
                        "class_id": {
                            "type": "string",
                            "enum": sorted(VALID_CLASS_IDS),
                            "description": "Canonical unified pantry or refrigerator food class_id.",
                        },
                        "display_name": {
                            "type": "string",
                            "description": "Specific recognizable food brand or item name.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Detection confidence score between 0.0 and 1.0.",
                        },
                        "estimated_calories": {
                            "type": "integer",
                            "description": "Estimated food energy in kcal for one visible unit or package of this item.",
                        },
                    },
                    "required": [
                        "box_2d",
                        "class_id",
                        "display_name",
                        "confidence",
                        "estimated_calories",
                    ],
                },
            }
        },
        "required": ["boxes"],
    }


def build_gemini_detection_prompt(excluded_classes=None):
    excluded_list = normalize_excluded_classes(excluded_classes)
    excluded_set = set(excluded_list)
    class_lines = []
    for entry in UNIFIED_CLASSES:
        if entry["class_id"] in excluded_set:
            continue
        examples = ", ".join(entry["prompts"])
        class_lines.append(f"- {entry['class_id']}: {entry['display_name']} ({examples})")
    classes_block = "\n".join(class_lines)
    exclusion_note = ""
    if excluded_list:
        exclusion_note = (
            "\nDo NOT detect or return any items belonging to these excluded_classes: "
            + ", ".join(excluded_list)
            + "."
        )
    return (
        "Detect only visible pantry and refrigerator food items in this image, "
        "including partially occluded, shadowed, tilted, or stacked packages, cans, jars, bottles, cartons, eggs, and produce.\n"
        "Strictly ignore non-food items including books, cups, bowls, plates, utensils, appliances, furniture, people, shelves, and countertops."
        f"{exclusion_note}\n"
        "Map every detected food item to the closest unified class_id from this list:\n"
        f"{classes_block}\n\n"
        "Return JSON matching the schema. For each item:\n"
        "- box_2d: [ymin, xmin, ymax, xmax] normalized to 0-1000.\n"
        "- class_id: exact unified class_id from the list above.\n"
        "- display_name: specific recognizable brand or food item name.\n"
        "- confidence: detection confidence score between 0.0 and 1.0.\n"
        "- estimated_calories: per-item integer calorie estimate in kcal."
    )


def clamp_estimated_calories(raw_value, class_id):
    if raw_value is None or isinstance(raw_value, bool):
        return get_class_calories(class_id, 1)
    try:
        val = float(raw_value)
    except (TypeError, ValueError):
        return get_class_calories(class_id, 1)
    if not math.isfinite(val) or val <= 0.0 or val > 3000.0:
        return get_class_calories(class_id, 1)
    return max(15, min(2500, int(round(val))))


def parse_gemini_boxes(raw_text, conf_threshold=DEFAULT_CONF_THRESHOLD, excluded_classes=None):
    if not raw_text:
        return []

    cleaned = str(raw_text).strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                cleaned = candidate
                break

    try:
        parsed = json.loads(cleaned)
    except Exception:
        return []

    if isinstance(parsed, dict) and "boxes" in parsed:
        boxes_list = parsed["boxes"]
    elif isinstance(parsed, list):
        boxes_list = parsed
    else:
        return []

    if not isinstance(boxes_list, list):
        return []

    valid_ids = get_valid_unified_class_ids()
    excluded_set = set(normalize_excluded_classes(excluded_classes))
    detections = []

    for item in boxes_list:
        if not isinstance(item, dict):
            continue
        box_2d = item.get("box_2d")
        if not isinstance(box_2d, (list, tuple)) or len(box_2d) < 4:
            continue

        try:
            ymin = max(0.0, min(1.0, float(box_2d[0]) / 1000.0))
            xmin = max(0.0, min(1.0, float(box_2d[1]) / 1000.0))
            ymax = max(0.0, min(1.0, float(box_2d[2]) / 1000.0))
            xmax = max(0.0, min(1.0, float(box_2d[3]) / 1000.0))
        except (TypeError, ValueError):
            continue

        x1 = xmin
        y1 = ymin
        x2 = xmax
        y2 = ymax
        if x2 < x1:
            temp_x = x1
            x1 = x2
            x2 = temp_x
        if y2 < y1:
            temp_y = y1
            y1 = y2
            y2 = temp_y

        bw = x2 - x1
        bh = y2 - y1
        if bw < 0.015 or bh < 0.015:
            continue

        raw_cls_field = item.get("class_id")
        if raw_cls_field is None:
            raw_cls_field = item.get("label")
        raw_disp_field = item.get("display_name")

        if raw_cls_field is not None:
            cls_clean = str(raw_cls_field).strip().lower()
            cls_spaced = cls_clean.replace("_", " ").replace("-", " ")
            if cls_clean in NON_FOOD_LABELS or cls_spaced in NON_FOOD_LABELS:
                continue

        cls_id = ""
        if raw_cls_field is not None and str(raw_cls_field).strip():
            if is_valid_food_class(raw_cls_field):
                cls_id = normalize_class_id(raw_cls_field)
            elif raw_disp_field is not None and is_valid_food_class(raw_disp_field):
                cls_id = normalize_class_id(raw_disp_field)
        elif raw_disp_field is not None and is_valid_food_class(raw_disp_field):
            cls_id = normalize_class_id(raw_disp_field)

        if not cls_id or cls_id not in valid_ids or not is_valid_food_class(cls_id) or cls_id in excluded_set:
            continue

        try:
            conf_val = float(item.get("confidence", 0.85))
        except (TypeError, ValueError):
            conf_val = 0.85
        conf_val = max(0.0, min(1.0, conf_val))

        req_conf = float(conf_threshold)
        if conf_val < req_conf:
            continue

        disp_raw = str(item.get("display_name") or "").strip()
        disp_name = disp_raw if disp_raw else get_display_name(cls_id)

        detections.append({
            "class_id": cls_id,
            "display_name": disp_name,
            "confidence": round(conf_val, 2),
            "bbox_norm": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            "polygon_norm": None,
            "temporally_confirmed": False,
            "estimated_calories": clamp_estimated_calories(item.get("estimated_calories"), cls_id),
            "use_base_conf": True,
        })

    return detections


def query_gemini_vision_api(
    bgr_image,
    api_key=None,
    model_name=None,
    timeout_sec=GEMINI_VISION_TIMEOUT_SECONDS,
    thinking_level=None,
    excluded_classes=None,
):
    resolved_model = str(model_name or GEMINI_VISION_MODEL).strip() or GEMINI_VISION_MODEL
    raw_thinking = str(thinking_level or "").strip().lower()
    if raw_thinking in ("minimal", "low", "medium", "high"):
        resolved_thinking = raw_thinking
    else:
        resolved_thinking = get_gemini_thinking_level(resolved_model)

    GEMINI_RUNTIME_STATE["last_model"] = resolved_model
    GEMINI_RUNTIME_STATE["last_thinking_level"] = resolved_thinking

    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key or not str(resolved_key).strip():
        GEMINI_RUNTIME_STATE["last_error"] = "GEMINI_API_KEY is not configured"
        GEMINI_RUNTIME_STATE["last_live_api"] = False
        return None

    b64_jpeg = encode_image_base64_jpeg(bgr_image, quality=82)
    prompt_text = build_gemini_detection_prompt(excluded_classes=excluded_classes)
    schema_dict = build_gemini_detection_schema()

    def worker():
        client = get_cached_gemini_client(resolved_key)
        if client is None or not hasattr(client, "interactions"):
            raise RuntimeError("Gemini interactions client is unavailable")
        interaction = client.interactions.create(
            model=resolved_model,
            input=[
                {"type": "text", "text": prompt_text},
                {"type": "image", "data": b64_jpeg, "mime_type": "image/jpeg"},
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema_dict,
            },
            generation_config={"thinking_level": resolved_thinking},
            store=False,
        )
        if getattr(interaction, "errors", None) and not getattr(interaction, "output_text", None):
            raise RuntimeError(str(interaction.errors))
        return getattr(interaction, "output_text", None)

    future = GEMINI_EXECUTOR.submit(worker)
    try:
        result_text = future.result(timeout=float(timeout_sec))
        if result_text is None:
            GEMINI_RUNTIME_STATE["last_error"] = "Gemini Vision API returned empty output"
            GEMINI_RUNTIME_STATE["last_live_api"] = False
            return None
        GEMINI_RUNTIME_STATE["last_error"] = None
        GEMINI_RUNTIME_STATE["last_live_api"] = True
        return result_text
    except FuturesTimeoutError:
        future.cancel()
        GEMINI_RUNTIME_STATE["last_error"] = f"Gemini Vision API timed out after {float(timeout_sec):.1f}s"
        GEMINI_RUNTIME_STATE["last_live_api"] = False
        return None
    except Exception as exc:
        GEMINI_RUNTIME_STATE["last_error"] = f"{type(exc).__name__}: {exc}"
        GEMINI_RUNTIME_STATE["last_live_api"] = False
        return None


def clone_detections(detections):
    cloned = []
    for det in detections:
        cls_id = det["class_id"]
        entry = {
            "class_id": cls_id,
            "display_name": det["display_name"],
            "confidence": float(det["confidence"]),
            "bbox_norm": list(det["bbox_norm"]),
            "polygon_norm": list(det["polygon_norm"]) if det.get("polygon_norm") else None,
            "temporally_confirmed": bool(det.get("temporally_confirmed", False)),
            "estimated_calories": int(det.get("estimated_calories") or get_class_calories(cls_id, 1)),
        }
        if det.get("use_base_conf"):
            entry["use_base_conf"] = True
        cloned.append(entry)
    return cloned


def store_cached_detections(cache_key, detections, backend, live_api):
    if len(GEMINI_DETECTION_CACHE) >= GEMINI_CACHE_MAX_ENTRIES:
        oldest_key = None
        oldest_ts = None
        for k, entry in GEMINI_DETECTION_CACHE.items():
            if oldest_ts is None or entry["ts"] < oldest_ts:
                oldest_ts = entry["ts"]
                oldest_key = k
        if oldest_key is not None:
            del GEMINI_DETECTION_CACHE[oldest_key]

    GEMINI_DETECTION_CACHE[cache_key] = {
        "ts": time.time(),
        "detections": clone_detections(detections),
        "backend": backend,
        "live_api": bool(live_api),
    }


def run_gemini_detection(
    bgr_image,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    iou_threshold=NMS_IOU_THRESHOLD,
    api_key=None,
    model_name=None,
    timeout_sec=GEMINI_VISION_TIMEOUT_SECONDS,
    use_cache=True,
    excluded_classes=None,
):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")

    resolved_model = model_name or GEMINI_VISION_MODEL
    excluded_list = normalize_excluded_classes(excluded_classes)
    excluded_set = set(excluded_list)
    sig = compute_frame_signature(bgr_image)
    excluded_key = ",".join(excluded_list)
    cache_key = f"{sig}:{round(float(conf_threshold), 2)}:{resolved_model}:{excluded_key}"

    if use_cache and cache_key in GEMINI_DETECTION_CACHE:
        entry = GEMINI_DETECTION_CACHE[cache_key]
        if (time.time() - entry["ts"]) <= GEMINI_CACHE_TTL_SEC:
            return clone_detections(entry["detections"]), {
                "detector_mode": "gemini_vision",
                "backend": entry["backend"],
                "live_api": entry["live_api"],
                "cached": True,
                "last_error": GEMINI_RUNTIME_STATE.get("last_error"),
            }

    raw_text = query_gemini_vision_api(
        bgr_image,
        api_key=api_key,
        model_name=resolved_model,
        timeout_sec=float(timeout_sec),
        excluded_classes=excluded_list,
    )

    if raw_text is not None:
        parsed_dets = parse_gemini_boxes(
            raw_text,
            conf_threshold=float(conf_threshold),
            excluded_classes=excluded_list,
        )
        deduped = apply_class_nms(parsed_dets, iou_threshold=float(iou_threshold))
        deduped = apply_cross_class_nms(deduped, iou_threshold=CROSS_CLASS_NMS_IOU)
        if use_cache:
            store_cached_detections(cache_key, deduped, resolved_model, True)
        return deduped, {
            "detector_mode": "gemini_vision",
            "backend": resolved_model,
            "live_api": True,
            "cached": False,
            "last_error": GEMINI_RUNTIME_STATE.get("last_error"),
        }

    yolo_dets, yolo_meta = run_yolo_detection(
        bgr_image,
        mode="yolov8",
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
    )
    filtered_yolo = []
    for yd in yolo_dets:
        if normalize_class_id(yd.get("class_id")) not in excluded_set:
            filtered_yolo.append(yd)

    if yolo_meta.get("backend") == "ultralytics":
        deduped_yolo = apply_class_nms(filtered_yolo, iou_threshold=float(iou_threshold))
        deduped_yolo = apply_cross_class_nms(deduped_yolo, iou_threshold=CROSS_CLASS_NMS_IOU)
        return deduped_yolo, {
            "detector_mode": "gemini_vision",
            "backend": "yolov8_offline_fallback",
            "live_api": False,
            "cached": False,
            "last_error": GEMINI_RUNTIME_STATE.get("last_error"),
        }

    cv_dets = run_opencv_region_fallback(
        bgr_image,
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
    )

    combined = []
    for yd in filtered_yolo:
        combined.append(yd)
    for cd in cv_dets:
        if normalize_class_id(cd.get("class_id")) in excluded_set:
            continue
        overlaps_yolo = False
        for yd in filtered_yolo:
            if compute_iou(cd.get("bbox_norm", [0, 0, 0, 0]), yd.get("bbox_norm", [0, 0, 0, 0])) >= 0.35:
                overlaps_yolo = True
                break
        if not overlaps_yolo:
            combined.append(cd)

    deduped_fallback = apply_class_nms(combined, iou_threshold=float(iou_threshold))
    deduped_fallback = apply_cross_class_nms(deduped_fallback, iou_threshold=CROSS_CLASS_NMS_IOU)
    return deduped_fallback, {
        "detector_mode": "gemini_vision",
        "backend": "opencv_offline_fallback",
        "live_api": False,
        "cached": False,
        "last_error": GEMINI_RUNTIME_STATE.get("last_error"),
    }


def run_hybrid_gemini_detection(
    bgr_image,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    iou_threshold=NMS_IOU_THRESHOLD,
    api_key=None,
    model_name=None,
    timeout_sec=GEMINI_VISION_TIMEOUT_SECONDS,
    excluded_classes=None,
):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")

    excluded_list = normalize_excluded_classes(excluded_classes)
    excluded_set = set(excluded_list)

    yolo_dets, yolo_meta = run_yolo_detection(
        bgr_image,
        mode="yolov8",
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
    )
    if excluded_set:
        filtered_yolo = []
        for yd in yolo_dets:
            if normalize_class_id(yd.get("class_id")) not in excluded_set:
                filtered_yolo.append(yd)
        yolo_dets = filtered_yolo

    max_yolo_conf = 0.0
    for yd in yolo_dets:
        c_val = float(yd.get("confidence", 0.0))
        if c_val > max_yolo_conf:
            max_yolo_conf = c_val

    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    gemini_dets = []
    gemini_meta = {
        "backend": "not_invoked",
        "live_api": False,
        "cached": False,
        "last_error": GEMINI_RUNTIME_STATE.get("last_error"),
    }

    if resolved_key or len(yolo_dets) < 2 or max_yolo_conf < 0.70:
        gemini_dets, gemini_meta = run_gemini_detection(
            bgr_image,
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
            api_key=resolved_key,
            model_name=model_name,
            timeout_sec=float(timeout_sec),
            use_cache=True,
            excluded_classes=excluded_list,
        )

    fused = []
    matched_gemini_indices = set()

    for yd in yolo_dets:
        best_g_idx = -1
        best_iou = 0.0
        for g_idx, gd in enumerate(gemini_dets):
            if g_idx in matched_gemini_indices:
                continue
            if gd["class_id"] != yd["class_id"]:
                continue
            iou_val = compute_iou(yd["bbox_norm"], gd["bbox_norm"])
            if iou_val >= float(iou_threshold) and iou_val > best_iou:
                best_iou = iou_val
                best_g_idx = g_idx

        if best_g_idx >= 0:
            matched_gemini_indices.add(best_g_idx)
            gd = gemini_dets[best_g_idx]
            y_conf = float(yd["confidence"])
            g_conf = float(gd["confidence"])
            chosen_box = list(gd["bbox_norm"]) if g_conf > y_conf else list(yd["bbox_norm"])
            boosted_conf = min(0.99, round(max(y_conf, g_conf) + 0.05, 2))
            gem_cal = extract_positive_calories(gd.get("estimated_calories"))
            if gem_cal is not None:
                est_cal = gem_cal
            else:
                yolo_cal = extract_positive_calories(yd.get("estimated_calories"))
                est_cal = yolo_cal if yolo_cal is not None else get_class_calories(yd["class_id"], 1)
            fused_item = {
                "class_id": yd["class_id"],
                "display_name": gd["display_name"] or yd["display_name"],
                "confidence": boosted_conf,
                "bbox_norm": chosen_box,
                "polygon_norm": None,
                "temporally_confirmed": False,
                "estimated_calories": est_cal,
            }
            if gd.get("use_base_conf") or yd.get("use_base_conf"):
                fused_item["use_base_conf"] = True
            fused.append(fused_item)
        else:
            yd_item = {
                "class_id": yd["class_id"],
                "display_name": yd["display_name"],
                "confidence": float(yd["confidence"]),
                "bbox_norm": list(yd["bbox_norm"]),
                "polygon_norm": yd.get("polygon_norm"),
                "temporally_confirmed": bool(yd.get("temporally_confirmed", False)),
                "estimated_calories": int(yd.get("estimated_calories") or get_class_calories(yd["class_id"], 1)),
            }
            if yd.get("use_base_conf"):
                yd_item["use_base_conf"] = True
            fused.append(yd_item)

    for g_idx, gd in enumerate(gemini_dets):
        if g_idx in matched_gemini_indices:
            continue
        gd_item = {
            "class_id": gd["class_id"],
            "display_name": gd["display_name"],
            "confidence": float(gd["confidence"]),
            "bbox_norm": list(gd["bbox_norm"]),
            "polygon_norm": gd.get("polygon_norm"),
            "temporally_confirmed": bool(gd.get("temporally_confirmed", False)),
            "estimated_calories": int(gd.get("estimated_calories") or get_class_calories(gd["class_id"], 1)),
        }
        if gd.get("use_base_conf"):
            gd_item["use_base_conf"] = True
        fused.append(gd_item)

    deduped = apply_class_nms(fused, iou_threshold=float(iou_threshold))
    deduped = apply_cross_class_nms(deduped, iou_threshold=CROSS_CLASS_NMS_IOU)

    yolo_backend = yolo_meta.get("backend", "none")
    gemini_backend = gemini_meta.get("backend", "none")
    if gemini_backend == "not_invoked":
        hybrid_backend = yolo_backend
    else:
        hybrid_backend = f"{yolo_backend}+{gemini_backend}"

    return deduped, {
        "detector_mode": "hybrid_gemini",
        "backend": hybrid_backend,
        "yolo_backend": yolo_backend,
        "gemini_backend": gemini_backend,
        "live_api": bool(gemini_meta.get("live_api", False)),
        "cached": bool(gemini_meta.get("cached", False)),
        "last_error": gemini_meta.get("last_error", GEMINI_RUNTIME_STATE.get("last_error")),
    }


def get_gemini_detector_status():
    return {
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
        "vision_model": GEMINI_VISION_MODEL,
        "recipe_model": GEMINI_RECIPE_MODEL,
        "thinking_level": GEMINI_RUNTIME_STATE.get("last_thinking_level") or get_gemini_thinking_level(GEMINI_VISION_MODEL),
        "live_api": bool(GEMINI_RUNTIME_STATE.get("last_live_api", False)),
        "last_error": GEMINI_RUNTIME_STATE.get("last_error"),
        "cache_entries": len(GEMINI_DETECTION_CACHE),
    }
