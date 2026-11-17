import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import cv2
import numpy as np

from src.cv.preprocess import encode_image_base64_jpeg
from src.cv.temporal_tracker import apply_class_nms, compute_iou
from src.cv.yolo_detector import run_opencv_region_fallback, run_yolo_detection
from src.server.config import (
    DEFAULT_CONF_THRESHOLD,
    GEMINI_VISION_MODEL,
    NMS_IOU_THRESHOLD,
    UNIFIED_CLASSES,
    get_class_conf_threshold,
    get_display_name,
    normalize_class_id,
)

GEMINI_DETECTION_CACHE = {}
GEMINI_CACHE_TTL_SEC = 15.0
GEMINI_CACHE_MAX_ENTRIES = 64


def compute_frame_signature(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    quantized = (thumb // 8).astype(np.uint8)
    return hashlib.sha1(quantized.tobytes()).hexdigest()


def get_valid_unified_class_ids():
    valid_ids = set()
    for entry in UNIFIED_CLASSES:
        valid_ids.add(entry["class_id"])
    return valid_ids


def build_gemini_detection_schema():
    return {
        "type": "object",
        "properties": {
            "boxes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "box_2d": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "class_id": {"type": "string"},
                        "display_name": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["box_2d", "class_id", "display_name", "confidence"],
                },
            }
        },
        "required": ["boxes"],
    }


def build_gemini_detection_prompt():
    class_lines = []
    for entry in UNIFIED_CLASSES:
        examples = ", ".join(entry["prompts"])
        class_lines.append(f"- {entry['class_id']}: {entry['display_name']} ({examples})")
    classes_block = "\n".join(class_lines)
    return (
        "Detect all visible pantry and refrigerator food items in this image, "
        "including partially occluded, shadowed, tilted, or stacked packages, cans, jars, bottles, cartons, eggs, and produce.\n"
        "Map every detected item to the closest unified class_id from this list:\n"
        f"{classes_block}\n\n"
        "Return JSON matching the schema. For each item:\n"
        "- box_2d: [ymin, xmin, ymax, xmax] normalized to 0-1000.\n"
        "- class_id: exact unified class_id from the list above.\n"
        "- display_name: specific recognizable brand or food item name.\n"
        "- confidence: detection confidence score between 0.0 and 1.0."
    )


def parse_gemini_boxes(raw_text, conf_threshold=DEFAULT_CONF_THRESHOLD):
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
        if bw <= 0.005 or bh <= 0.005:
            continue

        raw_cls = item.get("class_id") or item.get("label") or item.get("display_name")
        cls_id = normalize_class_id(raw_cls)
        if cls_id not in valid_ids and item.get("display_name"):
            alt_cls = normalize_class_id(item.get("display_name"))
            if alt_cls in valid_ids:
                cls_id = alt_cls
        if cls_id not in valid_ids:
            cls_id = "pasta_noodles"

        try:
            conf_val = float(item.get("confidence", 0.85))
        except (TypeError, ValueError):
            conf_val = 0.85
        conf_val = max(0.0, min(1.0, conf_val))

        req_conf = get_class_conf_threshold(cls_id, float(conf_threshold))
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
        })

    return detections


def query_gemini_vision_api(bgr_image, api_key=None, model_name=None, timeout_sec=4.0):
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        return None

    resolved_model = model_name or GEMINI_VISION_MODEL
    b64_jpeg = encode_image_base64_jpeg(bgr_image, quality=82)
    prompt_text = build_gemini_detection_prompt()
    schema_dict = build_gemini_detection_schema()

    def worker():
        from google import genai

        client = genai.Client(api_key=resolved_key)
        if hasattr(client, "interactions"):
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
                generation_config={"temperature": 0.1},
                store=False,
            )
            return interaction.output_text
        return None

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(worker)
    try:
        result_text = future.result(timeout=float(timeout_sec))
        executor.shutdown(wait=False, cancel_futures=True)
        return result_text
    except (FuturesTimeoutError, Exception):
        executor.shutdown(wait=False, cancel_futures=True)
        return None


def clone_detections(detections):
    cloned = []
    for det in detections:
        cloned.append({
            "class_id": det["class_id"],
            "display_name": det["display_name"],
            "confidence": float(det["confidence"]),
            "bbox_norm": list(det["bbox_norm"]),
            "polygon_norm": list(det["polygon_norm"]) if det.get("polygon_norm") else None,
            "temporally_confirmed": bool(det.get("temporally_confirmed", False)),
        })
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
    timeout_sec=4.0,
    use_cache=True,
):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")

    resolved_model = model_name or GEMINI_VISION_MODEL
    sig = compute_frame_signature(bgr_image)
    cache_key = f"{sig}:{round(float(conf_threshold), 2)}:{resolved_model}"

    if use_cache and cache_key in GEMINI_DETECTION_CACHE:
        entry = GEMINI_DETECTION_CACHE[cache_key]
        if (time.time() - entry["ts"]) <= GEMINI_CACHE_TTL_SEC:
            return clone_detections(entry["detections"]), {
                "detector_mode": "gemini_vision",
                "backend": entry["backend"],
                "live_api": entry["live_api"],
                "cached": True,
            }

    raw_text = query_gemini_vision_api(
        bgr_image,
        api_key=api_key,
        model_name=resolved_model,
        timeout_sec=float(timeout_sec),
    )

    if raw_text is not None:
        parsed_dets = parse_gemini_boxes(raw_text, conf_threshold=float(conf_threshold))
        deduped = apply_class_nms(parsed_dets, iou_threshold=float(iou_threshold))
        if use_cache:
            store_cached_detections(cache_key, deduped, resolved_model, True)
        return deduped, {
            "detector_mode": "gemini_vision",
            "backend": resolved_model,
            "live_api": True,
            "cached": False,
        }

    yolo_dets, _ = run_yolo_detection(
        bgr_image,
        mode="yolov8",
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
    )
    cv_dets = run_opencv_region_fallback(
        bgr_image,
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
    )

    combined = []
    for yd in yolo_dets:
        combined.append(yd)
    for cd in cv_dets:
        overlaps_yolo = False
        for yd in yolo_dets:
            if compute_iou(cd.get("bbox_norm", [0, 0, 0, 0]), yd.get("bbox_norm", [0, 0, 0, 0])) >= 0.35:
                overlaps_yolo = True
                break
        if not overlaps_yolo:
            combined.append(cd)

    deduped_fallback = apply_class_nms(combined, iou_threshold=float(iou_threshold))
    return deduped_fallback, {
        "detector_mode": "gemini_vision",
        "backend": "gemini_offline_fallback",
        "live_api": False,
        "cached": False,
    }


def run_hybrid_gemini_detection(
    bgr_image,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    iou_threshold=NMS_IOU_THRESHOLD,
    api_key=None,
    model_name=None,
):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")

    yolo_dets, yolo_meta = run_yolo_detection(
        bgr_image,
        mode="yolov8",
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
    )

    max_yolo_conf = 0.0
    for yd in yolo_dets:
        c_val = float(yd.get("confidence", 0.0))
        if c_val > max_yolo_conf:
            max_yolo_conf = c_val

    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    gemini_dets = []
    gemini_meta = {"backend": "not_invoked", "live_api": False}

    if resolved_key or len(yolo_dets) < 2 or max_yolo_conf < 0.70:
        gemini_dets, gemini_meta = run_gemini_detection(
            bgr_image,
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
            api_key=resolved_key,
            model_name=model_name,
            use_cache=True,
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
            fused.append({
                "class_id": yd["class_id"],
                "display_name": gd["display_name"] or yd["display_name"],
                "confidence": boosted_conf,
                "bbox_norm": chosen_box,
                "polygon_norm": None,
                "temporally_confirmed": False,
            })
        else:
            fused.append({
                "class_id": yd["class_id"],
                "display_name": yd["display_name"],
                "confidence": float(yd["confidence"]),
                "bbox_norm": list(yd["bbox_norm"]),
                "polygon_norm": yd.get("polygon_norm"),
                "temporally_confirmed": bool(yd.get("temporally_confirmed", False)),
            })

    for g_idx, gd in enumerate(gemini_dets):
        if g_idx in matched_gemini_indices:
            continue
        fused.append({
            "class_id": gd["class_id"],
            "display_name": gd["display_name"],
            "confidence": float(gd["confidence"]),
            "bbox_norm": list(gd["bbox_norm"]),
            "polygon_norm": gd.get("polygon_norm"),
            "temporally_confirmed": bool(gd.get("temporally_confirmed", False)),
        })

    deduped = apply_class_nms(fused, iou_threshold=float(iou_threshold))
    return deduped, {
        "detector_mode": "hybrid_gemini",
        "yolo_backend": yolo_meta.get("backend", "none"),
        "gemini_backend": gemini_meta.get("backend", "none"),
        "live_api": bool(gemini_meta.get("live_api", False)),
    }


def get_gemini_detector_status():
    return {
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
        "vision_model": GEMINI_VISION_MODEL,
        "cache_entries": len(GEMINI_DETECTION_CACHE),
    }
