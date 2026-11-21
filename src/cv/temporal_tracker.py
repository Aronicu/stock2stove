import math
from collections import deque

from src.server.config import (
    CROSS_CLASS_NMS_IOU,
    DEFAULT_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    TEMPORAL_MIN_HITS,
    TEMPORAL_WINDOW_SIZE,
    get_class_calories,
    get_class_conf_threshold,
    get_display_name,
    is_valid_food_class,
    normalize_class_id,
)


def compute_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
    area_b = max(0.0, (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1])))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def apply_class_nms(detections, iou_threshold=NMS_IOU_THRESHOLD):
    sorted_dets = sorted(
        list(detections),
        key=lambda d: float(d.get("confidence", 0.0)),
        reverse=True,
    )

    kept = []
    for det in sorted_dets:
        det_cls = det.get("class_id", "")
        det_box = det.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])
        overlap = False
        for existing in kept:
            if existing.get("class_id", "") == det_cls:
                iou = compute_iou(existing.get("bbox_norm", [0.0, 0.0, 0.0, 0.0]), det_box)
                if iou >= float(iou_threshold):
                    overlap = True
                    break
        if not overlap:
            kept.append(det)
    return kept


def apply_cross_class_nms(detections, iou_threshold=CROSS_CLASS_NMS_IOU):
    thr = CROSS_CLASS_NMS_IOU if iou_threshold is None else float(iou_threshold)
    sorted_dets = sorted(
        list(detections),
        key=lambda d: float(d.get("confidence", 0.0)),
        reverse=True,
    )

    kept = []
    for det in sorted_dets:
        det_box = det.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])
        overlap = False
        for existing in kept:
            iou = compute_iou(existing.get("bbox_norm", [0.0, 0.0, 0.0, 0.0]), det_box)
            if iou >= thr:
                overlap = True
                break
        if not overlap:
            kept.append(det)
    return kept


def create_tracker_state(
    window_size=TEMPORAL_WINDOW_SIZE,
    min_hits=TEMPORAL_MIN_HITS,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    iou_threshold=NMS_IOU_THRESHOLD,
):
    return {
        "window_size": int(window_size),
        "min_hits": int(min_hits),
        "conf_threshold": float(conf_threshold),
        "iou_threshold": float(iou_threshold),
        "frames_processed": 0,
        "window": deque(maxlen=int(window_size)),
    }


def reset_tracker_state(tracker_state):
    tracker_state["frames_processed"] = 0
    tracker_state["window"].clear()
    return tracker_state


def purge_class_from_tracker(tracker_state, class_id):
    if not isinstance(tracker_state, dict):
        return tracker_state
    raw_clean = str(class_id or "").strip().lower()
    target_cls = normalize_class_id(class_id)
    window = tracker_state.get("window")
    if not window:
        return tracker_state
    for frame_entry in window:
        if not isinstance(frame_entry, dict):
            continue
        keys_to_remove = []
        for key in frame_entry.keys():
            norm_key = normalize_class_id(key)
            if key == class_id or str(key).strip().lower() == raw_clean or (target_cls and norm_key == target_cls):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del frame_entry[key]
    return tracker_state


def extract_positive_calories(raw_value):
    if raw_value is None or isinstance(raw_value, bool):
        return None
    try:
        val = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val) or val <= 0.0:
        return None
    return int(round(val))


def compute_median_unit_calories(unit_calories, class_id):
    if unit_calories:
        sorted_cals = sorted(unit_calories)
        mid = len(sorted_cals) // 2
        if len(sorted_cals) % 2 == 1:
            resolved = int(sorted_cals[mid])
        else:
            resolved = int(round((sorted_cals[mid - 1] + sorted_cals[mid]) / 2.0))
        if resolved > 0:
            return resolved
    return get_class_calories(class_id, 1)


def build_confirmed_inventory_from_window(
    window,
    min_hits=TEMPORAL_MIN_HITS,
    fast_confirm_conf=None,
    excluded_classes=None,
):
    excluded_set = set()
    if excluded_classes:
        for exc in excluded_classes:
            norm_exc = normalize_class_id(exc)
            if norm_exc:
                excluded_set.add(norm_exc)

    resolved_min_hits = max(1, int(min_hits))
    aggregate = {}
    for frame_entry in window:
        if not isinstance(frame_entry, dict):
            continue
        for raw_cls, info in frame_entry.items():
            if not is_valid_food_class(raw_cls):
                continue
            cls_id = normalize_class_id(raw_cls)
            if not is_valid_food_class(cls_id) or cls_id in excluded_set:
                continue
            info_crop = info.get("crop_jpeg_b64")
            cnt_val = max(1, int(info.get("count", 1)))
            conf_val = float(info.get("max_conf", 0.5))

            frame_unit_cals = []
            raw_unit_cals = info.get("unit_calories")
            if isinstance(raw_unit_cals, list):
                for c_item in raw_unit_cals:
                    c_int = extract_positive_calories(c_item)
                    if c_int is not None:
                        frame_unit_cals.append(c_int)
            if not frame_unit_cals:
                fallback_cal = extract_positive_calories(info.get("estimated_calories"))
                if fallback_cal is not None:
                    frame_unit_cals.append(fallback_cal)

            if cls_id not in aggregate:
                aggregate[cls_id] = {
                    "hits": 1,
                    "counts": [cnt_val],
                    "max_conf": conf_val,
                    "display_name": info.get("display_name") or get_display_name(cls_id),
                    "crop_jpeg_b64": info_crop,
                    "unit_calories": list(frame_unit_cals),
                }
            else:
                aggregate[cls_id]["hits"] += 1
                aggregate[cls_id]["counts"].append(cnt_val)
                aggregate[cls_id]["unit_calories"].extend(frame_unit_cals)
                if conf_val > aggregate[cls_id]["max_conf"]:
                    aggregate[cls_id]["max_conf"] = conf_val
                    if info_crop:
                        aggregate[cls_id]["crop_jpeg_b64"] = info_crop
                elif not aggregate[cls_id].get("crop_jpeg_b64") and info_crop:
                    aggregate[cls_id]["crop_jpeg_b64"] = info_crop

    confirmed_inventory = []
    sorted_class_ids = sorted(list(aggregate.keys()))
    for cls_id in sorted_class_ids:
        stats = aggregate[cls_id]
        is_hit_confirmed = stats["hits"] >= resolved_min_hits
        is_fast_confirmed = fast_confirm_conf is not None and float(stats["max_conf"]) >= float(fast_confirm_conf)
        if is_hit_confirmed or is_fast_confirmed:
            counts_sorted = sorted(stats["counts"])
            consensus_count = int(counts_sorted[len(counts_sorted) // 2])
            calories_per_unit = compute_median_unit_calories(stats.get("unit_calories", []), cls_id)
            estimated_calories = int(calories_per_unit * consensus_count)
            confirmed_inventory.append({
                "class_id": cls_id,
                "display_name": stats["display_name"],
                "count": consensus_count,
                "max_conf": round(float(stats["max_conf"]), 2),
                "crop_jpeg_b64": stats.get("crop_jpeg_b64"),
                "calories_per_unit": calories_per_unit,
                "estimated_calories": estimated_calories,
            })
    return confirmed_inventory


def update_temporal_tracker(
    tracker_state,
    detections,
    conf_threshold=None,
    min_hits_override=None,
    fast_confirm_conf=None,
    excluded_classes=None,
):
    active_conf = float(tracker_state.get("conf_threshold", DEFAULT_CONF_THRESHOLD))
    if conf_threshold is not None:
        active_conf = float(conf_threshold)

    iou_threshold = float(tracker_state.get("iou_threshold", NMS_IOU_THRESHOLD))
    min_hits = int(tracker_state.get("min_hits", TEMPORAL_MIN_HITS))
    if min_hits_override is not None:
        min_hits = max(1, int(min_hits_override))

    excluded_set = set()
    if excluded_classes:
        for exc in excluded_classes:
            norm_exc = normalize_class_id(exc)
            if norm_exc:
                excluded_set.add(norm_exc)

    filtered_input = []
    for det in detections:
        conf_val = float(det.get("confidence", 0.0))
        raw_cls = det.get("class_id")
        if not is_valid_food_class(raw_cls):
            continue
        cls_id = normalize_class_id(raw_cls)
        if not is_valid_food_class(cls_id) or cls_id in excluded_set:
            continue
        req_conf = get_class_conf_threshold(cls_id, active_conf)
        if conf_val < req_conf:
            continue
        disp = det.get("display_name") or get_display_name(cls_id)
        bbox = det.get("bbox_norm", [0.0, 0.0, 1.0, 1.0])
        poly = det.get("polygon_norm", None)
        est_cal = extract_positive_calories(det.get("estimated_calories"))
        if est_cal is None:
            est_cal = get_class_calories(cls_id, 1)

        filtered_input.append({
            "class_id": cls_id,
            "display_name": disp,
            "confidence": round(conf_val, 2),
            "bbox_norm": list(bbox),
            "polygon_norm": poly,
            "temporally_confirmed": False,
            "crop_jpeg_b64": det.get("crop_jpeg_b64"),
            "estimated_calories": int(est_cal),
        })

    deduped = apply_class_nms(filtered_input, iou_threshold=iou_threshold)
    deduped = apply_cross_class_nms(deduped, iou_threshold=CROSS_CLASS_NMS_IOU)

    frame_class_summary = {}
    for det in deduped:
        cls_id = det["class_id"]
        conf_val = det["confidence"]
        crop_b64 = det.get("crop_jpeg_b64")
        unit_cal = extract_positive_calories(det.get("estimated_calories"))
        if unit_cal is None:
            unit_cal = get_class_calories(cls_id, 1)
        if cls_id not in frame_class_summary:
            frame_class_summary[cls_id] = {
                "count": 1,
                "max_conf": conf_val,
                "display_name": det["display_name"],
                "crop_jpeg_b64": crop_b64,
                "unit_calories": [unit_cal],
                "estimated_calories": unit_cal,
            }
        else:
            frame_class_summary[cls_id]["count"] += 1
            frame_class_summary[cls_id]["unit_calories"].append(unit_cal)
            if conf_val > frame_class_summary[cls_id]["max_conf"]:
                frame_class_summary[cls_id]["max_conf"] = conf_val
                frame_class_summary[cls_id]["estimated_calories"] = unit_cal
                if crop_b64:
                    frame_class_summary[cls_id]["crop_jpeg_b64"] = crop_b64
            elif not frame_class_summary[cls_id].get("crop_jpeg_b64") and crop_b64:
                frame_class_summary[cls_id]["crop_jpeg_b64"] = crop_b64

    tracker_state["frames_processed"] = int(tracker_state.get("frames_processed", 0)) + 1
    tracker_state["window"].append(frame_class_summary)

    confirmed_inventory = build_confirmed_inventory_from_window(
        tracker_state["window"],
        min_hits=min_hits,
        fast_confirm_conf=fast_confirm_conf,
        excluded_classes=excluded_set,
    )

    confirmed_set = set()
    for item in confirmed_inventory:
        confirmed_set.add(item["class_id"])

    updated_detections = []
    for det in deduped:
        is_confirmed = det["class_id"] in confirmed_set
        det_cal = extract_positive_calories(det.get("estimated_calories"))
        if det_cal is None:
            det_cal = get_class_calories(det["class_id"], 1)
        det_copy = {
            "class_id": det["class_id"],
            "display_name": det["display_name"],
            "confidence": det["confidence"],
            "bbox_norm": det["bbox_norm"],
            "polygon_norm": det["polygon_norm"],
            "temporally_confirmed": bool(is_confirmed),
            "crop_jpeg_b64": det.get("crop_jpeg_b64"),
            "estimated_calories": int(det_cal),
        }
        updated_detections.append(det_copy)

    return updated_detections, confirmed_inventory
