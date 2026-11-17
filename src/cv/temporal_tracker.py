from collections import deque

from src.server.config import (
    DEFAULT_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    TEMPORAL_MIN_HITS,
    TEMPORAL_WINDOW_SIZE,
    get_class_conf_threshold,
    get_display_name,
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
    sorted_dets = list(detections)
    n = len(sorted_dets)
    for i in range(n):
        for j in range(i + 1, n):
            if float(sorted_dets[j].get("confidence", 0.0)) > float(sorted_dets[i].get("confidence", 0.0)):
                temp = sorted_dets[i]
                sorted_dets[i] = sorted_dets[j]
                sorted_dets[j] = temp

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


def update_temporal_tracker(tracker_state, detections, conf_threshold=None):
    active_conf = float(tracker_state.get("conf_threshold", DEFAULT_CONF_THRESHOLD))
    if conf_threshold is not None:
        active_conf = float(conf_threshold)

    iou_threshold = float(tracker_state.get("iou_threshold", NMS_IOU_THRESHOLD))
    min_hits = int(tracker_state.get("min_hits", TEMPORAL_MIN_HITS))

    filtered_input = []
    for det in detections:
        conf_val = float(det.get("confidence", 0.0))
        raw_cls = det.get("class_id", "pasta_noodles")
        cls_id = normalize_class_id(raw_cls)
        req_conf = get_class_conf_threshold(cls_id, active_conf)
        if conf_val < req_conf:
            continue
        disp = det.get("display_name") or get_display_name(cls_id)
        bbox = det.get("bbox_norm", [0.0, 0.0, 1.0, 1.0])
        poly = det.get("polygon_norm", None)

        filtered_input.append({
            "class_id": cls_id,
            "display_name": disp,
            "confidence": round(conf_val, 2),
            "bbox_norm": list(bbox),
            "polygon_norm": poly,
            "temporally_confirmed": False,
        })

    deduped = apply_class_nms(filtered_input, iou_threshold=iou_threshold)

    frame_class_summary = {}
    for det in deduped:
        cls_id = det["class_id"]
        conf_val = det["confidence"]
        if cls_id not in frame_class_summary:
            frame_class_summary[cls_id] = {
                "count": 1,
                "max_conf": conf_val,
                "display_name": det["display_name"],
            }
        else:
            frame_class_summary[cls_id]["count"] += 1
            if conf_val > frame_class_summary[cls_id]["max_conf"]:
                frame_class_summary[cls_id]["max_conf"] = conf_val

    tracker_state["frames_processed"] = int(tracker_state.get("frames_processed", 0)) + 1
    tracker_state["window"].append(frame_class_summary)

    aggregate = {}
    for frame_entry in tracker_state["window"]:
        for cls_id, info in frame_entry.items():
            if cls_id not in aggregate:
                aggregate[cls_id] = {
                    "hits": 1,
                    "max_count": int(info["count"]),
                    "max_conf": float(info["max_conf"]),
                    "display_name": info["display_name"],
                }
            else:
                aggregate[cls_id]["hits"] += 1
                if int(info["count"]) > aggregate[cls_id]["max_count"]:
                    aggregate[cls_id]["max_count"] = int(info["count"])
                if float(info["max_conf"]) > aggregate[cls_id]["max_conf"]:
                    aggregate[cls_id]["max_conf"] = float(info["max_conf"])

    confirmed_set = set()
    confirmed_inventory = []
    sorted_class_ids = sorted(list(aggregate.keys()))
    for cls_id in sorted_class_ids:
        stats = aggregate[cls_id]
        if stats["hits"] >= min_hits:
            confirmed_set.add(cls_id)
            confirmed_inventory.append({
                "class_id": cls_id,
                "display_name": stats["display_name"],
                "count": int(stats["max_count"]),
                "max_conf": round(float(stats["max_conf"]), 2),
            })

    updated_detections = []
    for det in deduped:
        is_confirmed = det["class_id"] in confirmed_set
        det_copy = {
            "class_id": det["class_id"],
            "display_name": det["display_name"],
            "confidence": det["confidence"],
            "bbox_norm": det["bbox_norm"],
            "polygon_norm": det["polygon_norm"],
            "temporally_confirmed": bool(is_confirmed),
        }
        updated_detections.append(det_copy)

    return updated_detections, confirmed_inventory
