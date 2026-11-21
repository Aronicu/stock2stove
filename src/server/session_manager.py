from src.cv.temporal_tracker import (
    create_tracker_state,
    reset_tracker_state,
    update_temporal_tracker,
)
from src.server.config import (
    DEFAULT_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    TEMPORAL_MIN_HITS,
    TEMPORAL_WINDOW_SIZE,
    get_display_name,
    is_valid_food_class,
    normalize_class_id,
)

ACTIVE_SESSIONS = {}


def get_or_create_session(session_id="default"):
    sid = str(session_id or "default")
    if sid not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[sid] = {
            "session_id": sid,
            "frame_id": 0,
            "detector_mode": "gemini_vision",
            "enable_clahe": True,
            "conf_threshold": DEFAULT_CONF_THRESHOLD,
            "excluded_classes": set(),
            "last_min_hits": TEMPORAL_MIN_HITS,
            "last_fast_confirm_conf": None,
            "tracker": create_tracker_state(
                window_size=TEMPORAL_WINDOW_SIZE,
                min_hits=TEMPORAL_MIN_HITS,
                conf_threshold=DEFAULT_CONF_THRESHOLD,
                iou_threshold=NMS_IOU_THRESHOLD,
            ),
        }
    return ACTIVE_SESSIONS[sid]


def reset_session(session_id="default"):
    session = get_or_create_session(session_id)
    session["frame_id"] = 0
    session["excluded_classes"] = set()
    session["last_min_hits"] = int(session["tracker"].get("min_hits", TEMPORAL_MIN_HITS))
    session["last_fast_confirm_conf"] = None
    reset_tracker_state(session["tracker"])
    return session


def remove_session(session_id="default"):
    sid = str(session_id or "default")
    if sid in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[sid]


def update_session_config(session_id, detector_mode=None, enable_clahe=None, conf_threshold=None):
    session = get_or_create_session(session_id)
    if detector_mode is not None:
        session["detector_mode"] = str(detector_mode)
    if enable_clahe is not None:
        session["enable_clahe"] = bool(enable_clahe)
    if conf_threshold is not None:
        session["conf_threshold"] = float(conf_threshold)
        session["tracker"]["conf_threshold"] = float(conf_threshold)
    return session


def get_session_excluded_classes(session_id="default"):
    session = get_or_create_session(session_id)
    return sorted(list(session.get("excluded_classes", set())))


def compute_session_confirmed_inventory(session):
    excluded = session.get("excluded_classes", set())
    tracker = session.get("tracker", {})
    window = tracker.get("window", [])
    min_hits = max(1, int(session.get("last_min_hits", tracker.get("min_hits", TEMPORAL_MIN_HITS))))
    fast_confirm_conf = session.get("last_fast_confirm_conf")

    aggregate = {}
    for frame_entry in window:
        if not isinstance(frame_entry, dict):
            continue
        for raw_cls, info in frame_entry.items():
            cls_id = normalize_class_id(raw_cls)
            if not is_valid_food_class(cls_id) or cls_id in excluded:
                continue
            info_crop = info.get("crop_jpeg_b64")
            cnt_val = max(1, int(info.get("count", 1)))
            conf_val = float(info.get("max_conf", 0.5))
            disp = info.get("display_name") or get_display_name(cls_id)
            if cls_id not in aggregate:
                aggregate[cls_id] = {
                    "hits": 1,
                    "counts": [cnt_val],
                    "max_conf": conf_val,
                    "display_name": disp,
                    "crop_jpeg_b64": info_crop,
                }
            else:
                aggregate[cls_id]["hits"] += 1
                aggregate[cls_id]["counts"].append(cnt_val)
                if conf_val > aggregate[cls_id]["max_conf"]:
                    aggregate[cls_id]["max_conf"] = conf_val
                    if info_crop:
                        aggregate[cls_id]["crop_jpeg_b64"] = info_crop
                elif not aggregate[cls_id].get("crop_jpeg_b64") and info_crop:
                    aggregate[cls_id]["crop_jpeg_b64"] = info_crop

    confirmed_inventory = []
    for cls_id in sorted(list(aggregate.keys())):
        stats = aggregate[cls_id]
        is_hit_confirmed = stats["hits"] >= min_hits
        is_fast_confirmed = fast_confirm_conf is not None and float(stats["max_conf"]) >= float(fast_confirm_conf)
        if is_hit_confirmed or is_fast_confirmed:
            counts_sorted = sorted(stats["counts"])
            median_count = counts_sorted[len(counts_sorted) // 2]
            confirmed_inventory.append({
                "class_id": cls_id,
                "display_name": stats["display_name"],
                "count": int(median_count),
                "max_conf": round(float(stats["max_conf"]), 2),
                "crop_jpeg_b64": stats.get("crop_jpeg_b64"),
            })
    return confirmed_inventory


def delete_session_item(session_id="default", class_id=""):
    session = get_or_create_session(session_id)
    cls_id = normalize_class_id(class_id)
    if cls_id:
        session["excluded_classes"].add(cls_id)
        window = session.get("tracker", {}).get("window", [])
        for frame_entry in window:
            if not isinstance(frame_entry, dict):
                continue
            keys_to_remove = []
            for key in frame_entry.keys():
                if key == cls_id or normalize_class_id(key) == cls_id:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del frame_entry[key]
    confirmed_inventory = compute_session_confirmed_inventory(session)
    return confirmed_inventory, sorted(list(session["excluded_classes"]))


def restore_session_item(session_id="default", class_id=""):
    session = get_or_create_session(session_id)
    cls_id = normalize_class_id(class_id)
    if cls_id and cls_id in session["excluded_classes"]:
        session["excluded_classes"].discard(cls_id)
    confirmed_inventory = compute_session_confirmed_inventory(session)
    return confirmed_inventory, sorted(list(session["excluded_classes"]))


def record_session_frame(
    session_id,
    detections,
    conf_threshold=None,
    min_hits_override=None,
    fast_confirm_conf=None,
):
    session = get_or_create_session(session_id)
    session["frame_id"] = int(session.get("frame_id", 0)) + 1
    if min_hits_override is not None:
        session["last_min_hits"] = max(1, int(min_hits_override))
    else:
        session["last_min_hits"] = int(session["tracker"].get("min_hits", TEMPORAL_MIN_HITS))
    session["last_fast_confirm_conf"] = fast_confirm_conf

    excluded = session.get("excluded_classes", set())
    filtered_input = []
    for det in detections:
        raw_cls = det.get("class_id") or det.get("display_name") or ""
        if not is_valid_food_class(raw_cls):
            continue
        cls_id = normalize_class_id(raw_cls)
        if cls_id in excluded:
            continue
        det_copy = dict(det)
        det_copy["class_id"] = cls_id
        filtered_input.append(det_copy)

    annotated_dets, confirmed_inventory = update_temporal_tracker(
        session["tracker"],
        filtered_input,
        conf_threshold=conf_threshold,
        min_hits_override=min_hits_override,
        fast_confirm_conf=fast_confirm_conf,
    )

    clean_annotated = []
    for det in annotated_dets:
        cls_id = normalize_class_id(det.get("class_id") or "")
        if not is_valid_food_class(cls_id) or cls_id in excluded:
            continue
        clean_annotated.append(det)

    clean_confirmed = []
    for item in confirmed_inventory:
        cls_id = normalize_class_id(item.get("class_id") or "")
        if not is_valid_food_class(cls_id) or cls_id in excluded:
            continue
        clean_confirmed.append(item)

    return clean_annotated, clean_confirmed
