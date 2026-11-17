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
)

ACTIVE_SESSIONS = {}


def get_or_create_session(session_id="default"):
    sid = str(session_id or "default")
    if sid not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[sid] = {
            "session_id": sid,
            "frame_id": 0,
            "detector_mode": "yolov8",
            "enable_clahe": True,
            "conf_threshold": DEFAULT_CONF_THRESHOLD,
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


def record_session_frame(
    session_id,
    detections,
    conf_threshold=None,
    min_hits_override=None,
    fast_confirm_conf=None,
):
    session = get_or_create_session(session_id)
    session["frame_id"] = int(session.get("frame_id", 0)) + 1
    annotated_dets, confirmed_inventory = update_temporal_tracker(
        session["tracker"],
        detections,
        conf_threshold=conf_threshold,
        min_hits_override=min_hits_override,
        fast_confirm_conf=fast_confirm_conf,
    )
    return annotated_dets, confirmed_inventory
