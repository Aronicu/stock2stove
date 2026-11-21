import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cv.baseline_detector import run_baseline_detection
from src.cv.gemini_detector import run_gemini_detection, run_hybrid_gemini_detection
from src.cv.preprocess import (
    preprocess_frame,
    unletterbox_bbox_norm,
    unletterbox_polygon_norm,
)
from src.cv.temporal_tracker import (
    apply_class_nms,
    apply_cross_class_nms,
    create_tracker_state,
    update_temporal_tracker,
)
from src.cv.yolo_detector import run_yolo_detection
from src.llm_recipe import generate_structured_recipes
from src.server.config import (
    CROSS_CLASS_NMS_IOU,
    DEFAULT_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD,
    TARGET_IMG_SIZE,
    TEMPORAL_MIN_HITS,
    TEMPORAL_WINDOW_SIZE,
    UNIFIED_CLASSES,
    get_class_calories,
    get_display_name,
    is_valid_food_class,
    normalize_class_id,
)
from src.server.schemas import validate_detector_mode

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

CLASS_INDEX_MAP = {}
for cls_entry in UNIFIED_CLASSES:
    CLASS_INDEX_MAP[cls_entry["class_id"]] = int(cls_entry["id"])


def preprocess(frame, enable_clahe=True, target_size=TARGET_IMG_SIZE):
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty or None.")
    return preprocess_frame(
        frame,
        enable_clahe=bool(enable_clahe),
        target_size=int(target_size),
    )


def map_detections_to_original_frame(raw_dets, meta):
    target_size = float(meta.get("target_size", TARGET_IMG_SIZE))
    mapped = []
    for det in raw_dets:
        bbox = det.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])
        bbox_px = [
            float(bbox[0]) * target_size,
            float(bbox[1]) * target_size,
            float(bbox[2]) * target_size,
            float(bbox[3]) * target_size,
        ]
        orig_bbox = unletterbox_bbox_norm(bbox_px, meta)
        if orig_bbox[2] <= orig_bbox[0] or orig_bbox[3] <= orig_bbox[1]:
            continue

        orig_poly = None
        raw_poly = det.get("polygon_norm")
        if isinstance(raw_poly, list) and len(raw_poly) >= 3:
            poly_px = []
            for pt in raw_poly:
                poly_px.append([float(pt[0]) * target_size, float(pt[1]) * target_size])
            orig_poly = unletterbox_polygon_norm(poly_px, meta)

        det_copy = dict(det)
        det_copy["bbox_norm"] = orig_bbox
        det_copy["polygon_norm"] = orig_poly
        mapped.append(det_copy)
    return mapped


def infer(
    preprocessed,
    mode="yolov8",
    conf_threshold=0.45,
    iou_threshold=NMS_IOU_THRESHOLD,
    excluded_classes=None,
):
    resolved_mode = validate_detector_mode(mode)
    letterboxed = preprocessed["letterboxed_bgr"]
    meta = preprocessed["meta"]

    if resolved_mode in ("orb_baseline", "sift_baseline"):
        raw_dets, infer_meta = run_baseline_detection(
            letterboxed,
            mode=resolved_mode,
            conf_threshold=float(conf_threshold),
        )
    elif resolved_mode == "gemini_vision":
        raw_dets, infer_meta = run_gemini_detection(
            letterboxed,
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
            excluded_classes=excluded_classes,
        )
    elif resolved_mode == "hybrid_gemini":
        raw_dets, infer_meta = run_hybrid_gemini_detection(
            letterboxed,
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
            excluded_classes=excluded_classes,
        )
    else:
        raw_dets, infer_meta = run_yolo_detection(
            letterboxed,
            mode=resolved_mode,
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
        )

    mapped_dets = map_detections_to_original_frame(raw_dets, meta)
    return mapped_dets, infer_meta


def postprocess(
    raw_detections,
    tracker_state=None,
    conf_threshold=0.45,
    iou_threshold=NMS_IOU_THRESHOLD,
    excluded_classes=None,
):
    filtered_raw = []
    for det in raw_detections:
        raw_cls = det.get("class_id")
        if not is_valid_food_class(raw_cls):
            continue
        filtered_raw.append(det)

    nms_dets = apply_class_nms(filtered_raw, iou_threshold=float(iou_threshold))
    nms_dets = apply_cross_class_nms(nms_dets, iou_threshold=CROSS_CLASS_NMS_IOU)

    active_tracker = tracker_state
    if active_tracker is None:
        active_tracker = create_tracker_state(
            window_size=1,
            min_hits=1,
            conf_threshold=float(conf_threshold),
            iou_threshold=float(iou_threshold),
        )

    dets, confirmed_inventory = update_temporal_tracker(
        active_tracker,
        nms_dets,
        conf_threshold=float(conf_threshold),
        excluded_classes=excluded_classes,
    )

    frame_counts = {}
    for det in dets:
        cls_id = det["class_id"]
        frame_counts[cls_id] = frame_counts.get(cls_id, 0) + 1

    confirmed_counts = {}
    for item in confirmed_inventory:
        cls_id = item["class_id"]
        confirmed_counts[cls_id] = int(item["count"])

    use_confirmed = int(active_tracker.get("min_hits", 1)) > 1
    counts_by_class = confirmed_counts if use_confirmed else frame_counts
    total_count = sum(counts_by_class.values())

    return {
        "detections": dets,
        "confirmed_inventory": confirmed_inventory,
        "counts_by_class": counts_by_class,
        "total_count": int(total_count),
        "frame_counts_by_class": frame_counts,
        "frame_total_count": int(sum(frame_counts.values())),
    }


def render(frame, postprocessed, source_name="frame", elapsed_ms=0.0, mode="yolov8"):
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    raw_detections = postprocessed.get("detections", [])
    detections = []
    for det in raw_detections:
        raw_cls = det.get("class_id")
        if not is_valid_food_class(raw_cls):
            continue
        detections.append(det)

    raw_confirmed = postprocessed.get("confirmed_inventory", [])
    confirmed_inventory = []
    for item in raw_confirmed:
        raw_cls = item.get("class_id")
        if not is_valid_food_class(raw_cls):
            continue
        confirmed_inventory.append(item)

    raw_counts = postprocessed.get("counts_by_class", {})
    counts_by_class = {}
    for cls_key, cnt_val in raw_counts.items():
        if is_valid_food_class(cls_key):
            norm_key = normalize_class_id(cls_key)
            counts_by_class[norm_key] = counts_by_class.get(norm_key, 0) + int(cnt_val)

    if "counts_by_class" in postprocessed:
        total_count = int(sum(counts_by_class.values()))
    else:
        total_count = len(detections)

    for det in detections:
        bbox = det.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])
        poly = det.get("polygon_norm")
        confirmed = bool(det.get("temporally_confirmed", False))
        color = (40, 200, 80) if confirmed else (0, 175, 255)

        if isinstance(poly, list) and len(poly) == 4:
            pts = []
            for p in poly:
                pts.append([int(round(float(p[0]) * w)), int(round(float(p[1]) * h))])
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], True, color, 2)
        else:
            x1 = max(0, min(w - 1, int(round(float(bbox[0]) * w))))
            y1 = max(0, min(h - 1, int(round(float(bbox[1]) * h))))
            x2 = max(x1 + 1, min(w, int(round(float(bbox[2]) * w))))
            y2 = max(y1 + 1, min(h, int(round(float(bbox[3]) * h))))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        x_lbl = max(2, int(round(float(bbox[0]) * w)))
        y_lbl = max(20, int(round(float(bbox[1]) * h)) - 6)
        disp = det.get("display_name") or get_display_name(det.get("class_id", "item"))
        label_txt = f"{disp} {float(det.get('confidence', 0.0)):.2f}"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(canvas, (x_lbl, max(0, y_lbl - th - 4)), (min(w, x_lbl + tw + 6), y_lbl + 4), color, -1)
        cv2.putText(canvas, label_txt, (x_lbl + 3, y_lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)

    inf_ms = max(0.01, float(elapsed_ms))
    fps = 1000.0 / inf_ms
    hud_text = f"{mode} | {inf_ms:.1f} ms ({fps:.1f} FPS) | items: {total_count}"
    cv2.rectangle(canvas, (0, 0), (min(w, 460), 26), (20, 20, 20), -1)
    cv2.putText(canvas, hud_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

    labels = []
    label_parts = []
    for det in detections:
        cls_id = normalize_class_id(det["class_id"])
        conf = float(det.get("confidence", 0.0))
        label_entry = {
            "class_id": cls_id,
            "class_idx": CLASS_INDEX_MAP.get(cls_id, -1),
            "display_name": det.get("display_name", get_display_name(cls_id)),
            "confidence": conf,
            "bbox_norm": det.get("bbox_norm", [0.0, 0.0, 0.0, 0.0]),
            "temporally_confirmed": bool(det.get("temporally_confirmed", False)),
        }
        if det.get("estimated_calories") is not None:
            label_entry["estimated_calories"] = int(det["estimated_calories"])
        else:
            label_entry["estimated_calories"] = get_class_calories(cls_id, 1)
        if det.get("crop_jpeg_b64") is not None:
            label_entry["crop_jpeg_b64"] = det["crop_jpeg_b64"]
        labels.append(label_entry)
        label_parts.append(f"{cls_id}:{conf:.2f}")

    csv_row = {
        "source": str(source_name),
        "mode": str(mode),
        "inference_ms": round(inf_ms, 2),
        "fps": round(fps, 2),
        "total_count": int(total_count),
        "labels_summary": "; ".join(label_parts) if label_parts else "none",
        "counts_json": json.dumps(counts_by_class, sort_keys=True),
    }

    return {
        "annotated_frame": canvas,
        "labels": labels,
        "counts": counts_by_class,
        "confirmed_inventory": confirmed_inventory,
        "csv_row": csv_row,
    }


def run_single_frame(
    frame,
    mode="yolov8",
    enable_clahe=True,
    conf_threshold=0.45,
    iou_threshold=NMS_IOU_THRESHOLD,
    tracker_state=None,
    source_name="frame",
    excluded_classes=None,
):
    t_start = time.perf_counter()
    prep = preprocess(frame, enable_clahe=enable_clahe, target_size=TARGET_IMG_SIZE)
    raw_dets, infer_meta = infer(
        prep,
        mode=mode,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        excluded_classes=excluded_classes,
    )
    post = postprocess(
        raw_dets,
        tracker_state=tracker_state,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        excluded_classes=excluded_classes,
    )
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    rendered = render(
        frame,
        post,
        source_name=source_name,
        elapsed_ms=elapsed_ms,
        mode=mode,
    )
    return {
        "preprocessed": prep,
        "raw_detections": raw_dets,
        "infer_meta": infer_meta,
        "postprocessed": post,
        "rendered": rendered,
        "elapsed_ms": elapsed_ms,
    }


def write_yolo_and_json_labels(labels_dir, stem, labels):
    os.makedirs(labels_dir, exist_ok=True)
    txt_path = os.path.join(labels_dir, f"{stem}.txt")
    json_path = os.path.join(labels_dir, f"{stem}.json")

    lines = []
    for item in labels:
        cls_idx = int(item.get("class_idx", -1))
        if cls_idx < 0:
            continue
        x1, y1, x2, y2 = item["bbox_norm"]
        xc = (float(x1) + float(x2)) / 2.0
        yc = (float(y1) + float(y2)) / 2.0
        bw = max(0.0, float(x2) - float(x1))
        bh = max(0.0, float(y2) - float(y1))
        lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {float(item['confidence']):.4f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def write_csv_rows(csv_path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "source",
        "mode",
        "inference_ms",
        "fps",
        "total_count",
        "labels_summary",
        "counts_json",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_input_path(raw_input):
    cand = Path(raw_input)
    if cand.exists():
        return cand
    normalized = str(raw_input).replace("\\", "/").rstrip("/")
    if normalized == "data/test":
        fallback = Path(PROJECT_ROOT) / "data" / "dataset_yolo" / "test" / "images"
        if fallback.exists():
            return fallback
    if normalized == "demo.mp4":
        fallback = Path(PROJECT_ROOT) / "data" / "raw_home" / "demo_pantry_pan.mp4"
        if fallback.exists():
            return fallback
    rel_cand = Path(PROJECT_ROOT) / raw_input
    if rel_cand.exists():
        return rel_cand
    raise FileNotFoundError(f"Input path not found: {raw_input}")


def process_image_paths(
    image_paths,
    output_arg,
    mode="yolov8",
    enable_clahe=True,
    conf_threshold=0.45,
    iou_threshold=NMS_IOU_THRESHOLD,
    generate_recipes=False,
):
    out_path = Path(output_arg)
    single_file_target = len(image_paths) == 1 and out_path.suffix.lower() in IMAGE_EXTENSIONS
    if single_file_target:
        out_dir = out_path.parent if str(out_path.parent) else Path(".")
        annotated_dir = out_dir
    else:
        out_dir = out_path
        annotated_dir = out_dir / "annotated"

    labels_dir = out_dir / "labels"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    warmup_img = cv2.imread(str(image_paths[0]))
    if warmup_img is not None and warmup_img.size > 0:
        run_single_frame(
            warmup_img,
            mode=mode,
            enable_clahe=enable_clahe,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            source_name="warmup",
        )

    csv_rows = []
    latencies = []
    aggregated_counts = {}
    aggregated_conf = {}

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None or bgr.size == 0:
            continue
        res = run_single_frame(
            bgr,
            mode=mode,
            enable_clahe=enable_clahe,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            source_name=img_path.name,
        )
        rendered = res["rendered"]
        latencies.append(res["elapsed_ms"])
        csv_rows.append(rendered["csv_row"])

        if single_file_target:
            save_img_path = out_path
        else:
            save_img_path = annotated_dir / img_path.name
        cv2.imwrite(str(save_img_path), rendered["annotated_frame"])
        write_yolo_and_json_labels(str(labels_dir), img_path.stem, rendered["labels"])

        for cls_id, count_val in rendered["counts"].items():
            aggregated_counts[cls_id] = aggregated_counts.get(cls_id, 0) + int(count_val)
        for lbl in rendered["labels"]:
            cid = lbl["class_id"]
            cval = float(lbl["confidence"])
            if cval > aggregated_conf.get(cid, 0.0):
                aggregated_conf[cid] = cval

    csv_path = out_dir / "predictions.csv"
    write_csv_rows(str(csv_path), csv_rows)

    mean_ms = float(np.mean(latencies)) if latencies else 0.0
    fps = (1000.0 / mean_ms) if mean_ms > 0.0 else 0.0
    summary = {
        "input_count": len(csv_rows),
        "mode": validate_detector_mode(mode),
        "enable_clahe": bool(enable_clahe),
        "conf_threshold": float(conf_threshold),
        "iou_threshold": float(iou_threshold),
        "mean_inference_ms": round(mean_ms, 2),
        "fps": round(fps, 2),
        "total_detections": int(sum(aggregated_counts.values())),
        "counts_by_class": aggregated_counts,
        "output_dir": str(out_dir),
        "predictions_csv": str(csv_path),
    }

    if generate_recipes:
        inv_items = []
        for cls_id in sorted(aggregated_counts.keys()):
            inv_items.append({
                "class_id": cls_id,
                "display_name": get_display_name(cls_id),
                "count": aggregated_counts[cls_id],
                "max_conf": round(aggregated_conf.get(cls_id, 0.5), 2),
            })
        recipes_payload = generate_structured_recipes(inv_items)
        recipes_path = out_dir / "recipes.json"
        with open(recipes_path, "w", encoding="utf-8") as f:
            json.dump(recipes_payload, f, indent=2)
        summary["recipes_json"] = str(recipes_path)

    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def process_video_file(
    video_path,
    output_arg,
    mode="yolov8",
    enable_clahe=True,
    conf_threshold=0.45,
    iou_threshold=NMS_IOU_THRESHOLD,
    skip_frames=1,
    generate_recipes=False,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps is None or native_fps <= 0.0 or np.isnan(native_fps):
        native_fps = 15.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    out_path = Path(output_arg)
    if out_path.suffix.lower() in VIDEO_EXTENSIONS:
        video_out_path = out_path
        out_dir = out_path.parent if str(out_path.parent) else Path(".")
        csv_path = out_dir / f"{out_path.stem}_predictions.csv"
        summary_path = out_dir / f"{out_path.stem}_summary.json"
    else:
        out_dir = out_path
        video_out_path = out_dir / f"{Path(video_path).stem}_annotated.mp4"
        csv_path = out_dir / "predictions.csv"
        summary_path = out_dir / "run_summary.json"

    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out_path), fourcc, float(native_fps), (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open VideoWriter: {video_out_path}")

    tracker = create_tracker_state(
        window_size=TEMPORAL_WINDOW_SIZE,
        min_hits=TEMPORAL_MIN_HITS,
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
    )

    step = max(1, int(skip_frames))
    frame_idx = 0
    processed_count = 0
    csv_rows = []
    latencies = []
    peak_confirmed = {}
    last_rendered_frame = None
    warmed_up = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if not warmed_up:
            warmup_tracker = create_tracker_state(
                window_size=1,
                min_hits=1,
                conf_threshold=float(conf_threshold),
                iou_threshold=float(iou_threshold),
            )
            run_single_frame(
                frame,
                mode=mode,
                enable_clahe=enable_clahe,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                tracker_state=warmup_tracker,
                source_name="warmup",
            )
            warmed_up = True

        if frame_idx % step == 0 or last_rendered_frame is None:
            res = run_single_frame(
                frame,
                mode=mode,
                enable_clahe=enable_clahe,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                tracker_state=tracker,
                source_name=f"frame_{frame_idx:04d}",
            )
            rendered = res["rendered"]
            last_rendered_frame = rendered["annotated_frame"]
            latencies.append(res["elapsed_ms"])
            csv_rows.append(rendered["csv_row"])
            processed_count += 1

            for item in rendered["confirmed_inventory"]:
                cid = item["class_id"]
                cnt = int(item["count"])
                mconf = float(item["max_conf"])
                if cid not in peak_confirmed:
                    peak_confirmed[cid] = {
                        "class_id": cid,
                        "display_name": item["display_name"],
                        "count": cnt,
                        "max_conf": mconf,
                    }
                else:
                    if cnt > peak_confirmed[cid]["count"]:
                        peak_confirmed[cid]["count"] = cnt
                    if mconf > peak_confirmed[cid]["max_conf"]:
                        peak_confirmed[cid]["max_conf"] = mconf

        writer.write(last_rendered_frame)
        frame_idx += 1

    cap.release()
    writer.release()

    write_csv_rows(str(csv_path), csv_rows)

    confirmed_list = []
    counts_by_class = {}
    for cid in sorted(peak_confirmed.keys()):
        confirmed_list.append(peak_confirmed[cid])
        counts_by_class[cid] = int(peak_confirmed[cid]["count"])

    mean_ms = float(np.mean(latencies)) if latencies else 0.0
    fps = (1000.0 / mean_ms) if mean_ms > 0.0 else 0.0

    summary = {
        "video_path": str(video_path),
        "annotated_video": str(video_out_path),
        "predictions_csv": str(csv_path),
        "total_video_frames": int(frame_idx),
        "frames_processed": int(processed_count),
        "mode": validate_detector_mode(mode),
        "enable_clahe": bool(enable_clahe),
        "conf_threshold": float(conf_threshold),
        "temporal_window": int(TEMPORAL_WINDOW_SIZE),
        "temporal_min_hits": int(TEMPORAL_MIN_HITS),
        "mean_inference_ms": round(mean_ms, 2),
        "fps": round(fps, 2),
        "confirmed_inventory": confirmed_list,
        "counts_by_class": counts_by_class,
        "total_confirmed_items": int(sum(counts_by_class.values())),
    }

    if generate_recipes:
        recipes_payload = generate_structured_recipes(confirmed_list)
        recipes_path = out_dir / "recipes.json"
        with open(recipes_path, "w", encoding="utf-8") as f:
            json.dump(recipes_payload, f, indent=2)
        summary["recipes_json"] = str(recipes_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def process_webcam(
    camera_index,
    output_arg="outputs/webcam_run",
    mode="yolov8",
    enable_clahe=True,
    conf_threshold=0.45,
    iou_threshold=NMS_IOU_THRESHOLD,
    max_frames=300,
    no_display=False,
):
    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open webcam device index {camera_index}.")

    out_dir = Path(output_arg)
    os.makedirs(out_dir, exist_ok=True)
    tracker = create_tracker_state(
        window_size=TEMPORAL_WINDOW_SIZE,
        min_hits=TEMPORAL_MIN_HITS,
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
    )

    csv_rows = []
    latencies = []
    peak_confirmed = {}
    writer = None
    video_out_path = out_dir / f"webcam_{int(camera_index)}_annotated.mp4"
    frame_idx = 0

    try:
        while cap.isOpened():
            if int(max_frames) > 0 and frame_idx >= int(max_frames):
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_out_path), fourcc, 15.0, (fw, fh))

            res = run_single_frame(
                frame,
                mode=mode,
                enable_clahe=enable_clahe,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                tracker_state=tracker,
                source_name=f"webcam_{int(camera_index)}_{frame_idx:04d}",
            )
            rendered = res["rendered"]
            latencies.append(res["elapsed_ms"])
            csv_rows.append(rendered["csv_row"])

            for item in rendered["confirmed_inventory"]:
                cid = item["class_id"]
                cnt = int(item["count"])
                mconf = float(item["max_conf"])
                if cid not in peak_confirmed:
                    peak_confirmed[cid] = {
                        "class_id": cid,
                        "display_name": item["display_name"],
                        "count": cnt,
                        "max_conf": mconf,
                    }
                else:
                    if cnt > peak_confirmed[cid]["count"]:
                        peak_confirmed[cid]["count"] = cnt
                    if mconf > peak_confirmed[cid]["max_conf"]:
                        peak_confirmed[cid]["max_conf"] = mconf

            if writer is not None and writer.isOpened():
                writer.write(rendered["annotated_frame"])

            if not no_display:
                try:
                    cv2.imshow("Stock2Stove Pipeline", rendered["annotated_frame"])
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                except cv2.error:
                    no_display = True

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    csv_path = out_dir / "predictions.csv"
    write_csv_rows(str(csv_path), csv_rows)

    mean_ms = float(np.mean(latencies)) if latencies else 0.0
    fps = (1000.0 / mean_ms) if mean_ms > 0.0 else 0.0
    summary = {
        "webcam_index": int(camera_index),
        "frames_processed": int(frame_idx),
        "annotated_video": str(video_out_path),
        "predictions_csv": str(csv_path),
        "mean_inference_ms": round(mean_ms, 2),
        "fps": round(fps, 2),
        "confirmed_inventory": list(peak_confirmed.values()),
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/run_latest/")
    parser.add_argument("--webcam", type=int, default=None)
    parser.add_argument("--mode", type=str, default="yolov8")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=NMS_IOU_THRESHOLD)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--generate-recipes", action="store_true")
    args = parser.parse_args()

    enable_clahe = not bool(args.no_clahe)

    if args.webcam is not None:
        summary = process_webcam(
            camera_index=args.webcam,
            output_arg=args.output,
            mode=args.mode,
            enable_clahe=enable_clahe,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_frames=args.max_frames,
            no_display=args.no_display,
        )
        print(json.dumps(summary, indent=2))
        return

    if not args.input:
        raise ValueError("Provide either --input <path> or --webcam <index>.")

    input_path = resolve_input_path(args.input)

    if input_path.is_dir():
        image_paths = []
        for entry in sorted(input_path.iterdir()):
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(entry)
        if not image_paths:
            raise FileNotFoundError(f"No images found in directory: {input_path}")
        summary = process_image_paths(
            image_paths,
            output_arg=args.output,
            mode=args.mode,
            enable_clahe=enable_clahe,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            generate_recipes=args.generate_recipes,
        )
        print(json.dumps(summary, indent=2))
        return

    suffix = input_path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        summary = process_video_file(
            input_path,
            output_arg=args.output,
            mode=args.mode,
            enable_clahe=enable_clahe,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            skip_frames=args.skip_frames,
            generate_recipes=args.generate_recipes,
        )
        print(json.dumps(summary, indent=2))
        return

    if suffix in IMAGE_EXTENSIONS:
        summary = process_image_paths(
            [input_path],
            output_arg=args.output,
            mode=args.mode,
            enable_clahe=enable_clahe,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            generate_recipes=args.generate_recipes,
        )
        print(json.dumps(summary, indent=2))
        return

    raise ValueError(f"Unsupported input format: {input_path}")


if __name__ == "__main__":
    main()
