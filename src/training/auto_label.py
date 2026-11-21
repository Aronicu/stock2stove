import argparse
import logging
import os
import shutil
import sys
import cv2
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cv import yolo_detector
from src.cv.gemini_detector import run_gemini_detection
from src.cv.yolo_detector import check_ultralytics_available, run_yolo_detection
from src.server.config import GEMINI_AUTOLABEL_MODEL, UNIFIED_CLASSES, normalize_class_id

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
logger = logging.getLogger(__name__)


def class_id_to_index(class_id):
    if class_id is None:
        return -1
    cleaned = str(class_id).strip().lower()
    if not cleaned:
        return -1
    for entry in UNIFIED_CLASSES:
        if entry["class_id"] == cleaned:
            return int(entry["id"])
    canonical = normalize_class_id(cleaned)
    for entry in UNIFIED_CLASSES:
        if entry["class_id"] == canonical:
            return int(entry["id"])
    return -1


def bbox_xyxy_to_yolo(bbox_norm):
    if not bbox_norm or len(bbox_norm) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = max(0.0, min(1.0, float(bbox_norm[0])))
    y1 = max(0.0, min(1.0, float(bbox_norm[1])))
    x2 = max(0.0, min(1.0, float(bbox_norm[2])))
    y2 = max(0.0, min(1.0, float(bbox_norm[3])))
    if x2 < x1:
        temp = x1
        x1 = x2
        x2 = temp
    if y2 < y1:
        temp = y1
        y1 = y2
        y2 = temp
    w = max(0.0, min(1.0, x2 - x1))
    h = max(0.0, min(1.0, y2 - y1))
    xc = max(0.0, min(1.0, (x1 + x2) / 2.0))
    yc = max(0.0, min(1.0, (y1 + y2) / 2.0))
    return [round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)]


def auto_label_image_with_meta(
    bgr_image,
    conf_threshold=0.45,
    model_path=None,
    backend="yolo_world",
    gemini_timeout_sec=25.0,
):
    backend_name = str(backend).strip().lower()
    if bgr_image is None or bgr_image.size == 0:
        return [], {
            "backend": backend_name,
            "live_api": False,
            "skipped_fallback": False,
        }

    if backend_name == "gemini":
        detections, det_meta = run_gemini_detection(
            bgr_image,
            conf_threshold=float(conf_threshold),
            model_name=GEMINI_AUTOLABEL_MODEL,
            timeout_sec=float(gemini_timeout_sec),
            use_cache=False,
        )
        meta = dict(det_meta) if isinstance(det_meta, dict) else {}
        is_live = bool(meta.get("live_api", False))
        meta["live_api"] = is_live
        if not is_live:
            meta["skipped_fallback"] = True
            logger.warning(
                "Skipping Gemini auto-label frame: live_api=False (backend=%s, last_error=%s)",
                meta.get("backend"),
                meta.get("last_error"),
            )
            return [], meta
        meta["skipped_fallback"] = False
    else:
        if model_path and os.path.exists(str(model_path)) and check_ultralytics_available():
            resolved_model = os.path.abspath(str(model_path))
            if yolo_detector.MODEL_REGISTRY.get("yolo_world_path") != resolved_model:
                try:
                    from ultralytics import YOLO

                    loaded_model = YOLO(resolved_model)
                    if "world" in os.path.basename(resolved_model).lower():
                        prompts, _ = yolo_detector.build_world_prompts()
                        loaded_model.set_classes(prompts)
                    yolo_detector.MODEL_REGISTRY["yolo_world_model"] = loaded_model
                    yolo_detector.MODEL_REGISTRY["yolo_world_path"] = resolved_model
                except Exception:
                    pass

        detections, det_meta = run_yolo_detection(
            bgr_image,
            mode="yolo_world",
            conf_threshold=float(conf_threshold),
        )
        meta = dict(det_meta) if isinstance(det_meta, dict) else {}
        meta["skipped_fallback"] = False

    annotations = []
    for det in detections:
        cls_idx = class_id_to_index(det.get("class_id", ""))
        if cls_idx < 0:
            continue
        canonical_cls = UNIFIED_CLASSES[cls_idx]["class_id"]
        yolo_box = bbox_xyxy_to_yolo(det.get("bbox_norm", [0.0, 0.0, 0.0, 0.0]))
        if yolo_box[2] <= 0.0 or yolo_box[3] <= 0.0:
            continue
        annotations.append({
            "class_idx": int(cls_idx),
            "class_id": canonical_cls,
            "confidence": float(det.get("confidence", 0.0)),
            "xc": float(yolo_box[0]),
            "yc": float(yolo_box[1]),
            "w": float(yolo_box[2]),
            "h": float(yolo_box[3]),
        })
    return annotations, meta


def auto_label_image(
    bgr_image,
    conf_threshold=0.45,
    model_path=None,
    backend="yolo_world",
    gemini_timeout_sec=25.0,
):
    annotations, _ = auto_label_image_with_meta(
        bgr_image,
        conf_threshold=conf_threshold,
        model_path=model_path,
        backend=backend,
        gemini_timeout_sec=gemini_timeout_sec,
    )
    return annotations


def auto_label_directory(
    input_dir,
    output_dir,
    conf_threshold=0.45,
    model_path=None,
    backend="yolo_world",
    gemini_timeout_sec=25.0,
):
    input_path = str(input_dir)
    output_path = str(output_dir)
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    images_out_dir = os.path.join(output_path, "images")
    labels_out_dir = os.path.join(output_path, "labels")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    class_counts = {}
    names_map = {}
    for entry in UNIFIED_CLASSES:
        class_counts[entry["class_id"]] = 0
        names_map[int(entry["id"])] = entry["class_id"]

    image_files = []
    for entry_name in sorted(os.listdir(input_path)):
        full_path = os.path.join(input_path, entry_name)
        if not os.path.isfile(full_path):
            continue
        if entry_name.lower().endswith(IMAGE_EXTENSIONS):
            image_files.append(full_path)

    images_processed = 0
    images_with_labels = 0
    skipped_fallback_frames = 0
    total_detections = 0

    for img_path in image_files:
        bgr = cv2.imread(img_path)
        if bgr is None or bgr.size == 0:
            continue
        annotations, ann_meta = auto_label_image_with_meta(
            bgr,
            conf_threshold=conf_threshold,
            model_path=model_path,
            backend=backend,
            gemini_timeout_sec=gemini_timeout_sec,
        )
        if ann_meta.get("skipped_fallback"):
            skipped_fallback_frames += 1
            continue

        base_name = os.path.basename(img_path)
        stem = os.path.splitext(base_name)[0]
        dst_img = os.path.join(images_out_dir, base_name)
        if os.path.abspath(img_path) != os.path.abspath(dst_img):
            shutil.copy2(img_path, dst_img)

        label_lines = []
        for ann in annotations:
            label_lines.append(
                f"{ann['class_idx']} {ann['xc']:.6f} {ann['yc']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n"
            )
            cls_key = ann["class_id"]
            class_counts[cls_key] = class_counts.get(cls_key, 0) + 1
            total_detections += 1

        dst_label = os.path.join(labels_out_dir, f"{stem}.txt")
        with open(dst_label, "w", encoding="utf-8") as f:
            f.writelines(label_lines)

        images_processed += 1
        if len(label_lines) > 0:
            images_with_labels += 1

    yaml_path = os.path.join(output_path, "data.yaml")
    data_yaml = {
        "path": os.path.abspath(output_path),
        "train": "images",
        "val": "images",
        "nc": len(UNIFIED_CLASSES),
        "names": names_map,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    return {
        "input_dir": input_path,
        "output_dir": output_path,
        "images_processed": images_processed,
        "images_with_labels": images_with_labels,
        "skipped_fallback_frames": skipped_fallback_frames,
        "total_detections": total_detections,
        "class_counts": class_counts,
        "data_yaml": yaml_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/samples")
    parser.add_argument("--output", type=str, default="data/auto_labeled")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--model", type=str, default="models/yolov8s-worldv2.pt")
    parser.add_argument("--backend", type=str, choices=["yolo_world", "gemini"], default="yolo_world")
    parser.add_argument("--gemini-timeout", type=float, default=25.0)
    args = parser.parse_args()

    summary = auto_label_directory(
        args.input,
        args.output,
        conf_threshold=args.conf,
        model_path=args.model,
        backend=args.backend,
        gemini_timeout_sec=args.gemini_timeout,
    )
    print(f"Input directory: {summary['input_dir']}")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Images processed: {summary['images_processed']}")
    print(f"Images with labels: {summary['images_with_labels']}")
    print(f"Skipped fallback frames: {summary['skipped_fallback_frames']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Class counts: {summary['class_counts']}")
    print(f"Data YAML: {summary['data_yaml']}")


if __name__ == "__main__":
    main()
