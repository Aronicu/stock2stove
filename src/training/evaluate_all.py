import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cv.baseline_detector import compute_box_iou, load_templates_for_mode, run_baseline_detection
from src.cv.gemini_detector import run_gemini_detection, run_hybrid_gemini_detection
from src.cv.preprocess import preprocess_frame
from src.cv.yolo_detector import MODEL_REGISTRY, load_ultralytics_detector, run_yolo_detection
from src.server.config import UNIFIED_CLASSES, normalize_class_id

logger = logging.getLogger(__name__)


def extract_bbox_xyxy(box_item):
    if isinstance(box_item, (list, tuple)) and len(box_item) >= 4:
        return [float(box_item[0]), float(box_item[1]), float(box_item[2]), float(box_item[3])]
    if isinstance(box_item, dict):
        coords = box_item.get("bbox_norm")
        if coords is None:
            coords = box_item.get("bbox")
        if isinstance(coords, (list, tuple)) and len(coords) >= 4:
            return [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]
    raise ValueError("Box entry must contain bbox_norm, bbox, or 4 coordinates.")


def resolve_test_paths(data_yaml_path):
    yaml_path = Path(data_yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    yaml_dir = yaml_path.parent

    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    raw_root = meta.get("path")
    if raw_root:
        root_cand = Path(str(raw_root))
        if not root_cand.is_absolute():
            rel_cand = (yaml_dir / root_cand).resolve()
            dataset_root = rel_cand if rel_cand.exists() else root_cand.resolve()
        else:
            dataset_root = root_cand
        if not (dataset_root / "test").exists() and (yaml_dir / "test").exists():
            dataset_root = yaml_dir
    else:
        dataset_root = yaml_dir

    test_entry = meta.get("test", "test/images")
    test_p = Path(str(test_entry))
    test_images_dir = test_p if test_p.is_absolute() else (dataset_root / test_p)
    if not test_images_dir.exists() and (yaml_dir / "test" / "images").exists():
        test_images_dir = yaml_dir / "test" / "images"

    if test_images_dir.name == "images":
        test_labels_dir = test_images_dir.parent / "labels"
    else:
        test_labels_dir = dataset_root / "test" / "labels"

    yaml_names_raw = meta.get("names", {})
    yaml_names = {}
    if isinstance(yaml_names_raw, list):
        for idx, name in enumerate(yaml_names_raw):
            yaml_names[int(idx)] = str(name)
    elif isinstance(yaml_names_raw, dict):
        for k, v in yaml_names_raw.items():
            yaml_names[int(k)] = str(v)

    return test_images_dir, test_labels_dir, yaml_names


def load_yolo_test_split(data_yaml_path):
    test_images_dir, test_labels_dir, yaml_names = resolve_test_paths(data_yaml_path)
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

    idx_to_class_id = {}
    for entry in UNIFIED_CLASSES:
        idx_to_class_id[int(entry["id"])] = entry["class_id"]

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    for p in sorted(test_images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in image_exts:
            image_paths.append(p)

    samples = []
    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None or bgr.size == 0:
            continue
        h, w = bgr.shape[:2]

        gt_boxes = []
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_idx = int(float(parts[0]))
                    if cls_idx in idx_to_class_id:
                        cls_id = idx_to_class_id[cls_idx]
                    elif cls_idx in yaml_names:
                        cls_id = normalize_class_id(yaml_names[cls_idx])
                    else:
                        continue

                    if len(parts) == 5:
                        xc = float(parts[1])
                        yc = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])
                        x1 = max(0.0, min(1.0, xc - bw / 2.0))
                        y1 = max(0.0, min(1.0, yc - bh / 2.0))
                        x2 = max(0.0, min(1.0, xc + bw / 2.0))
                        y2 = max(0.0, min(1.0, yc + bh / 2.0))
                    else:
                        coords = [float(val) for val in parts[1:]]
                        xs = [coords[k] for k in range(0, len(coords) - 1, 2)]
                        ys = [coords[k + 1] for k in range(0, len(coords) - 1, 2)]
                        if not xs or not ys:
                            continue
                        x1 = max(0.0, min(1.0, min(xs)))
                        y1 = max(0.0, min(1.0, min(ys)))
                        x2 = max(0.0, min(1.0, max(xs)))
                        y2 = max(0.0, min(1.0, max(ys)))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    gt_boxes.append({
                        "class_idx": int(cls_idx),
                        "class_id": cls_id,
                        "bbox_norm": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
                    })

        samples.append({
            "image_path": str(img_path),
            "filename": img_path.name,
            "image": bgr,
            "width": int(w),
            "height": int(h),
            "gt_boxes": gt_boxes,
        })

    return samples


def simulate_low_light_shadow(bgr_image):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")

    h, w = bgr_image.shape[:2]
    img_float = np.clip(bgr_image.astype(np.float32) / 255.0, 0.0, 1.0)
    darkened = np.power(img_float, 2.4) * 0.32

    y_grad = np.linspace(0.55, 1.0, max(1, h), dtype=np.float32)[:, None]
    x_grad = np.linspace(0.75, 1.0, max(1, w), dtype=np.float32)[None, :]
    shadow_mask = (0.65 * y_grad + 0.35 * x_grad)[:, :, None]

    shadowed = np.clip(darkened * shadow_mask * 255.0, 0.0, 255.0).astype(np.uint8)
    return shadowed


def simulate_partial_occlusion(bgr_image, gt_boxes, occlusion_ratio=0.35):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")

    occluded = bgr_image.copy()
    h, w = occluded.shape[:2]
    ratio = max(0.05, min(0.90, float(occlusion_ratio)))
    palette = [
        (48, 52, 58),
        (64, 60, 56),
        (54, 62, 68),
        (58, 52, 48),
    ]

    for idx, box_entry in enumerate(gt_boxes):
        x1, y1, x2, y2 = extract_bbox_xyxy(box_entry)
        bx1 = max(0, min(w - 1, int(round(x1 * float(w)))))
        by1 = max(0, min(h - 1, int(round(y1 * float(h)))))
        bx2 = max(bx1 + 1, min(w, int(round(x2 * float(w)))))
        by2 = max(by1 + 1, min(h, int(round(y2 * float(h)))))

        bw = bx2 - bx1
        bh = by2 - by1
        target_area = ratio * float(bw * bh)
        width_frac = min(0.85, math.sqrt(ratio * 1.4))
        occ_w = max(1, min(bw, int(round(float(bw) * width_frac))))
        occ_h = max(1, min(bh, int(round(target_area / float(occ_w)))))

        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        ox1 = max(bx1, min(bx2 - occ_w, cx - (occ_w // 2)))
        oy1 = max(by1, min(by2 - occ_h, cy - (occ_h // 2)))
        ox2 = min(bx2, ox1 + occ_w)
        oy2 = min(by2, oy1 + occ_h)

        occluded[oy1:oy2, ox1:ox2] = palette[idx % len(palette)]

    occluded = cv2.GaussianBlur(occluded, (5, 5), 0)
    return occluded


def compute_average_precision(tp_flags, total_gt):
    if total_gt <= 0 or not tp_flags:
        return 0.0

    tp_arr = np.array(tp_flags, dtype=np.float64)
    fp_arr = 1.0 - tp_arr

    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)

    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i + 1] > mpre[i]:
            mpre[i] = mpre[i + 1]

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += float((mrec[i] - mrec[i - 1]) * mpre[i])
    return round(float(ap), 6)


def evaluate_predictions_against_gt(pred_boxes, gt_boxes, iou_threshold=0.50):
    gt_buckets = {}
    gt_counts_by_class = {}
    total_gt = 0

    for gt in gt_boxes:
        cls_id = normalize_class_id(gt.get("class_id", ""))
        img_id = gt.get("image_id", gt.get("filename", "__single__"))
        bbox = extract_bbox_xyxy(gt)
        key = (img_id, cls_id)
        if key not in gt_buckets:
            gt_buckets[key] = []
        gt_buckets[key].append({
            "bbox_norm": bbox,
            "matched": False,
        })
        gt_counts_by_class[cls_id] = gt_counts_by_class.get(cls_id, 0) + 1
        total_gt += 1

    indexed_preds = []
    for idx, pred in enumerate(pred_boxes):
        conf = float(pred.get("confidence", 0.0))
        indexed_preds.append((conf, -idx, pred))
    indexed_preds.sort(key=lambda item: (item[0], item[1]), reverse=True)

    tp = 0
    fp = 0
    class_tp_records = {}

    for conf, _, pred in indexed_preds:
        cls_id = normalize_class_id(pred.get("class_id", ""))
        img_id = pred.get("image_id", pred.get("filename", "__single__"))
        pbox = extract_bbox_xyxy(pred)
        key = (img_id, cls_id)

        if cls_id not in class_tp_records:
            class_tp_records[cls_id] = []

        best_iou = 0.0
        best_gt_idx = -1
        candidates = gt_buckets.get(key, [])
        for g_idx, g_entry in enumerate(candidates):
            if g_entry["matched"]:
                continue
            iou_val = compute_box_iou(pbox, g_entry["bbox_norm"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = g_idx

        if best_gt_idx >= 0 and best_iou >= float(iou_threshold):
            candidates[best_gt_idx]["matched"] = True
            tp += 1
            class_tp_records[cls_id].append(1)
        else:
            fp += 1
            class_tp_records[cls_id].append(0)

    fn = max(0, total_gt - tp)

    average_precisions_per_class = {}
    for cls_id in sorted(gt_counts_by_class.keys()):
        cls_total_gt = gt_counts_by_class[cls_id]
        flags = class_tp_records.get(cls_id, [])
        average_precisions_per_class[cls_id] = compute_average_precision(flags, cls_total_gt)

    return int(tp), int(fp), int(fn), average_precisions_per_class


def compute_map50_95(pred_boxes, gt_boxes):
    thresholds = [0.50 + 0.05 * i for i in range(10)]
    ap_means = []
    for thr in thresholds:
        _, _, _, ap_map = evaluate_predictions_against_gt(pred_boxes, gt_boxes, iou_threshold=thr)
        if ap_map:
            ap_means.append(sum(ap_map.values()) / float(len(ap_map)))
        else:
            ap_means.append(0.0)
    if not ap_means:
        return 0.0
    return round(sum(ap_means) / float(len(ap_means)), 4)


def compute_per_class_pr(pred_boxes, gt_boxes, iou_threshold=0.50):
    gt_buckets = {}
    gt_counts_by_class = {}

    for gt in gt_boxes:
        cls_id = normalize_class_id(gt.get("class_id", ""))
        img_id = gt.get("image_id", gt.get("filename", "__single__"))
        bbox = extract_bbox_xyxy(gt)
        key = (img_id, cls_id)
        if key not in gt_buckets:
            gt_buckets[key] = []
        gt_buckets[key].append({
            "bbox_norm": bbox,
            "matched": False,
        })
        gt_counts_by_class[cls_id] = gt_counts_by_class.get(cls_id, 0) + 1

    indexed_preds = []
    for idx, pred in enumerate(pred_boxes):
        conf = float(pred.get("confidence", 0.0))
        indexed_preds.append((conf, -idx, pred))
    indexed_preds.sort(key=lambda item: (item[0], item[1]), reverse=True)

    tp_by_class = {}
    fp_by_class = {}

    for _, _, pred in indexed_preds:
        cls_id = normalize_class_id(pred.get("class_id", ""))
        img_id = pred.get("image_id", pred.get("filename", "__single__"))
        pbox = extract_bbox_xyxy(pred)
        key = (img_id, cls_id)

        best_iou = 0.0
        best_gt_idx = -1
        candidates = gt_buckets.get(key, [])
        for g_idx, g_entry in enumerate(candidates):
            if g_entry["matched"]:
                continue
            iou_val = compute_box_iou(pbox, g_entry["bbox_norm"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = g_idx

        if best_gt_idx >= 0 and best_iou >= float(iou_threshold):
            candidates[best_gt_idx]["matched"] = True
            tp_by_class[cls_id] = tp_by_class.get(cls_id, 0) + 1
        else:
            fp_by_class[cls_id] = fp_by_class.get(cls_id, 0) + 1

    all_classes = sorted(set(list(gt_counts_by_class.keys()) + list(tp_by_class.keys()) + list(fp_by_class.keys())))
    precision_per_class = {}
    recall_per_class = {}
    for cls_id in all_classes:
        cls_tp = tp_by_class.get(cls_id, 0)
        cls_fp = fp_by_class.get(cls_id, 0)
        cls_gt = gt_counts_by_class.get(cls_id, 0)
        precision_per_class[cls_id] = round(float(cls_tp) / float(cls_tp + cls_fp), 4) if (cls_tp + cls_fp) > 0 else 0.0
        recall_per_class[cls_id] = round(float(cls_tp) / float(cls_gt), 4) if cls_gt > 0 else 0.0

    return precision_per_class, recall_per_class


def evaluate_false_meat_rate(data_yaml_path, conf_threshold=0.50, include_gemini=False):
    samples = load_yolo_test_split(data_yaml_path)
    checked_names = set()
    eval_items = []

    for sample in samples:
        has_meat_gt = False
        for gt in sample["gt_boxes"]:
            if gt["class_id"] == "canned_tuna_meat":
                has_meat_gt = True
                break
        if not has_meat_gt:
            fname = os.path.basename(sample["filename"])
            if fname not in checked_names:
                checked_names.add(fname)
                eval_items.append((sample["filename"], sample["image"]))

    samples_dir = Path(PROJECT_ROOT) / "data" / "samples"
    if samples_dir.exists():
        for img_path in sorted(samples_dir.glob("pantry_images_cv_*.jpg")):
            fname = os.path.basename(str(img_path))
            if fname in checked_names:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is not None and bgr.size > 0:
                checked_names.add(fname)
                eval_items.append((img_path.name, bgr))

    pantry_neg_dir = Path(data_yaml_path).resolve().parent / "pantry_negatives"
    if pantry_neg_dir.exists():
        for img_path in sorted(pantry_neg_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue
            fname = os.path.basename(str(img_path))
            if fname in checked_names:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is not None and bgr.size > 0:
                checked_names.add(fname)
                eval_items.append((img_path.name, bgr))

    has_gemini_key = bool((os.getenv("GEMINI_API_KEY") or "").strip())
    effective_include_gemini = bool(include_gemini) and has_gemini_key
    if include_gemini and not has_gemini_key:
        notice = "GEMINI_API_KEY is not set; skipping gemini_vision and hybrid_gemini false-meat evaluation."
        logger.warning(notice)
        print(notice)

    prepped_items = []
    for name, bgr in eval_items:
        prep = preprocess_frame(bgr, enable_clahe=True, target_size=640)
        prepped_items.append((name, prep["letterboxed_bgr"]))

    eval_detectors = ["yolov8n_finetuned"]
    if effective_include_gemini:
        eval_detectors.extend(["gemini_vision", "hybrid_gemini"])

    total = len(prepped_items)
    false_meat_by_detector = {}

    for det_name in eval_detectors:
        false_meat_images = []
        for name, letterboxed in prepped_items:
            if det_name == "gemini_vision":
                dets, _ = run_gemini_detection(
                    letterboxed,
                    conf_threshold=float(conf_threshold),
                    use_cache=False,
                )
            elif det_name == "hybrid_gemini":
                dets, _ = run_hybrid_gemini_detection(
                    letterboxed,
                    conf_threshold=float(conf_threshold),
                )
            else:
                dets, _ = run_yolo_detection(
                    letterboxed,
                    mode="yolov8",
                    conf_threshold=float(conf_threshold),
                )
            found_meat = False
            for det in dets:
                if det.get("class_id") == "canned_tuna_meat":
                    found_meat = True
                    break
            if found_meat:
                false_meat_images.append(name)

        det_rate = round(float(len(false_meat_images)) / float(total), 4) if total > 0 else 0.0
        false_meat_by_detector[det_name] = {
            "false_meat_rate": det_rate,
            "false_meat_count": len(false_meat_images),
            "total_pantry_negatives": total,
            "false_meat_images": false_meat_images,
        }

    yolo_stats = false_meat_by_detector["yolov8n_finetuned"]
    return {
        "false_meat_rate": yolo_stats["false_meat_rate"],
        "false_meat_count": yolo_stats["false_meat_count"],
        "total_pantry_negatives": yolo_stats["total_pantry_negatives"],
        "false_meat_images": yolo_stats["false_meat_images"],
        "false_meat_by_detector": false_meat_by_detector,
    }


def render_benchmark_chart(rows, output_plot):
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)

    splits = ["normal", "shadowed_sub100lux", "occluded_20_50pct"]
    split_labels = ["Normal", "Shadowed (<100 lux)", "Occluded (20-50%)"]

    preferred_order = [
        "orb_baseline",
        "sift_baseline",
        "yolov8n_finetuned",
        "gemini_vision",
        "hybrid_gemini",
    ]
    seen_detectors = set()
    present_order = []
    for row in rows:
        det_name = row.get("detector")
        if det_name and det_name not in seen_detectors:
            seen_detectors.add(det_name)
            present_order.append(det_name)

    detectors = []
    for det_name in preferred_order:
        if det_name in seen_detectors:
            detectors.append(det_name)
    for det_name in present_order:
        if det_name not in detectors:
            detectors.append(det_name)
    if not detectors:
        detectors = ["orb_baseline", "sift_baseline", "yolov8n_finetuned"]

    detector_labels = {
        "orb_baseline": "ORB Baseline",
        "sift_baseline": "SIFT Baseline",
        "yolov8n_finetuned": "YOLOv8n Fine-Tuned",
        "gemini_vision": "Gemini Vision",
        "hybrid_gemini": "Hybrid Gemini",
    }
    colors = {
        "orb_baseline": "steelblue",
        "sift_baseline": "darkorange",
        "yolov8n_finetuned": "seagreen",
        "gemini_vision": "mediumpurple",
        "hybrid_gemini": "crimson",
    }
    fallback_palette = ["teal", "goldenrod", "slategray", "orchid"]

    lookup = {}
    for row in rows:
        lookup[(row["split"], row["detector"])] = row

    fig_size = (15, 9.5) if len(detectors) > 3 else (13, 9)
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    panels = [
        (axes[0, 0], "mAP50", "mAP@0.50", (0.0, 1.08)),
        (axes[0, 1], "recall", "Recall", (0.0, 1.08)),
        (axes[1, 0], "precision", "Precision", (0.0, 1.08)),
        (axes[1, 1], "fps", "Inference Speed (FPS)", None),
    ]

    x = np.arange(len(splits))
    n_detectors = max(1, len(detectors))
    bar_width = min(0.25, 0.80 / float(n_detectors))
    ann_font_size = 7 if n_detectors >= 5 else 8

    for ax, metric_key, title, ylim in panels:
        for d_idx, det in enumerate(detectors):
            vals = []
            for sp in splits:
                entry = lookup.get((sp, det), {})
                vals.append(float(entry.get(metric_key, 0.0)))
            offset = (float(d_idx) - (float(n_detectors - 1) / 2.0)) * bar_width
            label = detector_labels.get(det, det.replace("_", " ").title())
            color = colors.get(det, fallback_palette[d_idx % len(fallback_palette)])
            bars = ax.bar(
                x + offset,
                vals,
                bar_width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            for bar in bars:
                h_val = bar.get_height()
                fmt = f"{h_val:.2f}" if metric_key != "fps" else f"{h_val:.1f}"
                ax.annotate(
                    fmt,
                    xy=(bar.get_x() + bar.get_width() / 2.0, h_val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=ann_font_size,
                )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.legend(fontsize=8, loc="best")

    if "gemini_vision" in detectors or "hybrid_gemini" in detectors:
        chart_title = "Stock2Stove Comparative Benchmark: Classical Baselines, YOLOv8n & Gemini Vision"
    else:
        chart_title = "Stock2Stove Phase 2 Comparative Benchmark: Classical Baselines vs YOLOv8n"
    fig.suptitle(chart_title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)


def run_benchmark_suite(
    data_yaml_path,
    output_csv,
    output_json,
    output_plot,
    conf_threshold=0.45,
    iou_threshold=0.50,
    include_gemini=False,
):
    samples = load_yolo_test_split(data_yaml_path)
    if not samples:
        raise RuntimeError(f"No test images found for dataset: {data_yaml_path}")

    MODEL_REGISTRY["yolov8_model"] = None
    MODEL_REGISTRY["yolov8_path"] = None
    _, loaded_weights_path = load_ultralytics_detector(mode="yolov8")
    load_templates_for_mode("orb_baseline")
    load_templates_for_mode("sift_baseline")

    split_images = {
        "normal": [],
        "shadowed_sub100lux": [],
        "occluded_20_50pct": [],
    }
    all_gts = []

    for sample in samples:
        bgr = sample["image"]
        filename = sample["filename"]

        split_images["normal"].append((filename, bgr))

        shadowed_bgr = simulate_low_light_shadow(bgr)
        clahe_out = preprocess_frame(shadowed_bgr, enable_clahe=True)
        split_images["shadowed_sub100lux"].append((filename, clahe_out["enhanced_bgr"]))

        occluded_bgr = simulate_partial_occlusion(bgr, sample["gt_boxes"], occlusion_ratio=0.35)
        split_images["occluded_20_50pct"].append((filename, occluded_bgr))

        for gt in sample["gt_boxes"]:
            all_gts.append({
                "image_id": filename,
                "class_idx": gt["class_idx"],
                "class_id": gt["class_id"],
                "bbox_norm": list(gt["bbox_norm"]),
            })

    splits_order = ["normal", "shadowed_sub100lux", "occluded_20_50pct"]
    detectors_order = ["orb_baseline", "sift_baseline", "yolov8n_finetuned"]

    has_gemini_key = bool((os.getenv("GEMINI_API_KEY") or "").strip())
    effective_include_gemini = bool(include_gemini) and has_gemini_key
    if effective_include_gemini:
        detectors_order.extend(["gemini_vision", "hybrid_gemini"])
    elif include_gemini and not has_gemini_key:
        notice = "GEMINI_API_KEY is not set; skipping gemini_vision and hybrid_gemini benchmark evaluation."
        logger.warning(notice)
        print(notice)

    rows = []
    by_split = {}

    for split_name in splits_order:
        by_split[split_name] = {}
        images_list = split_images[split_name]

        for detector_name in detectors_order:
            all_preds = []
            latencies_ms = []

            if detector_name not in ("gemini_vision", "hybrid_gemini"):
                for warm_idx in range(min(3, len(images_list))):
                    _, warm_img = images_list[warm_idx]
                    if detector_name in ("orb_baseline", "sift_baseline"):
                        run_baseline_detection(
                            warm_img,
                            mode=detector_name,
                            conf_threshold=float(conf_threshold),
                        )
                    else:
                        run_yolo_detection(
                            warm_img,
                            mode="yolov8",
                            conf_threshold=float(conf_threshold),
                        )

            for filename, img_bgr in images_list:
                t_start = time.perf_counter()
                if detector_name in ("orb_baseline", "sift_baseline"):
                    dets, _ = run_baseline_detection(
                        img_bgr,
                        mode=detector_name,
                        conf_threshold=float(conf_threshold),
                    )
                elif detector_name == "gemini_vision":
                    dets, _ = run_gemini_detection(
                        img_bgr,
                        conf_threshold=float(conf_threshold),
                        use_cache=False,
                    )
                elif detector_name == "hybrid_gemini":
                    dets, _ = run_hybrid_gemini_detection(
                        img_bgr,
                        conf_threshold=float(conf_threshold),
                    )
                else:
                    dets, _ = run_yolo_detection(
                        img_bgr,
                        mode="yolov8",
                        conf_threshold=float(conf_threshold),
                    )
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                latencies_ms.append(elapsed_ms)

                for det in dets:
                    all_preds.append({
                        "image_id": filename,
                        "class_id": det["class_id"],
                        "confidence": float(det["confidence"]),
                        "bbox_norm": list(det["bbox_norm"]),
                    })

            tp, fp, fn, ap_per_class = evaluate_predictions_against_gt(
                all_preds,
                all_gts,
                iou_threshold=float(iou_threshold),
            )
            map50_95 = compute_map50_95(all_preds, all_gts)
            precision_per_class, recall_per_class = compute_per_class_pr(
                all_preds,
                all_gts,
                iou_threshold=float(iou_threshold),
            )

            if ap_per_class:
                map50 = sum(ap_per_class.values()) / float(len(ap_per_class))
            else:
                map50 = 0.0

            precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
            recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
            if (precision + recall) > 0.0:
                f1_score = (2.0 * precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
            fps = (1000.0 / mean_ms) if mean_ms > 0.0 else 0.0

            row_dict = {
                "split": split_name,
                "detector": detector_name,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "mean_inference_ms": round(mean_ms, 2),
                "fps": round(fps, 2),
            }
            rows.append(row_dict)

            json_entry = dict(row_dict)
            json_entry["ap_per_class"] = ap_per_class
            json_entry["precision_per_class"] = precision_per_class
            json_entry["recall_per_class"] = recall_per_class
            json_entry["canned_tuna_meat_precision"] = precision_per_class.get("canned_tuna_meat", 0.0)
            by_split[split_name][detector_name] = json_entry

    false_meat_eval = evaluate_false_meat_rate(
        data_yaml_path,
        conf_threshold=float(conf_threshold),
        include_gemini=effective_include_gemini,
    )
    normal_yolo_entry = by_split.get("normal", {}).get("yolov8n_finetuned", {})
    tuna_precision = float(normal_yolo_entry.get("canned_tuna_meat_precision", 0.0))

    csv_parent = os.path.dirname(os.path.abspath(output_csv))
    if csv_parent:
        os.makedirs(csv_parent, exist_ok=True)
    fieldnames = [
        "split",
        "detector",
        "tp",
        "fp",
        "fn",
        "mAP50",
        "mAP50_95",
        "precision",
        "recall",
        "f1_score",
        "mean_inference_ms",
        "fps",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "data_yaml": str(data_yaml_path),
        "num_test_images": len(samples),
        "num_gt_boxes": len(all_gts),
        "conf_threshold": float(conf_threshold),
        "iou_threshold": float(iou_threshold),
        "include_gemini": bool(effective_include_gemini),
        "weights_path": str(loaded_weights_path) if loaded_weights_path else "opencv_hybrid_fallback",
        "false_meat_rate": false_meat_eval["false_meat_rate"],
        "false_meat_by_detector": false_meat_eval["false_meat_by_detector"],
        "canned_tuna_meat_precision": tuna_precision,
        "false_meat_evaluation": false_meat_eval,
        "splits": splits_order,
        "detectors": detectors_order,
        "results": rows,
        "by_split": by_split,
    }

    json_parent = os.path.dirname(os.path.abspath(output_json))
    if json_parent:
        os.makedirs(json_parent, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    render_benchmark_chart(rows, output_plot)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset_yolo/data.yaml")
    parser.add_argument("--output-csv", type=str, default="outputs/benchmark_results.csv")
    parser.add_argument("--output-json", type=str, default="outputs/benchmark_summary.json")
    parser.add_argument("--output-plot", type=str, default="outputs/phase2_benchmark_comparison.png")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--include-gemini", action="store_true")
    args = parser.parse_args()

    summary = run_benchmark_suite(
        data_yaml_path=args.data,
        output_csv=args.output_csv,
        output_json=args.output_json,
        output_plot=args.output_plot,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        include_gemini=args.include_gemini,
    )
    print(
        json.dumps(
            {
                "weights_path": summary["weights_path"],
                "detectors": summary["detectors"],
                "rows": len(summary["results"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
