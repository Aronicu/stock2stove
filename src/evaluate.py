import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cv.baseline_detector import compute_box_iou
from src.run import infer, postprocess, preprocess, process_video_file, render
from src.server.config import NMS_IOU_THRESHOLD, UNIFIED_CLASSES, normalize_class_id
from src.training.evaluate_all import (
    compute_map50_95,
    compute_per_class_pr,
    evaluate_false_meat_rate,
    evaluate_predictions_against_gt,
    load_yolo_test_split,
    simulate_low_light_shadow,
    simulate_partial_occlusion,
)

CLASS_ORDER = [entry["class_id"] for entry in UNIFIED_CLASSES]


def simulate_motion_blur(bgr_image, kernel_size=11):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    mid = k // 2
    kernel[mid, :] = 1.0 / float(k)
    return cv2.filter2D(bgr_image, -1, kernel)


def simulate_angled_perspective(bgr_image, gt_boxes, angle_deg=16.0, scale=0.88):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")
    h, w = bgr_image.shape[:2]
    center = (float(w) / 2.0, float(h) / 2.0)
    mat = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale))
    warped = cv2.warpAffine(
        bgr_image,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )

    transformed_gts = []
    for gt in gt_boxes:
        x1, y1, x2, y2 = gt["bbox_norm"]
        corners = np.array(
            [
                [[x1 * float(w), y1 * float(h)]],
                [[x2 * float(w), y1 * float(h)]],
                [[x2 * float(w), y2 * float(h)]],
                [[x1 * float(w), y2 * float(h)]],
            ],
            dtype=np.float32,
        )
        warped_pts = cv2.transform(corners, mat).reshape(4, 2)
        nx1 = max(0.0, min(1.0, float(np.min(warped_pts[:, 0])) / float(w)))
        ny1 = max(0.0, min(1.0, float(np.min(warped_pts[:, 1])) / float(h)))
        nx2 = max(0.0, min(1.0, float(np.max(warped_pts[:, 0])) / float(w)))
        ny2 = max(0.0, min(1.0, float(np.max(warped_pts[:, 1])) / float(h)))
        if nx2 - nx1 < 0.01 or ny2 - ny1 < 0.01:
            continue
        gt_copy = dict(gt)
        gt_copy["bbox_norm"] = [round(nx1, 6), round(ny1, 6), round(nx2, 6), round(ny2, 6)]
        transformed_gts.append(gt_copy)

    return warped, transformed_gts


def build_condition_datasets(samples):
    conditions = {
        "normal": [],
        "shadowed_sub100lux": [],
        "motion_blur": [],
        "angled_perspective": [],
        "occluded_20_50pct": [],
        "cluttered_shelf": [],
    }

    for idx, sample in enumerate(samples):
        bgr = sample["image"]
        filename = sample["filename"]
        gts = sample["gt_boxes"]

        conditions["normal"].append({
            "filename": filename,
            "image": bgr,
            "gt_boxes": gts,
        })

        shadow_bgr = simulate_low_light_shadow(bgr)
        conditions["shadowed_sub100lux"].append({
            "filename": filename,
            "image": shadow_bgr,
            "gt_boxes": gts,
        })

        blur_bgr = simulate_motion_blur(bgr, kernel_size=11)
        conditions["motion_blur"].append({
            "filename": filename,
            "image": blur_bgr,
            "gt_boxes": gts,
        })

        angle = 16.0 if idx % 2 == 0 else -16.0
        angled_bgr, angled_gts = simulate_angled_perspective(bgr, gts, angle_deg=angle, scale=0.88)
        conditions["angled_perspective"].append({
            "filename": filename,
            "image": angled_bgr,
            "gt_boxes": angled_gts,
        })

        occluded_bgr = simulate_partial_occlusion(bgr, gts, occlusion_ratio=0.35)
        conditions["occluded_20_50pct"].append({
            "filename": filename,
            "image": occluded_bgr,
            "gt_boxes": gts,
        })

        if len(gts) >= 4:
            conditions["cluttered_shelf"].append({
                "filename": filename,
                "image": bgr,
                "gt_boxes": gts,
            })

    return conditions


def run_detector_on_split(
    split_items,
    mode="yolov8",
    enable_clahe=True,
    conf_threshold=0.45,
    iou_threshold=NMS_IOU_THRESHOLD,
):
    if not split_items:
        return [], [], {}, []

    for warm_idx in range(min(3, len(split_items))):
        warm_img = split_items[warm_idx]["image"]
        prep_w = preprocess(warm_img, enable_clahe=enable_clahe)
        raw_w, _ = infer(prep_w, mode=mode, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        postprocess(raw_w, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    all_preds = []
    all_gts = []
    per_image_records = {}
    latencies_ms = []

    for item in split_items:
        fname = item["filename"]
        bgr = item["image"]
        gts = item["gt_boxes"]

        for gt in gts:
            all_gts.append({
                "image_id": fname,
                "class_id": gt["class_id"],
                "bbox_norm": list(gt["bbox_norm"]),
            })

        t_start = time.perf_counter()
        prep = preprocess(bgr, enable_clahe=enable_clahe)
        raw_dets, _ = infer(
            prep,
            mode=mode,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        post = postprocess(
            raw_dets,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        latencies_ms.append(elapsed_ms)

        dets = post["detections"]
        for det in dets:
            all_preds.append({
                "image_id": fname,
                "class_id": det["class_id"],
                "confidence": float(det["confidence"]),
                "bbox_norm": list(det["bbox_norm"]),
            })

        per_image_records[fname] = {
            "filename": fname,
            "image": bgr,
            "gt_boxes": gts,
            "detections": dets,
            "counts": post["counts_by_class"],
            "elapsed_ms": round(elapsed_ms, 2),
        }

    return all_preds, all_gts, per_image_records, latencies_ms


def summarize_split_metrics(
    split_name,
    detector_label,
    all_preds,
    all_gts,
    latencies_ms,
    num_images,
    iou_threshold=0.50,
):
    tp, fp, fn, ap_per_class = evaluate_predictions_against_gt(
        all_preds,
        all_gts,
        iou_threshold=float(iou_threshold),
    )
    map50_95 = compute_map50_95(all_preds, all_gts)
    prec_per_class, rec_per_class = compute_per_class_pr(
        all_preds,
        all_gts,
        iou_threshold=float(iou_threshold),
    )
    map50 = sum(ap_per_class.values()) / float(len(ap_per_class)) if ap_per_class else 0.0
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0
    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = (1000.0 / mean_ms) if mean_ms > 0.0 else 0.0

    row = {
        "split": split_name,
        "detector": detector_label,
        "num_images": int(num_images),
        "num_gt": len(all_gts),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "mean_inference_ms": round(mean_ms, 2),
        "fps": round(fps, 2),
    }
    return row, ap_per_class, prec_per_class, rec_per_class


def analyze_image_errors(per_image_records, iou_match_thresh=0.45):
    y_true_all = []
    y_pred_all = []
    y_true_matched = []
    y_pred_matched = []
    image_failures = []

    for fname, rec in per_image_records.items():
        gts = rec["gt_boxes"]
        preds = list(rec["detections"])
        preds.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)

        gt_matched = [False] * len(gts)
        pred_matched = [False] * len(preds)
        matched_pairs = []

        for p_idx, pred in enumerate(preds):
            pbox = pred["bbox_norm"]
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, gt in enumerate(gts):
                if gt_matched[g_idx]:
                    continue
                iou_val = compute_box_iou(pbox, gt["bbox_norm"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_g_idx = g_idx
            if best_g_idx >= 0 and best_iou >= float(iou_match_thresh):
                gt_matched[best_g_idx] = True
                pred_matched[p_idx] = True
                gt_cls = normalize_class_id(gts[best_g_idx]["class_id"])
                pred_cls = normalize_class_id(pred["class_id"])
                y_true_all.append(gt_cls)
                y_pred_all.append(pred_cls)
                y_true_matched.append(gt_cls)
                y_pred_matched.append(pred_cls)
                matched_pairs.append({
                    "gt_class": gt_cls,
                    "pred_class": pred_cls,
                    "iou": round(best_iou, 3),
                    "conf": float(pred.get("confidence", 0.0)),
                    "gt_box": gts[best_g_idx]["bbox_norm"],
                    "pred_box": pbox,
                })

        missed_gts = []
        for g_idx, gt in enumerate(gts):
            if not gt_matched[g_idx]:
                gt_cls = normalize_class_id(gt["class_id"])
                y_true_all.append(gt_cls)
                y_pred_all.append("missed_bg")
                missed_gts.append(gt)

        false_preds = []
        for p_idx, pred in enumerate(preds):
            if not pred_matched[p_idx]:
                pred_cls = normalize_class_id(pred["class_id"])
                y_true_all.append("bg_fp")
                y_pred_all.append(pred_cls)
                false_preds.append(pred)

        misclassified = [m for m in matched_pairs if m["gt_class"] != m["pred_class"]]
        if misclassified or missed_gts or false_preds:
            image_failures.append({
                "filename": fname,
                "image": rec["image"],
                "gt_boxes": gts,
                "detections": preds,
                "matched_pairs": matched_pairs,
                "misclassified": misclassified,
                "missed_gts": missed_gts,
                "false_preds": false_preds,
            })

    return {
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "y_true_matched": y_true_matched,
        "y_pred_matched": y_pred_matched,
        "image_failures": image_failures,
    }


def evaluate_demo_video_ground_truth(video_path, output_dir, conf_threshold=0.45):
    samples_dir = Path(PROJECT_ROOT) / "data" / "samples"
    tiles = []
    tile_boxes = []
    x_cursor = 0
    frame_h = 480

    for idx in range(1, 9):
        spath = samples_dir / f"shelf_sample_{idx:02d}.jpg"
        img = cv2.imread(str(spath))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        scale = float(frame_h) / float(max(1, ih))
        new_w = max(1, int(round(float(iw) * scale)))
        resized = cv2.resize(img, (new_w, frame_h))
        tiles.append(resized)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        shelf_area = resized[: int(0.75 * frame_h), :]
        hsv = cv2.cvtColor(shelf_area, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] > 40) | (gray[: int(0.75 * frame_h), :] > 95)
        mask_u8 = (mask.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        contours_info = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw >= 50 and bh >= 80:
                tile_boxes.append((x_cursor + bx, by, x_cursor + bx + bw, by + bh))
        x_cursor += new_w

    total_expected_objects = len(tile_boxes)
    panorama_w = x_cursor
    frame_w = 640
    max_x = max(0, panorama_w - frame_w)

    video_out = Path(output_dir) / "demo_pantry_pan_annotated.mp4"
    summary = process_video_file(
        video_path,
        str(video_out),
        mode="yolov8",
        enable_clahe=True,
        conf_threshold=float(conf_threshold),
        iou_threshold=NMS_IOU_THRESHOLD,
        skip_frames=1,
    )

    csv_path = summary["predictions_csv"]
    pred_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_rows.append(row)

    total_frames = int(summary["total_video_frames"])
    frame_expected_counts = []
    frame_pred_counts = []
    missed_frames = 0
    false_pos_total = 0
    class_switches = 0
    prev_classes = None

    for idx in range(total_frames):
        ratio = float(idx) / float(max(1, total_frames - 1))
        x_off = int(round(ratio * float(max_x)))
        x_end = x_off + frame_w
        visible = 0
        for bx1, by1, bx2, by2 in tile_boxes:
            inter_w = min(x_end, bx2) - max(x_off, bx1)
            box_w = max(1, bx2 - bx1)
            if float(inter_w) / float(box_w) >= 0.40:
                visible += 1
        frame_expected_counts.append(visible)

        if idx < len(pred_rows):
            row = pred_rows[idx]
            pcnt = int(row.get("total_count", 0))
            frame_pred_counts.append(pcnt)
            if pcnt == 0 and visible > 0:
                missed_frames += 1
            if pcnt > visible:
                false_pos_total += (pcnt - visible)
            try:
                counts_dict = json.loads(row.get("counts_json", "{}"))
            except Exception:
                counts_dict = {}
            cur_classes = tuple(sorted(counts_dict.keys()))
            if prev_classes is not None and cur_classes != prev_classes:
                class_switches += 1
            prev_classes = cur_classes

    mean_expected = float(np.mean(frame_expected_counts)) if frame_expected_counts else 0.0
    mean_pred = float(np.mean(frame_pred_counts)) if frame_pred_counts else 0.0
    mae_per_frame = float(np.mean([abs(e - p) for e, p in zip(frame_expected_counts, frame_pred_counts)])) if frame_pred_counts else 0.0

    return {
        "clip_name": "demo_pantry_pan.mp4",
        "total_frames": total_frames,
        "resolution": "640x480",
        "native_fps": 15.0,
        "expected_unique_items": int(total_expected_objects),
        "predicted_confirmed_items": int(summary["total_confirmed_items"]),
        "clip_count_error": int(summary["total_confirmed_items"] - total_expected_objects),
        "mean_expected_per_frame": round(mean_expected, 2),
        "mean_predicted_per_frame": round(mean_pred, 2),
        "frame_mae": round(mae_per_frame, 2),
        "zero_detection_frames": int(missed_frames),
        "frame_overcounts": int(false_pos_total),
        "class_state_switches": int(class_switches),
        "mean_inference_ms": summary["mean_inference_ms"],
        "fps": summary["fps"],
        "confirmed_counts_by_class": summary["counts_by_class"],
    }


def save_confusion_matrix_and_report(err_info, output_dir):
    report_path = os.path.join(output_dir, "classification_report.txt")
    cm_path = os.path.join(output_dir, "test_confusion_matrix.png")

    y_true_matched = err_info["y_true_matched"]
    y_pred_matched = err_info["y_pred_matched"]
    y_true_all = err_info["y_true_all"]
    y_pred_all = err_info["y_pred_all"]

    matched_report = classification_report(
        y_true_matched,
        y_pred_matched,
        labels=CLASS_ORDER,
        digits=3,
        zero_division=0,
    )

    all_labels = CLASS_ORDER + ["missed_bg", "bg_fp"]
    full_report = classification_report(
        y_true_all,
        y_pred_all,
        labels=all_labels,
        digits=3,
        zero_division=0,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== 8-Class Classification Report on Spatially Matched Test Boxes (IoU >= 0.45) ===\n")
        f.write(matched_report)
        f.write("\n=== Full Detection + Background Classification Report (including Missed FN & False Positive FP) ===\n")
        f.write(full_report)

    cm_labels = CLASS_ORDER + ["missed_bg", "bg_fp"]
    fig, ax = plt.subplots(figsize=(10.5, 8.8))
    ConfusionMatrixDisplay.from_predictions(
        y_true_all,
        y_pred_all,
        labels=cm_labels,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax,
        colorbar=True,
        values_format="d",
    )
    ax.set_title("Stock2Stove YOLOv8n Held-Out Test Confusion Matrix (104 Images, conf=0.45)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("Ground Truth Label", fontsize=10)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    return report_path, cm_path


def draw_failure_tile(bgr_image, gt_boxes, pred_boxes, title_line, subtitle_line):
    canvas = bgr_image.copy()
    h, w = canvas.shape[:2]
    target_w = 480
    target_h = 360
    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    nw = max(1, int(round(float(w) * scale)))
    nh = max(1, int(round(float(h) * scale)))
    resized = cv2.resize(canvas, (nw, nh))
    tile = np.full((target_h + 52, target_w, 3), 245, dtype=np.uint8)
    x_off = (target_w - nw) // 2
    y_off = 52 + (target_h - nh) // 2
    tile[y_off : y_off + nh, x_off : x_off + nw] = resized

    for gt in gt_boxes:
        gx1, gy1, gx2, gy2 = gt["bbox_norm"]
        px1 = x_off + int(round(gx1 * float(nw)))
        py1 = y_off + int(round(gy1 * float(nh)))
        px2 = x_off + int(round(gx2 * float(nw)))
        py2 = y_off + int(round(gy2 * float(nh)))
        cv2.rectangle(tile, (px1, py1), (px2, py2), (35, 165, 55), 2)
        cv2.putText(tile, f"GT:{gt['class_id'][:10]}", (px1 + 2, max(y_off + 14, py1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 130, 35), 1, cv2.LINE_AA)

    for pred in pred_boxes:
        bx1, by1, bx2, by2 = pred["bbox_norm"]
        px1 = x_off + int(round(bx1 * float(nw)))
        py1 = y_off + int(round(by1 * float(nh)))
        px2 = x_off + int(round(bx2 * float(nw)))
        py2 = y_off + int(round(by2 * float(nh)))
        cv2.rectangle(tile, (px1, py1), (px2, py2), (35, 45, 220), 2)
        cv2.putText(tile, f"P:{pred['class_id'][:9]} {pred['confidence']:.2f}", (px1 + 2, min(y_off + nh - 4, py2 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 200), 1, cv2.LINE_AA)

    cv2.rectangle(tile, (0, 0), (target_w, 50), (30, 35, 45), -1)
    cv2.putText(tile, title_line[:58], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, subtitle_line[:66], (8, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (205, 220, 240), 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (target_w - 1, target_h + 51), (180, 180, 180), 1)
    return tile


def build_failure_gallery_and_counts(condition_results, video_eval, output_dir):
    failures_dir = os.path.join(output_dir, "failures")
    os.makedirs(failures_dir, exist_ok=True)

    normal_err = condition_results["normal"]["errors"]
    shadow_err = condition_results["shadowed_sub100lux"]["errors"]
    occ_err = condition_results["occluded_20_50pct"]["errors"]
    blur_err = condition_results["motion_blur"]["errors"]
    angle_err = condition_results["angled_perspective"]["errors"]
    clutter_err = condition_results["cluttered_shelf"]["errors"]

    cause_counts = {
        "Occlusion and clutter": 0,
        "Lighting (shadows/glare)": 0,
        "Viewpoint and scale": 0,
        "Look-alike classes": 0,
        "Image quality (blur/low-res)": 0,
        "Label errors / domain gap": 0,
    }

    normal_rec_map = condition_results["normal"]["records"]
    for item in normal_err["image_failures"]:
        cause_counts["Look-alike classes"] += len(item["misclassified"])
        for gt in item["missed_gts"]:
            bw = gt["bbox_norm"][2] - gt["bbox_norm"][0]
            bh = gt["bbox_norm"][3] - gt["bbox_norm"][1]
            if len(item["gt_boxes"]) >= 4:
                cause_counts["Occlusion and clutter"] += 1
            elif bw * bh < 0.035 or gt["bbox_norm"][0] < 0.02 or gt["bbox_norm"][2] > 0.98:
                cause_counts["Viewpoint and scale"] += 1
            else:
                cause_counts["Image quality (blur/low-res)"] += 1
        for fp_det in item["false_preds"]:
            if len(item["gt_boxes"]) == 0:
                cause_counts["Occlusion and clutter"] += 1
            else:
                cause_counts["Label errors / domain gap"] += 1

    if abs(int(video_eval.get("clip_count_error", 0))) > 2:
        cause_counts["Label errors / domain gap"] += 1

    shadow_fn = condition_results["shadowed_sub100lux"]["yolo_row"]["fn"]
    normal_fn = condition_results["normal"]["yolo_row"]["fn"]
    shadow_delta_fn = max(0, shadow_fn - normal_fn)
    cause_counts["Lighting (shadows/glare)"] += shadow_delta_fn

    occ_fn = condition_results["occluded_20_50pct"]["yolo_row"]["fn"]
    occ_delta_fn = max(0, occ_fn - normal_fn)
    cause_counts["Occlusion and clutter"] += occ_delta_fn

    blur_fn = condition_results["motion_blur"]["yolo_row"]["fn"]
    blur_delta_fn = max(0, blur_fn - normal_fn)
    cause_counts["Image quality (blur/low-res)"] += blur_delta_fn

    angle_fn = condition_results["angled_perspective"]["yolo_row"]["fn"]
    angle_delta_fn = max(0, angle_fn - normal_fn)
    cause_counts["Viewpoint and scale"] += angle_delta_fn

    selected_cases = []
    used_keys = set()

    def pick_from_error_list(err_list, cause_label, condition_name, limit, require_type="any"):
        picked = 0
        for item in err_list:
            if picked >= limit:
                break
            key = (condition_name, item["filename"])
            if key in used_keys:
                continue
            if require_type == "misclassified" and not item["misclassified"]:
                continue
            if require_type == "missed" and not item["missed_gts"]:
                continue
            if require_type == "fp" and not item["false_preds"]:
                continue
            if require_type == "missed_only" and (not item["missed_gts"] or item["misclassified"]):
                continue
            if require_type == "small_or_edge":
                has_small = False
                for gt in item["missed_gts"]:
                    area = (gt["bbox_norm"][2] - gt["bbox_norm"][0]) * (gt["bbox_norm"][3] - gt["bbox_norm"][1])
                    if area < 0.04 or gt["bbox_norm"][0] < 0.03 or gt["bbox_norm"][2] > 0.97:
                        has_small = True
                        break
                if not has_small:
                    continue

            used_keys.add(key)
            gt_names = ", ".join([g["class_id"] for g in item["gt_boxes"]]) or "none"
            pred_names = ", ".join([d["class_id"] for d in item["detections"]]) or "none"
            selected_cases.append({
                "id": f"F{len(selected_cases) + 1:02d}",
                "cause": cause_label,
                "condition": condition_name,
                "filename": item["filename"],
                "image": item["image"],
                "gt_boxes": item["gt_boxes"],
                "detections": item["detections"],
                "expected": gt_names,
                "got": pred_names,
            })
            picked += 1

    pick_from_error_list(clutter_err["image_failures"], "Occlusion and clutter", "cluttered_shelf", 2, "missed")
    pick_from_error_list(occ_err["image_failures"], "Occlusion and clutter", "occluded_20_50pct", 1, "missed")
    pick_from_error_list(shadow_err["image_failures"], "Lighting (shadows/glare)", "shadowed_sub100lux", 3, "missed")
    pick_from_error_list(normal_err["image_failures"], "Look-alike classes", "normal", 2, "misclassified")
    pick_from_error_list(angle_err["image_failures"], "Viewpoint and scale", "angled_perspective", 1, "missed")
    pick_from_error_list(normal_err["image_failures"], "Viewpoint and scale", "normal", 1, "small_or_edge")
    pick_from_error_list(blur_err["image_failures"], "Image quality (blur/low-res)", "motion_blur", 1, "missed")
    pick_from_error_list(normal_err["image_failures"], "Label errors / domain gap", "normal", 1, "fp")

    while len(selected_cases) < 12:
        before_len = len(selected_cases)
        pick_from_error_list(normal_err["image_failures"], "Image quality (blur/low-res)", "normal", 1, "missed")
        if len(selected_cases) == before_len:
            pick_from_error_list(occ_err["image_failures"], "Occlusion and clutter", "occluded_20_50pct", 1, "missed")
        if len(selected_cases) == before_len:
            break

    tiles = []
    gallery_meta = []
    for case in selected_cases[:12]:
        title_line = f"{case['id']} | {case['cause']} ({case['condition']})"
        subtitle_line = f"Exp: {case['expected'][:28]} | Got: {case['got'][:28]}"
        tile = draw_failure_tile(
            case["image"],
            case["gt_boxes"],
            case["detections"],
            title_line,
            subtitle_line,
        )
        tile_path = os.path.join(failures_dir, f"{case['id']}_{case['filename']}")
        cv2.imwrite(tile_path, tile)
        tiles.append(tile)
        gallery_meta.append({
            "id": case["id"],
            "cause": case["cause"],
            "condition": case["condition"],
            "filename": case["filename"],
            "expected": case["expected"],
            "got": case["got"],
            "tile_path": tile_path,
        })

    if tiles:
        cols = 3
        rows = math.ceil(float(len(tiles)) / float(cols))
        th, tw = tiles[0].shape[:2]
        grid = np.full((rows * th, cols * tw, 3), 235, dtype=np.uint8)
        for idx, tile in enumerate(tiles):
            r = idx // cols
            c = idx % cols
            grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
        gallery_path = os.path.join(output_dir, "failure_gallery.png")
        cv2.imwrite(gallery_path, grid)
    else:
        gallery_path = os.path.join(output_dir, "failure_gallery.png")

    return {
        "cause_counts": cause_counts,
        "total_attributed_failures": int(sum(cause_counts.values())),
        "gallery_cases": gallery_meta,
        "failure_gallery_png": gallery_path,
    }


def build_representative_test_log(condition_results, video_eval, output_dir):
    log_rows = []
    used = set()

    def add_case(cond_key, cond_label, sample_rec, force_pass=None):
        fname = sample_rec["filename"]
        gts = sample_rec["gt_boxes"]
        dets = sample_rec["detections"]

        gt_counts = {}
        for g in gts:
            cid = g["class_id"]
            gt_counts[cid] = gt_counts.get(cid, 0) + 1
        det_counts = {}
        for d in dets:
            cid = d["class_id"]
            det_counts[cid] = det_counts.get(cid, 0) + 1

        expected_str = ", ".join([f"{v}x {k}" for k, v in sorted(gt_counts.items())]) or "0 items (background)"
        got_str = ", ".join([f"{v}x {k}" for k, v in sorted(det_counts.items())]) or "0 items"

        if force_pass is not None:
            passed = bool(force_pass)
        else:
            passed = gt_counts == det_counts

        log_rows.append({
            "ID": f"T{len(log_rows) + 1:02d}",
            "Input": fname,
            "Condition": cond_label,
            "Expected": expected_str,
            "Got": got_str,
            "Status": "PASS" if passed else "FAIL",
        })

    def pick_cases(cond_key, cond_label, want_pass, want_fail):
        recs = list(condition_results[cond_key]["records"].values())
        pass_list = []
        fail_list = []
        for rec in recs:
            if (cond_key, rec["filename"]) in used:
                continue
            gt_c = {}
            for g in rec["gt_boxes"]:
                gt_c[g["class_id"]] = gt_c.get(g["class_id"], 0) + 1
            det_c = {}
            for d in rec["detections"]:
                det_c[d["class_id"]] = det_c.get(d["class_id"], 0) + 1
            if gt_c == det_c and len(rec["gt_boxes"]) > 0:
                pass_list.append(rec)
            elif gt_c != det_c and len(rec["gt_boxes"]) > 0:
                fail_list.append(rec)

        for rec in pass_list[:want_pass]:
            used.add((cond_key, rec["filename"]))
            add_case(cond_key, cond_label, rec, force_pass=True)
        for rec in fail_list[:want_fail]:
            used.add((cond_key, rec["filename"]))
            add_case(cond_key, cond_label, rec, force_pass=False)

    pick_cases("normal", "Normal daylight / standard shelf", 3, 1)
    pick_cases("shadowed_sub100lux", "Poor lighting (<100 lux shadow + CLAHE)", 1, 3)
    pick_cases("motion_blur", "Motion blur (handheld pan, k=11)", 1, 2)
    pick_cases("angled_perspective", "Angled viewpoint (+/-16 deg tilt)", 1, 2)
    pick_cases("occluded_20_50pct", "Partial occlusion (35% foreground box)", 1, 3)
    pick_cases("cluttered_shelf", "High clutter (>=4 items per shelf)", 1, 2)

    neg_recs = [r for r in condition_results["normal"]["records"].values() if len(r["gt_boxes"]) == 0]
    for rec in neg_recs[:2]:
        det_c = rec["counts"]
        add_case("normal", "Empty negative background", rec, force_pass=(len(det_c) == 0))

    vid_exp = f"{video_eval['expected_unique_items']} synthetic shelf boxes (pan)"
    vid_got = ", ".join([f"{v}x {k}" for k, v in sorted(video_eval["confirmed_counts_by_class"].items())])
    vid_pass = abs(video_eval["clip_count_error"]) <= 2
    log_rows.append({
        "ID": f"T{len(log_rows) + 1:02d}",
        "Input": "demo_pantry_pan.mp4",
        "Condition": "Video pan (60 frames, W=5, K=3)",
        "Expected": vid_exp,
        "Got": f"{video_eval['predicted_confirmed_items']} ({vid_got})",
        "Status": "PASS" if vid_pass else "FAIL",
    })

    csv_path = os.path.join(output_dir, "test_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Input", "Condition", "Expected", "Got", "Status"])
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)

    return log_rows, csv_path


def render_pipeline_stages_strip(samples, output_dir):
    chosen = []
    for s in samples:
        if len(s["gt_boxes"]) >= 2:
            chosen.append(("Normal Held-Out Test", s["image"]))
            break
    for s in samples:
        if "COFFEE" in s["filename"] or "TOMATO" in s["filename"] or len(s["gt_boxes"]) >= 3:
            if len(chosen) == 1:
                chosen.append(("Shadowed (<100 lux) + CLAHE", simulate_low_light_shadow(s["image"])))
                break
    cap = cv2.VideoCapture(os.path.join(PROJECT_ROOT, "data", "raw_home", "demo_pantry_pan.mp4"))
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        ret, frame = cap.read()
        if ret and frame is not None:
            chosen.append(("Video Pan Frame #10", frame))
    cap.release()

    col_w = 320
    col_h = 240
    header_h = 34
    rows = []

    for row_idx, (row_title, raw_bgr) in enumerate(chosen):
        t0 = time.perf_counter()
        prep = preprocess(raw_bgr, enable_clahe=True)
        raw_dets, _ = infer(prep, mode="yolov8", conf_threshold=0.45)
        post = postprocess(raw_dets, conf_threshold=0.45)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rendered = render(raw_bgr, post, source_name=row_title, elapsed_ms=elapsed_ms, mode="yolov8")

        stage1 = cv2.resize(raw_bgr, (col_w, col_h))
        stage2 = cv2.resize(prep["letterboxed_bgr"], (col_w, col_h))

        raw_canvas = raw_bgr.copy()
        rh, rw = raw_canvas.shape[:2]
        for det in raw_dets:
            bx1, by1, bx2, by2 = det["bbox_norm"]
            cv2.rectangle(
                raw_canvas,
                (int(bx1 * rw), int(by1 * rh)),
                (int(bx2 * rw), int(by2 * rh)),
                (255, 165, 0),
                2,
            )
        stage3 = cv2.resize(raw_canvas, (col_w, col_h))
        stage4 = cv2.resize(rendered["annotated_frame"], (col_w, col_h))

        stages = [
            ("1. Input Frame", stage1),
            ("2. preprocess() LAB+640", stage2),
            (f"3. infer() raw ({len(raw_dets)})", stage3),
            (f"4. postprocess()+render() ({post['total_count']})", stage4),
        ]
        row_tiles = []
        for idx, (col_lbl, img_tile) in enumerate(stages):
            tile = np.full((col_h + header_h, col_w, 3), 245, dtype=np.uint8)
            tile[header_h:, :] = img_tile
            cv2.rectangle(tile, (0, 0), (col_w, header_h), (28, 42, 62), -1)
            hdr_text = f"{col_lbl}" if row_idx == 0 else f"{row_title}: {col_lbl[:20]}"
            cv2.putText(tile, hdr_text[:34], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (col_w - 1, col_h + header_h - 1), (180, 180, 180), 1)
            row_tiles.append(tile)
        rows.append(np.hstack(row_tiles))

    strip = np.vstack(rows)
    out_path = os.path.join(output_dir, "pipeline_stages_strip.png")
    cv2.imwrite(out_path, strip)
    return out_path


def render_preliminary_comparison_chart(rows, output_path):
    splits = [
        "normal",
        "shadowed_sub100lux",
        "motion_blur",
        "angled_perspective",
        "occluded_20_50pct",
        "cluttered_shelf",
    ]
    split_labels = ["Normal", "Shadow\n(<100 lux)", "Motion\nBlur", "Angled\n(±16°)", "Occluded\n(35%)", "Cluttered\n(>=4 items)"]
    detectors = ["orb_baseline", "sift_baseline", "yolov8n_pipeline"]
    det_labels = {
        "orb_baseline": "ORB Baseline",
        "sift_baseline": "SIFT Baseline",
        "yolov8n_pipeline": "YOLOv8n (Main)",
    }
    colors = {
        "orb_baseline": "#4C78A8",
        "sift_baseline": "#F58518",
        "yolov8n_pipeline": "#2E8B57",
    }

    lookup = {}
    for row in rows:
        lookup[(row["split"], row["detector"])] = row

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    panels = [
        (axes[0, 0], "mAP50", "mAP@0.50 Across Conditions", (0.0, 1.05)),
        (axes[0, 1], "f1_score", "F1-Score Across Conditions", (0.0, 1.05)),
        (axes[1, 0], "recall", "Recall Across Conditions", (0.0, 1.05)),
        (axes[1, 1], "fps", "Post-Warmup CPU Speed (FPS)", None),
    ]

    x = np.arange(len(splits))
    width = 0.26

    for ax, metric, title, ylim in panels:
        for d_idx, det in enumerate(detectors):
            vals = [float(lookup.get((sp, det), {}).get(metric, 0.0)) for sp in splits]
            offset = (d_idx - 1) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=det_labels[det],
                color=colors[det],
                edgecolor="black",
                linewidth=0.5,
            )
            for bar in bars:
                hv = bar.get_height()
                fmt = f"{hv:.2f}" if metric != "fps" else f"{hv:.1f}"
                ax.annotate(
                    fmt,
                    xy=(bar.get_x() + bar.get_width() / 2.0, hv),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Stock-to-Stove Milestone 3: Classical Baselines vs. Fine-Tuned YOLOv8n (Held-Out Test Set)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_milestone3_evaluation(
    data_yaml="data/dataset_yolo/data.yaml",
    output_dir="docs/milestone3",
    conf_threshold=0.45,
    iou_threshold=0.50,
):
    os.makedirs(output_dir, exist_ok=True)
    samples = load_yolo_test_split(data_yaml)
    if not samples:
        raise RuntimeError(f"No held-out test samples found for {data_yaml}")

    conditions = build_condition_datasets(samples)
    splits_order = [
        "normal",
        "shadowed_sub100lux",
        "motion_blur",
        "angled_perspective",
        "occluded_20_50pct",
        "cluttered_shelf",
    ]
    detectors_order = ["orb_baseline", "sift_baseline", "yolov8n_pipeline"]

    prelim_rows = []
    condition_results = {}
    cluttered_names = set(item["filename"] for item in conditions["cluttered_shelf"])

    for split_name in splits_order:
        items = conditions[split_name]
        condition_results[split_name] = {}

        if split_name == "cluttered_shelf":
            for det_name in detectors_order:
                norm_res = condition_results["normal"][det_name]
                sub_preds = [p for p in norm_res["preds"] if p["image_id"] in cluttered_names]
                sub_gts = [g for g in norm_res["gts"] if g["image_id"] in cluttered_names]
                sub_lats = [norm_res["records"][fn]["elapsed_ms"] for fn in cluttered_names if fn in norm_res["records"]]
                sub_recs = {fn: norm_res["records"][fn] for fn in cluttered_names if fn in norm_res["records"]}
                row, ap_c, p_c, r_c = summarize_split_metrics(
                    split_name,
                    det_name,
                    sub_preds,
                    sub_gts,
                    sub_lats,
                    len(items),
                    iou_threshold=iou_threshold,
                )
                prelim_rows.append(row)
                condition_results[split_name][det_name] = {
                    "row": row,
                    "ap_per_class": ap_c,
                    "precision_per_class": p_c,
                    "recall_per_class": r_c,
                    "preds": sub_preds,
                    "gts": sub_gts,
                    "records": sub_recs,
                }
            condition_results[split_name]["yolo_row"] = condition_results[split_name]["yolov8n_pipeline"]["row"]
            condition_results[split_name]["records"] = condition_results[split_name]["yolov8n_pipeline"]["records"]
            condition_results[split_name]["errors"] = analyze_image_errors(condition_results[split_name]["records"], iou_match_thresh=0.45)
            continue

        for det_name in detectors_order:
            print(f"[evaluate] Running {det_name} on {split_name} ({len(items)} images)...", flush=True)
            mode_arg = "yolov8" if det_name == "yolov8n_pipeline" else det_name
            preds, gts, recs, lats = run_detector_on_split(
                items,
                mode=mode_arg,
                enable_clahe=True,
                conf_threshold=float(conf_threshold),
                iou_threshold=NMS_IOU_THRESHOLD,
            )
            row, ap_c, p_c, r_c = summarize_split_metrics(
                split_name,
                det_name,
                preds,
                gts,
                lats,
                len(items),
                iou_threshold=iou_threshold,
            )
            prelim_rows.append(row)
            condition_results[split_name][det_name] = {
                "row": row,
                "ap_per_class": ap_c,
                "precision_per_class": p_c,
                "recall_per_class": r_c,
                "preds": preds,
                "gts": gts,
                "records": recs,
            }

        condition_results[split_name]["yolo_row"] = condition_results[split_name]["yolov8n_pipeline"]["row"]
        condition_results[split_name]["records"] = condition_results[split_name]["yolov8n_pipeline"]["records"]
        condition_results[split_name]["errors"] = analyze_image_errors(condition_results[split_name]["records"], iou_match_thresh=0.45)

    ablation_rows = []
    for sp_name in ["normal", "shadowed_sub100lux"]:
        preds, gts, recs, lats = run_detector_on_split(
            conditions[sp_name],
            mode="yolov8",
            enable_clahe=False,
            conf_threshold=float(conf_threshold),
            iou_threshold=NMS_IOU_THRESHOLD,
        )
        row, ap_c, p_c, r_c = summarize_split_metrics(
            sp_name,
            "yolov8n_no_clahe",
            preds,
            gts,
            lats,
            len(conditions[sp_name]),
            iou_threshold=iou_threshold,
        )
        ablation_rows.append(row)

    prelim_csv_path = os.path.join(output_dir, "preliminary_results.csv")
    fieldnames = [
        "split",
        "detector",
        "num_images",
        "num_gt",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1_score",
        "mAP50",
        "mAP50_95",
        "mean_inference_ms",
        "fps",
    ]
    with open(prelim_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prelim_rows + ablation_rows:
            writer.writerow(row)

    chart_path = os.path.join(output_dir, "preliminary_results_comparison.png")
    render_preliminary_comparison_chart(prelim_rows, chart_path)

    report_path, cm_path = save_confusion_matrix_and_report(condition_results["normal"]["errors"], output_dir)

    video_eval_dir = os.path.join(PROJECT_ROOT, "outputs", "eval_video")
    os.makedirs(video_eval_dir, exist_ok=True)
    video_eval = evaluate_demo_video_ground_truth(
        os.path.join(PROJECT_ROOT, "data", "raw_home", "demo_pantry_pan.mp4"),
        video_eval_dir,
        conf_threshold=float(conf_threshold),
    )

    test_log_rows, test_log_path = build_representative_test_log(condition_results, video_eval, output_dir)
    failure_analysis = build_failure_gallery_and_counts(condition_results, video_eval, output_dir)
    strip_path = render_pipeline_stages_strip(samples, output_dir)
    false_meat = evaluate_false_meat_rate(data_yaml, conf_threshold=float(conf_threshold))

    normal_yolo = condition_results["normal"]["yolov8n_pipeline"]
    per_class_rows = []
    for cls_id in CLASS_ORDER:
        per_class_rows.append({
            "class_id": cls_id,
            "ap50": round(float(normal_yolo["ap_per_class"].get(cls_id, 0.0)), 4),
            "precision": round(float(normal_yolo["precision_per_class"].get(cls_id, 0.0)), 4),
            "recall": round(float(normal_yolo["recall_per_class"].get(cls_id, 0.0)), 4),
        })

    hardware = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_threads": int(os.cpu_count() or 16),
    }

    summary_payload = {
        "hardware": hardware,
        "conf_threshold": float(conf_threshold),
        "nms_iou_threshold": float(NMS_IOU_THRESHOLD),
        "eval_iou_threshold": float(iou_threshold),
        "num_test_images": len(samples),
        "num_gt_boxes": len(condition_results["normal"]["yolov8n_pipeline"]["gts"]),
        "preliminary_results": prelim_rows,
        "clahe_ablation": ablation_rows,
        "per_class_normal_yolov8n": per_class_rows,
        "video_evaluation": video_eval,
        "false_meat_evaluation": false_meat,
        "failure_analysis": failure_analysis,
        "test_log_summary": {
            "total_cases": len(test_log_rows),
            "passed": sum(1 for r in test_log_rows if r["Status"] == "PASS"),
            "failed": sum(1 for r in test_log_rows if r["Status"] == "FAIL"),
        },
        "artifacts": {
            "preliminary_results_csv": prelim_csv_path,
            "preliminary_comparison_png": chart_path,
            "classification_report_txt": report_path,
            "confusion_matrix_png": cm_path,
            "test_log_csv": test_log_path,
            "failure_gallery_png": failure_analysis["failure_gallery_png"],
            "pipeline_stages_strip_png": strip_path,
        },
    }

    json_path = os.path.join(output_dir, "milestone3_evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset_yolo/data.yaml")
    parser.add_argument("--output-dir", type=str, default="docs/milestone3")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    summary = run_milestone3_evaluation(
        data_yaml=args.data,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    print(json.dumps({
        "num_test_images": summary["num_test_images"],
        "prelim_rows": len(summary["preliminary_results"]),
        "failure_causes": summary["failure_analysis"]["cause_counts"],
        "test_log_summary": summary["test_log_summary"],
        "video_eval": summary["video_evaluation"],
    }, indent=2))


if __name__ == "__main__":
    main()
