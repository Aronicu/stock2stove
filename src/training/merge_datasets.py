import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.server.config import SAMPLES_DIR, TEMPLATES_DIR, UNIFIED_CLASSES
from src.training.download_roboflow import is_valid_yolo_box

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_unified_class_names():
    names = {}
    for entry in UNIFIED_CLASSES:
        names[int(entry["id"])] = str(entry["class_id"])
    return names


def parse_yolo_label_file(label_path, num_classes=8, max_area=0.80):
    boxes = []
    label_lines = []
    if not os.path.exists(label_path):
        return boxes, label_lines

    with open(label_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_idx = int(float(parts[0]))
            except ValueError:
                continue
            if cls_idx < 0 or cls_idx >= num_classes:
                continue

            if len(parts) == 5:
                try:
                    xc = float(parts[1])
                    yc = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError:
                    continue
            else:
                coords = []
                parse_err = False
                for v in parts[1:]:
                    try:
                        coords.append(float(v))
                    except ValueError:
                        parse_err = True
                        break
                if parse_err:
                    continue
                xs = []
                ys = []
                k = 0
                while k + 1 < len(coords):
                    xs.append(coords[k])
                    ys.append(coords[k + 1])
                    k += 2
                if not xs or not ys:
                    continue
                x1 = max(0.0, min(1.0, min(xs)))
                y1 = max(0.0, min(1.0, min(ys)))
                x2 = max(0.0, min(1.0, max(xs)))
                y2 = max(0.0, min(1.0, max(ys)))
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1

            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))
            if not is_valid_yolo_box(xc, yc, bw, bh, max_area=max_area):
                continue

            boxes.append((cls_idx, xc, yc, bw, bh))
            label_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    return boxes, label_lines


def scan_roboflow_directory(source_dir, include_empty=True):
    root = Path(source_dir)
    if not root.exists():
        return []

    split_candidates = []
    for split_name in ("train", "valid", "val", "test"):
        cand = root / split_name
        if cand.exists() and cand.is_dir():
            norm_split = "valid" if split_name == "val" else split_name
            split_candidates.append((norm_split, cand))

    if not split_candidates:
        split_candidates.append(("train", root))

    source_name = root.name
    records = []
    for norm_split, split_dir in split_candidates:
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists():
            images_dir = split_dir
        if not labels_dir.exists():
            labels_dir = split_dir
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for img_path in sorted(images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            boxes, label_lines = parse_yolo_label_file(str(lbl_path))
            is_negative = False
            if not boxes:
                if not include_empty:
                    continue
                is_negative = True
            cls_set_raw = set()
            for b in boxes:
                cls_set_raw.add(b[0])
            cls_set = sorted(list(cls_set_raw))
            records.append({
                "image_path": str(img_path),
                "label_path": str(lbl_path),
                "source": source_name,
                "orig_split": norm_split,
                "classes": cls_set,
                "boxes": boxes,
                "label_lines": label_lines,
                "is_negative": is_negative,
            })

    return records


def collect_existing_roboflow_samples(source_dirs, max_per_class=80, include_empty=True):
    if isinstance(source_dirs, (str, os.PathLike)):
        dir_list = [str(source_dirs)]
    else:
        dir_list = []
        for d in source_dirs:
            dir_list.append(str(d))

    all_candidates = []
    for src_dir in dir_list:
        all_candidates.extend(scan_roboflow_directory(src_dir, include_empty=include_empty))

    if not all_candidates:
        return []

    class_freq = {}
    for i in range(8):
        class_freq[i] = 0
    for cand in all_candidates:
        for cls_idx in cand["classes"]:
            class_freq[cls_idx] += 1

    selected = []
    selected_paths = set()
    selected_class_counts = {}
    for i in range(8):
        selected_class_counts[i] = 0

    for cand in all_candidates:
        if cand["is_negative"]:
            selected.append(cand)
            selected_paths.add(cand["image_path"])

    non_empty_classes = []
    for c in range(8):
        if class_freq[c] > 0:
            non_empty_classes.append(c)
    ordered_classes = sorted(non_empty_classes, key=lambda c: (class_freq[c], c))

    for cls_idx in ordered_classes:
        while selected_class_counts[cls_idx] < int(max_per_class):
            best_cand = None
            best_score = None
            for cand in all_candidates:
                if cand["image_path"] in selected_paths:
                    continue
                if cls_idx not in cand["classes"]:
                    continue
                under_cap_gain = 0
                over_cap_penalty = 0
                for c_id in cand["classes"]:
                    if selected_class_counts[c_id] < int(max_per_class):
                        under_cap_gain += 1
                    else:
                        over_cap_penalty += 1
                score = (under_cap_gain, -over_cap_penalty, -len(cand["boxes"]))
                if best_score is None or score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand is None:
                break
            selected.append(best_cand)
            selected_paths.add(best_cand["image_path"])
            for c_id in best_cand["classes"]:
                selected_class_counts[c_id] += 1

    return selected


def load_template_paths_by_class(templates_dir):
    class_names = get_unified_class_names()
    templates_by_cls = {}
    for i in range(8):
        templates_by_cls[i] = []

    for cls_idx, cls_name in class_names.items():
        cls_dir = Path(templates_dir) / cls_name
        if not cls_dir.exists() or not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                templates_by_cls[cls_idx].append(str(p))

    return templates_by_cls


def load_background_paths(samples_dir):
    samples_root = Path(samples_dir)
    if not samples_root.exists():
        return []
    negative_real = []
    fallback = []
    for p in sorted(samples_root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            name = p.name.lower()
            if name.startswith("pantry_images_cv_"):
                idx_part = name.replace("pantry_images_cv_", "").split(".")[0]
                if idx_part.isdigit() and int(idx_part) >= 8:
                    negative_real.append(str(p))
            elif not name.startswith("shelf_sample_"):
                fallback.append(str(p))
    if negative_real:
        return negative_real
    return fallback


def boxes_overlap_with_margin(box_a, box_b, margin_px=6):
    if box_a[2] + margin_px <= box_b[0]:
        return False
    if box_b[2] + margin_px <= box_a[0]:
        return False
    if box_a[3] + margin_px <= box_b[1]:
        return False
    if box_b[3] + margin_px <= box_a[1]:
        return False
    return True


def prepare_rotated_template(template_bgr, target_scale, frame_size, angle_deg, brightness_factor):
    th, tw = template_bgr.shape[:2]
    target_max = float(target_scale) * float(frame_size)
    ratio = target_max / float(max(1, max(th, tw)))
    new_w = max(24, int(round(float(tw) * ratio)))
    new_h = max(24, int(round(float(th) * ratio)))

    resized = cv2.resize(template_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    bright_img = np.clip(resized.astype(np.float32) * float(brightness_factor), 0.0, 255.0).astype(np.uint8)

    mask = np.full((new_h, new_w), 255, dtype=np.uint8)
    cx = float(new_w) / 2.0
    cy = float(new_h) / 2.0
    rot_mat = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
    cos_a = abs(float(rot_mat[0, 0]))
    sin_a = abs(float(rot_mat[0, 1]))
    rot_w = max(1, int(math.ceil(float(new_h) * sin_a + float(new_w) * cos_a)))
    rot_h = max(1, int(math.ceil(float(new_h) * cos_a + float(new_w) * sin_a)))
    rot_mat[0, 2] += (float(rot_w) / 2.0) - cx
    rot_mat[1, 2] += (float(rot_h) / 2.0) - cy

    rot_img = cv2.warpAffine(
        bright_img,
        rot_mat,
        (rot_w, rot_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )
    rot_mask = cv2.warpAffine(
        mask,
        rot_mat,
        (rot_w, rot_h),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return rot_img, rot_mask


def find_non_overlapping_origin(rot_w, rot_h, mask_ys, mask_xs, placed_boxes, frame_size, rng):
    max_x = int(frame_size) - int(rot_w)
    max_y = int(frame_size) - int(rot_h)
    if max_x < 0 or max_y < 0 or len(mask_xs) == 0 or len(mask_ys) == 0:
        return None

    rel_x1 = int(mask_xs.min())
    rel_y1 = int(mask_ys.min())
    rel_x2 = int(mask_xs.max()) + 1
    rel_y2 = int(mask_ys.max()) + 1

    for _ in range(90):
        px = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        py = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        cand_box = (px + rel_x1, py + rel_y1, px + rel_x2, py + rel_y2)
        conflict = False
        for existing in placed_boxes:
            if boxes_overlap_with_margin(cand_box, existing, margin_px=6):
                conflict = True
                break
        if not conflict:
            return px, py, cand_box

    for py in range(0, max_y + 1, 16):
        for px in range(0, max_x + 1, 16):
            cand_box = (px + rel_x1, py + rel_y1, px + rel_x2, py + rel_y2)
            conflict = False
            for existing in placed_boxes:
                if boxes_overlap_with_margin(cand_box, existing, margin_px=4):
                    conflict = True
                    break
            if not conflict:
                return px, py, cand_box

    return None


def composite_single_scene(bg_path, item_specs, templates_cache, frame_size, rng, apply_cutout=False, scene_idx=0):
    bg_img = cv2.imread(bg_path)
    if bg_img is None or bg_img.size == 0:
        canvas = np.full((frame_size, frame_size, 3), 120, dtype=np.uint8)
    else:
        canvas = cv2.resize(bg_img, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        if float(rng.random()) < 0.5:
            canvas = cv2.flip(canvas, 1)
        bg_bright = float(rng.uniform(0.85, 1.12))
        canvas = np.clip(canvas.astype(np.float32) * bg_bright, 0.0, 255.0).astype(np.uint8)

    placed_boxes = []
    yolo_labels = []

    for item_pos, (cls_idx, tmpl_path) in enumerate(item_specs):
        tmpl_bgr = templates_cache.get(tmpl_path)
        if tmpl_bgr is None:
            continue

        base_scale = float(rng.uniform(0.22, 0.40))
        angle_deg = float(rng.uniform(-8.0, 8.0))
        bright_val = float(rng.uniform(0.80, 1.15))

        placed = None
        for shrink_factor in (1.0, 0.85, 0.72, 0.60):
            try_scale = max(0.20, base_scale * shrink_factor) if shrink_factor < 1.0 else base_scale
            if shrink_factor <= 0.72:
                try_scale = max(0.20, min(try_scale, 0.28))
            rot_img, rot_mask = prepare_rotated_template(
                tmpl_bgr,
                try_scale,
                frame_size,
                angle_deg,
                bright_val,
            )
            rot_h, rot_w = rot_mask.shape[:2]
            ys, xs = np.where(rot_mask > 0)
            origin = find_non_overlapping_origin(
                rot_w,
                rot_h,
                ys,
                xs,
                placed_boxes,
                frame_size,
                rng,
            )
            if origin is not None:
                placed = (origin[0], origin[1], origin[2], rot_img, rot_mask)
                break

        if placed is None:
            continue

        px, py, cand_box, rot_img, rot_mask = placed
        rot_h, rot_w = rot_mask.shape[:2]
        roi = canvas[py : py + rot_h, px : px + rot_w].astype(np.float32)
        feathered = cv2.GaussianBlur(rot_mask, (7, 7), 2.0).astype(np.float32) / 255.0
        alpha = feathered[:, :, None]
        blended = (rot_img.astype(np.float32) * alpha) + (roi * (1.0 - alpha))
        canvas[py : py + rot_h, px : px + rot_w] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        placed_boxes.append(cand_box)

        bx1, by1, bx2, by2 = cand_box
        if apply_cutout and ((scene_idx + item_pos) % 5 < 2):
            bw_px = max(1, bx2 - bx1)
            bh_px = max(1, by2 - by1)
            occ_frac = float(rng.uniform(0.30, 0.35))
            patch_w = max(6, min(bw_px, int(round(float(bw_px) * 0.72))))
            patch_h = max(6, min(int(round(float(bh_px) * 0.65)), int(round(float(bh_px) * (occ_frac / 0.72)))))
            cx_px = (bx1 + bx2) // 2
            cy_px = (by1 + by2) // 2
            ox1 = max(bx1, cx_px - patch_w // 2)
            ox2 = min(bx2, ox1 + patch_w)
            oy1 = max(by1, cy_px - patch_h // 2)
            oy2 = min(by2, oy1 + patch_h)
            canvas[oy1:oy2, ox1:ox2] = (20, 20, 24)

        xc = ((float(bx1) + float(bx2)) / 2.0) / float(frame_size)
        yc = ((float(by1) + float(by2)) / 2.0) / float(frame_size)
        bw = float(bx2 - bx1) / float(frame_size)
        bh = float(by2 - by1) / float(frame_size)
        yolo_labels.append((int(cls_idx), xc, yc, bw, bh))

    return canvas, yolo_labels


def build_balanced_class_schedule(total_items, available_classes, rng):
    if not available_classes:
        return []
    schedule = []
    while len(schedule) < total_items:
        block = list(available_classes)
        rng.shuffle(block)
        schedule.extend(block)
    return schedule[:total_items]


def generate_balanced_shelf_composites(
    templates_dir,
    samples_dir,
    output_dir,
    num_train=60,
    num_val=0,
    num_test=0,
):
    frame_size = 640
    rng = np.random.default_rng(42)

    templates_by_cls = load_template_paths_by_class(templates_dir)
    available_classes = []
    for c in range(8):
        if len(templates_by_cls[c]) > 0:
            available_classes.append(c)
    if not available_classes:
        raise FileNotFoundError(f"No reference templates found under {templates_dir}")

    templates_cache = {}
    for cls_idx in available_classes:
        for path in templates_by_cls[cls_idx]:
            img = cv2.imread(path)
            if img is not None and img.size > 0:
                templates_cache[path] = img

    bg_paths = load_background_paths(samples_dir)
    if not bg_paths:
        bg_splits = {"train": [], "valid": [], "test": []}
    elif len(bg_paths) >= 10:
        train_end = max(1, int(round(0.70 * len(bg_paths))))
        val_end = max(train_end + 1, int(round(0.85 * len(bg_paths))))
        bg_splits = {
            "train": bg_paths[:train_end],
            "valid": bg_paths[train_end:val_end],
            "test": bg_paths[val_end:],
        }
        if not bg_splits["test"]:
            bg_splits["test"] = bg_paths[-2:]
    else:
        bg_splits = {
            "train": list(bg_paths),
            "valid": list(bg_paths),
            "test": list(bg_paths),
        }

    split_targets = [
        ("train", int(num_train)),
        ("valid", int(num_val)),
        ("test", int(num_test)),
    ]

    class_instance_counts = {}
    for i in range(8):
        class_instance_counts[i] = 0
    split_image_counts = {"train": 0, "valid": 0, "test": 0}
    total_instances = 0

    count_pattern = [3, 4, 2, 3, 4, 3]
    out_root = Path(output_dir)

    for split_name, target_num in split_targets:
        images_dir = out_root / split_name / "images"
        labels_dir = out_root / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        if target_num <= 0:
            continue

        counts_per_scene = []
        for idx in range(target_num):
            counts_per_scene.append(count_pattern[idx % len(count_pattern)])
        rng.shuffle(counts_per_scene)

        total_split_items = sum(counts_per_scene)
        class_schedule = build_balanced_class_schedule(total_split_items, available_classes, rng)

        cursor = 0
        split_bgs = bg_splits[split_name]
        for scene_idx in range(target_num):
            n_items = counts_per_scene[scene_idx]
            scene_classes = class_schedule[cursor : cursor + n_items]
            cursor += n_items

            item_specs = []
            for cls_idx in scene_classes:
                tmpl_choices = templates_by_cls[cls_idx]
                chosen_tmpl = tmpl_choices[int(rng.integers(0, len(tmpl_choices)))]
                item_specs.append((cls_idx, chosen_tmpl))

            chosen_bg = split_bgs[scene_idx % len(split_bgs)] if split_bgs else ""
            canvas, yolo_labels = composite_single_scene(
                chosen_bg,
                item_specs,
                templates_cache,
                frame_size,
                rng,
                apply_cutout=(split_name == "train"),
                scene_idx=scene_idx,
            )

            stem = f"composite_{split_name}_{scene_idx:04d}"
            img_out = images_dir / f"{stem}.jpg"
            lbl_out = labels_dir / f"{stem}.txt"

            cv2.imwrite(str(img_out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(lbl_out, "w", encoding="utf-8") as f:
                for cls_idx, xc, yc, bw, bh in yolo_labels:
                    f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    class_instance_counts[cls_idx] += 1
                    total_instances += 1

            split_image_counts[split_name] += 1

    return {
        "split_image_counts": split_image_counts,
        "class_instance_counts": class_instance_counts,
        "total_images": sum(split_image_counts.values()),
        "total_instances": total_instances,
    }


def write_dataset_yaml_files(output_dir):
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    names = get_unified_class_names()

    yaml_payload = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }

    data_yaml_path = out_root / "data.yaml"
    pantry_fridge_yaml_path = out_root / "pantry_fridge.yaml"

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_payload, f, sort_keys=False)

    with open(pantry_fridge_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_payload, f, sort_keys=False)

    return str(data_yaml_path), str(pantry_fridge_yaml_path)


def count_dataset_statistics(dataset_dir):
    root = Path(dataset_dir)
    split_image_counts = {}
    class_counts = {}
    for i in range(8):
        class_counts[i] = 0
    per_split_class_counts = {}
    negative_image_counts = {}

    for split_name in ("train", "valid", "test"):
        images_dir = root / split_name / "images"
        labels_dir = root / split_name / "labels"
        img_count = 0
        if images_dir.exists():
            for p in images_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    img_count += 1
        split_image_counts[split_name] = img_count

        split_cls = {}
        for i in range(8):
            split_cls[i] = 0
        neg_count = 0
        if labels_dir.exists():
            for lp in labels_dir.glob("*.txt"):
                boxes, _ = parse_yolo_label_file(str(lp))
                if not boxes:
                    neg_count += 1
                for b in boxes:
                    cls_idx = b[0]
                    class_counts[cls_idx] += 1
                    split_cls[cls_idx] += 1
        per_split_class_counts[split_name] = split_cls
        negative_image_counts[split_name] = neg_count

    return {
        "split_image_counts": split_image_counts,
        "class_instance_counts": class_counts,
        "per_split_class_counts": per_split_class_counts,
        "negative_image_counts": negative_image_counts,
        "total_images": sum(split_image_counts.values()),
        "total_instances": sum(class_counts.values()),
    }


def extract_templates_from_datasets(source_dirs, templates_dir=TEMPLATES_DIR, samples_dir=SAMPLES_DIR, min_per_class=10):
    tmpl_root = Path(templates_dir)
    if tmpl_root.exists():
        shutil.rmtree(tmpl_root)

    class_names = get_unified_class_names()
    for cls_idx in range(8):
        (tmpl_root / class_names[cls_idx]).mkdir(parents=True, exist_ok=True)

    saved_per_cls = {}
    for cls_idx in range(8):
        saved_per_cls[cls_idx] = 0

    priority_sample = Path(samples_dir) / "pantry_images_cv_01.jpg"
    if priority_sample.exists():
        img01 = cv2.imread(str(priority_sample))
        if img01 is not None and img01.size > 0:
            h01, w01 = img01.shape[:2]
            seed_crops = [
                (0, 0.09, 0.20, 0.38, 0.47),
                (1, 0.36, 0.19, 0.65, 0.47),
                (3, 0.60, 0.18, 0.90, 0.46),
            ]
            for cls_idx, x1f, y1f, x2f, y2f in seed_crops:
                px1 = max(0, int(round(x1f * w01)))
                py1 = max(0, int(round(y1f * h01)))
                px2 = min(w01, int(round(x2f * w01)))
                py2 = min(h01, int(round(y2f * h01)))
                if px2 - px1 >= 16 and py2 - py1 >= 16:
                    crop = img01[py1:py2, px1:px2]
                    ch, cw = crop.shape[:2]
                    if cw < 220 or ch < 220:
                        scale_up = max(220.0 / float(cw), 220.0 / float(ch))
                        crop = cv2.resize(
                            crop,
                            (int(round(cw * scale_up)), int(round(ch * scale_up))),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    idx_num = saved_per_cls[cls_idx]
                    out_p = tmpl_root / class_names[cls_idx] / f"template_{idx_num:03d}.jpg"
                    cv2.imwrite(str(out_p), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_per_cls[cls_idx] += 1

    all_records = []
    for src_dir in source_dirs:
        all_records.extend(scan_roboflow_directory(src_dir, include_empty=False))

    for rec in all_records:
        if rec["orig_split"] != "train":
            continue
        all_done = True
        for cls_idx in range(8):
            if saved_per_cls[cls_idx] < int(min_per_class):
                all_done = False
                break
        if all_done:
            break

        img = cv2.imread(rec["image_path"])
        if img is None or img.size == 0:
            continue
        ih, iw = img.shape[:2]

        for cls_idx, xc, yc, bw, bh in rec["boxes"]:
            if saved_per_cls[cls_idx] >= int(min_per_class):
                continue
            x1 = max(0, int(round((xc - bw / 2.0) * float(iw))))
            y1 = max(0, int(round((yc - bh / 2.0) * float(ih))))
            x2 = min(iw, int(round((xc + bw / 2.0) * float(iw))))
            y2 = min(ih, int(round((yc + bh / 2.0) * float(ih))))
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            crop = img[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            if cw < 160 or ch < 160:
                scale_up = max(160.0 / float(cw), 160.0 / float(ch))
                crop = cv2.resize(
                    crop,
                    (int(round(cw * scale_up)), int(round(ch * scale_up))),
                    interpolation=cv2.INTER_CUBIC,
                )
            idx_num = saved_per_cls[cls_idx]
            out_p = tmpl_root / class_names[cls_idx] / f"template_{idx_num:03d}.jpg"
            cv2.imwrite(str(out_p), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_per_cls[cls_idx] += 1

    return saved_per_cls


def generate_negative_samples(samples_dir, output_dir, train_count, val_count, test_count):
    bg_paths = load_background_paths(samples_dir)
    if not bg_paths:
        return {"train": 0, "valid": 0, "test": 0}

    n_bg = len(bg_paths)
    train_end = max(1, min(n_bg - 2, int(round(0.70 * n_bg)))) if n_bg >= 3 else n_bg
    val_end = max(train_end + 1, min(n_bg - 1, int(round(0.85 * n_bg)))) if n_bg >= 3 else n_bg

    split_bgs = {
        "train": bg_paths[:train_end],
        "valid": bg_paths[train_end:val_end] if val_end > train_end else bg_paths[-1:],
        "test": bg_paths[val_end:] if val_end < n_bg else bg_paths[-1:],
    }
    split_targets = {
        "train": int(train_count),
        "valid": int(val_count),
        "test": int(test_count),
    }

    rng = np.random.default_rng(77)
    out_root = Path(output_dir)
    saved = {"train": 0, "valid": 0, "test": 0}

    for split_name in ("train", "valid", "test"):
        target = split_targets[split_name]
        bgs = split_bgs[split_name]
        images_dst = out_root / split_name / "images"
        labels_dst = out_root / split_name / "labels"
        images_dst.mkdir(parents=True, exist_ok=True)
        labels_dst.mkdir(parents=True, exist_ok=True)

        for idx in range(target):
            src_bg = bgs[idx % len(bgs)]
            bgr = cv2.imread(src_bg)
            if bgr is None or bgr.size == 0:
                continue
            resized = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_CUBIC)
            variant = idx // len(bgs)
            if variant % 3 == 1:
                resized = cv2.flip(resized, 1)
            elif variant % 3 == 2:
                crop = resized[40:600, 40:600]
                resized = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)
            bright = float(rng.uniform(0.82, 1.15))
            resized = np.clip(resized.astype(np.float32) * bright, 0.0, 255.0).astype(np.uint8)

            stem = f"neg_{split_name}_{idx:04d}"
            cv2.imwrite(str(images_dst / f"{stem}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(labels_dst / f"{stem}.txt", "w", encoding="utf-8") as f:
                f.write("")
            saved[split_name] += 1

    return saved


def populate_pantry_negatives_eval_dir(samples_dir, output_dir):
    neg_eval_dir = Path(output_dir) / "pantry_negatives"
    if neg_eval_dir.exists():
        shutil.rmtree(neg_eval_dir)
    neg_eval_dir.mkdir(parents=True, exist_ok=True)

    samples_root = Path(samples_dir)
    if samples_root.exists():
        for p in sorted(samples_root.glob("pantry_images_cv_*.jpg")):
            shutil.copy2(p, neg_eval_dir / p.name)

    test_images_dir = Path(output_dir) / "test" / "images"
    test_labels_dir = Path(output_dir) / "test" / "labels"
    if test_images_dir.exists() and test_labels_dir.exists():
        for img_path in sorted(test_images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            lbl_path = test_labels_dir / f"{img_path.stem}.txt"
            boxes, _ = parse_yolo_label_file(str(lbl_path))
            has_meat = False
            for b in boxes:
                if b[0] == 2:
                    has_meat = True
                    break
            if not has_meat:
                shutil.copy2(img_path, neg_eval_dir / img_path.name)


def copy_records_to_split(records, images_dst, labels_dst):
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for idx, rec in enumerate(records):
        src_img = Path(rec["image_path"])
        stem = f"rf_{rec['source']}_{idx:04d}_{src_img.stem}"
        dst_img = images_dst / f"{stem}{src_img.suffix.lower()}"
        dst_lbl = labels_dst / f"{stem}.txt"
        shutil.copy2(src_img, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.writelines(rec["label_lines"])
        copied += 1
    return copied


def balance_source_caps(records, max_fraction=0.40):
    if not records:
        return []
    by_source = {}
    for rec in records:
        src = rec["source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(rec)

    if len(by_source) <= 1:
        return list(records)

    total_est = len(records)
    cap = max(10, int(math.floor(float(max_fraction) * float(total_est))))
    kept = []
    for src in sorted(by_source.keys()):
        src_recs = by_source[src]
        if len(src_recs) <= cap:
            kept.extend(src_recs)
        else:
            step = float(len(src_recs)) / float(cap)
            picked = []
            for k in range(cap):
                idx = min(len(src_recs) - 1, int(math.floor(float(k) * step)))
                picked.append(src_recs[idx])
            kept.extend(picked)
    return kept


def build_unified_dataset(output_dir="data/dataset_yolo"):
    out_root = Path(output_dir)
    for split_name in ("train", "valid", "test"):
        split_path = out_root / split_name
        if split_path.exists():
            shutil.rmtree(split_path)
    for cache_file in out_root.glob("*.cache"):
        cache_file.unlink()
    for split_name in ("train", "valid", "test"):
        (out_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

    source_dirs = [
        os.path.join(PROJECT_ROOT, "data", "raw_rf_frieburg"),
        os.path.join(PROJECT_ROOT, "data", "raw_rf_kitchen_yolo"),
        os.path.join(PROJECT_ROOT, "data", "raw_roboflow_fridge"),
        os.path.join(PROJECT_ROOT, "data", "raw_rf_pantry_samples"),
    ]
    for extra_name in ("raw_rf_yolo_pantry", "raw_rf_smart_pantry"):
        extra_path = os.path.join(PROJECT_ROOT, "data", extra_name)
        if os.path.isdir(extra_path):
            source_dirs.append(extra_path)

    all_real_records = []
    for src_dir in source_dirs:
        if os.path.isdir(src_dir):
            all_real_records.extend(scan_roboflow_directory(src_dir, include_empty=True))

    pantry_produce_dir = os.path.join(PROJECT_ROOT, "data", "raw_roboflow_pantry")
    if os.path.isdir(pantry_produce_dir):
        produce_recs = scan_roboflow_directory(pantry_produce_dir, include_empty=False)
        split_limits = {"train": 14, "valid": 3, "test": 3}
        split_seen = {"train": 0, "valid": 0, "test": 0}
        for rec in produce_recs:
            sp = rec["orig_split"]
            if sp not in split_limits:
                sp = "train"
            if split_seen[sp] < split_limits[sp]:
                all_real_records.append(rec)
                split_seen[sp] += 1

    split_records = {"train": [], "valid": [], "test": []}
    for rec in all_real_records:
        sp = rec["orig_split"]
        if sp not in split_records:
            sp = "train"
        split_records[sp].append(rec)

    split_records["train"] = balance_source_caps(split_records["train"], max_fraction=0.40)

    extract_templates_from_datasets(
        source_dirs=source_dirs,
        templates_dir=TEMPLATES_DIR,
        samples_dir=SAMPLES_DIR,
        min_per_class=10,
    )

    real_train_count = len(split_records["train"])
    comp_train_count = min(65, max(20, int(math.floor(0.25 * float(real_train_count)))))

    generate_balanced_shelf_composites(
        templates_dir=TEMPLATES_DIR,
        samples_dir=SAMPLES_DIR,
        output_dir=str(out_root),
        num_train=comp_train_count,
        num_val=0,
        num_test=0,
    )

    for split_name in ("train", "valid", "test"):
        copy_records_to_split(
            split_records[split_name],
            out_root / split_name / "images",
            out_root / split_name / "labels",
        )

    neg_train = max(12, int(round(0.14 * float(real_train_count + comp_train_count))))
    neg_val = max(4, int(round(0.14 * float(len(split_records["valid"])))))
    neg_test = max(4, int(round(0.14 * float(len(split_records["test"])))))
    generate_negative_samples(SAMPLES_DIR, str(out_root), neg_train, neg_val, neg_test)

    populate_pantry_negatives_eval_dir(SAMPLES_DIR, str(out_root))

    data_yaml_path, pantry_fridge_yaml_path = write_dataset_yaml_files(str(out_root))
    stats = count_dataset_statistics(str(out_root))
    stats["data_yaml"] = data_yaml_path
    stats["pantry_fridge_yaml"] = pantry_fridge_yaml_path
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/dataset_yolo")
    args = parser.parse_args()

    stats = build_unified_dataset(output_dir=args.output_dir)
    print(f"Unified dataset written to {args.output_dir}")
    print(f"Split image counts: {stats['split_image_counts']} (total={stats['total_images']})")
    print(f"Negative image counts: {stats['negative_image_counts']}")
    print(f"Class instance counts: {stats['class_instance_counts']} (total={stats['total_instances']})")
    print(f"Per-split class counts: {stats['per_split_class_counts']}")
    print(f"YAML files: {stats['data_yaml']}, {stats['pantry_fridge_yaml']}")


if __name__ == "__main__":
    main()
