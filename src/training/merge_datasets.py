import argparse
import glob
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_unified_class_names():
    names = {}
    for entry in UNIFIED_CLASSES:
        names[int(entry["id"])] = str(entry["class_id"])
    return names


def parse_yolo_label_file(label_path, num_classes=8):
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
                try:
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                xs = [coords[k] for k in range(0, len(coords) - 1, 2)]
                ys = [coords[k + 1] for k in range(0, len(coords) - 1, 2)]
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
            if bw <= 0.001 or bh <= 0.001:
                continue

            boxes.append((cls_idx, xc, yc, bw, bh))
            label_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    return boxes, label_lines


def scan_roboflow_directory(source_dir):
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
            boxes, label_lines = parse_yolo_label_file(str(lbl_path))
            if not boxes:
                continue
            cls_set = sorted(list({b[0] for b in boxes}))
            records.append({
                "image_path": str(img_path),
                "label_path": str(lbl_path),
                "source": source_name,
                "orig_split": norm_split,
                "classes": cls_set,
                "boxes": boxes,
                "label_lines": label_lines,
            })

    return records


def collect_existing_roboflow_samples(source_dirs, max_per_class=45):
    if isinstance(source_dirs, (str, os.PathLike)):
        dir_list = [str(source_dirs)]
    else:
        dir_list = [str(d) for d in source_dirs]

    all_candidates = []
    for src_dir in dir_list:
        all_candidates.extend(scan_roboflow_directory(src_dir))

    if not all_candidates:
        return []

    class_freq = {i: 0 for i in range(8)}
    for cand in all_candidates:
        for cls_idx in cand["classes"]:
            class_freq[cls_idx] += 1

    selected = []
    selected_paths = set()
    selected_class_counts = {i: 0 for i in range(8)}

    sources_seen = []
    for cand in all_candidates:
        if cand["source"] not in sources_seen:
            sources_seen.append(cand["source"])

    pantry_quota = min(15, max(3, int(max_per_class) // 3))
    for src_name in sources_seen:
        src_cands = [c for c in all_candidates if c["source"] == src_name]
        has_rare = False
        for c in src_cands:
            for cls_idx in c["classes"]:
                if cls_idx != 7:
                    has_rare = True
                    break
            if has_rare:
                break
        if not has_rare and src_cands:
            per_split_targets = {"train": pantry_quota - 4, "valid": 2, "test": 2}
            for split_name in ("train", "valid", "test"):
                split_cands = [c for c in src_cands if c["orig_split"] == split_name]
                limit = per_split_targets.get(split_name, 2)
                for cand in split_cands[:limit]:
                    if cand["image_path"] in selected_paths:
                        continue
                    selected.append(cand)
                    selected_paths.add(cand["image_path"])
                    for cls_idx in cand["classes"]:
                        selected_class_counts[cls_idx] += 1

    ordered_classes = sorted(
        [c for c in range(8) if class_freq[c] > 0],
        key=lambda c: (class_freq[c], c),
    )

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
    templates_by_cls = {i: [] for i in range(8)}

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
    paths = []
    for p in sorted(samples_root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(str(p))
    return paths


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
        canvas = cv2.GaussianBlur(canvas, (31, 31), 10)

    placed_boxes = []
    yolo_labels = []

    for item_pos, (cls_idx, tmpl_path) in enumerate(item_specs):
        tmpl_bgr = templates_cache.get(tmpl_path)
        if tmpl_bgr is None:
            continue

        base_scale = float(rng.uniform(0.22, 0.42))
        angle_deg = float(rng.uniform(-8.0, 8.0))
        bright_val = float(rng.uniform(0.80, 1.15))

        placed = None
        for shrink_factor in (1.0, 0.85, 0.72, 0.60):
            try_scale = max(0.22, base_scale * shrink_factor) if shrink_factor < 1.0 else base_scale
            if shrink_factor <= 0.72:
                try_scale = max(0.22, min(try_scale, 0.28))
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
        roi = canvas[py : py + rot_h, px : px + rot_w]
        mask_bool = rot_mask > 0
        roi[mask_bool] = rot_img[mask_bool]
        canvas[py : py + rot_h, px : px + rot_w] = roi
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
    num_train=140,
    num_val=36,
    num_test=36,
):
    frame_size = 640
    rng = np.random.default_rng(42)

    templates_by_cls = load_template_paths_by_class(templates_dir)
    available_classes = [c for c in range(8) if len(templates_by_cls[c]) > 0]
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
        raise FileNotFoundError(f"No background sample images found under {samples_dir}")

    n_bg = len(bg_paths)
    if n_bg >= 12:
        train_end = max(1, int(round(0.70 * n_bg)))
        val_end = max(train_end + 1, int(round(0.85 * n_bg)))
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

    class_instance_counts = {i: 0 for i in range(8)}
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

            chosen_bg = split_bgs[scene_idx % len(split_bgs)]
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


def allocate_roboflow_to_splits(samples, target_counts):
    pantry_by_split = {"train": [], "valid": [], "test": []}
    fridge_pool = []

    for sample in samples:
        has_non_produce = any(c != 7 for c in sample["classes"])
        if has_non_produce:
            fridge_pool.append(sample)
        else:
            orig_s = sample.get("orig_split", "train")
            if orig_s not in pantry_by_split:
                orig_s = "train"
            pantry_by_split[orig_s].append(sample)

    allocated = {"train": [], "valid": [], "test": []}
    pantry_caps = {"train": 12, "valid": 3, "test": 2}
    for split_name in ("train", "valid", "test"):
        cap = min(pantry_caps[split_name], target_counts[split_name])
        allocated[split_name].extend(pantry_by_split[split_name][:cap])

    fridge_sorted = sorted(
        fridge_pool,
        key=lambda s: (min(s["classes"]), len(s["classes"]), s["image_path"]),
    )
    need_val = max(0, target_counts["valid"] - len(allocated["valid"]))
    need_test = max(0, target_counts["test"] - len(allocated["test"]))
    need_train = max(0, target_counts["train"] - len(allocated["train"]))

    rng = np.random.default_rng(123)
    indices = list(range(len(fridge_sorted)))
    rng.shuffle(indices)

    val_picked = []
    test_picked = []
    train_picked = []
    for pos, idx in enumerate(indices):
        item = fridge_sorted[idx]
        if len(val_picked) < need_val and pos % 7 == 1:
            val_picked.append(item)
        elif len(test_picked) < need_test and pos % 11 == 3:
            test_picked.append(item)
        elif len(train_picked) < need_train:
            train_picked.append(item)

    remaining = [
        fridge_sorted[idx]
        for idx in indices
        if fridge_sorted[idx] not in val_picked
        and fridge_sorted[idx] not in test_picked
        and fridge_sorted[idx] not in train_picked
    ]

    for item in remaining:
        if len(val_picked) < need_val:
            val_picked.append(item)
        elif len(test_picked) < need_test:
            test_picked.append(item)
        elif len(train_picked) < need_train:
            train_picked.append(item)

    allocated["valid"].extend(val_picked)
    allocated["test"].extend(test_picked)
    allocated["train"].extend(train_picked)
    return allocated


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
    class_counts = {i: 0 for i in range(8)}
    per_split_class_counts = {}

    for split_name in ("train", "valid", "test"):
        images_dir = root / split_name / "images"
        labels_dir = root / split_name / "labels"
        img_count = 0
        if images_dir.exists():
            for p in images_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    img_count += 1
        split_image_counts[split_name] = img_count

        split_cls = {i: 0 for i in range(8)}
        if labels_dir.exists():
            for lp in labels_dir.glob("*.txt"):
                boxes, _ = parse_yolo_label_file(str(lp))
                for b in boxes:
                    cls_idx = b[0]
                    class_counts[cls_idx] += 1
                    split_cls[cls_idx] += 1
        per_split_class_counts[split_name] = split_cls

    return {
        "split_image_counts": split_image_counts,
        "class_instance_counts": class_counts,
        "per_split_class_counts": per_split_class_counts,
        "total_images": sum(split_image_counts.values()),
        "total_instances": sum(class_counts.values()),
    }


def collect_auto_labeled_samples(auto_labeled_dir):
    root = Path(auto_labeled_dir)
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return []

    samples = []
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        boxes, label_lines = parse_yolo_label_file(str(lbl_path))
        if not boxes:
            continue
        samples.append({
            "image_path": str(img_path),
            "label_path": str(lbl_path),
            "label_lines": label_lines,
        })
    return samples


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

    auto_dir = os.path.join(PROJECT_ROOT, "data", "auto_labeled")
    auto_samples = collect_auto_labeled_samples(auto_dir)
    num_auto = len(auto_samples)
    comp_train = 140 if num_auto == 0 else max(90, 155 - num_auto)
    rf_train_target = 60 if num_auto == 0 else max(30, 200 - comp_train - num_auto)

    generate_balanced_shelf_composites(
        templates_dir=TEMPLATES_DIR,
        samples_dir=SAMPLES_DIR,
        output_dir=str(out_root),
        num_train=comp_train,
        num_val=36,
        num_test=36,
    )

    source_dirs = [
        os.path.join(PROJECT_ROOT, "data", "raw_roboflow_pantry"),
        os.path.join(PROJECT_ROOT, "data", "raw_roboflow_fridge"),
    ]
    rf_samples = collect_existing_roboflow_samples(source_dirs, max_per_class=45)
    allocated = allocate_roboflow_to_splits(
        rf_samples,
        target_counts={"train": rf_train_target, "valid": 9, "test": 4},
    )

    for split_name in ("train", "valid", "test"):
        images_dst = out_root / split_name / "images"
        labels_dst = out_root / split_name / "labels"
        for idx, sample in enumerate(allocated[split_name]):
            src_img = Path(sample["image_path"])
            stem = f"rf_{sample['source']}_{idx:04d}_{src_img.stem}"
            dst_img = images_dst / f"{stem}{src_img.suffix.lower()}"
            dst_lbl = labels_dst / f"{stem}.txt"
            shutil.copy2(src_img, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.writelines(sample["label_lines"])

    train_images_dst = out_root / "train" / "images"
    train_labels_dst = out_root / "train" / "labels"
    for idx, sample in enumerate(auto_samples):
        src_img = Path(sample["image_path"])
        stem = f"autolabel_{idx:04d}_{src_img.stem}"
        dst_img = train_images_dst / f"{stem}{src_img.suffix.lower()}"
        dst_lbl = train_labels_dst / f"{stem}.txt"
        shutil.copy2(src_img, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.writelines(sample["label_lines"])

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
    print(f"Class instance counts: {stats['class_instance_counts']} (total={stats['total_instances']})")
    print(f"YAML files: {stats['data_yaml']}, {stats['pantry_fridge_yaml']}")


if __name__ == "__main__":
    main()
