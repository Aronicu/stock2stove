import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

TARGET_CLASSES = [
    "pasta_noodles",
    "canned_tomato",
    "canned_tuna_meat",
    "cereal_grains",
    "coffee_condiment",
    "milk_dairy",
    "eggs",
    "fresh_produce",
]

EXACT_CLASS_MAP = {
    "pasta": 0,
    "pasta_noodles": 0,
    "pasta noodles": 0,
    "noodles": 0,
    "spaghetti": 0,
    "macaroni": 0,
    "ramen": 0,
    "instant_noodle": 0,
    "instant noodle": 0,
    "canned_tomato": 1,
    "canned tomato": 1,
    "canned tomatoes": 1,
    "tomato_sauce": 1,
    "tomato sauce": 1,
    "tomato_paste": 1,
    "tomato paste": 1,
    "tomato_can": 1,
    "tomato can": 1,
    "spaghetti_sauce": 1,
    "spaghetti sauce": 1,
    "canned_tuna_meat": 2,
    "canned tuna meat": 2,
    "canned_tuna": 2,
    "canned tuna": 2,
    "tuna": 2,
    "tuna_can": 2,
    "tuna can": 2,
    "canned_salmon": 2,
    "canned salmon": 2,
    "sardine": 2,
    "sardines": 2,
    "corned_beef": 2,
    "corned beef": 2,
    "canned_meat": 2,
    "canned meat": 2,
    "spam": 2,
    "fish": 2,
    "cereal_grains": 3,
    "cereal grains": 3,
    "cereal": 3,
    "oats": 3,
    "oatmeal": 3,
    "grain": 3,
    "grains": 3,
    "rice": 3,
    "wheat": 3,
    "flour": 3,
    "coffee_condiment": 4,
    "coffee condiment": 4,
    "coffee": 4,
    "tea": 4,
    "oil": 4,
    "cooking_oil": 4,
    "cooking oil": 4,
    "condiment": 4,
    "vinegar": 4,
    "soy_sauce": 4,
    "soy sauce": 4,
    "sugar": 4,
    "salt": 4,
    "chilli": 4,
    "chili": 4,
    "spices": 4,
    "milk_dairy": 5,
    "milk dairy": 5,
    "milk": 5,
    "dairy": 5,
    "cheese": 5,
    "goat_cheese": 5,
    "goat cheese": 5,
    "butter": 5,
    "yogurt": 5,
    "heavy_cream": 5,
    "heavy cream": 5,
    "cream": 5,
    "eggs": 6,
    "egg": 6,
    "fresh_produce": 7,
    "fresh produce": 7,
    "vegetable": 7,
    "vegetables": 7,
    "fruit": 7,
    "onion": 7,
    "garlic": 7,
    "carrot": 7,
    "potato": 7,
    "sweet_potato": 7,
    "sweet potato": 7,
    "apple": 7,
    "banana": 7,
    "tomato": 7,
    "bell_pepper": 7,
    "bell pepper": 7,
    "blueberries": 7,
    "lime": 7,
    "mushrooms": 7,
    "peas_and_carrots": 7,
    "peas and carrots": 7,
    "spinach": 7,
    "strawberries": 7,
    "green_beans": 7,
    "green beans": 7,
    "corn": 7,
}

FORBIDDEN_RAW_MEAT_CLASSES = {
    "beef",
    "ground_beef",
    "ground beef",
    "chicken",
    "chicken_breast",
    "chicken breast",
    "ham",
    "shrimp",
    "pork",
    "raw_meat",
    "raw meat",
    "meat",
}

SOURCE_CLASS_MAPS = {
    "frieburg": {
        "pasta": 0,
        "tomato_sauce": 1,
        "fish": 2,
        "cereal": 3,
        "rice": 3,
        "flour": 3,
        "coffee": 4,
        "oil": 4,
        "vinegar": 4,
        "sugar": 4,
        "tea": 4,
        "spices": 4,
        "milk": 5,
    },
    "kitchen_yolo": {
        "pasta": 0,
        "tomato sauce": 1,
        "tomato_sauce": 1,
        "chilli": 4,
        "butter": 5,
        "cheese": 5,
        "bell pepper": 7,
        "bell_pepper": 7,
        "garlic": 7,
        "mushrooms": 7,
        "peas and carrots": 7,
        "peas_and_carrots": 7,
    },
    "fridge_food": {
        "milk": 5,
        "cheese": 5,
        "goat_cheese": 5,
        "butter": 5,
        "heavy_cream": 5,
        "eggs": 6,
        "apple": 7,
        "banana": 7,
        "blueberries": 7,
        "carrot": 7,
        "corn": 7,
        "green_beans": 7,
        "lime": 7,
        "mushrooms": 7,
        "onion": 7,
        "potato": 7,
        "spinach": 7,
        "strawberries": 7,
        "sweet_potato": 7,
        "tomato": 7,
    },
}

FRIEBURG_NAMES = [
    "beans",
    "cake",
    "candy",
    "cereal",
    "chips",
    "chocolate",
    "coffee",
    "corn",
    "fish",
    "flour",
    "honey",
    "jam",
    "juice",
    "milk",
    "nuts",
    "oil",
    "pasta",
    "rice",
    "soda",
    "spices",
    "sugar",
    "tea",
    "tomato_sauce",
    "vinegar",
    "water",
]


def normalize_raw_label(source_name):
    if source_name is None:
        return ""
    cleaned = str(source_name).lower().replace("-", " ").strip()
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned


def map_label_name(source_name, source_name_key=None):
    cleaned = normalize_raw_label(source_name)
    if not cleaned:
        return -1
    underscored = cleaned.replace(" ", "_")
    if cleaned in FORBIDDEN_RAW_MEAT_CLASSES or underscored in FORBIDDEN_RAW_MEAT_CLASSES:
        return -1
    if source_name_key and source_name_key in SOURCE_CLASS_MAPS:
        src_map = SOURCE_CLASS_MAPS[source_name_key]
        if cleaned in src_map:
            return int(src_map[cleaned])
        if underscored in src_map:
            return int(src_map[underscored])
        return -1
    if cleaned in EXACT_CLASS_MAP:
        return int(EXACT_CLASS_MAP[cleaned])
    if underscored in EXACT_CLASS_MAP:
        return int(EXACT_CLASS_MAP[underscored])
    return -1


def is_valid_yolo_box(xc, yc, bw, bh, max_area=0.80, max_aspect=6.0):
    if bw <= 0.005 or bh <= 0.005:
        return False
    if xc < 0.0 or xc > 1.0 or yc < 0.0 or yc > 1.0:
        return False
    area = float(bw) * float(bh)
    if area > float(max_area):
        return False
    if bw > 0.90 and bh > 0.35:
        return False
    aspect_a = float(bw) / float(bh)
    aspect_b = float(bh) / float(bw)
    if aspect_a > float(max_aspect) or aspect_b > float(max_aspect):
        return False
    return True


def parse_box_coords(parts):
    if len(parts) == 5:
        try:
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            return xc, yc, bw, bh
        except ValueError:
            return None
    if len(parts) > 5:
        coords = []
        for val_str in parts[1:]:
            try:
                coords.append(float(val_str))
            except ValueError:
                return None
        xs = []
        ys = []
        idx = 0
        while idx + 1 < len(coords):
            xs.append(coords[idx])
            ys.append(coords[idx + 1])
            idx += 2
        if not xs or not ys:
            return None
        x1 = max(0.0, min(1.0, min(xs)))
        y1 = max(0.0, min(1.0, min(ys)))
        x2 = max(0.0, min(1.0, max(xs)))
        y2 = max(0.0, min(1.0, max(ys)))
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
    return None


def mask_unmapped_regions_on_image(img_bgr, drop_boxes_norm, keep_boxes_norm):
    if img_bgr is None or img_bgr.size == 0 or not drop_boxes_norm:
        return img_bgr
    h, w = img_bgr.shape[:2]
    paint_mask = np.zeros((h, w), dtype=bool)
    for xc, yc, bw, bh in drop_boxes_norm:
        x1 = max(0, min(w, int(round((xc - bw / 2.0) * float(w)))))
        y1 = max(0, min(h, int(round((yc - bh / 2.0) * float(h)))))
        x2 = max(0, min(w, int(round((xc + bw / 2.0) * float(w)))))
        y2 = max(0, min(h, int(round((yc + bh / 2.0) * float(h)))))
        if x2 > x1 and y2 > y1:
            paint_mask[y1:y2, x1:x2] = True
    for xc, yc, bw, bh in keep_boxes_norm:
        x1 = max(0, min(w, int(round((xc - bw / 2.0) * float(w)))))
        y1 = max(0, min(h, int(round((yc - bh / 2.0) * float(h)))))
        x2 = max(0, min(w, int(round((xc + bw / 2.0) * float(w)))))
        y2 = max(0, min(h, int(round((yc + bh / 2.0) * float(h)))))
        if x2 > x1 and y2 > y1:
            paint_mask[y1:y2, x1:x2] = False
    out_img = img_bgr.copy()
    if np.any(paint_mask):
        flat = out_img.reshape(-1, 3)
        median_bgr = np.median(flat, axis=0).astype(np.uint8)
        out_img[paint_mask] = median_bgr
    return out_img


def write_standard_data_yaml(output_dir):
    names_dict = {}
    for idx, name in enumerate(TARGET_CLASSES):
        names_dict[idx] = name
    out_yaml = {
        "path": str(Path(output_dir).resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": names_dict,
    }
    with open(Path(output_dir) / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)


def remap_split(
    split_dir,
    source_names,
    output_dir,
    split_name,
    source_name_key=None,
    drop_forbidden_images=True,
    mask_dropped_boxes=False,
    max_box_area=0.60,
):
    split_path = Path(split_dir)
    out_path = Path(output_dir)
    images_src = split_path / "images"
    labels_src = split_path / "labels"
    if not images_src.exists():
        images_src = split_path
    if not labels_src.exists():
        labels_src = split_path

    images_dst = out_path / split_name / "images"
    labels_dst = out_path / split_name / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    mapping = {}
    forbidden_ids = set()
    for idx, name in enumerate(source_names):
        cleaned = normalize_raw_label(name)
        underscored = cleaned.replace(" ", "_")
        if cleaned in FORBIDDEN_RAW_MEAT_CLASSES or underscored in FORBIDDEN_RAW_MEAT_CLASSES:
            forbidden_ids.add(idx)
        mapped = map_label_name(str(name), source_name_key=source_name_key)
        if mapped != -1:
            mapping[idx] = mapped

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    saved_count = 0
    for img_path in sorted(images_src.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in image_extensions:
            continue
        label_path = labels_src / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        raw_lines = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    raw_lines.append(stripped)

        if len(raw_lines) == 0:
            shutil.copy2(img_path, images_dst / img_path.name)
            with open(labels_dst / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
                f.write("")
            saved_count += 1
            continue

        remapped_lines = []
        keep_boxes_norm = []
        drop_boxes_norm = []
        has_forbidden = False
        has_unmapped = False

        for line in raw_lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                src_cls = int(float(parts[0]))
            except ValueError:
                continue
            coords = parse_box_coords(parts)
            if coords is None:
                continue
            xc, yc, bw, bh = coords
            if src_cls in forbidden_ids:
                has_forbidden = True
                drop_boxes_norm.append((xc, yc, bw, bh))
                continue
            if src_cls not in mapping:
                has_unmapped = True
                drop_boxes_norm.append((xc, yc, bw, bh))
                continue
            if not is_valid_yolo_box(xc, yc, bw, bh, max_area=max_box_area):
                continue
            target_cls = mapping[src_cls]
            remapped_lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            keep_boxes_norm.append((xc, yc, bw, bh))

        if has_forbidden and drop_forbidden_images and not mask_dropped_boxes:
            continue
        if has_unmapped and not mask_dropped_boxes:
            continue
        if not remapped_lines:
            continue

        target_img = images_dst / img_path.name
        if mask_dropped_boxes and drop_boxes_norm:
            bgr = cv2.imread(str(img_path))
            if bgr is None or bgr.size == 0:
                continue
            cleaned_bgr = mask_unmapped_regions_on_image(bgr, drop_boxes_norm, keep_boxes_norm)
            cv2.imwrite(str(target_img), cleaned_bgr)
        else:
            shutil.copy2(img_path, target_img)

        with open(labels_dst / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
            f.writelines(remapped_lines)
        saved_count += 1

    return saved_count


def process_dataset_directory(
    source_path,
    output_dir,
    source_name_key=None,
    drop_forbidden_images=True,
    mask_dropped_boxes=False,
    max_box_area=0.60,
):
    candidates = list(Path(source_path).rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(f"data.yaml not found under {source_path}")

    data_yaml_path = candidates[0]
    actual_source_dir = data_yaml_path.parent

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    raw_names = meta.get("names", [])
    source_names = []
    if isinstance(raw_names, dict):
        for k in sorted(raw_names.keys()):
            source_names.append(raw_names[k])
    elif isinstance(raw_names, list):
        source_names = list(raw_names)

    total_saved = 0
    for split in ["train", "valid", "test", "val"]:
        cand = actual_source_dir / split
        if cand.exists():
            norm_split = "valid" if split == "val" else split
            total_saved += remap_split(
                cand,
                source_names,
                output_dir,
                norm_split,
                source_name_key=source_name_key,
                drop_forbidden_images=drop_forbidden_images,
                mask_dropped_boxes=mask_dropped_boxes,
                max_box_area=max_box_area,
            )

    write_standard_data_yaml(output_dir)
    return total_saved


def fetch_url_bytes(url, retries=4, timeout=15):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * float(attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def extract_frieburg_base_stem(filename):
    base = filename.split("_png")[0]
    return base


def fetch_frieburg_yolo_subset(output_dir="data/raw_rf_frieburg", per_prefix_limit=30):
    out_root = Path(output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    for split_name in ("train", "valid", "test"):
        (out_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

    target_prefixes = [
        "FISH",
        "PASTA",
        "TOMATO_SAUCE",
        "CEREAL",
        "RICE",
        "FLOUR",
        "COFFEE",
        "OIL",
        "VINEGAR",
        "SUGAR",
        "MILK",
    ]

    contents_raw = fetch_url_bytes("https://api.github.com/repos/bhimar/GrocerEye/contents/frieburg")
    contents_list = json.loads(contents_raw.decode("utf-8"))

    split_trees = {}
    for entry in contents_list:
        if entry.get("type") == "dir" and entry.get("name") in ("train", "test"):
            tree_raw = fetch_url_bytes(entry["git_url"], timeout=25)
            tree_json = json.loads(tree_raw.decode("utf-8"))
            split_trees[entry["name"]] = tree_json.get("tree", [])

    selected_items = []
    train_limits = {
        "PASTA": 75,
        "TOMATO_SAUCE": 45,
        "FISH": 65,
        "CEREAL": 30,
        "RICE": 18,
        "FLOUR": 18,
        "COFFEE": 18,
        "OIL": 12,
        "VINEGAR": 12,
        "SUGAR": 12,
        "MILK": 25,
    }
    eval_limits = {
        "PASTA": 16,
        "TOMATO_SAUCE": 16,
        "FISH": 20,
        "CEREAL": 12,
        "RICE": 6,
        "FLOUR": 6,
        "COFFEE": 8,
        "OIL": 6,
        "VINEGAR": 6,
        "SUGAR": 6,
        "MILK": 14,
    }

    for gh_split in ("train", "test"):
        tree_entries = split_trees.get(gh_split, [])
        file_set = set()
        for entry in tree_entries:
            file_set.add(entry.get("path", ""))

        jpg_names = []
        for entry in tree_entries:
            p = entry.get("path", "")
            if p.endswith(".jpg"):
                jpg_names.append(p)
        jpg_names.sort()

        prefix_counts = {}
        for pfx in target_prefixes:
            prefix_counts[pfx] = 0
        split_limit_map = train_limits if gh_split == "train" else eval_limits

        for jpg_name in jpg_names:
            matched_prefix = None
            for pfx in target_prefixes:
                if jpg_name.startswith(pfx) and jpg_name[len(pfx) : len(pfx) + 1].isdigit():
                    matched_prefix = pfx
                    break
            if matched_prefix is None:
                continue
            if prefix_counts[matched_prefix] >= split_limit_map[matched_prefix]:
                continue
            txt_name = jpg_name[:-4] + ".txt"
            if txt_name not in file_set:
                continue
            prefix_counts[matched_prefix] += 1
            selected_items.append({
                "gh_split": gh_split,
                "prefix": matched_prefix,
                "jpg_name": jpg_name,
                "txt_name": txt_name,
                "base_stem": extract_frieburg_base_stem(jpg_name),
            })

    staging_dir = Path("data/_tmp/frieburg_raw")
    staging_dir.mkdir(parents=True, exist_ok=True)

    download_jobs = []
    for item in selected_items:
        gh_split = item["gh_split"]
        for fname in (item["jpg_name"], item["txt_name"]):
            local_file = staging_dir / f"{gh_split}__{fname}"
            if not local_file.exists():
                url = f"https://raw.githubusercontent.com/bhimar/GrocerEye/master/frieburg/{gh_split}/{fname}"
                download_jobs.append((url, local_file))

    def download_one(job):
        url, dst = job
        data = fetch_url_bytes(url, retries=4, timeout=15)
        with open(dst, "wb") as f:
            f.write(data)
        return str(dst)

    if download_jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(download_one, download_jobs))

    valid_records = []
    frieburg_map = SOURCE_CLASS_MAPS["frieburg"]
    for item in selected_items:
        gh_split = item["gh_split"]
        jpg_file = staging_dir / f"{gh_split}__{item['jpg_name']}"
        txt_file = staging_dir / f"{gh_split}__{item['txt_name']}"
        if not jpg_file.exists() or not txt_file.exists():
            continue

        remapped_lines = []
        has_unmapped = False
        with open(txt_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                parts = raw_line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    src_idx = int(float(parts[0]))
                except ValueError:
                    continue
                if src_idx < 0 or src_idx >= len(FRIEBURG_NAMES):
                    has_unmapped = True
                    break
                src_name = FRIEBURG_NAMES[src_idx]
                if src_name not in frieburg_map:
                    has_unmapped = True
                    break
                coords = parse_box_coords(parts)
                if coords is None:
                    continue
                xc, yc, bw, bh = coords
                if not is_valid_yolo_box(xc, yc, bw, bh, max_area=0.80):
                    continue
                target_cls = frieburg_map[src_name]
                remapped_lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        if has_unmapped or not remapped_lines:
            continue

        valid_records.append({
            "gh_split": gh_split,
            "prefix": item["prefix"],
            "base_stem": item["base_stem"],
            "jpg_file": jpg_file,
            "label_lines": remapped_lines,
        })

    by_prefix = {}
    for rec in valid_records:
        pfx = rec["prefix"]
        if pfx not in by_prefix:
            by_prefix[pfx] = []
        by_prefix[pfx].append(rec)

    saved_counts = {"train": 0, "valid": 0, "test": 0}
    for pfx in sorted(by_prefix.keys()):
        recs = by_prefix[pfx]
        test_recs = []
        train_recs = []
        for rec in recs:
            if rec["gh_split"] == "test":
                test_recs.append(rec)
            else:
                train_recs.append(rec)

        val_group = []
        test_group = []
        for idx, rec in enumerate(test_recs):
            if idx % 2 == 0:
                val_group.append(rec)
            else:
                test_group.append(rec)

        if not val_group and len(train_recs) >= 3:
            val_group.append(train_recs.pop())
        if not test_group and len(train_recs) >= 3:
            test_group.append(train_recs.pop())

        split_pairs = [
            ("train", train_recs),
            ("valid", val_group),
            ("test", test_group),
        ]
        for target_split, group_recs in split_pairs:
            for rec in group_recs:
                out_stem = rec["jpg_file"].name.split("__", 1)[1][:-4]
                dst_img = out_root / target_split / "images" / f"{out_stem}.jpg"
                dst_lbl = out_root / target_split / "labels" / f"{out_stem}.txt"
                shutil.copy2(rec["jpg_file"], dst_img)
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    f.writelines(rec["label_lines"])
                saved_counts[target_split] += 1

    write_standard_data_yaml(output_dir)
    return saved_counts


def convert_kitchen_yolo_staging(staging_dir="data/_tmp/kitchen_yolo", output_dir="data/raw_rf_kitchen_yolo"):
    src_root = Path(staging_dir)
    out_root = Path(output_dir)
    classes_file = src_root / "classes.txt"
    images_src = src_root / "my_images"
    labels_src = src_root / "labels"
    if not classes_file.exists() or not images_src.exists() or not labels_src.exists():
        raise FileNotFoundError(f"Kitchen YOLO staging directory incomplete: {staging_dir}")

    if out_root.exists():
        shutil.rmtree(out_root)
    for split_name in ("train", "valid", "test"):
        (out_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

    source_names = []
    with open(classes_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                source_names.append(stripped)

    kitchen_map = SOURCE_CLASS_MAPS["kitchen_yolo"]
    records = []
    for img_path in sorted(images_src.iterdir()):
        if not img_path.is_file():
            continue
        lbl_path = labels_src / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        remapped_lines = []
        keep_boxes_norm = []
        drop_boxes_norm = []

        with open(lbl_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                parts = raw_line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    src_idx = int(float(parts[0]))
                except ValueError:
                    continue
                if src_idx < 0 or src_idx >= len(source_names):
                    continue
                coords = parse_box_coords(parts)
                if coords is None:
                    continue
                xc, yc, bw, bh = coords
                src_name = normalize_raw_label(source_names[src_idx])
                if src_name not in kitchen_map:
                    drop_boxes_norm.append((xc, yc, bw, bh))
                    continue
                if not is_valid_yolo_box(xc, yc, bw, bh, max_area=0.60):
                    continue
                target_cls = kitchen_map[src_name]
                remapped_lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                keep_boxes_norm.append((xc, yc, bw, bh))

        if not remapped_lines:
            continue

        records.append({
            "img_path": img_path,
            "label_lines": remapped_lines,
            "keep_boxes": keep_boxes_norm,
            "drop_boxes": drop_boxes_norm,
        })

    saved_counts = {"train": 0, "valid": 0, "test": 0}
    for idx, rec in enumerate(records):
        mod = idx % 10
        if mod == 8:
            split_name = "valid"
        elif mod == 9:
            split_name = "test"
        else:
            split_name = "train"

        bgr = cv2.imread(str(rec["img_path"]))
        if bgr is None or bgr.size == 0:
            continue
        if rec["drop_boxes"]:
            bgr = mask_unmapped_regions_on_image(bgr, rec["drop_boxes"], rec["keep_boxes"])

        stem = f"kitchen_{rec['img_path'].stem}"
        dst_img = out_root / split_name / "images" / f"{stem}.jpg"
        dst_lbl = out_root / split_name / "labels" / f"{stem}.txt"
        cv2.imwrite(str(dst_img), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.writelines(rec["label_lines"])
        saved_counts[split_name] += 1

    write_standard_data_yaml(output_dir)
    return saved_counts


def reprocess_local_raw_roboflow(source_dir="data/raw_roboflow", output_dir="data/raw_roboflow_fridge"):
    src_root = Path(source_dir)
    out_root = Path(output_dir)
    yaml_path = src_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    raw_names = meta.get("names", [])
    source_names = []
    if isinstance(raw_names, dict):
        for k in sorted(raw_names.keys()):
            source_names.append(raw_names[k])
    elif isinstance(raw_names, list):
        source_names = list(raw_names)

    if out_root.exists():
        shutil.rmtree(out_root)
    for split_name in ("train", "valid", "test"):
        (out_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

    fridge_map = SOURCE_CLASS_MAPS["fridge_food"]
    grouped_records = {}

    for split_cand in ("train", "valid", "test", "val"):
        split_dir = src_root / split_cand
        if not split_dir.exists():
            continue
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists():
            images_dir = split_dir
        if not labels_dir.exists():
            labels_dir = split_dir

        for img_path in sorted(images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            remapped_lines = []
            keep_boxes_norm = []
            drop_boxes_norm = []

            with open(lbl_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    parts = raw_line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        src_idx = int(float(parts[0]))
                    except ValueError:
                        continue
                    if src_idx < 0 or src_idx >= len(source_names):
                        continue
                    coords = parse_box_coords(parts)
                    if coords is None:
                        continue
                    xc, yc, bw, bh = coords
                    raw_cls_name = normalize_raw_label(source_names[src_idx]).replace(" ", "_")
                    if raw_cls_name in FORBIDDEN_RAW_MEAT_CLASSES or raw_cls_name not in fridge_map:
                        drop_boxes_norm.append((xc, yc, bw, bh))
                        continue
                    if not is_valid_yolo_box(xc, yc, bw, bh, max_area=0.60):
                        continue
                    target_cls = fridge_map[raw_cls_name]
                    remapped_lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    keep_boxes_norm.append((xc, yc, bw, bh))

            if not remapped_lines:
                continue

            base_key = img_path.name.split("_JPG")[0]
            if base_key not in grouped_records:
                grouped_records[base_key] = []
            grouped_records[base_key].append({
                "img_path": img_path,
                "label_lines": remapped_lines,
                "keep_boxes": keep_boxes_norm,
                "drop_boxes": drop_boxes_norm,
            })

    group_keys = sorted(grouped_records.keys())
    saved_counts = {"train": 0, "valid": 0, "test": 0}
    for g_idx, g_key in enumerate(group_keys):
        if g_idx % 8 == 6:
            split_name = "valid"
        elif g_idx % 8 == 7:
            split_name = "test"
        else:
            split_name = "train"

        for rec in grouped_records[g_key]:
            bgr = cv2.imread(str(rec["img_path"]))
            if bgr is None or bgr.size == 0:
                continue
            if rec["drop_boxes"]:
                bgr = mask_unmapped_regions_on_image(bgr, rec["drop_boxes"], rec["keep_boxes"])
            dst_img = out_root / split_name / "images" / rec["img_path"].name
            dst_lbl = out_root / split_name / "labels" / f"{rec['img_path'].stem}.txt"
            cv2.imwrite(str(dst_img), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.writelines(rec["label_lines"])
            saved_counts[split_name] += 1

    write_standard_data_yaml(output_dir)
    return saved_counts


def build_real_pantry_samples_dataset(samples_dir="data/samples", output_dir="data/raw_rf_pantry_samples"):
    src_root = Path(samples_dir)
    out_root = Path(output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    for split_name in ("train", "valid", "test"):
        (out_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

    sample_annotations = {
        "pantry_images_cv_01.jpg": [
            (0, 0.10, 0.21, 0.19, 0.30),
            (0, 0.48, 0.54, 0.56, 0.63),
            (0, 0.10, 0.68, 0.16, 0.79),
            (1, 0.59, 0.08, 0.68, 0.16),
            (1, 0.40, 0.21, 0.54, 0.30),
            (1, 0.38, 0.37, 0.49, 0.46),
            (1, 0.79, 0.54, 0.85, 0.63),
            (3, 0.62, 0.19, 0.74, 0.30),
            (3, 0.74, 0.33, 0.89, 0.44),
            (3, 0.62, 0.54, 0.66, 0.63),
            (4, 0.21, 0.38, 0.38, 0.47),
            (4, 0.51, 0.38, 0.64, 0.46),
        ],
        "pantry_images_cv_02.jpg": [
            (1, 0.63, 0.43, 0.69, 0.53),
            (4, 0.13, 0.55, 0.28, 0.64),
            (3, 0.78, 0.43, 0.86, 0.52),
        ],
        "pantry_images_cv_03.jpg": [
            (0, 0.89, 0.73, 0.98, 0.94),
            (1, 0.63, 0.24, 0.76, 0.38),
            (3, 0.81, 0.44, 0.96, 0.67),
            (3, 0.70, 0.47, 0.81, 0.67),
            (3, 0.09, 0.47, 0.19, 0.67),
            (4, 0.40, 0.16, 0.48, 0.38),
            (4, 0.76, 0.22, 0.89, 0.37),
            (4, 0.12, 0.79, 0.28, 0.94),
            (4, 0.55, 0.81, 0.66, 0.94),
        ],
        "pantry_images_cv_04.jpg": [
            (1, 0.38, 0.48, 0.51, 0.63),
            (4, 0.21, 0.26, 0.46, 0.43),
            (4, 0.53, 0.27, 0.61, 0.43),
            (4, 0.72, 0.22, 0.83, 0.42),
            (5, 0.18, 0.85, 0.83, 0.99),
        ],
        "pantry_images_cv_05.jpg": [
            (4, 0.10, 0.02, 0.20, 0.13),
            (3, 0.35, 0.37, 0.55, 0.49),
        ],
        "pantry_images_cv_06.jpg": [
            (1, 0.22, 0.17, 0.48, 0.34),
            (1, 0.72, 0.04, 0.99, 0.29),
            (1, 0.21, 0.41, 0.46, 0.60),
        ],
        "pantry_images_cv_07.jpg": [
            (4, 0.17, 0.10, 0.45, 0.33),
            (4, 0.45, 0.20, 0.59, 0.33),
        ],
    }

    split_assignment = {
        "pantry_images_cv_01.jpg": "train",
        "pantry_images_cv_02.jpg": "train",
        "pantry_images_cv_03.jpg": "train",
        "pantry_images_cv_04.jpg": "train",
        "pantry_images_cv_05.jpg": "train",
        "pantry_images_cv_06.jpg": "valid",
        "pantry_images_cv_07.jpg": "test",
    }

    saved_counts = {"train": 0, "valid": 0, "test": 0}
    for filename, boxes in sample_annotations.items():
        img_path = src_root / filename
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None or bgr.size == 0:
            continue

        split_name = split_assignment[filename]
        lines = []
        for cls_id, x1, y1, x2, y2 in boxes:
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1
            if is_valid_yolo_box(xc, yc, bw, bh, max_area=0.60):
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        if not lines:
            continue

        up_bgr = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_CUBIC)
        stem = Path(filename).stem
        dst_img = out_root / split_name / "images" / f"{stem}.jpg"
        dst_lbl = out_root / split_name / "labels" / f"{stem}.txt"
        cv2.imwrite(str(dst_img), up_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.writelines(lines)
        saved_counts[split_name] += 1

    write_standard_data_yaml(output_dir)
    return saved_counts


def download_from_roboflow(api_key, workspace, project, version, download_dir):
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)
    if download_dir.exists():
        shutil.rmtree(download_dir)
    target_loc = str(download_dir.resolve())
    dataset = ver.download("yolov8", location=target_loc, overwrite=True)
    return Path(dataset.location)


def extract_zip(zip_path, extract_dir):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def prepare_all_real_datasets():
    frieburg_counts = fetch_frieburg_yolo_subset(output_dir="data/raw_rf_frieburg", per_prefix_limit=30)
    if os.path.isdir("data/_tmp/kitchen_yolo"):
        kitchen_counts = convert_kitchen_yolo_staging(
            staging_dir="data/_tmp/kitchen_yolo",
            output_dir="data/raw_rf_kitchen_yolo",
        )
    elif os.path.exists("data/raw_rf_kitchen_yolo/data.yaml"):
        kitchen_counts = {"status": "kept_existing"}
    else:
        kitchen_counts = convert_kitchen_yolo_staging(
            staging_dir="data/_tmp/kitchen_yolo",
            output_dir="data/raw_rf_kitchen_yolo",
        )
    fridge_counts = reprocess_local_raw_roboflow(
        source_dir="data/raw_roboflow",
        output_dir="data/raw_roboflow_fridge",
    )
    pantry_sample_counts = build_real_pantry_samples_dataset(
        samples_dir="data/samples",
        output_dir="data/raw_rf_pantry_samples",
    )

    api_key = os.getenv("ROBOFLOW_API_KEY")
    roboflow_downloads = {}
    if api_key:
        extra_sources = [
            ("yolo", "pantry", 1, "raw_rf_yolo_pantry"),
            ("stvio", "smart-pantry", 1, "raw_rf_smart_pantry"),
        ]
        for ws, proj, ver, slug in extra_sources:
            try:
                staging = Path("data/_tmp") / slug
                downloaded = download_from_roboflow(api_key, ws, proj, ver, staging)
                out_dir = Path("data") / slug
                saved = process_dataset_directory(downloaded, out_dir)
                roboflow_downloads[slug] = saved
            except Exception as exc:
                roboflow_downloads[slug] = f"skipped: {exc}"

    return {
        "frieburg": frieburg_counts,
        "kitchen_yolo": kitchen_counts,
        "fridge": fridge_counts,
        "pantry_samples": pantry_sample_counts,
        "roboflow_extra": roboflow_downloads,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY"))
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--zip", type=str, default=None)
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/dataset_yolo")
    parser.add_argument("--source-key", type=str, default=None)
    parser.add_argument("--prepare-all", action="store_true")
    args = parser.parse_args()

    if args.prepare_all:
        summary = prepare_all_real_datasets()
        print("Prepared real datasets:", summary)
        return

    output_path = Path(args.output_dir)
    raw_staging = Path("data/raw_roboflow")

    if args.zip:
        extracted = extract_zip(Path(args.zip), raw_staging)
        process_dataset_directory(extracted, output_path, source_name_key=args.source_key)
    elif args.source_dir:
        process_dataset_directory(Path(args.source_dir), output_path, source_name_key=args.source_key)
    elif args.workspace and args.project:
        if not args.api_key:
            sys.exit("ROBOFLOW_API_KEY required. Set env var or pass --api-key.")
        slug = f"raw_rf_{args.project.replace('-', '_')}"
        downloaded = download_from_roboflow(args.api_key, args.workspace, args.project, args.version, Path("data/_tmp") / slug)
        process_dataset_directory(downloaded, output_path, source_name_key=args.source_key)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
