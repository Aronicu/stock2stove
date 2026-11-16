import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
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

CLASS_KEYWORDS = {
    0: ["pasta", "noodle", "spaghetti", "macaroni", "ramen"],
    1: ["tomato", "sauce", "ketchup", "paste"],
    2: ["tuna", "canned", "can", "meat", "sardine", "beef", "corned", "spam"],
    3: ["cereal", "oat", "grain", "rice", "wheat", "flour"],
    4: ["coffee", "tea", "oil", "condiment", "vinegar", "soy", "sugar", "salt"],
    5: ["milk", "dairy", "cheese", "butter", "yogurt", "cream"],
    6: ["egg", "eggs"],
    7: ["vegetable", "fruit", "onion", "garlic", "carrot", "potato", "apple", "banana"],
}

def map_label_name(source_name: str) -> int:
    clean = source_name.lower().replace("-", " ").replace("_", " ").strip()
    for class_id, keywords in CLASS_KEYWORDS.items():
        for kw in keywords:
            if kw in clean:
                return class_id
    return -1

def remap_split(split_dir: Path, source_names: list, output_dir: Path, split_name: str):
    images_src = split_dir / "images"
    labels_src = split_dir / "labels"
    if not images_src.exists():
        images_src = split_dir
    if not labels_src.exists():
        labels_src = split_dir

    images_dst = output_dir / split_name / "images"
    labels_dst = output_dir / split_name / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for idx, name in enumerate(source_names):
        mapped = map_label_name(str(name))
        if mapped != -1:
            mapping[idx] = mapped

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img_path in images_src.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
        label_path = labels_src / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        remapped_lines = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    src_cls = int(parts[0])
                    if src_cls in mapping:
                        target_cls = mapping[src_cls]
                        remapped_lines.append(f"{target_cls} {' '.join(parts[1:])}\n")

        if remapped_lines:
            target_img = images_dst / img_path.name
            shutil.copy2(img_path, target_img)
            with open(labels_dst / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
                f.writelines(remapped_lines)

def process_dataset_directory(source_path: Path, output_dir: Path):
    candidates = list(source_path.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(f"data.yaml not found under {source_path}")

    data_yaml_path = candidates[0]
    actual_source_dir = data_yaml_path.parent

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    source_names = meta.get("names", [])
    if isinstance(source_names, dict):
        source_names = [source_names[k] for k in sorted(source_names.keys())]

    for split in ["train", "valid", "test", "val"]:
        cand = actual_source_dir / split
        if cand.exists():
            norm_split = "valid" if split == "val" else split
            remap_split(cand, source_names, output_dir, norm_split)

    out_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": {i: name for i, name in enumerate(TARGET_CLASSES)},
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)

def download_from_roboflow(api_key: str, workspace: str, project: str, version: int, download_dir: Path) -> Path:
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)
    if download_dir.exists():
        shutil.rmtree(download_dir)
    target_loc = str(download_dir.resolve())
    dataset = ver.download("yolov8", location=target_loc, overwrite=True)
    return Path(dataset.location)

def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY"))
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--zip", type=str, default=None)
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/dataset_yolo")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    raw_staging = Path("data/raw_roboflow")

    if args.zip:
        extracted = extract_zip(Path(args.zip), raw_staging)
        process_dataset_directory(extracted, output_path)
    elif args.source_dir:
        process_dataset_directory(Path(args.source_dir), output_path)
    elif args.workspace and args.project:
        if not args.api_key:
            sys.exit("ROBOFLOW_API_KEY required. Set env var or pass --api-key.")
        downloaded = download_from_roboflow(args.api_key, args.workspace, args.project, args.version, raw_staging)
        process_dataset_directory(downloaded, output_path)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
