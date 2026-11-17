import argparse
import shutil
from pathlib import Path


def copy_run_artifacts(src_dir, dst_dir):
    if src_dir.resolve() == dst_dir.resolve():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def ensure_expected_plots(run_dir):
    pr_path = run_dir / "PR_curve.png"
    box_pr_path = run_dir / "BoxPR_curve.png"
    if box_pr_path.exists() and (not pr_path.exists() or box_pr_path.stat().st_mtime >= pr_path.stat().st_mtime):
        shutil.copy2(box_pr_path, pr_path)

    f1_path = run_dir / "F1_curve.png"
    box_f1_path = run_dir / "BoxF1_curve.png"
    if box_f1_path.exists() and (not f1_path.exists() or box_f1_path.stat().st_mtime >= f1_path.stat().st_mtime):
        shutil.copy2(box_f1_path, f1_path)


def train_and_export_yolo(
    data_yaml,
    epochs=120,
    batch=8,
    imgsz=640,
    freeze=0,
    patience=30,
    base_weights="yolov8s.pt",
    run_name="stock2stove_yolov8n",
    export_onnx=True,
    close_mosaic=15,
    seed=0,
):
    from ultralytics import YOLO

    runs_detect_dir = Path("runs") / "detect"
    runs_detect_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_detect_dir / run_name

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(base_weights)
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        freeze=freeze,
        mosaic=1.0,
        mixup=0.15,
        hsv_v=0.4,
        close_mosaic=close_mosaic,
        seed=seed,
        project="runs/detect",
        name=run_name,
        exist_ok=True,
        workers=2,
    )

    actual_save_dir = None
    if train_results is not None and hasattr(train_results, "save_dir"):
        actual_save_dir = Path(train_results.save_dir)
    elif hasattr(model, "trainer") and model.trainer is not None and hasattr(model.trainer, "save_dir"):
        actual_save_dir = Path(model.trainer.save_dir)

    if actual_save_dir is not None and actual_save_dir.exists():
        copy_run_artifacts(actual_save_dir, run_dir)

    weights_dir = run_dir / "weights"
    best_pt_src = weights_dir / "best.pt"
    if not best_pt_src.exists():
        last_pt_src = weights_dir / "last.pt"
        if last_pt_src.exists():
            best_pt_src = last_pt_src
        else:
            raise FileNotFoundError(f"Trained weights not found in {weights_dir}")

    dest_pt = models_dir / "yolov8n_stock2stove.pt"
    shutil.copy2(best_pt_src, dest_pt)

    dest_onnx = None
    if export_onnx:
        best_model = YOLO(str(best_pt_src))
        exported_path = best_model.export(format="onnx", imgsz=imgsz)
        onnx_src = Path(exported_path) if exported_path else (weights_dir / "best.onnx")
        if not onnx_src.exists():
            onnx_src = weights_dir / "best.onnx"
        if not onnx_src.exists():
            raise FileNotFoundError(f"Exported ONNX file not found at {onnx_src}")
        dest_onnx = models_dir / "yolov8n_stock2stove.onnx"
        shutil.copy2(onnx_src, dest_onnx)

    ensure_expected_plots(run_dir)

    metrics = {}
    results_dict = getattr(train_results, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                metrics[str(key)] = str(value)

    summary = {
        "run_dir": str(run_dir),
        "best_pt": str(dest_pt),
        "best_onnx": str(dest_onnx) if dest_onnx is not None else None,
        "plots": {
            "results": str(run_dir / "results.png"),
            "confusion_matrix_normalized": str(run_dir / "confusion_matrix_normalized.png"),
            "pr_curve": str(run_dir / "PR_curve.png"),
            "f1_curve": str(run_dir / "F1_curve.png"),
        },
        "metrics": metrics,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset_yolo/data.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--name", type=str, default="stock2stove_yolov8n")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    summary = train_and_export_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze,
        patience=args.patience,
        base_weights=args.weights,
        run_name=args.name,
        export_onnx=args.export_onnx,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
    )
    print(f"Saved PyTorch weights: {summary['best_pt']}")
    if summary["best_onnx"]:
        print(f"Saved ONNX weights: {summary['best_onnx']}")
    print(f"Run artifacts directory: {summary['run_dir']}")
    if summary["metrics"]:
        print(f"Training metrics: {summary['metrics']}")


if __name__ == "__main__":
    main()
