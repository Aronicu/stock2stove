import argparse
import math
import os
import shutil
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.server.config import SAMPLES_DIR

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def create_synthetic_pan_video(samples_dir, output_video_path, duration_sec=4.0, fps=15):
    if not samples_dir or not os.path.isdir(str(samples_dir)):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    all_images = []
    shelf_images = []
    for entry_name in sorted(os.listdir(str(samples_dir))):
        full_path = os.path.join(str(samples_dir), entry_name)
        if not os.path.isfile(full_path):
            continue
        lower_name = entry_name.lower()
        if lower_name.endswith(IMAGE_EXTENSIONS):
            all_images.append(full_path)
            if lower_name.startswith("shelf_sample_"):
                shelf_images.append(full_path)

    chosen_paths = shelf_images[:8] if shelf_images else all_images[:8]
    if not chosen_paths:
        raise FileNotFoundError(f"No sample images found in {samples_dir}")

    frame_w = 640
    frame_h = 480
    tiles = []
    for img_path in chosen_paths:
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            continue
        ih, iw = img.shape[:2]
        scale = float(frame_h) / float(max(1, ih))
        new_w = max(1, int(round(float(iw) * scale)))
        interp = cv2.INTER_AREA if ih > frame_h else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, frame_h), interpolation=interp)
        tiles.append(resized)

    if not tiles:
        raise ValueError(f"Could not decode any sample images in {samples_dir}")

    panorama = np.hstack(tiles)
    while panorama.shape[1] < frame_w * 2:
        panorama = np.hstack([panorama, panorama])

    pano_w = int(panorama.shape[1])
    max_x = max(0, pano_w - frame_w)
    eff_fps = float(fps) if float(fps) > 0.0 else 15.0
    total_frames = max(1, int(round(float(duration_sec) * eff_fps)))

    out_parent = os.path.dirname(os.path.abspath(str(output_video_path)))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, eff_fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_video_path}")

    for idx in range(total_frames):
        ratio = float(idx) / float(max(1, total_frames - 1))
        x_offset = int(round(ratio * float(max_x)))
        crop = panorama[:, x_offset : x_offset + frame_w]
        writer.write(crop)

    writer.release()
    return str(output_video_path)


def extract_frames_from_video(video_path, output_dir, target_fps=3.0, prefix=""):
    os.makedirs(str(output_dir), exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps is None or math.isnan(native_fps) or native_fps <= 0.0:
        native_fps = 30.0

    eff_target_fps = float(target_fps)
    if eff_target_fps <= 0.0:
        eff_target_fps = float(native_fps)

    step = float(native_fps) / eff_target_fps
    if step < 1.0:
        step = 1.0

    video_stem = os.path.splitext(os.path.basename(str(video_path)))[0]
    if prefix:
        raw_prefix = str(prefix)
        name_prefix = raw_prefix if raw_prefix.endswith(("_", "-")) else f"{raw_prefix}_"
    else:
        name_prefix = f"{video_stem}_"

    saved_paths = []
    frame_idx = 0
    saved_idx = 0
    next_sample_idx = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if float(frame_idx) + 1e-6 >= next_sample_idx:
            out_name = f"{name_prefix}frame_{saved_idx:04d}.jpg"
            out_path = os.path.join(str(output_dir), out_name)
            ok = cv2.imwrite(out_path, frame)
            if ok:
                saved_paths.append(out_path)
                saved_idx += 1
            while float(frame_idx) + 1e-6 >= next_sample_idx:
                next_sample_idx += step
        frame_idx += 1

    cap.release()
    return saved_paths


def extract_frames_from_directory(input_dir, output_dir, target_fps=3.0, synthesize_demo=False):
    input_path = str(input_dir)
    output_path = str(output_dir)
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    video_files = []
    still_files = []
    for entry_name in sorted(os.listdir(input_path)):
        full_path = os.path.join(input_path, entry_name)
        if not os.path.isfile(full_path):
            continue
        lower_name = entry_name.lower()
        if lower_name.endswith(VIDEO_EXTENSIONS):
            video_files.append(full_path)
        elif lower_name.endswith(IMAGE_EXTENSIONS):
            still_files.append(full_path)

    synthesized = False
    if len(video_files) == 0 and len(still_files) == 0 and synthesize_demo:
        demo_video_path = os.path.join(input_path, "demo_pantry_pan.mp4")
        samples_source = SAMPLES_DIR if os.path.isdir(SAMPLES_DIR) else os.path.join("data", "samples")
        create_synthetic_pan_video(samples_source, demo_video_path, duration_sec=4.0, fps=15)
        video_files.append(demo_video_path)
        synthesized = True

    saved_paths = []
    videos_processed = 0
    frames_extracted = 0
    stills_copied = 0

    for vid_path in video_files:
        vid_stem = os.path.splitext(os.path.basename(vid_path))[0]
        extracted = extract_frames_from_video(
            vid_path,
            output_path,
            target_fps=target_fps,
            prefix=f"{vid_stem}_",
        )
        saved_paths.extend(extracted)
        frames_extracted += len(extracted)
        videos_processed += 1

    for still_path in still_files:
        dst_path = os.path.join(output_path, os.path.basename(still_path))
        if os.path.abspath(still_path) != os.path.abspath(dst_path):
            shutil.copy2(still_path, dst_path)
        saved_paths.append(dst_path)
        stills_copied += 1

    return {
        "input_dir": input_path,
        "output_dir": output_path,
        "videos_processed": videos_processed,
        "stills_copied": stills_copied,
        "frames_extracted": frames_extracted,
        "total_output_files": len(saved_paths),
        "synthesized_demo": synthesized,
        "saved_paths": saved_paths,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw_home")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--output", type=str, default="data/raw_home/frames")
    parser.add_argument("--synthesize-demo", action="store_true")
    args = parser.parse_args()

    summary = extract_frames_from_directory(
        args.input,
        args.output,
        target_fps=args.fps,
        synthesize_demo=args.synthesize_demo,
    )
    print(f"Input directory: {summary['input_dir']}")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Synthesized demo video: {summary['synthesized_demo']}")
    print(f"Videos processed: {summary['videos_processed']}")
    print(f"Stills copied: {summary['stills_copied']}")
    print(f"Frames extracted: {summary['frames_extracted']}")
    print(f"Total output files: {summary['total_output_files']}")


if __name__ == "__main__":
    main()
