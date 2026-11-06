"""
CSST 106 Computer Vision Final Project - Feasibility Spike
Project: Stock-to-Stove (Automated Pantry Shelf Detection & Recipe Suggestion)
Dataset Source: images.cv Pantry Image Classification Dataset
(https://images.cv/dataset/pantry-image-classification-dataset)
"""

import os
import glob
import cv2
import numpy as np

def run_feasibility_spike(samples_dir="data/samples", output_dir="outputs", pattern="pantry_images_cv_*.jpg"):
    """
    Runs the CV feasibility spike proving edge detection, ORB keypoint extraction,
    and generating the 20-sample contact sheet using real images from images.cv.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_paths = sorted(glob.glob(os.path.join(samples_dir, pattern)))

    # Fallback to any jpg if images.cv pattern not found
    if not sample_paths:
        sample_paths = sorted(glob.glob(os.path.join(samples_dir, "*.jpg")))

    if not sample_paths:
        print(f"[ERROR] No sample images found in {samples_dir}.")
        return

    print(f"[SPIKE] Processing {len(sample_paths)} real pantry dataset samples from images.cv...")
    print(f"[SPIKE] Dataset Source: https://images.cv/dataset/pantry-image-classification-dataset")

    # 1. Classical Edge Detection & ORB Keypoint Extraction
    orb = cv2.ORB_create(nfeatures=500)
    
    print("\n--- Feasibility Diagnostics (First 5 Samples) ---")
    for idx, path in enumerate(sample_paths[:5]):
        img = cv2.imread(path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_density = (np.count_nonzero(edges) / (h * w)) * 100.0

        # Keypoints with ORB (for classical feature matching baseline)
        keypoints, _ = orb.detectAndCompute(gray, None)
        img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

        filename = os.path.basename(path)
        edge_out_path = os.path.join(output_dir, f"spike_edge_{filename}")
        kp_out_path = os.path.join(output_dir, f"spike_orb_{filename}")

        cv2.imwrite(edge_out_path, edges)
        cv2.imwrite(kp_out_path, img_kp)

        print(f"Sample #{idx+1} ({filename}): Resolution={w}x{h}, ORB Keypoints={len(keypoints)}, Edge Density={edge_density:.2f}%")

    print(f"\n[SPIKE] Individual edge and ORB keypoint images saved to '{output_dir}'.")

    # 2. Build 20-Sample Contact Sheet from real images.cv samples
    contact_sheet_path = os.path.join(output_dir, "contact_sheet_20_samples.jpg")
    build_contact_sheet(sample_paths[:20], output_path=contact_sheet_path)

def build_contact_sheet(paths, output_path, cols=5, rows=4, thumb_w=240, thumb_h=180):
    """
    Creates a contact sheet grid of 20 samples from images.cv with index and filename labels.
    """
    canvas_w = cols * thumb_w
    canvas_h = rows * thumb_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for idx, path in enumerate(paths[:cols * rows]):
        r = idx // cols
        c = idx % cols
        img = cv2.imread(path)
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h))

        # Add clean header bar
        label = f"#{idx+1}: {os.path.basename(path)[:20]}"
        cv2.rectangle(thumb, (0, 0), (thumb_w, 22), (20, 20, 20), -1)
        cv2.putText(thumb, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (100, 100, 100), 1)

        y_start = r * thumb_h
        x_start = c * thumb_w
        canvas[y_start:y_start + thumb_h, x_start:x_start + thumb_w] = thumb

    cv2.imwrite(output_path, canvas)
    print(f"[SPIKE] 20-sample contact sheet created from images.cv dataset: '{output_path}'.")

if __name__ == "__main__":
    run_feasibility_spike()
