import glob
import os
import cv2
import numpy as np

from src.server.config import (
    DEFAULT_CONF_THRESHOLD,
    LOWE_RATIO_THRESHOLD,
    MIN_HOMOGRAPHY_INLIERS,
    NMS_IOU_THRESHOLD,
    ORB_NFEATURES,
    RANSAC_REPROJ_THRESHOLD,
    SAMPLES_DIR,
    SIFT_NFEATURES,
    TEMPLATES_DIR,
    UNIFIED_CLASSES,
    get_display_name,
    normalize_class_id,
)

TEMPLATE_CACHE = {
    "orb_baseline": [],
    "sift_baseline": [],
    "templates_dir": None,
}


def compute_box_iou(box_a, box_b):
    x_left = max(float(box_a[0]), float(box_b[0]))
    y_top = max(float(box_a[1]), float(box_b[1]))
    x_right = min(float(box_a[2]), float(box_b[2]))
    y_bottom = min(float(box_a[3]), float(box_b[3]))

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
    area_b = max(0.0, (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1])))
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def deduplicate_detections_nms(detections, iou_threshold=NMS_IOU_THRESHOLD):
    ordered = list(detections)
    n = len(ordered)
    for i in range(n):
        for j in range(i + 1, n):
            if ordered[j]["confidence"] > ordered[i]["confidence"]:
                temp = ordered[i]
                ordered[i] = ordered[j]
                ordered[j] = temp

    kept = []
    for cand in ordered:
        should_keep = True
        for existing in kept:
            if existing["class_id"] == cand["class_id"]:
                iou_val = compute_box_iou(existing["bbox_norm"], cand["bbox_norm"])
                if iou_val >= float(iou_threshold):
                    should_keep = False
                    break
        if should_keep:
            kept.append(cand)
    return kept


def ensure_reference_templates(templates_dir=TEMPLATES_DIR, samples_dir=SAMPLES_DIR):
    os.makedirs(templates_dir, exist_ok=True)
    existing_jpgs = glob.glob(os.path.join(templates_dir, "*", "*.jpg"))
    if len(existing_jpgs) >= 8:
        return templates_dir

    sample_files = sorted(glob.glob(os.path.join(samples_dir, "pantry_images_cv_*.jpg")))
    if not sample_files:
        sample_files = sorted(glob.glob(os.path.join(samples_dir, "*.jpg")))

    crop_specs = [
        ("pasta_noodles", 0, 0.10, 0.15, 0.48, 0.85),
        ("pasta_noodles", 1, 0.45, 0.12, 0.88, 0.82),
        ("canned_tomato", 2, 0.12, 0.20, 0.52, 0.86),
        ("canned_tomato", 3, 0.42, 0.18, 0.85, 0.84),
        ("canned_tuna_meat", 4, 0.15, 0.22, 0.58, 0.85),
        ("cereal_grains", 5, 0.10, 0.10, 0.55, 0.88),
        ("coffee_condiment", 6, 0.18, 0.15, 0.62, 0.86),
        ("milk_dairy", 7, 0.14, 0.14, 0.56, 0.86),
        ("eggs", 8, 0.16, 0.20, 0.64, 0.84),
        ("fresh_produce", 9, 0.12, 0.18, 0.60, 0.85),
    ]

    for spec in crop_specs:
        cls_id = spec[0]
        sample_idx = spec[1]
        x1_f = spec[2]
        y1_f = spec[3]
        x2_f = spec[4]
        y2_f = spec[5]

        cls_dir = os.path.join(templates_dir, cls_id)
        os.makedirs(cls_dir, exist_ok=True)
        out_path = os.path.join(cls_dir, f"template_{sample_idx:02d}.jpg")
        if os.path.exists(out_path):
            continue

        if sample_files:
            chosen_path = sample_files[sample_idx % len(sample_files)]
            img = cv2.imread(chosen_path)
            if img is not None:
                h, w = img.shape[:2]
                cx1 = max(0, int(w * x1_f))
                cy1 = max(0, int(h * y1_f))
                cx2 = min(w, int(w * x2_f))
                cy2 = min(h, int(h * y2_f))
                if cx2 - cx1 >= 30 and cy2 - cy1 >= 30:
                    crop = img[cy1:cy2, cx1:cx2]
                    ch_c, cw_c = crop.shape[:2]
                    if cw_c < 220 or ch_c < 220:
                        scale_up = max(220.0 / float(cw_c), 220.0 / float(ch_c))
                        crop = cv2.resize(
                            crop,
                            (int(round(cw_c * scale_up)), int(round(ch_c * scale_up))),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    cv2.imwrite(out_path, crop)
                    continue

        synth = np.zeros((200, 160, 3), dtype=np.uint8)
        synth[:] = (45, 70, 120)
        cv2.rectangle(synth, (12, 12), (148, 188), (220, 200, 160), 2)
        cv2.putText(synth, cls_id[:10].upper(), (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imwrite(out_path, synth)

    return templates_dir


def create_feature_extractor(mode="orb_baseline"):
    if mode == "sift_baseline":
        return cv2.SIFT_create(nfeatures=int(SIFT_NFEATURES))
    return cv2.ORB_create(
        nfeatures=int(ORB_NFEATURES),
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=15,
        fastThreshold=10,
    )


def create_descriptor_matcher(mode="orb_baseline"):
    if mode == "sift_baseline":
        index_params = {"algorithm": 1, "trees": 5}
        search_params = {"checks": 50}
        return cv2.FlannBasedMatcher(index_params, search_params)
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def load_templates_for_mode(mode="orb_baseline", templates_dir=TEMPLATES_DIR):
    normalized_mode = "orb_baseline"
    if mode == "sift_baseline":
        normalized_mode = "sift_baseline"

    ensure_reference_templates(templates_dir=templates_dir)

    if (
        TEMPLATE_CACHE["templates_dir"] == templates_dir
        and len(TEMPLATE_CACHE[normalized_mode]) > 0
    ):
        return TEMPLATE_CACHE[normalized_mode]

    TEMPLATE_CACHE["templates_dir"] = templates_dir
    TEMPLATE_CACHE[normalized_mode] = []

    extractor = create_feature_extractor(normalized_mode)
    class_dirs = sorted(glob.glob(os.path.join(templates_dir, "*")))

    for class_dir in class_dirs:
        if not os.path.isdir(class_dir):
            continue
        raw_class = os.path.basename(class_dir)
        class_id = normalize_class_id(raw_class)
        display_name = get_display_name(class_id)

        img_paths = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
        png_paths = sorted(glob.glob(os.path.join(class_dir, "*.png")))
        for p in png_paths:
            img_paths.append(p)

        for path in img_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = extractor.detectAndCompute(gray, None)
            if descriptors is None or keypoints is None or len(keypoints) < 4:
                continue
            if normalized_mode == "sift_baseline" and descriptors.dtype != np.float32:
                descriptors = descriptors.astype(np.float32)

            th, tw = gray.shape[:2]
            TEMPLATE_CACHE[normalized_mode].append({
                "class_id": class_id,
                "display_name": display_name,
                "template_path": path,
                "width": int(tw),
                "height": int(th),
                "keypoints": keypoints,
                "descriptors": descriptors,
            })

    return TEMPLATE_CACHE[normalized_mode]


def is_valid_homography_quad(quad_pts, scene_w, scene_h):
    if quad_pts is None or len(quad_pts) != 4:
        return False

    xs = []
    ys = []
    for pt in quad_pts:
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    if max_x - min_x < 16.0 or max_y - min_y < 16.0:
        return False
    if max_x - min_x > float(scene_w) * 1.35 or max_y - min_y > float(scene_h) * 1.35:
        return False

    contour = np.array(quad_pts, dtype=np.float32).reshape((-1, 1, 2))
    area = abs(cv2.contourArea(contour))
    scene_area = float(scene_w * scene_h)
    if area < scene_area * 0.002 or area > scene_area * 0.98:
        return False
    if not cv2.isContourConvex(contour.astype(np.int32)):
        return False

    return True


def run_baseline_detection(
    bgr_image,
    mode="orb_baseline",
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    templates_dir=TEMPLATES_DIR,
    ratio_thresh=LOWE_RATIO_THRESHOLD,
    ransac_thresh=RANSAC_REPROJ_THRESHOLD,
    min_inliers=MIN_HOMOGRAPHY_INLIERS,
):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")

    normalized_mode = "orb_baseline"
    if mode == "sift_baseline":
        normalized_mode = "sift_baseline"

    scene_h, scene_w = bgr_image.shape[:2]
    gray_scene = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    extractor = create_feature_extractor(normalized_mode)
    scene_kps, scene_descs = extractor.detectAndCompute(gray_scene, None)

    scene_kp_count = 0
    if scene_kps is not None:
        scene_kp_count = len(scene_kps)

    if scene_descs is None or scene_kp_count < 4:
        return [], {
            "detector_mode": normalized_mode,
            "scene_keypoints": int(scene_kp_count),
            "good_matches": 0,
            "ransac_inliers": 0,
            "templates_checked": 0,
        }

    if normalized_mode == "sift_baseline" and scene_descs.dtype != np.float32:
        scene_descs = scene_descs.astype(np.float32)

    templates = load_templates_for_mode(mode=normalized_mode, templates_dir=templates_dir)
    matcher = create_descriptor_matcher(normalized_mode)

    raw_detections = []
    total_good_matches = 0
    total_inliers = 0

    for tmpl in templates:
        tmpl_descs = tmpl["descriptors"]
        tmpl_kps = tmpl["keypoints"]
        if tmpl_descs is None or len(tmpl_kps) < 4:
            continue

        try:
            knn_matches = matcher.knnMatch(tmpl_descs, scene_descs, k=2)
        except cv2.error:
            continue

        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m = pair[0]
            n = pair[1]
            if m.distance < float(ratio_thresh) * n.distance:
                good_matches.append(m)

        total_good_matches += len(good_matches)
        if len(good_matches) < max(4, int(min_inliers)):
            continue

        src_pts_list = []
        dst_pts_list = []
        for m in good_matches:
            src_pts_list.append(tmpl_kps[m.queryIdx].pt)
            dst_pts_list.append(scene_kps[m.trainIdx].pt)

        src_pts = np.array(src_pts_list, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(dst_pts_list, dtype=np.float32).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(ransac_thresh))
        if homography is None or mask is None:
            continue

        inliers_count = int(np.sum(mask))
        total_inliers += inliers_count
        if inliers_count < int(min_inliers):
            continue

        tw = float(tmpl["width"])
        th = float(tmpl["height"])
        corners = np.array(
            [[[0.0, 0.0]], [[tw, 0.0]], [[tw, th]], [[0.0, th]]],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(corners, homography)
        quad_px = projected.reshape(4, 2)

        if not is_valid_homography_quad(quad_px, scene_w, scene_h):
            continue

        polygon_norm = []
        xs_norm = []
        ys_norm = []
        for pt in quad_px:
            nx = max(0.0, min(1.0, float(pt[0]) / float(scene_w)))
            ny = max(0.0, min(1.0, float(pt[1]) / float(scene_h)))
            polygon_norm.append([round(nx, 4), round(ny, 4)])
            xs_norm.append(nx)
            ys_norm.append(ny)

        bbox_norm = [
            round(min(xs_norm), 4),
            round(min(ys_norm), 4),
            round(max(xs_norm), 4),
            round(max(ys_norm), 4),
        ]

        inlier_ratio = float(inliers_count) / float(max(1, len(good_matches)))
        raw_conf = 0.45 + min(0.45, (float(inliers_count) / 40.0) * 0.35 + inlier_ratio * 0.15)
        confidence = round(min(0.98, max(0.0, raw_conf)), 2)

        if confidence < float(conf_threshold):
            continue

        raw_detections.append({
            "class_id": tmpl["class_id"],
            "display_name": tmpl["display_name"],
            "confidence": confidence,
            "bbox_norm": bbox_norm,
            "polygon_norm": polygon_norm,
            "temporally_confirmed": False,
            "inliers": inliers_count,
            "good_matches": len(good_matches),
        })

    deduped = deduplicate_detections_nms(raw_detections, iou_threshold=NMS_IOU_THRESHOLD)

    cleaned_detections = []
    for det in deduped:
        cleaned_detections.append({
            "class_id": det["class_id"],
            "display_name": det["display_name"],
            "confidence": det["confidence"],
            "bbox_norm": det["bbox_norm"],
            "polygon_norm": det["polygon_norm"],
            "temporally_confirmed": bool(det.get("temporally_confirmed", False)),
        })

    diagnostics_extra = {
        "detector_mode": normalized_mode,
        "scene_keypoints": int(scene_kp_count),
        "good_matches": int(total_good_matches),
        "ransac_inliers": int(total_inliers),
        "templates_checked": len(templates),
    }
    return cleaned_detections, diagnostics_extra
