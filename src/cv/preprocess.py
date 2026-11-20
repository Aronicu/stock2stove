import base64
import cv2
import numpy as np


def apply_clahe_lab(bgr_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    mean_l = float(np.mean(l_channel))
    if mean_l < 45.0:
        norm_img = np.clip(bgr_image.astype(np.float32) / 255.0, 0.0, 1.0)
        restored = np.clip(np.power(norm_img / 0.26, 1.0 / 2.4) * 255.0, 0.0, 255.0).astype(np.uint8)
        lab_restored = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
        l_rest, a_rest, b_rest = cv2.split(lab_restored)
        clahe_limit = min(float(clip_limit), 1.8)
        clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=tuple(tile_grid_size))
        l_equalized = clahe.apply(l_rest)
        merged_lab = cv2.merge((l_equalized, a_rest, b_rest))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return bgr_image.copy()


def letterbox_image(bgr_image, target_size=640, pad_color=(114, 114, 114)):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")
    orig_h, orig_w = bgr_image.shape[:2]
    target_w = int(target_size)
    target_h = int(target_size)

    scale_w = float(target_w) / float(orig_w)
    scale_h = float(target_h) / float(orig_h)
    scale = scale_w
    if scale_h < scale_w:
        scale = scale_h

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    if new_w != orig_w or new_h != orig_h:
        resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = bgr_image.copy()

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    meta = {
        "scale": float(scale),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "target_size": int(target_size),
    }
    return padded, meta


def unletterbox_bbox_norm(bbox_xyxy_px, meta):
    scale = float(meta.get("scale", 1.0))
    pad_left = float(meta.get("pad_left", 0))
    pad_top = float(meta.get("pad_top", 0))
    orig_w = float(meta.get("orig_w", 640))
    orig_h = float(meta.get("orig_h", 640))

    if scale <= 0.0:
        scale = 1.0
    if orig_w <= 0.0:
        orig_w = 640.0
    if orig_h <= 0.0:
        orig_h = 640.0

    x1 = (float(bbox_xyxy_px[0]) - pad_left) / scale
    y1 = (float(bbox_xyxy_px[1]) - pad_top) / scale
    x2 = (float(bbox_xyxy_px[2]) - pad_left) / scale
    y2 = (float(bbox_xyxy_px[3]) - pad_top) / scale

    x1_norm = max(0.0, min(1.0, x1 / orig_w))
    y1_norm = max(0.0, min(1.0, y1 / orig_h))
    x2_norm = max(0.0, min(1.0, x2 / orig_w))
    y2_norm = max(0.0, min(1.0, y2 / orig_h))

    if x2_norm < x1_norm:
        temp = x1_norm
        x1_norm = x2_norm
        x2_norm = temp
    if y2_norm < y1_norm:
        temp = y1_norm
        y1_norm = y2_norm
        y2_norm = temp

    return [
        round(x1_norm, 4),
        round(y1_norm, 4),
        round(x2_norm, 4),
        round(y2_norm, 4),
    ]


def unletterbox_polygon_norm(polygon_pts_px, meta):
    scale = float(meta.get("scale", 1.0))
    pad_left = float(meta.get("pad_left", 0))
    pad_top = float(meta.get("pad_top", 0))
    orig_w = float(meta.get("orig_w", 640))
    orig_h = float(meta.get("orig_h", 640))

    if scale <= 0.0:
        scale = 1.0
    if orig_w <= 0.0:
        orig_w = 640.0
    if orig_h <= 0.0:
        orig_h = 640.0

    normalized_pts = []
    for pt in polygon_pts_px:
        px = (float(pt[0]) - pad_left) / scale
        py = (float(pt[1]) - pad_top) / scale
        nx = max(0.0, min(1.0, px / orig_w))
        ny = max(0.0, min(1.0, py / orig_h))
        normalized_pts.append([round(nx, 4), round(ny, 4)])
    return normalized_pts


def unletterbox_detections(detections, meta):
    if not detections or not meta:
        return list(detections) if detections else []

    target_size = float(meta.get("target_size", 640))
    if target_size <= 0.0:
        target_size = 640.0

    mapped = []
    for det in detections:
        copied = dict(det)
        bbox = det.get("bbox_norm")
        if bbox and len(bbox) >= 4:
            bbox_px = [
                float(bbox[0]) * target_size,
                float(bbox[1]) * target_size,
                float(bbox[2]) * target_size,
                float(bbox[3]) * target_size,
            ]
            copied["bbox_norm"] = unletterbox_bbox_norm(bbox_px, meta)

        poly = det.get("polygon_norm")
        if isinstance(poly, list) and len(poly) > 0:
            poly_px = []
            for pt in poly:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    poly_px.append([float(pt[0]) * target_size, float(pt[1]) * target_size])
            if len(poly_px) > 0:
                copied["polygon_norm"] = unletterbox_polygon_norm(poly_px, meta)

        mapped.append(copied)

    return mapped


def compute_frame_diagnostics(bgr_image, clahe_applied=True, orb_nfeatures=500):
    h, w = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    total_pixels = float(h * w)
    if total_pixels <= 0.0:
        total_pixels = 1.0
    edge_density_pct = (float(np.count_nonzero(edges)) / total_pixels) * 100.0

    orb = cv2.ORB_create(nfeatures=int(orb_nfeatures))
    keypoints = orb.detect(gray, None)
    kp_count = 0
    if keypoints is not None:
        kp_count = len(keypoints)

    return {
        "edge_density_pct": round(edge_density_pct, 2),
        "orb_keypoints": int(kp_count),
        "clahe_applied": bool(clahe_applied),
    }


def preprocess_frame(bgr_image, enable_clahe=True, target_size=640, clip_limit=2.0, tile_grid_size=(8, 8)):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")

    if enable_clahe:
        enhanced_bgr = apply_clahe_lab(bgr_image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    else:
        enhanced_bgr = bgr_image.copy()

    letterboxed_bgr, meta = letterbox_image(enhanced_bgr, target_size=target_size)
    orig_h, orig_w = enhanced_bgr.shape[:2]
    new_w = int(meta["new_w"])
    new_h = int(meta["new_h"])
    if new_w != orig_w or new_h != orig_h:
        resized_bgr = cv2.resize(enhanced_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_bgr = enhanced_bgr.copy()
    diagnostics = compute_frame_diagnostics(enhanced_bgr, clahe_applied=enable_clahe)

    return {
        "enhanced_bgr": enhanced_bgr,
        "resized_bgr": resized_bgr,
        "letterboxed_bgr": letterboxed_bgr,
        "meta": meta,
        "diagnostics": diagnostics,
    }


def decode_image_bytes(raw_bytes):
    if not raw_bytes:
        raise ValueError("Empty image bytes provided.")
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode image bytes.")
    return bgr


def decode_base64_image(b64_string):
    if not b64_string:
        raise ValueError("Empty base64 image string provided.")
    cleaned = str(b64_string).strip()
    if "," in cleaned and cleaned.startswith("data:"):
        parts = cleaned.split(",", 1)
        cleaned = parts[1]
    raw_bytes = base64.b64decode(cleaned)
    return decode_image_bytes(raw_bytes)


def encode_image_base64_jpeg(bgr_image, quality=85):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", bgr_image, encode_params)
    if not ok:
        raise ValueError("Failed to encode image as JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def crop_detection_base64_jpeg(bgr_image, bbox_norm, max_side=128, quality=78):
    if bgr_image is None or bgr_image.size == 0:
        return None
    if not bbox_norm or len(bbox_norm) < 4:
        return None

    h, w = bgr_image.shape[:2]
    if h <= 0 or w <= 0:
        return None

    try:
        x1_f = float(bbox_norm[0])
        y1_f = float(bbox_norm[1])
        x2_f = float(bbox_norm[2])
        y2_f = float(bbox_norm[3])
    except (TypeError, ValueError):
        return None

    x1_n = max(0.0, min(1.0, x1_f))
    y1_n = max(0.0, min(1.0, y1_f))
    x2_n = max(0.0, min(1.0, x2_f))
    y2_n = max(0.0, min(1.0, y2_f))

    if x2_n < x1_n:
        temp_x = x1_n
        x1_n = x2_n
        x2_n = temp_x
    if y2_n < y1_n:
        temp_y = y1_n
        y1_n = y2_n
        y2_n = temp_y

    px1 = max(0, min(w - 1, int(x1_n * float(w))))
    py1 = max(0, min(h - 1, int(y1_n * float(h))))
    px2 = max(px1 + 1, min(w, int(round(x2_n * float(w)))))
    py2 = max(py1 + 1, min(h, int(round(y2_n * float(h)))))

    crop = bgr_image[py1:py2, px1:px2]
    if crop is None or crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return None

    if max_side is not None and int(max_side) > 0:
        largest = ch if ch > cw else cw
        if largest > int(max_side):
            scale = float(max_side) / float(largest)
            new_w = max(1, int(round(float(cw) * scale)))
            new_h = max(1, int(round(float(ch) * scale)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    try:
        return encode_image_base64_jpeg(crop, quality=quality)
    except Exception:
        return None
