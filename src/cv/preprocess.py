import base64
import cv2
import numpy as np


def apply_clahe_lab(bgr_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input BGR image is empty or None.")
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    l_equalized = clahe.apply(l_channel)
    merged_lab = cv2.merge((l_equalized, a_channel, b_channel))
    equalized_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return equalized_bgr


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
    diagnostics = compute_frame_diagnostics(enhanced_bgr, clahe_applied=enable_clahe)

    return {
        "enhanced_bgr": enhanced_bgr,
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
