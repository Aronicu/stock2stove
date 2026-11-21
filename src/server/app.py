import json
import os
import platform
import tempfile
import time
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.cv.baseline_detector import ensure_reference_templates, run_baseline_detection
from src.cv.gemini_detector import (
    get_gemini_detector_status,
    run_gemini_detection,
    run_hybrid_gemini_detection,
)
from src.cv.preprocess import (
    crop_detection_base64_jpeg,
    decode_base64_image,
    decode_image_bytes,
    encode_image_base64_jpeg,
    preprocess_frame,
)
from src.cv.temporal_tracker import create_tracker_state, update_temporal_tracker
from src.cv.yolo_detector import get_yolo_model_status, run_yolo_detection
from src.llm_recipe import generate_structured_recipes
from src.run import render, run_single_frame
from src.server.config import (
    DEFAULT_CONF_THRESHOLD,
    FRONTEND_DIST_DIR,
    HOST,
    PORT,
    SAMPLES_DIR,
    SUPPORTED_DETECTOR_MODES,
    TARGET_IMG_SIZE,
    TEMPLATES_DIR,
    UNIFIED_CLASSES,
    get_class_calories,
    get_display_name,
    is_valid_food_class,
    normalize_class_id,
)
from src.server.schemas import (
    format_confirmed_inventory_item,
    format_detection_item,
    format_scan_response,
    parse_recipe_request,
    parse_scan_frame_request,
    validate_detector_mode,
)
from src.server.session_manager import (
    delete_session_item,
    get_or_create_session,
    get_session_excluded_classes,
    record_session_frame,
    remove_session,
    reset_session,
    restore_session_item,
    update_session_config,
)


def parse_excluded_classes_value(raw_value):
    excluded = set()
    if raw_value is None:
        return excluded
    if isinstance(raw_value, (list, tuple, set)):
        for item in raw_value:
            for sub in parse_excluded_classes_value(item):
                excluded.add(sub)
        return excluded
    raw_str = str(raw_value).strip()
    if not raw_str:
        return excluded
    if raw_str.startswith("["):
        try:
            parsed = json.loads(raw_str)
            if isinstance(parsed, list):
                for entry in parsed:
                    norm = normalize_class_id(entry)
                    if norm:
                        excluded.add(norm)
                return excluded
        except Exception:
            pass
    for part in raw_str.split(","):
        cleaned = part.strip()
        if cleaned:
            norm = normalize_class_id(cleaned)
            if norm:
                excluded.add(norm)
    return excluded


def filter_valid_detections(detections, excluded_classes=None):
    excluded = set()
    if excluded_classes:
        for cls_item in excluded_classes:
            norm = normalize_class_id(cls_item)
            if norm:
                excluded.add(norm)
    kept = []
    for det in detections:
        raw_cls = det.get("class_id") or det.get("display_name") or ""
        if not is_valid_food_class(raw_cls):
            continue
        cls_id = normalize_class_id(raw_cls)
        if cls_id in excluded:
            continue
        det_copy = dict(det)
        det_copy["class_id"] = cls_id
        kept.append(det_copy)
    return kept


def build_session_item_response(status_label, session_id, cls_id, confirmed_inventory, excluded_classes):
    formatted_inv = []
    total_estimated_calories = 0
    for item in confirmed_inventory:
        formatted_item = format_confirmed_inventory_item(item)
        formatted_inv.append(formatted_item)
        total_estimated_calories += int(formatted_item["estimated_calories"])
    return {
        "status": str(status_label),
        "session_id": str(session_id),
        "class_id": str(cls_id),
        "excluded_classes": list(excluded_classes),
        "confirmed_inventory": formatted_inv,
        "total_estimated_calories": int(total_estimated_calories),
    }


def execute_single_frame_pipeline(
    bgr_image,
    detector_mode="gemini_vision",
    enable_clahe=True,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    session_id=None,
    frame_id=0,
):
    t_start = time.perf_counter()
    mode = validate_detector_mode(detector_mode)

    prep = preprocess_frame(
        bgr_image,
        enable_clahe=bool(enable_clahe),
        target_size=TARGET_IMG_SIZE,
    )
    resized_bgr = prep.get("resized_bgr", prep["enhanced_bgr"])
    diagnostics = dict(prep["diagnostics"])

    if mode in ("orb_baseline", "sift_baseline"):
        raw_dets, extra_diag = run_baseline_detection(
            resized_bgr,
            mode=mode,
            conf_threshold=float(conf_threshold),
        )
        diagnostics["good_matches"] = int(extra_diag.get("good_matches", 0))
        diagnostics["ransac_inliers"] = int(extra_diag.get("ransac_inliers", 0))
    elif mode == "gemini_vision":
        raw_dets, _ = run_gemini_detection(
            resized_bgr,
            conf_threshold=float(conf_threshold),
        )
    elif mode == "hybrid_gemini":
        raw_dets, _ = run_hybrid_gemini_detection(
            resized_bgr,
            conf_threshold=float(conf_threshold),
        )
    else:
        raw_dets, _ = run_yolo_detection(
            resized_bgr,
            mode=mode,
            conf_threshold=float(conf_threshold),
        )

    excluded_list = get_session_excluded_classes(session_id) if session_id is not None else []
    filtered_raw_dets = filter_valid_detections(raw_dets, excluded_classes=excluded_list)

    for det in filtered_raw_dets:
        if not det.get("crop_jpeg_b64"):
            det["crop_jpeg_b64"] = crop_detection_base64_jpeg(
                resized_bgr,
                det.get("bbox_norm"),
            )

    if session_id is not None:
        min_hits_override = 1 if mode == "gemini_vision" else None
        fast_confirm_conf = 0.80 if mode in ("gemini_vision", "hybrid_gemini") else None
        dets, confirmed = record_session_frame(
            session_id,
            filtered_raw_dets,
            conf_threshold=float(conf_threshold),
            min_hits_override=min_hits_override,
            fast_confirm_conf=fast_confirm_conf,
        )
    else:
        temp_tracker = create_tracker_state(
            window_size=1,
            min_hits=1,
            conf_threshold=float(conf_threshold),
        )
        dets, confirmed = update_temporal_tracker(
            temp_tracker,
            filtered_raw_dets,
            conf_threshold=float(conf_threshold),
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return format_scan_response(
        frame_id=frame_id,
        detector_mode=mode,
        inference_ms=elapsed_ms,
        diagnostics=diagnostics,
        detections=dets,
        confirmed_inventory=confirmed,
        excluded_classes=excluded_list,
    )


def annotate_frame_visual(bgr_image, detections):
    canvas = bgr_image.copy()
    h, w = canvas.shape[:2]

    for det in detections:
        bbox = det.get("bbox_norm", [0.0, 0.0, 0.0, 0.0])
        poly = det.get("polygon_norm")
        confirmed = bool(det.get("temporally_confirmed", False))
        color = (40, 200, 80) if confirmed else (0, 180, 255)

        if isinstance(poly, list) and len(poly) == 4:
            pts = []
            for p in poly:
                pts.append([int(float(p[0]) * w), int(float(p[1]) * h)])
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], True, color, 2)
        else:
            x1 = int(float(bbox[0]) * w)
            y1 = int(float(bbox[1]) * h)
            x2 = int(float(bbox[2]) * w)
            y2 = int(float(bbox[3]) * h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        x_lbl = int(float(bbox[0]) * w)
        y_lbl = max(18, int(float(bbox[1]) * h) - 6)
        label = f"{det.get('display_name', det.get('class_id', 'item'))} {det.get('confidence', 0.0):.2f}"
        cv2.putText(canvas, label, (x_lbl, y_lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return canvas


async def extract_bgr_from_request(request):
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file") or form.get("image")
        enable_clahe_raw = form.get("enable_clahe", "true")
        enable_clahe = str(enable_clahe_raw).lower() not in ("false", "0", "no")
        conf_raw = form.get("conf_threshold", str(DEFAULT_CONF_THRESHOLD))
        conf_threshold = float(conf_raw)
        if upload is not None and hasattr(upload, "read"):
            raw_bytes = await upload.read()
            bgr = decode_image_bytes(raw_bytes)
            return bgr, enable_clahe, conf_threshold

    try:
        body = await request.json()
    except Exception:
        body = {}

    if not isinstance(body, dict):
        body = {}

    enable_clahe = bool(body.get("enable_clahe", True))
    conf_threshold = float(body.get("conf_threshold", DEFAULT_CONF_THRESHOLD))

    b64_str = body.get("image_b64") or body.get("image")
    if b64_str:
        bgr = decode_base64_image(b64_str)
        return bgr, enable_clahe, conf_threshold

    sample_name = body.get("sample_name") or "pantry_images_cv_01.jpg"
    sample_path = os.path.join(SAMPLES_DIR, os.path.basename(str(sample_name)))
    if os.path.exists(sample_path):
        bgr = cv2.imread(sample_path)
        if bgr is not None:
            return bgr, enable_clahe, conf_threshold

    raise ValueError("No valid image provided via file upload, image_b64, or sample_name.")


async def handle_health(request: Request):
    ensure_reference_templates(templates_dir=TEMPLATES_DIR, samples_dir=SAMPLES_DIR)
    model_status = get_yolo_model_status()
    model_status["gemini_status"] = get_gemini_detector_status()
    hardware_info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "AMD Ryzen 7 5700G with Radeon Graphics",
        "cpu_threads": int(os.cpu_count() or 16),
        "opencv_version": cv2.__version__,
    }
    return JSONResponse({
        "status": "ok",
        "service": "stock2stove-inference-server",
        "model_status": model_status,
        "hardware": hardware_info,
        "supported_detector_modes": SUPPORTED_DETECTOR_MODES,
        "supported_classes": UNIFIED_CLASSES,
    })


async def handle_scan_image(request: Request):
    try:
        bgr_image, enable_clahe, conf_threshold = await extract_bgr_from_request(request)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    yolo_res = execute_single_frame_pipeline(
        bgr_image,
        detector_mode="yolov8",
        enable_clahe=enable_clahe,
        conf_threshold=conf_threshold,
        session_id=None,
        frame_id=1,
    )
    orb_res = execute_single_frame_pipeline(
        bgr_image,
        detector_mode="orb_baseline",
        enable_clahe=enable_clahe,
        conf_threshold=conf_threshold,
        session_id=None,
        frame_id=1,
    )
    sift_res = execute_single_frame_pipeline(
        bgr_image,
        detector_mode="sift_baseline",
        enable_clahe=enable_clahe,
        conf_threshold=conf_threshold,
        session_id=None,
        frame_id=1,
    )
    gemini_res = execute_single_frame_pipeline(
        bgr_image,
        detector_mode="gemini_vision",
        enable_clahe=enable_clahe,
        conf_threshold=conf_threshold,
        session_id=None,
        frame_id=1,
    )
    hybrid_res = execute_single_frame_pipeline(
        bgr_image,
        detector_mode="hybrid_gemini",
        enable_clahe=enable_clahe,
        conf_threshold=conf_threshold,
        session_id=None,
        frame_id=1,
    )

    yolo_confs = []
    for d in yolo_res["detections"]:
        yolo_confs.append(float(d["confidence"]))
    orb_confs = []
    for d in orb_res["detections"]:
        orb_confs.append(float(d["confidence"]))
    sift_confs = []
    for d in sift_res["detections"]:
        sift_confs.append(float(d["confidence"]))
    gemini_confs = []
    for d in gemini_res["detections"]:
        gemini_confs.append(float(d["confidence"]))
    hybrid_confs = []
    for d in hybrid_res["detections"]:
        hybrid_confs.append(float(d["confidence"]))

    yolo_mean_conf = round(float(np.mean(yolo_confs)), 2) if yolo_confs else 0.0
    orb_mean_conf = round(float(np.mean(orb_confs)), 2) if orb_confs else 0.0
    sift_mean_conf = round(float(np.mean(sift_confs)), 2) if sift_confs else 0.0
    gemini_mean_conf = round(float(np.mean(gemini_confs)), 2) if gemini_confs else 0.0
    hybrid_mean_conf = round(float(np.mean(hybrid_confs)), 2) if hybrid_confs else 0.0

    comparison = {
        "yolov8": {
            "inference_ms": yolo_res["inference_ms"],
            "fps": yolo_res["fps"],
            "detection_count": len(yolo_res["detections"]),
            "mean_confidence": yolo_mean_conf,
            "detections": yolo_res["detections"],
        },
        "gemini_vision": {
            "inference_ms": gemini_res["inference_ms"],
            "fps": gemini_res["fps"],
            "detection_count": len(gemini_res["detections"]),
            "mean_confidence": gemini_mean_conf,
            "detections": gemini_res["detections"],
        },
        "hybrid_gemini": {
            "inference_ms": hybrid_res["inference_ms"],
            "fps": hybrid_res["fps"],
            "detection_count": len(hybrid_res["detections"]),
            "mean_confidence": hybrid_mean_conf,
            "detections": hybrid_res["detections"],
        },
        "orb_baseline": {
            "inference_ms": orb_res["inference_ms"],
            "fps": orb_res["fps"],
            "detection_count": len(orb_res["detections"]),
            "mean_confidence": orb_mean_conf,
            "good_matches": orb_res["diagnostics"].get("good_matches", 0),
            "ransac_inliers": orb_res["diagnostics"].get("ransac_inliers", 0),
            "detections": orb_res["detections"],
        },
        "sift_baseline": {
            "inference_ms": sift_res["inference_ms"],
            "fps": sift_res["fps"],
            "detection_count": len(sift_res["detections"]),
            "mean_confidence": sift_mean_conf,
            "good_matches": sift_res["diagnostics"].get("good_matches", 0),
            "ransac_inliers": sift_res["diagnostics"].get("ransac_inliers", 0),
            "detections": sift_res["detections"],
        },
        "diagnostics": yolo_res["diagnostics"],
    }
    return JSONResponse(comparison)


async def handle_scan_video(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    frames_bgr = []
    detector_mode = "gemini_vision"
    enable_clahe = True
    conf_threshold = DEFAULT_CONF_THRESHOLD

    if "multipart/form-data" in content_type:
        form = await request.form()
        detector_mode = validate_detector_mode(form.get("detector_mode", "gemini_vision"))
        enable_clahe = str(form.get("enable_clahe", "true")).lower() not in ("false", "0", "no")
        conf_threshold = float(form.get("conf_threshold", str(DEFAULT_CONF_THRESHOLD)))
        upload = form.get("file") or form.get("video")
        if upload is not None and hasattr(upload, "read"):
            raw_video = await upload.read()
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            try:
                tmp_file.write(raw_video)
                tmp_file.close()
                cap = cv2.VideoCapture(tmp_file.name)
                total_read = 0
                while cap.isOpened() and len(frames_bgr) < 15:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if total_read % 3 == 0:
                        frames_bgr.append(frame)
                    total_read += 1
                cap.release()
            finally:
                if os.path.exists(tmp_file.name):
                    try:
                        os.remove(tmp_file.name)
                    except OSError:
                        pass
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict):
            detector_mode = validate_detector_mode(body.get("detector_mode", "gemini_vision"))
            enable_clahe = bool(body.get("enable_clahe", True))
            conf_threshold = float(body.get("conf_threshold", DEFAULT_CONF_THRESHOLD))
            frames_b64 = body.get("frames_b64") or []
            for b64_item in frames_b64[:15]:
                try:
                    frames_bgr.append(decode_base64_image(b64_item))
                except Exception:
                    continue

    if len(frames_bgr) == 0:
        for idx in range(1, 6):
            sample_path = os.path.join(SAMPLES_DIR, f"pantry_images_cv_{idx:02d}.jpg")
            if os.path.exists(sample_path):
                img = cv2.imread(sample_path)
                if img is not None:
                    frames_bgr.append(img)

    if len(frames_bgr) == 0:
        return JSONResponse({"error": "No video frames could be decoded."}, status_code=400)

    session_id = f"video_{uuid.uuid4().hex[:8]}"
    reset_session(session_id)

    frame_results = []
    keyframes = []
    final_confirmed = []
    total_estimated_calories = 0

    for idx, frame in enumerate(frames_bgr):
        res = execute_single_frame_pipeline(
            frame,
            detector_mode=detector_mode,
            enable_clahe=enable_clahe,
            conf_threshold=conf_threshold,
            session_id=session_id,
            frame_id=idx + 1,
        )
        final_confirmed = res["confirmed_inventory"]
        total_estimated_calories = int(res.get("total_estimated_calories", 0))
        frame_results.append({
            "frame_id": res["frame_id"],
            "inference_ms": res["inference_ms"],
            "detections": res["detections"],
        })

        if idx == 0 or idx == len(frames_bgr) // 2 or idx == len(frames_bgr) - 1:
            annotated = annotate_frame_visual(frame, res["detections"])
            keyframes.append({
                "frame_id": res["frame_id"],
                "annotated_jpeg_b64": encode_image_base64_jpeg(annotated, quality=80),
                "detection_count": len(res["detections"]),
            })

    remove_session(session_id)

    return JSONResponse({
        "frames_processed": len(frames_bgr),
        "detector_mode": detector_mode,
        "confirmed_inventory": final_confirmed,
        "total_estimated_calories": int(total_estimated_calories),
        "frame_summaries": frame_results,
        "keyframes": keyframes,
    })


async def handle_recipes_generate(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    req_data = parse_recipe_request(body)
    result = generate_structured_recipes(
        req_data["items"],
        api_key=req_data["api_key"],
        timeout_sec=req_data["timeout_sec"],
    )
    return JSONResponse(result)


async def handle_session_reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = "default"
    if isinstance(body, dict) and body.get("session_id"):
        session_id = str(body["session_id"])
    reset_session(session_id)
    return JSONResponse({"status": "reset", "session_id": session_id})


async def handle_session_item_delete_param(class_id: str, session_id: str = "default"):
    cls_id = normalize_class_id(class_id)
    if not cls_id:
        return JSONResponse({"error": "class_id is required."}, status_code=400)
    confirmed_inv, excluded_list = delete_session_item(session_id, cls_id)
    return JSONResponse(
        build_session_item_response("item_deleted", session_id, cls_id, confirmed_inv, excluded_list)
    )


async def handle_session_item_delete_post(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    session_id = str(body.get("session_id") or "default")
    raw_cls = body.get("class_id") or ""
    cls_id = normalize_class_id(raw_cls)
    if not cls_id:
        return JSONResponse({"error": "class_id is required."}, status_code=400)
    confirmed_inv, excluded_list = delete_session_item(session_id, cls_id)
    return JSONResponse(
        build_session_item_response("item_deleted", session_id, cls_id, confirmed_inv, excluded_list)
    )


async def handle_session_item_restore_post(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    session_id = str(body.get("session_id") or "default")
    raw_cls = body.get("class_id") or ""
    cls_id = normalize_class_id(raw_cls)
    if not cls_id:
        return JSONResponse({"error": "class_id is required."}, status_code=400)
    confirmed_inv, excluded_list = restore_session_item(session_id, cls_id)
    return JSONResponse(
        build_session_item_response("item_restored", session_id, cls_id, confirmed_inv, excluded_list)
    )


async def handle_ws_scan(websocket: WebSocket):
    await websocket.accept()
    ws_session_id = f"ws_{uuid.uuid4().hex[:8]}"
    session = get_or_create_session(ws_session_id)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await websocket.send_json({"error": "Invalid JSON payload."})
                    continue

                if isinstance(payload, dict):
                    action = str(payload.get("action") or "").strip().lower()
                    if action == "delete_item":
                        raw_cls = payload.get("class_id") or ""
                        cls_id = normalize_class_id(raw_cls)
                        if not cls_id:
                            await websocket.send_json({"error": "class_id is required for delete_item."})
                            continue
                        confirmed_inv, excluded_list = delete_session_item(ws_session_id, cls_id)
                        await websocket.send_json(
                            build_session_item_response(
                                "item_deleted",
                                ws_session_id,
                                cls_id,
                                confirmed_inv,
                                excluded_list,
                            )
                        )
                        continue
                    if action == "restore_item":
                        raw_cls = payload.get("class_id") or ""
                        cls_id = normalize_class_id(raw_cls)
                        if not cls_id:
                            await websocket.send_json({"error": "class_id is required for restore_item."})
                            continue
                        confirmed_inv, excluded_list = restore_session_item(ws_session_id, cls_id)
                        await websocket.send_json(
                            build_session_item_response(
                                "item_restored",
                                ws_session_id,
                                cls_id,
                                confirmed_inv,
                                excluded_list,
                            )
                        )
                        continue

                parsed = parse_scan_frame_request(payload, session_defaults=session)
                if parsed["reset_session"] or (isinstance(payload, dict) and str(payload.get("action") or "").strip().lower() == "reset_session"):
                    reset_session(ws_session_id)

                update_session_config(
                    ws_session_id,
                    detector_mode=parsed["detector_mode"],
                    enable_clahe=parsed["enable_clahe"],
                    conf_threshold=parsed["conf_threshold"],
                )

                if not parsed["image_b64"]:
                    await websocket.send_json({
                        "status": "config_updated",
                        "session_id": ws_session_id,
                        "detector_mode": session["detector_mode"],
                        "enable_clahe": session["enable_clahe"],
                        "conf_threshold": session["conf_threshold"],
                        "excluded_classes": get_session_excluded_classes(ws_session_id),
                    })
                    continue

                try:
                    bgr_frame = decode_base64_image(parsed["image_b64"])
                except Exception as exc:
                    await websocket.send_json({"error": f"Image decode failed: {exc}"})
                    continue

                response_payload = execute_single_frame_pipeline(
                    bgr_frame,
                    detector_mode=parsed["detector_mode"],
                    enable_clahe=parsed["enable_clahe"],
                    conf_threshold=parsed["conf_threshold"],
                    session_id=ws_session_id,
                    frame_id=parsed["frame_id"],
                )
                await websocket.send_json(response_payload)

            elif "bytes" in message and message["bytes"] is not None:
                raw_bytes = message["bytes"]
                try:
                    bgr_frame = decode_image_bytes(raw_bytes)
                except Exception as exc:
                    await websocket.send_json({"error": f"Binary frame decode failed: {exc}"})
                    continue

                next_frame_id = int(session.get("frame_id", 0)) + 1
                response_payload = execute_single_frame_pipeline(
                    bgr_frame,
                    detector_mode=session.get("detector_mode", "gemini_vision"),
                    enable_clahe=session.get("enable_clahe", True),
                    conf_threshold=session.get("conf_threshold", DEFAULT_CONF_THRESHOLD),
                    session_id=ws_session_id,
                    frame_id=next_frame_id,
                )
                await websocket.send_json(response_payload)
    except WebSocketDisconnect:
        pass
    finally:
        remove_session(ws_session_id)


async def handle_ui_detect(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    detector_mode = "yolov8"
    enable_clahe = True
    conf_threshold = 0.45
    bgr_image = None
    source_name = "upload"
    excluded_set = set()
    req_session_id = None

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            detector_mode = validate_detector_mode(form.get("detector_mode", "yolov8"))
            enable_clahe = str(form.get("enable_clahe", "true")).lower() not in ("false", "0", "no")
            conf_threshold = float(form.get("conf_threshold", "0.45"))
            req_session_id = form.get("session_id")
            if hasattr(form, "getlist"):
                for raw_exc in form.getlist("excluded_classes"):
                    for item in parse_excluded_classes_value(raw_exc):
                        excluded_set.add(item)
            elif "excluded_classes" in form:
                for item in parse_excluded_classes_value(form.get("excluded_classes")):
                    excluded_set.add(item)

            upload = form.get("file") or form.get("image")
            if upload is not None and hasattr(upload, "read"):
                source_name = getattr(upload, "filename", "upload") or "upload"
                raw_bytes = await upload.read()
                bgr_image = decode_image_bytes(raw_bytes)
            else:
                sample_name = str(form.get("sample_name", "pantry_images_cv_01.jpg"))
                source_name = sample_name
                sample_path = os.path.join(SAMPLES_DIR, os.path.basename(sample_name))
                if os.path.exists(sample_path):
                    bgr_image = cv2.imread(sample_path)
        else:
            try:
                body = await request.json()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                body = {}
            detector_mode = validate_detector_mode(body.get("detector_mode", "yolov8"))
            enable_clahe = bool(body.get("enable_clahe", True))
            conf_threshold = float(body.get("conf_threshold", 0.45))
            req_session_id = body.get("session_id")
            for item in parse_excluded_classes_value(body.get("excluded_classes")):
                excluded_set.add(item)

            b64_str = body.get("image_b64") or body.get("image")
            if b64_str:
                bgr_image = decode_base64_image(b64_str)
            else:
                sample_name = str(body.get("sample_name") or "pantry_images_cv_01.jpg")
                source_name = sample_name
                sample_path = os.path.join(SAMPLES_DIR, os.path.basename(sample_name))
                if os.path.exists(sample_path):
                    bgr_image = cv2.imread(sample_path)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    if req_session_id is not None and str(req_session_id).strip():
        for cls_val in get_session_excluded_classes(str(req_session_id).strip()):
            excluded_set.add(cls_val)

    if bgr_image is None or bgr_image.size == 0:
        return JSONResponse({"error": "No valid image provided."}, status_code=400)

    res = run_single_frame(
        bgr_image,
        mode=detector_mode,
        enable_clahe=enable_clahe,
        conf_threshold=conf_threshold,
        source_name=source_name,
    )
    post = res["postprocessed"]
    filtered_dets = filter_valid_detections(post.get("detections", []), excluded_classes=excluded_set)

    counts_by_class = {}
    grouped = {}
    for det in filtered_dets:
        cls_id = det["class_id"]
        conf_val = float(det.get("confidence", 0.0))
        crop_b64 = det.get("crop_jpeg_b64")
        if not crop_b64:
            crop_b64 = crop_detection_base64_jpeg(bgr_image, det.get("bbox_norm"))
            det["crop_jpeg_b64"] = crop_b64
        counts_by_class[cls_id] = counts_by_class.get(cls_id, 0) + 1
        if cls_id not in grouped:
            grouped[cls_id] = {
                "class_id": cls_id,
                "display_name": det.get("display_name") or get_display_name(cls_id),
                "count": 1,
                "max_conf": conf_val,
                "crop_jpeg_b64": crop_b64,
            }
        else:
            grouped[cls_id]["count"] += 1
            if conf_val > grouped[cls_id]["max_conf"]:
                grouped[cls_id]["max_conf"] = conf_val
                if crop_b64:
                    grouped[cls_id]["crop_jpeg_b64"] = crop_b64
            elif not grouped[cls_id].get("crop_jpeg_b64") and crop_b64:
                grouped[cls_id]["crop_jpeg_b64"] = crop_b64

    confirmed_inventory = []
    total_estimated_calories = 0
    for cls_id in sorted(grouped.keys()):
        formatted_item = format_confirmed_inventory_item(grouped[cls_id])
        confirmed_inventory.append(formatted_item)
        total_estimated_calories += int(formatted_item["estimated_calories"])

    filtered_post = {
        "detections": filtered_dets,
        "confirmed_inventory": confirmed_inventory,
        "counts_by_class": counts_by_class,
        "total_count": int(sum(counts_by_class.values())),
    }
    rendered = render(
        bgr_image,
        filtered_post,
        source_name=source_name,
        elapsed_ms=res["elapsed_ms"],
        mode=detector_mode,
    )
    annotated_b64 = encode_image_base64_jpeg(rendered["annotated_frame"], quality=88)

    detections_out = []
    for idx, lbl in enumerate(rendered["labels"]):
        lbl_copy = dict(lbl)
        cls_id = normalize_class_id(lbl_copy.get("class_id") or "")
        lbl_copy["estimated_calories"] = get_class_calories(cls_id, count=1)
        if idx < len(filtered_dets):
            lbl_copy["crop_jpeg_b64"] = filtered_dets[idx].get("crop_jpeg_b64")
        detections_out.append(lbl_copy)

    items_for_recipes = []
    for inv_item in confirmed_inventory:
        items_for_recipes.append({"class_id": inv_item["class_id"], "count": int(inv_item["count"])})

    recipes_payload = {
        "recipes": [],
        "source": "none",
        "latency_ms": 0.0,
        "total_inventory_calories": 0,
    }
    if items_for_recipes:
        recipes_payload = generate_structured_recipes(items_for_recipes, timeout_sec=2.0)

    elapsed_ms = round(float(res["elapsed_ms"]), 2)
    fps = round(1000.0 / max(0.01, float(res["elapsed_ms"])), 2)

    return JSONResponse({
        "source": source_name,
        "detector_mode": detector_mode,
        "enable_clahe": bool(enable_clahe),
        "conf_threshold": float(conf_threshold),
        "inference_ms": elapsed_ms,
        "fps": fps,
        "total_count": int(filtered_post["total_count"]),
        "counts_by_class": counts_by_class,
        "detections": detections_out,
        "confirmed_inventory": confirmed_inventory,
        "total_estimated_calories": int(total_estimated_calories),
        "excluded_classes": sorted(list(excluded_set)),
        "diagnostics": res["preprocessed"]["diagnostics"],
        "annotated_jpeg_b64": annotated_b64,
        "recipes": recipes_payload.get("recipes", []),
        "recipe_source": recipes_payload.get("source", "offline"),
        "total_inventory_calories": int(recipes_payload.get("total_inventory_calories", total_estimated_calories)),
    })


async def handle_web_ui(request: Request):
    sample_options = []
    for idx in range(1, 11):
        fname = f"pantry_images_cv_{idx:02d}.jpg"
        sample_options.append(f'<option value="{fname}">{fname}</option>')
    options_html = "".join(sample_options)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock-to-Stove | Pantry CV Dashboard</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
  header {{ background: #1e293b; padding: 14px 24px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
  header h1 {{ margin: 0; font-size: 18px; color: #38bdf8; }}
  header span {{ font-size: 12px; color: #94a3b8; }}
  .container {{ max-width: 1200px; margin: 18px auto; padding: 0 18px; display: grid; grid-template-columns: 340px 1fr; gap: 18px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; }}
  .card h2 {{ margin: 0 0 10px 0; font-size: 14px; color: #f8fafc; border-bottom: 1px solid #334155; padding-bottom: 6px; }}
  label {{ display: block; font-size: 12px; color: #cbd5e1; margin-top: 8px; margin-bottom: 4px; }}
  select, input[type="file"], input[type="range"], button {{ width: 100%; box-sizing: border-box; padding: 8px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; font-size: 13px; }}
  button {{ background: #0284c7; border-color: #0284c7; font-weight: 600; cursor: pointer; margin-top: 10px; }}
  button:hover {{ background: #0369a1; }}
  button.secondary {{ background: #334155; border-color: #475569; }}
  .row {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px; }}
  .stat-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; text-align: center; }}
  .stat-val {{ font-size: 18px; font-weight: 700; color: #38bdf8; }}
  .stat-lbl {{ font-size: 11px; color: #94a3b8; }}
  .preview-wrap {{ text-align: center; background: #0b1120; border-radius: 8px; padding: 8px; border: 1px solid #334155; min-height: 280px; display: flex; align-items: center; justify-content: center; }}
  .preview-wrap img {{ max-width: 100%; max-height: 460px; border-radius: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  th, td {{ border-bottom: 1px solid #334155; padding: 6px 8px; text-align: left; }}
  th {{ color: #94a3b8; }}
  .recipe-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; }}
  .recipe-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; font-size: 12px; }}
  .recipe-card h3 {{ margin: 0 0 6px 0; font-size: 13px; color: #38bdf8; }}
  video {{ width: 100%; border-radius: 6px; margin-top: 8px; display: none; background: #000; }}
</style>
</head>
<body>
<header>
  <h1>Stock-to-Stove | Pantry Vision Application</h1>
  <span>Model cached once in server memory (YOLOv8n / Gemini Vision / ORB / SIFT)</span>
</header>
<div class="container">
  <div class="card">
    <h2>1. Input & Detector Settings</h2>
    <label>Detector Mode</label>
    <select id="mode">
      <option value="yolov8" selected>YOLOv8n Fine-Tuned (Main)</option>
      <option value="hybrid_gemini">Hybrid YOLOv8n + Gemini Vision</option>
      <option value="gemini_vision">Gemini Vision (Cloud / Offline Fallback)</option>
      <option value="orb_baseline">ORB + RANSAC (Baseline)</option>
      <option value="sift_baseline">SIFT + RANSAC (Baseline)</option>
    </select>
    <label>Confidence Threshold: <span id="confVal">0.45</span></label>
    <input type="range" id="conf" min="0.20" max="0.85" step="0.05" value="0.45" oninput="document.getElementById('confVal').innerText=this.value">
    <div class="row">
      <input type="checkbox" id="clahe" checked style="width:auto;">
      <label for="clahe" style="margin:0;">Luminance-Gated Low-Light CLAHE (&lt;100 lux)</label>
    </div>

    <label style="margin-top:14px;">Upload Pantry/Fridge Image</label>
    <input type="file" id="imageFile" accept="image/*">
    <button onclick="runImageUpload()">Detect Uploaded Image</button>

    <label style="margin-top:14px;">Or Select Built-In Sample</label>
    <select id="sampleSelect">{options_html}</select>
    <button class="secondary" onclick="runSample()">Detect Sample Image</button>

    <label style="margin-top:14px;">Upload Video Clip (.mp4)</label>
    <input type="file" id="videoFile" accept="video/*">
    <button class="secondary" onclick="runVideoUpload()">Analyze Video Clip</button>

    <label style="margin-top:14px;">Live Webcam Snapshot</label>
    <div class="row">
      <button class="secondary" onclick="toggleWebcam()">Start/Stop Cam</button>
      <button onclick="captureWebcam()">Capture & Detect</button>
    </div>
    <video id="camVideo" autoplay playsinline></video>
    <canvas id="camCanvas" style="display:none;"></canvas>
  </div>

  <div>
    <div class="stats">
      <div class="stat-box"><div class="stat-val" id="statCount">0</div><div class="stat-lbl">Items Detected</div></div>
      <div class="stat-box"><div class="stat-val" id="statMs">-- ms</div><div class="stat-lbl">Processing Time</div></div>
      <div class="stat-box"><div class="stat-val" id="statFps">-- FPS</div><div class="stat-lbl">Throughput</div></div>
      <div class="stat-box"><div class="stat-val" id="statMode">yolov8</div><div class="stat-lbl">Active Detector</div></div>
    </div>

    <div class="card">
      <h2>2. Annotated Detection Overlay</h2>
      <div class="preview-wrap"><img id="resultImg" alt="Detection overlay preview will appear here"></div>
    </div>

    <div class="card" style="margin-top:14px;">
      <h2>3. Detections, Confidence & Counts</h2>
      <div id="countsSummary" style="font-size:13px; color:#cbd5e1; margin-bottom:6px;">No scan run yet.</div>
      <table>
        <thead><tr><th>#</th><th>Class ID</th><th>Display Name</th><th>Confidence</th><th>Normalized Box [x1, y1, x2, y2]</th></tr></thead>
        <tbody id="detTable"></tbody>
      </table>
    </div>

    <div class="card" style="margin-top:14px;">
      <h2>4. Zero-Waste Recipe Suggestions</h2>
      <div class="recipe-grid" id="recipeGrid"></div>
    </div>
  </div>
</div>

<script>
let camStream = null;
async function renderDetectionResponse(data) {{
  if (data.error) {{ alert(data.error); return; }}
  document.getElementById('statCount').innerText = data.total_count ?? 0;
  document.getElementById('statMs').innerText = (data.inference_ms ?? 0) + ' ms';
  document.getElementById('statFps').innerText = (data.fps ?? 0) + ' FPS';
  document.getElementById('statMode').innerText = data.detector_mode || '--';
  if (data.annotated_jpeg_b64) {{
    document.getElementById('resultImg').src = 'data:image/jpeg;base64,' + data.annotated_jpeg_b64;
  }}
  const counts = data.counts_by_class || {{}};
  const countStr = Object.entries(counts).map(([k, v]) => `${{k}}: ${{v}}`).join(' | ') || 'None';
  document.getElementById('countsSummary').innerText = 'Counts by class: ' + countStr;

  const tbody = document.getElementById('detTable');
  tbody.innerHTML = '';
  (data.detections || []).forEach((d, idx) => {{
    const tr = document.createElement('tr');
    const boxStr = (d.bbox_norm || []).map(v => Number(v).toFixed(3)).join(', ');
    tr.innerHTML = `<td>${{idx + 1}}</td><td>${{d.class_id}}</td><td>${{d.display_name}}</td><td>${{Number(d.confidence).toFixed(2)}}</td><td>[${{boxStr}}]</td>`;
    tbody.appendChild(tr);
  }});

  const rGrid = document.getElementById('recipeGrid');
  rGrid.innerHTML = '';
  (data.recipes || []).forEach(r => {{
    const div = document.createElement('div');
    div.className = 'recipe-card';
    const stepsHtml = (r.steps || []).slice(0, 3).map(s => `<li>${{s}}</li>`).join('');
    div.innerHTML = `<h3>${{r.title}}</h3><div>Prep: ${{r.prep_time_mins}} mins</div><ol style="padding-left:16px;margin:6px 0;">${{stepsHtml}}</ol>`;
    rGrid.appendChild(div);
  }});
}}

async function runImageUpload() {{
  const fileInput = document.getElementById('imageFile');
  if (!fileInput.files.length) {{ alert('Select an image file first.'); return; }}
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('detector_mode', document.getElementById('mode').value);
  fd.append('conf_threshold', document.getElementById('conf').value);
  fd.append('enable_clahe', document.getElementById('clahe').checked ? 'true' : 'false');
  const res = await fetch('/api/v1/ui/detect', {{ method: 'POST', body: fd }});
  await renderDetectionResponse(await res.json());
}}

async function runSample() {{
  const payload = {{
    sample_name: document.getElementById('sampleSelect').value,
    detector_mode: document.getElementById('mode').value,
    conf_threshold: parseFloat(document.getElementById('conf').value),
    enable_clahe: document.getElementById('clahe').checked
  }};
  const res = await fetch('/api/v1/ui/detect', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(payload)
  }});
  await renderDetectionResponse(await res.json());
}}

async function runVideoUpload() {{
  const fileInput = document.getElementById('videoFile');
  if (!fileInput.files.length) {{ alert('Select a video file first.'); return; }}
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('detector_mode', document.getElementById('mode').value);
  fd.append('conf_threshold', document.getElementById('conf').value);
  fd.append('enable_clahe', document.getElementById('clahe').checked ? 'true' : 'false');
  const res = await fetch('/api/v1/scan/video', {{ method: 'POST', body: fd }});
  const data = await res.json();
  if (data.error) {{ alert(data.error); return; }}
  const kf = (data.keyframes && data.keyframes.length) ? data.keyframes[0] : null;
  const inv = data.confirmed_inventory || [];
  const counts = {{}};
  let total = 0;
  inv.forEach(item => {{ counts[item.class_id] = item.count; total += item.count; }});
  await renderDetectionResponse({{
    detector_mode: data.detector_mode,
    inference_ms: data.frame_summaries?.[0]?.inference_ms ?? 0,
    fps: data.frame_summaries?.[0]?.inference_ms ? (1000.0 / data.frame_summaries[0].inference_ms).toFixed(1) : 0,
    total_count: total,
    counts_by_class: counts,
    detections: data.frame_summaries?.[0]?.detections ?? [],
    annotated_jpeg_b64: kf ? kf.annotated_jpeg_b64 : null,
    recipes: []
  }});
}}

async function toggleWebcam() {{
  const vid = document.getElementById('camVideo');
  if (camStream) {{
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
    vid.style.display = 'none';
    return;
  }}
  camStream = await navigator.mediaDevices.getUserMedia({{ video: true }});
  vid.srcObject = camStream;
  vid.style.display = 'block';
}}

async function captureWebcam() {{
  const vid = document.getElementById('camVideo');
  if (!camStream || !vid.videoWidth) {{ alert('Start webcam first.'); return; }}
  const canvas = document.getElementById('camCanvas');
  canvas.width = vid.videoWidth;
  canvas.height = vid.videoHeight;
  canvas.getContext('2d').drawImage(vid, 0, 0);
  const b64 = canvas.toDataURL('image/jpeg', 0.88);
  const res = await fetch('/api/v1/ui/detect', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{
      image_b64: b64,
      detector_mode: document.getElementById('mode').value,
      conf_threshold: parseFloat(document.getElementById('conf').value),
      enable_clahe: document.getElementById('clahe').checked
    }})
  }});
  await renderDetectionResponse(await res.json());
}}

window.addEventListener('DOMContentLoaded', runSample);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


async def handle_root(request: Request):
    index_path = os.path.join(FRONTEND_DIST_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return await handle_web_ui(request)


def create_app():
    ensure_reference_templates(templates_dir=TEMPLATES_DIR, samples_dir=SAMPLES_DIR)
    application = FastAPI(title="Stock-to-Stove Local Inference Server", version="1.0.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_api_route("/", handle_root, methods=["GET"])
    application.add_api_route("/api/v1/ui/detect", handle_ui_detect, methods=["POST"])
    application.add_api_route("/api/v1/health", handle_health, methods=["GET"])
    application.add_api_route("/api/v1/scan/image", handle_scan_image, methods=["POST"])
    application.add_api_route("/api/v1/scan/video", handle_scan_video, methods=["POST"])
    application.add_api_route("/api/v1/recipes/generate", handle_recipes_generate, methods=["POST"])
    application.add_api_route("/api/v1/session/reset", handle_session_reset, methods=["POST"])
    application.add_api_route("/api/v1/session/items/delete", handle_session_item_delete_post, methods=["POST"])
    application.add_api_route("/api/v1/session/items/restore", handle_session_item_restore_post, methods=["POST"])
    application.add_api_route("/api/v1/session/items/{class_id}", handle_session_item_delete_param, methods=["DELETE"])
    application.add_api_websocket_route("/ws/scan", handle_ws_scan)
    assets_dir = os.path.join(FRONTEND_DIST_DIR, "assets")
    application.mount("/assets", StaticFiles(directory=assets_dir, check_dir=False), name="frontend_assets")
    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server.app:app",
        host=HOST,
        port=PORT,
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
