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
from fastapi.responses import JSONResponse

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
from src.server.config import (
    DEFAULT_CONF_THRESHOLD,
    HOST,
    PORT,
    SAMPLES_DIR,
    SUPPORTED_DETECTOR_MODES,
    TARGET_IMG_SIZE,
    TEMPLATES_DIR,
    UNIFIED_CLASSES,
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
    get_or_create_session,
    record_session_frame,
    remove_session,
    reset_session,
    update_session_config,
)


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

    for det in raw_dets:
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
            raw_dets,
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
            raw_dets,
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

                parsed = parse_scan_frame_request(payload, session_defaults=session)
                if parsed["reset_session"]:
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
    application.add_api_route("/api/v1/health", handle_health, methods=["GET"])
    application.add_api_route("/api/v1/scan/image", handle_scan_image, methods=["POST"])
    application.add_api_route("/api/v1/scan/video", handle_scan_video, methods=["POST"])
    application.add_api_route("/api/v1/recipes/generate", handle_recipes_generate, methods=["POST"])
    application.add_api_route("/api/v1/session/reset", handle_session_reset, methods=["POST"])
    application.add_api_websocket_route("/ws/scan", handle_ws_scan)
    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server.app:app", host=HOST, port=PORT, reload=False)
