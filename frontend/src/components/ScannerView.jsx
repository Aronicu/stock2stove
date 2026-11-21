import { useEffect, useRef, useState } from 'react'
import {
  Camera,
  CameraOff,
  Film,
  Image as ImageIcon,
  Layers,
  Pause,
  Play,
  RotateCcw,
  ShoppingBasket,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react'
import {
  buildScanWsUrl,
  detectImageFile,
  detectSampleImage,
  detectSnapshotBase64,
  scanVideoClip,
} from '../api/client.js'
import {
  BUILTIN_SAMPLES,
  DETECTOR_MODES,
  formatCropDataUrl,
  getCaloriesPerUnit,
  getClassMeta,
  normalizeClassId,
} from '../constants/catalog.js'

function drawDetectionOverlays(canvas, detections) {
  if (!canvas) {
    return
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return
  }
  const w = canvas.width
  const h = canvas.height
  ctx.clearRect(0, 0, w, h)
  if (!detections || detections.length === 0 || w <= 0 || h <= 0) {
    return
  }

  for (let i = 0; i < detections.length; i += 1) {
    const det = detections[i]
    const isConfirmed = Boolean(det.temporally_confirmed)
    const strokeColor = isConfirmed ? '#28C850' : '#FFB300'
    const fillTint = isConfirmed ? 'rgba(40, 200, 80, 0.14)' : 'rgba(255, 179, 0, 0.14)'
    const bbox = Array.isArray(det.bbox_norm) ? det.bbox_norm : [0, 0, 0, 0]
    const poly = Array.isArray(det.polygon_norm) ? det.polygon_norm : null

    let labelAnchorX = 0
    let labelAnchorY = 0

    if (poly && poly.length >= 3) {
      ctx.beginPath()
      let minX = w
      let minY = h
      for (let p = 0; p < poly.length; p += 1) {
        const px = Math.max(0, Math.min(w, Number(poly[p][0]) * w))
        const py = Math.max(0, Math.min(h, Number(poly[p][1]) * h))
        if (px < minX) minX = px
        if (py < minY) minY = py
        if (p === 0) {
          ctx.moveTo(px, py)
        } else {
          ctx.lineTo(px, py)
        }
      }
      ctx.closePath()
      ctx.fillStyle = fillTint
      ctx.fill()
      ctx.lineWidth = 3
      ctx.strokeStyle = strokeColor
      ctx.stroke()
      labelAnchorX = minX
      labelAnchorY = minY
    } else if (bbox.length >= 4) {
      const x1 = Math.max(0, Math.min(w, Number(bbox[0]) * w))
      const y1 = Math.max(0, Math.min(h, Number(bbox[1]) * h))
      const x2 = Math.max(0, Math.min(w, Number(bbox[2]) * w))
      const y2 = Math.max(0, Math.min(h, Number(bbox[3]) * h))
      const left = Math.min(x1, x2)
      const top = Math.min(y1, y2)
      const boxW = Math.max(1, Math.abs(x2 - x1))
      const boxH = Math.max(1, Math.abs(y2 - y1))

      ctx.fillStyle = fillTint
      ctx.fillRect(left, top, boxW, boxH)
      ctx.lineWidth = 3
      ctx.strokeStyle = strokeColor
      ctx.strokeRect(left, top, boxW, boxH)
      labelAnchorX = left
      labelAnchorY = top
    }

    const name = det.display_name || getClassMeta(det.class_id).display_name
    const pct = Math.round(Number(det.confidence || 0) * 100)
    const kcal = det.estimated_calories != null ? det.estimated_calories : getCaloriesPerUnit(det.class_id)
    const text = `${name} ${pct}% · ${kcal} kcal`

    ctx.font = 'bold 13px system-ui, -apple-system, sans-serif'
    const metrics = ctx.measureText(text)
    const pillW = metrics.width + 16
    const pillH = 24
    const pillX = Math.max(2, Math.min(w - pillW - 2, labelAnchorX))
    const pillY = Math.max(2, labelAnchorY - pillH - 4)

    ctx.fillStyle = 'rgba(17, 25, 21, 0.88)'
    ctx.fillRect(pillX, pillY, pillW, pillH)
    ctx.lineWidth = 1.5
    ctx.strokeStyle = strokeColor
    ctx.strokeRect(pillX, pillY, pillW, pillH)

    ctx.fillStyle = '#FFFFFF'
    ctx.fillText(text, pillX + 8, pillY + 16)
  }
}

export default function ScannerView({
  sessionId,
  detectorMode,
  onChangeDetectorMode,
  enableClahe,
  onToggleClahe,
  confThreshold,
  onChangeConfThreshold,
  excludedClasses,
  basketItems,
  onScanUpdate,
  onDeleteItem,
  onResetSession,
  onGoToBasket,
  onGenerateRecipes,
  onRegisterWsSend,
  onLogEvent,
}) {
  const [inputMode, setInputMode] = useState('sample')
  const [selectedSample, setSelectedSample] = useState(BUILTIN_SAMPLES[0].value)
  const [selectedImageFile, setSelectedImageFile] = useState(null)
  const [selectedVideoFile, setSelectedVideoFile] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [previewData, setPreviewData] = useState(null)
  const [selectedKeyframeIdx, setSelectedKeyframeIdx] = useState(0)

  const [cameraActive, setCameraActive] = useState(false)
  const [cameraPaused, setCameraPaused] = useState(false)
  const [wsConnected, setWsConnected] = useState(false)
  const [videoAspect, setVideoAspect] = useState('4 / 3')
  const [liveDetections, setLiveDetections] = useState([])
  const [liveMetrics, setLiveMetrics] = useState({
    fps: 0,
    inference_ms: 0,
    diagnostics: {
      edge_density_pct: 0,
      orb_keypoints: 0,
      clahe_applied: true,
      good_matches: 0,
      ransac_inliers: 0,
    },
  })

  const videoRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const captureCanvasRef = useRef(null)
  const streamRef = useRef(null)
  const wsRef = useRef(null)
  const inFlightRef = useRef(false)
  const lastSentAtRef = useRef(0)
  const frameCounterRef = useRef(0)
  const loopTimerRef = useRef(null)
  const excludedRef = useRef(excludedClasses)
  const configRef = useRef({ detectorMode, enableClahe, confThreshold, cameraPaused, sessionId })

  useEffect(function syncConfigRefs() {
    excludedRef.current = excludedClasses
    configRef.current = { detectorMode, enableClahe, confThreshold, cameraPaused, sessionId }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        frame_id: frameCounterRef.current,
        detector_mode: detectorMode,
        enable_clahe: Boolean(enableClahe),
        conf_threshold: Number(confThreshold),
        image_b64: null,
        session_id: sessionId,
        reset_session: false,
      }))
    }
  }, [detectorMode, enableClahe, confThreshold, cameraPaused, sessionId, excludedClasses])

  useEffect(function redrawLiveOverlay() {
    const canvas = overlayCanvasRef.current
    const video = videoRef.current
    if (!canvas || !video) {
      return
    }
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }
    }
    const filtered = liveDetections.filter(function keepVisible(det) {
      const norm = normalizeClassId(det.class_id)
      return !excludedClasses.includes(norm)
    })
    drawDetectionOverlays(canvas, filtered)
  }, [liveDetections, excludedClasses])

  function stopCameraAndSocket() {
    if (loopTimerRef.current) {
      window.clearInterval(loopTimerRef.current)
      loopTimerRef.current = null
    }
    if (wsRef.current) {
      try {
        wsRef.current.close(1000, 'Camera stopped')
      } catch {}
      wsRef.current = null
    }
    if (onRegisterWsSend) {
      onRegisterWsSend(null)
    }
    if (streamRef.current) {
      const tracks = streamRef.current.getTracks()
      for (let i = 0; i < tracks.length; i += 1) {
        tracks[i].stop()
      }
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    inFlightRef.current = false
    setWsConnected(false)
    setCameraActive(false)
    setCameraPaused(false)
    setLiveDetections([])
  }

  useEffect(function cleanupOnUnmount() {
    return function releaseResources() {
      stopCameraAndSocket()
    }
  }, [])

  function openScanWebSocket() {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return
    }
    const wsUrl = buildScanWsUrl()
    const socket = new WebSocket(wsUrl)
    wsRef.current = socket

    socket.onopen = function handleWsOpen() {
      setWsConnected(true)
      setErrorMessage('')
      if (onLogEvent) {
        onLogEvent('INFO', 'WS', `Connected to ${wsUrl}`)
      }
      if (onRegisterWsSend) {
        onRegisterWsSend(function sendWsJson(payloadObj) {
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(payloadObj))
          }
        })
      }
      const cfg = configRef.current
      socket.send(JSON.stringify({
        frame_id: 0,
        detector_mode: cfg.detectorMode,
        enable_clahe: Boolean(cfg.enableClahe),
        conf_threshold: Number(cfg.confThreshold),
        image_b64: null,
        session_id: cfg.sessionId,
        reset_session: true,
      }))
      const currentExcluded = excludedRef.current || []
      for (let i = 0; i < currentExcluded.length; i += 1) {
        socket.send(JSON.stringify({
          action: 'delete_item',
          class_id: currentExcluded[i],
        }))
      }
    }

    socket.onmessage = function handleWsMessage(event) {
      inFlightRef.current = false
      let payload = null
      try {
        payload = JSON.parse(event.data)
      } catch {
        return
      }
      if (!payload) {
        return
      }
      if (payload.error) {
        setErrorMessage(payload.error)
        if (onLogEvent) {
          onLogEvent('ERROR', 'WS', payload.error)
        }
        return
      }
      if (payload.status === 'config_updated') {
        return
      }
      if (payload.status === 'item_deleted' || payload.status === 'item_restored') {
        onScanUpdate({
          confirmed_inventory: payload.confirmed_inventory || [],
          excluded_classes: payload.excluded_classes || [],
          replaceInventory: true,
        })
        return
      }

      const rawDets = Array.isArray(payload.detections) ? payload.detections : []
      const activeExcluded = excludedRef.current || []
      const filteredDets = rawDets.filter(function filterDet(d) {
        return !activeExcluded.includes(normalizeClassId(d.class_id))
      })

      setLiveDetections(filteredDets)
      setLiveMetrics({
        fps: Number(payload.fps || 0),
        inference_ms: Number(payload.inference_ms || 0),
        diagnostics: payload.diagnostics || {},
      })

      const inv = Array.isArray(payload.confirmed_inventory) ? payload.confirmed_inventory : []
      if (inv.length > 0 || (payload.excluded_classes && payload.excluded_classes.length > 0)) {
        onScanUpdate({
          confirmed_inventory: inv,
          excluded_classes: payload.excluded_classes || [],
          replaceInventory: false,
        })
      }
      if (onLogEvent) {
        onLogEvent(
          'TELEMETRY',
          'WS_FRAME',
          `#${payload.frame_id} ${payload.detector_mode} ${payload.inference_ms}ms (${payload.fps} FPS) dets=${filteredDets.length}`
        )
      }
    }

    socket.onerror = function handleWsError() {
      inFlightRef.current = false
      setWsConnected(false)
      setErrorMessage('WebSocket connection error. Verify FastAPI server is running on port 8000.')
      if (onLogEvent) {
        onLogEvent('ERROR', 'WS', 'WebSocket stream error')
      }
    }

    socket.onclose = function handleWsClose() {
      inFlightRef.current = false
      setWsConnected(false)
      if (onRegisterWsSend) {
        onRegisterWsSend(null)
      }
    }
  }

  async function startLiveCamera() {
    setErrorMessage('')
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage('Webcam capture requires a modern browser with getUserMedia support.')
      return
    }
    try {
      let mediaStream = null
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        })
      } catch {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      }
      streamRef.current = mediaStream
      setCameraActive(true)
      setCameraPaused(false)
      openScanWebSocket()

      window.setTimeout(function attachStreamToVideo() {
        const video = videoRef.current
        if (!video) {
          return
        }
        video.srcObject = mediaStream
        video.onloadedmetadata = function handleMetaLoaded() {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            setVideoAspect(`${video.videoWidth} / ${video.videoHeight}`)
          }
        }
      }, 40)

      if (loopTimerRef.current) {
        window.clearInterval(loopTimerRef.current)
      }
      loopTimerRef.current = window.setInterval(function streamFrameTick() {
        const video = videoRef.current
        const socket = wsRef.current
        const cfg = configRef.current
        if (!video || !socket || socket.readyState !== WebSocket.OPEN) {
          return
        }
        if (cfg.cameraPaused || !video.videoWidth || !video.videoHeight) {
          return
        }
        const now = performance.now()
        const isGemini = cfg.detectorMode === 'gemini_vision' || cfg.detectorMode === 'hybrid_gemini'
        const maxWaitMs = isGemini ? 4500 : 1200
        const minIntervalMs = isGemini ? 900 : 180
        if (inFlightRef.current && now - lastSentAtRef.current < maxWaitMs) {
          return
        }
        if (now - lastSentAtRef.current < minIntervalMs) {
          return
        }

        let capCanvas = captureCanvasRef.current
        if (!capCanvas) {
          capCanvas = document.createElement('canvas')
          captureCanvasRef.current = capCanvas
        }
        const targetW = Math.min(640, video.videoWidth)
        const targetH = Math.round((video.videoHeight / video.videoWidth) * targetW)
        capCanvas.width = targetW
        capCanvas.height = targetH
        const capCtx = capCanvas.getContext('2d')
        if (!capCtx) {
          return
        }
        capCtx.drawImage(video, 0, 0, targetW, targetH)
        const dataUrl = capCanvas.toDataURL('image/jpeg', 0.8)

        frameCounterRef.current += 1
        inFlightRef.current = true
        lastSentAtRef.current = now

        socket.send(JSON.stringify({
          frame_id: frameCounterRef.current,
          detector_mode: cfg.detectorMode,
          enable_clahe: Boolean(cfg.enableClahe),
          conf_threshold: Number(cfg.confThreshold),
          image_b64: dataUrl,
          session_id: cfg.sessionId,
          reset_session: false,
        }))
      }, 150)
    } catch (err) {
      setErrorMessage(err.message || 'Unable to access webcam device.')
      stopCameraAndSocket()
    }
  }

  async function handleWebcamSnapshotDetect() {
    const video = videoRef.current
    if (!video || !video.videoWidth || !video.videoHeight) {
      setErrorMessage('Start the webcam first before taking a snapshot.')
      return
    }
    setIsProcessing(true)
    setErrorMessage('')
    try {
      let capCanvas = captureCanvasRef.current
      if (!capCanvas) {
        capCanvas = document.createElement('canvas')
        captureCanvasRef.current = capCanvas
      }
      capCanvas.width = video.videoWidth
      capCanvas.height = video.videoHeight
      capCanvas.getContext('2d').drawImage(video, 0, 0)
      const dataUrl = capCanvas.toDataURL('image/jpeg', 0.88)
      const result = await detectSnapshotBase64({
        imageB64: dataUrl,
        detectorMode,
        enableClahe,
        confThreshold,
        sessionId,
        excludedClasses,
      })
      applySingleFramePreview(result, 'Webcam Snapshot')
    } catch (err) {
      setErrorMessage(err.message || 'Snapshot detection failed.')
    } finally {
      setIsProcessing(false)
    }
  }

  function applySingleFramePreview(result, label) {
    const filteredDets = (result.detections || []).filter(function keepDet(d) {
      return !excludedClasses.includes(normalizeClassId(d.class_id))
    })
    setPreviewData({
      label,
      detector_mode: result.detector_mode || detectorMode,
      inference_ms: Number(result.inference_ms || 0),
      fps: Number(result.fps || 0),
      total_count: Number(result.total_count || filteredDets.length),
      counts_by_class: result.counts_by_class || {},
      detections: filteredDets,
      diagnostics: result.diagnostics || {},
      keyframes: result.annotated_jpeg_b64
        ? [{ frame_id: 1, annotated_jpeg_b64: result.annotated_jpeg_b64, detection_count: filteredDets.length }]
        : [],
      total_estimated_calories: Number(result.total_estimated_calories || 0),
    })
    setSelectedKeyframeIdx(0)
    onScanUpdate({
      confirmed_inventory: result.confirmed_inventory || [],
      excluded_classes: result.excluded_classes || [],
      recipes: result.recipes && result.recipes.length > 0 ? result : null,
      replaceInventory: false,
    })
    if (onLogEvent) {
      onLogEvent(
        'INFO',
        'DETECT',
        `${label}: ${filteredDets.length} items (${result.inference_ms} ms, mode=${result.detector_mode})`
      )
    }
  }

  async function handleUploadImageSubmit() {
    if (!selectedImageFile) {
      setErrorMessage('Select an image file first.')
      return
    }
    setIsProcessing(true)
    setErrorMessage('')
    try {
      const result = await detectImageFile({
        file: selectedImageFile,
        detectorMode,
        enableClahe,
        confThreshold,
        sessionId,
        excludedClasses,
      })
      applySingleFramePreview(result, selectedImageFile.name)
    } catch (err) {
      setErrorMessage(err.message || 'Image detection failed.')
    } finally {
      setIsProcessing(false)
    }
  }

  async function handleSampleDetectSubmit() {
    setIsProcessing(true)
    setErrorMessage('')
    try {
      const result = await detectSampleImage({
        sampleName: selectedSample,
        detectorMode,
        enableClahe,
        confThreshold,
        sessionId,
        excludedClasses,
      })
      applySingleFramePreview(result, selectedSample)
    } catch (err) {
      setErrorMessage(err.message || 'Sample detection failed.')
    } finally {
      setIsProcessing(false)
    }
  }

  async function handleVideoUploadSubmit() {
    if (!selectedVideoFile) {
      setErrorMessage('Select a video file (.mp4 / .mov) first.')
      return
    }
    setIsProcessing(true)
    setErrorMessage('')
    try {
      const result = await scanVideoClip({
        file: selectedVideoFile,
        detectorMode,
        enableClahe,
        confThreshold,
      })
      const firstSummary = (result.frame_summaries && result.frame_summaries[0]) || { detections: [], inference_ms: 0 }
      const rawDets = firstSummary.detections || []
      const filteredDets = rawDets.filter(function keepDet(d) {
        return !excludedClasses.includes(normalizeClassId(d.class_id))
      })
      const inv = result.confirmed_inventory || []
      const infMs = Number(firstSummary.inference_ms || 0)
      const fpsVal = infMs > 0 ? Number((1000 / infMs).toFixed(1)) : 0
      let totalCnt = 0
      for (let i = 0; i < inv.length; i += 1) {
        totalCnt += Number(inv[i].count || 1)
      }
      setPreviewData({
        label: `${selectedVideoFile.name} (${result.frames_processed || 0} frames)`,
        detector_mode: result.detector_mode || detectorMode,
        inference_ms: infMs,
        fps: fpsVal,
        total_count: totalCnt,
        detections: filteredDets,
        diagnostics: {},
        keyframes: result.keyframes || [],
        total_estimated_calories: Number(result.total_estimated_calories || 0),
      })
      setSelectedKeyframeIdx(0)
      onScanUpdate({
        confirmed_inventory: inv,
        excluded_classes: [],
        replaceInventory: false,
      })
      if (onLogEvent) {
        onLogEvent(
          'INFO',
          'VIDEO',
          `Processed ${result.frames_processed} frames from ${selectedVideoFile.name}`
        )
      }
    } catch (err) {
      setErrorMessage(err.message || 'Video scan failed.')
    } finally {
      setIsProcessing(false)
    }
  }

  function handleInputTabChange(nextMode) {
    if (nextMode !== 'camera' && cameraActive) {
      stopCameraAndSocket()
    }
    setInputMode(nextMode)
    setErrorMessage('')
  }

  async function handleResetClick() {
    setPreviewData(null)
    setLiveDetections([])
    frameCounterRef.current = 0
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        frame_id: 0,
        detector_mode: detectorMode,
        enable_clahe: Boolean(enableClahe),
        conf_threshold: Number(confThreshold),
        image_b64: null,
        session_id: sessionId,
        reset_session: true,
      }))
    }
    await onResetSession()
  }

  function handleDeleteDetection(classId) {
    const norm = normalizeClassId(classId)
    setLiveDetections(function removeFromLive(prev) {
      return prev.filter(function keepItem(d) {
        return normalizeClassId(d.class_id) !== norm
      })
    })
    setPreviewData(function removeFromPreview(prev) {
      if (!prev) {
        return prev
      }
      return {
        ...prev,
        detections: (prev.detections || []).filter(function keepItem(d) {
          return normalizeClassId(d.class_id) !== norm
        }),
      }
    })
    onDeleteItem(norm)
  }

  const rawDisplayedDetections = inputMode === 'camera'
    ? liveDetections
    : (previewData ? previewData.detections : [])

  const displayedDetections = rawDisplayedDetections.filter(function keepNonExcluded(det) {
    return !excludedClasses.includes(normalizeClassId(det.class_id))
  })

  const activeStats = inputMode === 'camera'
    ? {
        count: displayedDetections.length,
        inferenceMs: liveMetrics.inference_ms,
        fps: liveMetrics.fps,
        mode: detectorMode,
        diag: liveMetrics.diagnostics || {},
      }
    : {
        count: previewData ? displayedDetections.length : 0,
        inferenceMs: previewData ? previewData.inference_ms : 0,
        fps: previewData ? previewData.fps : 0,
        mode: previewData ? previewData.detector_mode : detectorMode,
        diag: previewData ? previewData.diagnostics || {} : {},
      }

  const activeKeyframe = previewData && previewData.keyframes && previewData.keyframes.length > 0
    ? previewData.keyframes[Math.min(selectedKeyframeIdx, previewData.keyframes.length - 1)]
    : null

  return (
    <div className="scanner-layout">
      <aside className="card controls-panel">
        <div className="panel-heading">
          <h2>Detector &amp; Input Controls</h2>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={handleResetClick}
            title="Reset session & clear excluded items"
          >
            <RotateCcw size={15} />
            <span>Reset Session</span>
          </button>
        </div>

        <div className="input-mode-grid" role="tablist" aria-label="Scanner input modes">
          <button
            type="button"
            className={`mode-chip ${inputMode === 'camera' ? 'mode-chip-active' : ''}`}
            onClick={function selectCam() { handleInputTabChange('camera') }}
          >
            <Camera size={15} />
            <span>Camera</span>
          </button>
          <button
            type="button"
            className={`mode-chip ${inputMode === 'upload' ? 'mode-chip-active' : ''}`}
            onClick={function selectUpload() { handleInputTabChange('upload') }}
          >
            <Upload size={15} />
            <span>Upload</span>
          </button>
          <button
            type="button"
            className={`mode-chip ${inputMode === 'sample' ? 'mode-chip-active' : ''}`}
            onClick={function selectSample() { handleInputTabChange('sample') }}
          >
            <ImageIcon size={15} />
            <span>Samples</span>
          </button>
          <button
            type="button"
            className={`mode-chip ${inputMode === 'video' ? 'mode-chip-active' : ''}`}
            onClick={function selectVideo() { handleInputTabChange('video') }}
          >
            <Film size={15} />
            <span>Video</span>
          </button>
        </div>

        <div className="control-group">
          <label htmlFor="detector-mode-select">Detector Mode</label>
          <select
            id="detector-mode-select"
            value={detectorMode}
            onChange={function handleModeSelect(e) { onChangeDetectorMode(e.target.value) }}
          >
            {DETECTOR_MODES.map(function renderModeOption(m) {
              return (
                <option key={m.value} value={m.value}>
                  {m.label} ({m.value})
                </option>
              )
            })}
          </select>
        </div>

        <div className="control-group">
          <div className="slider-header">
            <label htmlFor="conf-slider">Confidence Threshold</label>
            <span className="mono-badge">{Math.round(confThreshold * 100)}%</span>
          </div>
          <input
            id="conf-slider"
            type="range"
            min="0.20"
            max="0.85"
            step="0.05"
            value={confThreshold}
            onChange={function handleSliderChange(e) { onChangeConfThreshold(parseFloat(e.target.value)) }}
          />
        </div>

        <label className="toggle-row">
          <input
            type="checkbox"
            checked={enableClahe}
            onChange={function handleClaheCheck(e) { onToggleClahe(e.target.checked) }}
          />
          <span>Low-Light CLAHE Enhancement (&lt;100 lux)</span>
        </label>

        <div className="divider" />

        {inputMode === 'camera' && (
          <div className="input-action-box">
            <p className="helper-text">
              Streams webcam frames over WebSocket <code>/ws/scan</code> with real-time temporal bounding boxes.
            </p>
            <div className="action-row">
              {!cameraActive ? (
                <button
                  type="button"
                  className="btn btn-primary btn-full"
                  onClick={startLiveCamera}
                >
                  <Camera size={17} />
                  <span>Start Live Camera</span>
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={stopCameraAndSocket}
                  >
                    <CameraOff size={16} />
                    <span>Stop</span>
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={function togglePauseClick() { setCameraPaused(function flip(p) { return !p }) }}
                  >
                    {cameraPaused ? <Play size={16} /> : <Pause size={16} />}
                    <span>{cameraPaused ? 'Resume' : 'Pause'}</span>
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={isProcessing}
                    onClick={handleWebcamSnapshotDetect}
                  >
                    <Sparkles size={16} />
                    <span>Snapshot</span>
                  </button>
                </>
              )}
            </div>
          </div>
        )}

        {inputMode === 'upload' && (
          <div className="input-action-box">
            <label htmlFor="image-upload-input">Upload Pantry / Fridge Image</label>
            <input
              id="image-upload-input"
              type="file"
              accept="image/*"
              onChange={function handleImagePick(e) {
                setSelectedImageFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)
              }}
            />
            <button
              type="button"
              className="btn btn-primary btn-full"
              disabled={isProcessing}
              onClick={handleUploadImageSubmit}
            >
              <Upload size={16} />
              <span>{isProcessing ? 'Detecting...' : 'Detect Uploaded Image'}</span>
            </button>
          </div>
        )}

        {inputMode === 'sample' && (
          <div className="input-action-box">
            <label htmlFor="sample-select">Built-In Pantry Sample</label>
            <select
              id="sample-select"
              value={selectedSample}
              onChange={function handleSampleChange(e) { setSelectedSample(e.target.value) }}
            >
              {BUILTIN_SAMPLES.map(function renderSampleOpt(s) {
                return (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                )
              })}
            </select>
            <button
              type="button"
              className="btn btn-primary btn-full"
              disabled={isProcessing}
              onClick={handleSampleDetectSubmit}
            >
              <ImageIcon size={16} />
              <span>{isProcessing ? 'Detecting...' : 'Detect Sample Image'}</span>
            </button>
          </div>
        )}

        {inputMode === 'video' && (
          <div className="input-action-box">
            <label htmlFor="video-upload-input">Upload Pantry Video Clip (.mp4)</label>
            <input
              id="video-upload-input"
              type="file"
              accept="video/*"
              onChange={function handleVideoPick(e) {
                setSelectedVideoFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)
              }}
            />
            <button
              type="button"
              className="btn btn-primary btn-full"
              disabled={isProcessing}
              onClick={handleVideoUploadSubmit}
            >
              <Film size={16} />
              <span>{isProcessing ? 'Analyzing Clip...' : 'Analyze Video Clip'}</span>
            </button>
          </div>
        )}

        {errorMessage && (
          <div className="alert-error" role="alert">
            {errorMessage}
          </div>
        )}

        <div className="basket-mini-summary">
          <div className="basket-mini-header">
            <span className="mini-title">
              <ShoppingBasket size={16} />
              <span>Confirmed Basket ({basketItems.length})</span>
            </span>
            <button
              type="button"
              className="link-btn"
              onClick={onGoToBasket}
            >
              Open Basket
            </button>
          </div>
          {basketItems.length === 0 ? (
            <p className="helper-text">Scan food items to build your verified basket.</p>
          ) : (
            <div className="mini-chips-wrap">
              {basketItems.map(function renderMiniChip(item) {
                const cropSrc = formatCropDataUrl(item.crop_jpeg_b64)
                return (
                  <span key={item.class_id} className="mini-chip">
                    {cropSrc && <img src={cropSrc} alt={item.display_name} className="mini-chip-img" />}
                    <span>{item.display_name} x{item.count}</span>
                  </span>
                )
              })}
            </div>
          )}
          <button
            type="button"
            className="btn btn-accent btn-full"
            disabled={basketItems.length === 0}
            onClick={onGenerateRecipes}
          >
            <Sparkles size={16} />
            <span>Generate Recipes ({basketItems.length})</span>
          </button>
        </div>
      </aside>

      <section className="scanner-stage-col">
        <div className="hud-strip">
          <div className="hud-card">
            <span className="hud-label">Detections</span>
            <strong className="hud-value">{activeStats.count}</strong>
          </div>
          <div className="hud-card">
            <span className="hud-label">Latency</span>
            <strong className="hud-value">{Number(activeStats.inferenceMs || 0).toFixed(1)} ms</strong>
          </div>
          <div className="hud-card">
            <span className="hud-label">Throughput</span>
            <strong className="hud-value">{Number(activeStats.fps || 0).toFixed(1)} FPS</strong>
          </div>
          <div className="hud-card">
            <span className="hud-label">Edge Density</span>
            <strong className="hud-value">{Number(activeStats.diag.edge_density_pct || 0).toFixed(1)}%</strong>
          </div>
          <div className="hud-card">
            <span className="hud-label">ORB Keypoints</span>
            <strong className="hud-value">{activeStats.diag.orb_keypoints || 0}</strong>
          </div>
        </div>

        <div className="card viewport-card">
          <div className="viewport-header">
            <div className="viewport-title-row">
              <h2>
                {inputMode === 'camera' ? 'Live Webcam Overlay' : (previewData ? previewData.label : 'Detection Preview')}
              </h2>
              {inputMode === 'camera' && (
                <span className={`ws-badge ${wsConnected ? 'ws-badge-live' : 'ws-badge-off'}`}>
                  {wsConnected ? (cameraPaused ? 'WS Paused' : 'WS Live') : 'WS Disconnected'}
                </span>
              )}
            </div>
            <div className="legend-row">
              <span className="legend-item">
                <span className="legend-swatch legend-confirmed" />
                <span>Confirmed</span>
              </span>
              <span className="legend-item">
                <span className="legend-swatch legend-pending" />
                <span>Pending</span>
              </span>
            </div>
          </div>

          {inputMode === 'camera' ? (
            <div className="camera-stage-wrap">
              {!cameraActive ? (
                <div className="stage-placeholder">
                  <Camera size={42} />
                  <h3>Webcam Ready</h3>
                  <p>Click &ldquo;Start Live Camera&rdquo; to stream frames and draw live green/amber detections.</p>
                </div>
              ) : (
                <div className="camera-stage" style={{ aspectRatio: videoAspect }}>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="camera-video"
                  />
                  <canvas
                    ref={overlayCanvasRef}
                    className="camera-overlay-canvas"
                  />
                </div>
              )}
            </div>
          ) : (
            <div className="image-stage-wrap">
              {activeKeyframe && activeKeyframe.annotated_jpeg_b64 ? (
                <>
                  <img
                    src={`data:image/jpeg;base64,${activeKeyframe.annotated_jpeg_b64}`}
                    alt="Annotated food detection preview"
                    className="annotated-preview-img"
                  />
                  {previewData && previewData.keyframes && previewData.keyframes.length > 1 && (
                    <div className="keyframe-strip">
                      <Layers size={15} />
                      <span>Keyframes:</span>
                      {previewData.keyframes.map(function renderKeyframeBtn(kf, idx) {
                        return (
                          <button
                            key={kf.frame_id}
                            type="button"
                            className={`kf-btn ${selectedKeyframeIdx === idx ? 'kf-btn-active' : ''}`}
                            onClick={function selectKf() { setSelectedKeyframeIdx(idx) }}
                          >
                            Frame #{kf.frame_id} ({kf.detection_count})
                          </button>
                        )
                      })}
                    </div>
                  )}
                </>
              ) : (
                <div className="stage-placeholder">
                  <ImageIcon size={42} />
                  <h3>No Frame Analyzed Yet</h3>
                  <p>Run a built-in pantry sample, upload an image, or analyze a video clip.</p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="card detections-card">
          <div className="panel-heading">
            <h2>Frame Detections ({displayedDetections.length})</h2>
            <span className="helper-text">Unwanted items can be removed directly from frame or basket</span>
          </div>

          {displayedDetections.length === 0 ? (
            <p className="helper-text">No food detections currently visible in this frame.</p>
          ) : (
            <div className="table-scroll">
              <table className="det-table">
                <thead>
                  <tr>
                    <th>Item</th>
                    <th>Status</th>
                    <th>Confidence</th>
                    <th>Est. kcal</th>
                    <th>BBox Norm</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {displayedDetections.map(function renderDetRow(det, idx) {
                    const meta = getClassMeta(det.class_id)
                    const disp = det.display_name || meta.display_name
                    const cropSrc = formatCropDataUrl(det.crop_jpeg_b64)
                    const confPct = Math.round(Number(det.confidence || 0) * 100)
                    const kcal = det.estimated_calories != null ? det.estimated_calories : meta.calories_per_unit
                    const boxStr = (det.bbox_norm || []).map(function fmtCoord(v) { return Number(v).toFixed(2) }).join(', ')
                    return (
                      <tr key={`${det.class_id}-${idx}`}>
                        <td>
                          <div className="table-item-cell">
                            {cropSrc && <img src={cropSrc} alt={disp} className="table-crop" />}
                            <div>
                              <strong>{disp}</strong>
                              <small>{det.class_id}</small>
                            </div>
                          </div>
                        </td>
                        <td>
                          <span className={`state-pill ${det.temporally_confirmed ? 'state-pill-ok' : 'state-pill-pending'}`}>
                            {det.temporally_confirmed ? 'Confirmed' : 'Pending'}
                          </span>
                        </td>
                        <td>{confPct}%</td>
                        <td>{kcal} kcal</td>
                        <td className="mono-cell">[{boxStr}]</td>
                        <td>
                          <button
                            type="button"
                            className="btn-icon-danger"
                            onClick={function clickDeleteDet() { handleDeleteDetection(det.class_id) }}
                            title={`Remove ${disp}`}
                          >
                            <Trash2 size={15} />
                            <span>Remove</span>
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
