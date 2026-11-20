package com.kaggleing.stock2stove.ui.scanner

import androidx.lifecycle.ViewModel
import com.kaggleing.stock2stove.data.api.ScanWebSocketClient
import com.kaggleing.stock2stove.data.model.ConfirmedItem
import com.kaggleing.stock2stove.data.model.DetectionItem
import com.kaggleing.stock2stove.data.model.DetectorDefenseMode
import com.kaggleing.stock2stove.data.model.DiagnosticsData
import com.kaggleing.stock2stove.data.model.ScanFrameRequest
import com.kaggleing.stock2stove.data.model.ScanMode
import com.kaggleing.stock2stove.data.model.ScanResponse
import com.kaggleing.stock2stove.data.model.UnifiedItemClass
import com.kaggleing.stock2stove.data.repository.AlphaTelemetryLogger
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.UUID

data class ScannerUiState(
    val selectedMode: DetectorDefenseMode = DetectorDefenseMode.GEMINI_VISION,
    val claheEnabled: Boolean = true,
    val confThreshold: Float = 0.5f,
    val fps: Float = 0.0f,
    val inferenceMs: Float = 0.0f,
    val diagnostics: DiagnosticsData = DiagnosticsData(),
    val detections: List<DetectionItem> = emptyList(),
    val confirmedInventory: List<ConfirmedItem> = emptyList(),
    val progressSec: Float = 0.0f,
    val totalSec: Float = 5.0f,
    val isScanning: Boolean = true,
    val isPaused: Boolean = false,
    val isComplete: Boolean = false,
    val isConnected: Boolean = false,
    val isAnalyzingFrame: Boolean = false,
    val showScanMorePrompt: Boolean = false,
    val scanMode: ScanMode = ScanMode.PANTRY_SHELF,
    val activeWsUrl: String = "ws://127.0.0.1:8000/ws/scan",
    val sessionId: String = UUID.randomUUID().toString().take(8),
    val frameCount: Long = 0L,
    val errorMessage: String? = null
)

class ScannerViewModel(
    val webSocketClient: ScanWebSocketClient = ScanWebSocketClient(),
    private val coroutineDispatcher: CoroutineDispatcher = Dispatchers.Default
) : ViewModel() {

    private val scope = CoroutineScope(SupervisorJob() + coroutineDispatcher)
    private val _uiState = MutableStateFlow(ScannerUiState())
    val uiState: StateFlow<ScannerUiState> = _uiState.asStateFlow()

    val tracker = TemporalConfirmationTracker()
    private var tickerJob: Job? = null
    private var lastGeminiFrameTimeMs: Long = 0L
    private var awaitingGeminiResponse: Boolean = false
    private var lastGeminiSentMs: Long = 0L
    private var hasInitializedUrl: String? = null
    private val frameCacheLock = Any()
    private val recentCapturedFrames = LinkedHashMap<Long, ByteArray>()
    private val bestCropByClass = mutableMapOf<String, Pair<Float, String>>()

    init {
        startCountdownTicker()
    }

    fun initScan(
        wsUrl: String,
        mode: ScanMode = ScanMode.PANTRY_SHELF,
        sessionId: String? = null,
        clearItems: Boolean = false
    ) {
        if (!clearItems && hasInitializedUrl == wsUrl && webSocketClient.isConnected()) {
            return
        }
        hasInitializedUrl = wsUrl
        val sId = sessionId ?: _uiState.value.sessionId.ifBlank { UUID.randomUUID().toString().take(8) }
        _uiState.update { state ->
            state.copy(
                activeWsUrl = wsUrl,
                scanMode = mode,
                sessionId = sId,
                progressSec = 0.0f,
                isScanning = true,
                isPaused = false,
                isComplete = false,
                isAnalyzingFrame = false,
                showScanMorePrompt = false,
                frameCount = 0L,
                detections = emptyList(),
                confirmedInventory = if (clearItems) emptyList() else state.confirmedInventory,
                errorMessage = null
            )
        }
        tracker.reset()
        lastGeminiFrameTimeMs = 0L
        lastGeminiSentMs = 0L
        awaitingGeminiResponse = false
        synchronized(frameCacheLock) {
            recentCapturedFrames.clear()
            if (clearItems) {
                bestCropByClass.clear()
            }
        }
        AlphaTelemetryLogger.getDefault().logEvent("SCANNER", "INFO", "Scan session started: $sId")
        connectWebSocket(wsUrl)
    }

    fun seedInventoryIfEmpty(items: List<ConfirmedItem>) {
        if (items.isEmpty()) {
            return
        }
        if (_uiState.value.confirmedInventory.isNotEmpty()) {
            return
        }
        for (item in items) {
            recordCropForClass(item.class_id, item.max_conf, item.crop_jpeg_b64)
        }
        _uiState.update { state ->
            state.copy(confirmedInventory = items)
        }
    }

    fun syncConfirmedInventory(items: List<ConfirmedItem>) {
        for (item in items) {
            recordCropForClass(item.class_id, item.max_conf, item.crop_jpeg_b64)
        }
        _uiState.update { state ->
            state.copy(confirmedInventory = items)
        }
    }

    fun connectWebSocket(wsUrl: String) {
        webSocketClient.connect(
            wsUrl = wsUrl,
            onConnected = {
                _uiState.update { state -> state.copy(isConnected = true, errorMessage = null) }
                val currentState = _uiState.value
                webSocketClient.resetSession(
                    sessionId = currentState.sessionId,
                    detectorMode = currentState.selectedMode.wireMode,
                    enableClahe = currentState.claheEnabled,
                    confThreshold = currentState.confThreshold
                )
            },
            onMessage = { response ->
                handleScanResponse(response)
            },
            onError = { error ->
                awaitingGeminiResponse = false
                _uiState.update { state ->
                    state.copy(
                        isConnected = false,
                        isAnalyzingFrame = false,
                        errorMessage = error.message ?: "Connection failure"
                    )
                }
            },
            onClosed = {
                awaitingGeminiResponse = false
                _uiState.update { state ->
                    state.copy(
                        isConnected = false,
                        isAnalyzingFrame = false
                    )
                }
            }
        )
    }

    private fun recordCropForClass(classId: String, confidence: Float, cropB64: String?) {
        if (classId.isBlank() || cropB64.isNullOrBlank()) {
            return
        }
        synchronized(frameCacheLock) {
            val existing = bestCropByClass[classId]
            if (existing == null || confidence >= existing.first) {
                bestCropByClass[classId] = Pair(confidence, cropB64)
            }
        }
    }

    private fun resolveBestCrop(classId: String, directCrop: String?): String? {
        synchronized(frameCacheLock) {
            val exact = bestCropByClass[classId]
            if (exact != null) {
                return exact.second
            }
            val targetCanonical = UnifiedItemClass.normalizeClassId(classId)
            for ((key, entry) in bestCropByClass) {
                if (UnifiedItemClass.normalizeClassId(key) == targetCanonical) {
                    return entry.second
                }
            }
        }
        if (!directCrop.isNullOrBlank()) {
            return directCrop
        }
        return null
    }

    fun handleScanResponse(response: ScanResponse) {
        awaitingGeminiResponse = false
        var frameBytes: ByteArray? = null
        synchronized(frameCacheLock) {
            frameBytes = recentCapturedFrames[response.frame_id]
        }

        val enrichedDetections = mutableListOf<DetectionItem>()
        for (det in response.detections) {
            var cropB64 = det.crop_jpeg_b64
            if (cropB64.isNullOrBlank() && frameBytes != null && det.bbox_norm.size >= 4) {
                cropB64 = CameraFrameProcessor.cropJpegBase64(frameBytes!!, det.bbox_norm)
            }
            if (!cropB64.isNullOrBlank()) {
                recordCropForClass(det.class_id, det.confidence, cropB64)
            }
            enrichedDetections.add(det.copy(crop_jpeg_b64 = cropB64))
        }

        tracker.recordFrame(enrichedDetections)
        val trackerTally = tracker.calculateConfirmedTally()
        AlphaTelemetryLogger.getDefault().logTelemetry(
            frameId = response.frame_id,
            detectorMode = response.detector_mode,
            inferenceMs = response.inference_ms,
            fps = response.fps,
            detectionsCount = enrichedDetections.size
        )

        _uiState.update { state ->
            val confirmedServerItems = mutableListOf<ConfirmedItem>()
            val isGeminiMode = response.detector_mode == "gemini_vision" || response.detector_mode == "hybrid_gemini"
            for (item in response.confirmed_inventory) {
                var detConfirmed = false
                for (det in enrichedDetections) {
                    if (det.class_id == item.class_id && det.temporally_confirmed) {
                        detConfirmed = true
                        break
                    }
                }
                var alreadyInInventory = false
                for (prev in state.confirmedInventory) {
                    if (prev.class_id == item.class_id) {
                        alreadyInInventory = true
                        break
                    }
                }
                if (detConfirmed || tracker.isConfirmed(item.class_id) || isGeminiMode || alreadyInInventory) {
                    recordCropForClass(item.class_id, item.max_conf, item.crop_jpeg_b64)
                    val bestCrop = resolveBestCrop(item.class_id, item.crop_jpeg_b64)
                    confirmedServerItems.add(item.copy(crop_jpeg_b64 = bestCrop))
                }
            }

            val incomingItems = if (confirmedServerItems.isNotEmpty()) {
                confirmedServerItems
            } else if (trackerTally.isNotEmpty()) {
                val tallyItems = mutableListOf<ConfirmedItem>()
                for ((cls, count) in trackerTally) {
                    if (!tracker.isConfirmed(cls)) {
                        continue
                    }
                    var matchingDetection: DetectionItem? = null
                    for (det in enrichedDetections) {
                        if (det.class_id == cls && (det.temporally_confirmed || tracker.isConfirmed(cls))) {
                            matchingDetection = det
                            break
                        }
                    }
                    val cropB64 = resolveBestCrop(cls, matchingDetection?.crop_jpeg_b64)
                    tallyItems.add(
                        ConfirmedItem(
                            class_id = cls,
                            display_name = matchingDetection?.display_name ?: cls,
                            count = count,
                            max_conf = matchingDetection?.confidence ?: 0.5f,
                            crop_jpeg_b64 = cropB64
                        )
                    )
                }
                tallyItems
            } else {
                emptyList()
            }

            val mergedMap = LinkedHashMap<String, ConfirmedItem>()
            for (prev in state.confirmedInventory) {
                val refreshedCrop = resolveBestCrop(prev.class_id, prev.crop_jpeg_b64)
                mergedMap[prev.class_id] = prev.copy(crop_jpeg_b64 = refreshedCrop)
            }
            for (inc in incomingItems) {
                val existing = mergedMap[inc.class_id]
                if (existing != null) {
                    val bestCount = maxOf(existing.count, inc.count)
                    val bestConf = maxOf(existing.max_conf, inc.max_conf)
                    val bestName = if (inc.display_name.isNotBlank()) inc.display_name else existing.display_name
                    val resolvedIncCrop = resolveBestCrop(inc.class_id, inc.crop_jpeg_b64)
                    val bestCrop = if (!resolvedIncCrop.isNullOrBlank() && (inc.max_conf >= existing.max_conf || existing.crop_jpeg_b64.isNullOrBlank())) {
                        resolvedIncCrop
                    } else {
                        existing.crop_jpeg_b64
                    }
                    mergedMap[inc.class_id] = ConfirmedItem(
                        class_id = inc.class_id,
                        display_name = bestName,
                        count = bestCount,
                        max_conf = bestConf,
                        crop_jpeg_b64 = bestCrop
                    )
                } else {
                    mergedMap[inc.class_id] = inc.copy(
                        crop_jpeg_b64 = resolveBestCrop(inc.class_id, inc.crop_jpeg_b64)
                    )
                }
            }

            val mergedInventory = mergedMap.values.toList()
            val reachedEnd = state.progressSec >= state.totalSec
            val completeNow = state.isComplete || reachedEnd

            state.copy(
                fps = response.fps,
                inferenceMs = response.inference_ms,
                diagnostics = response.diagnostics,
                detections = enrichedDetections,
                confirmedInventory = mergedInventory,
                isAnalyzingFrame = false,
                isComplete = completeNow,
                isScanning = if (completeNow) false else state.isScanning,
                showScanMorePrompt = state.showScanMorePrompt || completeNow
            )
        }
    }

    fun onFrameCaptured(jpegBytes: ByteArray, b64: String) {
        val state = _uiState.value
        if (!state.isScanning || state.isPaused || state.isComplete) {
            return
        }

        if (state.selectedMode == DetectorDefenseMode.GEMINI_VISION || state.selectedMode == DetectorDefenseMode.HYBRID_GEMINI) {
            val nowMs = System.currentTimeMillis()
            if (awaitingGeminiResponse && (nowMs - lastGeminiSentMs) < 4500L) {
                return
            }
            if (lastGeminiFrameTimeMs != 0L && (nowMs - lastGeminiFrameTimeMs) < 800L) {
                return
            }
            lastGeminiFrameTimeMs = nowMs
            lastGeminiSentMs = nowMs
            awaitingGeminiResponse = true
            _uiState.update { current -> current.copy(isAnalyzingFrame = true) }
        }

        val nextFrameId = state.frameCount + 1L
        synchronized(frameCacheLock) {
            recentCapturedFrames[nextFrameId] = jpegBytes
            while (recentCapturedFrames.size > 24) {
                val oldestKey = recentCapturedFrames.keys.iterator().next()
                recentCapturedFrames.remove(oldestKey)
            }
        }
        _uiState.update { current -> current.copy(frameCount = nextFrameId) }

        val request = ScanFrameRequest(
            frame_id = nextFrameId,
            detector_mode = state.selectedMode.wireMode,
            enable_clahe = state.claheEnabled,
            conf_threshold = state.confThreshold,
            image_b64 = b64,
            session_id = state.sessionId,
            reset_session = false
        )
        webSocketClient.sendFrame(request)
    }

    fun setDetectorMode(mode: DetectorDefenseMode) {
        lastGeminiFrameTimeMs = 0L
        lastGeminiSentMs = 0L
        awaitingGeminiResponse = false
        _uiState.update { state -> state.copy(selectedMode = mode, isAnalyzingFrame = false) }
    }

    fun setClaheEnabled(enabled: Boolean) {
        _uiState.update { state -> state.copy(claheEnabled = enabled) }
    }

    fun setConfThreshold(threshold: Float) {
        _uiState.update { state -> state.copy(confThreshold = threshold) }
    }

    fun togglePause() {
        _uiState.update { state -> state.copy(isPaused = !state.isPaused) }
    }

    fun resetScan(clearItems: Boolean = false) {
        tracker.reset()
        lastGeminiFrameTimeMs = 0L
        lastGeminiSentMs = 0L
        awaitingGeminiResponse = false
        synchronized(frameCacheLock) {
            recentCapturedFrames.clear()
            if (clearItems) {
                bestCropByClass.clear()
            }
        }
        _uiState.update { state ->
            state.copy(
                progressSec = 0.0f,
                isScanning = true,
                isPaused = false,
                isComplete = false,
                isAnalyzingFrame = false,
                showScanMorePrompt = false,
                frameCount = 0L,
                detections = emptyList(),
                confirmedInventory = if (clearItems) emptyList() else state.confirmedInventory,
                errorMessage = null
            )
        }
        val state = _uiState.value
        webSocketClient.resetSession(
            sessionId = state.sessionId,
            detectorMode = state.selectedMode.wireMode,
            enableClahe = state.claheEnabled,
            confThreshold = state.confThreshold
        )
    }

    fun retryScan() {
        resetScan(clearItems = false)
    }

    fun scanMore() {
        resetScan(clearItems = false)
    }

    fun clearInventory() {
        resetScan(clearItems = true)
    }

    fun removeConfirmedItem(classId: String) {
        val targetClass = UnifiedItemClass.fromClassId(classId)
        _uiState.update { state ->
            val filtered = mutableListOf<ConfirmedItem>()
            for (item in state.confirmedInventory) {
                val exactMatch = item.class_id == classId
                val classMatch = targetClass != null && UnifiedItemClass.fromClassId(item.class_id) == targetClass
                if (!exactMatch && !classMatch) {
                    filtered.add(item)
                }
            }
            state.copy(confirmedInventory = filtered)
        }
    }

    fun triggerComplete() {
        _uiState.update { state ->
            state.copy(
                progressSec = state.totalSec,
                isComplete = true,
                isScanning = false,
                isAnalyzingFrame = false,
                showScanMorePrompt = true
            )
        }
    }

    fun advanceProgress(deltaSec: Float) {
        _uiState.update { state ->
            val nextSec = (state.progressSec + deltaSec).coerceAtMost(state.totalSec)
            val complete = nextSec >= state.totalSec
            state.copy(
                progressSec = nextSec,
                isComplete = complete,
                isScanning = if (complete) false else state.isScanning,
                showScanMorePrompt = if (complete) true else state.showScanMorePrompt
            )
        }
    }

    fun startCountdownTicker() {
        tickerJob?.cancel()
        tickerJob = scope.launch {
            while (isActive) {
                delay(100L)
                val state = _uiState.value
                if (state.isScanning && !state.isPaused && !state.isComplete) {
                    val nextSec = (state.progressSec + 0.1f).coerceAtMost(state.totalSec)
                    if (nextSec >= state.totalSec) {
                        val waitingForFirstGemini = state.selectedMode == DetectorDefenseMode.GEMINI_VISION &&
                            awaitingGeminiResponse &&
                            state.confirmedInventory.isEmpty() &&
                            (System.currentTimeMillis() - lastGeminiSentMs) < 4500L
                        if (waitingForFirstGemini) {
                            _uiState.update { current ->
                                current.copy(
                                    progressSec = current.totalSec,
                                    isAnalyzingFrame = true
                                )
                            }
                        } else {
                            _uiState.update { current ->
                                current.copy(
                                    progressSec = current.totalSec,
                                    isComplete = true,
                                    isScanning = false,
                                    isAnalyzingFrame = false,
                                    showScanMorePrompt = true
                                )
                            }
                        }
                    } else {
                        _uiState.update { current ->
                            current.copy(progressSec = nextSec)
                        }
                    }
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        tickerJob?.cancel()
        scope.cancel()
        webSocketClient.disconnect()
    }
}
