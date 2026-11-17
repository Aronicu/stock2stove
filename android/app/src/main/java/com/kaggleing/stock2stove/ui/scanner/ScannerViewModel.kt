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
    val selectedMode: DetectorDefenseMode = DetectorDefenseMode.YOLO_V8N,
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

    init {
        startCountdownTicker()
    }

    fun initScan(wsUrl: String, mode: ScanMode, sessionId: String? = null) {
        val sId = sessionId ?: UUID.randomUUID().toString().take(8)
        _uiState.update {
            it.copy(
                activeWsUrl = wsUrl,
                scanMode = mode,
                sessionId = sId,
                progressSec = 0.0f,
                isScanning = true,
                isPaused = false,
                isComplete = false,
                frameCount = 0L,
                detections = emptyList(),
                confirmedInventory = emptyList(),
                errorMessage = null
            )
        }
        tracker.reset()
        lastGeminiFrameTimeMs = 0L
        AlphaTelemetryLogger.getDefault().logEvent("SCANNER", "INFO", "Scan session started: $sId")
        connectWebSocket(wsUrl)
    }

    fun connectWebSocket(wsUrl: String) {
        webSocketClient.connect(
            wsUrl = wsUrl,
            onConnected = {
                _uiState.update { it.copy(isConnected = true, errorMessage = null) }
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
                _uiState.update {
                    it.copy(
                        isConnected = false,
                        errorMessage = error.message ?: "Connection failure"
                    )
                }
            },
            onClosed = {
                _uiState.update { it.copy(isConnected = false) }
            }
        )
    }

    fun handleScanResponse(response: ScanResponse) {
        tracker.recordFrame(response.detections)
        val trackerTally = tracker.calculateConfirmedTally()
        AlphaTelemetryLogger.getDefault().logTelemetry(
            frameId = response.frame_id,
            detectorMode = response.detector_mode,
            inferenceMs = response.inference_ms,
            fps = response.fps,
            detectionsCount = response.detections.size
        )

        _uiState.update { state ->
            val confirmedServerItems = mutableListOf<ConfirmedItem>()
            for (item in response.confirmed_inventory) {
                var detConfirmed = false
                for (det in response.detections) {
                    if (det.class_id == item.class_id && det.temporally_confirmed) {
                        detConfirmed = true
                        break
                    }
                }
                if (detConfirmed || tracker.isConfirmed(item.class_id)) {
                    confirmedServerItems.add(item)
                }
            }
            val mergedInventory = if (confirmedServerItems.isNotEmpty()) {
                confirmedServerItems
            } else if (trackerTally.isNotEmpty()) {
                val tallyItems = mutableListOf<ConfirmedItem>()
                for ((cls, count) in trackerTally) {
                    if (!tracker.isConfirmed(cls)) {
                        continue
                    }
                    var matchingDetection: DetectionItem? = null
                    for (det in response.detections) {
                        if (det.class_id == cls && (det.temporally_confirmed || tracker.isConfirmed(cls))) {
                            matchingDetection = det
                            break
                        }
                    }
                    tallyItems.add(
                        ConfirmedItem(
                            class_id = cls,
                            display_name = matchingDetection?.display_name ?: cls,
                            count = count,
                            max_conf = matchingDetection?.confidence ?: 0.5f
                        )
                    )
                }
                tallyItems
            } else {
                state.confirmedInventory
            }

            state.copy(
                fps = response.fps,
                inferenceMs = response.inference_ms,
                diagnostics = response.diagnostics,
                detections = response.detections,
                confirmedInventory = mergedInventory
            )
        }
    }

    fun onFrameCaptured(jpegBytes: ByteArray, b64: String) {
        val state = _uiState.value
        if (!state.isScanning || state.isPaused || state.isComplete) {
            return
        }

        if (state.selectedMode == DetectorDefenseMode.GEMINI_VISION) {
            val nowMs = System.currentTimeMillis()
            if (lastGeminiFrameTimeMs != 0L && (nowMs - lastGeminiFrameTimeMs) < 600L) {
                return
            }
            lastGeminiFrameTimeMs = nowMs
        }

        val nextFrameId = state.frameCount + 1L
        _uiState.update { it.copy(frameCount = nextFrameId) }

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
        _uiState.update { it.copy(selectedMode = mode) }
    }

    fun setClaheEnabled(enabled: Boolean) {
        _uiState.update { it.copy(claheEnabled = enabled) }
    }

    fun setConfThreshold(threshold: Float) {
        _uiState.update { it.copy(confThreshold = threshold) }
    }

    fun togglePause() {
        _uiState.update { it.copy(isPaused = !it.isPaused) }
    }

    fun resetScan() {
        tracker.reset()
        lastGeminiFrameTimeMs = 0L
        _uiState.update {
            it.copy(
                progressSec = 0.0f,
                isScanning = true,
                isPaused = false,
                isComplete = false,
                frameCount = 0L,
                detections = emptyList(),
                confirmedInventory = emptyList(),
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

    fun triggerComplete() {
        _uiState.update {
            it.copy(
                progressSec = it.totalSec,
                isComplete = true,
                isScanning = false
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
                isScanning = if (complete) false else state.isScanning
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
                    val complete = nextSec >= state.totalSec
                    _uiState.update {
                        it.copy(
                            progressSec = nextSec,
                            isComplete = complete,
                            isScanning = if (complete) false else it.isScanning
                        )
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
