package com.kaggleing.stock2stove.data.api

import com.kaggleing.stock2stove.data.model.ScanFrameRequest
import com.kaggleing.stock2stove.data.model.ScanResponse
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import java.util.Base64
import java.util.concurrent.CopyOnWriteArrayList

typealias FrameMessage = ScanFrameRequest

interface ScanWebSocketListener {
    fun onConnected()
    fun onMessage(response: ScanResponse)
    fun onError(error: Throwable)
    fun onClosed()
}

class ScanWebSocketClient(
    private val okHttpClient: OkHttpClient = NetworkClientProvider.okHttpClient
) {
    private var webSocket: WebSocket? = null
    private var isConnectedState: Boolean = false
    private val listeners = CopyOnWriteArrayList<ScanWebSocketListener>()

    private val _lastResponse = MutableStateFlow<ScanResponse?>(null)
    val lastResponse: StateFlow<ScanResponse?> = _lastResponse.asStateFlow()

    private val _connectionState = MutableStateFlow(false)
    val connectionState: StateFlow<Boolean> = _connectionState.asStateFlow()

    private val _responseFlow = MutableSharedFlow<ScanResponse>(extraBufferCapacity = 64)
    val responseFlow: SharedFlow<ScanResponse> = _responseFlow.asSharedFlow()

    fun addListener(listener: ScanWebSocketListener) {
        listeners.add(listener)
    }

    fun removeListener(listener: ScanWebSocketListener) {
        listeners.remove(listener)
    }

    fun connect(
        wsUrl: String,
        onConnected: () -> Unit = {},
        onMessage: (ScanResponse) -> Unit = {},
        onError: (Throwable) -> Unit = {},
        onClosed: () -> Unit = {}
    ) {
        disconnect()

        val request = Request.Builder()
            .url(wsUrl)
            .build()

        val listener = object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                this@ScanWebSocketClient.webSocket = webSocket
                this@ScanWebSocketClient.isConnectedState = true
                _connectionState.value = true
                onConnected()
                for (l in listeners) {
                    l.onConnected()
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val parsed = NetworkClientProvider.json.decodeFromString<ScanResponse>(text)
                    _lastResponse.value = parsed
                    _responseFlow.tryEmit(parsed)
                    onMessage(parsed)
                    for (l in listeners) {
                        l.onMessage(parsed)
                    }
                } catch (t: Throwable) {
                    onError(t)
                    for (l in listeners) {
                        l.onError(t)
                    }
                }
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                try {
                    val text = bytes.utf8()
                    val parsed = NetworkClientProvider.json.decodeFromString<ScanResponse>(text)
                    _lastResponse.value = parsed
                    _responseFlow.tryEmit(parsed)
                    onMessage(parsed)
                    for (l in listeners) {
                        l.onMessage(parsed)
                    }
                } catch (t: Throwable) {
                    onError(t)
                    for (l in listeners) {
                        l.onError(t)
                    }
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                this@ScanWebSocketClient.isConnectedState = false
                _connectionState.value = false
                this@ScanWebSocketClient.webSocket = null
                onError(t)
                for (l in listeners) {
                    l.onError(t)
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                this@ScanWebSocketClient.isConnectedState = false
                _connectionState.value = false
                this@ScanWebSocketClient.webSocket = null
                onClosed()
                for (l in listeners) {
                    l.onClosed()
                }
            }
        }

        webSocket = okHttpClient.newWebSocket(request, listener)
    }

    fun sendFrame(request: ScanFrameRequest): Boolean {
        val currentSocket = webSocket ?: return false
        val jsonPayload = NetworkClientProvider.json.encodeToString(ScanFrameRequest.serializer(), request)
        return currentSocket.send(jsonPayload)
    }

    fun sendFrameBytes(
        frameId: Long,
        detectorMode: String,
        enableClahe: Boolean,
        confThreshold: Float,
        jpegBytes: ByteArray,
        sessionId: String = "default",
        resetSession: Boolean = false
    ): Boolean {
        val b64 = Base64.getEncoder().encodeToString(jpegBytes)
        val req = ScanFrameRequest(
            frame_id = frameId,
            detector_mode = detectorMode,
            enable_clahe = enableClahe,
            conf_threshold = confThreshold,
            image_b64 = b64,
            session_id = sessionId,
            reset_session = resetSession
        )
        return sendFrame(req)
    }

    fun resetSession(
        sessionId: String = "default",
        detectorMode: String = "yolov8",
        enableClahe: Boolean = true,
        confThreshold: Float = 0.5f
    ): Boolean {
        val req = ScanFrameRequest(
            frame_id = 0L,
            detector_mode = detectorMode,
            enable_clahe = enableClahe,
            conf_threshold = confThreshold,
            image_b64 = null,
            session_id = sessionId,
            reset_session = true
        )
        return sendFrame(req)
    }

    fun disconnect() {
        try {
            webSocket?.close(1000, "Normal closure")
        } catch (_: Throwable) {
        } finally {
            webSocket = null
            isConnectedState = false
            _connectionState.value = false
        }
    }

    fun isConnected(): Boolean = isConnectedState
}
