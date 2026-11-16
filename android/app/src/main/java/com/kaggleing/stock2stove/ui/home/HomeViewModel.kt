package com.kaggleing.stock2stove.ui.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaggleing.stock2stove.data.api.InferenceApiService
import com.kaggleing.stock2stove.data.api.NetworkClientProvider
import com.kaggleing.stock2stove.data.model.ConnectionStatus
import com.kaggleing.stock2stove.data.model.HealthResponse
import com.kaggleing.stock2stove.data.model.ScanMode
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class HomeUiState(
    val selectedPresetId: String = "usb_adb",
    val customHost: String = "192.168.1.100",
    val customPort: Int = 8000,
    val activeBaseUrl: String = "http://127.0.0.1:8000",
    val scanMode: ScanMode = ScanMode.PANTRY_SHELF,
    val connectionStatus: ConnectionStatus = ConnectionStatus.DISCONNECTED,
    val isCheckingHealth: Boolean = false,
    val healthResponse: HealthResponse? = null,
    val errorMessage: String? = null,
    val pingLatencyMs: Long? = null
)

class HomeViewModel(
    private val apiService: InferenceApiService = NetworkClientProvider.apiService,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO
) : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    fun selectPreset(presetId: String) {
        val newUrl = when (presetId) {
            "emulator" -> "http://10.0.2.2:8000"
            "custom_lan" -> "http://${_uiState.value.customHost}:${_uiState.value.customPort}"
            else -> "http://127.0.0.1:8000"
        }
        NetworkClientProvider.setBaseUrl(newUrl)
        _uiState.update {
            it.copy(
                selectedPresetId = presetId,
                activeBaseUrl = newUrl,
                connectionStatus = ConnectionStatus.DISCONNECTED,
                errorMessage = null
            )
        }
    }

    fun setCustomHost(host: String) {
        val newUrl = "http://$host:${_uiState.value.customPort}"
        NetworkClientProvider.setBaseUrl(newUrl)
        _uiState.update {
            it.copy(
                customHost = host,
                activeBaseUrl = newUrl,
                connectionStatus = ConnectionStatus.DISCONNECTED,
                errorMessage = null
            )
        }
    }

    fun setCustomPort(port: Int) {
        val newUrl = "http://${_uiState.value.customHost}:$port"
        NetworkClientProvider.setBaseUrl(newUrl)
        _uiState.update {
            it.copy(
                customPort = port,
                activeBaseUrl = newUrl,
                connectionStatus = ConnectionStatus.DISCONNECTED,
                errorMessage = null
            )
        }
    }

    fun setScanMode(mode: ScanMode) {
        _uiState.update {
            it.copy(scanMode = mode)
        }
    }

    fun checkHealth() {
        val targetUrl = _uiState.value.activeBaseUrl
        _uiState.update {
            it.copy(
                isCheckingHealth = true,
                connectionStatus = ConnectionStatus.CONNECTING,
                errorMessage = null
            )
        }
        viewModelScope.launch(ioDispatcher) {
            val startTime = System.currentTimeMillis()
            try {
                NetworkClientProvider.setBaseUrl(targetUrl)
                val response = apiService.getHealth()
                val latency = System.currentTimeMillis() - startTime
                com.kaggleing.stock2stove.data.repository.AlphaTelemetryLogger.getDefault().logEvent(
                    "NETWORK",
                    "INFO",
                    "Health ping succeeded: ${response.service} (${response.status}) in ${latency}ms"
                )
                _uiState.update {
                    it.copy(
                        isCheckingHealth = false,
                        connectionStatus = ConnectionStatus.CONNECTED,
                        healthResponse = response,
                        pingLatencyMs = latency,
                        errorMessage = null
                    )
                }
            } catch (e: Exception) {
                val latency = System.currentTimeMillis() - startTime
                _uiState.update {
                    it.copy(
                        isCheckingHealth = false,
                        connectionStatus = ConnectionStatus.ERROR,
                        healthResponse = null,
                        pingLatencyMs = latency,
                        errorMessage = e.message ?: "Failed to connect to server"
                    )
                }
            }
        }
    }
}
