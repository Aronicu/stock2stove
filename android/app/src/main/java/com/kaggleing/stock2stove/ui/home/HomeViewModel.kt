package com.kaggleing.stock2stove.ui.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaggleing.stock2stove.data.api.InferenceApiService
import com.kaggleing.stock2stove.data.api.NetworkClientProvider
import com.kaggleing.stock2stove.data.model.ConnectionStatus
import com.kaggleing.stock2stove.data.model.HealthResponse
import com.kaggleing.stock2stove.data.repository.AlphaTelemetryLogger
import com.kaggleing.stock2stove.data.repository.InMemoryServerConfigStore
import com.kaggleing.stock2stove.data.repository.ServerAddressParser
import com.kaggleing.stock2stove.data.repository.ServerConfigStore
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class HomeUiState(
    val selectedPresetId: String = "custom_lan",
    val scheme: String = "http",
    val customHost: String = "192.168.1.100",
    val customPort: Int = 8000,
    val portInput: String = "8000",
    val activeBaseUrl: String = "http://192.168.1.100:8000",
    val connectionStatus: ConnectionStatus = ConnectionStatus.DISCONNECTED,
    val isCheckingHealth: Boolean = false,
    val healthResponse: HealthResponse? = null,
    val errorMessage: String? = null,
    val pingLatencyMs: Long? = null,
    val isSaved: Boolean = true
)

class HomeViewModel(
    private val apiService: InferenceApiService = NetworkClientProvider.apiService,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val configStore: ServerConfigStore = InMemoryServerConfigStore()
) : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    init {
        val savedScheme = configStore.loadScheme()
        val savedHost = configStore.loadHost()
        val savedPort = configStore.loadPort()
        val savedPresetId = configStore.loadPresetId()
        val baseUrl = ServerAddressParser.buildBaseUrl(savedScheme, savedHost, savedPort)
        NetworkClientProvider.setServerConfig(savedScheme, savedHost, savedPort)
        _uiState.value = HomeUiState(
            selectedPresetId = savedPresetId,
            scheme = savedScheme,
            customHost = savedHost,
            customPort = savedPort,
            portInput = savedPort.toString(),
            activeBaseUrl = baseUrl,
            isSaved = savedHost.isNotBlank()
        )
    }

    fun selectPreset(presetId: String) {
        val currentState = _uiState.value
        val targetScheme: String
        val targetHost: String
        val targetPort: Int
        if (presetId == "usb_adb") {
            targetScheme = "http"
            targetHost = "127.0.0.1"
            targetPort = 8000
        } else if (presetId == "emulator") {
            targetScheme = "http"
            targetHost = "10.0.2.2"
            targetPort = 8000
        } else if (presetId == "https_tunnel") {
            targetScheme = "https"
            val prevHost = currentState.customHost.trim()
            targetHost = if (prevHost == "127.0.0.1" || prevHost == "10.0.2.2" || prevHost == "192.168.1.100") {
                ""
            } else {
                prevHost
            }
            targetPort = if (currentState.scheme == "https" && currentState.customPort in 1..65535) {
                currentState.customPort
            } else {
                443
            }
        } else {
            targetScheme = "http"
            val prevHost = currentState.customHost.trim()
            targetHost = if (prevHost.isBlank() || prevHost == "127.0.0.1" || prevHost == "10.0.2.2") {
                "192.168.1.100"
            } else {
                prevHost
            }
            targetPort = if (currentState.scheme == "http" && currentState.customPort in 1..65535) {
                currentState.customPort
            } else {
                8000
            }
        }

        val newUrl = ServerAddressParser.buildBaseUrl(targetScheme, targetHost, targetPort)
        val hasHost = targetHost.isNotBlank()
        if (hasHost) {
            configStore.saveServerConfig(targetScheme, targetHost, targetPort, presetId)
            NetworkClientProvider.setServerConfig(targetScheme, targetHost, targetPort)
        } else {
            NetworkClientProvider.setServerConfig(targetScheme, targetHost, targetPort)
        }

        _uiState.update { state ->
            state.copy(
                selectedPresetId = presetId,
                scheme = targetScheme,
                customHost = targetHost,
                customPort = targetPort,
                portInput = targetPort.toString(),
                activeBaseUrl = newUrl,
                connectionStatus = ConnectionStatus.DISCONNECTED,
                healthResponse = null,
                errorMessage = null,
                isSaved = hasHost
            )
        }
    }

    fun setUseHttps(enabled: Boolean) {
        val currentState = _uiState.value
        val nextScheme = if (enabled) "https" else "http"
        if (currentState.scheme == nextScheme) {
            return
        }
        val nextPort = if (enabled) {
            if (currentState.customPort == 8000 || currentState.customPort == 80) 443 else currentState.customPort
        } else {
            if (currentState.customPort == 443) 8000 else currentState.customPort
        }
        val nextPreset = if (enabled) {
            "https_tunnel"
        } else if (currentState.selectedPresetId == "https_tunnel") {
            "custom_lan"
        } else {
            currentState.selectedPresetId
        }
        val newUrl = ServerAddressParser.buildBaseUrl(nextScheme, currentState.customHost, nextPort)
        val hasHost = currentState.customHost.isNotBlank()
        if (hasHost) {
            configStore.saveServerConfig(nextScheme, currentState.customHost, nextPort, nextPreset)
        }
        NetworkClientProvider.setServerConfig(nextScheme, currentState.customHost, nextPort)

        _uiState.update { state ->
            state.copy(
                selectedPresetId = nextPreset,
                scheme = nextScheme,
                customPort = nextPort,
                portInput = nextPort.toString(),
                activeBaseUrl = newUrl,
                connectionStatus = ConnectionStatus.DISCONNECTED,
                healthResponse = null,
                errorMessage = null,
                isSaved = hasHost
            )
        }
    }

    fun setCustomHost(host: String) {
        val currentState = _uiState.value
        val parsed = ServerAddressParser.parseAddress(host, currentState.scheme, currentState.customPort)
        val cleanHost = parsed.host
        val resolvedScheme = parsed.scheme
        val resolvedPort = parsed.port
        val inferredPreset = if (cleanHost == "127.0.0.1" || cleanHost == "localhost") {
            if (resolvedScheme == "https") "https_tunnel" else "usb_adb"
        } else if (cleanHost == "10.0.2.2") {
            if (resolvedScheme == "https") "https_tunnel" else "emulator"
        } else if (resolvedScheme == "https") {
            "https_tunnel"
        } else {
            "custom_lan"
        }

        val newUrl = ServerAddressParser.buildBaseUrl(resolvedScheme, cleanHost, resolvedPort)
        val hasHost = cleanHost.isNotBlank()
        if (hasHost) {
            configStore.saveServerConfig(resolvedScheme, cleanHost, resolvedPort, inferredPreset)
            NetworkClientProvider.setServerConfig(resolvedScheme, cleanHost, resolvedPort)
        }

        _uiState.update { state ->
            state.copy(
                selectedPresetId = inferredPreset,
                scheme = resolvedScheme,
                customHost = cleanHost,
                customPort = resolvedPort,
                portInput = resolvedPort.toString(),
                activeBaseUrl = newUrl,
                connectionStatus = ConnectionStatus.DISCONNECTED,
                errorMessage = null,
                isSaved = hasHost
            )
        }
    }

    fun setCustomPort(port: Int) {
        val currentState = _uiState.value
        val defaultPort = if (currentState.scheme == "https") 443 else 8000
        val validPort = if (port in 1..65535) port else defaultPort
        val newUrl = ServerAddressParser.buildBaseUrl(currentState.scheme, currentState.customHost, validPort)
        if (currentState.customHost.isNotBlank()) {
            configStore.saveServerConfig(currentState.scheme, currentState.customHost, validPort, currentState.selectedPresetId)
            NetworkClientProvider.setServerConfig(currentState.scheme, currentState.customHost, validPort)
        }
        _uiState.update { state ->
            state.copy(
                customPort = validPort,
                portInput = validPort.toString(),
                activeBaseUrl = newUrl,
                connectionStatus = ConnectionStatus.DISCONNECTED,
                errorMessage = null,
                isSaved = currentState.customHost.isNotBlank()
            )
        }
    }

    fun setCustomPortInput(rawPort: String) {
        val currentState = _uiState.value
        val builder = StringBuilder()
        for (i in 0 until rawPort.length) {
            val ch = rawPort[i]
            if (ch in '0'..'9') {
                builder.append(ch)
            }
        }
        val digits = builder.toString()
        val parsedPort = digits.toIntOrNull()
        val validPort = if (parsedPort != null && parsedPort in 1..65535) {
            parsedPort
        } else {
            currentState.customPort
        }

        val newUrl = ServerAddressParser.buildBaseUrl(currentState.scheme, currentState.customHost, validPort)
        val canSave = parsedPort != null && parsedPort in 1..65535 && currentState.customHost.isNotBlank()
        if (canSave) {
            configStore.saveServerConfig(currentState.scheme, currentState.customHost, validPort, currentState.selectedPresetId)
            NetworkClientProvider.setServerConfig(currentState.scheme, currentState.customHost, validPort)
        }

        _uiState.update { state ->
            state.copy(
                portInput = digits,
                customPort = validPort,
                activeBaseUrl = newUrl,
                connectionStatus = ConnectionStatus.DISCONNECTED,
                errorMessage = null,
                isSaved = currentState.customHost.isNotBlank()
            )
        }
    }

    fun checkHealth() {
        val currentState = _uiState.value
        val cleanHost = currentState.customHost.trim()
        if (cleanHost.isBlank()) {
            _uiState.update { state ->
                state.copy(
                    isCheckingHealth = false,
                    connectionStatus = ConnectionStatus.ERROR,
                    errorMessage = "Please enter a valid server host or tunnel URL"
                )
            }
            return
        }

        val defaultPort = if (currentState.scheme == "https") 443 else 8000
        val targetPort = if (currentState.customPort in 1..65535) currentState.customPort else defaultPort
        val targetUrl = ServerAddressParser.buildBaseUrl(currentState.scheme, cleanHost, targetPort)
        configStore.saveServerConfig(currentState.scheme, cleanHost, targetPort, currentState.selectedPresetId)
        NetworkClientProvider.setServerConfig(currentState.scheme, cleanHost, targetPort)

        _uiState.update { state ->
            state.copy(
                activeBaseUrl = targetUrl,
                isCheckingHealth = true,
                connectionStatus = ConnectionStatus.CONNECTING,
                errorMessage = null,
                isSaved = true
            )
        }

        viewModelScope.launch(ioDispatcher) {
            val startTime = System.currentTimeMillis()
            try {
                NetworkClientProvider.setServerConfig(currentState.scheme, cleanHost, targetPort)
                val response = apiService.getHealth()
                val latency = System.currentTimeMillis() - startTime
                AlphaTelemetryLogger.getDefault().logEvent(
                    "NETWORK",
                    "INFO",
                    "Health ping succeeded: ${response.service} (${response.status}) in ${latency}ms"
                )
                _uiState.update { state ->
                    state.copy(
                        isCheckingHealth = false,
                        connectionStatus = ConnectionStatus.CONNECTED,
                        healthResponse = response,
                        pingLatencyMs = latency,
                        errorMessage = null
                    )
                }
            } catch (e: Exception) {
                val latency = System.currentTimeMillis() - startTime
                _uiState.update { state ->
                    state.copy(
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
