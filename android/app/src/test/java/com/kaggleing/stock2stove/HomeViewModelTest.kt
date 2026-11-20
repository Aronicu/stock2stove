package com.kaggleing.stock2stove

import com.kaggleing.stock2stove.data.api.InferenceApiService
import com.kaggleing.stock2stove.data.model.ConnectionStatus
import com.kaggleing.stock2stove.data.model.HardwareInfo
import com.kaggleing.stock2stove.data.model.HealthResponse
import com.kaggleing.stock2stove.data.model.ModelStatus
import com.kaggleing.stock2stove.data.model.RecipeGenerateRequest
import com.kaggleing.stock2stove.data.model.RecipeGenerateResponse
import com.kaggleing.stock2stove.data.repository.InMemoryServerConfigStore
import com.kaggleing.stock2stove.ui.home.HomeViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException

@OptIn(ExperimentalCoroutinesApi::class)
class HomeViewModelTest {

    private class FakeInferenceApiService(
        var shouldFail: Boolean = false,
        var failureMessage: String = "Connection refused"
    ) : InferenceApiService {
        override suspend fun getHealth(): HealthResponse {
            if (shouldFail) {
                throw IOException(failureMessage)
            }
            return HealthResponse(
                status = "ok",
                service = "stock2stove-cv-inference",
                model_status = ModelStatus(ultralytics_installed = true, yolov8n_loaded = true),
                hardware = HardwareInfo(device = "cpu", processor = "Test CPU")
            )
        }

        override suspend fun generateRecipes(request: RecipeGenerateRequest): RecipeGenerateResponse {
            return RecipeGenerateResponse()
        }
    }

    @Test
    fun testInitialUiState() {
        val testDispatcher = StandardTestDispatcher()
        val store = InMemoryServerConfigStore()
        val viewModel = HomeViewModel(
            apiService = FakeInferenceApiService(),
            ioDispatcher = testDispatcher,
            configStore = store
        )
        val state = viewModel.uiState.value

        assertEquals("custom_lan", state.selectedPresetId)
        assertEquals("192.168.1.100", state.customHost)
        assertEquals(8000, state.customPort)
        assertEquals("http://192.168.1.100:8000", state.activeBaseUrl)
        assertEquals(ConnectionStatus.DISCONNECTED, state.connectionStatus)
        assertFalse(state.isCheckingHealth)
        assertNull(state.healthResponse)
        assertNull(state.errorMessage)
        assertTrue(state.isSaved)
    }

    @Test
    fun testPresetSwitchingUpdatesActiveBaseUrl() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = HomeViewModel(
            apiService = FakeInferenceApiService(),
            ioDispatcher = testDispatcher,
            configStore = InMemoryServerConfigStore()
        )

        viewModel.selectPreset("emulator")
        assertEquals("emulator", viewModel.uiState.value.selectedPresetId)
        assertEquals("http://10.0.2.2:8000", viewModel.uiState.value.activeBaseUrl)

        viewModel.selectPreset("custom_lan")
        assertEquals("custom_lan", viewModel.uiState.value.selectedPresetId)
        assertEquals("http://192.168.1.100:8000", viewModel.uiState.value.activeBaseUrl)

        viewModel.selectPreset("usb_adb")
        assertEquals("usb_adb", viewModel.uiState.value.selectedPresetId)
        assertEquals("http://127.0.0.1:8000", viewModel.uiState.value.activeBaseUrl)
    }

    @Test
    fun testCustomHostAndPortMutation() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = HomeViewModel(
            apiService = FakeInferenceApiService(),
            ioDispatcher = testDispatcher,
            configStore = InMemoryServerConfigStore()
        )

        viewModel.selectPreset("custom_lan")
        viewModel.setCustomHost("192.168.1.222")
        assertEquals("192.168.1.222", viewModel.uiState.value.customHost)
        assertEquals("http://192.168.1.222:8000", viewModel.uiState.value.activeBaseUrl)

        viewModel.setCustomPort(8080)
        assertEquals(8080, viewModel.uiState.value.customPort)
        assertEquals("http://192.168.1.222:8080", viewModel.uiState.value.activeBaseUrl)
    }

    @Test
    fun testServerIpPersistenceAcrossViewModelInstances() {
        val testDispatcher = StandardTestDispatcher()
        val sharedStore = InMemoryServerConfigStore()
        val firstVm = HomeViewModel(
            apiService = FakeInferenceApiService(),
            ioDispatcher = testDispatcher,
            configStore = sharedStore
        )

        firstVm.setCustomHost("192.168.1.45")
        firstVm.setCustomPort(8123)

        val secondVm = HomeViewModel(
            apiService = FakeInferenceApiService(),
            ioDispatcher = testDispatcher,
            configStore = sharedStore
        )
        val restored = secondVm.uiState.value
        assertEquals("192.168.1.45", restored.customHost)
        assertEquals(8123, restored.customPort)
        assertEquals("http://192.168.1.45:8123", restored.activeBaseUrl)
        assertTrue(restored.isSaved)
    }

    @Test
    fun testHostInputParsingWithPortAndScheme() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = HomeViewModel(
            apiService = FakeInferenceApiService(),
            ioDispatcher = testDispatcher,
            configStore = InMemoryServerConfigStore()
        )

        viewModel.setCustomHost("http://192.168.1.50:9000/")
        val state = viewModel.uiState.value
        assertEquals("192.168.1.50", state.customHost)
        assertEquals(9000, state.customPort)
        assertEquals("9000", state.portInput)
        assertEquals("http://192.168.1.50:9000", state.activeBaseUrl)
    }

    @Test
    fun testHealthCheckSuccess() = runTest {
        val testDispatcher = StandardTestDispatcher(testScheduler)
        val fakeApi = FakeInferenceApiService(shouldFail = false)
        val viewModel = HomeViewModel(
            apiService = fakeApi,
            ioDispatcher = testDispatcher,
            configStore = InMemoryServerConfigStore()
        )

        viewModel.checkHealth()
        assertTrue(viewModel.uiState.value.isCheckingHealth)
        assertEquals(ConnectionStatus.CONNECTING, viewModel.uiState.value.connectionStatus)

        testScheduler.advanceUntilIdle()

        val state = viewModel.uiState.value
        assertFalse(state.isCheckingHealth)
        assertEquals(ConnectionStatus.CONNECTED, state.connectionStatus)
        assertNotNull(state.healthResponse)
        assertEquals("ok", state.healthResponse?.status)
        assertEquals("stock2stove-cv-inference", state.healthResponse?.service)
        assertNotNull(state.pingLatencyMs)
        assertNull(state.errorMessage)
    }

    @Test
    fun testHealthCheckFailure() = runTest {
        val testDispatcher = StandardTestDispatcher(testScheduler)
        val fakeApi = FakeInferenceApiService(shouldFail = true, failureMessage = "SocketTimeoutException")
        val viewModel = HomeViewModel(
            apiService = fakeApi,
            ioDispatcher = testDispatcher,
            configStore = InMemoryServerConfigStore()
        )

        viewModel.checkHealth()
        testScheduler.advanceUntilIdle()

        val state = viewModel.uiState.value
        assertFalse(state.isCheckingHealth)
        assertEquals(ConnectionStatus.ERROR, state.connectionStatus)
        assertNull(state.healthResponse)
        assertEquals("SocketTimeoutException", state.errorMessage)
        assertNotNull(state.pingLatencyMs)
    }

    @Test
    fun testHttpsCloudflareTunnelUrlParsingAndPersistence() {
        val testDispatcher = StandardTestDispatcher()
        val sharedStore = InMemoryServerConfigStore()
        val firstVm = HomeViewModel(
            apiService = FakeInferenceApiService(),
            ioDispatcher = testDispatcher,
            configStore = sharedStore
        )

        firstVm.setCustomHost("https://kitchen-demo.trycloudflare.com/")
        val state = firstVm.uiState.value
        assertEquals("https_tunnel", state.selectedPresetId)
        assertEquals("https", state.scheme)
        assertEquals("kitchen-demo.trycloudflare.com", state.customHost)
        assertEquals(443, state.customPort)
        assertEquals("443", state.portInput)
        assertEquals("https://kitchen-demo.trycloudflare.com", state.activeBaseUrl)

        val secondVm = HomeViewModel(
            apiService = FakeInferenceApiService(),
            ioDispatcher = testDispatcher,
            configStore = sharedStore
        )
        val restored = secondVm.uiState.value
        assertEquals("https", restored.scheme)
        assertEquals("kitchen-demo.trycloudflare.com", restored.customHost)
        assertEquals(443, restored.customPort)
        assertEquals("https://kitchen-demo.trycloudflare.com", restored.activeBaseUrl)
    }

    @Test
    fun testHttpsToggleAndTunnelDomainAutoDetection() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = HomeViewModel(
            apiService = FakeInferenceApiService(),
            ioDispatcher = testDispatcher,
            configStore = InMemoryServerConfigStore()
        )

        viewModel.setCustomHost("my-stove.trycloudflare.com")
        assertEquals("https", viewModel.uiState.value.scheme)
        assertEquals(443, viewModel.uiState.value.customPort)
        assertEquals("https://my-stove.trycloudflare.com", viewModel.uiState.value.activeBaseUrl)

        viewModel.setUseHttps(false)
        assertEquals("http", viewModel.uiState.value.scheme)
        assertEquals(8000, viewModel.uiState.value.customPort)
        assertEquals("http://my-stove.trycloudflare.com:8000", viewModel.uiState.value.activeBaseUrl)

        viewModel.setUseHttps(true)
        assertEquals("https", viewModel.uiState.value.scheme)
        assertEquals(443, viewModel.uiState.value.customPort)
        assertEquals("https://my-stove.trycloudflare.com", viewModel.uiState.value.activeBaseUrl)
    }
}
