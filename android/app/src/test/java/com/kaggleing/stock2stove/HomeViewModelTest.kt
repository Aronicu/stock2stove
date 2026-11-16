package com.kaggleing.stock2stove

import com.kaggleing.stock2stove.data.api.InferenceApiService
import com.kaggleing.stock2stove.data.model.ConnectionStatus
import com.kaggleing.stock2stove.data.model.HealthResponse
import com.kaggleing.stock2stove.data.model.HardwareInfo
import com.kaggleing.stock2stove.data.model.ModelStatus
import com.kaggleing.stock2stove.data.model.RecipeGenerateRequest
import com.kaggleing.stock2stove.data.model.RecipeGenerateResponse
import com.kaggleing.stock2stove.data.model.ScanMode
import com.kaggleing.stock2stove.ui.home.HomeViewModel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
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
        val viewModel = HomeViewModel(apiService = FakeInferenceApiService(), ioDispatcher = testDispatcher)
        val state = viewModel.uiState.value

        assertEquals("usb_adb", state.selectedPresetId)
        assertEquals("http://127.0.0.1:8000", state.activeBaseUrl)
        assertEquals(ScanMode.PANTRY_SHELF, state.scanMode)
        assertEquals(ConnectionStatus.DISCONNECTED, state.connectionStatus)
        assertFalse(state.isCheckingHealth)
        assertNull(state.healthResponse)
        assertNull(state.errorMessage)
    }

    @Test
    fun testPresetSwitchingUpdatesActiveBaseUrl() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = HomeViewModel(apiService = FakeInferenceApiService(), ioDispatcher = testDispatcher)

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
        val viewModel = HomeViewModel(apiService = FakeInferenceApiService(), ioDispatcher = testDispatcher)

        viewModel.selectPreset("custom_lan")
        viewModel.setCustomHost("192.168.1.222")
        assertEquals("192.168.1.222", viewModel.uiState.value.customHost)
        assertEquals("http://192.168.1.222:8000", viewModel.uiState.value.activeBaseUrl)

        viewModel.setCustomPort(8080)
        assertEquals(8080, viewModel.uiState.value.customPort)
        assertEquals("http://192.168.1.222:8080", viewModel.uiState.value.activeBaseUrl)
    }

    @Test
    fun testScanModeSwitching() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = HomeViewModel(apiService = FakeInferenceApiService(), ioDispatcher = testDispatcher)

        viewModel.setScanMode(ScanMode.REFRIGERATOR)
        assertEquals(ScanMode.REFRIGERATOR, viewModel.uiState.value.scanMode)

        viewModel.setScanMode(ScanMode.PANTRY_SHELF)
        assertEquals(ScanMode.PANTRY_SHELF, viewModel.uiState.value.scanMode)
    }

    @Test
    fun testHealthCheckSuccess() = runTest {
        val testDispatcher = StandardTestDispatcher(testScheduler)
        val fakeApi = FakeInferenceApiService(shouldFail = false)
        val viewModel = HomeViewModel(apiService = fakeApi, ioDispatcher = testDispatcher)

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
        val viewModel = HomeViewModel(apiService = fakeApi, ioDispatcher = testDispatcher)

        viewModel.checkHealth()
        testScheduler.advanceUntilIdle()

        val state = viewModel.uiState.value
        assertFalse(state.isCheckingHealth)
        assertEquals(ConnectionStatus.ERROR, state.connectionStatus)
        assertNull(state.healthResponse)
        assertEquals("SocketTimeoutException", state.errorMessage)
        assertNotNull(state.pingLatencyMs)
    }
}
