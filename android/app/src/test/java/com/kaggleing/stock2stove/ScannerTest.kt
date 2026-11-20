package com.kaggleing.stock2stove

import com.kaggleing.stock2stove.data.api.FrameMessage
import com.kaggleing.stock2stove.data.api.NetworkClientProvider
import com.kaggleing.stock2stove.data.model.ConfirmedItem
import com.kaggleing.stock2stove.data.model.DetectionItem
import com.kaggleing.stock2stove.data.model.DetectorDefenseMode
import com.kaggleing.stock2stove.data.model.DiagnosticsData
import com.kaggleing.stock2stove.data.model.ScanFrameRequest
import com.kaggleing.stock2stove.data.model.ScanMode
import com.kaggleing.stock2stove.data.model.ScanResponse
import com.kaggleing.stock2stove.ui.scanner.CameraFrameProcessor
import com.kaggleing.stock2stove.ui.scanner.OverlayMath
import com.kaggleing.stock2stove.ui.scanner.ProgressRingMath
import com.kaggleing.stock2stove.ui.scanner.ScannerViewModel
import com.kaggleing.stock2stove.ui.scanner.TemporalConfirmationTracker
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class ScannerTest {

    @Test
    fun testWebSocketFrameRequestSerialization() {
        val request = ScanFrameRequest(
            frame_id = 42L,
            detector_mode = "yolov8",
            enable_clahe = true,
            conf_threshold = 0.65f,
            image_b64 = "dGVzdF9pbWFnZQ==",
            session_id = "sess_123",
            reset_session = false
        )

        val jsonString = NetworkClientProvider.json.encodeToString(ScanFrameRequest.serializer(), request)
        assertTrue(jsonString.contains("\"frame_id\":42"))
        assertTrue(jsonString.contains("\"detector_mode\":\"yolov8\""))
        assertTrue(jsonString.contains("\"enable_clahe\":true"))
        assertTrue(jsonString.contains("\"session_id\":\"sess_123\""))
        assertTrue(jsonString.contains("\"reset_session\":false"))

        val decoded = NetworkClientProvider.json.decodeFromString<ScanFrameRequest>(jsonString)
        assertEquals(42L, decoded.frame_id)
        assertEquals("yolov8", decoded.detector_mode)
        assertTrue(decoded.enable_clahe)
        assertEquals(0.65f, decoded.conf_threshold, 0.001f)
        assertEquals("dGVzdF9pbWFnZQ==", decoded.image_b64)
        assertEquals("sess_123", decoded.session_id)
        assertFalse(decoded.reset_session)
    }

    @Test
    fun testFrameMessageTypeAlias() {
        val frame: FrameMessage = ScanFrameRequest(frame_id = 100L, detector_mode = "orb_baseline")
        assertEquals(100L, frame.frame_id)
        assertEquals("orb_baseline", frame.detector_mode)
    }

    @Test
    fun testWebSocketScanResponseDeserialization() {
        val rawJson = """
            {
              "frame_id": 105,
              "detector_mode": "yolov8",
              "inference_ms": 14.8,
              "fps": 8.5,
              "diagnostics": {
                "edge_density_pct": 22.4,
                "edge_density": 0.224,
                "orb_keypoints": 450,
                "keypoints_detected": 450,
                "inliers": 38,
                "ransac_inliers": 38,
                "good_matches": 55,
                "clahe_applied": true
              },
              "detections": [
                {
                  "class_id": "canned_tuna",
                  "display_name": "Canned Tuna",
                  "confidence": 0.88,
                  "bbox_norm": [0.1, 0.2, 0.5, 0.7],
                  "polygon_norm": [
                    [0.1, 0.2],
                    [0.5, 0.2],
                    [0.5, 0.7],
                    [0.1, 0.7]
                  ],
                  "temporally_confirmed": true
                }
              ],
              "confirmed_inventory": [
                {
                  "class_id": "canned_tuna",
                  "display_name": "Canned Tuna",
                  "count": 2,
                  "max_conf": 0.88
                }
              ]
            }
        """.trimIndent()

        val parsed = NetworkClientProvider.json.decodeFromString<ScanResponse>(rawJson)
        assertEquals(105L, parsed.frame_id)
        assertEquals("yolov8", parsed.detector_mode)
        assertEquals(14.8f, parsed.inference_ms, 0.01f)
        assertEquals(8.5f, parsed.fps, 0.01f)
        assertEquals(22.4f, parsed.diagnostics.edge_density_pct, 0.01f)
        assertEquals(450, parsed.diagnostics.orb_keypoints)
        assertEquals(38, parsed.diagnostics.inliers)
        assertTrue(parsed.diagnostics.clahe_applied)

        assertEquals(1, parsed.detections.size)
        val det = parsed.detections[0]
        assertEquals("canned_tuna", det.class_id)
        assertEquals("Canned Tuna", det.display_name)
        assertEquals(0.88f, det.confidence, 0.01f)
        assertTrue(det.temporally_confirmed)
        assertNotNull(det.polygon_norm)
        assertEquals(4, det.polygon_norm?.size)

        assertEquals(1, parsed.confirmed_inventory.size)
        val item = parsed.confirmed_inventory[0]
        assertEquals("canned_tuna", item.class_id)
        assertEquals(2, item.count)
        assertEquals(0.88f, item.max_conf, 0.01f)
    }

    @Test
    fun testOverlayBoundingBoxNormalization() {
        val bbox = listOf(0.1f, 0.2f, 0.6f, 0.8f)
        val rect = OverlayMath.mapNormalizedBbox(bbox, 1000f, 500f)

        assertEquals(100f, rect.left, 0.01f)
        assertEquals(100f, rect.top, 0.01f)
        assertEquals(600f, rect.right, 0.01f)
        assertEquals(400f, rect.bottom, 0.01f)
        assertEquals(500f, rect.width, 0.01f)
        assertEquals(300f, rect.height, 0.01f)

        val emptyRect = OverlayMath.mapNormalizedBbox(emptyList(), 1000f, 500f)
        assertEquals(0f, emptyRect.width, 0.01f)
        assertEquals(0f, emptyRect.height, 0.01f)

        val clampedRect = OverlayMath.mapNormalizedBbox(listOf(-0.2f, 1.5f, 0.5f, 0.5f), 100f, 100f)
        assertEquals(0f, clampedRect.left, 0.01f)
        assertEquals(100f, clampedRect.bottom, 0.01f)
    }

    @Test
    fun testOverlayPolygonNormalization() {
        val polygon = listOf(
            listOf(0.1f, 0.1f),
            listOf(0.9f, 0.1f),
            listOf(0.8f, 0.8f),
            listOf(0.2f, 0.8f)
        )
        val offsets = OverlayMath.mapNormalizedPolygon(polygon, 800f, 600f)
        assertEquals(4, offsets.size)
        assertEquals(80f, offsets[0].x, 0.01f)
        assertEquals(60f, offsets[0].y, 0.01f)
        assertEquals(720f, offsets[1].x, 0.01f)
        assertEquals(60f, offsets[1].y, 0.01f)
        assertEquals(640f, offsets[2].x, 0.01f)
        assertEquals(480f, offsets[2].y, 0.01f)
        assertEquals(160f, offsets[3].x, 0.01f)
        assertEquals(480f, offsets[3].y, 0.01f)
    }

    @Test
    fun testOverlayFormatLabelText() {
        val label = OverlayMath.formatLabelText("Eggs", 0.942f)
        assertEquals("Eggs 94%", label)
    }

    @Test
    fun testTemporalConfirmationLogic() {
        assertTrue(OverlayMath.isTemporallyConfirmed(temporallyConfirmed = true, hitCount = 1))
        assertFalse(OverlayMath.isTemporallyConfirmed(temporallyConfirmed = false, hitCount = 2))
        assertTrue(OverlayMath.isTemporallyConfirmed(temporallyConfirmed = false, hitCount = 3))
        assertTrue(OverlayMath.isTemporallyConfirmed(temporallyConfirmed = false, hitCount = 5))

        val confirmedColor = OverlayMath.getConfirmationColor(true, 0)
        val trackingColor = OverlayMath.getConfirmationColor(false, 2)
        val hitConfirmedColor = OverlayMath.getConfirmationColor(false, 3)

        assertEquals(OverlayMath.CONFIRMED_COLOR, confirmedColor)
        assertEquals(OverlayMath.TRACKING_COLOR, trackingColor)
        assertEquals(OverlayMath.CONFIRMED_COLOR, hitConfirmedColor)
    }

    @Test
    fun testTemporalConfirmationTrackerSlidingWindow() {
        val tracker = TemporalConfirmationTracker(thresholdHits = 3, windowSize = 5)

        val det1 = DetectionItem(class_id = "milk", display_name = "Milk", confidence = 0.9f)
        tracker.recordFrame(listOf(det1))
        assertEquals(1, tracker.getHitCount("milk"))
        assertFalse(tracker.isConfirmed("milk"))

        tracker.recordFrame(listOf(det1))
        assertEquals(2, tracker.getHitCount("milk"))
        assertFalse(tracker.isConfirmed("milk"))

        tracker.recordFrame(listOf(det1))
        assertEquals(3, tracker.getHitCount("milk"))
        assertTrue(tracker.isConfirmed("milk"))

        val tally = tracker.calculateConfirmedTally()
        assertEquals(1, tally.size)
        assertEquals(3, tally["milk"])

        tracker.recordFrame(emptyList())
        tracker.recordFrame(emptyList())
        tracker.recordFrame(emptyList())
        assertEquals(2, tracker.getHitCount("milk"))
        assertFalse(tracker.isConfirmed("milk"))

        tracker.recordFrame(emptyList())
        assertEquals(1, tracker.getHitCount("milk"))

        tracker.reset()
        assertEquals(0, tracker.getHitCount("milk"))
    }

    @Test
    fun testProgressRingMathCalculations() {
        assertEquals(0.0f, ProgressRingMath.calculateProgressFraction(0.0f, 5.0f), 0.001f)
        assertEquals(0.5f, ProgressRingMath.calculateProgressFraction(2.5f, 5.0f), 0.001f)
        assertEquals(1.0f, ProgressRingMath.calculateProgressFraction(5.0f, 5.0f), 0.001f)
        assertEquals(1.0f, ProgressRingMath.calculateProgressFraction(7.0f, 5.0f), 0.001f)

        assertEquals(0f, ProgressRingMath.calculateSweepAngle(0.0f, 5.0f), 0.01f)
        assertEquals(180f, ProgressRingMath.calculateSweepAngle(2.5f, 5.0f), 0.01f)
        assertEquals(360f, ProgressRingMath.calculateSweepAngle(5.0f, 5.0f), 0.01f)

        assertEquals("2.5s / 5.0s", ProgressRingMath.formatTimerText(2.5f, 5.0f))
        assertFalse(ProgressRingMath.isCountdownComplete(4.9f, 5.0f))
        assertTrue(ProgressRingMath.isCountdownComplete(5.0f, 5.0f))
        assertTrue(ProgressRingMath.isCountdownComplete(5.2f, 5.0f))
    }

    @Test
    fun testCameraFrameProcessorThrottling() {
        assertTrue(CameraFrameProcessor.shouldCaptureFrame(1000L, 0L, 8))
        assertFalse(CameraFrameProcessor.shouldCaptureFrame(1100L, 1000L, 8))
        assertTrue(CameraFrameProcessor.shouldCaptureFrame(1125L, 1000L, 8))
        assertTrue(CameraFrameProcessor.shouldCaptureFrame(1200L, 1000L, 8))

        val base64 = CameraFrameProcessor.encodeBase64("hello world".toByteArray())
        assertEquals("aGVsbG8gd29ybGQ=", base64)
    }

    @Test
    fun testScannerViewModelStateTransitions() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = ScannerViewModel(coroutineDispatcher = testDispatcher)

        viewModel.setDetectorMode(DetectorDefenseMode.ORB_BASELINE)
        assertEquals(DetectorDefenseMode.ORB_BASELINE, viewModel.uiState.value.selectedMode)

        viewModel.setClaheEnabled(false)
        assertFalse(viewModel.uiState.value.claheEnabled)

        viewModel.setConfThreshold(0.75f)
        assertEquals(0.75f, viewModel.uiState.value.confThreshold, 0.001f)

        assertFalse(viewModel.uiState.value.isPaused)
        viewModel.togglePause()
        assertTrue(viewModel.uiState.value.isPaused)
        viewModel.togglePause()
        assertFalse(viewModel.uiState.value.isPaused)

        viewModel.advanceProgress(2.5f)
        assertEquals(2.5f, viewModel.uiState.value.progressSec, 0.01f)
        assertFalse(viewModel.uiState.value.isComplete)
        assertTrue(viewModel.uiState.value.isScanning)

        viewModel.advanceProgress(3.0f)
        assertEquals(5.0f, viewModel.uiState.value.progressSec, 0.01f)
        assertTrue(viewModel.uiState.value.isComplete)
        assertFalse(viewModel.uiState.value.isScanning)

        viewModel.resetScan()
        assertEquals(0.0f, viewModel.uiState.value.progressSec, 0.01f)
        assertFalse(viewModel.uiState.value.isComplete)
        assertTrue(viewModel.uiState.value.isScanning)
        assertFalse(viewModel.uiState.value.isPaused)
    }

    @Test
    fun testScannerViewModelResponseHandling() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = ScannerViewModel(coroutineDispatcher = testDispatcher)

        val mockResponse = ScanResponse(
            frame_id = 12L,
            detector_mode = "yolov8",
            inference_ms = 18.5f,
            fps = 9.2f,
            diagnostics = DiagnosticsData(edge_density_pct = 15.0f, orb_keypoints = 200, inliers = 15),
            detections = listOf(
                DetectionItem(class_id = "pasta", display_name = "Pasta", confidence = 0.92f, temporally_confirmed = true)
            ),
            confirmed_inventory = listOf(
                ConfirmedItem(class_id = "pasta", display_name = "Pasta", count = 1, max_conf = 0.92f)
            )
        )

        viewModel.handleScanResponse(mockResponse)
        val state = viewModel.uiState.value
        assertEquals(18.5f, state.inferenceMs, 0.01f)
        assertEquals(9.2f, state.fps, 0.01f)
        assertEquals(1, state.detections.size)
        assertEquals("pasta", state.detections[0].class_id)
        assertEquals(1, state.confirmedInventory.size)
        assertEquals("pasta", state.confirmedInventory[0].class_id)
        assertEquals(1, state.confirmedInventory[0].count)
    }

    @Test
    fun testDefaultDetectorModeIsGeminiVision() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = ScannerViewModel(coroutineDispatcher = testDispatcher)
        assertEquals(DetectorDefenseMode.GEMINI_VISION, viewModel.uiState.value.selectedMode)
    }

    @Test
    fun testRetryScanPreservesConfirmedInventoryAndMergesNewPass() {
        val testDispatcher = StandardTestDispatcher()
        val viewModel = ScannerViewModel(coroutineDispatcher = testDispatcher)

        val passOneResponse = ScanResponse(
            frame_id = 1L,
            detector_mode = "gemini_vision",
            inference_ms = 240.0f,
            fps = 4.2f,
            detections = listOf(
                DetectionItem(
                    class_id = "pasta_noodles",
                    display_name = "Spaghetti",
                    confidence = 0.91f,
                    temporally_confirmed = true
                )
            ),
            confirmed_inventory = listOf(
                ConfirmedItem(
                    class_id = "pasta_noodles",
                    display_name = "Spaghetti",
                    count = 1,
                    max_conf = 0.91f
                )
            )
        )
        viewModel.handleScanResponse(passOneResponse)
        viewModel.triggerComplete()
        assertTrue(viewModel.uiState.value.isComplete)
        assertEquals(1, viewModel.uiState.value.confirmedInventory.size)

        viewModel.retryScan()
        assertFalse(viewModel.uiState.value.isComplete)
        assertTrue(viewModel.uiState.value.isScanning)
        assertEquals(0.0f, viewModel.uiState.value.progressSec, 0.01f)
        assertTrue(viewModel.uiState.value.detections.isEmpty())
        assertEquals(1, viewModel.uiState.value.confirmedInventory.size)
        assertEquals("pasta_noodles", viewModel.uiState.value.confirmedInventory[0].class_id)

        val passTwoResponse = ScanResponse(
            frame_id = 1L,
            detector_mode = "gemini_vision",
            inference_ms = 210.0f,
            fps = 4.7f,
            detections = listOf(
                DetectionItem(
                    class_id = "eggs",
                    display_name = "Eggs",
                    confidence = 0.95f,
                    temporally_confirmed = true
                )
            ),
            confirmed_inventory = listOf(
                ConfirmedItem(
                    class_id = "eggs",
                    display_name = "Eggs",
                    count = 6,
                    max_conf = 0.95f
                )
            )
        )
        viewModel.handleScanResponse(passTwoResponse)
        val merged = viewModel.uiState.value.confirmedInventory
        assertEquals(2, merged.size)
        assertEquals("pasta_noodles", merged[0].class_id)
        assertEquals("eggs", merged[1].class_id)

        viewModel.removeConfirmedItem("pasta_noodles")
        assertEquals(1, viewModel.uiState.value.confirmedInventory.size)
        assertEquals("eggs", viewModel.uiState.value.confirmedInventory[0].class_id)

        viewModel.clearInventory()
        assertTrue(viewModel.uiState.value.confirmedInventory.isEmpty())
    }
}
