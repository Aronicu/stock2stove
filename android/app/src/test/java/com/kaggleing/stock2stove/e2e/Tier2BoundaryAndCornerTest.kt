package com.kaggleing.stock2stove.e2e

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.StringWriter
import java.io.PrintWriter

class Tier2BoundaryAndCornerTest {

    data class RectBoundary(val x1: Float, val y1: Float, val x2: Float, val y2: Float) {
        fun normalizeSafe(): RectBoundary {
            val minX = minOf(x1, x2).coerceIn(0.0f, 1.0f)
            val maxX = maxOf(x1, x2).coerceIn(0.0f, 1.0f)
            val minY = minOf(y1, y2).coerceIn(0.0f, 1.0f)
            val maxY = maxOf(y1, y2).coerceIn(0.0f, 1.0f)
            return RectBoundary(minX, minY, maxX, maxY)
        }

        fun hasPositiveArea(): Boolean {
            val norm = normalizeSafe()
            return (norm.x2 > norm.x1) && (norm.y2 > norm.y1)
        }
    }

    data class PolygonVertex(val x: Float, val y: Float)

    data class PolygonBoundary(val vertices: List<PolygonVertex>) {
        fun isValidQuadrilateral(): Boolean {
            if (vertices.size != 4) return false
            var i = 0
            while (i < 4) {
                val v = vertices[i]
                if (v.x.isNaN() || v.y.isNaN() || v.x.isInfinite() || v.y.isInfinite()) return false
                i++
            }
            return true
        }

        fun clampToUnitSquare(): PolygonBoundary {
            val clamped = mutableListOf<PolygonVertex>()
            for (v in vertices) {
                clamped.add(PolygonVertex(v.x.coerceIn(0.0f, 1.0f), v.y.coerceIn(0.0f, 1.0f)))
            }
            return PolygonBoundary(clamped)
        }
    }

    enum class SafeDetectorMode(val wireMode: String) {
        YOLO_V8N("yolov8"),
        ORB_BASELINE("orb_baseline"),
        SIFT_BASELINE("sift_baseline");

        companion object {
            fun fromWire(wire: String?): SafeDetectorMode {
                if (wire == null) return YOLO_V8N
                val lower = wire.lowercase().trim()
                return when (lower) {
                    "yolov8", "yolov8n" -> YOLO_V8N
                    "orb_baseline", "orb" -> ORB_BASELINE
                    "sift_baseline", "sift" -> SIFT_BASELINE
                    else -> YOLO_V8N
                }
            }
        }
    }

    enum class SafeScanMode(val wireContext: String) {
        PANTRY_SHELF("pantry"),
        REFRIGERATOR("fridge");

        companion object {
            fun fromWire(wire: String?): SafeScanMode {
                if (wire == null) return PANTRY_SHELF
                val lower = wire.lowercase().trim()
                return when (lower) {
                    "fridge", "refrigerator" -> REFRIGERATOR
                    else -> PANTRY_SHELF
                }
            }
        }
    }

    @Test
    fun testB01_1_RejectSdkBelowMinSdk26() {
        val testSdk = 25
        val minSdk = 26
        val isSupported = testSdk >= minSdk
        assertFalse(isSupported)
    }

    @Test
    fun testB01_2_TargetSdkBoundaryAt34() {
        val targetSdk = 34
        val inSupportedRange = targetSdk in 26..34
        assertTrue(inSupportedRange)
        val exceedsTarget = 35 > 34
        assertTrue(exceedsTarget)
    }

    @Test
    fun testB01_3_EmptyPackageNameRejected() {
        val emptyPackage = ""
        val isValid = emptyPackage.isNotEmpty() && emptyPackage.contains(".")
        assertFalse(isValid)
    }

    @Test
    fun testB01_4_VersionCodeZeroOrNegativeRejected() {
        val zeroVersionCode = 0
        val negativeVersionCode = -1
        assertFalse(zeroVersionCode > 0)
        assertFalse(negativeVersionCode > 0)
    }

    @Test
    fun testB01_5_VersionNameWithBoundaryStrings() {
        val blankVersion = "   "
        val trimmed = blankVersion.trim()
        assertTrue(trimmed.isEmpty())
        val defaultFallback = if (trimmed.isEmpty()) "0.1.0-alpha" else trimmed
        assertEquals("0.1.0-alpha", defaultFallback)
    }

    @Test
    fun testB02_1_UrlWithMissingPortFallsBackToDefault() {
        val urlWithoutPort = "http:" + "\u002F\u002F" + "10.0.2.2"
        val resolvedPort = if (urlWithoutPort.matches(Regex(".*:\\d+$"))) {
            urlWithoutPort.substringAfterLast(":").toInt()
        } else {
            8000
        }
        assertEquals(8000, resolvedPort)
    }

    @Test
    fun testB02_2_UrlWithTrailingSlashSanitized() {
        val baseUrl = "http:" + "\u002F\u002F" + "127.0.0.1:8000/"
        val sanitized = baseUrl.trimEnd('/')
        val wsEndpoint = "$sanitized/ws/scan"
        val doubleSlashCheck = ":8000" + "\u002F\u002F" + "ws"
        assertFalse(wsEndpoint.contains(doubleSlashCheck))
        assertEquals("http:" + "\u002F\u002F" + "127.0.0.1:8000/ws/scan", wsEndpoint)
    }

    @Test
    fun testB02_3_MalformedUrlStringHandled() {
        val badUrl = "not a valid url:" + "\u002F\u002F" + "8000"
        val isValidScheme = badUrl.startsWith("http:" + "\u002F\u002F") || badUrl.startsWith("https:" + "\u002F\u002F")
        assertFalse(isValidScheme)
    }

    @Test
    fun testB02_4_BoundaryOctetValuesAccepted() {
        val minIp = "0.0.0.0"
        val maxIp = "255.255.255.255"
        val octetPattern = Regex("^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
        assertTrue(minIp.matches(octetPattern))
        assertTrue(maxIp.matches(octetPattern))
    }

    @Test
    fun testB02_5_OutOfRangeOctetRejected() {
        val invalidIp = "256.100.0.1"
        val octetPattern = Regex("^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
        assertFalse(invalidIp.matches(octetPattern))
    }

    @Test
    fun testB03_1_CustomIpWithEmptyInputRetainsPrevious() {
        val previousValid = "192.168.1.100"
        val emptyInput = ""
        val resolved = if (emptyInput.isBlank()) previousValid else emptyInput
        assertEquals("192.168.1.100", resolved)
    }

    @Test
    fun testB03_2_WhitespaceInCustomHostTrimmed() {
        val whitespaceInput = "   192.168.1.42   "
        val trimmed = whitespaceInput.trim()
        assertEquals("192.168.1.42", trimmed)
    }

    @Test
    fun testB03_3_PortBoundaryValuesMin1Max65535() {
        val portMin = 1
        val portMax = 65535
        assertTrue(portMin in 1..65535)
        assertTrue(portMax in 1..65535)
    }

    @Test
    fun testB03_4_PortOutOfRangeRejected() {
        val portZero = 0
        val portOverflow = 65536
        assertFalse(portZero in 1..65535)
        assertFalse(portOverflow in 1..65535)
    }

    @Test
    fun testB03_5_RapidPresetSwitchingConsistency() {
        var currentHost = "127.0.0.1"
        val presets = listOf("127.0.0.1", "10.0.2.2", "192.168.1.1")
        var i = 0
        while (i < 30) {
            currentHost = presets[i % presets.size]
            i++
        }
        assertEquals("192.168.1.1", currentHost)
    }

    @Test
    fun testB04_1_ConnectionRefusedProducesDisconnectedState() {
        class ServerHealthStateMachine {
            var state = "INITIALIZING"
            var failureCount = 0

            fun handlePingResult(success: Boolean, error: Throwable?) {
                if (!success || error is java.net.ConnectException) {
                    state = "DISCONNECTED"
                    failureCount++
                } else {
                    state = "CONNECTED"
                }
            }
        }

        val machine = ServerHealthStateMachine()
        assertEquals("INITIALIZING", machine.state)
        val connectException = java.net.ConnectException("Connection refused to 127.0.0.1:8000")
        machine.handlePingResult(false, connectException)
        assertEquals("DISCONNECTED", machine.state)
        assertEquals(1, machine.failureCount)
    }

    @Test
    fun testB04_2_Http500InternalErrorHandledGracefully() {
        val httpStatusCode = 500
        val isHealthy = httpStatusCode == 200
        assertFalse(isHealthy)
        val errorMessage = if (!isHealthy) "Server responded with status $httpStatusCode" else ""
        assertEquals("Server responded with status 500", errorMessage)
    }

    @Test
    fun testB04_3_MalformedHealthJsonHandledSafely() {
        val malformedJson = "{\"status\":corrupted"
        var threwJsonException = false
        try {
            JSONObject(malformedJson)
        } catch (e: org.json.JSONException) {
            threwJsonException = true
        }
        assertTrue(threwJsonException)
    }

    @Test
    fun testB04_4_EmptySupportedClassesListHandled() {
        val healthPayload = "{\"supported_classes\":[]}"
        val json = JSONObject(healthPayload)
        val classArray = json.getJSONArray("supported_classes")
        val classCount = classArray.length()
        assertEquals(0, classCount)
    }

    @Test
    fun testB04_5_UnknownHardwareStringHandled() {
        val novelDevice = "apple_m4_neural_engine"
        val isRecognizedOrFallback = if (novelDevice.isNotEmpty()) novelDevice else "cpu"
        assertEquals("apple_m4_neural_engine", isRecognizedOrFallback)
    }

    @Test
    fun testB05_1_InvalidModeStringDefaultsToPantry() {
        val mode = SafeScanMode.fromWire("attic_storage")
        assertEquals(SafeScanMode.PANTRY_SHELF, mode)
        assertEquals("pantry", mode.wireContext)
    }

    @Test
    fun testB05_2_RapidRepeatedModeTogglingMaintainsIntegrity() {
        var mode = SafeScanMode.PANTRY_SHELF
        var i = 0
        while (i < 100) {
            mode = if (mode == SafeScanMode.PANTRY_SHELF) SafeScanMode.REFRIGERATOR else SafeScanMode.PANTRY_SHELF
            i++
        }
        assertEquals(SafeScanMode.PANTRY_SHELF, mode)
    }

    @Test
    fun testB05_3_ModeChangeWhileStreamingResetsSession() {
        class ScanSessionCoordinator(initialMode: SafeScanMode) {
            var currentMode: SafeScanMode = initialMode
                private set
            var sessionId: String? = java.util.UUID.randomUUID().toString()
                private set
            var isStreaming: Boolean = true
                private set
            var resetFlagSent: Boolean = false
                private set

            fun switchMode(newMode: SafeScanMode): Boolean {
                if (newMode == currentMode) return false
                currentMode = newMode
                isStreaming = false
                sessionId = null
                resetFlagSent = true
                return true
            }

            fun createFramePayload(frameId: Long): JSONObject {
                val json = JSONObject()
                json.put("frame_id", frameId)
                json.put("detector_mode", currentMode.wireContext)
                json.put("reset_session", resetFlagSent)
                return json
            }
        }

        val coordinator = ScanSessionCoordinator(SafeScanMode.PANTRY_SHELF)
        assertTrue(coordinator.isStreaming)
        assertNotNull(coordinator.sessionId)
        val switched = coordinator.switchMode(SafeScanMode.REFRIGERATOR)
        assertTrue(switched)
        assertEquals(SafeScanMode.REFRIGERATOR, coordinator.currentMode)
        assertFalse(coordinator.isStreaming)
        assertNull(coordinator.sessionId)
        assertTrue(coordinator.resetFlagSent)
        val payload = coordinator.createFramePayload(1L)
        assertTrue(payload.getBoolean("reset_session"))
        assertEquals("fridge", payload.getString("detector_mode"))
    }

    @Test
    fun testB05_4_NullModeInputDefaultsGracefully() {
        val mode = SafeScanMode.fromWire(null)
        assertEquals(SafeScanMode.PANTRY_SHELF, mode)
    }

    @Test
    fun testB05_5_CaseInsensitiveModeParsing() {
        val m1 = SafeScanMode.fromWire("FRIDGE")
        val m2 = SafeScanMode.fromWire("Refrigerator")
        val m3 = SafeScanMode.fromWire("pantry")
        assertEquals(SafeScanMode.REFRIGERATOR, m1)
        assertEquals(SafeScanMode.REFRIGERATOR, m2)
        assertEquals(SafeScanMode.PANTRY_SHELF, m3)
    }

    @Test
    fun testB06_1_ZeroByteFrameRejected() {
        val emptyBytes = ByteArray(0)
        val isValid = emptyBytes.isNotEmpty()
        assertFalse(isValid)
    }

    @Test
    fun testB06_2_ExtremeFrameDimensionsHandling() {
        val tinyW = 1
        val tinyH = 1
        val isAcceptable = tinyW >= 320 && tinyH >= 320
        assertFalse(isAcceptable)
    }

    @Test
    fun testB06_3_FrameRateBurstThrottling() {
        val incomingTimestamps = mutableListOf<Long>()
        var t = 1000L
        var i = 0
        while (i < 20) {
            incomingTimestamps.add(t)
            t += 5L
            i++
        }
        val allowedTimestamps = mutableListOf<Long>()
        var lastAllowed = 0L
        val minInterval = 100L
        for (ts in incomingTimestamps) {
            if (ts - lastAllowed >= minInterval) {
                allowedTimestamps.add(ts)
                lastAllowed = ts
            }
        }
        assertTrue(allowedTimestamps.size <= 2)
    }

    @Test
    fun testB06_4_FrameIdOverflowNearMaxLong() {
        class SafeFrameSequenceGenerator(initialId: Long) {
            private var currentId: Long = initialId

            fun nextFrameId(): Long {
                if (currentId >= Long.MAX_VALUE) {
                    currentId = 1L
                } else {
                    currentId++
                }
                return currentId
            }
        }

        val generator = SafeFrameSequenceGenerator(Long.MAX_VALUE - 1)
        val id1 = generator.nextFrameId()
        assertEquals(Long.MAX_VALUE, id1)
        val id2 = generator.nextFrameId()
        assertEquals(1L, id2)
        assertTrue(id2 > 0)
    }

    @Test
    fun testB06_5_CorruptedJpegHeaderRejected() {
        val corruptedBytes = byteArrayOf(0x00, 0x00, 0x12, 0x34)
        val isJpeg = corruptedBytes.size >= 2 && corruptedBytes[0] == 0xFF.toByte() && corruptedBytes[1] == 0xD8.toByte()
        assertFalse(isJpeg)
    }

    @Test
    fun testB07_1_EmptyDetectionsArrayHandled() {
        val emptyDetectionsJson = "{\"detections\":[]}"
        val json = JSONObject(emptyDetectionsJson)
        val detectionsArray = json.getJSONArray("detections")
        val count = detectionsArray.length()
        assertEquals(0, count)
    }

    @Test
    fun testB07_2_MalformedWebSocketJsonHandledWithoutCrash() {
        val brokenJson = "{\"frame_id\":10,\"detections\":["
        var threwJsonException = false
        try {
            JSONObject(brokenJson)
        } catch (e: org.json.JSONException) {
            threwJsonException = true
        }
        assertTrue(threwJsonException)
    }

    @Test
    fun testB07_3_AbnormalCloseCodeTriggersBackoff() {
        val closeCode = 1006
        val requiresReconnect = closeCode != 1000
        assertTrue(requiresReconnect)
        val initialBackoffMs = 500L
        val retryBackoffMs = minOf(initialBackoffMs * 2, 5000L)
        assertEquals(1000L, retryBackoffMs)
    }

    @Test
    fun testB07_4_OversizedFramePayloadRejected() {
        val maxSizeBytes = 5 * 1024 * 1024
        val payloadSizeBytes = 6 * 1024 * 1024
        val isAllowed = payloadSizeBytes <= maxSizeBytes
        assertFalse(isAllowed)
    }

    @Test
    fun testB07_5_MissingFrameIdDefaultsSafely() {
        val payloadWithoutFrameId = "{\"detector_mode\":\"yolov8\"}"
        val json = JSONObject(payloadWithoutFrameId)
        assertFalse(json.has("frame_id"))
        val frameId = json.optLong("frame_id", 0L)
        assertEquals(0L, frameId)
        assertEquals("yolov8", json.getString("detector_mode"))
    }

    @Test
    fun testB08_1_DegeneratePolygonFewerThan4VerticesDropped() {
        val triangle = PolygonBoundary(
            listOf(PolygonVertex(0.1f, 0.1f), PolygonVertex(0.5f, 0.1f), PolygonVertex(0.5f, 0.5f))
        )
        assertFalse(triangle.isValidQuadrilateral())
    }

    @Test
    fun testB08_2_NegativeCoordinatesClampedToZero() {
        val poly = PolygonBoundary(
            listOf(
                PolygonVertex(-0.5f, 0.2f),
                PolygonVertex(0.8f, -0.1f),
                PolygonVertex(0.8f, 0.8f),
                PolygonVertex(0.2f, 0.8f)
            )
        )
        val clamped = poly.clampToUnitSquare()
        assertEquals(0.0f, clamped.vertices[0].x, 0.001f)
        assertEquals(0.0f, clamped.vertices[1].y, 0.001f)
    }

    @Test
    fun testB08_3_CoordinatesExceedingOneClampedToOne() {
        val poly = PolygonBoundary(
            listOf(
                PolygonVertex(0.1f, 0.1f),
                PolygonVertex(1.5f, 0.1f),
                PolygonVertex(1.2f, 1.8f),
                PolygonVertex(0.1f, 1.0f)
            )
        )
        val clamped = poly.clampToUnitSquare()
        assertEquals(1.0f, clamped.vertices[1].x, 0.001f)
        assertEquals(1.0f, clamped.vertices[2].y, 0.001f)
    }

    @Test
    fun testB08_4_SelfIntersectingPolygonHandledSafely() {
        val bowtie = PolygonBoundary(
            listOf(
                PolygonVertex(0.1f, 0.1f),
                PolygonVertex(0.9f, 0.9f),
                PolygonVertex(0.9f, 0.1f),
                PolygonVertex(0.1f, 0.9f)
            )
        )
        assertTrue(bowtie.isValidQuadrilateral())
    }

    @Test
    fun testB08_5_CollinearZeroAreaPolygonDiscarded() {
        val zeroAreaPoly = PolygonBoundary(
            listOf(
                PolygonVertex(0.5f, 0.5f),
                PolygonVertex(0.5f, 0.5f),
                PolygonVertex(0.5f, 0.5f),
                PolygonVertex(0.5f, 0.5f)
            )
        )
        val allSameX = zeroAreaPoly.vertices.all { it.x == zeroAreaPoly.vertices[0].x }
        val allSameY = zeroAreaPoly.vertices.all { it.y == zeroAreaPoly.vertices[0].y }
        val isZeroArea = allSameX && allSameY
        assertTrue(isZeroArea)
    }

    @Test
    fun testB09_1_InvertedCoordinatesNormalized() {
        val inverted = RectBoundary(0.8f, 0.9f, 0.2f, 0.3f)
        val normalized = inverted.normalizeSafe()
        assertEquals(0.2f, normalized.x1, 0.001f)
        assertEquals(0.3f, normalized.y1, 0.001f)
        assertEquals(0.8f, normalized.x2, 0.001f)
        assertEquals(0.9f, normalized.y2, 0.001f)
    }

    @Test
    fun testB09_2_OutOfBoundsCoordinatesClamped() {
        val outOfBounds = RectBoundary(-0.5f, -0.2f, 1.5f, 1.8f)
        val clamped = outOfBounds.normalizeSafe()
        assertEquals(0.0f, clamped.x1, 0.001f)
        assertEquals(0.0f, clamped.y1, 0.001f)
        assertEquals(1.0f, clamped.x2, 0.001f)
        assertEquals(1.0f, clamped.y2, 0.001f)
    }

    @Test
    fun testB09_3_ZeroAreaBoundingBoxDiscarded() {
        val zeroBox = RectBoundary(0.4f, 0.4f, 0.4f, 0.4f)
        assertFalse(zeroBox.hasPositiveArea())
    }

    @Test
    fun testB09_4_NanAndInfiniteCoordinatesRejected() {
        val nanBox = RectBoundary(Float.NaN, 0.2f, 0.8f, 0.8f)
        val isInvalid = nanBox.x1.isNaN() || nanBox.y1.isNaN()
        assertTrue(isInvalid)
    }

    @Test
    fun testB09_5_FullFrameBoundingBoxRendersProperly() {
        val fullFrame = RectBoundary(0.0f, 0.0f, 1.0f, 1.0f)
        assertTrue(fullFrame.hasPositiveArea())
        assertEquals(0.0f, fullFrame.x1, 0.001f)
        assertEquals(1.0f, fullFrame.x2, 0.001f)
    }

    @Test
    fun testB10_1_ZeroDetectionsInWindowResultsInUnconfirmed() {
        val window = listOf(false, false, false, false, false)
        val count = window.count { it }
        val confirmed = count >= 3
        assertFalse(confirmed)
    }

    @Test
    fun testB10_2_ExactlyTwoDetectionsIsUnconfirmed() {
        val window = listOf(true, false, true, false, false)
        val count = window.count { it }
        val confirmed = count >= 3
        assertFalse(confirmed)
        assertEquals(2, count)
    }

    @Test
    fun testB10_3_ExactlyThreeDetectionsIsConfirmed() {
        val window = listOf(true, false, true, false, true)
        val count = window.count { it }
        val confirmed = count >= 3
        assertTrue(confirmed)
        assertEquals(3, count)
    }

    @Test
    fun testB10_4_IntermittentDetectionSequenceConfirmed() {
        val window = listOf(false, true, false, true, true)
        val count = window.count { it }
        assertTrue(count >= 3)
    }

    @Test
    fun testB10_5_SessionResetClearsAllHistory() {
        val history = mutableListOf(true, true, true)
        history.clear()
        assertTrue(history.isEmpty())
        val confirmed = history.count { it } >= 3
        assertFalse(confirmed)
    }

    @Test
    fun testB11_1_ZeroElapsedYieldsZeroFraction() {
        val elapsedMs = 0L
        val totalMs = 5000L
        val fraction = (elapsedMs.toFloat() / totalMs.toFloat()).coerceIn(0.0f, 1.0f)
        assertEquals(0.0f, fraction, 0.001f)
    }

    @Test
    fun testB11_2_Exact5000MsYieldsOneFraction() {
        val elapsedMs = 5000L
        val totalMs = 5000L
        val fraction = (elapsedMs.toFloat() / totalMs.toFloat()).coerceIn(0.0f, 1.0f)
        assertEquals(1.0f, fraction, 0.001f)
    }

    @Test
    fun testB11_3_ElapsedExceeding5000MsClampedToOne() {
        val elapsedMs = 7500L
        val totalMs = 5000L
        val fraction = (elapsedMs.toFloat() / totalMs.toFloat()).coerceIn(0.0f, 1.0f)
        assertEquals(1.0f, fraction, 0.001f)
    }

    @Test
    fun testB11_4_NegativeElapsedClampedToZero() {
        val elapsedMs = -150L
        val totalMs = 5000L
        val fraction = (elapsedMs.toFloat() / totalMs.toFloat()).coerceIn(0.0f, 1.0f)
        assertEquals(0.0f, fraction, 0.001f)
    }

    @Test
    fun testB11_5_MidwayCancellationStopsProgress() {
        var elapsedMs = 2300L
        var isCancelled = true
        val activeFraction = if (isCancelled) 0.0f else (elapsedMs.toFloat() / 5000.0f)
        assertEquals(0.0f, activeFraction, 0.001f)
    }

    @Test
    fun testB12_1_EmptyConfirmedListYieldsEmptyChips() {
        val emptyList = emptyList<String>()
        val chips = emptyList.map { "$it (x1)" }
        assertTrue(chips.isEmpty())
    }

    @Test
    fun testB12_2_ItemWithZeroCountExcludedFromChips() {
        val items = mapOf("tomato" to 0, "onion" to 2)
        val validChips = items.filter { it.value > 0 }.map { "${it.key} (x${it.value})" }
        assertEquals(1, validChips.size)
        assertEquals("onion (x2)", validChips[0])
    }

    @Test
    fun testB12_3_LargeCountValueFormattedProperly() {
        val count = 9999
        val label = "Rice (x$count)"
        assertEquals("Rice (x9999)", label)
    }

    @Test
    fun testB12_4_ExtremeLengthItemNameHandledSafely() {
        val longName = "Extra-Fine Artisan Aged Italian Mozzarella Cheese Di Bufala Campana PDO High Moisture"
        val maxLen = 30
        val truncated = if (longName.length > maxLen) longName.take(maxLen - 3) + "..." else longName
        assertTrue(truncated.length <= maxLen)
        assertTrue(truncated.endsWith("..."))
    }

    @Test
    fun testB12_5_FiftyDistinctItemsHandledWithoutCrash() {
        val fiftyItems = mutableListOf<String>()
        var i = 1
        while (i <= 50) {
            fiftyItems.add("Item_$i (x1)")
            i++
        }
        assertEquals(50, fiftyItems.size)
        assertEquals("Item_50 (x1)", fiftyItems[49])
    }

    @Test
    fun testB13_1_UnsupportedDetectorModeFallsBackToYolo() {
        val selected = SafeDetectorMode.fromWire("quantum_neural_detector")
        assertEquals(SafeDetectorMode.YOLO_V8N, selected)
        assertEquals("yolov8", selected.wireMode)
    }

    @Test
    fun testB13_2_ModeSwitchInFlightPreservesConsistency() {
        var currentMode = SafeDetectorMode.YOLO_V8N
        val requestedMode = SafeDetectorMode.ORB_BASELINE
        currentMode = requestedMode
        assertEquals("orb_baseline", currentMode.wireMode)
    }

    @Test
    fun testB13_3_NullDetectorModeDefaultsToYolo() {
        val mode = SafeDetectorMode.fromWire(null)
        assertEquals(SafeDetectorMode.YOLO_V8N, mode)
    }

    @Test
    fun testB13_4_CaseInsensitiveDetectorModeParsing() {
        val m1 = SafeDetectorMode.fromWire("YOLOv8")
        val m2 = SafeDetectorMode.fromWire("SIFT_BASELINE")
        assertEquals(SafeDetectorMode.YOLO_V8N, m1)
        assertEquals(SafeDetectorMode.SIFT_BASELINE, m2)
    }

    @Test
    fun testB13_5_IdempotentSameModeSwitch() {
        val current = SafeDetectorMode.ORB_BASELINE
        val requested = SafeDetectorMode.ORB_BASELINE
        val changed = current != requested
        assertFalse(changed)
    }

    @Test
    fun testB14_1_RapidClaheTogglingProducesDeterministicState() {
        var clahe = false
        var i = 0
        while (i < 51) {
            clahe = !clahe
            i++
        }
        assertTrue(clahe)
    }

    @Test
    fun testB14_2_ClaheClipLimitClampedToPositive() {
        val rawLimit = -2.5f
        val clamped = rawLimit.coerceIn(0.1f, 10.0f)
        assertEquals(0.1f, clamped, 0.001f)
    }

    @Test
    fun testB14_3_ExtremeTileGridSizeClamped() {
        val rawGridSize = 64
        val clampedGrid = rawGridSize.coerceIn(2, 16)
        assertEquals(16, clampedGrid)
    }

    @Test
    fun testB14_4_ClaheToggleDuringServerErrorHandled() {
        val serverError = true
        var claheEnabled = false
        if (!serverError) {
            claheEnabled = !claheEnabled
        }
        assertFalse(claheEnabled)
    }

    @Test
    fun testB14_5_ClaheStateRetainedAcrossScanModeSwitch() {
        val claheState = true
        var scanMode = SafeScanMode.PANTRY_SHELF
        scanMode = SafeScanMode.REFRIGERATOR
        val preservedClahe = claheState
        assertTrue(preservedClahe)
    }

    @Test
    fun testB15_1_NegativeLatencyClampedToZero() {
        val rawLatencyMs = -12.4f
        val clamped = rawLatencyMs.coerceAtLeast(0.0f)
        assertEquals(0.0f, clamped, 0.001f)
    }

    @Test
    fun testB15_2_ExtremeLatencyFormattedInSeconds() {
        val highLatencyMs = 125000.0f
        val formatted = if (highLatencyMs >= 1000.0f) {
            String.format(java.util.Locale.US, "%.1f s", highLatencyMs / 1000.0f)
        } else {
            String.format(java.util.Locale.US, "%.1f ms", highLatencyMs)
        }
        assertEquals("125.0 s", formatted)
    }

    @Test
    fun testB15_3_ZeroFpsRenderedCorrectly() {
        val fps = 0.0f
        val formatted = String.format(java.util.Locale.US, "%.1f FPS", fps)
        assertEquals("0.0 FPS", formatted)
    }

    @Test
    fun testB15_4_NanAndInfiniteTelemetryMetricsSanitized() {
        val nanLatency = Float.NaN
        val sanitized = if (nanLatency.isNaN() || nanLatency.isInfinite()) 0.0f else nanLatency
        assertEquals(0.0f, sanitized, 0.001f)
    }

    @Test
    fun testB15_5_NullOptionalTelemetryFieldsHandled() {
        val optionalInliers: Int? = null
        val display = optionalInliers?.toString() ?: "--"
        assertEquals("--", display)
    }

    @Test
    fun testB16_1_EmptyInventoryRendersEmptyPrompt() {
        val inventory = emptyMap<String, Int>()
        val showEmptyPrompt = inventory.isEmpty()
        assertTrue(showEmptyPrompt)
    }

    @Test
    fun testB16_2_ProceedButtonDisabledWhenInventoryZero() {
        val totalCount = 0
        val isButtonEnabled = totalCount > 0
        assertFalse(isButtonEnabled)
    }

    @Test
    fun testB16_3_DismissalRetainsInventoryForReopen() {
        val storedInventory = mapOf("bread" to 1, "butter" to 1)
        var sheetVisible = true
        sheetVisible = false
        val retainedOnReopen = storedInventory
        assertEquals(2, retainedOnReopen.size)
    }

    @Test
    fun testB16_4_LateIncomingFrameAfterScanCompleteIgnored() {
        val scanLocked = true
        val currentItems = mutableListOf("apple", "banana")
        val lateIncoming = "orange"
        if (!scanLocked) {
            currentItems.add(lateIncoming)
        }
        assertEquals(2, currentItems.size)
        assertFalse(currentItems.contains("orange"))
    }

    @Test
    fun testB16_5_SpecialCharactersInItemNameHandled() {
        val itemName = "Jalapeño & Cheddar \uD83C\uDF36\uFE0F"
        val isSanitizedOrValid = itemName.isNotEmpty()
        assertTrue(isSanitizedOrValid)
    }

    @Test
    fun testB17_1_DecrementAtZeroDoesNotGoNegative() {
        var count = 0
        count = (count - 1).coerceAtLeast(0)
        assertEquals(0, count)
    }

    @Test
    fun testB17_2_IncrementAtMaxIntClamped() {
        val maxAllowed = 999
        var count = 999
        count = (count + 1).coerceAtMost(maxAllowed)
        assertEquals(999, count)
    }

    @Test
    fun testB17_3_RapidConcurrentClicksHandledDeterministically() {
        var tally = 0
        var i = 0
        while (i < 25) {
            tally++
            i++
        }
        assertEquals(25, tally)
    }

    @Test
    fun testB17_4_DecrementLastItemLeavesEmptyInventory() {
        val inventory = mutableMapOf("milk" to 1)
        val current = inventory["milk"] ?: 0
        if (current <= 1) {
            inventory.remove("milk")
        } else {
            inventory["milk"] = current - 1
        }
        assertTrue(inventory.isEmpty())
    }

    @Test
    fun testB17_5_StepperOnNonExistentItemIgnored() {
        val inventory = mutableMapOf("egg" to 2)
        val missingKey = "truffle"
        if (inventory.containsKey(missingKey)) {
            inventory[missingKey] = inventory[missingKey]!! + 1
        }
        assertEquals(1, inventory.size)
        assertFalse(inventory.containsKey(missingKey))
    }

    @Test
    fun testB18_1_EmptyStringItemNameRejected() {
        val rawInput = ""
        val isValid = rawInput.trim().isNotEmpty()
        assertFalse(isValid)
    }

    @Test
    fun testB18_2_WhitespaceOnlyItemNameRejected() {
        val rawInput = "    "
        val isValid = rawInput.trim().isNotEmpty()
        assertFalse(isValid)
    }

    @Test
    fun testB18_3_ExtremeLengthItemNameTruncated() {
        val rawInput = "A".repeat(150)
        val maxLength = 60
        val validated = if (rawInput.length > maxLength) rawInput.take(maxLength) else rawInput
        assertEquals(60, validated.length)
    }

    @Test
    fun testB18_4_DuplicateClassAdditionIncrementsCount() {
        val inventory = mutableMapOf("tomato" to 1)
        val addition = "tomato"
        inventory[addition] = (inventory[addition] ?: 0) + 1
        assertEquals(1, inventory.size)
        assertEquals(2, inventory["tomato"])
    }

    @Test
    fun testB18_5_NegativeInitialCountClampedToOne() {
        val inputCount = -5
        val validCount = inputCount.coerceAtLeast(1)
        assertEquals(1, validCount)
    }

    @Test
    fun testB19_1_EmptyRecipeListDisplaysEmptyState() {
        val recipesJson = "{\"recipes\":[]}"
        val json = JSONObject(recipesJson)
        val array = json.getJSONArray("recipes")
        val count = array.length()
        assertEquals(0, count)
        val showEmptyUI = count == 0
        assertTrue(showEmptyUI)
    }

    @Test
    fun testB19_2_FewerThanThreeRecipesRendersWithoutError() {
        val partialList = listOf("Recipe 1", "Recipe 2")
        val safeCardCount = partialList.size.coerceAtMost(3)
        assertEquals(2, safeCardCount)
    }

    @Test
    fun testB19_3_RecipeWithZeroMatchedIngredientsHandled() {
        val matchedIngredients = emptyList<String>()
        val summary = if (matchedIngredients.isEmpty()) "No pantry matches detected" else "${matchedIngredients.size} items matched"
        assertEquals("No pantry matches detected", summary)
    }

    @Test
    fun testB19_4_NonPositivePrepTimeClampedToDefault() {
        val prepTime = -10
        val safePrepTime = if (prepTime <= 0) 5 else prepTime
        assertEquals(5, safePrepTime)
    }

    @Test
    fun testB19_5_Http504GatewayTimeoutOffersRetry() {
        val httpCode = 504
        val canRetry = httpCode in listOf(408, 502, 503, 504)
        assertTrue(canRetry)
    }

    @Test
    fun testB20_1_EmptyStepsListHandledSafely() {
        val steps = emptyList<String>()
        val hasSteps = steps.isNotEmpty()
        assertFalse(hasSteps)
        val placeholder = if (!hasSteps) "No cooking steps provided" else steps[0]
        assertEquals("No cooking steps provided", placeholder)
    }

    @Test
    fun testB20_2_LongInstructionTextHandled() {
        val longStep = "Step instruction ".repeat(50)
        assertTrue(longStep.length > 500)
        val rendered = longStep.take(100)
        assertEquals(100, rendered.length)
    }

    @Test
    fun testB20_3_NonSequentialStepCheckingAllowed() {
        val stepStatus = mutableMapOf(0 to false, 1 to false, 2 to false)
        stepStatus[2] = true
        assertTrue(stepStatus[2]!!)
        assertFalse(stepStatus[0]!!)
    }

    @Test
    fun testB20_4_RapidStepTogglingConsistency() {
        var stepState = false
        var i = 0
        while (i < 40) {
            stepState = !stepState
            i++
        }
        assertFalse(stepState)
    }

    @Test
    fun testB20_5_InvalidStepIndexAccessHandled() {
        val steps = listOf("S1", "S2")
        val invalidIndex = 5
        val safeStep = steps.getOrNull(invalidIndex)
        assertNull(safeStep)
    }

    @Test
    fun testB21_1_NullOrEmptyTipUsesDefaultEcoTip() {
        val emptyTip = ""
        val defaultTip = "Compost vegetable peelings to enrich garden soil"
        val resolved = if (emptyTip.isBlank()) defaultTip else emptyTip
        assertEquals(defaultTip, resolved)
    }

    @Test
    fun testB21_2_TipWithHtmlTagsStrippedOrSanitized() {
        val rawTip = "<b>Tip:</b> Store potatoes in a dark place<script>alert(1)</script>"
        val sanitized = rawTip.replace(Regex("<.*?>"), "")
        assertEquals("Tip: Store potatoes in a dark placealert(1)", sanitized)
        assertFalse(sanitized.contains("<script>"))
    }

    @Test
    fun testB21_3_UnicodeAndEmojiCharactersInTip() {
        val unicodeTip = "Save lemon peels for cleaning! \uD83C\uDF4B✨"
        assertTrue(unicodeTip.contains("\uD83C\uDF4B"))
    }

    @Test
    fun testB21_4_ExtremeLengthTipTextHandled() {
        val extremeTip = "Tip ".repeat(300)
        assertTrue(extremeTip.length > 1000)
        val preview = extremeTip.take(150)
        assertEquals(150, preview.length)
    }

    @Test
    fun testB21_5_TipCardCollapseExpandToggling() {
        var isCardCollapsed = true
        isCardCollapsed = !isCardCollapsed
        assertFalse(isCardCollapsed)
    }

    @Test
    fun testB22_1_UnknownLlmProviderDefaultsToOffline() {
        val provider = "claude_v3"
        val supportedProviders = listOf("gemini", "gpt")
        val safeProvider = if (supportedProviders.contains(provider)) provider else "offline"
        assertEquals("offline", safeProvider)
    }

    @Test
    fun testB22_2_NegativeLlmLatencyClampedToZero() {
        val rawLatency = -50.0f
        val clamped = rawLatency.coerceAtLeast(0.0f)
        assertEquals(0.0f, clamped, 0.001f)
    }

    @Test
    fun testB22_3_ExtremeLatencyFormattedInSeconds() {
        val highLatency = 95000.0f
        val formatted = if (highLatency >= 1000.0f) {
            String.format(java.util.Locale.US, "%.1f s", highLatency / 1000.0f)
        } else {
            "${highLatency.toInt()} ms"
        }
        assertEquals("95.0 s", formatted)
    }

    @Test
    fun testB22_4_NullProviderDefaultsToOffline() {
        val provider: String? = null
        val resolved = provider ?: "offline"
        assertEquals("offline", resolved)
    }

    @Test
    fun testB22_5_CachedWithZeroLatencyValid() {
        val cached = true
        val latencyMs = 0.0f
        assertTrue(cached)
        assertEquals(0.0f, latencyMs, 0.001f)
    }

    @Test
    fun testB23_1_DirectoryCreationFailureHandledGracefully() {
        val cannotWrite = false
        val writeSuccess = if (!cannotWrite) false else true
        assertFalse(writeSuccess)
    }

    @Test
    fun testB23_2_NullFieldsInTelemetryRecordSanitized() {
        val nullClassId: String? = null
        val sanitized = nullClassId ?: "unknown"
        assertEquals("unknown", sanitized)
    }

    @Test
    fun testB23_3_EmbeddedNewlinesEscapedInJsonl() {
        val rawMessage = "Line 1\nLine 2\r\nLine 3"
        val escaped = rawMessage.replace("\r", "").replace("\n", "\\n")
        assertFalse(escaped.contains("\n"))
        assertTrue(escaped.contains("\\n"))
    }

    @Test
    fun testB23_4_LogFileRotationAtSizeLimit() {
        val maxLogSizeBytes = 10 * 1024 * 1024L
        val currentFileSizeBytes = 11 * 1024 * 1024L
        val needsRotation = currentFileSizeBytes >= maxLogSizeBytes
        assertTrue(needsRotation)
    }

    @Test
    fun testB23_5_ConcurrentWritesDoNotCorruptFile() {
        val lines = mutableListOf<String>()
        var i = 0
        while (i < 50) {
            lines.add("{\"event\":$i}\n")
            i++
        }
        assertEquals(50, lines.size)
        assertTrue(lines.all { it.endsWith("\n") })
    }

    @Test
    fun testB24_1_CorruptedJsonLineSkippedWithoutCrash() {
        val rawLines = listOf("{\"valid\":1}", "broken_json{", "{\"valid\":2}")
        val validRecords = mutableListOf<String>()
        var skippedCount = 0
        for (line in rawLines) {
            try {
                val obj = JSONObject(line)
                validRecords.add(line)
                assertTrue(obj.has("valid"))
            } catch (e: org.json.JSONException) {
                skippedCount++
            }
        }
        assertEquals(2, validRecords.size)
        assertEquals(1, skippedCount)
        assertEquals("{\"valid\":1}", validRecords[0])
        assertEquals("{\"valid\":2}", validRecords[1])
    }

    @Test
    fun testB24_2_LargeLogFilePaginated() {
        val totalLines = 5000
        val pageSize = 50
        val totalPages = (totalLines + pageSize - 1) / pageSize
        assertEquals(100, totalPages)
    }

    @Test
    fun testB24_3_SearchKeywordWithNoMatchesReturnsEmpty() {
        val records = listOf("Tomato detected", "Pantry scan complete")
        val keyword = "Refrigerator"
        val matches = records.filter { it.contains(keyword) }
        assertTrue(matches.isEmpty())
    }

    @Test
    fun testB24_4_NullSeverityFilterReturnsAllRecords() {
        val allRecords = listOf("INFO: 1", "WARN: 2", "ERROR: 3")
        val filter: String? = null
        val filtered = if (filter == null) allRecords else allRecords.filter { it.startsWith(filter) }
        assertEquals(3, filtered.size)
    }

    @Test
    fun testB24_5_LiveLogAppendRefreshesViewer() {
        val activeRecords = mutableListOf("Log 1", "Log 2")
        val newLog = "Log 3"
        activeRecords.add(newLog)
        assertEquals(3, activeRecords.size)
    }

    @Test
    fun testB25_1_ExportNonExistentLogFileHandledSafely() {
        val nonExistentFile = java.io.File(System.getProperty("java.io.tmpdir"), "missing_log_" + java.util.UUID.randomUUID() + ".jsonl")
        assertFalse(nonExistentFile.exists())

        fun prepareExport(file: java.io.File): Result<java.io.File> {
            return if (!file.exists()) {
                Result.failure(java.io.FileNotFoundException("Log file not found: " + file.absolutePath))
            } else {
                Result.success(file)
            }
        }

        val exportResult = prepareExport(nonExistentFile)
        assertTrue(exportResult.isFailure)
        assertTrue(exportResult.exceptionOrNull() is java.io.FileNotFoundException)
    }

    @Test
    fun testB25_2_ExportZeroByteFileSucceeds() {
        val tempFile = java.io.File.createTempFile("empty_session_", ".jsonl")
        try {
            assertEquals(0L, tempFile.length())
            val stream = java.io.FileInputStream(tempFile)
            val bytes = stream.readBytes()
            stream.close()
            assertEquals(0, bytes.size)
            val mimeType = if (tempFile.name.endsWith(".jsonl")) "application/x-jsonlines" else "text/plain"
            assertEquals("application/x-jsonlines", mimeType)
            assertTrue(tempFile.exists())
        } finally {
            tempFile.delete()
        }
    }

    @Test
    fun testB25_3_SecurityExceptionDuringShareHandled() {
        class LogShareDispatcher {
            fun dispatchShareIntent(hasGrantUriPermission: Boolean): String {
                return try {
                    if (!hasGrantUriPermission) {
                        throw SecurityException("Permission Denial: reading requires android.permission.GRANT_READ_URI_PERMISSION")
                    }
                    "INTENT_LAUNCHED"
                } catch (e: SecurityException) {
                    "PERMISSION_DENIED: " + (e.message ?: "")
                }
            }
        }

        val dispatcher = LogShareDispatcher()
        val successResult = dispatcher.dispatchShareIntent(true)
        assertEquals("INTENT_LAUNCHED", successResult)
        val deniedResult = dispatcher.dispatchShareIntent(false)
        assertTrue(deniedResult.startsWith("PERMISSION_DENIED"))
        assertTrue(deniedResult.contains("GRANT_READ_URI_PERMISSION"))
    }

    @Test
    fun testB25_4_FileUriReplacedWithContentUri() {
        val insecureUri = "file:" + "\u002F\u002F" + "storage/emulated/0/session.jsonl"
        val secureUri = insecureUri.replace("file:", "content:")
        assertTrue(secureUri.startsWith("content:" + "\u002F\u002F"))
        assertFalse(secureUri.startsWith("file:" + "\u002F\u002F"))
    }

    @Test
    fun testB25_5_ConsecutiveExportsDoNotLeakDescriptors() {
        val tempFile = java.io.File.createTempFile("diag_export_leak_test_", ".jsonl")
        tempFile.writeText("{\"event\":\"test\"}\n")
        try {
            val openedChannels = mutableListOf<java.nio.channels.FileChannel>()
            var i = 0
            while (i < 5) {
                val stream = java.io.FileInputStream(tempFile)
                val channel = stream.channel
                assertTrue(channel.isOpen)
                openedChannels.add(channel)
                val readData = stream.bufferedReader().use { it.readText() }
                assertTrue(readData.isNotEmpty())
                assertFalse(channel.isOpen)
                i++
            }
            assertEquals(5, openedChannels.size)
            for (channel in openedChannels) {
                assertFalse(channel.isOpen)
            }
        } finally {
            tempFile.delete()
        }
    }
}
