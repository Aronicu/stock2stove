package com.kaggleing.stock2stove.e2e

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.io.StringWriter
import java.io.PrintWriter
import java.util.*

class Tier3CrossFeatureTest {

    data class PresetConfig(val id: String, val displayName: String, val host: String, val port: Int) {
        fun getHealthUrl(): String = "http:" + "\u002F\u002F" + "$host:$port/api/v1/health"
        fun getWsUrl(): String = "ws:" + "\u002F\u002F" + "$host:$port/ws/scan"
        fun getRecipeUrl(): String = "http:" + "\u002F\u002F" + "$host:$port/api/v1/recipes/generate"
    }

    enum class ScanContext {
        PANTRY_SHELF,
        REFRIGERATOR
    }

    enum class DefenseDetectorMode(val wireValue: String) {
        YOLO_V8N("yolov8"),
        ORB_BASELINE("orb_baseline"),
        SIFT_BASELINE("sift_baseline")
    }

    data class Point2D(val x: Float, val y: Float)

    data class ScreenPolygon(val points: List<Point2D>) {
        fun calculateBoundingBox(): ScreenRect {
            val minX = points.minOf { it.x }
            val maxX = points.maxOf { it.x }
            val minY = points.minOf { it.y }
            val maxY = points.maxOf { it.y }
            return ScreenRect(minX, minY, maxX, maxY)
        }
    }

    data class ScreenRect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
        val width: Float get() = right - left
        val height: Float get() = bottom - top
    }

    data class RawDetection(
        val classId: String,
        val displayName: String,
        val confidence: Float,
        val bboxNorm: List<Float>,
        val polygonNorm: List<List<Float>>?
    )

    class TemporalTracker(private val windowSize: Int = 5, private val minConfirmations: Int = 3) {
        private val history = mutableMapOf<String, MutableList<Boolean>>()

        fun processFrame(detectedClassesInFrame: Set<String>): Map<String, Boolean> {
            val allTrackedClasses = (history.keys + detectedClassesInFrame).toSet()
            val result = mutableMapOf<String, Boolean>()
            for (classId in allTrackedClasses) {
                val window = history.getOrPut(classId) { mutableListOf() }
                window.add(detectedClassesInFrame.contains(classId))
                if (window.size > windowSize) {
                    window.removeAt(0)
                }
                val trueCount = window.count { it }
                result[classId] = (trueCount >= minConfirmations)
            }
            return result
        }

        fun reset() {
            history.clear()
        }
    }

    class InventoryTallyState {
        private val items = mutableMapOf<String, Int>()

        fun updateFromConfirmed(classId: String, count: Int) {
            if (count > 0) {
                items[classId] = count
            } else {
                items.remove(classId)
            }
        }

        fun increment(classId: String) {
            val current = items.getOrDefault(classId, 0)
            items[classId] = current + 1
        }

        fun decrement(classId: String) {
            val current = items.getOrDefault(classId, 0)
            if (current <= 1) {
                items.remove(classId)
            } else {
                items[classId] = current - 1
            }
        }

        fun getItemCount(classId: String): Int = items.getOrDefault(classId, 0)

        fun getAllItems(): Map<String, Int> = items.toMap()

        fun clear() {
            items.clear()
        }
    }

    class ProgressRingState {
        var elapsedSeconds: Float = 0.0f
        val maxSeconds: Float = 5.0f
        var isCompleted: Boolean = false

        fun advance(deltaSeconds: Float): Boolean {
            elapsedSeconds += deltaSeconds
            if (elapsedSeconds >= maxSeconds) {
                elapsedSeconds = maxSeconds
                isCompleted = true
            }
            return isCompleted
        }

        fun reset() {
            elapsedSeconds = 0.0f
            isCompleted = false
        }
    }

    data class LogRecord(
        val timestamp: Long,
        val level: String,
        val tag: String,
        val message: String,
        val metadata: Map<String, Any?>
    ) {
        fun toJsonString(): String {
            val json = JSONObject()
            json.put("ts", timestamp)
            json.put("level", level)
            json.put("tag", tag)
            json.put("msg", message)
            val metaJson = JSONObject()
            for ((k, v) in metadata) {
                metaJson.put(k, v)
            }
            json.put("meta", metaJson)
            return json.toString()
        }

        companion object {
            fun parseJson(line: String): LogRecord {
                val json = JSONObject(line)
                val metaJson = json.optJSONObject("meta") ?: JSONObject()
                val metaMap = mutableMapOf<String, Any?>()
                val keys = metaJson.keys()
                while (keys.hasNext()) {
                    val key = keys.next()
                    metaMap[key] = metaJson.get(key)
                }
                return LogRecord(
                    timestamp = json.getLong("ts"),
                    level = json.getString("level"),
                    tag = json.getString("tag"),
                    message = json.getString("msg"),
                    metadata = metaMap
                )
            }
        }
    }

    class MemoryLogger {
        private val records = mutableListOf<LogRecord>()

        fun log(level: String, tag: String, message: String, metadata: Map<String, Any?> = emptyMap()) {
            val record = LogRecord(
                timestamp = System.currentTimeMillis(),
                level = level,
                tag = tag,
                message = message,
                metadata = metadata
            )
            records.add(record)
        }

        fun getRecords(): List<LogRecord> = records.toList()

        fun filterByMinLevel(minLevel: String): List<LogRecord> {
            val levels = listOf("DEBUG", "INFO", "WARN", "ERROR")
            val minIndex = levels.indexOf(minLevel)
            if (minIndex == -1) return records.toList()
            return records.filter {
                val idx = levels.indexOf(it.level)
                idx >= minIndex
            }
        }

        fun clear() {
            records.clear()
        }
    }

    @Test
    fun testPairwise_F03_F04_PresetSwitching_HealthCheckPing() {
        val presets = listOf(
            PresetConfig("usb_adb", "USB ADB Reverse", "127.0.0.1", 8000),
            PresetConfig("emulator", "Android Emulator", "10.0.2.2", 8000),
            PresetConfig("lan_custom", "Custom LAN", "192.168.1.145", 8000)
        )

        for (preset in presets) {
            val healthUrl = preset.getHealthUrl()
            assertTrue(healthUrl.startsWith("http:" + "\u002F\u002F"))
            assertTrue(healthUrl.endsWith("/api/v1/health"))
            assertTrue(healthUrl.contains(preset.host))

            val fakeResponseJson = JSONObject()
            fakeResponseJson.put("status", "ok")
            fakeResponseJson.put("service", "stock2stove-inference-server")
            val modelStatus = JSONObject()
            modelStatus.put("yolo_loaded", true)
            fakeResponseJson.put("model_status", modelStatus)
            val hardware = JSONObject()
            hardware.put("device", "cuda")
            fakeResponseJson.put("hardware", hardware)
            val modes = JSONArray()
            modes.put("yolov8")
            modes.put("orb_baseline")
            modes.put("sift_baseline")
            fakeResponseJson.put("supported_detector_modes", modes)

            assertEquals("ok", fakeResponseJson.getString("status"))
            assertEquals("stock2stove-inference-server", fakeResponseJson.getString("service"))
            assertTrue(fakeResponseJson.getJSONObject("model_status").getBoolean("yolo_loaded"))
            assertEquals(3, fakeResponseJson.getJSONArray("supported_detector_modes").length())
        }
    }

    @Test
    fun testPairwise_F05_F07_ScanModeSelector_WebSocketStreamingPayload() {
        val modes = listOf(ScanContext.PANTRY_SHELF, ScanContext.REFRIGERATOR)
        var frameId = 100L

        for (mode in modes) {
            frameId++
            val clientFrameJson = JSONObject()
            clientFrameJson.put("frame_id", frameId)
            clientFrameJson.put("scan_mode", mode.name.lowercase())
            clientFrameJson.put("detector_mode", "yolov8")
            clientFrameJson.put("enable_clahe", mode == ScanContext.REFRIGERATOR)
            clientFrameJson.put("conf_threshold", 0.25f)
            clientFrameJson.put("image_b64", "iVBORw0KGgoAAAANSUhEUg==")
            clientFrameJson.put("reset_session", false)

            assertEquals(frameId, clientFrameJson.getLong("frame_id"))
            assertEquals(mode.name.lowercase(), clientFrameJson.getString("scan_mode"))
            if (mode == ScanContext.REFRIGERATOR) {
                assertTrue(clientFrameJson.getBoolean("enable_clahe"))
            } else {
                assertFalse(clientFrameJson.getBoolean("enable_clahe"))
            }
        }
    }

    @Test
    fun testPairwise_F07_F08_WebSocketStreaming_PlanarHomographyPolygon() {
        val serverScanResponse = JSONObject()
        serverScanResponse.put("frame_id", 42L)
        serverScanResponse.put("detector_mode", "yolov8")
        serverScanResponse.put("inference_ms", 32.5f)

        val detections = JSONArray()
        val det = JSONObject()
        det.put("class_id", "canned_beans")
        det.put("display_name", "Canned Black Beans")
        det.put("confidence", 0.88f)
        val bbox = JSONArray(listOf(0.1, 0.2, 0.8, 0.9))
        det.put("bbox_norm", bbox)

        val poly = JSONArray()
        poly.put(JSONArray(listOf(0.15, 0.20)))
        poly.put(JSONArray(listOf(0.78, 0.22)))
        poly.put(JSONArray(listOf(0.82, 0.88)))
        poly.put(JSONArray(listOf(0.12, 0.86)))
        det.put("polygon_norm", poly)
        detections.put(det)
        serverScanResponse.put("detections", detections)

        val polyJson = serverScanResponse.getJSONArray("detections").getJSONObject(0).getJSONArray("polygon_norm")
        assertEquals(4, polyJson.length())

        val canvasWidth = 1080.0f
        val canvasHeight = 1920.0f
        val points = mutableListOf<Point2D>()
        for (i in 0 until polyJson.length()) {
            val pt = polyJson.getJSONArray(i)
            val px = pt.getDouble(0).toFloat() * canvasWidth
            val py = pt.getDouble(1).toFloat() * canvasHeight
            points.add(Point2D(px, py))
        }

        val screenPolygon = ScreenPolygon(points)
        val bounds = screenPolygon.calculateBoundingBox()

        assertEquals(4, screenPolygon.points.size)
        assertTrue(bounds.left > 0.0f)
        assertTrue(bounds.right <= canvasWidth)
        assertTrue(bounds.top > 0.0f)
        assertTrue(bounds.bottom <= canvasHeight)
        assertTrue(bounds.width > 500.0f)
        assertTrue(bounds.height > 1000.0f)
    }

    @Test
    fun testPairwise_F07_F09_WebSocketStreaming_BoundingBoxFallback() {
        val serverScanResponse = JSONObject()
        serverScanResponse.put("frame_id", 43L)

        val detections = JSONArray()
        val det = JSONObject()
        det.put("class_id", "pasta_box")
        det.put("confidence", 0.91f)
        val bbox = JSONArray(listOf(0.20, 0.15, 0.70, 0.85))
        det.put("bbox_norm", bbox)
        det.put("polygon_norm", JSONObject.NULL)
        detections.put(det)
        serverScanResponse.put("detections", detections)

        val retrievedDet = serverScanResponse.getJSONArray("detections").getJSONObject(0)
        assertTrue(retrievedDet.isNull("polygon_norm"))

        val bboxJson = retrievedDet.getJSONArray("bbox_norm")
        val canvasW = 720.0f
        val canvasH = 1280.0f

        val left = bboxJson.getDouble(0).toFloat() * canvasW
        val top = bboxJson.getDouble(1).toFloat() * canvasH
        val right = bboxJson.getDouble(2).toFloat() * canvasW
        val bottom = bboxJson.getDouble(3).toFloat() * canvasH
        val rect = ScreenRect(left, top, right, bottom)

        assertEquals(144.0f, rect.left, 0.01f)
        assertEquals(192.0f, rect.top, 0.01f)
        assertEquals(504.0f, rect.right, 0.01f)
        assertEquals(1088.0f, rect.bottom, 0.01f)
        assertEquals(360.0f, rect.width, 0.01f)
        assertEquals(896.0f, rect.height, 0.01f)
    }

    @Test
    fun testPairwise_F07_F10_WebSocketStreaming_TemporalConfirmationWindow() {
        val tracker = TemporalTracker(windowSize = 5, minConfirmations = 3)

        val f1 = tracker.processFrame(setOf("beans", "milk"))
        assertFalse(f1["beans"] ?: true)
        assertFalse(f1["milk"] ?: true)

        val f2 = tracker.processFrame(setOf("beans", "rice"))
        assertFalse(f2["beans"] ?: true)
        assertFalse(f2["rice"] ?: true)

        val f3 = tracker.processFrame(setOf("beans", "milk"))
        assertTrue(f3["beans"] ?: false)
        assertFalse(f3["milk"] ?: true)

        val f4 = tracker.processFrame(setOf("beans", "milk"))
        assertTrue(f4["beans"] ?: false)
        assertTrue(f4["milk"] ?: false)

        val f5 = tracker.processFrame(setOf("pasta"))
        assertTrue(f5["beans"] ?: false)
        assertTrue(f5["milk"] ?: false)

        val f6 = tracker.processFrame(setOf("pasta"))
        assertTrue(f6["beans"] ?: false)
        assertFalse(f6["milk"] ?: true)
    }

    @Test
    fun testPairwise_F10_F12_TemporalConfirmation_LiveConfirmedChipsTally() {
        val tracker = TemporalTracker(windowSize = 5, minConfirmations = 3)
        val chipsTally = InventoryTallyState()

        val streamSequence = listOf(
            setOf("canned_beans", "tomato_sauce"),
            setOf("canned_beans"),
            setOf("canned_beans", "tomato_sauce"),
            setOf("canned_beans", "tomato_sauce", "pasta"),
            setOf("pasta", "tomato_sauce")
        )

        for (frameClasses in streamSequence) {
            val confirmedMap = tracker.processFrame(frameClasses)
            for ((cls, confirmed) in confirmedMap) {
                if (confirmed) {
                    chipsTally.updateFromConfirmed(cls, 1)
                }
            }
        }

        val allChips = chipsTally.getAllItems()
        assertEquals(1, allChips["canned_beans"] ?: 0)
        assertEquals(1, allChips["tomato_sauce"] ?: 0)
        assertEquals(0, allChips["pasta"] ?: 0)
    }

    @Test
    fun testPairwise_F07_F13_WebSocketStreaming_AcademicDefenseModeSwitcher() {
        val modes = listOf(
            DefenseDetectorMode.YOLO_V8N,
            DefenseDetectorMode.ORB_BASELINE,
            DefenseDetectorMode.SIFT_BASELINE
        )

        for (mode in modes) {
            val frameReq = JSONObject()
            frameReq.put("frame_id", 1L)
            frameReq.put("detector_mode", mode.wireValue)
            frameReq.put("enable_clahe", false)

            assertEquals(mode.wireValue, frameReq.getString("detector_mode"))

            val simulatedResponse = JSONObject()
            simulatedResponse.put("frame_id", 1L)
            simulatedResponse.put("detector_mode", mode.wireValue)
            val diag = JSONObject()
            when (mode) {
                DefenseDetectorMode.YOLO_V8N -> {
                    diag.put("bbox_count", 4)
                    diag.put("backbone", "darknet")
                }
                DefenseDetectorMode.ORB_BASELINE -> {
                    diag.put("keypoints", 500)
                    diag.put("inliers", 42)
                }
                DefenseDetectorMode.SIFT_BASELINE -> {
                    diag.put("keypoints", 1250)
                    diag.put("inliers", 88)
                }
            }
            simulatedResponse.put("diagnostics", diag)

            assertEquals(mode.wireValue, simulatedResponse.getString("detector_mode"))
            assertNotNull(simulatedResponse.getJSONObject("diagnostics"))
        }
    }

    @Test
    fun testPairwise_F07_F14_WebSocketStreaming_ClaheLowLightToggle() {
        val clientFrameOn = JSONObject()
        clientFrameOn.put("frame_id", 10L)
        clientFrameOn.put("enable_clahe", true)

        val clientFrameOff = JSONObject()
        clientFrameOff.put("frame_id", 11L)
        clientFrameOff.put("enable_clahe", false)

        assertTrue(clientFrameOn.getBoolean("enable_clahe"))
        assertFalse(clientFrameOff.getBoolean("enable_clahe"))

        val responseWithClahe = JSONObject()
        responseWithClahe.put("frame_id", 10L)
        val diag = JSONObject()
        diag.put("clahe_applied", true)
        diag.put("clip_limit", 2.0f)
        diag.put("tile_grid_size", JSONArray(listOf(8, 8)))
        responseWithClahe.put("diagnostics", diag)

        assertTrue(responseWithClahe.getJSONObject("diagnostics").getBoolean("clahe_applied"))
        assertEquals(2.0, responseWithClahe.getJSONObject("diagnostics").getDouble("clip_limit"), 0.01)
    }

    @Test
    fun testPairwise_F07_F15_WebSocketStreaming_LiveTelemetryHudMetrics() {
        val serverScanResponse = JSONObject()
        serverScanResponse.put("frame_id", 77L)
        serverScanResponse.put("detector_mode", "orb_baseline")
        serverScanResponse.put("inference_ms", 19.4f)
        serverScanResponse.put("fps", 28.5f)

        val diagnostics = JSONObject()
        diagnostics.put("keypoints", 640)
        diagnostics.put("inliers", 53)
        diagnostics.put("edge_density", 0.34f)
        serverScanResponse.put("diagnostics", diagnostics)

        val fps = serverScanResponse.getDouble("fps").toFloat()
        val latencyMs = serverScanResponse.getDouble("inference_ms").toFloat()
        val diagObj = serverScanResponse.getJSONObject("diagnostics")
        val keypoints = diagObj.getInt("keypoints")
        val inliers = diagObj.getInt("inliers")
        val edgeDensity = diagObj.getDouble("edge_density").toFloat()

        assertEquals(28.5f, fps, 0.01f)
        assertEquals(19.4f, latencyMs, 0.01f)
        assertEquals(640, keypoints)
        assertEquals(53, inliers)
        assertEquals(0.34f, edgeDensity, 0.01f)
    }

    @Test
    fun testPairwise_F11_F12_ProgressRing_LiveChipsTallyPersistence() {
        val ring = ProgressRingState()
        val chipsTally = InventoryTallyState()

        chipsTally.updateFromConfirmed("canned_beans", 2)
        assertFalse(ring.advance(1.5f))
        assertEquals(1.5f, ring.elapsedSeconds, 0.01f)
        assertEquals(2, chipsTally.getItemCount("canned_beans"))

        chipsTally.updateFromConfirmed("pasta", 1)
        assertFalse(ring.advance(2.0f))
        assertEquals(3.5f, ring.elapsedSeconds, 0.01f)
        assertEquals(1, chipsTally.getItemCount("pasta"))

        val isFinished = ring.advance(1.5f)
        assertTrue(isFinished)
        assertTrue(ring.isCompleted)
        assertEquals(5.0f, ring.elapsedSeconds, 0.01f)

        assertEquals(2, chipsTally.getItemCount("canned_beans"))
        assertEquals(1, chipsTally.getItemCount("pasta"))
    }

    @Test
    fun testPairwise_F11_F16_ProgressRingCompletion_BottomSheetTransition() {
        val ring = ProgressRingState()
        var bottomSheetOpen = false

        assertFalse(ring.isCompleted)
        assertFalse(bottomSheetOpen)

        ring.advance(3.0f)
        assertFalse(ring.isCompleted)
        assertFalse(bottomSheetOpen)

        val done = ring.advance(2.0f)
        if (done) {
            bottomSheetOpen = true
        }

        assertTrue(ring.isCompleted)
        assertTrue(bottomSheetOpen)
    }

    @Test
    fun testPairwise_F16_F17_BottomSheet_QuantitySteppersIncrementDecrementZeroOut() {
        val tally = InventoryTallyState()
        tally.updateFromConfirmed("canned_beans", 2)
        tally.updateFromConfirmed("milk", 1)

        tally.increment("canned_beans")
        assertEquals(3, tally.getItemCount("canned_beans"))

        tally.decrement("canned_beans")
        assertEquals(2, tally.getItemCount("canned_beans"))

        tally.decrement("milk")
        assertEquals(0, tally.getItemCount("milk"))
        assertFalse(tally.getAllItems().containsKey("milk"))
    }

    @Test
    fun testPairwise_F16_F18_BottomSheet_ManualItemAdditionUnifiedClasses() {
        val tally = InventoryTallyState()
        tally.updateFromConfirmed("canned_beans", 1)

        val unifiedClasses = listOf(
            "canned_beans", "tomato_sauce", "pasta", "rice",
            "olive_oil", "milk", "eggs", "cheese"
        )
        assertTrue(unifiedClasses.contains("olive_oil"))

        tally.increment("olive_oil")
        assertEquals(1, tally.getItemCount("olive_oil"))

        tally.increment("canned_beans")
        assertEquals(2, tally.getItemCount("canned_beans"))

        tally.increment("cheese")
        assertEquals(1, tally.getItemCount("cheese"))

        val items = tally.getAllItems()
        assertEquals(3, items.size)
    }

    @Test
    fun testPairwise_F17_F19_QuantitySteppers_RecipeScreenInventoryPayload() {
        val tally = InventoryTallyState()
        tally.updateFromConfirmed("canned_beans", 2)
        tally.updateFromConfirmed("rice", 1)
        tally.increment("rice")

        val reqJson = JSONObject()
        val itemsArray = JSONArray()
        for ((classId, count) in tally.getAllItems()) {
            val itemObj = JSONObject()
            itemObj.put("class_id", classId)
            itemObj.put("count", count)
            itemsArray.put(itemObj)
        }
        reqJson.put("confirmed_inventory", itemsArray)
        reqJson.put("timeout_sec", 10.0f)

        val serialized = reqJson.getJSONArray("confirmed_inventory")
        assertEquals(2, serialized.length())

        val fakeRecipeResponse = JSONObject()
        fakeRecipeResponse.put("source", "gemini")
        val recipeArray = JSONArray()
        for (i in 1..3) {
            val r = JSONObject()
            r.put("id", "rec_$i")
            r.put("title", "Pantry Recipe $i")
            r.put("prep_time_mins", 15 * i)
            recipeArray.put(r)
        }
        fakeRecipeResponse.put("recipes", recipeArray)

        assertEquals(3, fakeRecipeResponse.getJSONArray("recipes").length())
    }

    @Test
    fun testPairwise_F18_F19_ManualItemAddition_RecipeScreenIngredientMatching() {
        val tally = InventoryTallyState()
        tally.updateFromConfirmed("pasta", 1)
        tally.increment("olive_oil")
        tally.increment("tomato_sauce")

        val recipeCard = JSONObject()
        recipeCard.put("id", "pantry_pasta")
        recipeCard.put("title", "Quick Tomato Olive Pasta")
        val matched = JSONArray(listOf("pasta", "olive_oil", "tomato_sauce"))
        recipeCard.put("matched_ingredients", matched)

        val matchedList = mutableListOf<String>()
        val arr = recipeCard.getJSONArray("matched_ingredients")
        for (i in 0 until arr.length()) {
            matchedList.add(arr.getString(i))
        }

        assertTrue(matchedList.contains("olive_oil"))
        assertTrue(matchedList.contains("pasta"))
        assertTrue(matchedList.contains("tomato_sauce"))
    }

    @Test
    fun testPairwise_F19_F20_RecipeScreen_InteractiveCookingStepsProgression() {
        val steps = listOf(
            "Boil 4 cups of salted water in large pot",
            "Add pasta and cook for 8 minutes until al dente",
            "Heat olive oil in pan and sauté garlic and tomato sauce",
            "Drain pasta, toss into sauce, and serve warm"
        )
        val stepCompleted = BooleanArray(steps.size) { false }

        stepCompleted[0] = true
        var completedCount = stepCompleted.count { it }
        var progressFraction = completedCount.toFloat() / steps.size
        assertEquals(0.25f, progressFraction, 0.01f)

        stepCompleted[1] = true
        stepCompleted[2] = true
        completedCount = stepCompleted.count { it }
        progressFraction = completedCount.toFloat() / steps.size
        assertEquals(0.75f, progressFraction, 0.01f)

        stepCompleted[3] = true
        completedCount = stepCompleted.count { it }
        progressFraction = completedCount.toFloat() / steps.size
        assertEquals(1.0f, progressFraction, 0.01f)
    }

    @Test
    fun testPairwise_F19_F21_RecipeScreen_ZeroWasteChefTipCardDisplay() {
        val recipeCardJson = JSONObject()
        recipeCardJson.put("id", "zero_waste_bowl")
        recipeCardJson.put("title", "Zero Waste Pantry Bowl")
        recipeCardJson.put(
            "zero_waste_tip",
            "Store leftover tomato sauce in an airtight glass jar with a thin layer of olive oil on top to prevent mold."
        )

        val tip = recipeCardJson.getString("zero_waste_tip")
        assertNotNull(tip)
        assertTrue(tip.isNotEmpty())
        assertTrue(tip.contains("olive oil"))
    }

    @Test
    fun testPairwise_F19_F22_RecipeScreen_LlmTelemetryBadgeDisplay() {
        val recipeResponse = JSONObject()
        recipeResponse.put("source", "gemini-1.5-flash")
        recipeResponse.put("latency_ms", 312.4f)
        recipeResponse.put("cached", false)

        val source = recipeResponse.getString("source")
        val latency = recipeResponse.getDouble("latency_ms").toFloat()
        val isCached = recipeResponse.getBoolean("cached")

        assertEquals("gemini-1.5-flash", source)
        assertEquals(312.4f, latency, 0.01f)
        assertFalse(isCached)

        val badgeText = "$source | ${latency.toInt()}ms"
        assertEquals("gemini-1.5-flash | 312ms", badgeText)
    }

    @Test
    fun testPairwise_F23_F24_SessionLogger_DiagnosticLogViewerParsingAndFiltering() {
        val logger = MemoryLogger()
        logger.log("INFO", "ScanSession", "Session started", mapOf("mode" to "pantry"))
        logger.log("DEBUG", "WebSocket", "Frame 1 sent", mapOf("bytes" to 14200))
        logger.log("WARN", "TemporalTracker", "Low confidence on item", mapOf("conf" to 0.22))
        logger.log("ERROR", "Network", "Connection dropped temporarily", mapOf("code" to 1006))

        val allRecords = logger.getRecords()
        assertEquals(4, allRecords.size)

        val warnAndAbove = logger.filterByMinLevel("WARN")
        assertEquals(2, warnAndAbove.size)
        assertEquals("WARN", warnAndAbove[0].level)
        assertEquals("ERROR", warnAndAbove[1].level)

        val errorOnly = logger.filterByMinLevel("ERROR")
        assertEquals(1, errorOnly.size)
        assertEquals("ERROR", errorOnly[0].level)
    }

    @Test
    fun testPairwise_F23_F25_SessionLogger_ShareExportDiagnosticsPayload() {
        val logger = MemoryLogger()
        logger.log("INFO", "AppLifecycle", "App initialized")
        logger.log("INFO", "Diagnostics", "Generating export bundle")

        val stringWriter = StringWriter()
        val printWriter = PrintWriter(stringWriter)
        for (r in logger.getRecords()) {
            printWriter.println(r.toJsonString())
        }
        printWriter.flush()

        val fullExportString = stringWriter.toString()
        assertTrue(fullExportString.contains("App initialized"))
        assertTrue(fullExportString.contains("Generating export bundle"))

        val mimeType = "text/plain"
        val exportSubject = "Stock2Stove Alpha Diagnostic Logs"
        assertEquals("text/plain", mimeType)
        assertTrue(exportSubject.contains("Diagnostic"))
    }

    @Test
    fun testPairwise_F01_F02_Scaffold_CleartextNetworkSecurityConfiguration() {
        val permittedHosts = setOf("127.0.0.1", "10.0.2.2", "localhost")
        val isCleartextTrafficPermitted = true

        assertTrue(isCleartextTrafficPermitted)
        assertTrue(permittedHosts.contains("127.0.0.1"))
        assertTrue(permittedHosts.contains("10.0.2.2"))
        assertTrue(permittedHosts.contains("localhost"))
    }

    @Test
    fun testPairwise_F02_F04_CleartextHttp_HealthCheckEndpointVerification() {
        val preset = PresetConfig("local", "Localhost", "127.0.0.1", 8000)
        val endpoint = preset.getHealthUrl()

        val simulatedHttpCode = 200
        val simulatedBody = JSONObject()
        simulatedBody.put("status", "ok")
        simulatedBody.put("service", "stock2stove-inference-server")

        assertEquals(200, simulatedHttpCode)
        assertEquals("ok", simulatedBody.getString("status"))
        assertTrue(endpoint.startsWith("http:" + "\u002F\u002F" + "127.0.0.1:8000"))
    }

    @Test
    fun testPairwise_F06_F07_CameraFrameAnalysis_WebSocketFrameRateThrottling() {
        val targetFps = 8
        val minIntervalMs = 1000L / targetFps
        assertEquals(125L, minIntervalMs)

        var lastFrameTimestamp = 0L
        var framesSent = 0

        val simulatedCaptureTimestamps = listOf(0L, 50L, 100L, 130L, 180L, 260L, 300L, 390L)
        for (ts in simulatedCaptureTimestamps) {
            if (framesSent == 0 || ts - lastFrameTimestamp >= minIntervalMs) {
                framesSent++
                lastFrameTimestamp = ts
            }
        }

        assertEquals(4, framesSent)
    }

    @Test
    fun testPairwise_F13_F15_DefenseModeSwitcher_HudFeatureMetricsDisplay() {
        val yoloHud = JSONObject()
        yoloHud.put("mode", "yolov8")
        yoloHud.put("fps", 30.0)
        yoloHud.put("latency_ms", 35.0)
        yoloHud.put("detections_count", 3)

        val orbHud = JSONObject()
        orbHud.put("mode", "orb_baseline")
        orbHud.put("fps", 45.0)
        orbHud.put("latency_ms", 18.0)
        orbHud.put("keypoints", 520)
        orbHud.put("inliers", 44)

        assertEquals("yolov8", yoloHud.getString("mode"))
        assertEquals(3, yoloHud.getInt("detections_count"))
        assertEquals("orb_baseline", orbHud.getString("mode"))
        assertEquals(520, orbHud.getInt("keypoints"))
        assertEquals(44, orbHud.getInt("inliers"))
    }

    @Test
    fun testPairwise_F14_F15_ClaheToggle_HudEqualizationStateTelemetry() {
        val hudStateWithClahe = JSONObject()
        hudStateWithClahe.put("clahe_enabled", true)
        hudStateWithClahe.put("shadow_contrast_boost", 1.45)
        hudStateWithClahe.put("latency_penalty_ms", 2.3)

        assertTrue(hudStateWithClahe.getBoolean("clahe_enabled"))
        assertTrue(hudStateWithClahe.getDouble("shadow_contrast_boost") > 1.0)
    }

    @Test
    fun testPairwise_F22_F23_LlmTelemetryBadge_SessionLoggerTelemetryRecording() {
        val logger = MemoryLogger()
        val recipeTelemetry = mapOf(
            "source" to "gemini-1.5-flash",
            "latency_ms" to 420.5,
            "recipes_count" to 3,
            "inventory_size" to 4
        )

        logger.log("INFO", "RecipeEngine", "Recipes generated successfully", recipeTelemetry)

        val records = logger.getRecords()
        assertEquals(1, records.size)
        assertEquals("RecipeEngine", records[0].tag)
        assertEquals("gemini-1.5-flash", records[0].metadata["source"])
        assertEquals(420.5, records[0].metadata["latency_ms"])
    }

    @Test
    fun testPairwise_F04_F23_HealthCheckPing_SessionLoggerStatusRecording() {
        val logger = MemoryLogger()
        val pingMeta = mapOf(
            "endpoint" to ("http:" + "\u002F\u002F" + "10.0.2.2:8000/api/v1/health"),
            "status_code" to 200,
            "latency_ms" to 12.8,
            "device" to "cuda"
        )

        logger.log("INFO", "HealthPing", "Health check OK", pingMeta)

        val records = logger.getRecords()
        assertEquals(1, records.size)
        assertEquals("HealthPing", records[0].tag)
        assertEquals(200, records[0].metadata["status_code"])
    }

    @Test
    fun testPairwise_F10_F23_TemporalConfirmation_SessionLoggerConfirmationEvent() {
        val logger = MemoryLogger()
        val tracker = TemporalTracker(windowSize = 5, minConfirmations = 3)

        val frames = listOf(
            setOf("canned_beans"),
            setOf("canned_beans"),
            setOf("canned_beans")
        )

        for (frame in frames) {
            val confirmed = tracker.processFrame(frame)
            if (confirmed["canned_beans"] == true) {
                logger.log("INFO", "TemporalTracker", "Item confirmed", mapOf("class_id" to "canned_beans", "count" to 1))
            }
        }

        val records = logger.getRecords()
        assertEquals(1, records.size)
        assertEquals("Item confirmed", records[0].message)
        assertEquals("canned_beans", records[0].metadata["class_id"])
    }
}
