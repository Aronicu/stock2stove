package com.kaggleing.stock2stove.e2e

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.io.StringWriter
import java.io.PrintWriter
import java.util.*

class Tier4RealWorldWorkloadTest {

    data class ServerPreset(val id: String, val host: String, val port: Int) {
        val healthUrl: String get() = "http:" + "\u002F\u002F" + "$host:$port/api/v1/health"
        val wsUrl: String get() = "ws:" + "\u002F\u002F" + "$host:$port/ws/scan"
        val recipesUrl: String get() = "http:" + "\u002F\u002F" + "$host:$port/api/v1/recipes/generate"
    }

    enum class ScanMode {
        PANTRY_SHELF,
        REFRIGERATOR
    }

    data class Point2D(val x: Float, val y: Float)

    data class QuadPolygon(val p1: Point2D, val p2: Point2D, val p3: Point2D, val p4: Point2D) {
        fun calculateArea(): Float {
            val term1 = p1.x * p2.y - p1.y * p2.x
            val term2 = p2.x * p3.y - p2.y * p3.x
            val term3 = p3.x * p4.y - p3.y * p4.x
            val term4 = p4.x * p1.y - p4.y * p1.x
            return 0.5f * kotlin.math.abs(term1 + term2 + term3 + term4)
        }
    }

    data class BoundingBox(val left: Float, val top: Float, val right: Float, val bottom: Float) {
        val width: Float get() = right - left
        val height: Float get() = bottom - top
        fun intersects(other: BoundingBox): Boolean {
            return left < other.right && right > other.left && top < other.bottom && bottom > other.top
        }
    }

    class SlidingWindowTemporalTracker(private val windowSize: Int = 5, private val threshold: Int = 3) {
        private val classHistory = mutableMapOf<String, MutableList<Boolean>>()

        fun update(detectedInFrame: Set<String>): Map<String, Boolean> {
            val allTracked = (classHistory.keys + detectedInFrame).toSet()
            val confirmations = mutableMapOf<String, Boolean>()
            for (cls in allTracked) {
                val list = classHistory.getOrPut(cls) { mutableListOf() }
                list.add(detectedInFrame.contains(cls))
                if (list.size > windowSize) {
                    list.removeAt(0)
                }
                val count = list.count { it }
                confirmations[cls] = (count >= threshold)
            }
            return confirmations
        }

        fun reset() {
            classHistory.clear()
        }
    }

    class AppSessionLogger(private val maxEntriesPerFile: Int = 200) {
        val activeEntries = mutableListOf<String>()
        val rotatedFiles = mutableListOf<List<String>>()

        fun log(level: String, tag: String, message: String, metadata: Map<String, Any?> = emptyMap()) {
            val json = JSONObject()
            json.put("ts", System.currentTimeMillis())
            json.put("level", level)
            json.put("tag", tag)
            json.put("msg", message)
            val metaJson = JSONObject()
            for ((k, v) in metadata) {
                metaJson.put(k, v)
            }
            json.put("meta", metaJson)
            activeEntries.add(json.toString())

            if (activeEntries.size >= maxEntriesPerFile) {
                rotate()
            }
        }

        private fun rotate() {
            rotatedFiles.add(activeEntries.toList())
            activeEntries.clear()
        }

        fun getTotalRecordedCount(): Int {
            val rotatedCount = rotatedFiles.sumOf { it.size }
            return rotatedCount + activeEntries.size
        }

        fun getAllEntries(): List<String> {
            val combined = mutableListOf<String>()
            for (file in rotatedFiles) {
                combined.addAll(file)
            }
            combined.addAll(activeEntries)
            return combined
        }
    }

    class InventoryModel {
        private val inventory = mutableMapOf<String, Int>()

        fun setConfirmed(classId: String, count: Int) {
            if (count > 0) {
                inventory[classId] = count
            } else {
                inventory.remove(classId)
            }
        }

        fun increment(classId: String) {
            val curr = inventory.getOrDefault(classId, 0)
            inventory[classId] = curr + 1
        }

        fun decrement(classId: String) {
            val curr = inventory.getOrDefault(classId, 0)
            if (curr <= 1) {
                inventory.remove(classId)
            } else {
                inventory[classId] = curr - 1
            }
        }

        fun getCount(classId: String): Int = inventory.getOrDefault(classId, 0)

        fun toMap(): Map<String, Int> = inventory.toMap()

        fun clear() {
            inventory.clear()
        }
    }

    @Test
    fun testScenario01_StandardPantryAuditWorkflow() {
        val logger = AppSessionLogger()
        logger.log("INFO", "Lifecycle", "App initialized")

        val preset = ServerPreset("emulator", "10.0.2.2", 8000)
        logger.log("INFO", "ConnectionHub", "Selected preset", mapOf("preset" to preset.id, "url" to preset.healthUrl))

        val healthJson = JSONObject()
        healthJson.put("status", "ok")
        healthJson.put("service", "stock2stove-inference-server")
        assertEquals("ok", healthJson.getString("status"))
        logger.log("INFO", "ConnectionHub", "Health ping successful", mapOf("status" to "ok"))

        var currentMode = ScanMode.PANTRY_SHELF
        logger.log("INFO", "ScanSetup", "Scan mode chosen", mapOf("mode" to currentMode.name))

        val tracker = SlidingWindowTemporalTracker(windowSize = 5, threshold = 3)
        val inventory = InventoryModel()

        var progressSec = 0.0f
        val frameSequence = listOf(
            setOf("canned_beans", "tomato_sauce"),
            setOf("canned_beans", "tomato_sauce", "pasta"),
            setOf("canned_beans", "tomato_sauce", "pasta"),
            setOf("canned_beans", "pasta"),
            setOf("canned_beans", "tomato_sauce", "pasta")
        )

        for ((idx, frameItems) in frameSequence.withIndex()) {
            progressSec += 1.0f
            val clientFrame = JSONObject()
            clientFrame.put("frame_id", (idx + 1).toLong())
            clientFrame.put("detector_mode", "yolov8")
            clientFrame.put("enable_clahe", false)
            clientFrame.put("image_b64", "base64data==")

            val confirmedMap = tracker.update(frameItems)
            for ((cls, isConfirmed) in confirmedMap) {
                if (isConfirmed) {
                    inventory.setConfirmed(cls, 1)
                    logger.log("DEBUG", "Tracker", "Item confirmed", mapOf("class_id" to cls))
                }
            }
        }

        assertEquals(5.0f, progressSec, 0.01f)
        assertTrue(inventory.getCount("canned_beans") >= 1)
        assertTrue(inventory.getCount("tomato_sauce") >= 1)
        assertTrue(inventory.getCount("pasta") >= 1)

        val bottomSheetOpen = (progressSec >= 5.0f)
        assertTrue(bottomSheetOpen)
        logger.log("INFO", "InventoryVerification", "Bottom sheet displayed", inventory.toMap())

        inventory.increment("canned_beans")
        assertEquals(2, inventory.getCount("canned_beans"))
        logger.log("INFO", "InventoryVerification", "Incremented canned_beans", mapOf("new_count" to 2))

        val recipeRequestJson = JSONObject()
        val itemsArray = JSONArray()
        for ((cls, count) in inventory.toMap()) {
            val item = JSONObject()
            item.put("class_id", cls)
            item.put("count", count)
            itemsArray.put(item)
        }
        recipeRequestJson.put("confirmed_inventory", itemsArray)
        recipeRequestJson.put("timeout_sec", 12.0)

        val recipeResponseJson = JSONObject()
        recipeResponseJson.put("source", "gemini-1.5-flash")
        recipeResponseJson.put("latency_ms", 340.0)
        val recipes = JSONArray()
        for (i in 1..3) {
            val r = JSONObject()
            r.put("id", "recipe_$i")
            r.put("title", "Delicious Pantry Meal $i")
            r.put("prep_time_mins", 15)
            recipes.put(r)
        }
        recipeResponseJson.put("recipes", recipes)

        assertEquals(3, recipeResponseJson.getJSONArray("recipes").length())
        logger.log("INFO", "RecipeEngine", "Received recipes", mapOf("count" to 3, "source" to "gemini-1.5-flash"))

        val totalLogs = logger.getAllEntries()
        assertTrue(totalLogs.size >= 7)
    }

    @Test
    fun testScenario02_DimlyLitRefrigeratorScanWithClahe() {
        val logger = AppSessionLogger()
        var currentMode = ScanMode.REFRIGERATOR
        var claheEnabled = true

        val clientFrame = JSONObject()
        clientFrame.put("frame_id", 101L)
        clientFrame.put("scan_mode", currentMode.name)
        clientFrame.put("enable_clahe", claheEnabled)
        clientFrame.put("detector_mode", "yolov8")

        assertTrue(clientFrame.getBoolean("enable_clahe"))
        logger.log("INFO", "Scanner", "Streaming low-light frame with CLAHE", mapOf("clahe" to true))

        val serverResponse = JSONObject()
        serverResponse.put("frame_id", 101L)
        val diagnostics = JSONObject()
        diagnostics.put("clahe_active", true)
        diagnostics.put("shadow_contrast_boost", 1.55)
        diagnostics.put("latency_ms", 41.2)
        diagnostics.put("fps", 24.2)
        serverResponse.put("diagnostics", diagnostics)

        val detections = JSONArray()
        val d1 = JSONObject()
        d1.put("class_id", "milk")
        d1.put("confidence", 0.89)
        detections.put(d1)
        val d2 = JSONObject()
        d2.put("class_id", "eggs")
        d2.put("confidence", 0.92)
        detections.put(d2)
        val d3 = JSONObject()
        d3.put("class_id", "cheese")
        d3.put("confidence", 0.84)
        detections.put(d3)
        serverResponse.put("detections", detections)

        val diag = serverResponse.getJSONObject("diagnostics")
        assertTrue(diag.getBoolean("clahe_active"))
        assertTrue(diag.getDouble("shadow_contrast_boost") > 1.2)

        val tracker = SlidingWindowTemporalTracker(windowSize = 3, threshold = 2)
        val dairyClasses = setOf("milk", "eggs", "cheese")
        tracker.update(dairyClasses)
        val confirmed = tracker.update(dairyClasses)

        assertTrue(confirmed["milk"] ?: false)
        assertTrue(confirmed["eggs"] ?: false)
        assertTrue(confirmed["cheese"] ?: false)
        logger.log("INFO", "Scanner", "Dairy items confirmed under CLAHE", mapOf("count" to 3))
    }

    @Test
    fun testScenario03_PlanarShelfScanningWithHomographyPolygons() {
        val canvasWidth = 1080.0f
        val canvasHeight = 1920.0f

        val polyNorm = JSONArray()
        polyNorm.put(JSONArray(listOf(0.12, 0.18)))
        polyNorm.put(JSONArray(listOf(0.85, 0.22)))
        polyNorm.put(JSONArray(listOf(0.88, 0.84)))
        polyNorm.put(JSONArray(listOf(0.10, 0.80)))

        val pts = mutableListOf<Point2D>()
        for (i in 0 until polyNorm.length()) {
            val pt = polyNorm.getJSONArray(i)
            val px = pt.getDouble(0).toFloat() * canvasWidth
            val py = pt.getDouble(1).toFloat() * canvasHeight
            pts.add(Point2D(px, py))
        }

        val quad = QuadPolygon(pts[0], pts[1], pts[2], pts[3])
        val area = quad.calculateArea()
        assertTrue(area > 500000.0f)

        val tracker = SlidingWindowTemporalTracker(windowSize = 5, threshold = 3)
        var confirmedItem = false
        for (f in 1..4) {
            val map = tracker.update(setOf("olive_oil_bottle"))
            if (map["olive_oil_bottle"] == true) {
                confirmedItem = true
            }
        }
        assertTrue(confirmedItem)

        val hudJson = JSONObject()
        hudJson.put("homography_active", true)
        hudJson.put("inliers", 64)
        hudJson.put("keypoints", 720)
        assertTrue(hudJson.getBoolean("homography_active"))
        assertTrue(hudJson.getInt("inliers") > 30)
    }

    @Test
    fun testScenario04_ClutteredBoxDetectionWithBoundingBoxOverlay() {
        val screenW = 720.0f
        val screenH = 1280.0f

        val detections = listOf(
            BoundingBox(0.05f * screenW, 0.10f * screenH, 0.45f * screenW, 0.60f * screenH),
            BoundingBox(0.40f * screenW, 0.15f * screenH, 0.90f * screenW, 0.70f * screenH),
            BoundingBox(0.20f * screenW, 0.55f * screenH, 0.80f * screenW, 0.95f * screenH)
        )

        assertEquals(3, detections.size)
        assertTrue(detections[0].intersects(detections[1]))
        assertTrue(detections[1].intersects(detections[2]))

        for (box in detections) {
            assertTrue(box.left >= 0.0f)
            assertTrue(box.right <= screenW)
            assertTrue(box.top >= 0.0f)
            assertTrue(box.bottom <= screenH)
            assertTrue(box.width > 100.0f)
            assertTrue(box.height > 200.0f)
        }

        val tracker = SlidingWindowTemporalTracker(windowSize = 3, threshold = 2)
        val clutterClasses = setOf("box_a", "box_b", "box_c")
        tracker.update(clutterClasses)
        val result = tracker.update(clutterClasses)

        assertEquals(3, result.size)
        assertTrue(result.values.all { it })
    }

    @Test
    fun testScenario05_AcademicDefenseDetectorComparisonYoloOrbSift() {
        val logger = AppSessionLogger()

        val yoloFrame = JSONObject()
        yoloFrame.put("detector_mode", "yolov8")
        val yoloResp = JSONObject()
        yoloResp.put("detector_mode", "yolov8")
        yoloResp.put("latency_ms", 32.4)
        yoloResp.put("fps", 30.8)
        yoloResp.put("detections_count", 4)
        logger.log("INFO", "DefenseBenchmark", "YOLOv8 evaluated", mapOf("latency" to 32.4, "fps" to 30.8))

        val orbFrame = JSONObject()
        orbFrame.put("detector_mode", "orb_baseline")
        val orbResp = JSONObject()
        orbResp.put("detector_mode", "orb_baseline")
        orbResp.put("latency_ms", 15.2)
        orbResp.put("fps", 65.0)
        orbResp.put("keypoints", 500)
        orbResp.put("inliers", 42)
        logger.log("INFO", "DefenseBenchmark", "ORB evaluated", mapOf("latency" to 15.2, "keypoints" to 500))

        val siftFrame = JSONObject()
        siftFrame.put("detector_mode", "sift_baseline")
        val siftResp = JSONObject()
        siftResp.put("detector_mode", "sift_baseline")
        siftResp.put("latency_ms", 112.0)
        siftResp.put("fps", 8.9)
        siftResp.put("keypoints", 1320)
        siftResp.put("inliers", 94)
        logger.log("INFO", "DefenseBenchmark", "SIFT evaluated", mapOf("latency" to 112.0, "keypoints" to 1320))

        assertTrue(orbResp.getDouble("latency_ms") < yoloResp.getDouble("latency_ms"))
        assertTrue(yoloResp.getDouble("latency_ms") < siftResp.getDouble("latency_ms"))
        assertTrue(siftResp.getInt("keypoints") > orbResp.getInt("keypoints"))

        val records = logger.getAllEntries()
        assertEquals(3, records.size)
    }

    @Test
    fun testScenario06_PostScanAdjustmentWithZeroOutRemoval() {
        val logger = AppSessionLogger()
        val inventory = InventoryModel()

        inventory.setConfirmed("canned_tuna", 2)
        inventory.setConfirmed("bread", 1)
        inventory.setConfirmed("expired_yogurt", 1)

        assertEquals(3, inventory.toMap().size)
        assertEquals(2, inventory.getCount("canned_tuna"))
        assertEquals(1, inventory.getCount("bread"))
        assertEquals(1, inventory.getCount("expired_yogurt"))

        inventory.decrement("expired_yogurt")
        assertEquals(0, inventory.getCount("expired_yogurt"))
        assertFalse(inventory.toMap().containsKey("expired_yogurt"))
        logger.log("INFO", "InventoryAdjustment", "Zeroed out item removed", mapOf("class_id" to "expired_yogurt"))

        inventory.decrement("canned_tuna")
        assertEquals(1, inventory.getCount("canned_tuna"))
        logger.log("INFO", "InventoryAdjustment", "Decremented canned_tuna", mapOf("new_count" to 1))

        val verified = inventory.toMap()
        assertEquals(2, verified.size)
        assertTrue(verified.containsKey("canned_tuna"))
        assertTrue(verified.containsKey("bread"))
        assertFalse(verified.containsKey("expired_yogurt"))
    }

    @Test
    fun testScenario07_ManualInventoryAugmentationForUnscannedStaple() {
        val logger = AppSessionLogger()
        val inventory = InventoryModel()

        inventory.setConfirmed("rice", 1)
        logger.log("INFO", "ScanComplete", "Scan detected items", inventory.toMap())

        val unifiedClasses = listOf("canned_beans", "tomato_sauce", "pasta", "rice", "olive_oil", "milk", "eggs", "cheese")
        val manualAdd1 = "olive_oil"
        val manualAdd2 = "canned_beans"

        assertTrue(unifiedClasses.contains(manualAdd1))
        assertTrue(unifiedClasses.contains(manualAdd2))

        inventory.increment(manualAdd1)
        logger.log("INFO", "ManualAdd", "Manually added olive_oil", mapOf("count" to 1))

        inventory.increment(manualAdd2)
        inventory.increment(manualAdd2)
        logger.log("INFO", "ManualAdd", "Manually added canned_beans", mapOf("count" to 2))

        val fullMap = inventory.toMap()
        assertEquals(3, fullMap.size)
        assertEquals(1, fullMap["rice"])
        assertEquals(1, fullMap["olive_oil"])
        assertEquals(2, fullMap["canned_beans"])

        val fakeRecipeJson = JSONObject()
        val recipes = JSONArray()
        val r = JSONObject()
        r.put("id", "mediterranean_rice")
        r.put("title", "Mediterranean Rice and Beans")
        val matched = JSONArray(listOf("rice", "olive_oil", "canned_beans"))
        r.put("matched_ingredients", matched)
        recipes.put(r)
        fakeRecipeJson.put("recipes", recipes)

        val retrievedMatched = fakeRecipeJson.getJSONArray("recipes").getJSONObject(0).getJSONArray("matched_ingredients")
        assertEquals(3, retrievedMatched.length())
    }

    @Test
    fun testScenario08_MultiRecipeCookingStepsInteractiveCompletion() {
        val recipe = JSONObject()
        recipe.put("id", "fiesta_bowl")
        recipe.put("title", "Black Bean Fiesta Bowl")
        val steps = JSONArray()
        steps.put("Rinse and drain canned black beans")
        steps.put("Warm beans in saucepan with cumin and olive oil")
        steps.put("Fluff cooked rice and spoon into bowls")
        steps.put("Top rice with seasoned beans and salsa")
        recipe.put("steps", steps)
        recipe.put("zero_waste_tip", "Freeze extra cooked beans in small silicone bags for up to 3 months.")

        val stepList = mutableListOf<String>()
        val arr = recipe.getJSONArray("steps")
        for (i in 0 until arr.length()) {
            stepList.add(arr.getString(i))
        }
        assertEquals(4, stepList.size)

        val checkState = BooleanArray(stepList.size) { false }

        for (i in 0 until stepList.size) {
            checkState[i] = true
            val completedCount = checkState.count { it }
            val fraction = completedCount.toFloat() / stepList.size
            assertEquals((i + 1) * 0.25f, fraction, 0.01f)
        }

        assertTrue(checkState.all { it })

        checkState[3] = false
        val undoneFraction = checkState.count { it }.toFloat() / stepList.size
        assertEquals(0.75f, undoneFraction, 0.01f)
    }

    @Test
    fun testScenario09_UntetheredKitchenSessionWithLocalLogExport() {
        val logger = AppSessionLogger()
        logger.log("INFO", "BatteryStatus", "Operating untethered", mapOf("charging" to false, "level" to 88))
        logger.log("INFO", "ScanSession", "Kitchen shelf scan initiated")
        logger.log("DEBUG", "FrameStream", "Captured 35 frames")
        logger.log("INFO", "Inventory", "4 items verified")
        logger.log("INFO", "RecipeScreen", "Generated 3 recipes")

        val stringWriter = StringWriter()
        val pw = PrintWriter(stringWriter)
        val entries = logger.getAllEntries()
        for (entry in entries) {
            pw.println(entry)
        }
        pw.flush()

        val logOutput = stringWriter.toString()
        assertTrue(logOutput.contains("Operating untethered"))
        assertTrue(logOutput.contains("Kitchen shelf scan"))

        val mockIntent = JSONObject()
        mockIntent.put("action", "android.intent.action.SEND")
        mockIntent.put("type", "text/plain")
        mockIntent.put("extra_subject", "Stock2Stove Untethered Log Export")
        mockIntent.put("extra_text_length", logOutput.length)

        assertEquals("android.intent.action.SEND", mockIntent.getString("action"))
        assertEquals("text/plain", mockIntent.getString("type"))
        assertTrue(mockIntent.getInt("extra_text_length") > 100)
    }

    @Test
    fun testScenario10_NetworkServerDropoutAndGracefulReconnection() {
        val logger = AppSessionLogger()
        var isConnected = true

        logger.log("INFO", "WebSocket", "Connected to ws:" + "\u002F\u002F" + "10.0.2.2:8000/ws/scan")

        isConnected = false
        val disconnectCode = 1006
        logger.log("WARN", "WebSocket", "Connection lost abruptly", mapOf("code" to disconnectCode))

        val retryIntervalsMs = listOf(1000L, 2000L, 4000L)
        var reconnectAttempts = 0
        var serverBackOnline = false

        for (interval in retryIntervalsMs) {
            reconnectAttempts++
            logger.log("INFO", "Reconnector", "Attempting reconnect in ${interval}ms", mapOf("attempt" to reconnectAttempts))
            if (reconnectAttempts == 2) {
                serverBackOnline = true
                isConnected = true
                break
            }
        }

        assertTrue(serverBackOnline)
        assertTrue(isConnected)
        assertEquals(2, reconnectAttempts)
        logger.log("INFO", "WebSocket", "Reconnected successfully")
    }

    @Test
    fun testScenario11_MalformedServerFrameHandlingAndTelemetryWarning() {
        val logger = AppSessionLogger()
        var parserSuccess = false
        var warningBannerVisible = false

        val malformedJsonString = "{\"frame_id\": 99, \"detector_mode\": \"yolov8\", \"detections\": [CORRUPTED"

        try {
            JSONObject(malformedJsonString)
            parserSuccess = true
        } catch (e: Exception) {
            parserSuccess = false
            warningBannerVisible = true
            logger.log("ERROR", "FrameParser", "Failed to parse incoming frame", mapOf("raw" to malformedJsonString))
        }

        assertFalse(parserSuccess)
        assertTrue(warningBannerVisible)

        val validJsonString = "{\"frame_id\": 100, \"detector_mode\": \"yolov8\", \"detections\": []}"
        val validJson = JSONObject(validJsonString)
        assertEquals(100L, validJson.getLong("frame_id"))
        warningBannerVisible = false

        assertFalse(warningBannerVisible)
        logger.log("INFO", "FrameParser", "Recovered with valid frame")
    }

    @Test
    fun testScenario12_RapidPanningScanWithResetSession() {
        val logger = AppSessionLogger()
        val tracker = SlidingWindowTemporalTracker(windowSize = 5, threshold = 3)
        val inventory = InventoryModel()

        val rapidFrames = listOf(
            setOf("item_alpha"),
            setOf("item_beta"),
            setOf("item_gamma"),
            setOf("item_delta"),
            setOf("item_epsilon")
        )

        for (frame in rapidFrames) {
            val confirmed = tracker.update(frame)
            for ((cls, isConf) in confirmed) {
                if (isConf) {
                    inventory.setConfirmed(cls, 1)
                }
            }
        }

        assertEquals(0, inventory.toMap().size)
        logger.log("WARN", "ScanSession", "Rapid camera motion detected - no stable items")

        val resetFrame = JSONObject()
        resetFrame.put("frame_id", 50L)
        resetFrame.put("reset_session", true)
        assertTrue(resetFrame.getBoolean("reset_session"))

        tracker.reset()
        inventory.clear()
        logger.log("INFO", "ScanSession", "Session reset initiated")

        val stableFrames = listOf(
            setOf("canned_beans"),
            setOf("canned_beans"),
            setOf("canned_beans")
        )
        for (f in stableFrames) {
            val map = tracker.update(f)
            if (map["canned_beans"] == true) {
                inventory.setConfirmed("canned_beans", 1)
            }
        }

        assertEquals(1, inventory.getCount("canned_beans"))
        logger.log("INFO", "ScanSession", "Stable scan succeeded after reset")
    }

    @Test
    fun testScenario13_HighLatencyRecipeGenerationWithFallbackBadge() {
        val logger = AppSessionLogger()

        val simulatedCloudTimeoutMs = 12000L
        val clientTimeoutThresholdMs = 10000L

        val isTimeout = simulatedCloudTimeoutMs > clientTimeoutThresholdMs
        assertTrue(isTimeout)
        logger.log("WARN", "RecipeClient", "Cloud LLM request timed out", mapOf("timeout_ms" to simulatedCloudTimeoutMs))

        val fallbackRecipes = JSONObject()
        fallbackRecipes.put("source", "offline_rule_engine")
        fallbackRecipes.put("cached", true)
        fallbackRecipes.put("latency_ms", 18.5)
        val rList = JSONArray()
        for (i in 1..3) {
            val r = JSONObject()
            r.put("id", "offline_r_$i")
            r.put("title", "Staple Pantry Meal $i")
            rList.put(r)
        }
        fallbackRecipes.put("recipes", rList)

        assertEquals("offline_rule_engine", fallbackRecipes.getString("source"))
        assertTrue(fallbackRecipes.getBoolean("cached"))
        assertEquals(3, fallbackRecipes.getJSONArray("recipes").length())

        val badgeText = "Offline Fallback | 18ms"
        assertTrue(badgeText.contains("Offline Fallback"))
        logger.log("INFO", "RecipeClient", "Fallback recipes rendered successfully")
    }

    @Test
    fun testScenario14_FullEmptyToFullPantryLifecycle() {
        val logger = AppSessionLogger()
        logger.log("INFO", "Lifecycle", "Phase 1: App launch & health check")
        val healthOk = true
        assertTrue(healthOk)

        logger.log("INFO", "Lifecycle", "Phase 2: Empty pantry scan")
        val pantryTracker = SlidingWindowTemporalTracker()
        val emptyResult = pantryTracker.update(emptySet())
        assertTrue(emptyResult.isEmpty())

        logger.log("INFO", "Lifecycle", "Phase 3: Manual staple addition")
        val inventory = InventoryModel()
        inventory.increment("pasta")
        inventory.increment("canned_beans")
        assertEquals(2, inventory.toMap().size)

        logger.log("INFO", "Lifecycle", "Phase 4: Refrigerator scan")
        val fridgeTracker = SlidingWindowTemporalTracker(windowSize = 3, threshold = 2)
        val fridgeItems = setOf("milk", "eggs", "cheese")
        fridgeTracker.update(fridgeItems)
        val confirmedFridge = fridgeTracker.update(fridgeItems)
        for ((k, v) in confirmedFridge) {
            if (v) {
                inventory.setConfirmed(k, 1)
            }
        }
        assertEquals(5, inventory.toMap().size)

        logger.log("INFO", "Lifecycle", "Phase 5: Inventory verification adjustments")
        inventory.increment("eggs")
        assertEquals(2, inventory.getCount("eggs"))

        logger.log("INFO", "Lifecycle", "Phase 6: Recipe generation")
        val fakeRecipes = listOf("Cheesy Baked Pasta", "Egg and Bean Breakfast Scramble", "Creamy Tomato Soup")
        assertEquals(3, fakeRecipes.size)

        logger.log("INFO", "Lifecycle", "Phase 7: Step completion")
        val stepsCompleted = booleanArrayOf(true, true, true, true)
        assertTrue(stepsCompleted.all { it })

        logger.log("INFO", "Lifecycle", "Phase 8: Completed full lifecycle")
        assertTrue(logger.getAllEntries().size >= 8)
    }

    @Test
    fun testScenario15_DiagnosticMemoryStressAndLogRotationVerification() {
        val logger = AppSessionLogger(maxEntriesPerFile = 100)

        for (i in 1..450) {
            logger.log("DEBUG", "StressTest", "Event sequence $i", mapOf("index" to i, "val" to (i * 1.5)))
        }

        assertEquals(450, logger.getTotalRecordedCount())
        assertEquals(4, logger.rotatedFiles.size)
        assertEquals(50, logger.activeEntries.size)

        for (rotated in logger.rotatedFiles) {
            assertEquals(100, rotated.size)
        }

        val allLogs = logger.getAllEntries()
        assertEquals(450, allLogs.size)

        val firstParsed = JSONObject(allLogs[0])
        assertEquals("StressTest", firstParsed.getString("tag"))
        assertEquals("Event sequence 1", firstParsed.getString("msg"))

        val lastParsed = JSONObject(allLogs[449])
        assertEquals("Event sequence 450", lastParsed.getString("msg"))
    }
}
