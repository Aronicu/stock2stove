package com.kaggleing.stock2stove

import com.kaggleing.stock2stove.data.repository.AlphaTelemetryLogger
import com.kaggleing.stock2stove.data.repository.LogLevel
import com.kaggleing.stock2stove.ui.diagnostics.DiagnosticsViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

@OptIn(ExperimentalCoroutinesApi::class)
class TelemetryLoggerTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private lateinit var logDir: File
    private lateinit var logger: AlphaTelemetryLogger

    @Before
    fun setUp() {
        logDir = tempFolder.newFolder("logs")
        logger = AlphaTelemetryLogger(baseDirectory = logDir, sessionId = "test_alpha_01")
    }

    @Test
    fun testLogEventCreatesValidJsonlRecord() {
        val entry = logger.logEvent(
            tag = "NETWORK",
            level = "INFO",
            message = "WebSocket connected",
            payloadJson = "{\"url\":\"ws://127.0.0.1:8000/ws/scan\"}"
        )

        assertNotNull(entry)
        assertEquals("NETWORK", entry.tag)
        assertEquals("INFO", entry.level)
        assertEquals("INFO", entry.severity)
        assertEquals("NETWORK", entry.eventType)
        assertEquals("WebSocket connected", entry.message)
        assertEquals("test_alpha_01", entry.sessionId)
        assertTrue(entry.timestamp > 0L)
        assertNotNull(entry.payloadJson)

        val logFile = logger.getActiveLogFile()
        assertTrue(logFile.exists())
        val lines = logFile.readLines()
        assertEquals(1, lines.size)

        val rawJson = JSONObject(lines[0])
        assertEquals("INFO", rawJson.getString("level"))
        assertEquals("NETWORK", rawJson.getString("tag"))
        assertEquals("WebSocket connected", rawJson.getString("message"))
        assertEquals("test_alpha_01", rawJson.getString("sessionId"))
    }

    @Test
    fun testLogTelemetrySerialization() {
        val entry = logger.logTelemetry(
            frameId = 101L,
            detectorMode = "YOLOv8n",
            inferenceMs = 24.5f,
            fps = 10.2f,
            detectionsCount = 4
        )

        assertEquals("TELEMETRY", entry.tag)
        assertEquals("TELEMETRY", entry.level)
        assertTrue(entry.message.contains("Frame #101"))
        assertTrue(entry.message.contains("YOLOv8n"))

        val payload = JSONObject(entry.payloadJson!!)
        assertEquals(101L, payload.getLong("frame_id"))
        assertEquals("YOLOv8n", payload.getString("detector_mode"))
        assertEquals(24.5, payload.getDouble("inference_ms"), 0.01)
        assertEquals(10.2, payload.getDouble("fps"), 0.01)
        assertEquals(4, payload.getInt("detections_count"))

        val readBack = logger.readLogLines()
        assertEquals(1, readBack.size)
        assertEquals("TELEMETRY", readBack[0].level)
        assertEquals(entry.message, readBack[0].message)
    }

    @Test
    fun testLogErrorFormattingAndContext() {
        val testException = IllegalStateException("Camera stream timed out")
        val entry = logger.logError(
            tag = "CAMERA_STREAM",
            error = testException,
            context = "FrameAcquisition"
        )

        assertEquals("CAMERA_STREAM", entry.tag)
        assertEquals("ERROR", entry.level)
        assertEquals("FrameAcquisition: Camera stream timed out", entry.message)

        val payload = JSONObject(entry.payloadJson!!)
        assertEquals("java.lang.IllegalStateException", payload.getString("error_class"))
        assertEquals("Camera stream timed out", payload.getString("error_message"))

        val readBack = logger.readLogLines()
        assertEquals(1, readBack.size)
        assertEquals("ERROR", readBack[0].level)
    }

    @Test
    fun testSequentialAppendsPreserveHistory() {
        logger.logEvent(tag = "APP", level = "INFO", message = "App launched")
        logger.logEvent(tag = "CAMERA", level = "WARN", message = "Low light condition")
        logger.logTelemetry(frameId = 1L, detectorMode = "ORB Baseline", inferenceMs = 8.1f, fps = 15.0f, detectionsCount = 2)
        logger.logError(tag = "WS", error = RuntimeException("Connection reset"))

        val entries = logger.readLogLines()
        assertEquals(4, entries.size)
        assertEquals("App launched", entries[0].message)
        assertEquals("Low light condition", entries[1].message)
        assertEquals("TELEMETRY", entries[2].level)
        assertEquals("ERROR", entries[3].level)

        val logFiles = logger.getLogFiles()
        assertTrue(logFiles.isNotEmpty())
        assertTrue(logger.getLogFileSize() > 0L)
    }

    @Test
    fun testReadLogLinesWithCustomFile() {
        val customFile = File(logDir, "custom_export.jsonl")
        customFile.writeText("{\"timestamp\":1000,\"sessionId\":\"s99\",\"level\":\"WARN\",\"tag\":\"SYS\",\"message\":\"Disk space low\"}\n")

        val entries = logger.readLogLines(customFile)
        assertEquals(1, entries.size)
        assertEquals("SYS", entries[0].tag)
        assertEquals("WARN", entries[0].level)
        assertEquals("Disk space low", entries[0].message)
    }

    @Test
    fun testSeverityFiltering() {
        logger.logEvent(tag = "E1", level = "INFO", message = "Info 1")
        logger.logEvent(tag = "E2", level = "WARN", message = "Warn 1")
        logger.logEvent(tag = "E3", level = "ERROR", message = "Error 1")
        logger.logEvent(tag = "E4", level = "ERROR", message = "Error 2")
        logger.logTelemetry(frameId = 5L, detectorMode = "YOLO", inferenceMs = 12.0f, fps = 20.0f, detectionsCount = 1)

        val all = logger.readLogLines()
        assertEquals(5, all.size)

        val errors = all.filter { it.level == "ERROR" }
        assertEquals(2, errors.size)
        assertTrue(errors.all { it.level == "ERROR" })

        val warnings = all.filter { it.level == "WARN" }
        assertEquals(1, warnings.size)
        assertEquals("Warn 1", warnings[0].message)

        val infos = all.filter { it.level == "INFO" }
        assertEquals(1, infos.size)
        assertEquals("Info 1", infos[0].message)

        val telemetry = all.filter { it.level == "TELEMETRY" }
        assertEquals(1, telemetry.size)
    }

    @Test
    fun testClearLogsRemovesFilesOrTruncates() {
        logger.logEvent(tag = "INIT", level = "INFO", message = "Init event")
        logger.logEvent(tag = "SCAN", level = "INFO", message = "Scan event")
        assertTrue(logger.readLogLines().isNotEmpty())
        assertTrue(logger.getLogFileSize() > 0L)

        logger.clearLogs()

        val afterClear = logger.readLogLines()
        assertEquals(0, afterClear.size)
        assertEquals(0L, logger.getLogFileSize())
    }

    @Test
    fun testGetLogFilesReturnsAllJsonlFiles() {
        logger.logEvent(tag = "E1", level = "INFO", message = "Message 1")

        val secondFile = File(logDir, "session_test_alpha_02.jsonl")
        secondFile.writeText("{\"timestamp\":2000,\"sessionId\":\"test_alpha_02\",\"level\":\"INFO\",\"tag\":\"T2\",\"message\":\"Other session\"}\n")

        val textFile = File(logDir, "readme.txt")
        textFile.writeText("Not a log file")

        val files = logger.getLogFiles()
        assertEquals(2, files.size)
        assertTrue(files.all { it.name.endsWith(".jsonl") })
        assertFalse(files.any { it.name == "readme.txt" })
    }

    @Test
    fun testSharePayloadContentAndContracts() {
        logger.logEvent(tag = "PRESET", level = "INFO", message = "Preset selected: emulator")
        logger.logTelemetry(frameId = 20L, detectorMode = "CLAHE", inferenceMs = 30.0f, fps = 8.5f, detectionsCount = 5)

        val payload = logger.getSharePayload()
        assertEquals("android.intent.action.SEND", payload.action)
        assertEquals("text/plain", payload.type)
        assertEquals("Stock-to-Stove Alpha Diagnostics", payload.subject)
        assertEquals(2, payload.lineCount)
        assertTrue(payload.text.contains("Stock-to-Stove Untethered Log Export"))
        assertTrue(payload.text.contains("Preset selected: emulator"))
        assertTrue(payload.text.contains("Total Entries: 2"))
    }

    @Test
    fun testCorruptedLinesHandling() {
        val target = logger.getActiveLogFile()
        target.parentFile?.mkdirs()
        target.writeText("{\"timestamp\":1000,\"sessionId\":\"s1\",\"level\":\"INFO\",\"tag\":\"OK1\",\"message\":\"First valid\"}\n" +
                "THIS_IS_CORRUPTED_JSON_NOT_VALID_SYNTAX\n" +
                "{\"timestamp\":2000,\"sessionId\":\"s1\",\"level\":\"INFO\",\"tag\":\"OK2\",\"message\":\"Second valid\"}\n")

        val entries = logger.readLogLines()
        assertEquals(2, entries.size)
        assertEquals("OK1", entries[0].tag)
        assertEquals("OK2", entries[1].tag)
    }

    @Test
    fun testEmptyLogFileHandledGracefully() {
        val emptyDir = tempFolder.newFolder("empty_logs")
        val emptyLogger = AlphaTelemetryLogger(baseDirectory = emptyDir, sessionId = "empty_sess")

        val entries = emptyLogger.readLogLines()
        assertTrue(entries.isEmpty())
        assertEquals(0L, emptyLogger.getLogFileSize())

        emptyLogger.clearLogs()
        val files = emptyLogger.getLogFiles()
        assertTrue(files.isEmpty())

        val sharePayload = emptyLogger.getSharePayload()
        assertEquals(0, sharePayload.lineCount)
        assertTrue(sharePayload.text.contains("No log entries found"))
    }

    @Test
    fun testDiagnosticsViewModelLoadAndFilter() = runTest {
        val testDispatcher = StandardTestDispatcher(testScheduler)
        val testScope = CoroutineScope(testDispatcher)

        logger.logEvent(tag = "AUTH", level = "INFO", message = "Connected to host")
        logger.logEvent(tag = "DETECTION", level = "WARN", message = "Low confidence detection")
        logger.logError(tag = "CRASH", error = IllegalStateException("Inference core fault"))
        logger.logTelemetry(frameId = 1L, detectorMode = "YOLOv8n", inferenceMs = 15.0f, fps = 12.0f, detectionsCount = 3)

        val viewModel = DiagnosticsViewModel(logger = logger, ioDispatcher = testDispatcher, customScope = testScope)
        testScheduler.advanceUntilIdle()

        val state = viewModel.uiState.value
        assertFalse(state.isLoading)
        assertEquals(4, state.totalEntriesCount)
        assertEquals(4, state.filteredEntries.size)
        assertEquals(1, state.infoCount)
        assertEquals(1, state.warnCount)
        assertEquals(1, state.errorCount)
        assertEquals(1, state.telemetryCount)
        assertNull(state.selectedFilter)
        assertTrue(state.totalSizeBytes > 0L)

        viewModel.setFilter("ERROR")
        val errorState = viewModel.uiState.value
        assertEquals("ERROR", errorState.selectedFilter)
        assertEquals(1, errorState.filteredEntries.size)
        assertEquals("CRASH", errorState.filteredEntries[0].tag)

        viewModel.setFilter("WARN")
        val warnState = viewModel.uiState.value
        assertEquals(1, warnState.filteredEntries.size)
        assertEquals("DETECTION", warnState.filteredEntries[0].tag)

        viewModel.setFilter("ALL")
        val allState = viewModel.uiState.value
        assertNull(allState.selectedFilter)
        assertEquals(4, allState.filteredEntries.size)

        viewModel.clearLogs()
        testScheduler.advanceUntilIdle()

        val clearedState = viewModel.uiState.value
        assertEquals(0, clearedState.totalEntriesCount)
        assertEquals(0, clearedState.filteredEntries.size)
    }

    @Test
    fun testLogEventWithLogLevelEnum() {
        val entry = logger.logEvent(
            level = LogLevel.WARN,
            tag = "BATTERY",
            message = "Battery low (15%)"
        )
        assertEquals("WARN", entry.level)
        assertEquals("BATTERY", entry.tag)
        assertEquals("Battery low (15%)", entry.message)
    }

    @Test
    fun testLogWithMetadataMap() {
        val meta = mapOf("device" to "Pixel 7", "battery_pct" to 85)
        val entry = logger.log(
            level = "INFO",
            tag = "HARDWARE",
            message = "Device info acquired",
            metadata = meta
        )
        assertEquals("INFO", entry.level)
        assertNotNull(entry.metadata)
        assertEquals("Pixel 7", entry.metadata?.get("device"))
        assertEquals("85", entry.metadata?.get("battery_pct"))
    }

    @Test
    fun testLogErrorWithWindowsFilePathProducesParseableJson() {
        val path = "C:\\users\\test\\file.log"
        val error = RuntimeException("File error at $path")
        val entry = logger.logError(
            tag = "FILE_SYSTEM",
            error = error
        )
        assertNotNull(entry.payloadJson)
        val payload = JSONObject(entry.payloadJson!!)
        assertEquals("File error at C:\\users\\test\\file.log", payload.getString("error_message"))
        assertEquals("java.lang.RuntimeException", payload.getString("error_class"))

        val readBack = logger.readLogLines()
        assertEquals(1, readBack.size)
        val readPayload = JSONObject(readBack[0].payloadJson!!)
        assertEquals("File error at C:\\users\\test\\file.log", readPayload.getString("error_message"))
    }

    @Test
    fun testDiagnosticsViewModelWarningFilterMatchesWarnAndWarning() = runTest {
        val testDispatcher = StandardTestDispatcher(testScheduler)
        val testScope = CoroutineScope(testDispatcher)

        logger.logEvent(tag = "NET", level = "WARN", message = "Network high latency")
        logger.logEvent(tag = "MEM", level = "WARNING", message = "Memory threshold reached")
        logger.logEvent(tag = "INFO", level = "INFO", message = "Normal heartbeat")

        val viewModel = DiagnosticsViewModel(logger = logger, ioDispatcher = testDispatcher, customScope = testScope)
        testScheduler.advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals(3, state.totalEntriesCount)
        assertEquals(2, state.warnCount)
        assertEquals(1, state.infoCount)

        viewModel.setFilter("WARN")
        val filteredState = viewModel.uiState.value
        assertEquals("WARN", filteredState.selectedFilter)
        assertEquals(2, filteredState.filteredEntries.size)
        val levels = filteredState.filteredEntries.map { it.level.uppercase() }.toSet()
        assertTrue(levels.contains("WARN"))
        assertTrue(levels.contains("WARNING"))

        viewModel.setFilter("WARNING")
        val filteredStateWarning = viewModel.uiState.value
        assertEquals("WARNING", filteredStateWarning.selectedFilter)
        assertEquals(2, filteredStateWarning.filteredEntries.size)
    }
}
