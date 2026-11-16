package com.kaggleing.stock2stove.data.repository

import android.content.Context
import android.content.Intent
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    TELEMETRY
}

@Serializable
data class LogEntry(
    val timestamp: Long = System.currentTimeMillis(),
    val sessionId: String = "",
    val level: String = LogLevel.INFO.name,
    val tag: String = "",
    val message: String = "",
    val payloadJson: String? = null,
    val metadata: Map<String, String>? = null
) {
    val severity: String get() = level
    val eventType: String get() = tag

    fun getFormattedTime(): String {
        val sdf = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
        return sdf.format(Date(timestamp))
    }

    fun getFormattedDateTime(): String {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
        return sdf.format(Date(timestamp))
    }

    fun toFormattedLogLine(): String {
        val payloadStr = if (payloadJson != null) " | $payloadJson" else ""
        return "[${getFormattedDateTime()}] [$level] [$tag] $message$payloadStr"
    }
}

data class SharePayload(
    val action: String,
    val type: String,
    val subject: String,
    val text: String,
    val lineCount: Int
)

class AlphaTelemetryLogger(
    val baseDirectory: File,
    val sessionId: String = UUID.randomUUID().toString().take(8),
    val canonicalFile: File? = null
) {
    constructor(context: Context) : this(
        File(context.filesDir, "logs"),
        UUID.randomUUID().toString().take(8),
        File(context.filesDir, "stock2stove_telemetry.jsonl")
    )

    private val lock = Any()
    private val maxFileSizeBytes: Long = 4 * 1024 * 1024L

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
        isLenient = true
    }

    private val activeLogFile: File = if (baseDirectory.name.endsWith(".jsonl")) {
        baseDirectory
    } else {
        File(baseDirectory, "session_${sessionId}.jsonl")
    }

    fun getActiveLogFile(): File = activeLogFile

    fun logEvent(
        tag: String,
        level: String,
        message: String,
        payloadJson: String? = null
    ): LogEntry {
        val entry = LogEntry(
            timestamp = System.currentTimeMillis(),
            sessionId = sessionId,
            level = level.uppercase(),
            tag = tag,
            message = message,
            payloadJson = payloadJson,
            metadata = null
        )
        writeEntry(entry)
        return entry
    }

    fun logEvent(
        level: LogLevel,
        tag: String,
        message: String,
        payloadJson: String? = null
    ): LogEntry {
        return logEvent(
            tag = tag,
            level = level.name,
            message = message,
            payloadJson = payloadJson
        )
    }

    fun log(
        level: String,
        tag: String,
        message: String,
        metadata: Map<String, Any?> = emptyMap()
    ): LogEntry {
        val strMeta: Map<String, String>? = if (metadata.isEmpty()) {
            null
        } else {
            val converted = mutableMapOf<String, String>()
            for ((key, value) in metadata) {
                if (value != null) {
                    converted[key] = value.toString()
                }
            }
            if (converted.isEmpty()) null else converted
        }

        val entry = LogEntry(
            timestamp = System.currentTimeMillis(),
            sessionId = sessionId,
            level = level.uppercase(),
            tag = tag,
            message = message,
            payloadJson = null,
            metadata = strMeta
        )
        writeEntry(entry)
        return entry
    }

    fun logTelemetry(
        frameId: Long,
        detectorMode: String,
        inferenceMs: Float,
        fps: Float,
        detectionsCount: Int
    ): LogEntry {
        val payload = "{\"frame_id\":$frameId,\"detector_mode\":\"$detectorMode\",\"inference_ms\":$inferenceMs,\"fps\":$fps,\"detections_count\":$detectionsCount}"
        return logEvent(
            tag = "TELEMETRY",
            level = LogLevel.TELEMETRY.name,
            message = "Frame #$frameId ($detectorMode): ${inferenceMs}ms, ${fps} FPS, $detectionsCount detections",
            payloadJson = payload
        )
    }

    fun logScanFrame(
        frameId: Long,
        latencyMs: Long,
        detectorMode: String,
        confirmedCount: Int,
        metadata: Map<String, Any?> = emptyMap()
    ): LogEntry {
        return logTelemetry(
            frameId = frameId,
            detectorMode = detectorMode,
            inferenceMs = latencyMs.toFloat(),
            fps = if (latencyMs > 0) 1000f / latencyMs else 0f,
            detectionsCount = confirmedCount
        )
    }

    fun logError(
        tag: String,
        error: Throwable,
        context: String? = null
    ): LogEntry {
        val sanitizedMsg = (error.message ?: "").replace("\\", "\\\\").replace("\"", "\\\"").replace("\r", " ").replace("\n", " ")
        val sanitizedClass = error.javaClass.name.replace("\\", "\\\\").replace("\"", "\\\"")
        val payload = "{\"error_class\":\"$sanitizedClass\",\"error_message\":\"$sanitizedMsg\"}"
        val formattedMsg = if (context != null) {
            "$context: ${error.message ?: error.javaClass.simpleName}"
        } else {
            error.message ?: error.javaClass.simpleName
        }
        return logEvent(
            tag = tag,
            level = LogLevel.ERROR.name,
            message = formattedMsg,
            payloadJson = payload
        )
    }

    fun logError(
        tag: String,
        message: String,
        throwable: Throwable? = null,
        metadata: Map<String, Any?> = emptyMap()
    ): LogEntry {
        val errorMeta = mutableMapOf<String, String>()
        for ((k, v) in metadata) {
            if (v != null) {
                errorMeta[k] = v.toString()
            }
        }
        if (throwable != null) {
            errorMeta["error_class"] = throwable.javaClass.simpleName
            errorMeta["error_message"] = throwable.message ?: ""
        }
        val entry = LogEntry(
            timestamp = System.currentTimeMillis(),
            sessionId = sessionId,
            level = LogLevel.ERROR.name,
            tag = tag,
            message = message,
            payloadJson = null,
            metadata = if (errorMeta.isEmpty()) null else errorMeta
        )
        writeEntry(entry)
        return entry
    }

    private fun writeEntry(entry: LogEntry) {
        val line = json.encodeToString(entry)
        synchronized(lock) {
            trimFileIfNeeded()
            activeLogFile.parentFile?.mkdirs()
            FileWriter(activeLogFile, true).use { fw ->
                fw.write(line + "\n")
            }
            if (canonicalFile != null) {
                try {
                    canonicalFile.parentFile?.mkdirs()
                    FileWriter(canonicalFile, true).use { cfw ->
                        cfw.write(line + "\n")
                    }
                } catch (_: Exception) {
                }
            }
        }
    }

    fun getLogFiles(): List<File> {
        synchronized(lock) {
            if (baseDirectory.name.endsWith(".jsonl")) {
                return if (baseDirectory.exists()) listOf(baseDirectory) else emptyList()
            }
            if (!baseDirectory.exists() || !baseDirectory.isDirectory) {
                return if (activeLogFile.exists()) listOf(activeLogFile) else emptyList()
            }
            val files = baseDirectory.listFiles { f ->
                f.isFile && f.name.endsWith(".jsonl")
            } ?: return emptyList()
            return files.sortedByDescending { it.lastModified() }
        }
    }

    fun readLogLines(file: File? = null): List<LogEntry> {
        val target = file ?: activeLogFile
        synchronized(lock) {
            if (!target.exists() || target.length() == 0L) {
                return emptyList()
            }
            val entries = mutableListOf<LogEntry>()
            target.forEachLine { rawLine ->
                val line = rawLine.trim()
                if (line.isNotEmpty()) {
                    try {
                        val parsed = json.decodeFromString<LogEntry>(line)
                        entries.add(parsed)
                    } catch (_: Exception) {
                    }
                }
            }
            return entries
        }
    }

    fun getLogEntries(limit: Int = 500): List<LogEntry> {
        val all = readLogLines()
        return if (all.size > limit) all.takeLast(limit) else all
    }

    fun getLogFileSize(): Long {
        synchronized(lock) {
            return if (activeLogFile.exists()) activeLogFile.length() else 0L
        }
    }

    fun clearLogs() {
        synchronized(lock) {
            if (baseDirectory.name.endsWith(".jsonl")) {
                if (baseDirectory.exists()) {
                    PrintWriter(baseDirectory).use { pw ->
                        pw.print("")
                    }
                }
                return
            }
            val files = getLogFiles()
            for (f in files) {
                try {
                    f.delete()
                } catch (_: Exception) {
                }
            }
            if (activeLogFile.exists()) {
                PrintWriter(activeLogFile).use { pw ->
                    pw.print("")
                }
            }
        }
    }

    fun exportLogsText(): String {
        synchronized(lock) {
            return if (activeLogFile.exists()) activeLogFile.readText() else ""
        }
    }

    fun exportToFile(targetFile: File): File {
        synchronized(lock) {
            targetFile.parentFile?.mkdirs()
            if (activeLogFile.exists()) {
                activeLogFile.copyTo(targetFile, overwrite = true)
            } else {
                targetFile.writeText("")
            }
            return targetFile
        }
    }

    fun getSharePayload(): SharePayload {
        val entries = readLogLines()
        val textContent = if (entries.isEmpty()) {
            "Stock-to-Stove Untethered Log Export\nSession: $sessionId\nNo log entries found."
        } else {
            val sb = StringBuilder()
            sb.append("Stock-to-Stove Untethered Log Export\n")
            sb.append("Session: ").append(sessionId).append("\n")
            sb.append("Total Entries: ").append(entries.size).append("\n\n")
            for (entry in entries) {
                sb.append(entry.toFormattedLogLine()).append("\n")
            }
            sb.toString()
        }
        return SharePayload(
            action = Intent.ACTION_SEND,
            type = "text/plain",
            subject = "Stock-to-Stove Alpha Diagnostics",
            text = textContent,
            lineCount = entries.size
        )
    }

    fun createShareIntent(context: Context): Intent {
        val payload = getSharePayload()
        val intent = Intent(payload.action).apply {
            type = payload.type
            putExtra(Intent.EXTRA_SUBJECT, payload.subject)
            putExtra(Intent.EXTRA_TEXT, payload.text)
        }
        return intent
    }

    private fun trimFileIfNeeded() {
        if (activeLogFile.exists() && activeLogFile.length() > maxFileSizeBytes) {
            try {
                val lines = activeLogFile.readLines()
                if (lines.size > 200) {
                    val retained = lines.takeLast(100)
                    activeLogFile.writeText(retained.joinToString("\n") + "\n")
                }
            } catch (_: Exception) {
            }
        }
    }

    companion object {
        @Volatile
        private var instance: AlphaTelemetryLogger? = null

        fun getInstance(context: Context): AlphaTelemetryLogger {
            return instance ?: synchronized(this) {
                instance ?: AlphaTelemetryLogger(context).also { instance = it }
            }
        }

        fun getDefault(baseDir: File? = null): AlphaTelemetryLogger {
            return instance ?: synchronized(this) {
                val dir = baseDir ?: File(System.getProperty("java.io.tmpdir"), "stock2stove_logs")
                instance ?: AlphaTelemetryLogger(dir).also { instance = it }
            }
        }

        fun createForTesting(baseDir: File, sessionId: String = "test_sess"): AlphaTelemetryLogger {
            return AlphaTelemetryLogger(baseDir, sessionId)
        }

        fun resetInstance() {
            synchronized(this) {
                instance = null
            }
        }
    }
}
