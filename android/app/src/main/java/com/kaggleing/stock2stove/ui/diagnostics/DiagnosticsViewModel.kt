package com.kaggleing.stock2stove.ui.diagnostics

import android.content.Context
import android.content.Intent
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaggleing.stock2stove.data.repository.AlphaTelemetryLogger
import com.kaggleing.stock2stove.data.repository.LogEntry
import com.kaggleing.stock2stove.data.repository.SharePayload
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class DiagnosticsUiState(
    val isLoading: Boolean = false,
    val entries: List<LogEntry> = emptyList(),
    val filteredEntries: List<LogEntry> = emptyList(),
    val selectedFilter: String? = null,
    val totalEntriesCount: Int = 0,
    val totalSizeBytes: Long = 0L,
    val infoCount: Int = 0,
    val warnCount: Int = 0,
    val errorCount: Int = 0,
    val telemetryCount: Int = 0,
    val activeLogFilePath: String = ""
)

class DiagnosticsViewModel(
    val logger: AlphaTelemetryLogger = AlphaTelemetryLogger.getDefault(),
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
    customScope: CoroutineScope? = null
) : ViewModel() {

    private val scope: CoroutineScope = customScope ?: viewModelScope
    private val _uiState = MutableStateFlow(DiagnosticsUiState())
    val uiState: StateFlow<DiagnosticsUiState> = _uiState.asStateFlow()

    init {
        loadLogs()
    }

    fun loadLogs() {
        _uiState.update { it.copy(isLoading = true) }
        scope.launch(ioDispatcher) {
            val rawEntries = logger.readLogLines()
            val allEntries = rawEntries.sortedByDescending { it.timestamp }
            val logFiles = logger.getLogFiles()
            var totalBytes = 0L
            for (file in logFiles) {
                totalBytes += file.length()
            }
            if (totalBytes == 0L) {
                totalBytes = logger.getLogFileSize()
            }

            var info = 0
            var warn = 0
            var error = 0
            var telemetry = 0
            for (entry in allEntries) {
                when (entry.level.uppercase()) {
                    "INFO" -> info++
                    "WARN", "WARNING" -> warn++
                    "ERROR" -> error++
                    "TELEMETRY" -> telemetry++
                }
            }

            val currentFilter = _uiState.value.selectedFilter
            val filtered = applyFilter(allEntries, currentFilter)

            _uiState.update {
                it.copy(
                    isLoading = false,
                    entries = allEntries,
                    filteredEntries = filtered,
                    totalEntriesCount = allEntries.size,
                    totalSizeBytes = totalBytes,
                    infoCount = info,
                    warnCount = warn,
                    errorCount = error,
                    telemetryCount = telemetry,
                    activeLogFilePath = logger.getActiveLogFile().absolutePath
                )
            }
        }
    }

    fun loadLogsSync() {
        val rawEntries = logger.readLogLines()
        val allEntries = rawEntries.sortedByDescending { it.timestamp }
        val logFiles = logger.getLogFiles()
        var totalBytes = 0L
        for (file in logFiles) {
            totalBytes += file.length()
        }
        if (totalBytes == 0L) {
            totalBytes = logger.getLogFileSize()
        }

        var info = 0
        var warn = 0
        var error = 0
        var telemetry = 0
        for (entry in allEntries) {
            when (entry.level.uppercase()) {
                "INFO" -> info++
                "WARN", "WARNING" -> warn++
                "ERROR" -> error++
                "TELEMETRY" -> telemetry++
            }
        }

        val currentFilter = _uiState.value.selectedFilter
        val filtered = applyFilter(allEntries, currentFilter)

        _uiState.update {
            it.copy(
                isLoading = false,
                entries = allEntries,
                filteredEntries = filtered,
                totalEntriesCount = allEntries.size,
                totalSizeBytes = totalBytes,
                infoCount = info,
                warnCount = warn,
                errorCount = error,
                telemetryCount = telemetry,
                activeLogFilePath = logger.getActiveLogFile().absolutePath
            )
        }
    }

    fun setFilter(level: String?) {
        val normalized = if (level == null || level.equals("ALL", ignoreCase = true)) {
            null
        } else {
            level.uppercase()
        }
        val allEntries = _uiState.value.entries
        val filtered = applyFilter(allEntries, normalized)
        _uiState.update {
            it.copy(
                selectedFilter = normalized,
                filteredEntries = filtered
            )
        }
    }

    fun clearLogs() {
        scope.launch(ioDispatcher) {
            logger.clearLogs()
            loadLogs()
        }
    }

    fun createShareIntent(context: Context): Intent {
        return logger.createShareIntent(context)
    }

    fun getSharePayload(): SharePayload {
        return logger.getSharePayload()
    }

    private fun applyFilter(entries: List<LogEntry>, filter: String?): List<LogEntry> {
        if (filter == null) {
            return entries
        }
        if (filter.equals("WARN", ignoreCase = true) || filter.equals("WARNING", ignoreCase = true)) {
            return entries.filter {
                it.level.equals("WARN", ignoreCase = true) || it.level.equals("WARNING", ignoreCase = true)
            }
        }
        return entries.filter { it.level.equals(filter, ignoreCase = true) }
    }
}
