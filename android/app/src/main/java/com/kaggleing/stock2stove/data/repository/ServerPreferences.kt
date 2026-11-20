package com.kaggleing.stock2stove.data.repository

import android.content.Context
import android.content.SharedPreferences

interface ServerConfigStore {
    fun loadHost(): String
    fun loadPort(): Int
    fun loadPresetId(): String
    fun loadScheme(): String = "http"
    fun saveServerConfig(host: String, port: Int, presetId: String)
    fun saveServerConfig(scheme: String, host: String, port: Int, presetId: String) {
        saveServerConfig(host, port, presetId)
    }
}

class InMemoryServerConfigStore(
    initialHost: String = "192.168.1.100",
    initialPort: Int = 8000,
    initialPresetId: String = "custom_lan",
    initialScheme: String = "http"
) : ServerConfigStore {
    private var host: String = initialHost
    private var port: Int = initialPort
    private var presetId: String = initialPresetId
    private var scheme: String = if (initialScheme == "https") "https" else "http"

    override fun loadHost(): String {
        return host
    }

    override fun loadPort(): Int {
        return port
    }

    override fun loadPresetId(): String {
        return presetId
    }

    override fun loadScheme(): String {
        return scheme
    }

    override fun saveServerConfig(host: String, port: Int, presetId: String) {
        this.host = host
        this.port = port
        this.presetId = presetId
    }

    override fun saveServerConfig(scheme: String, host: String, port: Int, presetId: String) {
        this.scheme = if (scheme.equals("https", ignoreCase = true)) "https" else "http"
        this.host = host
        this.port = port
        this.presetId = presetId
    }
}

class SharedPrefsServerConfigStore(context: Context) : ServerConfigStore {
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    override fun loadHost(): String {
        val stored = prefs.getString(KEY_HOST, DEFAULT_HOST)
        if (stored.isNullOrBlank()) {
            return DEFAULT_HOST
        }
        return stored.trim()
    }

    override fun loadPort(): Int {
        val stored = prefs.getInt(KEY_PORT, DEFAULT_PORT)
        if (stored in 1..65535) {
            return stored
        }
        return if (loadScheme() == "https") 443 else DEFAULT_PORT
    }

    override fun loadPresetId(): String {
        val stored = prefs.getString(KEY_PRESET_ID, DEFAULT_PRESET_ID)
        if (stored.isNullOrBlank()) {
            return DEFAULT_PRESET_ID
        }
        return stored
    }

    override fun loadScheme(): String {
        val stored = prefs.getString(KEY_SCHEME, DEFAULT_SCHEME)
        return if (stored.equals("https", ignoreCase = true)) "https" else "http"
    }

    override fun saveServerConfig(host: String, port: Int, presetId: String) {
        saveServerConfig(loadScheme(), host, port, presetId)
    }

    override fun saveServerConfig(scheme: String, host: String, port: Int, presetId: String) {
        val cleanScheme = if (scheme.equals("https", ignoreCase = true)) "https" else "http"
        val cleanHost = host.trim()
        val fallbackPort = if (cleanScheme == "https") 443 else DEFAULT_PORT
        val validPort = if (port in 1..65535) port else fallbackPort
        val editor = prefs.edit()
        editor.putString(KEY_SCHEME, cleanScheme)
        editor.putString(KEY_HOST, cleanHost)
        editor.putInt(KEY_PORT, validPort)
        editor.putString(KEY_PRESET_ID, presetId)
        editor.apply()
    }

    companion object {
        private const val PREFS_NAME = "stock2stove_server_prefs"
        private const val KEY_SCHEME = "server_scheme"
        private const val KEY_HOST = "server_host"
        private const val KEY_PORT = "server_port"
        private const val KEY_PRESET_ID = "server_preset_id"
        private const val DEFAULT_SCHEME = "http"
        private const val DEFAULT_HOST = "192.168.1.100"
        private const val DEFAULT_PORT = 8000
        private const val DEFAULT_PRESET_ID = "custom_lan"
    }
}

data class ParsedServerAddress(
    val scheme: String,
    val host: String,
    val port: Int,
    val hasExplicitScheme: Boolean,
    val hasExplicitPort: Boolean
)

object ServerAddressParser {
    fun parseAddress(
        rawInput: String,
        fallbackScheme: String = "http",
        fallbackPort: Int = 8000
    ): ParsedServerAddress {
        var cleaned = rawInput.trim()
        val lower = cleaned.lowercase()
        var resolvedScheme = if (fallbackScheme.equals("https", ignoreCase = true)) "https" else "http"
        var hasExplicitScheme = false

        if (lower.startsWith("http://")) {
            cleaned = cleaned.substring(7)
            resolvedScheme = "http"
            hasExplicitScheme = true
        } else if (lower.startsWith("https://")) {
            cleaned = cleaned.substring(8)
            resolvedScheme = "https"
            hasExplicitScheme = true
        } else if (lower.startsWith("ws://")) {
            cleaned = cleaned.substring(5)
            resolvedScheme = "http"
            hasExplicitScheme = true
        } else if (lower.startsWith("wss://")) {
            cleaned = cleaned.substring(6)
            resolvedScheme = "https"
            hasExplicitScheme = true
        }

        val slashIndex = cleaned.indexOf('/')
        if (slashIndex >= 0) {
            cleaned = cleaned.substring(0, slashIndex)
        }

        var resolvedHost = cleaned.trim()
        var hasExplicitPort = false
        var parsedPortValue: Int? = null

        val colonIndex = resolvedHost.indexOf(':')
        if (colonIndex >= 0) {
            val hostPart = resolvedHost.substring(0, colonIndex).trim()
            val portPart = resolvedHost.substring(colonIndex + 1).trim()
            val parsedPort = portPart.toIntOrNull()
            if (parsedPort != null && parsedPort in 1..65535) {
                parsedPortValue = parsedPort
                hasExplicitPort = true
            }
            resolvedHost = hostPart
        }

        val lowerHost = resolvedHost.lowercase()
        if (!hasExplicitScheme && (lowerHost.endsWith(".trycloudflare.com") || lowerHost.endsWith(".cfargotunnel.com"))) {
            resolvedScheme = "https"
        }

        val resolvedPort = if (parsedPortValue != null) {
            parsedPortValue
        } else if (resolvedScheme == "https") {
            if (fallbackScheme.equals("https", ignoreCase = true) && fallbackPort in 1..65535) {
                fallbackPort
            } else {
                443
            }
        } else {
            if (fallbackScheme.equals("https", ignoreCase = true) && fallbackPort == 443) {
                8000
            } else if (fallbackPort in 1..65535) {
                fallbackPort
            } else {
                8000
            }
        }

        return ParsedServerAddress(
            scheme = resolvedScheme,
            host = resolvedHost,
            port = resolvedPort,
            hasExplicitScheme = hasExplicitScheme,
            hasExplicitPort = hasExplicitPort
        )
    }

    fun parseHostAndPort(rawInput: String, fallbackPort: Int): Pair<String, Int> {
        val parsed = parseAddress(rawInput, "http", fallbackPort)
        return Pair(parsed.host, parsed.port)
    }

    fun buildBaseUrl(scheme: String, host: String, port: Int): String {
        val cleanScheme = if (scheme.equals("https", ignoreCase = true)) "https" else "http"
        val cleanHost = host.trim()
        val defaultPort = if (cleanScheme == "https") 443 else 8000
        val validPort = if (port in 1..65535) port else defaultPort
        val isDefaultHttpPort = (cleanScheme == "https" && validPort == 443) || (cleanScheme == "http" && validPort == 80)
        return if (isDefaultHttpPort) {
            "$cleanScheme://$cleanHost"
        } else {
            "$cleanScheme://$cleanHost:$validPort"
        }
    }

    fun buildHttpBaseUrl(host: String, port: Int): String {
        return buildBaseUrl("http", host, port)
    }
}
