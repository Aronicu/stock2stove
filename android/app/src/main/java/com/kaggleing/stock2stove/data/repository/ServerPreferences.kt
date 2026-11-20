package com.kaggleing.stock2stove.data.repository

import android.content.Context
import android.content.SharedPreferences

interface ServerConfigStore {
    fun loadHost(): String
    fun loadPort(): Int
    fun loadPresetId(): String
    fun saveServerConfig(host: String, port: Int, presetId: String)
}

class InMemoryServerConfigStore(
    initialHost: String = "192.168.1.100",
    initialPort: Int = 8000,
    initialPresetId: String = "custom_lan"
) : ServerConfigStore {
    private var host: String = initialHost
    private var port: Int = initialPort
    private var presetId: String = initialPresetId

    override fun loadHost(): String {
        return host
    }

    override fun loadPort(): Int {
        return port
    }

    override fun loadPresetId(): String {
        return presetId
    }

    override fun saveServerConfig(host: String, port: Int, presetId: String) {
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
        return DEFAULT_PORT
    }

    override fun loadPresetId(): String {
        val stored = prefs.getString(KEY_PRESET_ID, DEFAULT_PRESET_ID)
        if (stored.isNullOrBlank()) {
            return DEFAULT_PRESET_ID
        }
        return stored
    }

    override fun saveServerConfig(host: String, port: Int, presetId: String) {
        val cleanHost = host.trim()
        val validPort = if (port in 1..65535) port else DEFAULT_PORT
        val editor = prefs.edit()
        editor.putString(KEY_HOST, cleanHost)
        editor.putInt(KEY_PORT, validPort)
        editor.putString(KEY_PRESET_ID, presetId)
        editor.apply()
    }

    companion object {
        private const val PREFS_NAME = "stock2stove_server_prefs"
        private const val KEY_HOST = "server_host"
        private const val KEY_PORT = "server_port"
        private const val KEY_PRESET_ID = "server_preset_id"
        private const val DEFAULT_HOST = "192.168.1.100"
        private const val DEFAULT_PORT = 8000
        private const val DEFAULT_PRESET_ID = "custom_lan"
    }
}

object ServerAddressParser {
    fun parseHostAndPort(rawInput: String, fallbackPort: Int): Pair<String, Int> {
        var cleaned = rawInput.trim()
        val lower = cleaned.lowercase()
        if (lower.startsWith("http://")) {
            cleaned = cleaned.substring(7)
        } else if (lower.startsWith("https://")) {
            cleaned = cleaned.substring(8)
        } else if (lower.startsWith("ws://")) {
            cleaned = cleaned.substring(5)
        } else if (lower.startsWith("wss://")) {
            cleaned = cleaned.substring(6)
        }

        val slashIndex = cleaned.indexOf('/')
        if (slashIndex >= 0) {
            cleaned = cleaned.substring(0, slashIndex)
        }

        var resolvedHost = cleaned.trim()
        var resolvedPort = if (fallbackPort in 1..65535) fallbackPort else 8000

        val colonIndex = resolvedHost.indexOf(':')
        if (colonIndex >= 0) {
            val hostPart = resolvedHost.substring(0, colonIndex).trim()
            val portPart = resolvedHost.substring(colonIndex + 1).trim()
            val parsedPort = portPart.toIntOrNull()
            if (parsedPort != null && parsedPort in 1..65535) {
                resolvedPort = parsedPort
            }
            resolvedHost = hostPart
        }

        return Pair(resolvedHost, resolvedPort)
    }

    fun buildHttpBaseUrl(host: String, port: Int): String {
        val cleanHost = host.trim()
        val validPort = if (port in 1..65535) port else 8000
        return "http://$cleanHost:$validPort"
    }
}
