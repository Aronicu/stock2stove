package com.kaggleing.stock2stove.data.api

import com.kaggleing.stock2stove.data.model.NetworkPreset
import retrofit2.converter.kotlinx.serialization.asConverterFactory
import kotlinx.serialization.json.Json
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit

object NetworkClientProvider {
    @Volatile
    private var currentHost: String = "127.0.0.1"

    @Volatile
    private var currentPort: Int = 8000

    @Volatile
    private var currentScheme: String = "http"

    val PRESET_USB_ADB = NetworkPreset("usb_adb", "USB adb reverse", "127.0.0.1", 8000, true, "http")
    val PRESET_EMULATOR = NetworkPreset("emulator", "Android Emulator", "10.0.2.2", 8000, false, "http")
    val PRESET_CUSTOM_LAN = NetworkPreset("custom_lan", "Custom LAN IP", "192.168.1.100", 8000, false, "http")
    val PRESET_HTTPS_TUNNEL = NetworkPreset("https_tunnel", "HTTPS Tunnel", "", 443, false, "https")
    val PRESETS = listOf(PRESET_HTTPS_TUNNEL, PRESET_USB_ADB, PRESET_EMULATOR, PRESET_CUSTOM_LAN)

    val json: Json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
        encodeDefaults = true
    }

    private val dynamicHostInterceptor = Interceptor { chain ->
        val originalRequest = chain.request()
        val originalUrl = originalRequest.url
        if (currentHost.isBlank()) {
            return@Interceptor chain.proceed(originalRequest)
        }
        val updatedUrl = originalUrl.newBuilder()
            .scheme(currentScheme)
            .host(currentHost)
            .port(currentPort)
            .build()
        val updatedRequest = originalRequest.newBuilder()
            .url(updatedUrl)
            .build()
        chain.proceed(updatedRequest)
    }

    val okHttpClient: OkHttpClient = OkHttpClient.Builder()
        .addInterceptor(dynamicHostInterceptor)
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    val apiService: InferenceApiService by lazy {
        val contentType = "application/json".toMediaType()
        val retrofit = Retrofit.Builder()
            .baseUrl("http://127.0.0.1:8000/")
            .client(okHttpClient)
            .addConverterFactory(json.asConverterFactory(contentType))
            .build()
        retrofit.create(InferenceApiService::class.java)
    }

    fun setBaseUrl(url: String) {
        val trimmed = url.trim()
        val lower = trimmed.lowercase()
        val withScheme = if (lower.startsWith("http://") || lower.startsWith("https://")) {
            trimmed
        } else if (lower.startsWith("ws://")) {
            "http://" + trimmed.substring(5)
        } else if (lower.startsWith("wss://")) {
            "https://" + trimmed.substring(6)
        } else {
            "http://$trimmed"
        }
        val parsed = withScheme.toHttpUrlOrNull()
        if (parsed != null) {
            currentScheme = parsed.scheme
            currentHost = parsed.host
            currentPort = parsed.port
        }
    }

    fun setServerConfig(scheme: String, host: String, port: Int) {
        val cleanScheme = if (scheme.equals("https", ignoreCase = true)) "https" else "http"
        val defaultPort = if (cleanScheme == "https") 443 else 8000
        currentScheme = cleanScheme
        currentHost = host.trim()
        currentPort = if (port in 1..65535) port else defaultPort
    }

    fun setHostAndPort(host: String, port: Int) {
        currentHost = host.trim()
        currentPort = port
    }

    fun getBaseUrl(): String {
        val isDefaultPort = (currentScheme == "https" && currentPort == 443) || (currentScheme == "http" && currentPort == 80)
        return if (isDefaultPort) {
            "$currentScheme://$currentHost"
        } else {
            "$currentScheme://$currentHost:$currentPort"
        }
    }

    fun getWsScanUrl(): String {
        val wsScheme = if (currentScheme == "https") "wss" else "ws"
        val isDefaultPort = (currentScheme == "https" && currentPort == 443) || (currentScheme == "http" && currentPort == 80)
        return if (isDefaultPort) {
            "$wsScheme://$currentHost/ws/scan"
        } else {
            "$wsScheme://$currentHost:$currentPort/ws/scan"
        }
    }
}
