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

    val PRESET_USB_ADB = NetworkPreset("usb_adb", "USB adb reverse", "127.0.0.1", 8000, true)
    val PRESET_EMULATOR = NetworkPreset("emulator", "Android Emulator", "10.0.2.2", 8000, false)
    val PRESET_CUSTOM_LAN = NetworkPreset("custom_lan", "Custom LAN IP", "192.168.1.100", 8000, false)
    val PRESETS = listOf(PRESET_USB_ADB, PRESET_EMULATOR, PRESET_CUSTOM_LAN)

    val json: Json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
        encodeDefaults = true
    }

    private val dynamicHostInterceptor = Interceptor { chain ->
        val originalRequest = chain.request()
        val originalUrl = originalRequest.url
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
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
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
        val withScheme = if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
            trimmed
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

    fun setHostAndPort(host: String, port: Int) {
        currentHost = host.trim()
        currentPort = port
    }

    fun getBaseUrl(): String {
        return "$currentScheme://$currentHost:$currentPort"
    }

    fun getWsScanUrl(): String {
        val wsScheme = if (currentScheme == "https") "wss" else "ws"
        return "$wsScheme://$currentHost:$currentPort/ws/scan"
    }
}
