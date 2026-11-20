package com.kaggleing.stock2stove.data.model

import kotlinx.serialization.Serializable

@Serializable
data class HealthResponse(
    val status: String = "",
    val service: String = "",
    val model_status: ModelStatus = ModelStatus(),
    val hardware: HardwareInfo = HardwareInfo(),
    val supported_detector_modes: List<String> = emptyList(),
    val supported_classes: List<ClassInfo> = emptyList()
)

@Serializable
data class GeminiStatus(
    val api_key_configured: Boolean = false,
    val vision_model: String = "",
    val cache_entries: Int = 0
)

@Serializable
data class ModelStatus(
    val ultralytics_installed: Boolean = false,
    val fine_tuned_pt_available: Boolean = false,
    val fine_tuned_onnx_available: Boolean = false,
    val yolo_world_available: Boolean = false,
    val active_weights: String = "",
    val yolov8n_loaded: Boolean = false,
    val orb_ready: Boolean = false,
    val sift_ready: Boolean = false,
    val weights_path: String = "",
    val gemini_status: GeminiStatus = GeminiStatus()
)

@Serializable
data class HardwareInfo(
    val os: String = "",
    val os_release: String = "",
    val machine: String = "",
    val processor: String = "",
    val cpu_threads: Int = 0,
    val opencv_version: String = "",
    val device: String = "cpu",
    val cuda_available: Boolean = false,
    val gpu_name: String = "",
    val vram_total_mb: Double = 0.0,
    val vram_allocated_mb: Double = 0.0
)

@Serializable
data class ClassInfo(
    val id: Int = 0,
    val class_id: String = "",
    val display_name: String = "",
    val storage_zone: String = "",
    val default_zone: String = "",
    val prompts: List<String> = emptyList()
)

@Serializable
data class ScanFrameRequest(
    val frame_id: Long = 0L,
    val detector_mode: String = "gemini_vision",
    val enable_clahe: Boolean = true,
    val conf_threshold: Float = 0.5f,
    val image_b64: String? = null,
    val session_id: String = "default",
    val reset_session: Boolean = false
)

@Serializable
data class ScanResponse(
    val frame_id: Long = 0L,
    val detector_mode: String = "gemini_vision",
    val inference_ms: Float = 0.0f,
    val fps: Float = 0.0f,
    val diagnostics: DiagnosticsData = DiagnosticsData(),
    val detections: List<DetectionItem> = emptyList(),
    val confirmed_inventory: List<ConfirmedItem> = emptyList()
)

@Serializable
data class DiagnosticsData(
    val edge_density_pct: Float = 0.0f,
    val edge_density: Float = 0.0f,
    val orb_keypoints: Int = 0,
    val keypoints_detected: Int = 0,
    val inliers: Int = 0,
    val ransac_inliers: Int = 0,
    val good_matches: Int = 0,
    val clahe_applied: Boolean = false
)

@Serializable
data class DetectionItem(
    val class_id: String = "",
    val display_name: String = "",
    val confidence: Float = 0.0f,
    val bbox_norm: List<Float> = emptyList(),
    val polygon_norm: List<List<Float>>? = null,
    val temporally_confirmed: Boolean = false,
    val crop_jpeg_b64: String? = null
)

@Serializable
data class ConfirmedItem(
    val class_id: String = "",
    val display_name: String = "",
    val count: Int = 1,
    val max_conf: Float = 0.0f,
    val crop_jpeg_b64: String? = null
)

@Serializable
data class RecipeGenerateRequest(
    val confirmed_inventory: List<InventoryItemRequest> = emptyList(),
    val timeout_sec: Float = 2.5f,
    val api_key: String? = null
)

@Serializable
data class InventoryItemRequest(
    val class_id: String = "",
    val display_name: String = "",
    val count: Int = 1,
    val storage_zone: String = "shelf",
    val crop_jpeg_b64: String? = null
)

@Serializable
data class RecipeGenerateResponse(
    val source: String = "",
    val cached: Boolean = false,
    val latency_ms: Float = 0.0f,
    val inventory_used: List<String> = emptyList(),
    val recipes: List<RecipeCard> = emptyList()
)

@Serializable
data class RecipeCard(
    val id: String = "",
    val title: String = "",
    val prep_time_mins: Int = 0,
    val difficulty: String = "",
    val storage_zones_used: List<String> = emptyList(),
    val matched_ingredients: List<String> = emptyList(),
    val additional_staples: List<String> = emptyList(),
    val steps: List<String> = emptyList(),
    val zero_waste_tip: String = ""
)

@Serializable
data class NetworkPreset(
    val presetId: String = "",
    val label: String = "",
    val host: String = "127.0.0.1",
    val port: Int = 8000,
    val isDefault: Boolean = false
) {
    fun getHttpBaseUrl(): String = "http://$host:$port"
    fun getWsScanUrl(): String = "ws://$host:$port/ws/scan"
    fun getHealthUrl(): String = "http://$host:$port/api/v1/health"
    fun getRecipeUrl(): String = "http://$host:$port/api/v1/recipes/generate"
}
