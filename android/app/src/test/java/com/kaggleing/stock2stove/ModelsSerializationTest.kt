package com.kaggleing.stock2stove

import com.kaggleing.stock2stove.data.api.NetworkClientProvider
import com.kaggleing.stock2stove.data.model.ConfirmedItem
import com.kaggleing.stock2stove.data.model.DetectionItem
import com.kaggleing.stock2stove.data.model.DetectorDefenseMode
import com.kaggleing.stock2stove.data.model.HealthResponse
import com.kaggleing.stock2stove.data.model.InventoryItemRequest
import com.kaggleing.stock2stove.data.model.NetworkPreset
import com.kaggleing.stock2stove.data.model.RecipeGenerateRequest
import com.kaggleing.stock2stove.data.model.RecipeGenerateResponse
import com.kaggleing.stock2stove.data.model.ScanFrameRequest
import com.kaggleing.stock2stove.data.model.ScanMode
import com.kaggleing.stock2stove.data.model.ScanResponse
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelsSerializationTest {
    private val json = NetworkClientProvider.json

    @Test
    fun testHealthResponseDeserialization() {
        val payload = """
        {
          "status": "ok",
          "service": "stock2stove-inference-server",
          "model_status": {
            "ultralytics_installed": true,
            "fine_tuned_pt_available": true,
            "fine_tuned_onnx_available": false,
            "yolo_world_available": false,
            "active_weights": "models/yolov8n_stock2stove.pt",
            "yolov8n_loaded": true
          },
          "hardware": {
            "os": "Windows",
            "processor": "AMD Ryzen 7",
            "cpu_threads": 16,
            "opencv_version": "4.10.0",
            "device": "cpu"
          },
          "supported_detector_modes": ["yolov8", "orb_baseline", "sift_baseline"],
          "supported_classes": [
            {
              "id": 0,
              "class_id": "pasta_noodles",
              "display_name": "Pasta & Noodles",
              "storage_zone": "Pantry",
              "prompts": ["pasta box"]
            }
          ]
        }
        """.trimIndent()

        val parsed = json.decodeFromString<HealthResponse>(payload)
        assertEquals("ok", parsed.status)
        assertEquals("stock2stove-inference-server", parsed.service)
        assertTrue(parsed.model_status.ultralytics_installed)
        assertTrue(parsed.model_status.yolov8n_loaded)
        assertEquals("models/yolov8n_stock2stove.pt", parsed.model_status.active_weights)
        assertEquals("cpu", parsed.hardware.device)
        assertEquals(16, parsed.hardware.cpu_threads)
        assertEquals(3, parsed.supported_detector_modes.size)
        assertEquals(1, parsed.supported_classes.size)
        assertEquals("pasta_noodles", parsed.supported_classes[0].class_id)
    }

    @Test
    fun testHealthResponseMissingFieldsFallback() {
        val payload = """{"status":"healthy"}"""
        val parsed = json.decodeFromString<HealthResponse>(payload)
        assertEquals("healthy", parsed.status)
        assertEquals("", parsed.service)
        assertEquals("cpu", parsed.hardware.device)
        assertTrue(parsed.supported_classes.isEmpty())
    }

    @Test
    fun testScanFrameRequestSerialization() {
        val req = ScanFrameRequest(
            frame_id = 42L,
            detector_mode = "yolov8",
            enable_clahe = true,
            conf_threshold = 0.35f,
            image_b64 = "dGVzdF9pbWFnZQ==",
            session_id = "session_001",
            reset_session = false
        )
        val serialized = json.encodeToString(req)
        assertTrue(serialized.contains("\"frame_id\":42"))
        assertTrue(serialized.contains("\"detector_mode\":\"yolov8\""))

        val deserialized = json.decodeFromString<ScanFrameRequest>(serialized)
        assertEquals(42L, deserialized.frame_id)
        assertEquals(0.35f, deserialized.conf_threshold)
        assertEquals("session_001", deserialized.session_id)
    }

    @Test
    fun testScanResponseDeserializationWithPolygon() {
        val payload = """
        {
          "frame_id": 105,
          "detector_mode": "orb_baseline",
          "inference_ms": 18.4,
          "fps": 54.3,
          "diagnostics": {
            "edge_density_pct": 3.42,
            "orb_keypoints": 850,
            "clahe_applied": true
          },
          "detections": [
            {
              "class_id": "canned_tomato",
              "display_name": "Canned Tomatoes",
              "confidence": 0.92,
              "bbox_norm": [0.1, 0.2, 0.4, 0.6],
              "polygon_norm": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.8]],
              "temporally_confirmed": true
            }
          ],
          "confirmed_inventory": [
            {
              "class_id": "canned_tomato",
              "display_name": "Canned Tomatoes",
              "count": 2,
              "max_conf": 0.95
            }
          ]
        }
        """.trimIndent()

        val resp = json.decodeFromString<ScanResponse>(payload)
        assertEquals(105L, resp.frame_id)
        assertEquals("orb_baseline", resp.detector_mode)
        assertEquals(18.4f, resp.inference_ms)
        assertEquals(1, resp.detections.size)
        val det = resp.detections[0]
        assertEquals("canned_tomato", det.class_id)
        assertTrue(det.temporally_confirmed)
        assertNotNull(det.polygon_norm)
        assertEquals(4, det.polygon_norm?.size)
        assertEquals(1, resp.confirmed_inventory.size)
        assertEquals(2, resp.confirmed_inventory[0].count)
    }

    @Test
    fun testRecipeRequestAndResponseSerialization() {
        val req = RecipeGenerateRequest(
            confirmed_inventory = listOf(
                InventoryItemRequest("canned_tomato", "Canned Tomatoes", 2, "shelf"),
                InventoryItemRequest("pasta_noodles", "Pasta & Noodles", 1, "shelf")
            ),
            timeout_sec = 3.0f
        )
        val reqStr = json.encodeToString(req)
        assertTrue(reqStr.contains("canned_tomato"))
        assertTrue(reqStr.contains("pasta_noodles"))

        val responseJson = """
        {
          "source": "gemini-1.5-flash",
          "cached": false,
          "latency_ms": 1150.0,
          "inventory_used": ["canned_tomato", "pasta_noodles"],
          "recipes": [
            {
              "id": "recipe_one_pot_pasta",
              "title": "One-Pot Tomato Pasta",
              "prep_time_mins": 15,
              "difficulty": "Easy",
              "storage_zones_used": ["shelf"],
              "matched_ingredients": ["Canned Tomatoes", "Pasta & Noodles"],
              "additional_staples": ["Salt", "Olive Oil"],
              "steps": ["Boil water", "Add pasta and tomatoes", "Simmer 10 minutes"],
              "zero_waste_tip": "Use pasta water to loosen thick sauce."
            }
          ]
        }
        """.trimIndent()

        val resp = json.decodeFromString<RecipeGenerateResponse>(responseJson)
        assertEquals("gemini-1.5-flash", resp.source)
        assertFalse(resp.cached)
        assertEquals(1, resp.recipes.size)
        val card = resp.recipes[0]
        assertEquals("One-Pot Tomato Pasta", card.title)
        assertEquals(15, card.prep_time_mins)
        assertEquals(3, card.steps.size)
        assertEquals("Use pasta water to loosen thick sauce.", card.zero_waste_tip)
    }

    @Test
    fun testNetworkPresetUrlGenerators() {
        val preset = NetworkPreset("usb_adb", "USB adb reverse", "127.0.0.1", 8000, true)
        assertEquals("http://127.0.0.1:8000", preset.getHttpBaseUrl())
        assertEquals("ws://127.0.0.1:8000/ws/scan", preset.getWsScanUrl())
        assertEquals("http://127.0.0.1:8000/api/v1/health", preset.getHealthUrl())
        assertEquals("http://127.0.0.1:8000/api/v1/recipes/generate", preset.getRecipeUrl())
        assertTrue(preset.isDefault)
    }

    @Test
    fun testScanModeAndDefenseEnums() {
        val pantry = ScanMode.PANTRY_SHELF
        val fridge = ScanMode.REFRIGERATOR
        assertEquals("pantry", pantry.contextKey)
        assertEquals("fridge", fridge.contextKey)

        val yolo = DetectorDefenseMode.YOLO_V8N
        val orb = DetectorDefenseMode.ORB_BASELINE
        val sift = DetectorDefenseMode.SIFT_BASELINE
        assertEquals("yolov8", yolo.wireMode)
        assertEquals("orb_baseline", orb.wireMode)
        assertEquals("sift_baseline", sift.wireMode)
    }
}
