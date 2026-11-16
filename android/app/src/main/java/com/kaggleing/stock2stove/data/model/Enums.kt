package com.kaggleing.stock2stove.data.model

enum class ScanMode(val contextKey: String, val displayName: String) {
    PANTRY_SHELF("pantry", "Pantry Shelf"),
    REFRIGERATOR("fridge", "Refrigerator")
}

enum class DetectorDefenseMode(val wireMode: String, val displayLabel: String) {
    YOLO_V8N("yolov8", "YOLOv8n"),
    ORB_BASELINE("orb_baseline", "ORB Baseline"),
    SIFT_BASELINE("sift_baseline", "SIFT Baseline")
}

enum class StorageZone(val zoneKey: String, val displayName: String) {
    PANTRY("Pantry", "Pantry Shelf"),
    REFRIGERATOR("Refrigerator", "Refrigerator")
}

enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    ERROR
}

enum class UnifiedItemClass(
    val classId: String,
    val displayName: String,
    val defaultStorageZone: StorageZone
) {
    PASTA_NOODLES("pasta_noodles", "Pasta & Noodles", StorageZone.PANTRY),
    CANNED_TOMATO("canned_tomato", "Canned Tomatoes", StorageZone.PANTRY),
    CANNED_TUNA_MEAT("canned_tuna_meat", "Canned Tuna & Meat", StorageZone.PANTRY),
    CEREAL_GRAINS("cereal_grains", "Cereal & Grains", StorageZone.PANTRY),
    COFFEE_CONDIMENT("coffee_condiment", "Coffee & Condiments", StorageZone.PANTRY),
    MILK_DAIRY("milk_dairy", "Milk & Dairy", StorageZone.REFRIGERATOR),
    EGGS("eggs", "Eggs", StorageZone.REFRIGERATOR),
    FRESH_PRODUCE("fresh_produce", "Fresh Produce", StorageZone.REFRIGERATOR);

    companion object {
        private val ALIAS_MAP: Map<String, UnifiedItemClass> = mapOf(
            "pasta_noodles" to PASTA_NOODLES,
            "pasta_box" to PASTA_NOODLES,
            "pasta" to PASTA_NOODLES,
            "noodles" to PASTA_NOODLES,
            "spaghetti" to PASTA_NOODLES,
            "macaroni" to PASTA_NOODLES,
            "instant_noodles" to PASTA_NOODLES,
            "canned_tomato" to CANNED_TOMATO,
            "tomato_can" to CANNED_TOMATO,
            "canned_tomatoes" to CANNED_TOMATO,
            "tomato_sauce" to CANNED_TOMATO,
            "tomato_paste" to CANNED_TOMATO,
            "crushed_tomatoes" to CANNED_TOMATO,
            "canned_tuna_meat" to CANNED_TUNA_MEAT,
            "tuna_can" to CANNED_TUNA_MEAT,
            "canned_tuna" to CANNED_TUNA_MEAT,
            "tuna" to CANNED_TUNA_MEAT,
            "corned_beef" to CANNED_TUNA_MEAT,
            "corned_beef_can" to CANNED_TUNA_MEAT,
            "sardines" to CANNED_TUNA_MEAT,
            "sardines_can" to CANNED_TUNA_MEAT,
            "canned_meat" to CANNED_TUNA_MEAT,
            "cereal_grains" to CEREAL_GRAINS,
            "cereal_box" to CEREAL_GRAINS,
            "cereal" to CEREAL_GRAINS,
            "oats" to CEREAL_GRAINS,
            "oatmeal" to CEREAL_GRAINS,
            "rice" to CEREAL_GRAINS,
            "rice_pack" to CEREAL_GRAINS,
            "grains_box" to CEREAL_GRAINS,
            "coffee_condiment" to COFFEE_CONDIMENT,
            "coffee_bag" to COFFEE_CONDIMENT,
            "coffee_jar" to COFFEE_CONDIMENT,
            "coffee" to COFFEE_CONDIMENT,
            "condiment" to COFFEE_CONDIMENT,
            "oil" to COFFEE_CONDIMENT,
            "cooking_oil" to COFFEE_CONDIMENT,
            "soy_sauce" to COFFEE_CONDIMENT,
            "condiment_jar" to COFFEE_CONDIMENT,
            "milk_dairy" to MILK_DAIRY,
            "milk" to MILK_DAIRY,
            "milk_carton" to MILK_DAIRY,
            "cheese" to MILK_DAIRY,
            "cheese_block" to MILK_DAIRY,
            "butter" to MILK_DAIRY,
            "yogurt" to MILK_DAIRY,
            "dairy" to MILK_DAIRY,
            "eggs" to EGGS,
            "egg" to EGGS,
            "egg_carton" to EGGS,
            "egg_tray" to EGGS,
            "fresh_produce" to FRESH_PRODUCE,
            "vegetables" to FRESH_PRODUCE,
            "garlic" to FRESH_PRODUCE,
            "onion" to FRESH_PRODUCE,
            "tomato" to FRESH_PRODUCE,
            "bell_pepper" to FRESH_PRODUCE,
            "carrot" to FRESH_PRODUCE,
            "produce" to FRESH_PRODUCE
        )

        fun fromClassId(classId: String?): UnifiedItemClass? {
            if (classId.isNullOrBlank()) return null
            val cleaned = classId.trim().lowercase().replace(" ", "_").replace("-", "_")
            for (entry in entries) {
                if (entry.classId == cleaned) return entry
            }
            return ALIAS_MAP[cleaned]
        }

        fun normalizeClassId(rawName: String?): String {
            if (rawName.isNullOrBlank()) return PASTA_NOODLES.classId
            val itemClass = normalize(rawName)
            return itemClass.classId
        }

        fun normalize(rawName: String?): UnifiedItemClass {
            if (rawName.isNullOrBlank()) return PASTA_NOODLES
            val cleaned = rawName.trim().lowercase().replace(" ", "_").replace("-", "_")
            val direct = ALIAS_MAP[cleaned]
            if (direct != null) return direct
            for (entry in entries) {
                if (entry.classId == cleaned) return entry
            }
            for (entry in entries) {
                if (cleaned.contains(entry.classId) || entry.classId.contains(cleaned)) {
                    return entry
                }
            }
            return PASTA_NOODLES
        }
    }
}
