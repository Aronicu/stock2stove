export const UNIFIED_FOOD_CLASSES = [
  {
    id: 0,
    class_id: 'pasta_noodles',
    display_name: 'Pasta & Noodles',
    storage_zone: 'Pantry',
    calories_per_unit: 220,
  },
  {
    id: 1,
    class_id: 'canned_tomato',
    display_name: 'Canned Tomatoes',
    storage_zone: 'Pantry',
    calories_per_unit: 70,
  },
  {
    id: 2,
    class_id: 'canned_tuna_meat',
    display_name: 'Canned Tuna & Meat',
    storage_zone: 'Pantry',
    calories_per_unit: 180,
  },
  {
    id: 3,
    class_id: 'cereal_grains',
    display_name: 'Cereal & Grains',
    storage_zone: 'Pantry',
    calories_per_unit: 190,
  },
  {
    id: 4,
    class_id: 'coffee_condiment',
    display_name: 'Coffee & Condiments',
    storage_zone: 'Pantry',
    calories_per_unit: 45,
  },
  {
    id: 5,
    class_id: 'milk_dairy',
    display_name: 'Milk & Dairy',
    storage_zone: 'Refrigerator',
    calories_per_unit: 150,
  },
  {
    id: 6,
    class_id: 'eggs',
    display_name: 'Eggs',
    storage_zone: 'Refrigerator',
    calories_per_unit: 78,
  },
  {
    id: 7,
    class_id: 'fresh_produce',
    display_name: 'Fresh Produce',
    storage_zone: 'Refrigerator',
    calories_per_unit: 55,
  },
]

export const DETECTOR_MODES = [
  { value: 'yolov8', label: 'YOLOv8n Fine-Tuned' },
  { value: 'hybrid_gemini', label: 'Hybrid YOLOv8n + Gemini' },
  { value: 'gemini_vision', label: 'Gemini Vision' },
  { value: 'orb_baseline', label: 'ORB + RANSAC' },
  { value: 'sift_baseline', label: 'SIFT + RANSAC' },
]

export const BUILTIN_SAMPLES = Array.from({ length: 10 }, function createSampleEntry(_, idx) {
  const num = String(idx + 1).padStart(2, '0')
  const filename = `pantry_images_cv_${num}.jpg`
  return {
    value: filename,
    label: `Sample ${num} (${filename})`,
  }
})

export const CLASS_CALORIE_MAP = {
  pasta_noodles: 220,
  canned_tomato: 70,
  canned_tuna_meat: 180,
  cereal_grains: 190,
  coffee_condiment: 45,
  milk_dairy: 150,
  eggs: 78,
  fresh_produce: 55,
}

export const CLASS_ALIAS_MAP = {
  pasta_noodles: 'pasta_noodles',
  pasta_box: 'pasta_noodles',
  pasta: 'pasta_noodles',
  noodles: 'pasta_noodles',
  spaghetti: 'pasta_noodles',
  macaroni: 'pasta_noodles',
  canned_tomato: 'canned_tomato',
  tomato_can: 'canned_tomato',
  canned_tomatoes: 'canned_tomato',
  tomato_sauce: 'canned_tomato',
  tomato_paste: 'canned_tomato',
  canned_tuna_meat: 'canned_tuna_meat',
  tuna_can: 'canned_tuna_meat',
  canned_tuna: 'canned_tuna_meat',
  tuna: 'canned_tuna_meat',
  corned_beef: 'canned_tuna_meat',
  sardines: 'canned_tuna_meat',
  cereal_grains: 'cereal_grains',
  cereal_box: 'cereal_grains',
  cereal: 'cereal_grains',
  oats: 'cereal_grains',
  oatmeal: 'cereal_grains',
  rice: 'cereal_grains',
  coffee_condiment: 'coffee_condiment',
  coffee_bag: 'coffee_condiment',
  coffee: 'coffee_condiment',
  condiment: 'coffee_condiment',
  oil: 'coffee_condiment',
  milk_dairy: 'milk_dairy',
  milk: 'milk_dairy',
  cheese: 'milk_dairy',
  butter: 'milk_dairy',
  yogurt: 'milk_dairy',
  eggs: 'eggs',
  egg: 'eggs',
  fresh_produce: 'fresh_produce',
  vegetables: 'fresh_produce',
  garlic: 'fresh_produce',
  onion: 'fresh_produce',
  tomato: 'fresh_produce',
}

export function normalizeClassId(rawName) {
  if (!rawName) {
    return ''
  }
  const cleaned = String(rawName).trim().toLowerCase().replace(/[\s-]+/g, '_')
  if (Object.prototype.hasOwnProperty.call(CLASS_ALIAS_MAP, cleaned)) {
    return CLASS_ALIAS_MAP[cleaned]
  }
  for (let i = 0; i < UNIFIED_FOOD_CLASSES.length; i += 1) {
    if (UNIFIED_FOOD_CLASSES[i].class_id === cleaned) {
      return cleaned
    }
  }
  return cleaned
}

export function getClassMeta(classId) {
  const canonical = normalizeClassId(classId)
  for (let i = 0; i < UNIFIED_FOOD_CLASSES.length; i += 1) {
    if (UNIFIED_FOOD_CLASSES[i].class_id === canonical) {
      return UNIFIED_FOOD_CLASSES[i]
    }
  }
  return {
    id: -1,
    class_id: canonical || 'unknown',
    display_name: String(classId || 'Unknown Item'),
    storage_zone: 'Pantry',
    calories_per_unit: CLASS_CALORIE_MAP[canonical] || 100,
  }
}

export function getCaloriesPerUnit(classId) {
  const canonical = normalizeClassId(classId)
  if (Object.prototype.hasOwnProperty.call(CLASS_CALORIE_MAP, canonical)) {
    return CLASS_CALORIE_MAP[canonical]
  }
  return 100
}

export function computeTotalBasketCalories(items) {
  let total = 0
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i]
    const qty = Math.max(1, Number(item.count || 1))
    const perUnit = Number(item.calories_per_unit || getCaloriesPerUnit(item.class_id))
    const subtotal = item.estimated_calories != null ? Number(item.estimated_calories) : perUnit * qty
    total += subtotal
  }
  return Math.round(total)
}

export function formatCropDataUrl(cropB64) {
  if (!cropB64) {
    return null
  }
  if (String(cropB64).startsWith('data:')) {
    return cropB64
  }
  return `data:image/jpeg;base64,${cropB64}`
}

export function createClientSessionId() {
  const rand = Math.random().toString(16).slice(2, 10)
  return `spa_${rand}`
}
