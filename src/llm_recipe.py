"""
CSST 106 Computer Vision Final Project - Downstream LLM Module
Project: Stock-to-Stove
Downstream task: Takes deduplicated detected pantry items from YOLOv8 and generates recipe suggestions.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import json
import math
import os
import time
from typing import List

from src.cv.gemini_detector import get_cached_gemini_client
from src.server.config import (
    GEMINI_RECIPE_MODEL,
    LLM_TIMEOUT_SECONDS,
    NON_FOOD_LABELS,
    OUTPUTS_DIR,
    get_class_calories,
    get_display_name,
    get_gemini_thinking_level,
    get_storage_zone,
    is_valid_food_class,
    normalize_class_id,
)

RECIPE_MEMORY_CACHE = {}
RECIPE_CACHE_FILE = os.path.join(OUTPUTS_DIR, "recipe_cache.json")
RECIPE_EXECUTOR = ThreadPoolExecutor(max_workers=4)

OFFLINE_RECIPE_BANK = [
    {
        "id": "recipe_arrabbiata_tuna",
        "title": "One-Pan Pantry Tuna Arrabbiata Pasta",
        "prep_time_mins": 15,
        "difficulty": "Easy",
        "required_classes": ["pasta_noodles", "canned_tomato", "canned_tuna_meat"],
        "estimated_calories": 470,
        "ingredient_calories": {
            "Pasta & Noodles": 220,
            "Canned Tomatoes": 70,
            "Canned Tuna & Meat": 180,
        },
        "additional_staples": ["Cooking Oil", "Salt", "Black Pepper", "Chili Flakes"],
        "steps": [
            "Boil pasta or noodles in salted water until al dente, reserving 1/4 cup pasta water.",
            "Warm oil in a skillet, pour in canned tomatoes, and simmer for 5 minutes until thickened.",
            "Flake drained canned tuna or meat into the simmering tomato sauce.",
            "Toss drained pasta into the skillet with reserved water and serve hot."
        ],
        "zero_waste_tip": "Rinse the tomato can with 2 tablespoons of pasta water to capture all remaining sauce."
    },
    {
        "id": "recipe_creamy_fridge_pasta",
        "title": "Creamy Garlic Dairy & Vegetable Skillet Pasta",
        "prep_time_mins": 18,
        "difficulty": "Easy",
        "required_classes": ["pasta_noodles", "milk_dairy", "fresh_produce"],
        "estimated_calories": 425,
        "ingredient_calories": {
            "Pasta & Noodles": 220,
            "Milk & Dairy": 150,
            "Fresh Produce": 55,
        },
        "additional_staples": ["Salt", "Black Pepper", "Cooking Oil"],
        "steps": [
            "Cook pasta in salted boiling water until tender, then drain.",
            "Saute diced fresh produce (onions, tomatoes, or greens) in oil over medium heat for 4 minutes.",
            "Lower heat and stir in milk, cheese, or butter until a smooth pan sauce forms.",
            "Fold the cooked pasta into the creamy vegetable sauce and season to taste."
        ],
        "zero_waste_tip": "Finely chop vegetable stems and saute them first so no fresh produce goes to waste."
    },
    {
        "id": "recipe_shakshuka_eggs",
        "title": "Rustic Skillet Tomato & Poached Eggs (Shakshuka)",
        "prep_time_mins": 12,
        "difficulty": "Easy",
        "required_classes": ["canned_tomato", "eggs", "fresh_produce"],
        "estimated_calories": 203,
        "ingredient_calories": {
            "Canned Tomatoes": 70,
            "Eggs": 78,
            "Fresh Produce": 55,
        },
        "additional_staples": ["Cooking Oil", "Salt", "Black Pepper"],
        "steps": [
            "Saute chopped fresh produce in a skillet with oil until softened.",
            "Pour in canned tomatoes, season with salt and pepper, and simmer for 5 minutes.",
            "Make gentle wells in the tomato sauce and crack eggs directly into each well.",
            "Cover skillet and cook on low for 4 to 5 minutes until egg whites set."
        ],
        "zero_waste_tip": "Serve over toasted bread ends or leftover cooked grains to soak up every drop of sauce."
    },
    {
        "id": "recipe_protein_scramble",
        "title": "Farmhouse Fridge Egg, Dairy & Tuna Scramble",
        "prep_time_mins": 10,
        "difficulty": "Easy",
        "required_classes": ["eggs", "milk_dairy", "canned_tuna_meat"],
        "estimated_calories": 408,
        "ingredient_calories": {
            "Eggs": 78,
            "Milk & Dairy": 150,
            "Canned Tuna & Meat": 180,
        },
        "additional_staples": ["Salt", "Black Pepper", "Cooking Oil"],
        "steps": [
            "Whisk eggs with a splash of milk or dairy and a pinch of salt.",
            "Warm drained canned tuna or meat in a lightly oiled non-stick pan for 2 minutes.",
            "Pour whisked eggs over the protein and fold gently over medium-low heat until softly set.",
            "Top with any remaining cheese or diced vegetables and serve immediately."
        ],
        "zero_waste_tip": "Use any small leftover dairy splashes at the bottom of the carton to make eggs fluffier."
    },
    {
        "id": "recipe_savory_grain_bowl",
        "title": "Hearty Savory Grain, Tomato & Protein Bowl",
        "prep_time_mins": 15,
        "difficulty": "Easy",
        "required_classes": ["cereal_grains", "canned_tomato", "canned_tuna_meat", "eggs"],
        "estimated_calories": 518,
        "ingredient_calories": {
            "Cereal & Grains": 190,
            "Canned Tomatoes": 70,
            "Canned Tuna & Meat": 180,
            "Eggs": 78,
        },
        "additional_staples": ["Water", "Salt", "Black Pepper", "Soy Sauce or Oil"],
        "steps": [
            "Simmer grains or oats with water and a spoonful of canned tomatoes for a savory base.",
            "Warm canned tuna or meat in a small pan with condiments and black pepper.",
            "Spoon the savory tomato grains into a bowl and top with the seasoned protein.",
            "Finish with a pan-fried egg or fresh chopped produce on top."
        ],
        "zero_waste_tip": "Savory oatmeal and rice bowls rescue single-serving grain leftovers in under 15 minutes."
    },
    {
        "id": "recipe_cafe_breakfast_parfait",
        "title": "Mocha Morning Grain & Dairy Breakfast Bowl",
        "prep_time_mins": 8,
        "difficulty": "Easy",
        "required_classes": ["cereal_grains", "milk_dairy", "coffee_condiment"],
        "estimated_calories": 385,
        "ingredient_calories": {
            "Cereal & Grains": 190,
            "Milk & Dairy": 150,
            "Coffee & Condiments": 45,
        },
        "additional_staples": ["Warm Water", "Pinch of Salt"],
        "steps": [
            "Combine cereal or oats with milk or yogurt in a bowl and let soften for 3 minutes.",
            "Dissolve a half-teaspoon of instant coffee or condiment drizzle in a splash of warm milk.",
            "Swirl the coffee-dairy mixture over the grain bowl for a cafe-style breakfast.",
            "Serve chilled or gently warmed."
        ],
        "zero_waste_tip": "Crush the fine cereal flakes at the bottom of the box as a crunchy topping layer."
    },
]


def resolve_inventory_class_id(item):
    if isinstance(item, dict):
        raw_cls = item.get("class_id")
        raw_disp = item.get("display_name") or item.get("name")
        if raw_cls is not None and str(raw_cls).strip():
            cls_clean = str(raw_cls).strip().lower()
            cls_spaced = cls_clean.replace("_", " ").replace("-", " ")
            if cls_clean in NON_FOOD_LABELS or cls_spaced in NON_FOOD_LABELS:
                return ""
            if is_valid_food_class(raw_cls):
                return normalize_class_id(raw_cls)
        if raw_disp is not None and str(raw_disp).strip():
            if is_valid_food_class(raw_disp):
                return normalize_class_id(raw_disp)
        return ""
    if item is None:
        return ""
    raw_val = str(item).strip()
    if not raw_val or not is_valid_food_class(raw_val):
        return ""
    return normalize_class_id(raw_val)


def extract_normalized_inventory(inventory_items):
    normalized = []
    seen = set()
    if not inventory_items:
        return normalized

    for item in inventory_items:
        cls_id = resolve_inventory_class_id(item)
        if not cls_id:
            continue
        if cls_id not in seen:
            seen.add(cls_id)
            normalized.append(cls_id)

    normalized.sort()
    return normalized


def extract_detailed_inventory(inventory_items):
    detailed = []
    if not inventory_items:
        return detailed

    for item in inventory_items:
        cls_id = resolve_inventory_class_id(item)
        if not cls_id:
            continue

        disp_name = get_display_name(cls_id)
        qty = 1
        cpu = None
        est_kcal = None
        zone = get_storage_zone(cls_id)

        if isinstance(item, dict):
            raw_disp = item.get("display_name")
            disp_clean = " ".join(str(raw_disp or "").strip().split())
            if disp_clean:
                disp_name = disp_clean

            raw_cnt = item.get("count", 1)
            if not isinstance(raw_cnt, bool) and raw_cnt is not None:
                try:
                    qty = max(1, int(raw_cnt))
                except (TypeError, ValueError):
                    qty = 1

            raw_cpu = item.get("calories_per_unit")
            cpu_float = None
            if raw_cpu is not None and not isinstance(raw_cpu, bool):
                try:
                    parsed_cpu = float(raw_cpu)
                    if math.isfinite(parsed_cpu) and parsed_cpu > 0.0:
                        cpu_float = parsed_cpu
                        cpu = max(1, int(round(parsed_cpu)))
                except (TypeError, ValueError):
                    cpu = None

            raw_est = item.get("estimated_calories")
            est_from_raw = False
            if raw_est is not None and not isinstance(raw_est, bool):
                try:
                    parsed_est = float(raw_est)
                    if math.isfinite(parsed_est) and parsed_est > 0.0:
                        est_kcal = max(1, int(round(parsed_est)))
                        est_from_raw = True
                except (TypeError, ValueError):
                    est_kcal = None

            if est_kcal is None:
                if cpu_float is not None:
                    est_kcal = max(1, int(round(cpu_float * qty)))
                else:
                    est_kcal = get_class_calories(cls_id, count=qty)

            if cpu is None:
                if est_from_raw:
                    cpu = max(1, int(round(float(est_kcal) / float(qty))))
                else:
                    cpu = get_class_calories(cls_id, count=1)

            raw_zone = str(item.get("storage_zone") or "").strip()
            if raw_zone:
                zone = raw_zone
        else:
            cpu = get_class_calories(cls_id, count=1)
            est_kcal = get_class_calories(cls_id, count=1)

        detailed.append({
            "class_id": cls_id,
            "display_name": disp_name,
            "count": int(qty),
            "calories_per_unit": int(cpu),
            "estimated_calories": int(est_kcal),
            "storage_zone": zone,
        })

    keyed = []
    for idx, entry in enumerate(detailed):
        keyed.append((entry["class_id"], entry["display_name"].lower(), idx, entry))
    keyed.sort()

    sorted_detailed = []
    for item_tuple in keyed:
        sorted_detailed.append(item_tuple[3])
    return sorted_detailed


def build_inventory_cache_key(inventory_items, normalized_classes):
    has_custom_or_multi = False
    if inventory_items:
        for item in inventory_items:
            if not isinstance(item, dict):
                continue
            cls_id = resolve_inventory_class_id(item)
            if not cls_id:
                continue
            raw_disp = item.get("display_name")
            disp_clean = " ".join(str(raw_disp or "").strip().split())
            if disp_clean and disp_clean.lower() != get_display_name(cls_id).strip().lower():
                has_custom_or_multi = True
                break
            raw_cnt = item.get("count", 1)
            qty = 1
            if not isinstance(raw_cnt, bool) and raw_cnt is not None:
                try:
                    qty = max(1, int(raw_cnt))
                except (TypeError, ValueError):
                    qty = 1
            if qty > 1:
                has_custom_or_multi = True
                break

    if has_custom_or_multi:
        detailed = extract_detailed_inventory(inventory_items)
        if detailed:
            parts = []
            for entry in detailed:
                parts.append(f"{entry['class_id']}:{entry['display_name'].lower()}:{entry['count']}")
            parts.sort()
            return "|".join(parts)

    return "|".join(normalized_classes)


def compute_inventory_total_calories(inventory_items):
    if not inventory_items:
        return 0
    total_kcal = 0
    for item in inventory_items:
        cls_id = resolve_inventory_class_id(item)
        if not cls_id:
            continue
        qty = 1
        if isinstance(item, dict):
            raw_cnt = item.get("count", 1)
            if not isinstance(raw_cnt, bool) and raw_cnt is not None:
                try:
                    qty = max(1, int(raw_cnt))
                except (TypeError, ValueError):
                    qty = 1
        item_kcal = None
        if isinstance(item, dict):
            raw_est = item.get("estimated_calories")
            if raw_est is not None and not isinstance(raw_est, bool):
                try:
                    est_val = float(raw_est)
                    if math.isfinite(est_val) and est_val > 0.0:
                        item_kcal = int(round(est_val))
                except (TypeError, ValueError):
                    item_kcal = None
            if item_kcal is None:
                raw_cpu = item.get("calories_per_unit")
                if raw_cpu is not None and not isinstance(raw_cpu, bool):
                    try:
                        cpu_val = float(raw_cpu)
                        if math.isfinite(cpu_val) and cpu_val > 0.0:
                            item_kcal = int(round(cpu_val * qty))
                    except (TypeError, ValueError):
                        item_kcal = None
        if item_kcal is None:
            item_kcal = get_class_calories(cls_id, count=qty)
        total_kcal += item_kcal
    return int(total_kcal)


def compute_recipe_calorie_fallback(
    matched_ingredients,
    fallback_classes,
    raw_ingredient_calories=None,
    raw_estimated_calories=None,
    detailed_items=None,
):
    detailed_cal_by_name = {}
    if detailed_items:
        for d_item in detailed_items:
            d_name = str(d_item.get("display_name") or "").strip().lower()
            if d_name:
                detailed_cal_by_name[d_name] = int(
                    d_item.get("calories_per_unit") or get_class_calories(d_item.get("class_id", ""), count=1)
                )

    ing_cals = {}
    if isinstance(raw_ingredient_calories, dict) and len(raw_ingredient_calories) > 0:
        for k, v in raw_ingredient_calories.items():
            try:
                val_int = max(0, int(round(float(v))))
            except (TypeError, ValueError):
                key_lower = str(k).strip().lower()
                if key_lower in detailed_cal_by_name:
                    val_int = detailed_cal_by_name[key_lower]
                else:
                    val_int = get_class_calories(str(k), count=1)
            ing_cals[str(k)] = val_int
    else:
        for ing in matched_ingredients:
            ing_name = str(ing).strip()
            if not ing_name:
                continue
            ing_lower = ing_name.lower()
            if ing_lower in detailed_cal_by_name:
                ing_cals[ing_name] = detailed_cal_by_name[ing_lower]
            else:
                ing_cals[ing_name] = get_class_calories(ing_name, count=1)
        if not ing_cals and fallback_classes:
            for c in fallback_classes[:3]:
                ing_cals[get_display_name(c)] = get_class_calories(c, count=1)

    est_kcal = 0
    if raw_estimated_calories is not None:
        try:
            est_kcal = int(round(float(raw_estimated_calories)))
        except (TypeError, ValueError):
            est_kcal = 0
    if est_kcal <= 0:
        est_kcal = sum(ing_cals.values())
    if est_kcal <= 0:
        est_kcal = 250

    return int(est_kcal), ing_cals


def enrich_recipe_cards_calories(cards, fallback_classes, detailed_items=None):
    enriched = []
    changed = False
    for card in cards:
        if not isinstance(card, dict):
            continue
        card_copy = dict(card)
        has_est = "estimated_calories" in card_copy and isinstance(card_copy.get("estimated_calories"), (int, float)) and int(card_copy["estimated_calories"]) > 0
        has_ing = isinstance(card_copy.get("ingredient_calories"), dict) and len(card_copy.get("ingredient_calories")) > 0
        if not has_est or not has_ing:
            changed = True
        est_kcal, ing_cals = compute_recipe_calorie_fallback(
            card_copy.get("matched_ingredients") or [],
            fallback_classes,
            raw_ingredient_calories=card_copy.get("ingredient_calories"),
            raw_estimated_calories=card_copy.get("estimated_calories"),
            detailed_items=detailed_items,
        )
        card_copy["estimated_calories"] = est_kcal
        card_copy["ingredient_calories"] = ing_cals
        enriched.append(card_copy)
    return enriched, changed


def load_disk_recipe_cache():
    if not os.path.exists(RECIPE_CACHE_FILE):
        return {}
    try:
        with open(RECIPE_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_disk_recipe_cache(cache_dict):
    try:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        with open(RECIPE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, indent=2)
    except Exception:
        pass


def build_offline_recipe_cards(normalized_classes, detailed_items=None):
    class_set = set(normalized_classes)
    detailed_by_class = {}
    if detailed_items:
        for d_item in detailed_items:
            c_id = d_item.get("class_id")
            if not c_id:
                continue
            if c_id not in detailed_by_class:
                detailed_by_class[c_id] = []
            detailed_by_class[c_id].append(d_item)

    scored = []

    for idx, template in enumerate(OFFLINE_RECIPE_BANK):
        req = template["required_classes"]
        matched_cls = []
        zones_set = set()
        for c in req:
            if c in class_set:
                matched_cls.append(c)
                zones_set.add(get_storage_zone(c))

        if len(matched_cls) == 0 and len(normalized_classes) > 0:
            for c in normalized_classes[:2]:
                matched_cls.append(c)
                zones_set.add(get_storage_zone(c))

        matched_names = []
        seen_names = set()
        ingredient_calories = {}
        for c in matched_cls:
            class_items = detailed_by_class.get(c)
            if class_items:
                for d_item in class_items:
                    disp = str(d_item.get("display_name") or get_display_name(c))
                    item_zone = d_item.get("storage_zone") or get_storage_zone(c)
                    zones_set.add(item_zone)
                    item_kcal = int(d_item.get("calories_per_unit") or get_class_calories(c, count=1))
                    if disp not in seen_names:
                        seen_names.add(disp)
                        matched_names.append(disp)
                    if disp not in ingredient_calories:
                        ingredient_calories[disp] = item_kcal
            else:
                disp = get_display_name(c)
                if disp not in seen_names:
                    seen_names.add(disp)
                    matched_names.append(disp)
                ingredient_calories[disp] = get_class_calories(c, count=1)

        if not ingredient_calories:
            ingredient_calories = dict(template.get("ingredient_calories", {}))

        estimated_calories = sum(ingredient_calories.values())
        if estimated_calories <= 0:
            estimated_calories = int(template.get("estimated_calories", 250))

        zones_list = sorted(list(zones_set))
        if not zones_list:
            zones_list = ["Pantry"]

        overlap_score = len(set(req).intersection(class_set))
        card = {
            "id": template["id"],
            "title": template["title"],
            "prep_time_mins": int(template["prep_time_mins"]),
            "difficulty": template["difficulty"],
            "estimated_calories": int(estimated_calories),
            "ingredient_calories": ingredient_calories,
            "storage_zones_used": zones_list,
            "matched_ingredients": matched_names,
            "additional_staples": list(template["additional_staples"]),
            "steps": list(template["steps"]),
            "zero_waste_tip": template["zero_waste_tip"],
        }
        scored.append((overlap_score, -idx, card))

    n = len(scored)
    for i in range(n):
        for j in range(i + 1, n):
            if scored[j][0] > scored[i][0] or (
                scored[j][0] == scored[i][0] and scored[j][1] > scored[i][1]
            ):
                temp = scored[i]
                scored[i] = scored[j]
                scored[j] = temp

    selected = []
    for entry in scored[:3]:
        selected.append(entry[2])
    return selected


def build_llm_json_prompt(normalized_classes, detailed_items=None):
    display_items = []
    if detailed_items:
        for item in detailed_items:
            cls_id = item.get("class_id") or ""
            name = item.get("display_name") or get_display_name(cls_id)
            qty = max(1, int(item.get("count", 1)))
            zone = item.get("storage_zone") or get_storage_zone(cls_id)
            kcal = int(item.get("estimated_calories") or get_class_calories(cls_id, count=qty))
            display_items.append(f"{name} x{qty} [{cls_id}] ({zone}, ~{kcal} kcal)")
    else:
        for c in normalized_classes:
            zone = get_storage_zone(c)
            name = get_display_name(c)
            kcal = get_class_calories(c, count=1)
            display_items.append(f"{name} ({zone}, ~{kcal} kcal)")

    joined_items = ", ".join(display_items)
    if not joined_items:
        joined_items = "Pasta & Noodles (Pantry, ~220 kcal), Canned Tomatoes (Pantry, ~70 kcal)"

    if detailed_items:
        instruction = (
            "Generate exactly 3 practical zero-waste recipes tailored specifically to these exact named products and quantities "
            "(for example, using cornstarch as a sauce thickener or crispy coating rather than a breakfast grain, "
            "and referencing the specific product names in matched_ingredients and ingredient_calories) "
            "using mostly these items plus basic salt, pepper, and cooking oil. "
        )
    else:
        instruction = (
            "Generate exactly 3 practical zero-waste recipes using mostly these items plus basic salt, pepper, and cooking oil. "
        )

    prompt = (
        f"Available pantry and refrigerator ingredients: {joined_items}. "
        f"{instruction}"
        "Return ONLY valid JSON as an array of 3 objects with keys: "
        "id (string), title (string), prep_time_mins (int), difficulty (string), "
        "estimated_calories (int kcal per serving), ingredient_calories (object mapping ingredient display name to int kcal), "
        "storage_zones_used (array of strings), matched_ingredients (array of strings), "
        "additional_staples (array of strings), steps (array of 3-4 step strings), zero_waste_tip (string)."
    )
    return prompt


def parse_llm_recipes_json(raw_text, fallback_classes, detailed_items=None):
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
             candidate = part.strip()
             if candidate.startswith("json"):
                 candidate = candidate[4:].strip()
             if candidate.startswith("[") or candidate.startswith("{"):
                 cleaned = candidate
                 break

    parsed = json.loads(cleaned)
    if isinstance(parsed, dict) and "recipes" in parsed:
        parsed = parsed["recipes"]
    if not isinstance(parsed, list) or len(parsed) == 0:
        return None

    cards = []
    for idx, item in enumerate(parsed[:3]):
        if not isinstance(item, dict):
            continue
        matched = item.get("matched_ingredients") or []
        if not matched:
            if detailed_items:
                for d_item in detailed_items[:3]:
                    d_disp = d_item.get("display_name")
                    if d_disp and d_disp not in matched:
                        matched.append(d_disp)
            else:
                for c in fallback_classes[:3]:
                    matched.append(get_display_name(c))
        steps = item.get("steps") or []
        if not steps:
            steps = ["Prepare ingredients.", "Cook over medium heat.", "Season and serve warm."]
        est_kcal, ing_cals = compute_recipe_calorie_fallback(
            matched,
            fallback_classes,
            raw_ingredient_calories=item.get("ingredient_calories"),
            raw_estimated_calories=item.get("estimated_calories"),
            detailed_items=detailed_items,
        )
        cards.append({
            "id": str(item.get("id") or f"llm_recipe_{idx + 1}"),
            "title": str(item.get("title") or f"Zero-Waste Pantry Dish #{idx + 1}"),
            "prep_time_mins": int(item.get("prep_time_mins") or 15),
            "difficulty": str(item.get("difficulty") or "Easy"),
            "estimated_calories": int(est_kcal),
            "ingredient_calories": ing_cals,
            "storage_zones_used": list(item.get("storage_zones_used") or ["Pantry", "Refrigerator"]),
            "matched_ingredients": list(matched),
            "additional_staples": list(item.get("additional_staples") or ["Cooking Oil", "Salt", "Pepper"]),
            "steps": list(steps),
            "zero_waste_tip": str(item.get("zero_waste_tip") or "Use remaining broth or sauce before rinsing containers."),
        })

    if len(cards) < 3:
        fallback_cards = build_offline_recipe_cards(fallback_classes, detailed_items=detailed_items)
        for fc in fallback_cards:
            if len(cards) >= 3:
                break
            cards.append(fc)
    return cards


def query_live_llm_worker(prompt, gemini_key, openai_key):
    if gemini_key:
        client = get_cached_gemini_client(gemini_key)
        if client is not None:
            recipe_schema = {
                "type": "object",
                "properties": {
                    "recipes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "prep_time_mins": {"type": "integer"},
                                "difficulty": {"type": "string"},
                                "estimated_calories": {"type": "integer"},
                                "ingredient_calories": {
                                    "type": "object",
                                    "additionalProperties": {"type": "integer"},
                                },
                                "storage_zones_used": {"type": "array", "items": {"type": "string"}},
                                "matched_ingredients": {"type": "array", "items": {"type": "string"}},
                                "additional_staples": {"type": "array", "items": {"type": "string"}},
                                "steps": {"type": "array", "items": {"type": "string"}},
                                "zero_waste_tip": {"type": "string"},
                            },
                            "required": [
                                "id",
                                "title",
                                "prep_time_mins",
                                "difficulty",
                                "estimated_calories",
                                "ingredient_calories",
                                "storage_zones_used",
                                "matched_ingredients",
                                "additional_staples",
                                "steps",
                                "zero_waste_tip",
                            ],
                        },
                    }
                },
                "required": ["recipes"],
            }
            interaction = client.interactions.create(
                model=GEMINI_RECIPE_MODEL,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": recipe_schema,
                },
                generation_config={"thinking_level": get_gemini_thinking_level(GEMINI_RECIPE_MODEL)},
                store=False,
            )
            output_text = getattr(interaction, "output_text", None)
            if getattr(interaction, "errors", None) and not output_text:
                raise RuntimeError(str(interaction.errors))
            return output_text, GEMINI_RECIPE_MODEL
        raise RuntimeError("Gemini client is unavailable")

    if openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a zero-waste culinary assistant that outputs strict JSON arrays."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content, "gpt-4o-mini"

    return None, "offline_cache"


def generate_structured_recipes(
    inventory_items,
    api_key=None,
    timeout_sec=LLM_TIMEOUT_SECONDS,
    force_refresh=False,
):
    start_time = time.perf_counter()
    total_inventory_calories = compute_inventory_total_calories(inventory_items)
    normalized_classes = extract_normalized_inventory(inventory_items)
    detailed_items = extract_detailed_inventory(inventory_items)
    if not normalized_classes:
        normalized_classes = ["pasta_noodles", "canned_tomato"]

    cache_key = build_inventory_cache_key(inventory_items, normalized_classes)
    items_used = []
    for it in detailed_items:
        items_used.append(f"{it['display_name']} x{it['count']}")

    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if api_key and not os.getenv("GEMINI_API_KEY") and os.getenv("OPENAI_API_KEY"):
        openai_key = api_key

    llm_error = None
    if gemini_key or openai_key:
        prompt = build_llm_json_prompt(normalized_classes, detailed_items=detailed_items)
        future = RECIPE_EXECUTOR.submit(query_live_llm_worker, prompt, gemini_key, openai_key)
        try:
            elapsed_before_wait = time.perf_counter() - start_time
            wait_timeout = max(0.05, float(timeout_sec) - elapsed_before_wait - 0.08)
            raw_text, source_name = future.result(timeout=wait_timeout)
            cards = parse_llm_recipes_json(raw_text, normalized_classes, detailed_items=detailed_items)
            if cards:
                RECIPE_MEMORY_CACHE[cache_key] = cards
                disk_cache = load_disk_recipe_cache()
                disk_cache[cache_key] = cards
                save_disk_recipe_cache(disk_cache)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "source": source_name,
                    "cached": False,
                    "latency_ms": round(elapsed_ms, 2),
                    "inventory_used": normalized_classes,
                    "items_used": items_used,
                    "total_inventory_calories": int(total_inventory_calories),
                    "recipes": cards,
                }
            llm_error = "Live LLM returned empty or unparseable recipes"
        except FuturesTimeoutError:
            future.cancel()
            llm_error = f"Live LLM timed out after {float(timeout_sec):.1f}s"
        except Exception as exc:
            future.cancel()
            llm_error = f"{type(exc).__name__}: {exc}"
        if llm_error:
            print(f"[llm_recipe] live LLM fallback: {llm_error}")

    if not force_refresh and cache_key in RECIPE_MEMORY_CACHE:
        mem_cards, changed = enrich_recipe_cards_calories(
            RECIPE_MEMORY_CACHE[cache_key],
            normalized_classes,
            detailed_items=detailed_items,
        )
        if changed:
            RECIPE_MEMORY_CACHE[cache_key] = mem_cards
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "source": "offline_cache",
            "cached": True,
            "latency_ms": round(elapsed_ms, 2),
            "inventory_used": normalized_classes,
            "items_used": items_used,
            "total_inventory_calories": int(total_inventory_calories),
            "recipes": mem_cards,
        }

    disk_cache = load_disk_recipe_cache()
    if not force_refresh and cache_key in disk_cache:
        disk_cards, changed = enrich_recipe_cards_calories(
            disk_cache[cache_key],
            normalized_classes,
            detailed_items=detailed_items,
        )
        RECIPE_MEMORY_CACHE[cache_key] = disk_cards
        if changed:
            disk_cache[cache_key] = disk_cards
            save_disk_recipe_cache(disk_cache)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "source": "offline_cache",
            "cached": True,
            "latency_ms": round(elapsed_ms, 2),
            "inventory_used": normalized_classes,
            "items_used": items_used,
            "total_inventory_calories": int(total_inventory_calories),
            "recipes": disk_cards,
        }

    offline_cards = build_offline_recipe_cards(normalized_classes, detailed_items=detailed_items)
    RECIPE_MEMORY_CACHE[cache_key] = offline_cards
    disk_cache[cache_key] = offline_cards
    save_disk_recipe_cache(disk_cache)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "source": "offline_cache",
        "cached": True,
        "latency_ms": round(elapsed_ms, 2),
        "inventory_used": normalized_classes,
        "items_used": items_used,
        "total_inventory_calories": int(total_inventory_calories),
        "recipes": offline_cards,
    }


def generate_recipes_from_inventory(detected_items: List[str], api_key: str = None) -> str:
    """
    Receives detected pantry items and calls an LLM to generate 3 quick recipes.
    Falls back to a structured rule-based response if no API key is configured.
    """
    detected_items = sorted(list(set(detected_items)))
    prompt = (
        f"My pantry currently has these ingredients: {', '.join(detected_items)}. "
        "Give me 3 quick recipe ideas I can make using mostly these items. "
        "You can assume I have basic pantry staples like oil, salt, and pepper. "
        "Output the response in clean bullet points with recipe name, estimated time, and key steps."
    )

    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        # Fallback offline simulation for testing without active network keys
        return (
            f"[OFFLINE PREVIEW - Set GEMINI_API_KEY to query live LLM]\n"
            f"Detected Inventory: {detected_items}\n\n"
            f"Prompt Sent to Downstream Model:\n\"{prompt}\"\n\n"
            f"Simulated LLM Output:\n"
            f"1. Quick Pantry Arrabbiata Pasta (15 mins)\n"
            f"   - Boil pasta until al dente.\n"
            f"   - Sauté garlic in olive oil, stir in crushed canned tomatoes with red pepper flakes and salt.\n"
            f"   - Toss pasta directly in the sauce.\n\n"
            f"2. Rustic Tomato & Tuna Skillet (10 mins)\n"
            f"   - Simmer canned tomatoes with salt, pepper, and pantry spices in a skillet.\n"
            f"   - Fold in drained tuna gently and serve over crusty toast or pasta.\n\n"
            f"3. Hearty Tomato Pantry Soup (15 mins)\n"
            f"   - Combine crushed tomatoes with water/broth and pantry seasoning.\n"
            f"   - Simmer, then top with cracked black pepper and a drizzle of olive oil."
        )

    # If Gemini API Key is available
    if os.getenv("GEMINI_API_KEY") or (api_key and not os.getenv("OPENAI_API_KEY")):
        try:
            client = get_cached_gemini_client(api_key)
            interaction = client.interactions.create(
                model=GEMINI_RECIPE_MODEL,
                input=prompt,
                store=False,
            )
            return interaction.output_text
        except Exception as e:
            return f"Error contacting Gemini API: {e}"

    # If OpenAI API Key is available
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a pragmatic, zero-waste kitchen chef."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error contacting OpenAI API: {e}"

if __name__ == "__main__":
    test_pantry = ["pasta_noodles", "canned_tomato", "canned_tuna_meat", "eggs", "milk_dairy"]
    result = generate_structured_recipes(test_pantry)
    print(json.dumps(result, indent=2))
