"""
CSST 106 Computer Vision Final Project - Downstream LLM Module
Project: Stock-to-Stove
Downstream task: Takes deduplicated detected pantry items from YOLOv8 and generates recipe suggestions.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import json
import os
import time
from typing import List

from src.server.config import (
    LLM_TIMEOUT_SECONDS,
    OUTPUTS_DIR,
    get_display_name,
    get_storage_zone,
    normalize_class_id,
)

RECIPE_MEMORY_CACHE = {}
RECIPE_CACHE_FILE = os.path.join(OUTPUTS_DIR, "recipe_cache.json")

OFFLINE_RECIPE_BANK = [
    {
        "id": "recipe_arrabbiata_tuna",
        "title": "One-Pan Pantry Tuna Arrabbiata Pasta",
        "prep_time_mins": 15,
        "difficulty": "Easy",
        "required_classes": ["pasta_noodles", "canned_tomato", "canned_tuna_meat"],
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


def extract_normalized_inventory(inventory_items):
    normalized = []
    seen = set()
    if not inventory_items:
        return normalized

    for item in inventory_items:
        raw_val = None
        if isinstance(item, dict):
            raw_val = item.get("class_id") or item.get("display_name") or item.get("name")
        else:
            raw_val = str(item)
        if not raw_val:
            continue
        cls_id = normalize_class_id(raw_val)
        if cls_id not in seen:
            seen.add(cls_id)
            normalized.append(cls_id)

    normalized.sort()
    return normalized


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


def build_offline_recipe_cards(normalized_classes):
    class_set = set(normalized_classes)
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
        for c in matched_cls:
            matched_names.append(get_display_name(c))

        zones_list = sorted(list(zones_set))
        if not zones_list:
            zones_list = ["Pantry"]

        overlap_score = len(set(req).intersection(class_set))
        card = {
            "id": template["id"],
            "title": template["title"],
            "prep_time_mins": int(template["prep_time_mins"]),
            "difficulty": template["difficulty"],
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


def build_llm_json_prompt(normalized_classes):
    display_items = []
    for c in normalized_classes:
        zone = get_storage_zone(c)
        name = get_display_name(c)
        display_items.append(f"{name} ({zone})")

    joined_items = ", ".join(display_items)
    if not joined_items:
        joined_items = "Pasta & Noodles (Pantry), Canned Tomatoes (Pantry)"

    prompt = (
        f"Available pantry and refrigerator ingredients: {joined_items}. "
        "Generate exactly 3 practical zero-waste recipes using mostly these items plus basic salt, pepper, and cooking oil. "
        "Return ONLY valid JSON as an array of 3 objects with keys: "
        "id (string), title (string), prep_time_mins (int), difficulty (string), "
        "storage_zones_used (array of strings), matched_ingredients (array of strings), "
        "additional_staples (array of strings), steps (array of 3-4 step strings), zero_waste_tip (string)."
    )
    return prompt


def parse_llm_recipes_json(raw_text, fallback_classes):
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
            for c in fallback_classes[:3]:
                matched.append(get_display_name(c))
        steps = item.get("steps") or []
        if not steps:
            steps = ["Prepare ingredients.", "Cook over medium heat.", "Season and serve warm."]
        cards.append({
            "id": str(item.get("id") or f"llm_recipe_{idx + 1}"),
            "title": str(item.get("title") or f"Zero-Waste Pantry Dish #{idx + 1}"),
            "prep_time_mins": int(item.get("prep_time_mins") or 15),
            "difficulty": str(item.get("difficulty") or "Easy"),
            "storage_zones_used": list(item.get("storage_zones_used") or ["Pantry", "Refrigerator"]),
            "matched_ingredients": list(matched),
            "additional_staples": list(item.get("additional_staples") or ["Cooking Oil", "Salt", "Pepper"]),
            "steps": list(steps),
            "zero_waste_tip": str(item.get("zero_waste_tip") or "Use remaining broth or sauce before rinsing containers."),
        })

    if len(cards) < 3:
        fallback_cards = build_offline_recipe_cards(fallback_classes)
        for fc in fallback_cards:
            if len(cards) >= 3:
                break
            cards.append(fc)
    return cards


def query_live_llm_worker(prompt, gemini_key, openai_key):
    if gemini_key:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text, "gemini-2.5-flash"

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


def generate_structured_recipes(inventory_items, api_key=None, timeout_sec=LLM_TIMEOUT_SECONDS):
    start_time = time.perf_counter()
    normalized_classes = extract_normalized_inventory(inventory_items)
    if not normalized_classes:
        normalized_classes = ["pasta_noodles", "canned_tomato"]

    cache_key = "|".join(normalized_classes)

    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if api_key and not os.getenv("GEMINI_API_KEY") and os.getenv("OPENAI_API_KEY"):
        openai_key = api_key

    if gemini_key or openai_key:
        prompt = build_llm_json_prompt(normalized_classes)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(query_live_llm_worker, prompt, gemini_key, openai_key)
        try:
            raw_text, source_name = future.result(timeout=float(timeout_sec))
            executor.shutdown(wait=False, cancel_futures=True)
            cards = parse_llm_recipes_json(raw_text, normalized_classes)
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
                    "recipes": cards,
                }
        except (FuturesTimeoutError, Exception):
            executor.shutdown(wait=False, cancel_futures=True)

    if cache_key in RECIPE_MEMORY_CACHE:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "source": "offline_cache",
            "cached": True,
            "latency_ms": round(elapsed_ms, 2),
            "inventory_used": normalized_classes,
            "recipes": RECIPE_MEMORY_CACHE[cache_key],
        }

    disk_cache = load_disk_recipe_cache()
    if cache_key in disk_cache:
        RECIPE_MEMORY_CACHE[cache_key] = disk_cache[cache_key]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "source": "offline_cache",
            "cached": True,
            "latency_ms": round(elapsed_ms, 2),
            "inventory_used": normalized_classes,
            "recipes": disk_cache[cache_key],
        }

    offline_cards = build_offline_recipe_cards(normalized_classes)
    RECIPE_MEMORY_CACHE[cache_key] = offline_cards
    disk_cache[cache_key] = offline_cards
    save_disk_recipe_cache(disk_cache)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "source": "offline_cache",
        "cached": True,
        "latency_ms": round(elapsed_ms, 2),
        "inventory_used": normalized_classes,
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
    if os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return response.text
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
