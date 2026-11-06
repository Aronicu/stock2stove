"""
CSST 106 Computer Vision Final Project - Downstream LLM Module
Project: Stock-to-Stove
Downstream task: Takes deduplicated detected pantry items from YOLOv8 and generates recipe suggestions.
"""

import os
from typing import List

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
    test_pantry = ["pasta", "canned tomatoes", "garlic", "coffee", "tuna"]
    print("Testing Downstream LLM Module with detected inventory:", test_pantry)
    print("----------------------------------------------------------------------")
    print(generate_recipes_from_inventory(test_pantry))
