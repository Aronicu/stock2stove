"""
Download real sample pantry images directly from https://images.cv/dataset/pantry-image-classification-dataset
and save them to data/samples/
"""

import os
import time
import urllib.parse
import urllib.request
from playwright.sync_api import sync_playwright

def download_images_cv_pantry(output_dir="data/samples", count=20):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[FETCH] Opening images.cv Pantry Dataset via Edge...")

    image_urls = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.goto("https://images.cv/dataset/pantry-image-classification-dataset", wait_until="domcontentloaded", timeout=25000)
        time.sleep(2)

        # Scroll to load thumbnails
        for _ in range(6):
            page.mouse.wheel(0, 1200)
            time.sleep(0.8)

        # Extract all img elements
        srcs = page.eval_on_selector_all("img", "elements => elements.map(e => e.src)")
        browser.close()

    print(f"[FETCH] Found {len(srcs)} total image sources.")
    for s in srcs:
        if "thumbnail-" in s or "cvstock" in s:
            if "url=" in s:
                parsed = urllib.parse.urlparse(s)
                qs = urllib.parse.parse_qs(parsed.query)
                if "url" in qs:
                    image_urls.append(qs["url"][0])
            else:
                image_urls.append(s)

    unique_urls = list(dict.fromkeys(image_urls))
    print(f"[FETCH] Found {len(unique_urls)} unique pantry thumbnails from images.cv.")

    downloaded = 0
    for idx, url in enumerate(unique_urls):
        if downloaded >= count:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                filename = f"pantry_images_cv_{downloaded+1:02d}.jpg"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(data)
                downloaded += 1
                print(f"  [SAVED] {filename} ({len(data)} bytes) from {url}")
        except Exception as e:
            print(f"  [FAIL] {url}: {e}")

    print(f"[COMPLETE] Successfully downloaded {downloaded} real pantry images from images.cv into '{output_dir}'.")
    return downloaded

if __name__ == "__main__":
    download_images_cv_pantry(count=20)
