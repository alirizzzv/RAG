"""Capture a demo GIF from the real Chainlit UI in a headless browser.

Drives an already-running instance (default http://localhost:8000): sends a
retrieval question and a chart question, screenshots each stage, and assembles
docs/demo.gif from the actual rendered frames.

Prereqs (dev only):
    pip install playwright && playwright install chromium
    chainlit run chainlit_app.py --port 8000 --headless   # in another shell
Run:
    python scripts/make_demo_gif.py
"""
import asyncio
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

URL = "http://localhost:8000"
OUT = Path(__file__).resolve().parent.parent / "docs" / "demo.gif"
FRAMES_DIR = OUT.parent / "_frames"
VIEWPORT = {"width": 1000, "height": 720}


async def _send(page, text):
    box = page.locator("#chat-input")
    await box.click()
    await page.keyboard.type(text)
    await page.keyboard.press("Enter")


async def _wait_for(page, needle, timeout=60):
    for _ in range(timeout):
        if needle in (await page.inner_text("body")):
            return True
        await asyncio.sleep(1)
    return False


async def capture():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport=VIEWPORT)
        await page.goto(URL, wait_until="domcontentloaded")
        await page.wait_for_selector("text=knowledge base", timeout=25000)
        await asyncio.sleep(2)
        f0 = FRAMES_DIR / "0.png"; await page.screenshot(path=f0); frames.append(f0)

        await _send(page, "What was Helios Energy total revenue in 2024?")
        await _wait_for(page, "679")
        await asyncio.sleep(1)
        f1 = FRAMES_DIR / "1.png"; await page.screenshot(path=f1); frames.append(f1)

        await _send(page, "Plot quarterly revenue for Northwind Robotics as a bar chart")
        await _wait_for(page, "Chart generated")
        await asyncio.sleep(2)
        f2 = FRAMES_DIR / "2.png"; await page.screenshot(path=f2); frames.append(f2)

        await browser.close()

    imgs = [Image.open(f).convert("RGB") for f in frames]
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:],
                 duration=[2500, 4000, 4500], loop=0, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(imgs)} frames)")


if __name__ == "__main__":
    asyncio.run(capture())
