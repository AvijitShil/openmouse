import os
import asyncio
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

class BrowserEngine:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.headless = os.getenv("HEADLESS", "false").lower() == "true"
        self.width = int(os.getenv("BROWSER_VIEWPORT_W", 1280))
        self.height = int(os.getenv("BROWSER_VIEWPORT_H", 720))

    async def initialize(self):
        """Starts Playwright with an isolated profile and a consistent viewport."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        await self.page.set_viewport_size({"width": self.width, "height": self.height})
        return self.page

    async def get_frame_bytes(self) -> bytes:
        """Captures compressed JPEG byte arrays directly to RAM to bypass disk latency."""
        return await self.page.screenshot(type='jpeg', quality=75)

    async def execute_action(self, action: str, x: int, y: int, text: str = None):
        """Performs precise headless synthetic actions on target coordinate indices."""
        # Visual feedback: Render a temporary red tracking crosshair on screen for debugging
        await self.page.evaluate(f"""() => {{
            let dot = document.getElementById('openmouse-pointer');
            if (!dot) {{
                dot = document.createElement('div');
                dot.id = 'openmouse-pointer';
                dot.style.position = 'absolute';
                dot.style.width = '12px';
                dot.style.height = '12px';
                dot.style.borderRadius = '50%';
                dot.style.backgroundColor = 'red';
                dot.style.border = '2px solid white';
                dot.style.zIndex = '100000';
                dot.style.pointerEvents = 'none';
                document.body.appendChild(dot);
            }}
            dot.style.left = '{x - 6}px';
            dot.style.top = '{y - 6}px';
        }}""")

        # Execute physical action mechanics
        await self.page.mouse.move(x, y, steps=3)
        if action == "click":
            await self.page.mouse.click(x, y)
        elif action == "type" and text:
            await self.page.mouse.click(x, y)
            await self.page.keyboard.press("Control+A")
            await self.page.keyboard.press("Backspace")
            await self.page.keyboard.type(text, delay=30)

    async def shutdown(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()