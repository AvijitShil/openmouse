import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv
from browser import BrowserEngine
from vision import PerceptionPipeline
from brain import BrainController

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="OpenMouse - AI Browser Automation")
    parser.add_argument("url", nargs="?", default=None, help="Target website URL (e.g., https://example.com)")
    parser.add_argument("--objective", "-o", default=None, help="Objective for OpenMouse to execute")
    args = parser.parse_args()

    if not os.getenv("NVIDIA_API_KEY"):
        print("[Error] Missing NVIDIA_API_KEY inside the local .env configuration file.")
        sys.exit(1)

    # Use provided URL, .env value, or prompt
    target_url = args.url or os.getenv("TARGET_URL")
    if not target_url:
        target_url = input("Enter the website URL to navigate to: ").strip()

    # Use provided objective or prompt
    objective = args.objective
    if not objective:
        objective = input("Enter what you want OpenMouse to execute: ").strip()

    # Initialize Core Engines
    browser = BrowserEngine()
    perception = PerceptionPipeline()
    brain = BrainController()

    page = await browser.initialize()

    print(f"\n[OpenMouse] Navigating to: {target_url}")
    await page.goto(target_url)
    await asyncio.sleep(4)

    print("[OpenMouse] Loop engaged. Press Ctrl+C to terminate operation safely.\n")

    try:
        while True:
            # 1. Grab raw state directly into memory buffers
            frame_bytes = await browser.get_frame_bytes()

            # 2. Run fast localized layout updates
            b64_image, coordinates_map = perception.process_and_annotate(frame_bytes)

            # 3. Handle Latency-Saving Static States
            if b64_image is None:
                print("[OpenMouse Status] Screen is static. Waiting for changes or page loads...")
                await asyncio.sleep(1.0)
                continue

            # 4. Handle Empty Frame edge case
            if not coordinates_map:
                print("[OpenMouse Status] No interactive targets found. Waiting to reload layout tags...")
                await asyncio.sleep(1.5)
                continue

            # 5. Extract structural next steps via NIM VLM
            decision = await brain.request_next_step(b64_image, objective)
            action = decision.get("action")
            target_id = str(decision.get("target"))
            text_value = decision.get("text")
            reasoning = decision.get("reasoning", "No reason provided.")

            print(f"[Brain Processing] Reason: {reasoning}")
            print(f"[Action Selected] {action.upper()} -> Target ID Tag: {target_id}")

            # 6. Execute programmatic mouse pointer translation
            if target_id in coordinates_map:
                target_x, target_y = coordinates_map[target_id]
                await browser.execute_action(action, target_x, target_y, text_value)
            else:
                print(f"[Warning] Targeted identifier '{target_id}' was missing from the coordinates table map.")

            # Cooldown execution delay padding config
            cooldown_time = float(os.getenv("LOOP_DELAY_MS", 400)) / 1000.0
            await asyncio.sleep(cooldown_time)

    except KeyboardInterrupt:
        print("\n[System Execution Interrupted] OpenMouse autonomous process loop terminating safely.")
    finally:
        await browser.shutdown()

if __name__ == "__main__":
    asyncio.run(main())