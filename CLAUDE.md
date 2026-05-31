# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenMouse is an AI-powered browser automation tool that uses computer vision (YOLO) + a vision-language model (NVIDIA NIM) to autonomously navigate websites and complete user-specified objectives.

## Architecture

The system consists of three core modules orchestrated by an async loop in `main.py`:

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
│  (async loop: capture → detect → decide → act → cooldown)  │
└─────────────────────────────────────────────────────────────┘
         ↓                ↓                 ↓
   ┌──────────┐    ┌───────────┐    ┌─────────────┐
   │browser.py│    │vision.py  │    │brain.py     │
   │Playwright│    │YOLO + CV2 │    │NVIDIA NIM   │
   │automation│    │detection  │    │LLM decision │
   └──────────┘    └───────────┘    └─────────────┘
```

### Module Responsibilities

- **browser.py**: Manages Playwright browser lifecycle, captures screenshots, executes mouse/keyboard actions at coordinates, injects visual pointer feedback
- **vision.py**: Uses YOLOv8 to detect interactive elements, applies frame-difference filtering (OpenCV) to skip API calls on static screens, tags detected elements with sequential IDs
- **brain.py**: Sends annotated screenshots to NVIDIA NIM LLM with the user's objective, parses structured JSON responses (click/type/wait actions)

### Execution Flow

1. Capture screenshot via Playwright
2. Run YOLO detection + frame comparison in vision.py
3. If screen is static or no targets found, sleep and retry
4. Send annotated image to LLM with objective
5. LLM returns action + target ID + reasoning
6. Execute action at the coordinates mapped to that target ID
7. Cooldown delay, repeat

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Run the application
python main.py

# Configuration (via .env)
NVIDIA_API_KEY=nvapi-...     # Required: NVIDIA NIM API key
NVIDIA_NIM_MODEL=meta/llama3-70b-instruct  # Model selection
TARGET_URL=https://www.kaggle.com  # Starting URL
HEADLESS=false               # Show browser window
LOOP_DELAY_MS=400           # Delay between actions
FRAME_DIFF_THRESHOLD=0.995  # Skip VLM if frames match >99.5%
BROWSER_VIEWPORT_W=1280
BROWSER_VIEWPORT_H=720
```

## Key Implementation Details

- **Frame difference filter**: vision.py uses `cv2.matchTemplate` to compare consecutive frames. If similarity exceeds `FRAME_DIFF_THRESHOLD`, the system skips the expensive LLM call—critical for latency optimization.
- **Numerical tagging**: Detected YOLO boxes are tagged with sequential IDs (0, 1, 2...) drawn on the image. The LLM responds with which ID to click/type.
- **Coordinate mapping**: vision.py returns a `coordinates_map` dict mapping tag_id → (x, y) center coordinates. main.py looks up the LLM's target ID in this map.
- **JSON-structured LLM output**: brain.py uses `response_format={"type": "json_object"}` and Pydantic models to enforce reliable parsing. The LLM must return: `action` (click/type/wait), `target` (ID string), `text` (for typing), `reasoning`.
- **Visual feedback**: browser.py injects a red pointer dot via JavaScript to show where actions target.

## Edge Cases Handled

- Static screen: Skip LLM call, sleep and retry
- No targets detected: Sleep and retry
- Target ID not in coordinates_map: Warning log, continue loop
- API/parsing failure: Fallback to "wait" action
- KeyboardInterrupt: Graceful shutdown via finally block