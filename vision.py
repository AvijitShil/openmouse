import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

class PerceptionPipeline:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.previous_gray_frame = None
        self.diff_threshold = float(os.getenv("FRAME_DIFF_THRESHOLD", 0.995))

    def is_screen_static(self, current_gray) -> bool:
        """Compares current frame against the previous frame to avoid running slow logic on static pages."""
        if self.previous_gray_frame is None:
            self.previous_gray_frame = current_gray
            return False

        # Fast structural pixel variance computation
        res = cv2.matchTemplate(current_gray, self.previous_gray_frame, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        self.previous_gray_frame = current_gray
        return max_val >= self.diff_threshold

    def process_and_annotate(self, frame_bytes):
        """Processes raw bytes into actionable base64 strings and spatial mapping coordinates."""
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check if page has actually visually updated since the last run
        if self.is_screen_static(gray):
            return None, {}

        coordinates_map = {}
        results = self.model(img, verbose=False)

        element_counter = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Sanity sizing filters for web interaction points
            if w > 10 and h > 10:
                center_x = x1 + (w // 2)
                center_y = y1 + (h // 2)

                tag_id = str(element_counter)
                coordinates_map[tag_id] = (center_x, center_y)

                # Render explicit Set-of-Mark tags cleanly over items
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.rectangle(img, (x1, y1 - 15), (x1 + 30, y1), (0, 255, 0), -1)
                cv2.putText(img, tag_id, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                element_counter += 1

        # High-efficiency base64 encoding straight out of RAM
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        b64_string = base64.b64encode(buffer).decode('utf-8')

        return b64_string, coordinates_map