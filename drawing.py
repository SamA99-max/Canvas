import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ==============================================================================
# AI AIR CANVAS STUDIO (MediaPipe + OpenCV)
# ==============================================================================
# Key Gestures:
#   - ✌️ TWO FINGERS UP (Index + Middle): Hover / Select Palette & Colors
#   - ☝️ ONE FINGER UP (Index Only): Draw on Canvas
#   - ✊ FIST / ALL DOWN: Pause Drawing / Move Hand Free
#
# Key Controls:
#   - 'c': Clear Canvas
#   - 'b': Toggle Background Mode (Webcam Feed vs. Dark Drawing Board)
#   - 's': Save Canvas Screenshot
#   - 'q' or ESC: Quit
# ==============================================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class EnhancedAirCanvas:
    """Manages the air drawing canvas, color selections, brush sizes, and UI palette."""
    
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color & Tool Palette Items
        self.palette_items = [
            {"type": "color", "label": "RED", "color": (0, 0, 255)},
            {"type": "color", "label": "ORANGE", "color": (0, 140, 255)},
            {"type": "color", "label": "YELLOW", "color": (0, 230, 255)},
            {"type": "color", "label": "GREEN", "color": (0, 255, 0)},
            {"type": "color", "label": "CYAN", "color": (255, 255, 0)},
            {"type": "color", "label": "BLUE", "color": (255, 50, 0)},
            {"type": "color", "label": "MAGENTA", "color": (255, 0, 255)},
            {"type": "color", "label": "WHITE", "color": (255, 255, 255)},
            {"type": "tool", "label": "ERASER", "action": "eraser", "bg": (50, 50, 50)},
            {"type": "tool", "label": "CLEAR", "action": "clear", "bg": (30, 30, 180)},
            {"type": "tool", "label": "SIZE -", "action": "size_down", "bg": (70, 70, 70)},
            {"type": "tool", "label": "SIZE +", "action": "size_up", "bg": (70, 70, 70)},
        ]
        
        self.active_color_idx = 1  # Default to Orange
        self.is_eraser_active = False
        self.brush_size = 8
        self.eraser_size = 45
        self.prev_x, self.prev_y = 0, 0
        self.last_action_time = 0  # Debounce timer for tool buttons
        self.dark_mode = False  # Dark board vs live video feed

    def draw_palette_ui(self, frame):
        """Draws top toolbar palette with colors, eraser, clear button, and brush size tools."""
        palette_height = 80
        num_items = len(self.palette_items)
        item_width = self.width // num_items

        # Top container bar background
        cv2.rectangle(frame, (0, 0), (self.width, palette_height), (20, 20, 20), -1)
        cv2.line(frame, (0, palette_height), (self.width, palette_height), (100, 100, 100), 2)

        for i, item in enumerate(self.palette_items):
            x1 = i * item_width
            y1 = 8
            x2 = (i + 1) * item_width - 4
            y2 = palette_height - 8

            # Highlight selected item
            is_selected = (not self.is_eraser_active and item.get("type") == "color" and i == self.active_color_idx) or \
                          (self.is_eraser_active and item.get("action") == "eraser")
            
            border_color = (0, 255, 255) if is_selected else (80, 80, 80)
            border_thick = 3 if is_selected else 1

            if item["type"] == "color":
                cv2.rectangle(frame, (x1, y1), (x2, y2), item["color"], -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)
                # Label text with shadow
                cv2.putText(frame, item["label"], (x1 + 6, y2 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                cv2.putText(frame, item["label"], (x1 + 6, y2 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            elif item["type"] == "tool":
                cv2.rectangle(frame, (x1, y1), (x2, y2), item["bg"], -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)
                cv2.putText(frame, item["label"], (x1 + 6, y2 - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw Active Brush Size and Status HUD Indicator
        status_color = (0, 0, 0) if self.is_eraser_active else self.palette_items[self.active_color_idx]["color"]
        curr_tool_name = "ERASER" if self.is_eraser_active else self.palette_items[self.active_color_idx]["label"]
        
        cv2.rectangle(frame, (10, self.height - 45), (320, self.height - 10), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, self.height - 45), (320, self.height - 10), (100, 100, 100), 1)
        cv2.circle(frame, (30, self.height - 27), 12, status_color, -1)
        cv2.circle(frame, (30, self.height - 27), 12, (255, 255, 255), 1)
        
        status_text = f"Tool: {curr_tool_name} | Size: {self.eraser_size if self.is_eraser_active else self.brush_size}px"
        cv2.putText(frame, status_text, (52, self.height - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def handle_selection_hover(self, ix, iy, frame):
        """Processes finger hover collisions over toolbar items (2 Fingers Up Mode)."""
        palette_height = 80
        num_items = len(self.palette_items)
        item_width = self.width // num_items

        if iy < palette_height:
            item_idx = min(ix // item_width, num_items - 1)
            item = self.palette_items[item_idx]

            # Visual hover highlight target box
            hx1 = item_idx * item_width
            hx2 = (item_idx + 1) * item_width
            cv2.rectangle(frame, (hx1, 0), (hx2, palette_height), (0, 255, 255), 2)

            curr_time = time.time()

            if item["type"] == "color":
                self.active_color_idx = item_idx
                self.is_eraser_active = False

            elif item["type"] == "tool" and (curr_time - self.last_action_time > 0.4):
                action = item["action"]
                if action == "eraser":
                    self.is_eraser_active = True
                elif action == "clear":
                    self.clear()
                    self.last_action_time = curr_time
                elif action == "size_up":
                    if self.is_eraser_active:
                        self.eraser_size = min(self.eraser_size + 10, 100)
                    else:
                        self.brush_size = min(self.brush_size + 4, 40)
                    self.last_action_time = curr_time
                elif action == "size_down":
                    if self.is_eraser_active:
                        self.eraser_size = max(self.eraser_size - 10, 10)
                    else:
                        self.brush_size = max(self.brush_size - 2, 2)
                    self.last_action_time = curr_time

    def draw_on_canvas(self, ix, iy):
        """Draws smooth lines or erases on canvas based on active finger tracking."""
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = ix, iy
            return

        if self.is_eraser_active:
            # Erase by painting black (0, 0, 0)
            cv2.line(self.canvas, (self.prev_x, self.prev_y), (ix, iy), (0, 0, 0), self.eraser_size)
        else:
            active_color = self.palette_items[self.active_color_idx]["color"]
            cv2.line(self.canvas, (self.prev_x, self.prev_y), (ix, iy), active_color, self.brush_size)

        self.prev_x, self.prev_y = ix, iy

    def clear(self):
        """Clears all drawing strokes from virtual canvas."""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)


def get_finger_states(landmarks):
    """
    Analyzes hand landmarks to detect extended fingers.
    Returns boolean states: (index_up, middle_up, ring_up, pinky_up)
    """
    # Landmark tips vs PIP joints
    index_up = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_up = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_up = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_up = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y

    return index_up, middle_up, ring_up, pinky_up


def draw_digital_hand_mesh(frame, hand_landmarks, w, h):
    """Draws a futuristic, colorful digital hand mesh directly on the frame."""
    # Define finger connection groups with neon BGR color schemes
    finger_connections = [
        # Palm / Wrist Connections -> Purple
        ([(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)], (200, 50, 200)),
        # Thumb -> Magenta
        ([(1, 2), (2, 3), (3, 4)], (255, 0, 255)),
        # Index -> Cyan
        ([(5, 6), (6, 7), (7, 8)], (255, 255, 0)),
        # Middle -> Neon Green
        ([(9, 10), (10, 11), (11, 12)], (0, 255, 128)),
        # Ring -> Bright Yellow
        ([(13, 14), (14, 15), (15, 16)], (0, 255, 255)),
        # Pinky -> Neon Orange
        ([(17, 18), (18, 19), (19, 20)], (50, 100, 255))
    ]

    # Convert normalized landmarks to pixel coordinates
    points = {}
    for idx, lm in enumerate(hand_landmarks.landmark):
        points[idx] = (int(lm.x * w), int(lm.y * h))

    # Draw glowing connection lines
    for connections, color in finger_connections:
        for p1, p2 in connections:
            cv2.line(frame, points[p1], points[p2], color, 2, cv2.LINE_AA)

    # Draw digital joint nodes
    for idx, pt in points.items():
        # Highlight Index Fingertip (Landmark 8) with outer target ring
        if idx == mp_hands.HandLandmark.INDEX_FINGER_TIP:
            cv2.circle(frame, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 11, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (120, 120, 120), 1, cv2.LINE_AA)


def generate_synthetic_frame(frame_count, width=1280, height=720):
    """Generates animated video feed when no physical camera is attached."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    t = frame_count * 0.04
    
    # Grid pattern background
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (30, 30, 30), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (30, 30, 30), 1)

    cx = int(width / 2 + math.cos(t) * 250)
    cy = int(height / 2 + math.sin(t * 1.2) * 150)

    cv2.circle(frame, (cx, cy), 30, (0, 140, 255), -1)
    cv2.putText(frame, "SIMULATED WEBCAM FEED (NO PHYSICAL CAMERA DETECTED)",
                (width // 2 - 320, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


def main():
    print("=" * 60)
    print("AI AIR CANVAS STUDIO - Pure Mid-Air Gesture Drawing")
    print("=" * 60)
    print("Gestures:")
    print("  ✌️  2 FINGERS UP (Index + Middle) -> Hover & Select Colors/Tools")
    print("  ☝️  1 FINGER UP  (Index Only)     -> Draw on Canvas")
    print("  ✊  FIST / HAND CLOSED           -> Pause / Move Pointer Free")
    print("Controls:")
    print("  'c' -> Clear Canvas")
    print("  'b' -> Toggle Dark Canvas Background")
    print("  's' -> Save Canvas Screenshot")
    print("  'q' -> Quit Application")
    print("=" * 60)

    # Initialize Video Capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    using_synthetic = False
    if not cap.isOpened():
        print("[WARNING] Primary camera not found. Launching Synthetic Generator...")
        using_synthetic = True

    air_canvas = EnhancedAirCanvas(1280, 720)
    hands_detector = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )

    prev_time = time.time()
    frame_count = 0

    while True:
        frame_count += 1
        if using_synthetic:
            frame = generate_synthetic_frame(frame_count)
            time.sleep(0.03)
        else:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame capture failed. Exiting...")
                break

        # Horizontal Flip for natural mirror response
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        if air_canvas.dark_mode:
            output_frame = air_canvas.canvas.copy()
        else:
            # Mask blending to superimpose drawing onto live camera video
            canvas_gray = cv2.cvtColor(air_canvas.canvas, cv2.COLOR_BGR2GRAY)
            _, inv_mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY_INV)
            inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
            
            output_frame = cv2.bitwise_and(frame, inv_mask)
            output_frame = cv2.bitwise_or(output_frame, air_canvas.canvas)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_digital_hand_mesh(output_frame, hand_landmarks, w, h)

                landmarks = hand_landmarks.landmark
                index_up, middle_up, ring_up, pinky_up = get_finger_states(landmarks)

                # Index Tip Pixel Coordinates
                ix = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                iy = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

                # Middle Tip Pixel Coordinates
                mx = int(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * w)
                my = int(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * h)

                # --------------------------------------------------------------
                # GESTURE MODE 1: SELECTION / HOVER MODE (Index + Middle Up)
                # --------------------------------------------------------------
                if index_up and middle_up and not ring_up:
                    air_canvas.prev_x, air_canvas.prev_y = 0, 0  # Reset line path
                    
                    # Pointer cursor midpoint
                    cx, cy = (ix + mx) // 2, (iy + my) // 2
                    
                    # Visual Selection Ring Cursor on output frame
                    cv2.circle(output_frame, (cx, cy), 15, (0, 255, 255), 2)
                    cv2.circle(output_frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.putText(output_frame, "SELECTING", (cx + 18, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                    # Trigger top bar collision handling
                    air_canvas.handle_selection_hover(cx, cy, output_frame)

                # --------------------------------------------------------------
                # GESTURE MODE 2: DRAWING MODE (Index Up Only)
                # --------------------------------------------------------------
                elif index_up and not middle_up and not ring_up:
                    # Draw Pointer Circle on output frame
                    brush_color = (100, 100, 100) if air_canvas.is_eraser_active else \
                                  air_canvas.palette_items[air_canvas.active_color_idx]["color"]
                    
                    radius = (air_canvas.eraser_size // 2) if air_canvas.is_eraser_active else \
                             max(air_canvas.brush_size // 2, 4)

                    cv2.circle(output_frame, (ix, iy), radius, brush_color, -1)
                    cv2.circle(output_frame, (ix, iy), radius + 2, (255, 255, 255), 1)

                    # Immediate line preview on output frame
                    if air_canvas.prev_x != 0 and air_canvas.prev_y != 0:
                        stroke_color = (0, 0, 0) if air_canvas.is_eraser_active else brush_color
                        stroke_size = air_canvas.eraser_size if air_canvas.is_eraser_active else air_canvas.brush_size
                        cv2.line(output_frame, (air_canvas.prev_x, air_canvas.prev_y), (ix, iy), stroke_color, stroke_size)

                    # Draw on internal canvas matrix
                    air_canvas.draw_on_canvas(ix, iy)

                # --------------------------------------------------------------
                # GESTURE MODE 3: HOVER / PAUSE MODE (Fist or other)
                # --------------------------------------------------------------
                else:
                    air_canvas.prev_x, air_canvas.prev_y = 0, 0

        else:
            air_canvas.prev_x, air_canvas.prev_y = 0, 0

        # Draw UI Toolbar on top of frame
        air_canvas.draw_palette_ui(output_frame)

        # Draw On-Screen Instructions & FPS HUD
        cv2.putText(output_frame, f"FPS: {int(fps)}", (w - 100, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(output_frame, "✌️ 2 Fingers: Select | ☝️ 1 Finger: Draw | 'b': Dark Mode | 'c': Clear",
                    (340, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        # Display output window
        cv2.imshow("Air Canvas Studio", output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Exiting Air Canvas Studio...")
            break
        elif key == ord('c') or key == ord('C'):
            air_canvas.clear()
            print("[ACTION] Canvas Cleared")
        elif key == ord('b') or key == ord('B'):
            air_canvas.dark_mode = not air_canvas.dark_mode
            mode_str = "Dark Canvas Board" if air_canvas.dark_mode else "Live Camera Feed"
            print(f"[ACTION] Switched background mode to: {mode_str}")
        elif key == ord('s') or key == ord('S'):
            filename = f"air_canvas_{int(time.time())}.png"
            cv2.imwrite(filename, output_frame)
            print(f"[ACTION] Saved canvas image to {filename}")

    # Cleanup
    if not using_synthetic:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()