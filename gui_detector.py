import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

# 1. Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "car.mp4"))
MODEL_PATH = os.path.join(BASE_DIR, "trained.pt")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")

# Screenshots folder create karein
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# 2. Model Load
print("Model load ho raha hai...")
model = YOLO(MODEL_PATH)

# 3. Video Capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Video '{VIDEO_PATH}' open nahi ho saki.")
    exit()

WINDOW_NAME = "Car Damage Detector - Realtime"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)

# 4. Interactive Confidence Threshold Slider (Trackbar)
# Initial threshold: 25% (0.25)
def on_trackbar(val):
    pass

cv2.createTrackbar("Confidence (%)", WINDOW_NAME, 25, 100, on_trackbar)

# Screenshot timer variables (Har 30 second baad auto screenshot)
SCREENSHOT_INTERVAL = 30  # seconds
last_screenshot_time = time.time()
screenshot_notification_time = 0
notification_text = ""

print("\n--------------------------------------------------")
print("🎥 Window open ho gayi hai!")
print("• Slider se Confidence Threshold adjust karein (0-100%).")
print(f"• Har {SCREENSHOT_INTERVAL} second baad automatic screenshot alag folder 'screenshots/' mein save hoga.")
print("• Manual screenshot lene ke liye 'S' dabayein.")
print("• Window band karne ke liye 'Q' ya 'ESC' dabayein.")
print("--------------------------------------------------\n")

fps_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    
    # Agar video khatam ho jaye to dobara shuru se chalayein (Loop playback)
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Slider se current threshold score lein
    slider_val = cv2.getTrackbarPos("Confidence (%)", WINDOW_NAME)
    conf_threshold = max(slider_val / 100.0, 0.05)  # Kam se kam 5% threshold

    # YOLO Detection with dynamic threshold
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()

    current_time = time.time()

    # 5. Har 30 second baad automatic screenshot lena
    time_since_last_ss = current_time - last_screenshot_time
    countdown = max(0, int(SCREENSHOT_INTERVAL - time_since_last_ss))

    # Auto screenshot trigger
    trigger_screenshot = False
    if time_since_last_ss >= SCREENSHOT_INTERVAL:
        trigger_screenshot = True
        last_screenshot_time = current_time

    # FPS calculation
    frame_count += 1
    if current_time - fps_time >= 1.0:
        fps = frame_count / (current_time - fps_time)
        frame_count = 0
        fps_time = current_time

    # UI Information Overlay
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Threshold: {int(conf_threshold * 100)}%", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Next Auto-Screenshot: {countdown}s", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' ya ESC se exit
        break
    elif key == ord('s') or key == ord('S'):  # 's' se manual screenshot
        trigger_screenshot = True

    # Screenshot save logic
    if trigger_screenshot:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ss_filename = f"damage_detection_{timestamp_str}.jpg"
        ss_path = os.path.join(SCREENSHOT_DIR, ss_filename)
        cv2.imwrite(ss_path, annotated_frame)
        notification_text = f"Saved: {ss_filename}"
        screenshot_notification_time = current_time
        print(f"📸 Screenshot saved: {ss_path}")

    # Agar abhi screenshot save hua ho to screen par notification dikhayein (for 2.5 seconds)
    if current_time - screenshot_notification_time < 2.5 and notification_text:
        cv2.putText(annotated_frame, f"📸 {notification_text}", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display in window
    cv2.imshow(WINDOW_NAME, annotated_frame)

cap.release()
cv2.destroyAllWindows()
print("\nWindow closed successfully.")
