import cv2
import os
import time
from ultralytics import YOLO

VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "car.mp4"))
MODEL_PATH = "trained.pt"

print("Model load ho raha hai...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Video '{VIDEO_PATH}' nahi mili.")
    exit()

print("Har 5th frame process ho raha hai (Bina loop ke andar if-condition/decision liye)...")
frame_idx = 0
start_time = time.perf_counter()

# Loop ke andar koi decision (if count % 5 == 0) nahi liya ja raha.
# read() se sirf 5th frame decode hota hai, aur grab() se darmiyan ke 4 frames bina decode kiye skip hotay hain (Zero Decoding Latency)
while True:
    ret, frame = cap.read()  # Decode and read target frame
    if not ret:
        break
    
    frame_idx += 5
    
    # Fast Inference
    results = model.predict(source=frame, save=False, conf=0.25, verbose=False)
    
    # Detections
    damages = [model.names[int(b.cls[0])] for r in results for b in r.boxes]
    print(f"Frame {frame_idx:03d}: {len(damages)} damages -> {damages}")
    
    # Uncompressed fast skip of 4 frames (Zero Latency)
    cap.grab()
    cap.grab()
    cap.grab()
    cap.grab()

cap.release()
total_time = time.perf_counter() - start_time
print(f"\n✅ Video Completed in {total_time:.2f} seconds!")
