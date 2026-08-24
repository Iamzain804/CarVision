import cv2
import os
import time
from ultralytics import YOLO

# 1. Paths configure karein
VIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "car.mp4"))
MODEL_PATH = "trained.pt"

print(f"Loading Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 2. Video Capture - Direct Seek (Zero decoding latency, No loop decisions, No OOP Class)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Video '{VIDEO_PATH}' open nahi ho saki.")
    exit()

# 5th Frame par direct jump (0-indexed frame 4 = 5th frame)
# Is se pehle ke 4 frames decode nahi hotay aur loop ke andar koi if/else decision nahi lena parta
cap.set(cv2.CAP_PROP_POS_FRAMES, 4)

start_time = time.perf_counter()
success, frame = cap.read()
cap.release()

if not success or frame is None:
    print("Error: 5th frame read nahi ho saka.")
    exit()

# 3. Fast Inference (Low Latency)
results = model.predict(
    source=frame,
    save=True,
    conf=0.25,
    project="runs",
    name="video_frame_5",
    verbose=False
)

latency = (time.perf_counter() - start_time) * 1000

# 4. Results Display
print(f"\n✅ 5th Frame processed successfully!")
print(f"⏱️ Total Latency (Read + Inference): {latency:.2f} ms")

for r in results:
    print("\n--- Detected Damages on 5th Frame ---")
    if len(r.boxes) == 0:
        print("Koi damage detect nahi hua.")
    for box in r.boxes:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"• {cls_name}: {conf:.2%}")
        
    print(f"\nResult save location: {r.save_dir}")
