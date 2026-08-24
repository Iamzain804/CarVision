import cv2
import os
import time
from ultralytics import YOLO

# 1. Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "car.mp4"))
MODEL_PATH = os.path.join(BASE_DIR, "trained.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "annotated_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "annotated_car.mp4")

print(f"Loading Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 2. Video Capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Video '{VIDEO_PATH}' open nahi ho saki.")
    exit()

# Video Properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n🎬 Input Video Info:")
print(f"• Resolution: {width}x{height}")
print(f"• FPS: {fps:.2f}")
print(f"• Total Frames: {total_frames}")
print(f"• Saving to: {OUTPUT_VIDEO_PATH}\n")

# 3. Video Writer Setup (mp4v codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

frame_idx = 0
start_time = time.time()

print("Video processing start ho rahi hai...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    
    # YOLO Inference
    results = model.predict(source=frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()
    
    # Write annotated frame
    out.write(annotated_frame)
    
    # Progress Display
    if frame_idx % 25 == 0 or frame_idx == total_frames:
        percent = (frame_idx / total_frames) * 100
        elapsed = time.time() - start_time
        speed = frame_idx / elapsed if elapsed > 0 else 0
        print(f"Progress: [{frame_idx}/{total_frames}] ({percent:.1f}%) - Speed: {speed:.1f} FPS")

cap.release()
out.release()

total_time = time.time() - start_time
print(f"\n🎉 Mubarak ho! Annotated Video kamyabi se ban gayi hai:")
print(f"📁 Output File: {OUTPUT_VIDEO_PATH}")
print(f"⏱️ Total Time Taken: {total_time:.2f} seconds")
