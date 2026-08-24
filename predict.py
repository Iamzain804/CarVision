from ultralytics import YOLO
import sys
import os

def main():
    # Model load karein
    model_path = "trained.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} nahi mila.")
        return

    print("Model load ho raha hai...")
    model = YOLO(model_path)
    
    # Image path: agar argument pass kiya hai to wo lein, warna default 'public/1.png'
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("public", "1.png")
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' nahi mili.")
        return

    print(f"Detecting damage on: {image_path}")
    results = model.predict(source=image_path, save=True, conf=0.25, project="runs", name="predict")

    for result in results:
        print("\n--- Detection Results ---")
        boxes = result.boxes
        if len(boxes) == 0:
            print("Koi damage detect nahi hua.")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            print(f"Damage Found: {cls_name} (Confidence: {conf:.2%})")
            
        print(f"\nResult image save ho gayi hai: {result.save_dir}")

if __name__ == "__main__":
    main()
