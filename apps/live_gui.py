import sys
import os
from src.video_pipeline import VideoPipeline
from src.damage_detector import CarDamageDetector
from src.config import DEFAULT_DAMAGE_CONF

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/videos/car.mp4"
    print("Initializing Real-Time Interactive Car Damage GUI...")
    detector = CarDamageDetector(conf=DEFAULT_DAMAGE_CONF)
    pipeline = VideoPipeline(detector)
    pipeline.stream_realtime(video_path, seek_step=5, initial_conf=DEFAULT_DAMAGE_CONF, polygon_mode=True)

if __name__ == "__main__":
    main()
