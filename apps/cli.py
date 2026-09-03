import argparse
import os
import sys
import cv2
from src.config import (
    DEFAULT_DAMAGE_CONF,
    DEFAULT_FIRE_CONF,
    OUTPUT_IMAGES_DIR,
    OUTPUT_VIDEOS_DIR
)
from src.damage_detector import CarDamageDetector
from src.fire_segmenter import FireSegmenter
from src.video_pipeline import VideoPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="🚗 CarVision & FireGuard AI - Unified Inspection CLI"
    )
    parser.add_argument(
        "--mode",
        choices=["damage-image", "damage-video", "annotate-video", "fire-image", "fire-video"],
        default="damage-video",
        help="Inference mode (default: damage-video)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input image/video path, directory, or camera index '0'"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold score (0.05 to 1.0)"
    )
    parser.add_argument(
        "--polygon",
        action="store_true",
        default=True,
        help="Enable precise neon polygon contours for damages (default: True)"
    )
    parser.add_argument(
        "--bbox-only",
        action="store_true",
        help="Force bounding boxes only instead of polygon contours"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file or folder path"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Frame skip step for real-time video stream (default: 5)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run headless without opening OpenCV preview window"
    )
    return parser.parse_args()

def run_damage_image(source: str, conf: float, polygon_mode: bool, output: str, no_show: bool):
    detector = CarDamageDetector(conf=conf or DEFAULT_DAMAGE_CONF)
    source_path = source or "data/images/download.jpg"
    annotated, detections, latency, resolved_path = detector.process_image_file(
        source_path, conf=conf, polygon_mode=polygon_mode
    )

    base_name = os.path.basename(resolved_path)
    out_dir = output if output and os.path.isdir(output) else OUTPUT_IMAGES_DIR
    out_file = output if output and not os.path.isdir(output) else os.path.join(out_dir, f"annotated_{base_name}")
    cv2.imwrite(out_file, annotated)

    print(f"\n==================================================")
    print(f"🖼️ Car Damage Detection Complete")
    print(f"• Input: {resolved_path}")
    print(f"• Latency: {latency:.2f} ms")
    print(f"• Detected Defects: {len(detections)}")
    for i, d in enumerate(detections, 1):
        print(f"   [{i}] {d['class_name']} ({d['confidence']:.1%}) - Box: {d['bbox']}")
    print(f"• Saved To: {out_file}")
    print(f"==================================================")

    if not no_show:
        win_name = f"Car Damage Result - {base_name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, annotated)
        print("\nPress any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def run_fire_image(source: str, conf: float, output: str, no_show: bool):
    segmenter = FireSegmenter(conf=conf or DEFAULT_FIRE_CONF)
    source_path = source or "data/images/download_2.jpg"
    annotated, count, latency, resolved_path = segmenter.process_image_file(
        source_path, conf=conf
    )

    base_name = os.path.basename(resolved_path)
    out_dir = output if output and os.path.isdir(output) else OUTPUT_IMAGES_DIR
    out_file = output if output and not os.path.isdir(output) else os.path.join(out_dir, f"fire_seg_{base_name}")
    cv2.imwrite(out_file, annotated)

    print(f"\n==================================================")
    print(f"🔥 Fire Instance Segmentation Complete")
    print(f"• Input: {resolved_path}")
    print(f"• Latency: {latency:.2f} ms")
    print(f"• Fire Masks Found: {count}")
    print(f"• Saved To: {out_file}")
    print(f"==================================================")

    if not no_show:
        win_name = f"Fire Segmentation Result - {base_name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, annotated)
        print("\nPress any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def run_damage_video_stream(source: str, conf: float, polygon_mode: bool, step: int):
    detector = CarDamageDetector(conf=conf or DEFAULT_DAMAGE_CONF)
    pipeline = VideoPipeline(detector)
    source_path = source or "data/videos/car.mp4"
    pipeline.stream_realtime(
        video_source=source_path,
        seek_step=step,
        initial_conf=conf or DEFAULT_DAMAGE_CONF,
        polygon_mode=polygon_mode
    )

def run_annotate_video(source: str, conf: float, polygon_mode: bool, output: str):
    detector = CarDamageDetector(conf=conf or DEFAULT_DAMAGE_CONF)
    pipeline = VideoPipeline(detector)
    source_path = source or "data/videos/car.mp4"
    pipeline.annotate_video_file(
        video_source=source_path,
        output_path=output,
        conf=conf or DEFAULT_DAMAGE_CONF,
        polygon_mode=polygon_mode
    )

def main():
    args = parse_args()
    polygon_mode = False if args.bbox_only else args.polygon

    if args.mode == "damage-image":
        run_damage_image(args.source, args.conf, polygon_mode, args.output, args.no_show)
    elif args.mode == "fire-image":
        run_fire_image(args.source, args.conf, args.output, args.no_show)
    elif args.mode == "damage-video":
        run_damage_video_stream(args.source, args.conf, polygon_mode, args.step)
    elif args.mode == "annotate-video":
        run_annotate_video(args.source, args.conf, polygon_mode, args.output)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
