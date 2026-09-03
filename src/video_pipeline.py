import os
import cv2
import time
from datetime import datetime
from typing import Optional, Callable
from src.config import (
    DEFAULT_DAMAGE_CONF,
    SCREENSHOTS_DIR,
    OUTPUT_VIDEOS_DIR
)
from src.utils.media_loader import open_video
from src.damage_detector import CarDamageDetector

class VideoPipeline:
    """
    Video processing pipeline supporting:
    1. Realtime live playback with 5th frame hardware skip, OpenCV trackbar slider, and auto-screenshot logging.
    2. Batch video file annotation and MP4 export.
    """
    def __init__(self, detector: Optional[CarDamageDetector] = None):
        self.detector = detector or CarDamageDetector()

    def stream_realtime(
        self,
        video_source: str,
        seek_step: int = 5,
        initial_conf: float = DEFAULT_DAMAGE_CONF,
        screenshot_interval: int = 30,
        polygon_mode: bool = True
    ):
        """
        Runs real-time interactive playback with zero latency seek and dynamic confidence slider.
        """
        cap, meta, resolved_path = open_video(video_source)
        window_name = "CarVision AI - Realtime Damage Inspector"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1020, 580)

        # Trackbar for dynamic confidence
        init_val = int(initial_conf * 100)
        cv2.createTrackbar("Confidence (%)", window_name, init_val, 100, lambda x: None)

        print("\n==================================================")
        print(f"🎬 Video Stream Started: {os.path.basename(resolved_path)}")
        print(f"• Resolution: {meta['width']}x{meta['height']} @ {meta['fps']:.1f} FPS")
        print(f"• Frame Step: Every {seek_step}th frame")
        print(f"• Auto-screenshot: Every {screenshot_interval} seconds to outputs/screenshots/")
        print("• Controls:")
        print("   - Trackbar: Drag slider to adjust Confidence (0-100%)")
        print("   - 'S' key: Capture manual screenshot")
        print("   - 'Q' or ESC: Quit stream")
        print("==================================================\n")

        last_screenshot_time = time.time()
        fps_start = time.time()
        frame_counter = 0
        fps_val = 0.0

        current_frame_pos = 0

        while True:
            # 5th frame hardware direct seek
            if not meta['is_camera']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                current_frame_pos += seek_step

            ret, frame = cap.read()
            if not ret:
                if meta['is_camera']:
                    break
                # Loop video playback
                current_frame_pos = 0
                continue

            frame_counter += 1
            if time.time() - fps_start >= 1.0:
                fps_val = frame_counter / (time.time() - fps_start)
                frame_counter = 0
                fps_start = time.time()

            # Dynamic confidence score from slider
            slider_pos = cv2.getTrackbarPos("Confidence (%)", window_name)
            active_conf = max(slider_pos / 100.0, 0.05)

            # Inference
            annotated_frame, detections, latency = self.detector.process_frame(
                frame,
                conf=active_conf,
                polygon_mode=polygon_mode,
                add_hud=True,
                fps=fps_val
            )

            # Screenshot handling
            now = time.time()
            trigger_ss = False
            ss_type = "auto"

            if now - last_screenshot_time >= screenshot_interval:
                trigger_ss = True
                ss_type = "auto"
                last_screenshot_time = now

            # Render countdown on screen
            time_left = max(0, int(screenshot_interval - (now - last_screenshot_time)))
            cv2.putText(
                annotated_frame,
                f"Next Screenshot: {time_left}s",
                (annotated_frame.shape[1] - 220, annotated_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('s'), ord('S')]:
                trigger_ss = True
                ss_type = "manual"

            if trigger_ss:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ss_filename = f"damage_{ss_type}_{timestamp}.jpg"
                ss_path = os.path.join(SCREENSHOTS_DIR, ss_filename)
                cv2.imwrite(ss_path, annotated_frame)
                print(f"📸 [{ss_type.upper()}] Screenshot saved: {ss_path}")

        cap.release()
        cv2.destroyAllWindows()
        print("Playback ended.")

    def annotate_video_file(
        self,
        video_source: str,
        output_path: Optional[str] = None,
        conf: float = DEFAULT_DAMAGE_CONF,
        polygon_mode: bool = False,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> str:
        """
        Batch annotates an entire video file frame-by-frame and exports an annotated MP4.
        """
        cap, meta, resolved_path = open_video(video_source)
        if not output_path:
            base_name = os.path.splitext(os.path.basename(resolved_path))[0]
            output_path = os.path.join(OUTPUT_VIDEOS_DIR, f"{base_name}_annotated.mp4")

        width = meta['width']
        height = meta['height']
        fps = meta['fps']
        total_frames = meta['total_frames']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("\n==================================================")
        print(f"🎬 Batch Annotating Video: {os.path.basename(resolved_path)}")
        print(f"• Total Frames: {total_frames} | FPS: {fps:.1f} | Resolution: {width}x{height}")
        print(f"• Target Output: {output_path}")
        print("==================================================")

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            annotated_frame, _, _ = self.detector.process_frame(
                frame, conf=conf, polygon_mode=polygon_mode, add_hud=False
            )
            writer.write(annotated_frame)

            elapsed = time.time() - start_time
            speed = frame_idx / elapsed if elapsed > 0 else 0

            if frame_idx % 25 == 0 or frame_idx == total_frames:
                percent = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"Progress: [{frame_idx}/{total_frames}] ({percent:.1f}%) - Speed: {speed:.1f} FPS")
                if progress_callback:
                    progress_callback(frame_idx, total_frames, speed)

        cap.release()
        writer.release()

        total_time = time.time() - start_time
        print(f"\n🎉 Video annotation complete in {total_time:.1f}s!")
        print(f"📁 Output File: {output_path}")
        return output_path
