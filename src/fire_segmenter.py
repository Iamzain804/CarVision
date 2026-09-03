import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO
from src.config import FIRE_SEG_MODEL_PATH, DEFAULT_FIRE_CONF
from src.utils.visualizer import draw_fire_segmentation, draw_telemetry_hud
from src.utils.media_loader import load_image

class FireSegmenter:
    """
    Real-time fire and hazard segmentation engine using fine-tuned YOLO11n-seg.
    Detects fire instances and provides pixel-level instance segmentation masks.
    """
    def __init__(self, model_path: Optional[str] = None, conf: float = DEFAULT_FIRE_CONF):
        self.model_path = model_path or FIRE_SEG_MODEL_PATH
        self.conf = conf

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Fire segmentation model file not found: {self.model_path}")

        print(f"Loading Fire Segmentation YOLO11n model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

    def predict(self, frame: np.ndarray, conf: Optional[float] = None, verbose: bool = False):
        """Runs raw YOLO instance segmentation on image or frame."""
        conf_threshold = conf if conf is not None else self.conf
        return self.model.predict(source=frame, conf=conf_threshold, verbose=verbose)

    def process_frame(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        add_hud: bool = False,
        fps: float = 0.0
    ) -> Tuple[np.ndarray, int, float, Any]:
        """
        Processes a single video frame or image array.
        Returns: (annotated_frame, fire_instances_count, latency_ms, raw_results)
        """
        start = time.perf_counter()
        results = self.predict(frame, conf=conf, verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000.0

        annotated = draw_fire_segmentation(frame, results)

        mask_count = 0
        if results and len(results) > 0 and results[0].masks is not None:
            mask_count = len(results[0].masks)

        if add_hud:
            status = "FIRE HAZARD DETECTED!" if mask_count > 0 else "Hazard Clear"
            annotated = draw_telemetry_hud(
                annotated,
                fps=fps,
                latency_ms=latency_ms,
                detections_count=mask_count,
                status_text=status
            )

        return annotated, mask_count, latency_ms, results

    def process_image_file(
        self,
        image_path: str,
        conf: Optional[float] = None
    ) -> Tuple[np.ndarray, int, float, str]:
        """
        Loads an image from file path and runs fire segmentation.
        Returns: (annotated_image, fire_count, latency_ms, resolved_path)
        """
        frame, resolved_path = load_image(image_path)
        annotated, count, latency, _ = self.process_frame(frame, conf=conf, add_hud=False)
        return annotated, count, latency, resolved_path
