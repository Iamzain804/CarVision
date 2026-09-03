import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO
from src.config import CAR_DAMAGE_MODEL_PATH, DEFAULT_DAMAGE_CONF
from src.utils.visualizer import draw_damage_annotation, draw_telemetry_hud
from src.utils.media_loader import load_image

class CarDamageDetector:
    """
    Production-ready vehicle damage detector using fine-tuned YOLO11m.
    Detects dents, scratches, cracks, shattered glass, broken lamps, and flat tires.
    Supports both precise neon polygon contour overlays and bounding boxes.
    """
    def __init__(self, model_path: Optional[str] = None, conf: float = DEFAULT_DAMAGE_CONF):
        self.model_path = model_path or CAR_DAMAGE_MODEL_PATH
        self.conf = conf

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading Car Damage YOLO11m model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

    def predict(self, frame: np.ndarray, conf: Optional[float] = None, verbose: bool = False):
        """Runs raw YOLO inference on a single image or video frame."""
        conf_threshold = conf if conf is not None else self.conf
        return self.model.predict(source=frame, conf=conf_threshold, verbose=verbose)

    def extract_detections(self, results) -> List[Dict[str, Any]]:
        """Parses YOLO results into structured dictionary objects."""
        detections = []
        if not results:
            return detections

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                confidence = float(box.conf[0])

                detections.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'area': (x2 - x1) * (y2 - y1)
                })
        return detections

    def process_frame(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        polygon_mode: bool = True,
        add_hud: bool = False,
        fps: float = 0.0
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Processes a single video frame or image array.
        Returns: (annotated_frame, detections_list, latency_ms)
        """
        start = time.perf_counter()
        results = self.predict(frame, conf=conf, verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000.0

        detections = self.extract_detections(results)
        annotated = draw_damage_annotation(frame, detections, polygon_mode=polygon_mode)

        if add_hud:
            status = "Damage Detected" if detections else "Inspection Clear"
            annotated = draw_telemetry_hud(
                annotated,
                fps=fps,
                latency_ms=latency_ms,
                detections_count=len(detections),
                status_text=status
            )

        return annotated, detections, latency_ms

    def process_image_file(
        self,
        image_path: str,
        conf: Optional[float] = None,
        polygon_mode: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float, str]:
        """
        Loads an image from file path and detects damage.
        Returns: (annotated_image, detections, latency_ms, resolved_path)
        """
        frame, resolved_path = load_image(image_path)
        annotated, detections, latency = self.process_frame(
            frame, conf=conf, polygon_mode=polygon_mode, add_hud=False
        )
        return annotated, detections, latency, resolved_path
