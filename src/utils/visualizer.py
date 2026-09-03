import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from src.config import DAMAGE_COLOR_MAP, DEFAULT_DETECTION_COLOR, FIRE_COLOR

def draw_polygon_contour(
    image: np.ndarray,
    roi_bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (220, 20, 220),
    alpha: float = 0.35
) -> Tuple[np.ndarray, bool]:
    """
    Extracts precise defect contours inside the detection ROI using Canny edges
    and renders a semi-transparent colored polygon fill with a bright border.
    """
    x1, y1, x2, y2 = roi_bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image, False

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, False

    sig_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 25]
    if not sig_contours:
        return image, False

    largest_cnt = max(sig_contours, key=cv2.contourArea)
    shifted_cnt = largest_cnt + np.array([x1, y1])
    epsilon = 0.015 * cv2.arcLength(shifted_cnt, True)
    approx_polygon = cv2.approxPolyDP(shifted_cnt, epsilon, True)

    overlay = image.copy()
    cv2.fillPoly(overlay, [approx_polygon], color)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
    cv2.polylines(image, [approx_polygon], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    return image, True

def draw_damage_annotation(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    polygon_mode: bool = True
) -> np.ndarray:
    """
    Draws damage detections on the frame with either polygon contours or standard bounding boxes.
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['class_name']
        conf = det['confidence']
        color = DAMAGE_COLOR_MAP.get(cls_name, DEFAULT_DETECTION_COLOR)

        drawn = False
        if polygon_mode:
            annotated, drawn = draw_polygon_contour(annotated, (x1, y1, x2, y2), color=color)

        if not drawn:
            # Fallback to stylish bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label badge
        label = f"{cls_name.replace('_', ' ').title()} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y1 = max(0, y1 - text_h - 8)
        label_y2 = label_y1 + text_h + 8
        label_x2 = min(annotated.shape[1], x1 + text_w + 10)

        # Label background pill
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, label_y2 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    return annotated

def draw_fire_segmentation(
    frame: np.ndarray,
    results,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlays fire instance segmentation masks with vivid red/orange colors and borders.
    """
    annotated = frame.copy()
    if not results or len(results) == 0 or results[0].masks is None:
        return annotated

    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes

    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8)

        # Colored mask overlay
        color_layer = np.zeros_like(frame, dtype=np.uint8)
        color_layer[:] = FIRE_COLOR

        mask_indices = binary_mask == 1
        annotated[mask_indices] = cv2.addWeighted(
            annotated[mask_indices], 1.0 - alpha,
            color_layer[mask_indices], alpha, 0
        )

        # Find contours of the fire mask for a crisp neon border
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)

        # Label
        if i < len(boxes):
            conf = float(boxes[i].conf[0])
            x1, y1, _, _ = map(int, boxes[i].xyxy[0])
            lbl = f"FIRE {conf:.0%}"
            cv2.putText(annotated, lbl, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated

def draw_telemetry_hud(
    frame: np.ndarray,
    fps: float = 0.0,
    latency_ms: float = 0.0,
    detections_count: int = 0,
    status_text: str = "Active",
    extra_info: str = ""
) -> np.ndarray:
    """
    Renders a sleek, modern HUD card on the top of the video/image frame.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent HUD header bar
    hud_h = 44
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Accent separator line
    cv2.line(annotated, (0, hud_h), (w, hud_h), (0, 180, 255), 2)

    # Text elements
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, f"CarVision AI", (14, 28), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    metrics_str = f"FPS: {fps:4.1f} | Latency: {latency_ms:5.1f}ms | Detected: {detections_count}"
    if extra_info:
        metrics_str += f" | {extra_info}"

    cv2.putText(annotated, metrics_str, (170, 28), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Status badge on right
    status_w = 120
    status_x = max(200, w - status_w - 15)
    badge_color = (0, 200, 0) if "Active" in status_text or "OK" in status_text else (0, 140, 255)
    cv2.putText(annotated, f"● {status_text}", (status_x, 28), font, 0.55, badge_color, 2, cv2.LINE_AA)

    return annotated
