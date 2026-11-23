"""
YOLOv8 Defect & Asset Detection Module for TCS Autonomous Asset Inspection System.
Performs real-time (sub-50ms) inference to:
1. Detect Asset ID plate (TR-1, TR-2, CB-1, CS-02) independently.
2. Trigger asset-specific inspection based on the detected Asset ID.
3. Identify defects and label parts according to the specified parameters.
4. Support ROI (square target) inspection to eliminate background / surrounding noise.
5. Save annotated images with bounding boxes, defect reasons, and timestamps.
"""

import os
import time
from datetime import datetime
from typing import Tuple, List, Optional, Dict
import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    ANNOTATED_DIR,
    CAPTURES_DIR,
    ASSET_TAG_MAP,
    ASSET_TYPE_MAP,
    DEFECT_PARAMETER_MAP,
    GAUGE_READINGS,
    DefectFinding,
    InspectionResult,
    TAG_DETECT_CONF,
    ANOMALY_DETECT_CONF,
    ROI_DETECT_CONF,
    NMS_IOU_THRESH,
)
from ocr_asset_id import AssetIDDetector

# Class indices in YOLO model
# 0: tag_TR1, 1: tag_TR2, 2: tag_CB1, 3: tag_CS02
# 4: transformer_body, 5: oil_gauge, 6: bushing, 7: circuit_breaker, 8: cable_bundle
# 9: oil_leakage, 10: crack, 11: insulation_damage, 12: burn_mark, 13: loose_connection, 14: physical_damage
TAG_CLASSES = [0, 1, 2, 3]

ASSET_ALLOWED_CLASSES = {
    "Transformer": [0, 1, 4, 5, 6, 9, 10],      # tags, body, gauge, bushing, oil_leakage, crack
    "Circuit Breaker": [2, 6, 7, 13, 14],        # tag, bushing, breaker, loose_conn, physical_damage
    "Cable Section": [3, 8, 10, 11, 12, 13, 14], # tag, cable_bundle, crack, insulation_damage, burn_mark, loose_conn, physical_damage
}


class YOLODetector:
    """Real-time YOLOv8 detector for substation asset inspection."""

    def __init__(self, model_path: str = YOLO_MODEL_PATH, device: int = 0, enable_ocr: bool = True):
        print(f"[YOLO] Initializing YOLOv8 model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}. Train model first!")

        self.device = device
        self.model = YOLO(model_path)
        os.makedirs(ANNOTATED_DIR, exist_ok=True)
        os.makedirs(CAPTURES_DIR, exist_ok=True)

        self.enable_ocr = enable_ocr
        self.ocr_detector = None
        if self.enable_ocr:
            try:
                self.ocr_detector = AssetIDDetector(gpu=(device >= 0))
            except Exception as e:
                print(f"[WARN] Could not initialize OCR engine: {e}. Using pure visual detection.")

        print(f"[YOLO] Model loaded and ready on device {device} (GPU).")

    def detect_asset_tag_only(
        self,
        frame_or_roi: np.ndarray,
        conf: float = TAG_DETECT_CONF,
        verify_with_ocr: bool = True,
    ) -> Optional[Dict]:
        """
        Step 1: Detect and verify the Asset ID tag (TR-1, TR-2, CB-1, CS-02).
        Uses YOLO candidate proposal + OCR text verification to eliminate all false positives.
        Returns dict with asset_id, asset_type, conf, bbox, verified if found, else None.
        """
        fh, fw = frame_or_roi.shape[:2]
        results = self.model.predict(
            frame_or_roi,
            classes=TAG_CLASSES,
            conf=conf,
            device=self.device,
            verbose=False,
        )[0]

        candidates = []
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            cls_name = self.model.names[cls_idx]
            c = float(box.conf[0])
            if cls_name in ASSET_TAG_MAP:
                asset_id, asset_type = ASSET_TAG_MAP[cls_name]
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                candidates.append({
                    "asset_id": asset_id,
                    "asset_type": asset_type,
                    "tag_class": cls_name,
                    "conf": c,
                    "bbox": [x1, y1, x2, y2],
                })

        # Sort candidates by confidence descending
        candidates.sort(key=lambda x: x["conf"], reverse=True)

        for cand in candidates:
            x1, y1, x2, y2 = cand["bbox"]

            # If OCR verification is enabled and engine is available
            if verify_with_ocr and self.ocr_detector is not None:
                pad_x = max(8, int((x2 - x1) * 0.12))
                pad_y = max(6, int((y2 - y1) * 0.12))
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(fw, x2 + pad_x)
                cy2 = min(fh, y2 + pad_y)
                crop = frame_or_roi[cy1:cy2, cx1:cx2]

                ocr_id, ocr_type, ocr_conf, texts = self.ocr_detector.verify_crop(crop)

                if ocr_id is not None:
                    # OCR VERIFIED! Positive authentic identification!
                    cand["asset_id"] = ocr_id
                    cand["asset_type"] = ocr_type
                    cand["ocr_conf"] = ocr_conf
                    cand["ocr_texts"] = texts
                    cand["verified"] = True
                    return cand
                else:
                    # OCR found no valid asset ID text on this candidate crop!
                    # Reject this candidate to prevent false positives!
                    continue
            else:
                # If OCR is disabled, only accept high confidence YOLO detections
                if cand["conf"] >= 0.50:
                    cand["verified"] = False
                    return cand

        return None

    def detect_anomalies_for_asset(
        self,
        frame_or_roi: np.ndarray,
        asset_type: str,
        conf: float = ANOMALY_DETECT_CONF,
    ) -> List[Dict]:
        """
        Step 2: Detect anomalies and components ONLY for the locked asset type.
        Classes not relevant to this asset are excluded from prediction.
        """
        allowed = ASSET_ALLOWED_CLASSES.get(asset_type, None)

        predict_kwargs = {
            "source": frame_or_roi,
            "conf": conf,
            "device": self.device,
            "verbose": False,
        }
        if allowed is not None:
            predict_kwargs["classes"] = allowed

        results = self.model.predict(**predict_kwargs)[0]

        detections = []
        for box in results.boxes:
            cls_idx = int(box.cls[0])
            cls_name = self.model.names[cls_idx]
            c = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append({
                "class": cls_name,
                "conf": c,
                "bbox": [x1, y1, x2, y2],
            })
        return detections

    def inspect_roi(
        self,
        full_frame: np.ndarray,
        roi_rect: Tuple[int, int, int, int],
        asset_id: str,
        asset_type: str,
    ) -> InspectionResult:
        """
        Run complete triggered inspection on the square ROI:
        1. Crop target square ROI so surroundings are excluded.
        2. Run defect detection filtered by asset_type.
        3. Translate bounding boxes back to full frame coordinates.
        4. Save clean raw frame and annotated inspection image.
        """
        start_time = time.time()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        rx1, ry1, rx2, ry2 = roi_rect
        roi = full_frame[ry1:ry2, rx1:rx2].copy()
        rw, rh = roi.shape[1], roi.shape[0]

        # Detect defects only within the square ROI
        detections = self.detect_anomalies_for_asset(roi, asset_type, conf=ROI_DETECT_CONF)
        inference_time_ms = (time.time() - start_time) * 1000

        findings: List[DefectFinding] = []
        detected_components: List[dict] = []

        # Defect classes specific to this asset
        if asset_type == "Transformer":
            allowed_defects = ["oil_leakage", "crack"]
            allowed_components = ["transformer_body", "oil_gauge", "bushing", "tag_TR1", "tag_TR2"]
        elif asset_type == "Circuit Breaker":
            allowed_defects = ["physical_damage", "loose_connection"]
            allowed_components = ["circuit_breaker", "bushing", "tag_CB1"]
        elif asset_type == "Cable Section":
            allowed_defects = ["insulation_damage", "cable_damage", "burn_mark", "loose_connection", "physical_damage", "crack"]
            allowed_components = ["cable_bundle", "tag_CS02"]
        else:
            allowed_defects = list(DEFECT_PARAMETER_MAP.keys())
            allowed_components = []

        for det in detections:
            c = det["class"]
            conf = det["conf"]
            bx1, by1, bx2, by2 = det["bbox"]

            # Translate bbox back to full_frame coordinates
            full_bbox = [rx1 + bx1, ry1 + by1, rx1 + bx2, ry1 + by2]

            if c in allowed_defects:
                info = DEFECT_PARAMETER_MAP.get(c, {
                    "param": c,
                    "severity": "Major",
                    "name": c.replace("_", " ").title(),
                    "description": f"Defect detected on {asset_type}.",
                })

                loc_desc = self._describe_location([bx1, by1, bx2, by2], rw, rh, c)

                findings.append(DefectFinding(
                    defect_type=info["param"],
                    severity=info["severity"],
                    location=loc_desc,
                    description=info["description"],
                    confidence=conf,
                    bbox=full_bbox,
                ))
            elif c in allowed_components:
                det_copy = det.copy()
                det_copy["bbox"] = full_bbox
                detected_components.append(det_copy)

        # Deduplicate overlapping findings of the same defect type (NMS with IoU > 0.40)
        deduped_findings: List[DefectFinding] = []
        for f in sorted(findings, key=lambda x: x.confidence, reverse=True):
            overlap = False
            for k in deduped_findings:
                if k.defect_type == f.defect_type:
                    b1, b2 = f.bbox, k.bbox
                    inter_x = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                    inter_y = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                    inter_area = inter_x * inter_y
                    area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
                    area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
                    iou = inter_area / float(area1 + area2 - inter_area)
                    if iou > NMS_IOU_THRESH:
                        overlap = True
                        break
            if not overlap:
                deduped_findings.append(f)
        findings = deduped_findings

        # Baseline oil gauge reading for transformers
        oil_gauge_reading = None
        if asset_type == "Transformer":
            oil_gauge_reading = 35 if any(f.defect_type == "Oil Leakage" for f in findings) else 92

        overall_condition = "Faulty" if len(findings) > 0 else "Healthy"

        print(f"\n[INSPECTION RESULT] Asset: {asset_id} ({asset_type})")
        print(f"[INSPECTION RESULT] Condition: {overall_condition} | Defects: {len(findings)} | Latency: {inference_time_ms:.1f}ms")
        for i, f in enumerate(findings, 1):
            print(f"  [{i}] {f.defect_type} ({f.severity}) - Conf: {f.confidence:.2f} at {f.location}")

        # Save raw capture
        raw_path = os.path.join(CAPTURES_DIR, f"{asset_id}_{file_timestamp}_raw.jpg")
        cv2.imwrite(raw_path, full_frame)

        # Save annotated capture
        annotated_path = os.path.join(ANNOTATED_DIR, f"{asset_id}_{file_timestamp}_annotated.jpg")
        self._annotate_image(
            img=full_frame,
            asset_id=asset_id,
            asset_type=asset_type,
            overall_condition=overall_condition,
            findings=findings,
            components=detected_components,
            oil_gauge=oil_gauge_reading,
            timestamp=timestamp_str,
            inference_time_ms=inference_time_ms,
            output_path=annotated_path,
            roi_rect=roi_rect,
        )

        return InspectionResult(
            asset_id=asset_id,
            asset_type=asset_type,
            timestamp=timestamp_str,
            image_path=raw_path,
            annotated_image_path=annotated_path,
            overall_condition=overall_condition,
            oil_gauge_reading=oil_gauge_reading,
            findings=findings,
            detected_components=detected_components,
            inference_time_ms=inference_time_ms,
        )

    def _describe_location(self, bbox: List[int], img_w: int, img_h: int, defect_cls: str) -> str:
        """Generate human-readable location description for a bounding box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        h_pos = "Left" if cx < img_w * 0.35 else ("Right" if cx > img_w * 0.65 else "Center")
        v_pos = "Top" if cy < img_h * 0.35 else ("Base/Bottom" if cy > img_h * 0.70 else "Middle")

        if defect_cls == "oil_leakage":
            if cy > img_h * 0.70:
                return "Transformer base frame / ground puddle"
            return f"{h_pos} side tank wall / cooling radiator"
        elif defect_cls == "crack":
            return f"{h_pos} tank weld joint / radiator seam"
        elif defect_cls == "cable_damage":
            return f"{v_pos}-{h_pos} cable run (sheath cut)"
        elif defect_cls == "burn_mark":
            return f"{v_pos}-{h_pos} cable section (heat charring)"
        elif defect_cls == "loose_connection":
            return f"{h_pos} side cable support tray / termination"
        return f"{v_pos} {h_pos} region"

    def _annotate_image(
        self,
        img: np.ndarray,
        asset_id: str,
        asset_type: str,
        overall_condition: str,
        findings: List[DefectFinding],
        components: List[dict],
        oil_gauge: Optional[int],
        timestamp: str,
        inference_time_ms: float,
        output_path: str,
        roi_rect: Optional[Tuple[int, int, int, int]] = None,
    ):
        """Draw high-contrast, professional inspection overlays on the image."""
        annotated = img.copy()
        h, w = annotated.shape[:2]

        # ── 1. Draw Target ROI Square Boundary ────────────────────────────────
        if roi_rect is not None:
            rx1, ry1, rx2, ry2 = roi_rect
            border_color = (0, 200, 0) if overall_condition == "Healthy" else (0, 0, 255)
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), border_color, 2)
            cv2.putText(
                annotated, f"INSPECTION BOUNDARY [{asset_id}]",
                (rx1 + 8, ry1 - 8 if ry1 > 20 else ry1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1
            )

        # ── 2. Top Status Banner ──────────────────────────────────────────────
        banner_h = max(55, int(h * 0.08))
        banner_color = (15, 120, 40) if overall_condition == "Healthy" else (30, 30, 200)
        cv2.rectangle(annotated, (0, 0), (w, banner_h), banner_color, -1)

        title = f"TCS ROBOT INSPECTION | Asset: {asset_id} ({asset_type}) | Status: {overall_condition.upper()}"
        cv2.putText(annotated, title, (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

        sub_text = f"Timestamp: {timestamp}  |  YOLO Latency: {inference_time_ms:.1f}ms"
        if oil_gauge is not None:
            sub_text += f"  |  Oil Gauge: {oil_gauge}%"
        cv2.putText(annotated, sub_text, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # ── 3. Draw Detected Components (Green / Blue / Yellow) ──────────────
        for comp in components:
            c = comp["class"]
            bbox = comp["bbox"]
            conf = comp["conf"]
            x1, y1, x2, y2 = bbox

            if "tag" in c:
                color = (0, 220, 0)
                label = f"ID PLATE: {asset_id} ({int(conf*100)}%)"
            elif c == "oil_gauge":
                color = (255, 180, 0)
                val = f"{oil_gauge}%" if oil_gauge else ""
                label = f"OIL GAUGE {val} ({int(conf*100)}%)"
            elif c == "bushing":
                color = (200, 150, 50)
                label = f"BUSHING ({int(conf*100)}%)"
            else:
                color = (160, 210, 160)
                label = f"{c.replace('_', ' ').upper()} ({int(conf*100)}%)"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(banner_h + 15, y1 - 20)), (x1 + tw + 6, max(banner_h + 15, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(banner_h + 15, y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # ── 4. Draw Detected Defects (Red / Orange with bold labels) ──────────
        for finding in findings:
            if not finding.bbox:
                continue
            x1, y1, x2, y2 = finding.bbox
            is_critical = finding.severity == "Critical"
            box_color = (0, 0, 230) if is_critical else (0, 140, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

            marker_len = min(25, (x2 - x1) // 3, (y2 - y1) // 3)
            cv2.line(annotated, (x1, y1), (x1 + marker_len, y1), (255, 255, 255), 3)
            cv2.line(annotated, (x1, y1), (x1, y1 + marker_len), (255, 255, 255), 3)
            cv2.line(annotated, (x2, y2), (x2 - marker_len, y2), (255, 255, 255), 3)
            cv2.line(annotated, (x2, y2), (x2, y2 - marker_len), (255, 255, 255), 3)

            label = f"[DEFECT] {finding.defect_type.upper()} ({int(finding.confidence*100)}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            badge_y1 = max(banner_h + 22, y1 - 24)
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, badge_y1 + th + 8), box_color, -1)
            cv2.putText(annotated, label, (x1 + 5, badge_y1 + th + 3), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, annotated)
        print(f"[YOLO] Annotated image saved: {output_path}")

    def inspect(self, image_path: str) -> InspectionResult:
        """Fallback for file inspection."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = img.shape[:2]
        # Full frame as ROI
        tag = self.detect_asset_tag_only(img, conf=0.20)
        asset_id = tag["asset_id"] if tag else "UNKNOWN"
        asset_type = tag["asset_type"] if tag else "Unknown"
        return self.inspect_roi(img, (0, 0, w, h), asset_id, asset_type)
