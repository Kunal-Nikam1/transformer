import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np

from config import (
    BATCH_SIZE,
    REPORT_DIR,
    ANNOTATED_DIR,
    CAPTURES_DIR,
    InspectionResult,
)
from yolo_detector import YOLODetector
from report_generator import ReportGenerator


def auto_crop_equipment_by_corners(
    frame: np.ndarray,
    tag_bbox: Optional[List[int]] = None,
    component_bboxes: Optional[List[List[int]]] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
  
    fh, fw = frame.shape[:2]

    # Strategy 1: Contour & Corner boundary detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    max_area = 0
    min_area = (fw * fh) * 0.12  # At least 12% of frame

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if area > max_area and bw > fw * 0.25 and bh > fh * 0.25:
                max_area = area
                best_rect = (x, y, x + bw, y + bh)

    # Strategy 2: Component & Tag bounding boxes combination
    if component_bboxes:
        all_x1 = [b[0] for b in component_bboxes]
        all_y1 = [b[1] for b in component_bboxes]
        all_x2 = [b[2] for b in component_bboxes]
        all_y2 = [b[3] for b in component_bboxes]
        if tag_bbox:
            all_x1.append(tag_bbox[0])
            all_y1.append(tag_bbox[1])
            all_x2.append(tag_bbox[2])
            all_y2.append(tag_bbox[3])

        comp_x1 = max(0, min(all_x1))
        comp_y1 = max(0, min(all_y1))
        comp_x2 = min(fw, max(all_x2))
        comp_y2 = min(fh, max(all_y2))

        if best_rect is not None:
            best_rect = (
                min(best_rect[0], comp_x1),
                min(best_rect[1], comp_y1),
                max(best_rect[2], comp_x2),
                max(best_rect[3], comp_y2),
            )
        else:
            best_rect = (comp_x1, comp_y1, comp_x2, comp_y2)

    # Strategy 3: Tag vicinity expansion
    if best_rect is None and tag_bbox:
        tx1, ty1, tx2, ty2 = tag_bbox
        cx = (tx1 + tx2) // 2
        cy = (ty1 + ty2) // 2
        span_w = int(fw * 0.85)
        span_h = int(fh * 0.85)
        best_rect = (
            max(0, cx - span_w // 2),
            max(0, cy - span_h // 2),
            min(fw, cx + span_w // 2),
            min(fh, cy + span_h // 2),
        )

    # Apply 4% padding margin
    if best_rect is not None:
        bx1, by1, bx2, by2 = best_rect
        pad_x = int((bx2 - bx1) * 0.04)
        pad_y = int((by2 - by1) * 0.04)
        cx1 = max(0, bx1 - pad_x)
        cy1 = max(0, by1 - pad_y)
        cx2 = min(fw, bx2 + pad_x)
        cy2 = min(fh, by2 + pad_y)

        # Sanity check: must be at least 30% of image
        if (cx2 - cx1) > fw * 0.25 and (cy2 - cy1) > fh * 0.25:
            return frame[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2)

    # Fallback: full frame
    return frame.copy(), (0, 0, fw, fh)


class LiveCamInspector:
    """Manages mobile robot live camera inspection pipeline."""

    def __init__(
        self,
        camera_index: int = 4,
        batch_size: int = BATCH_SIZE,
        device: int = 0,
        tag_conf: float = 0.35,
        defect_conf: float = 0.15,
        lock_duration_sec: float = 4.0,
    ):
        self.camera_index = camera_index
        self.batch_size = batch_size
        self.device = device
        self.tag_conf = tag_conf
        self.defect_conf = defect_conf
        self.lock_duration_sec = lock_duration_sec

        # Initialize detector and report generator
        self.detector = YOLODetector(device=device)
        self.report_gen = ReportGenerator()

        # States: SEARCHING -> LOCKING -> CAPTURED_INSPECTION -> CONSECUTIVE_WAIT -> COMPLETE
        self.state = "SEARCHING"
        self.inspected_assets: List[InspectionResult] = []
        self.last_inspected_asset_id: Optional[str] = None
        self.time_last_asset_lost: float = time.time()

        # Locking tracking
        self.candidate_id: Optional[str] = None
        self.candidate_type: Optional[str] = None
        self.candidate_tag_bbox: Optional[List[int]] = None
        self.lock_start_time: float = 0.0
        self.last_tag_seen_time: float = 0.0

        # Visual feedback
        self.freeze_until: float = 0.0
        self.frozen_display_frame: Optional[np.ndarray] = None
        self.flash_frames: int = 0
        self.status_message: str = "SEARCHING FOR ASSET ID (TR-1, TR-2, CB-1, CS-02)..."
        self.status_color = (0, 220, 255)
        self.status_expiry: float = 0.0

    def open_camera(self) -> cv2.VideoCapture:
        """Open camera with DirectShow backend for Windows."""
        print(f"\n[CAM] Connecting to Camera Index {self.camera_index}...")
        cap = cv2.VideoCapture(5 , cv2.CAP_V4L2)
        
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print(f"[WARN] Failed to open Camera {self.camera_index} with CAP_DSHOW. Retrying CAP_ANY...")
            cap = cv2.VideoCapture(5 , cv2.CAP_V4L2)

        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_index}. Please verify connection.")
            return None

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

       	# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
       	# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAM] Camera connected. Resolution: {w}x{h}")
        return cap

    def run(self):
        """Main real-time loop."""
        cap = self.open_camera()
        if cap is None:
            sys.exit(1)

        window_name = "TCS Autonomous Substation Asset Inspection [Robot Camera - GPU]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 600)

        prev_time = time.time()
        fps = 0.0

        while True:
            # Handle post-inspection freeze display
            if time.time() < self.freeze_until and self.frozen_display_frame is not None:
                key = cv2.waitKey(30) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break
                cv2.imshow(window_name, self.frozen_display_frame)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame from camera.")
                time.sleep(0.05)
                continue

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, (curr_time - prev_time)))
            prev_time = curr_time

            fh, fw = frame.shape[:2]

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("[INFO] Quitting live camera inspection.")
                break
            elif key in [ord('r'), ord('R')]:
                print("[INFO] Resetting inspection batch.")
                self.inspected_assets = []
                self.last_inspected_asset_id = None
                self.state = "SEARCHING"
                self.candidate_id = None
                self.status_message = "Batch reset. Ready to inspect 4 assets."
                self.status_color = (0, 255, 255)
                self.status_expiry = time.time() + 3.0
            force_capture = (key in [32, 13])  # SPACE or ENTER

            # ──────────────────────────────────────────────────────────────────
            # STEP 1: DETECT ASSET ID TAG IN FRAME (NO DEFECT PREDICTION YET)
            # ──────────────────────────────────────────────────────────────────
            t0 = time.time()
            tag_detection = self.detector.detect_asset_tag_only(frame, conf=self.tag_conf)
            yolo_lat = (time.time() - t0) * 1000

            active_tag_bbox = None
            detected_tag_id = None
            detected_tag_type = None

            if tag_detection is not None:
                detected_tag_id = tag_detection["asset_id"]
                detected_tag_type = tag_detection["asset_type"]
                active_tag_bbox = tag_detection["bbox"]
                self.last_tag_seen_time = curr_time

            # ──────────────────────────────────────────────────────────────────
            # CONSECUTIVE SAME-ASSET CHECK
            # ──────────────────────────────────────────────────────────────────
            is_consecutively_same = False
            if self.last_inspected_asset_id is not None:
                if detected_tag_id == self.last_inspected_asset_id:
                    # Same asset is still in front of camera while robot is moving past!
                    is_consecutively_same = True
                    self.status_message = (
                        f"ASSET {self.last_inspected_asset_id} ALREADY LOGGED — "
                        f"WAITING FOR ROBOT TO NAVIGATE TO NEXT ASSET..."
                    )
                    self.status_color = (180, 180, 180)
                else:
                    # If this asset hasn't been seen for > 1.8 seconds, clear consecutive lock
                    if curr_time - self.last_tag_seen_time > 1.8:
                        self.last_inspected_asset_id = None

            # ──────────────────────────────────────────────────────────────────
            # STEP 2: 2-SECOND SHAKE-TOLERANT LOCK-IN
            # ──────────────────────────────────────────────────────────────────
            lock_progress = 0.0

            if self.state in ["SEARCHING", "LOCKING"] and not is_consecutively_same:
                if detected_tag_id is not None:
                    if self.state == "SEARCHING":
                        # Begin 2.0-second lock
                        self.state = "LOCKING"
                        self.candidate_id = detected_tag_id
                        self.candidate_type = detected_tag_type
                        self.candidate_tag_bbox = active_tag_bbox
                        self.lock_start_time = curr_time
                        lock_progress = 0.0
                    elif self.state == "LOCKING":
                        if detected_tag_id == self.candidate_id:
                            self.candidate_tag_bbox = active_tag_bbox
                            elapsed = curr_time - self.lock_start_time
                            lock_progress = min(1.0, elapsed / self.lock_duration_sec)

                            if elapsed >= self.lock_duration_sec or force_capture:
                                # 2 SECONDS CONFIRMED! TRIGGER CAPTURE & ANOMALY SCAN!
                                self.state = "TRIGGER_INSPECTION"
                        else:
                            # Candidate switched to different asset
                            self.candidate_id = detected_tag_id
                            self.candidate_type = detected_tag_type
                            self.candidate_tag_bbox = active_tag_bbox
                            self.lock_start_time = curr_time
                            lock_progress = 0.0
                else:
                    # Grace period for robot shaking/motion blur: tolerate up to 0.45s dropout
                    if self.state == "LOCKING" and not force_capture:
                        if curr_time - self.last_tag_seen_time > 0.45:
                            self.state = "SEARCHING"
                            self.candidate_id = None
                            lock_progress = 0.0

            # ──────────────────────────────────────────────────────────────────
            # STEP 3: CAPTURE, AUTO-CROP CORNERS & RUN DEFECT DETECTION
            # ──────────────────────────────────────────────────────────────────
            if (self.state == "TRIGGER_INSPECTION" or force_capture) and len(self.inspected_assets) < self.batch_size:
                target_id = self.candidate_id if self.candidate_id else (detected_tag_id if detected_tag_id else "UNKNOWN")
                target_type = self.candidate_type if self.candidate_type else (detected_tag_type if detected_tag_type else "Unknown")

                print("\n" + "#" * 68)
                print(f"  [{self.lock_duration_sec}s LOCK CONFIRMED] CAPTURING ASSET #{len(self.inspected_assets) + 1}/{self.batch_size}: {target_id}")
                print(f"  Auto-cropping equipment corners from surroundings...")
                print("#" * 68)

                # 1. Automatically detect equipment boundary corners & crop
                cropped_equipment, crop_rect = auto_crop_equipment_by_corners(
                    frame=frame,
                    tag_bbox=active_tag_bbox,
                )
                print(f"[CROP] Frame: {fw}x{fh} -> Cropped Equipment: {cropped_equipment.shape[1]}x{cropped_equipment.shape[0]} at {crop_rect}")

                # 2. Run YOLO defect analysis on that cropped equipment image
                result = self.detector.inspect_roi(
                    full_frame=frame,
                    roi_rect=crop_rect,
                    asset_id=target_id,
                    asset_type=target_type,
                )

                self.inspected_assets.append(result)
                self.last_inspected_asset_id = result.asset_id
                self.flash_frames = 6

                # 3. Build post-capture display with defect boxes to freeze briefly
                frozen_disp = self.draw_inspection_result(
                    frame=frame,
                    crop_rect=crop_rect,
                    result=result,
                )
                self.frozen_display_frame = frozen_disp
                self.freeze_until = curr_time + 1.8  # Show for 1.8s then resume

                cond_str = "FAULTY" if result.overall_condition == "Faulty" else "HEALTHY"
                self.status_message = (
                    f"Asset #{len(self.inspected_assets)}/4 ({result.asset_id}) Logged: {cond_str} "
                    f"({len(result.findings)} defects) — Moving to next asset..."
                )
                self.status_color = (0, 0, 255) if result.overall_condition == "Faulty" else (0, 255, 0)
                self.status_expiry = curr_time + 5.0

                # 4. Check if batch completed (4 assets)
                if len(self.inspected_assets) == self.batch_size:
                    print("\n" + "=" * 68)
                    print(f"  [COMPLETED] ALL {self.batch_size} ASSETS INSPECTED!")
                    print("  Generating TCS Automated Inspection Report PDF...")
                    print("=" * 68)
                    pdf_path = self.report_gen.generate(self.inspected_assets)
                    print(f"[REPORT] Final Report: {pdf_path}\n")
                    self.state = "COMPLETE"
                else:
                    self.state = "SEARCHING"
                    self.candidate_id = None

            # ──────────────────────────────────────────────────────────────────
            # STEP 4: RENDER ROBOT HEADS-UP DISPLAY (HUD)
            # ──────────────────────────────────────────────────────────────────
            display = self.render_hud(
                frame=frame,
                state=self.state,
                candidate_id=self.candidate_id,
                tag_bbox=self.candidate_tag_bbox if self.candidate_tag_bbox else active_tag_bbox,
                lock_progress=lock_progress,
                is_consecutive_same=is_consecutively_same,
                fps=fps,
                yolo_lat=yolo_lat,
            )

            # Flash animation
            if self.flash_frames > 0:
                cv2.rectangle(display, (0, 0), (fw, fh), (255, 255, 255), -1)
                self.flash_frames -= 1

            cv2.imshow(window_name, display)

        cap.release()
        cv2.destroyAllWindows()

    def render_hud(
        self,
        frame: np.ndarray,
        state: str,
        candidate_id: Optional[str],
        tag_bbox: Optional[List[int]],
        lock_progress: float,
        is_consecutive_same: bool,
        fps: float,
        yolo_lat: float,
    ) -> np.ndarray:
        """Render informative mobile-robot HUD on video stream."""
        display = frame.copy()
        h, w = display.shape[:2]

        # ── 1. Top HUD Banner ─────────────────────────────────────────────────
        banner_h = 58
        cv2.rectangle(display, (0, 0), (w, banner_h), (25, 25, 45), -1)
        cv2.line(display, (0, banner_h), (w, banner_h), (0, 180, 255), 2)

        cv2.putText(
            display, "TCS ROBOT VISION | AUTONOMOUS PATROL", (15, 24),
            cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 2
        )
        sub_title = f"  YOLO: {yolo_lat:.1f}ms  |  FPS: {fps:.1f}  |  Mode: Auto-Crop + 2s Lock"
        cv2.putText(
            display, sub_title, (15, 46),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 210, 255), 1
        )

        # Asset Progress Badge
        count_text = f"ASSETS: {len(self.inspected_assets)} / {self.batch_size}"
        badge_color = (0, 200, 0) if len(self.inspected_assets) == self.batch_size else (0, 140, 255)
        (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        bx1 = w - tw - 25
        cv2.rectangle(display, (bx1, 10), (w - 10, 48), badge_color, -1)
        cv2.putText(display, count_text, (bx1 + 8, 36), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

        # ── 2. Tag Reticle & 2-Second Lock Progress Bar ───────────────────────
        if tag_bbox is not None and not is_consecutive_same:
            tx1, ty1, tx2, ty2 = tag_bbox
            # Draw dynamic tracking reticle around the tag
            #reticle_color = (0, 200, 255) if state == "LOCKING" else (0, 255, 0)
           # cv2.rectangle(display, (tx1, ty1), (tx2, ty2), reticle_color, 2)

           # tag_label = f"ID: {candidate_id if candidate_id else 'DETECTED'} [OCR VERIFIED]"
            #cv2.putText(
             #   display, tag_label, (tx1, max(75, ty1 - 8)),
              #  cv2.FONT_HERSHEY_SIMPLEX, 0.5, reticle_color, 2
            #)

            if state == "LOCKING":
                # Progress bar underneath the tag
                bar_w = max(160, tx2 - tx1)
                bar_fill = int(bar_w * lock_progress)
                by1 = ty2 + 8
                by2 = ty2 + 20
                cv2.rectangle(display, (tx1, by1), (tx1 + bar_w, by2), (40, 40, 40), -1)
                cv2.rectangle(display, (tx1, by1), (tx1 + bar_fill, by2), (0, 200, 255), -1)
                cv2.rectangle(display, (tx1, by1), (tx1 + bar_w, by2), (255, 255, 255), 1)

                cv2.putText(
                    display, f"LOCKING 2.0s: {int(lock_progress * 100)}%",
                    (tx1, by2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1
                )

        # ── 3. Bottom Instruction Bar ─────────────────────────────────────────
        bot_h = 32
        cv2.rectangle(display, (0, h - bot_h), (w, h), (20, 20, 20), -1)

        if time.time() < self.status_expiry:
            msg = self.status_message
            m_color = self.status_color
        elif is_consecutive_same:
            msg = f"Consecutive asset '{self.last_inspected_asset_id}' in view — Robot moving to next asset..."
            m_color = (180, 180, 180)
        elif state == "COMPLETE":
            msg = "[MISSION COMPLETE] 4/4 Assets Inspected! Report generated in reports/. Press [R] or [Q]."
            m_color = (0, 255, 0)
        elif state == "LOCKING":
            msg = f"Locking onto {candidate_id} for {self.lock_duration_sec}s... Shake-tolerant tracking active. Auto-capture at 100%."
            m_color = (0, 200, 255)
        else:
            msg = "ROBOT NAVIGATING: Searching for Asset ID (TR-1, TR-2, CB-1, CS-02) | [SPACE] Force Capture | [Q] Quit"
            m_color = (200, 200, 200)

        cv2.putText(display, msg, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, m_color, 1)

        # ── 4. Mission Complete Modal ─────────────────────────────────────────
        if state == "COMPLETE":
            overlay = display.copy()
            mw, mh = int(w * 0.78), 140
            mx1, my1 = (w - mw) // 2, (h - mh) // 2
            cv2.rectangle(overlay, (mx1, my1), (mx1 + mw, my1 + mh), (15, 70, 25), -1)
            cv2.addWeighted(overlay, 0.88, display, 0.12, 0, display)
            cv2.rectangle(display, (mx1, my1), (mx1 + mw, my1 + mh), (0, 255, 100), 2)

            cv2.putText(
                display, "INSPECTION MISSION COMPLETE!", (mx1 + 30, my1 + 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2
            )
            cv2.putText(
                display, "4/4 Assets Successfully Analyzed & Documented", (mx1 + 30, my1 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1
            )
            cv2.putText(
                display, "TCS Inspection Report Generated in: reports/", (mx1 + 30, my1 + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 100), 1
            )
            cv2.putText(
                display, "Press [R] to Start New Session  |  Press [Q] to Exit", (mx1 + 30, my1 + 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
            )

        return display

    def draw_inspection_result(
        self,
        frame: np.ndarray,
        crop_rect: Tuple[int, int, int, int],
        result: InspectionResult,
    ) -> np.ndarray:
        """Render post-capture visual feedback with highlighted defects and parts."""
        display = frame.copy()
        h, w = display.shape[:2]
        cx1, cy1, cx2, cy2 = crop_rect

        # Highlight auto-cropped equipment boundary
        crop_border_col = (0, 0, 240) if result.overall_condition == "Faulty" else (0, 220, 0)
        cv2.rectangle(display, (cx1, cy1), (cx2, cy2), crop_border_col, 2)
        cv2.putText(
            display, f"AUTO-CROPPED EQUIPMENT BOUNDARY [{result.asset_id}]",
            (cx1 + 8, max(75, cy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, crop_border_col, 1
        )

        # Draw Components
        for comp in result.detected_components:
            bx1, by1, bx2, by2 = comp["bbox"]
            c = comp["class"]
            conf = comp["conf"]
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(
                display, f"{c.replace('_', ' ').upper()} ({int(conf*100)}%)",
                (bx1, max(75, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
            )

        # Draw Defects
        for finding in result.findings:
            if not finding.bbox:
                continue
            bx1, by1, bx2, by2 = finding.bbox
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            # Corner markers
            marker = 20
            cv2.line(display, (bx1, by1), (bx1 + marker, by1), (255, 255, 255), 3)
            cv2.line(display, (bx1, by1), (bx1, by1 + marker), (255, 255, 255), 3)
            label = f"[DEFECT] {finding.defect_type.upper()} ({int(finding.confidence*100)}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            cv2.rectangle(display, (bx1, max(75, by1 - 22)), (bx1 + tw + 8, max(75, by1)), (0, 0, 255), -1)
            cv2.putText(
                display, label, (bx1 + 4, max(75, by1) - 5),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1
            )

        # Top confirmation banner
        banner_h = 58
        cv2.rectangle(display, (0, 0), (w, banner_h), (10, 10, 10), -1)
        res_text = f"AUTO-CAPTURED: {result.asset_id} ({result.asset_type}) -> STATUS: {result.overall_condition.upper()}"
        res_color = (0, 0, 255) if result.overall_condition == "Faulty" else (0, 255, 0)
        cv2.putText(display, res_text, (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, res_color, 2)
        cv2.putText(
            display, f"Defects Found: {len(result.findings)}  |  Asset #{len(self.inspected_assets)}/{self.batch_size} Logged  |  Moving to next asset...",
            (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1
        )

        return display


def main():
    parser = argparse.ArgumentParser(
        description="TCS Substation Asset Live Camera Inspection (Robot Moving/Shaky Optimized)"
    )
    parser.add_argument(
        "--camera-index", "-c",
        type=int,
        default=1,
        help="Camera device index (default: 1)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of assets to inspect before triggering report (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index for GPU (default: 0)"
    )
    parser.add_argument(
        "--tag-conf",
        type=float,
        default=0.35,
        help="Tag proposal confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--defect-conf",
        type=float,
        default=0.15,
        help="Defect detection confidence threshold (default: 0.15)"
    )
    parser.add_argument(
        "--lock-time",
        type=float,
        default=4.0,
        help="Lock duration in seconds before capturing & inspecting (default: 2.0)"
    )

    args = parser.parse_args()

    inspector = LiveCamInspector(
        camera_index=args.camera_index,
        batch_size=args.batch_size,
        device=args.device,
        tag_conf=args.tag_conf,
        defect_conf=args.defect_conf,
        lock_duration_sec=args.lock_time,
    )
    inspector.run()


if __name__ == "__main__":
    main()
