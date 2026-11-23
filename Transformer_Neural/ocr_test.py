"""
Live camera test for ocr_asset_id.AssetIDDetector.

Put this file in the SAME folder as ocr_asset_id.py and config.py, then run:
    python live_ocr_test.py                # camera index 1
    python live_ocr_test.py --cam 0        # different camera
    python live_ocr_test.py --cpu          # if no GPU
    python live_ocr_test.py --debug        # print raw EasyOCR output

Keys (click the video window first):
    q / ESC : quit
    d       : toggle debug printing of raw OCR text
    r       : cycle search region (full -> lower_half -> center)
    s       : save current frame to ./captures/
    SPACE   : pause / resume OCR
"""

import argparse
import os
import threading
import time

import cv2

import ocr_asset_id
from ocr_asset_id import AssetIDDetector

REGIONS = ["full", "lower_half", "center"]
HOLD_SECONDS = 1.5   # keep drawing last detection for this long


class OCRWorker(threading.Thread):
    """Runs OCR in the background so the video preview never freezes."""

    def __init__(self, detector):
        super().__init__(daemon=True)
        self.det = detector
        self.lock = threading.Lock()
        self.frame = None
        self.region = "full"
        self.result = None
        self.result_time = 0.0
        self.latency = 0.0
        self.paused = False
        self.running = True
        self.event = threading.Event()

    def submit(self, frame, region):
        with self.lock:
            self.frame = frame.copy()
            self.region = region
        self.event.set()

    def run(self):
        while self.running:
            self.event.wait(timeout=0.5)
            self.event.clear()
            with self.lock:
                frame, region = self.frame, self.region
                self.frame = None
            if frame is None or self.paused:
                continue
            t0 = time.time()
            try:
                res = self.det.detect_in_frame(frame, search_region=region)
            except Exception as e:
                print(f"[ERR] OCR failed: {e}")
                res = None
            dt = time.time() - t0
            with self.lock:
                self.latency = dt
                if res is not None:
                    self.result = res
                    self.result_time = time.time()
                    print(f"[DETECT] {res['asset_id']} ({res['asset_type']}) "
                          f"conf={res['conf']:.2f} text='{res['text']}' bbox={res['bbox']} "
                          f"[{dt*1000:.0f} ms]")

    def snapshot(self):
        with self.lock:
            return self.result, self.result_time, self.latency


def open_camera(index):
    # Try DirectShow first on Windows (faster/more reliable), then default backend
    for backend in (getattr(cv2, "CAP_DSHOW", 0), cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def draw_overlay(frame, result, age, latency, fps, region, paused):
    h, w = frame.shape[:2]

    if result is not None and age < HOLD_SECONDS:
        x1, y1, x2, y2 = result["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{result['asset_id']} | {result['asset_type']} | {result['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 8
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 6), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Search region guide
    if region == "lower_half":
        cv2.rectangle(frame, (0, int(h * 0.45)), (w - 1, h - 1), (255, 200, 0), 1)
    elif region == "center":
        cv2.rectangle(frame, (int(w * 0.1), int(h * 0.2)),
                      (int(w * 0.9), int(h * 0.8)), (255, 200, 0), 1)

    # HUD
    status = "PAUSED" if paused else "RUNNING"
    if result is not None and age < HOLD_SECONDS:
        det_txt = f"Detected: {result['asset_id']}"
        det_col = (0, 255, 0)
    else:
        det_txt = "Detected: none"
        det_col = (0, 0, 255)
    lines = [
        (f"FPS {fps:4.1f} | OCR {latency*1000:4.0f} ms | region={region} | {status}", (255, 255, 255)),
        (det_txt, det_col),
        ("q:quit d:debug r:region s:save space:pause", (200, 200, 200)),
    ]
    y = 24
    for txt, col in lines:
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        y += 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1, help="camera index (default 1)")
    ap.add_argument("--cpu", action="store_true", help="run EasyOCR on CPU")
    ap.add_argument("--debug", action="store_true", help="print raw OCR output")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    ocr_asset_id.DEBUG = args.debug
    detector = AssetIDDetector(gpu=not args.cpu)

    cap = open_camera(args.cam)
    if cap is None:
        print(f"[ERR] Could not open camera index {args.cam}. Try --cam 0 or --cam 2.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"[CAM] Opened index {args.cam}: "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    worker = OCRWorker(detector)
    worker.start()

    region_idx = 0
    fps, last_t = 0.0, time.time()
    win = "Asset ID OCR - live"
    os.makedirs("captures", exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Frame grab failed, retrying...")
                time.sleep(0.05)
                continue

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now

            # Feed the worker the newest frame (it skips ahead if busy)
            worker.submit(frame, REGIONS[region_idx])

            result, rtime, latency = worker.snapshot()
            display = frame.copy()
            draw_overlay(display, result, time.time() - rtime, latency, fps,
                         REGIONS[region_idx], worker.paused)
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                ocr_asset_id.DEBUG = not ocr_asset_id.DEBUG
                print(f"[DEBUG] {'ON' if ocr_asset_id.DEBUG else 'OFF'}")
            elif key == ord("r"):
                region_idx = (region_idx + 1) % len(REGIONS)
                print(f"[REGION] {REGIONS[region_idx]}")
            elif key == ord("s"):
                path = os.path.join("captures", f"cap_{int(time.time())}.png")
                cv2.imwrite(path, frame)
                print(f"[SAVED] {path}")
            elif key == ord(" "):
                worker.paused = not worker.paused
                print(f"[OCR] {'paused' if worker.paused else 'resumed'}")
    finally:
        worker.running = False
        worker.event.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()