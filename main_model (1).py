"""
turret_pi.py  —  Raspberry Pi face-tracking turret controller
Fixes applied:
  1. Tilt direction was inverted (face below → tilt_us decreased toward clamp)
  2. picamera2 returned RGB; converted to BGR for OpenCV
  3. INFER_SIZE raised 160→320 for reliable YOLOv8n-face detection
  4. TRACKING strategy now avoids auto-locking when no target exists yet
  5. Serial port configurable via CLI argument
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
from enum import Enum
import time
import serial
import argparse


class LockingStrategy(Enum):
    CLOSEST           = "closest"
    CENTER            = "center"
    LARGEST           = "largest"
    HIGHEST_CONFIDENCE = "confidence"
    TRACKING          = "tracking"


class TurretController:
    """
    Raspberry Pi → Arduino turret controller over serial.

    Commands sent (all newline-terminated):
      'A'     : pan LEFT  (one step)
      'D'     : pan RIGHT (one step)
      'W'     : Arduino cmdTiltUp   — increases tiltUS → barrel physically DOWN
      'S'     : Arduino cmdTiltDown — decreases tiltUS → barrel physically UP
      'F'     : fire
      'R'     : reset / home
    """

    # ── PAN ────────────────────────────────────────────────────────────
    PAN_DEAD_ZONE_PX       = 10   # ignore small horizontal errors
    PAN_PX_PER_STEP        = 45   # pixels of error per one pan step
    PAN_MAX_STEPS_PER_TICK = 2    # cap steps per control tick

    # ── TILT ───────────────────────────────────────────────────────────
    TILT_DEAD_ZONE_PX       = 10  # ignore small vertical errors
    TILT_PX_PER_STEP        = 45  # pixels of error per one tilt step
    TILT_MAX_STEPS_PER_TICK = 2   # cap steps per control tick

    # ── FIRE ───────────────────────────────────────────────────────────
    FIRE_COOLDOWN_S = 1.0         # minimum seconds between auto-fire shots

    # ── RATE LIMITING ──────────────────────────────────────────────────
    MIN_TICK_INTERVAL_S = 0.04   # ~25 Hz control updates

    def __init__(self, port="/dev/ttyACM0", baud=115200, enabled=True):
        self.enabled = enabled
        self.ser     = None

        self._last_tick_time  = 0.0
        self._last_fire_time  = -float("inf")  # allow immediate first shot

        if not enabled:
            print("⚙️  Turret control DISABLED (preview only)")
            return

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)                   # wait for Arduino reset
            self.ser.reset_input_buffer()   # discard any boot garbage
            print(f"✓ Turret connected on {port} @ {baud}")
        except serial.SerialException as e:
            print(f"⚠️  Could not open {port}: {e}")
            print("   Running in PREVIEW mode (no turret control)")
            self.enabled = False
            self.ser     = None

    def _write_line(self, s: str):
        if not self.ser:
            return
        try:
            self.ser.write((s + "\n").encode("ascii", "ignore"))
        except serial.SerialException:
            pass

    def track(self, error_x: int, error_y: int):
        """
        Drive turret so that (error_x, error_y) → (0, 0).

        error_x = face_cx − frame_center_x   (+ve: face is to the RIGHT)
        error_y = face_cy − frame_center_y   (+ve: face is BELOW centre)

        Pan:  face right  → send 'D' (pan right)
              face left   → send 'A' (pan left)

        Tilt: the tilt servo is mounted so that increasing tiltUS physically
              points the barrel DOWN (TILT_HOME_US == TILT_MIN_US == 600 is
              the upward/level position).  Therefore:
              face below  → barrel must go DOWN → send 'W' (Arduino
                            cmdTiltUp: increases tiltUS toward TILT_MAX_US)
              face above  → barrel must go UP   → send 'S' (Arduino
                            cmdTiltDown: decreases tiltUS toward TILT_MIN_US)
        """
        if not self.enabled:
            return

        now = time.monotonic()
        if now - self._last_tick_time < self.MIN_TICK_INTERVAL_S:
            return
        self._last_tick_time = now

        # ── PAN ────────────────────────────────────────────────────────
        if abs(error_x) > self.PAN_DEAD_ZONE_PX:
            steps = int(error_x / self.PAN_PX_PER_STEP)
            steps = max(-self.PAN_MAX_STEPS_PER_TICK,
                        min(self.PAN_MAX_STEPS_PER_TICK, steps))

            if steps > 0:                   # face is right → pan right
                for _ in range(steps):
                    self._write_line("D")
            elif steps < 0:                 # face is left  → pan left
                for _ in range(-steps):
                    self._write_line("A")

        # ── TILT ───────────────────────────────────────────────────────
        if abs(error_y) > self.TILT_DEAD_ZONE_PX:
            steps = int(error_y / self.TILT_PX_PER_STEP)
            steps = max(-self.TILT_MAX_STEPS_PER_TICK,
                        min(self.TILT_MAX_STEPS_PER_TICK, steps))

            if steps > 0:                   # face is below → barrel DOWN → W
                for _ in range(steps):
                    self._write_line("W")
            elif steps < 0:                 # face is above → barrel UP  → S
                for _ in range(-steps):
                    self._write_line("S")

    def fire(self):
        now = time.monotonic()
        if now - self._last_fire_time < self.FIRE_COOLDOWN_S:
            return
        self._last_fire_time = now
        self._write_line("F")

    def reset(self):
        self._write_line("R")

    def close(self):
        if self.ser:
            self.ser.close()


class YOLOFaceDetectorRPi:
    # ── Performance settings ───────────────────────────────────────────
    CAPTURE_WIDTH  = 640
    CAPTURE_HEIGHT = 480

    # FIX #3: raised from 160 → 320; YOLOv8n-face is unreliable at 160 px
    INFER_SIZE  = 320
    MAX_FACES   = 3
    SKIP_FRAMES = 2
    CONFIDENCE  = 0.45

    # ── Locking ────────────────────────────────────────────────────────
    IOU_THRESHOLD    = 0.35
    LOST_FRAME_LIMIT = 20

    STRATEGIES = {
        ord("1"): LockingStrategy.CLOSEST,
        ord("2"): LockingStrategy.CENTER,
        ord("3"): LockingStrategy.LARGEST,
        ord("4"): LockingStrategy.HIGHEST_CONFIDENCE,
        ord("5"): LockingStrategy.TRACKING,
    }

    def __init__(
        self,
        model_name    = "yolov8n-face.pt",
        strategy      = LockingStrategy.CENTER,
        turret_port   = "/dev/ttyACM0",
        turret_enabled = True,
    ):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_name)

        if not os.path.exists(model_path):
            raise SystemExit(
                f"Error: Model file '{model_name}' not found in {script_dir}"
            )

        print(f"Loading model: {model_name}")
        self.model = YOLO(model_path)
        print("✓ Model loaded")

        self.lock_strategy      = strategy
        self.locked_target      = None
        self.tracking_frames    = 0
        self.auto_lock_enabled  = True
        self.manual_mode        = False

        self._last_boxes                = []
        self._current_boxes_for_click   = []

        self.turret = TurretController(port=turret_port, enabled=turret_enabled)

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _bbox(box):
        return box.xyxy[0].tolist()

    @staticmethod
    def _center(x1, y1, x2, y2):
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @staticmethod
    def _iou(b1, b2):
        x1_i = max(b1[0], b2[0]); y1_i = max(b1[1], b2[1])
        x2_i = min(b1[2], b2[2]); y2_i = min(b1[3], b2[3])
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        inter = (x2_i - x1_i) * (y2_i - y1_i)
        a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    # ── Lock scoring ───────────────────────────────────────────────────

    def _score_box(self, box, frame_center):
        x1, y1, x2, y2 = self._bbox(box)
        strat = self.lock_strategy

        if strat == LockingStrategy.CENTER:
            cx, cy = self._center(x1, y1, x2, y2)
            d = np.hypot(cx - frame_center[0], cy - frame_center[1])
            return 1.0 / max(d, 1.0)

        if strat == LockingStrategy.HIGHEST_CONFIDENCE:
            return box.conf[0].item()

        if strat == LockingStrategy.LARGEST:
            return (x2 - x1) * (y2 - y1)

        if strat == LockingStrategy.TRACKING:
            # FIX #4: only score against existing lock; without a lock,
            # return 0 so no new lock is acquired in TRACKING mode until
            # the user manually selects a target or switches strategy.
            if self.locked_target is not None:
                return self._iou([x1, y1, x2, y2], self.locked_target["bbox"])
            return 0.0

        # CLOSEST: approximate via bbox width (bigger ≈ closer)
        return max(x2 - x1, 1.0)

    def _select_target(self, boxes, frame_center):
        if len(boxes) == 0:
            return None

        # FIX #4: in TRACKING mode with no existing lock, don't acquire
        if self.lock_strategy == LockingStrategy.TRACKING and self.locked_target is None:
            return None

        best_idx = max(
            range(len(boxes)),
            key=lambda i: self._score_box(boxes[i], frame_center)
        )

        if (self.lock_strategy == LockingStrategy.TRACKING
                and self.locked_target is not None):
            if self._score_box(boxes[best_idx], frame_center) < self.IOU_THRESHOLD:
                return None

        return best_idx

    def _maintain_existing_lock(self, boxes):
        best_iou, best_idx = 0.0, None
        for i, box in enumerate(boxes):
            iou = self._iou(self._bbox(box), self.locked_target["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_idx is not None and best_iou > self.IOU_THRESHOLD:
            self.locked_target["bbox"]  = self._bbox(boxes[best_idx])
            self.locked_target["idx"]   = best_idx
            self.locked_target["lost"]  = 0
            self.tracking_frames       += 1
            return

        self.locked_target["lost"] = self.locked_target.get("lost", 0) + 1
        if self.locked_target["lost"] > self.LOST_FRAME_LIMIT:
            self.locked_target   = None
            self.tracking_frames = 0

    def _acquire_lock(self, boxes, idx):
        bb = self._bbox(boxes[idx])
        self.locked_target = {
            "idx": idx, "bbox": bb, "lost": 0,
            "lock_time": time.monotonic()
        }
        self.tracking_frames = 1
        print(
            f"🎯 LOCKED ({self.lock_strategy.value}) "
            f"conf={boxes[idx].conf[0].item():.2f}"
        )

    def _update_lock(self, boxes, frame_center):
        if not self.auto_lock_enabled:
            if self.locked_target is not None:
                self._maintain_existing_lock(boxes)
            return

        if self.locked_target is not None:
            self._maintain_existing_lock(boxes)
            return

        idx = self._select_target(boxes, frame_center)
        if idx is not None:
            self._acquire_lock(boxes, idx)

    def _unlock(self):
        self.locked_target   = None
        self.tracking_frames = 0

    # ── Mouse callback ─────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or not self.manual_mode:
            return
        for idx, box in enumerate(self._current_boxes_for_click):
            x1, y1, x2, y2 = self._bbox(box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.locked_target = {
                    "idx": idx, "bbox": [x1, y1, x2, y2],
                    "lost": 0, "lock_time": time.monotonic()
                }
                self.tracking_frames = 1
                break

    # ── Camera init ────────────────────────────────────────────────────

    @staticmethod
    def _try_picamera2(w, h):
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            # FIX #2: use BGR888 so OpenCV receives native BGR, not RGB
            cfg = cam.create_preview_configuration(
                main={"size": (w, h), "format": "BGR888"}
            )
            cam.configure(cfg)
            cam.start()
            print("✓ Using picamera2 (BGR888)")
            return (lambda: (True, cam.capture_array())), cam.stop
        except Exception:
            return None, None

    @staticmethod
    def _try_cv2(index, w, h):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return None, None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        print(f"✓ Using cv2.VideoCapture (index={index})")
        return cap.read, cap.release

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self, camera_index=0):
        W, H = self.CAPTURE_WIDTH, self.CAPTURE_HEIGHT

        grab, release = self._try_picamera2(W, H)
        if grab is None:
            grab, release = self._try_cv2(camera_index, W, H)
        if grab is None:
            print("Error: no camera available")
            return

        frame_center = (W // 2, H // 2)
        frame_count  = 0
        fps          = 0.0
        t_prev       = time.monotonic()

        win = "YOLO + Turret"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)

        try:
            while True:
                ok, frame = grab()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)

                # ── Inference (every SKIP_FRAMES frames) ──────────────
                if frame_count % self.SKIP_FRAMES == 0:
                    results = self.model(
                        frame,
                        conf    = self.CONFIDENCE,
                        imgsz   = self.INFER_SIZE,
                        max_det = self.MAX_FACES,
                        verbose = False,
                    )
                    self._last_boxes = results[0].boxes
                frame_count += 1

                boxes = self._last_boxes
                self._current_boxes_for_click = boxes

                self._update_lock(boxes, frame_center)

                # ── Turret control ─────────────────────────────────────
                error_x, error_y = 0, 0
                crosshair_in_box = False
                if self.locked_target is not None:
                    x1, y1, x2, y2 = self.locked_target["bbox"]
                    face_cx, face_cy = self._center(x1, y1, x2, y2)
                    error_x = int(face_cx - frame_center[0])
                    error_y = int(face_cy - frame_center[1])
                    self.turret.track(error_x, error_y)
                    cv2.line(frame, frame_center,
                             (face_cx, face_cy), (0, 0, 255), 2)

                    # Auto-fire when crosshair is inside the bounding box
                    cx, cy = frame_center
                    if x1 <= cx < x2 and y1 <= cy < y2:
                        crosshair_in_box = True
                        self.turret.fire()

                # Crosshair — green when inside target box, blue otherwise
                crosshair_color = (0, 255, 0) if crosshair_in_box else (255, 0, 0)
                cv2.drawMarker(frame, frame_center,
                               crosshair_color, cv2.MARKER_CROSS, 20, 2)

                # ── Draw bounding boxes ────────────────────────────────
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(v) for v in self._bbox(box)]
                    is_locked = (
                        self.locked_target is not None
                        and i == self.locked_target["idx"]
                    )
                    color = (0, 0, 255) if is_locked else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # ── FPS ────────────────────────────────────────────────
                now   = time.monotonic()
                fps   = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now

                # ── HUD ────────────────────────────────────────────────
                turret_ok = "OK" if self.turret.ser else "PREVIEW"
                lock_txt  = "LOCK" if self.locked_target else "NOLOCK"
                fire_txt  = "FIRE" if crosshair_in_box else ""
                cv2.putText(
                    frame,
                    f"{fps:.0f}fps {lock_txt} turret:{turret_ok} "
                    f"err=({error_x:+d},{error_y:+d}) {fire_txt}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    "Keys: q quit | u unlock | a auto | "
                    "m manual | 1-5 strategy | r reset | f fire",
                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA
                )

                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("u"):
                    self._unlock()
                elif key == ord("r"):
                    self.turret.reset()
                    self._unlock()
                elif key == ord("f"):
                    self.turret.fire()
                elif key == ord("a"):
                    self.auto_lock_enabled = not self.auto_lock_enabled
                    self.manual_mode = False
                elif key == ord("m"):
                    self.manual_mode       = not self.manual_mode
                    self.auto_lock_enabled = not self.manual_mode
                elif key in self.STRATEGIES:
                    self.lock_strategy = self.STRATEGIES[key]
                    self._unlock()

        finally:
            self.turret.close()
            release()
            cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # FIX #5: make serial port configurable at runtime
    parser = argparse.ArgumentParser(description="YOLO face-tracking turret")
    parser.add_argument(
        "--port", default="COM9",
        help="Arduino serial port (default: /dev/ttyACM0, try /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--no-turret", action="store_true",
        help="Run in preview-only mode without serial connection"
    )
    parser.add_argument(
        "--camera", type=int, default=1,
        help="cv2 camera index (default: 0)"
    )
    args = parser.parse_args()

    detector = YOLOFaceDetectorRPi(
        model_name     = "yolov8n-face.pt",
        strategy       = LockingStrategy.CENTER,
        turret_port    = args.port,
        turret_enabled = not args.no_turret,
    )
    detector.run(camera_index=args.camera)
