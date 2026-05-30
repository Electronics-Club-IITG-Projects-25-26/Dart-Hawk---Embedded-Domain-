"""
turret_pi.py - Main vision and control script for the Darthawk turret.

Notes from last testing session:
- The tilt servo is mounted upside down. Sending 'W' increases the PWM signal, 
  which physically aims the barrel DOWN. Don't change this unless we rebuild the mount.
- picamera2 feeds us RGB, but OpenCV wants BGR. Fixed the config so we don't look like Smurfs.
- YOLOv8n-face is blind as a bat at 160px. Bumped infer size to 320px. 
- Using pySerial for bare-metal Arduino communication.
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
import time
import serial
import argparse
from enum import Enum

class TargetStrategy(Enum):
    CLOSEST = "closest"
    CENTER = "center"
    LARGEST = "largest"
    CONFIDENCE = "confidence"
    TRACKING = "tracking"

class DarthawkTurret:
    """
    Handles serial comms to the Arduino. 
    Expects newline-terminated chars: W/A/S/D for movement, F for fire, R for reset.
    """
    # Deadzones so the servos aren't constantly jittering when locked on
    PAN_DEAD_ZONE = 10 
    TILT_DEAD_ZONE = 10 
    
    # Rough calibration: 1 step command to the Arduino equals about 45 pixels of movement
    PX_PER_STEP = 45 
    MAX_STEPS = 2 # Hard cap so we don't violently whip the motors around
    
    FIRE_COOLDOWN = 1.0 # Give the mechanism a second between shots
    TICK_RATE = 0.04 # Throttle serial spam to ~25Hz

    def __init__(self, port="/dev/ttyACM0", baud=115200, enabled=True):
        self.enabled = enabled
        self.ser = None
        self.last_tick = 0.0
        self.last_fire = -float("inf") # Allow immediate firing on startup

        if not enabled:
            print("⚙️ Turret hardware disabled. Running in vision-only preview mode.")
            return

        try:
            # Short timeout so we don't hang the main loop if the Arduino drops
            self.ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2) # Give the Arduino time to wake up after serial init
            self.ser.reset_input_buffer() 
            print(f"✓ Turret armed and connected on {port}")
        except serial.SerialException as e:
            print(f"⚠️ Failed to connect to Arduino: {e}")
            print("  Falling back to preview mode.")
            self.enabled = False

    def _send(self, cmd: str):
        # Fire-and-forget serial write
        if self.ser:
            try:
                self.ser.write((cmd + "\n").encode("ascii", "ignore"))
            except serial.SerialException:
                pass # Usually happens if the USB cable gets yanked

    def track(self, error_x: int, error_y: int):
        """
        Takes pixel error from the center of the frame and translates it to 
        stepper/servo movements.
        """
        if not self.enabled:
            return

        now = time.monotonic()
        if now - self.last_tick < self.TICK_RATE:
            return
        self.last_tick = now

        # --- Pan Logic ---
        # error_x > 0 means the target is to the right of the crosshair
        if abs(error_x) > self.PAN_DEAD_ZONE:
            steps = int(error_x / self.PX_PER_STEP)
            steps = max(-self.MAX_STEPS, min(self.MAX_STEPS, steps)) # Clamp it

            if steps > 0:
                for _ in range(steps): self._send("D")
            elif steps < 0:
                for _ in range(-steps): self._send("A")

        # --- Tilt Logic ---
        # error_y > 0 means target is below the crosshair.
        # Remember: servo is inverted. Pushing 'W' increases PWM -> barrel goes DOWN.
        if abs(error_y) > self.TILT_DEAD_ZONE:
            steps = int(error_y / self.PX_PER_STEP)
            steps = max(-self.MAX_STEPS, min(self.MAX_STEPS, steps))

            if steps > 0:
                for _ in range(steps): self._send("W")
            elif steps < 0:
                for _ in range(-steps): self._send("S")

    def fire(self):
        now = time.monotonic()
        if now - self.last_fire >= self.FIRE_COOLDOWN:
            self.last_fire = now
            self._send("F")

    def reset(self):
        self._send("R")

    def close(self):
        if self.ser:
            self.ser.close()


class VisionSystem:
    # Camera resolution
    W, H = 640, 480
    
    # YOLO params
    INFER_SIZE = 320 
    MAX_DETECTIONS = 3
    SKIP_FRAMES = 2 # Only run inference every 3rd frame to save CPU
    CONF_THRESH = 0.45

    # Target tracking thresholds
    IOU_THRESH = 0.35 
    LOST_PATIENCE = 20 # How many frames we'll wait for a lost target before giving up

    # Keybinds for changing how we pick targets on the fly
    STRAT_MAP = {
        ord("1"): TargetStrategy.CLOSEST,
        ord("2"): TargetStrategy.CENTER,
        ord("3"): TargetStrategy.LARGEST,
        ord("4"): TargetStrategy.CONFIDENCE,
        ord("5"): TargetStrategy.TRACKING,
    }

    def __init__(self, model_name="yolov8n-face.pt", strat=TargetStrategy.CENTER, port="/dev/ttyACM0", hardware_on=True):
        model_path = os.path.join(os.path.dirname(__file__), model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Couldn't find YOLO model at {model_path}. Did you download it?")

        print(f"Loading {model_name}...")
        self.model = YOLO(model_path)
        
        self.strategy = strat
        self.locked_target = None
        self.auto_lock = True
        self.manual_override = False
        
        self.latest_boxes = [] # Cache for mouse click detection

        self.turret = DarthawkTurret(port=port, enabled=hardware_on)

    # --- Math Helpers ---
    
    def _get_bbox(self, box):
        # YOLO returns tensors, just grab the raw list of coords
        return box.xyxy[0].tolist()

    def _get_center(self, x1, y1, x2, y2):
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _calc_iou(self, b1, b2):
        # Standard Intersection over Union to check if a bounding box 
        # is roughly the same target we were looking at last frame.
        x_left = max(b1[0], b2[0])
        y_top = max(b1[1], b2[1])
        x_right = min(b1[2], b2[2])
        y_bottom = min(b1[3], b2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.
