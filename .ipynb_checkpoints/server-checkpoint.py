#!/usr/bin/env python3
"""
JetBot Server - runs on each robot (Python 3.6.9 compatible)
Provides: MJPEG stream on :8080/stream, detection API on :8081
"""

import cv2
import numpy as np
import threading
import time
import json
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import os

# ──────────────────────────────────────────────
# Try to import Jetson / Jetbot libs gracefully
# ──────────────────────────────────────────────
from jetbot import Robot
ROBOT_AVAILABLE = True

robot = Robot()

# ─────────────────────────
# CONFIG
# ─────────────────────────
STREAM_PORT   = 8080
API_PORT      = 8081

# Detection runs at this resolution (higher = better accuracy)
DETECT_WIDTH  = 2000
DETECT_HEIGHT = 1800

# MJPEG stream is downscaled to this for low latency
STREAM_WIDTH  = 600
STREAM_HEIGHT = 600

STREAM_FPS    = 10
JPEG_QUALITY  = 40       # 0-100, lower = smaller = faster

# flip-method for nvvidconv:
#   0=none  1=ccw-90  2=rotate-180  3=cw-90  4=horiz-flip  6=vert-flip
FLIP_METHOD   = 0

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# HSV colour ranges for detection
#   TARGET  : red box   (two ranges because red wraps around H=0)
#   GREEN   : single range
#   YELLOW  : single range
#   BLUE    : single range
#   PURPLE  : single range
# ─────────────────────────────────────────────────────────────────────────────

# Remove JETBOT_HSV_* and replace with:
GREEN_HSV_LOWER  = np.array([40,  80,  40])
GREEN_HSV_UPPER  = np.array([90, 255, 255])

# ─────────────────────────────────────────────────────────────────────────────
TARGET_HSV_LOWER1 = np.array([0,   120,  70])
TARGET_HSV_UPPER1 = np.array([10,  255, 255])
TARGET_HSV_LOWER2 = np.array([160, 120,  70])
TARGET_HSV_UPPER2 = np.array([180, 255, 255])


# ── Reduce minimum area significantly ──────────────────────────────────────
MIN_CONTOUR_AREA  = 100   # was 800 — distant objects are tiny

# ── Widen yellow: lower saturation floor, broader hue ──────────────────────
YELLOW_HSV_LOWER = np.array([18,  120,  80])  # raise sat from 60→120
YELLOW_HSV_UPPER = np.array([35,  255, 255])
# ── Widen blue ──────────────────────────────────────────────────────────────
BLUE_HSV_LOWER = np.array([100,  40,  30])   # tighter hue, higher sat
BLUE_HSV_UPPER = np.array([210, 255, 255])   # was 140 — drops dark blue clothing
# ── Widen purple ────────────────────────────────────────────────────────────
PURPLE_HSV_LOWER = np.array([130,  80,  60])  # raise sat from 40→80
PURPLE_HSV_UPPER = np.array([158, 255, 255])  # tighten upper hue from 165→158
# Real-world object heights used for distance estimation
TARGET_REAL_HEIGHT_CM = 27.0    # red target box
OTHER_REAL_HEIGHT_CM  = 14.5    # all other colours

# ─────────────────────────
# Shared state
# ─────────────────────────
_lock             = threading.Lock()
_latest_frame     = None          # raw BGR @ DETECT resolution
_latest_annotated = None          # BGR with boxes @ DETECT resolution
_latest_detection = {
    "target":  None,
    "greens":  [],      # was "jetbots"
    "yellows": [],
    "blues":   [],
    "purples": [],
    "timestamp": 0.0
}
_current_cmd = {"left": 0.0, "right": 0.0}


def apply_motors(left, right):
    """Clamp and apply motor speeds."""
    left  = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))
    _current_cmd["left"]  = left
    _current_cmd["right"] = right
    robot.set_motors(left, right)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_from_contour(cnt):
    """Return (cx, cy, w, h, area) for a contour."""
    x, y, w, h = cv2.boundingRect(cnt)
    cx = x + w // 2
    cy = y + h // 2
    area = cv2.contourArea(cnt)
    return {"cx": cx, "cy": cy, "w": w, "h": h, "area": int(area)}



FOCAL_LENGTH_PX = 1233.3   # calibrated at DETECT_HEIGHT = 1800
# Real-world object heights

def estimate_distance(bbox_height, frame_height, real_height_cm):
    focal_px = 1233
    if bbox_height < 2:
        return 9999.0
    return round(focal_px * real_height_cm / bbox_height, 1)

def estimate_direction(bbox_cx, frame_width):
    """
    Returns angle in degrees from centre: negative=left, positive=right.
    Approximately ±30° for ±60° horizontal FOV camera.
    """
    norm = (bbox_cx - frame_width / 2) / (frame_width / 2)   # -1..1
    return round(norm * 30.0, 1)


def _detect_color(hsv, lower, upper, frame_h, frame_w,
                  lower2=None, upper2=None, real_height_cm=OTHER_REAL_HEIGHT_CM):
    # ── ADD THIS: blur the HSV image to merge small distant fragments ───────
    hsv_blurred = cv2.GaussianBlur(hsv, (7, 7), 0)
    mask = cv2.inRange(hsv_blurred, lower, upper)   # use blurred copy
    # ────────────────────────────────────────────────────────────────────────
    if lower2 is not None:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_blurred, lower2, upper2))
    
    # Also increase closing kernel — stitches distant fragments together
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))  # was 5×5
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) 

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the single largest contour that exceeds the minimum area
    best_cnt  = None
    best_area = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA and area > best_area:
            best_cnt  = cnt
            best_area = area

    if best_cnt is None:
        return []

    b = _bbox_from_contour(best_cnt)
    b["distance_cm"] = estimate_distance(b["h"], frame_h, real_height_cm)
    b["direction_deg"] = estimate_direction(b["cx"], frame_w)
    return [b]


def detect_objects(frame):
    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ── Red target (27 cm tall) ──────────────────────────────────────────────
    red_blobs = _detect_color(hsv,
                              TARGET_HSV_LOWER1, TARGET_HSV_UPPER1, h, w,
                              lower2=TARGET_HSV_LOWER2, upper2=TARGET_HSV_UPPER2,
                              real_height_cm=TARGET_REAL_HEIGHT_CM)
    target = red_blobs[0] if red_blobs else None

    # ── Green objects (14.5 cm) ──────────────────────────────────────────────
    greens  = _detect_color(hsv, GREEN_HSV_LOWER,  GREEN_HSV_UPPER,  h, w,
                            real_height_cm=OTHER_REAL_HEIGHT_CM)

    # ── Yellow objects (14.5 cm) ─────────────────────────────────────────────
    yellows = _detect_color(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER, h, w,
                            real_height_cm=OTHER_REAL_HEIGHT_CM)

    # ── Blue objects (14.5 cm) ───────────────────────────────────────────────
    blues   = _detect_color(hsv, BLUE_HSV_LOWER,   BLUE_HSV_UPPER,   h, w,
                            real_height_cm=OTHER_REAL_HEIGHT_CM)

    # ── Purple objects (14.5 cm) ─────────────────────────────────────────────
    purples = _detect_color(hsv, PURPLE_HSV_LOWER, PURPLE_HSV_UPPER, h, w,
                            real_height_cm=OTHER_REAL_HEIGHT_CM)

    return {
        "target":    target,
        "greens":    greens,    # was "jetbots"
        "yellows":   yellows,
        "blues":     blues,
        "purples":   purples,
        "frame_w":   w,
        "frame_h":   h,
        "timestamp": time.time()
    }

def _draw_boxes(out, blobs, bgr_color, label_prefix):
    """Draw labelled bounding boxes for a list of blobs."""
    for i, b in enumerate(blobs):
        x1 = b["cx"] - b["w"] // 2
        y1 = b["cy"] - b["h"] // 2
        x2 = b["cx"] + b["w"] // 2
        y2 = b["cy"] + b["h"] // 2
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr_color, 2)
        label = "{}{} {:.1f}m {:.0f}deg".format(
            label_prefix, i + 1 if len(blobs) > 1 else "",
            b["distance_cm"], b["direction_deg"])
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr_color, 1, cv2.LINE_AA)


def annotate_frame(frame, detection):
    out = frame.copy()
    h, w = out.shape[:2]

    # Red target
    if detection["target"]:
        _draw_boxes(out, [detection["target"]], (0, 0, 255), "TARGET")

    # Green objects  – BGR (0, 200, 0)
    _draw_boxes(out, detection["greens"],   (0, 200,   0), "GRN")  # was "BOT"

    # Yellow objects – BGR (0, 220, 255)
    _draw_boxes(out, detection["yellows"], (0, 220, 255), "YEL")

    # Blue objects   – BGR (255, 100, 0)
    _draw_boxes(out, detection["blues"],   (255, 100,   0), "BLU")

    # Purple objects – BGR (200, 0, 200)
    _draw_boxes(out, detection["purples"], (200,   0, 200), "PUR")

    # Overlay: motor state
    cv2.putText(out, "L:{:.2f} R:{:.2f}".format(
        _current_cmd["left"], _current_cmd["right"]),
        (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA CAPTURE THREAD
# ─────────────────────────────────────────────────────────────────────────────


def camera_thread():
    global _latest_frame, _latest_annotated, _latest_detection

    # Capture at DETECT resolution (640×480) for better detection quality.
    # The stream handler downscales to STREAM_WIDTH×STREAM_HEIGHT before encoding.
    gst_pipe = (
        "nvarguscamerasrc sensor-mode=2 ! "
        "video/x-raw(memory:NVMM),width=2000,height=1800,framerate=30/1 ! "
        "nvvidconv flip-method={flip} ! "
        "video/x-raw,width={w},height={h},format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1"
    ).format(w=DETECT_WIDTH, h=DETECT_HEIGHT, flip=FLIP_METHOD)

    print("[CAM] Trying GStreamer pipeline...")
    cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[CAM] GStreamer failed, trying /dev/video0")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DETECT_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DETECT_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, STREAM_FPS)

    if not cap.isOpened():
        print("[CAM] ERROR: no camera available")
        return

    print("[CAM] Camera opened OK at {}x{}".format(DETECT_WIDTH, DETECT_HEIGHT))

    interval = 1.0 / STREAM_FPS
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        # Normalise to 3-channel BGR
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Ensure we are at the target detection resolution
        if frame.shape[1] != DETECT_WIDTH or frame.shape[0] != DETECT_HEIGHT:
            frame = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))

        detection = detect_objects(frame)
        annotated  = annotate_frame(frame, detection)

        with _lock:
            _latest_frame     = frame.copy()
            _latest_annotated = annotated.copy()
            _latest_detection = detection

        elapsed = time.time() - t0
        time.sleep(max(0.0, interval - elapsed))


# ─────────────────────────────────────────────────────────────────────────────
# STREAM SERVER  (port 8080)
# ─────────────────────────────────────────────────────────────────────────────

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass
    
    def _serve_example(self):
        with _lock:
            frame = _latest_annotated
        if frame is None:
            self.send_error(503, "No frame yet")
            return
        cv2.imwrite("example.png", frame)
        self._json(200, {"ok": True, "saved": "example.png"})


    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/stream":
            self._serve_mjpeg()
        elif path == "/example":
            self._serve_example()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


    def _encode_stream_frame(self, frame):
        """Downscale annotated frame to stream resolution before JPEG encode."""
        small = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.imencode(".jpg", small,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    def _serve_snapshot(self):
        with _lock:
            frame = _latest_annotated
        if frame is None:
            self.send_error(503, "No frame yet")
            return
        ok, jpg = self._encode_stream_frame(frame)
        if not ok:
            self.send_error(500)
            return
        data = jpg.tobytes()
        self.send_response(200)
        self.send_header("Content-Type",   "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control",  "no-cache, no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                t0 = time.time()
                with _lock:
                    frame = _latest_annotated
                if frame is not None:
                    ok, jpg = self._encode_stream_frame(frame)
                    if ok:
                        data = jpg.tobytes()
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            ("Content-Length: {}\r\n\r\n".format(len(data))).encode())
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                elapsed = time.time() - t0
                time.sleep(max(0.0, 1.0 / STREAM_FPS - elapsed))
        except (BrokenPipeError, ConnectionResetError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# API SERVER  (port 8081)  – JSON commands + detection data
# ─────────────────────────────────────────────────────────────────────────────

class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/detection":
            with _lock:
                d = dict(_latest_detection)
            self._json(200, d)
        elif self.path == "/status":
            self._json(200, {
                "motors": _current_cmd,
                "robot_available": ROBOT_AVAILABLE,
                "timestamp": time.time()
            })
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            cmd = json.loads(body.decode())
        except Exception:
            self.send_error(400, "Bad JSON")
            return

        if self.path == "/motors":
            left  = float(cmd.get("left",  0.0))
            right = float(cmd.get("right", 0.0))
            apply_motors(left, right)
            self._json(200, {"ok": True, "left": left, "right": right})

        elif self.path == "/stop":
            apply_motors(0, 0)
            self._json(200, {"ok": True})

        elif self.path == "/move":
            action = cmd.get("action", "stop")
            speed  = float(cmd.get("speed", 0.35))
            if action == "forward":
                apply_motors( speed,  speed)
            elif action == "back":
                apply_motors(-speed, -speed)
            elif action == "left":
                apply_motors(-speed,  speed)
            elif action == "right":
                apply_motors( speed, -speed)
            else:
                apply_motors(0, 0)
            self._json(200, {"ok": True, "action": action})

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

def run_server(handler_class, port):
    server = ReusableHTTPServer(("0.0.0.0", port), handler_class)
    print("[SERVER] Listening on port {}".format(port))
    server.serve_forever()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hostname = socket.gethostname()
    ip       = socket.gethostbyname(hostname)
    print("=" * 55)
    print("  JetBot Server  |  {}".format(ip))
    print("  Stream : http://{}:{}/stream".format(ip, STREAM_PORT))
    print("  API    : http://{}:{}/detection".format(ip, API_PORT))
    print("=" * 55)

    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_cam.start()

    t_api = threading.Thread(
        target=run_server,
        args=(APIHandler, API_PORT),
        daemon=True)
    t_api.start()

    run_server(StreamHandler, STREAM_PORT)