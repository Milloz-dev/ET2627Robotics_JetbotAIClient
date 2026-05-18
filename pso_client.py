#!/usr/bin/env python3
"""
PSO Client - runs on your laptop (Python 3.11+)
Controls all JetBots using Particle Swarm Optimization logic.

Usage:
    python pso_client.py                  # full PSO mode
    python pso_client.py --manual 0      # manually drive robot index 0
"""

import requests
import threading
import time
import random
import math
import json
import argparse
from typing import Optional, Dict, List

# ─────────────────────────
# CONFIG
# ─────────────────────────
ROBOT_IPS = [
    "194.47.156.201",
    "194.47.156.39",
    "194.47.156.43",
    "194.47.156.213",
]
STREAM_PORT  = 8080
API_PORT     = 8081
TIMEOUT      = 1.5          # seconds per HTTP request
PSO_INTERVAL = 0.4          # seconds between PSO update cycles

# PSO hyper-parameters
# PSO hyper-parameters — tuned to reduce spin
W   = 0.4    # lower inertia so velocity doesn't snowball
C1  = 1.2
C2  = 1.2
MAX_SPEED = 0.30
MAX_VX = 5.0   # clamp angular velocity
MAX_VY = 0.5   # clamp linear velocity

# Exploration turn: much slower
SEARCH_TURN_SPEED = 0.1   # was 0.30
# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL ROBOT COMMUNICATION
# ─────────────────────────────────────────────────────────────────────────────

def api(ip, path, method="GET", payload=None, timeout=TIMEOUT):
    url = "http://{}:{}/{}".format(ip, API_PORT, path.lstrip("/"))
    try:
        if method == "GET":
            r = requests.get(url, timeout=timeout)
        else:
            r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def get_detection(ip) -> Optional[Dict]:
    """Fetch the latest detection dict from a robot."""
    return api(ip, "/detection")


def send_motors(ip, left: float, right: float):
    """Set motor speeds directly."""
    return api(ip, "/motors", method="POST",
                payload={"left": left, "right": right})


def stop_robot(ip):
    return api(ip, "/stop", method="POST", payload={})


def move(ip, action: str, speed: float = 0.2):
    return api(ip, "/move", method="POST",
                payload={"action": action, "speed": speed})


def stream_url(ip) -> str:
    return "http://{}:{}/stream".format(ip, STREAM_PORT)


# ─────────────────────────────────────────────────────────────────────────────
# PSO STATE
# ─────────────────────────────────────────────────────────────────────────────

class Particle:
    """
    Represents one JetBot in the PSO swarm.

    Positional state is abstract (direction_deg, distance_m) because we
    have no GPS.  We treat:
        x = lateral angle to target  (-30 .. +30 degrees)
        y = distance to target       (0 .. ~3 m)

    Velocity maps to motor commands:
        vx → turning  (positive = turn right)
        vy → forward/back (negative = closer)
    """
    def __init__(self, ip: str, idx: int):
        self.ip  = ip
        self.idx = idx

        # "position" in abstract (angle, distance) space
        self.x  = 0.0      # direction_deg   (unknown until first detection)
        self.y  = 9999.0     # distance_m

        # velocity in abstract space
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-0.1, 0.1)

        # personal best (lowest distance = best)
        self.best_x   = self.x
        self.best_y   = self.y
        self.best_dist = 9999.0

        self.target_visible = False
        self.last_detection = None

    def update_from_detection(self, det):
        if det is None:
            return
        self.last_detection = det
        target = det.get("target")
        if target:
            self.target_visible = True
            self.x = target.get("direction_deg", 0.0)
            self.y = target.get("distance_cm", 9999.0)
            dist = math.sqrt(self.x**2 + self.y**2)
            if dist < self.best_dist:
                self.best_dist = dist
                self.best_x    = self.x
                self.best_y    = self.y
        else:
            self.target_visible = False
            # Don't update x/y — keep last known position
            # so velocity update doesn't snap to stale (0, 99)

    def pso_update(self, global_best_x, global_best_y, global_target_visible):
        r1x, r1y = random.random(), random.random()
        r2x, r2y = random.random(), random.random()

        self.vx = (W * self.vx
                + C1 * r1x * (self.best_x - self.x)
                + C2 * r2x * (global_best_x - self.x))
        self.vy = (W * self.vy
                + C1 * r1y * (self.best_y - self.y)
                + C2 * r2y * (global_best_y - self.y))

        self.vx = _clamp(self.vx, -MAX_VX, MAX_VX)
        self.vy = _clamp(self.vy, -MAX_VY, MAX_VY)

        if self.target_visible:
            # ── Direct proportional control toward THIS robot's own sighting ──
            # Don't rely on PSO velocity — act on raw sensor data immediately
            angle_deg = self.x          # e.g. -20 = target left, +20 = target right
            distance  = self.y          # metres

            forward = _clamp(0.2, 0.0, MAX_SPEED)          # constant drive forward
            turn    = _clamp(angle_deg * 0.015, -0.2, 0.2) # steer toward target

            # Stop if very close
            if distance < 250.0:
                forward = 0.0
                turn    = 0.0

        elif global_target_visible:
            # ── Another robot sees it — use PSO to navigate toward global best ──
            forward = _clamp(-self.vy * 1.5, -MAX_SPEED, MAX_SPEED)
            turn    = _clamp(self.vx * 0.25, -MAX_SPEED, MAX_SPEED)

        else:
            # ── No robot sees target — slow search spin ──
            forward = 0.0
            turn    = random.choice([-1, 1]) * 0.18

        left_speed  = _clamp(forward - turn, -MAX_SPEED, MAX_SPEED)
        right_speed = _clamp(forward + turn, -MAX_SPEED, MAX_SPEED)
        return left_speed, right_speed
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────────────────────
# PSO SWARM CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class SwarmController:
    def __init__(self, ips: List[str]):
        self.particles = [Particle(ip, i) for i, ip in enumerate(ips)]
        self.global_best_x    = 0.0
        self.global_best_y    = 9999.0
        self.global_best_dist = 9999.0
        self.global_target_visible = False
        self._running = False
        self._lock    = threading.Lock()

    # ── fetch all detections in parallel ────────────────────────────────────
    def _fetch_all(self):
        results = [None] * len(self.particles)
        def _fetch(idx, p):
            results[idx] = get_detection(p.ip)
        threads = [threading.Thread(target=_fetch, args=(i, p))
                   for i, p in enumerate(self.particles)]
        for t in threads: t.start()
        for t in threads: t.join()
        return results

    # ── send motors in parallel ──────────────────────────────────────────────
    def _send_all(self, commands: list):
        def _send(p, l, r):
            send_motors(p.ip, l, r)
        threads = [threading.Thread(target=_send, args=(p, l, r))
                   for p, (l, r) in zip(self.particles, commands)]
        for t in threads: t.start()
        for t in threads: t.join()

    # ── one PSO iteration ────────────────────────────────────────────────────
    def step(self):
        detections = self._fetch_all()

        # Update particles from detection
        for p, det in zip(self.particles, detections):
            p.update_from_detection(det)

        # Update global best
        for p in self.particles:
            if p.best_dist < self.global_best_dist:
                self.global_best_dist = p.best_dist
                self.global_best_x    = p.best_x
                self.global_best_y    = p.best_y

        self.global_target_visible = any(p.target_visible for p in self.particles)

        # Compute and apply motor commands
        commands = []
        for p in self.particles:
            if p.best_dist < self.global_best_dist and p.best_dist < 90.0:  # ← add this condition
                self.global_best_dist = p.best_dist
                self.global_best_x    = p.best_x
                self.global_best_y    = p.best_y
            l, r = p.pso_update(
                self.global_best_x,
                self.global_best_y,
                self.global_target_visible)
            commands.append((l, r))

        self._send_all(commands)

        # Print summary
        vis = [i for i, p in enumerate(self.particles) if p.target_visible]
        print(
            "[PSO] global_best=({:.1f}°, {:.2f}m)  target_visible={}"
            .format(self.global_best_x, self.global_best_y,
                    vis if vis else "none"))
        for i, (p, (l, r)) in enumerate(zip(self.particles, commands)):
            print("  Bot{} ip={}  motors=({:.2f},{:.2f})  dist={:.2f}m".format(
                i, p.ip, l, r, p.best_dist))

    def run(self):
        self._running = True
        print("[PSO] Swarm started with {} robots".format(len(self.particles)))
        try:
            while self._running:
                t0 = time.time()
                self.step()
                elapsed = time.time() - t0
                time.sleep(max(0.0, PSO_INTERVAL - elapsed))
        except KeyboardInterrupt:
            print("\n[PSO] Stopping all robots...")
            self.stop_all()

    def stop_all(self):
        self._running = False
        threads = [threading.Thread(target=stop_robot, args=(p.ip,))
                   for p in self.particles]
        for t in threads: t.start()
        for t in threads: t.join()
        print("[PSO] All robots stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL DRIVE (keyboard)
# ─────────────────────────────────────────────────────────────────────────────

def manual_drive(ip: str):
    """
    Simple keyboard control using arrow keys / WASD.
    Requires: pip install keyboard   (or run as root on Linux)
    Falls back to input() prompts if keyboard not available.
    """
    try:
        import keyboard as kb
        print("[MANUAL] Driving {}  (WASD / arrows, Q to quit)".format(ip))
        speed = 0.2
        while True:
            if kb.is_pressed("q"):
                break
            elif kb.is_pressed("w") or kb.is_pressed("up"):
                move(ip, "forward", speed)
            elif kb.is_pressed("s") or kb.is_pressed("down"):
                move(ip, "back", speed)
            elif kb.is_pressed("a") or kb.is_pressed("left"):
                move(ip, "left", speed/2)
            elif kb.is_pressed("d") or kb.is_pressed("right"):
                move(ip, "right", speed/2)
            else:
                stop_robot(ip)
            time.sleep(0.08)
    except ImportError:
        print("[MANUAL] 'keyboard' not installed – using text prompts.")
        print("Commands: w=forward  s=back  a=left  d=right  q=quit")
        while True:
            cmd = input("cmd> ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "w":
                move(ip, "forward")
            elif cmd == "s":
                move(ip, "back")
            elif cmd == "a":
                move(ip, "left")
            elif cmd == "d":
                move(ip, "right")
            else:
                stop_robot(ip)
    stop_robot(ip)
    print("[MANUAL] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def print_stream_urls():
    print("\n  ── Live stream URLs ───────────────────────────────")
    for i, ip in enumerate(ROBOT_IPS):
        print("  Bot{}: http://{}:{}/stream".format(i, ip, STREAM_PORT))
    print("  ───────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JetBot PSO Swarm Client")
    parser.add_argument("--manual", type=int, default=-1,
                        help="Index of robot to drive manually (0-3)")
    parser.add_argument("--ping", action="store_true",
                        help="Ping all robots and print detection status")
    args = parser.parse_args()

    print_stream_urls()

    if args.ping:
        print("[PING] Checking all robots...")
        for i, ip in enumerate(ROBOT_IPS):
            det = get_detection(ip)
            if det:
                print("  Bot{} ({}) OK – target_visible={}".format(
                    i, ip, det.get("target") is not None))
            else:
                print("  Bot{} ({}) UNREACHABLE".format(i, ip))

    elif args.manual >= 0:
        ip = ROBOT_IPS[args.manual]
        print("[MANUAL] Controlling Bot{} at {}".format(args.manual, ip))
        manual_drive(ip)

    else:
        swarm = SwarmController(ROBOT_IPS)
        swarm.run()
