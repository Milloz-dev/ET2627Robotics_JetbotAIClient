#!/usr/bin/env python3
"""
Robot Follower Client - runs on your laptop (Python 3.11+)
Robots search for the target by spinning, then follow it.
If a robot can't see the target but another can, it drives forward.

Usage:
    python robot_client.py              # run all robots
    python robot_client.py --manual 0   # manually drive robot index 0
    python robot_client.py --ping       # check connectivity
"""

import requests
import threading
import time
import argparse
from typing import Optional, Dict, List
import sys


TIMEOUT = 2.0  # seconds
# ─────────────────────────
# CONFIG
# ─────────────────────────

ROBOTS = {
    "194.47.156.39":  "blues",
    "194.47.156.201": "greens",
    "194.47.156.43":  "purples",
    "194.47.156.213": "yellows",
}

COLORS = {
    "blues": False,
    "greens": False,
    "purples": False,
    "yellows": False
}


shared_state = {}

STREAM_PORT  = 8080
API_PORT     = 8081

# Shared stop event — set this to signal all threads to exit
stop_event = threading.Event()


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
    return api(ip, "/detection")


def send_motors(ip, left: float, right: float):
    return api(ip, "/motors", method="POST",
               payload={"left": left, "right": right})


def stop_robot(ip):
    return api(ip, "/stop", method="POST", payload={})


def move(ip, action: str, speed: float = 0.2):
    return api(ip, "/move", method="POST",
               payload={"action": action, "speed": speed})


def stream_url(ip) -> str:
    return "http://{}:{}/stream".format(ip, STREAM_PORT)


def ping_all():
    """Check which robots are reachable."""
    print("Pinging robots...")
    any_ok = False
    for ip, color in ROBOTS.items():
        result = api(ip, "/detection", timeout=3.0)
        if result is not None:
            print(f"  ✓ {color} ({ip}) — reachable")
            any_ok = True
        else:
            print(f"  ✗ {color} ({ip}) — no response")
    return any_ok


class RobotClient(threading.Thread):
    def __init__(self, idx: int, ip: str, shared_state: Dict, stop_event: threading.Event):
        super().__init__(daemon=False)  # NOT daemon — main thread waits for us
        self.idx = idx
        self.ip = ip
        self.color = ROBOTS.get(ip, "unknown")
        self.shared_state = shared_state
        self.stop_event = stop_event
        self.state = "searching"
        self.status = False
        self.running = True
        self.target_just_seen = False

    def stop(self):
        stop_robot(self.ip)

    def run(self):
        self.running = True
        try:
            while self.running:
                detection = get_detection(self.ip)

                if detection is None:
                    print(f"Bot{self.idx} ({self.color}) — no response, retrying...")
                    # Sleep in small increments so Ctrl+C is responsive
                    continue

                target   = detection.get("target")
                distance = target.get("distance_cm", 0) if target else None

                other_detections = [
                    detection.get(c)
                    for c in ROBOTS.values()
                    if c != self.color and detection.get(c) and COLORS.get(c, False)
                ]

                # Decide action based on detection and shared state

                if any(other_detections):
                    self.state = "following"
                    if self.color != "purples":  # purples are followers, never leaders
                        COLORS[self.color] = True
                    target2   = other_detections[0][0]
                    degree    = target2.get("direction_deg", 0.0)
                    distance2 = target2.get("distance_cm", 0)

                    if distance2 < 20:
                        self.running = False
                        self.state = "stopped"
                        self.stop()
                        return

                    left_speed  =  degree * 0.0005 + 0.15
                    right_speed = -degree * 0.0005 + 0.15
                    send_motors(self.ip, left_speed, right_speed)
                elif target:
                    self.status = True
                    COLORS[self.color] = True
                    if distance < 20:
                        self.status = False
                        self.state = "stopped"
                        print(f"Bot{self.idx} ({self.color}) — target reached, stopping")
                        self.stop()
                        return

                    degree = target.get("direction_deg", 0.0)
                    self.state = "following"
                    left_speed  =  degree * 0.0005 + 0.15
                    right_speed = -degree * 0.0005 + 0.15
                    self.target_just_seen = True
                    send_motors(self.ip, left_speed, right_speed)
                else:
                    self.running = True
                    COLORS[self.color] = False
                    left_speed, right_speed = 0.12, 0.0  # spin in place
                    send_motors(self.ip, left_speed, right_speed)


        finally:
            # Always stop the physical robot when the thread exits for any reason
            print(f"Bot{self.idx} ({self.color}) — stopping motors")
            self.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ping", action="store_true", help="Check robot connectivity and exit")
    args = parser.parse_args()

    if args.ping:
        ok = ping_all()
        sys.exit(0 if ok else 1)

    # Quick connectivity check before starting threads
    print("Checking robot connectivity...")
    reachable = []
    for ip, color in ROBOTS.items():
        result = api(ip, "/detection", timeout=3.0)
        if result is not None:
            print(f"  ✓ {color} ({ip})")
            reachable.append(ip)
        else:
            print(f"  ✗ {color} ({ip}) — skipping (unreachable)")

    if not reachable:
        print("\nERROR: No robots reachable. Check your network/VPN connection.")
        sys.exit(1)

    print(f"\nStarting {len(reachable)} robot(s)...\n")

    clients = [
        RobotClient(i, ip, shared_state, stop_event)
        for i, ip in enumerate(reachable)
    ]

    try:
        for c in clients:
            c.start()

        # Keep main thread alive; small sleep so Ctrl+C is felt immediately
        while not stop_event.is_set():
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nCtrl+C received — stopping all robots...")

    finally:
        stop_event.set()
        for c in clients:
            c.join(timeout=5)
        print("Done.")