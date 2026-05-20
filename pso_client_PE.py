#!/usr/bin/env python3

import requests
import threading
import time
import random
import math
import argparse
from typing import Optional, Dict, List

# ─────────────────────────
# CONFIG
# ─────────────────────────

ROBOT_IPS = [
    "194.47.156.43",
    "194.47.156.201",
    "194.47.156.39"
]

STREAM_PORT  = 8080
API_PORT     = 8081
TIMEOUT      = 1.5
PSO_INTERVAL = 0.4

W   = 0.12
C1  = 1.2
C2  = 1.2
MAX_SPEED = 0.15
MAX_VX = 0.1
MAX_VY = 0.1

SEARCH_TURN_SPEED = 0.15


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
    except Exception:
        return None


def get_detection(ip) -> Optional[Dict]:
    return api(ip, "/detection")


def send_motors(ip, left: float, right: float):
    return api(
        ip,
        "/motors",
        method="POST",
        payload={"left": left, "right": right}
    )


def stop_robot(ip):
    return api(ip, "/stop", method="POST", payload={})


def move(ip, action: str, speed: float = 0.2):
    return api(
        ip,
        "/move",
        method="POST",
        payload={"action": action, "speed": speed}
    )


def stream_url(ip) -> str:
    return "http://{}:{}/stream".format(ip, STREAM_PORT)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────────────────────
# LEADER / DEPENDENCY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def find_best_leader(states):
    leaders = [s for s in states if s["state"] in ["LEADER", "GOAL"]]

    if not leaders:
        return None

    best = min(leaders, key=lambda s: s["best_dist"])
    return best["id"]


def leader_can_move(leader_id, states):
    """
    Leader may move only if no non-goal/non-leader robot depends on it.
    """
    for s in states:
        if s["id"] == leader_id:
            continue

        if s["state"] in ["LEADER", "GOAL"]:
            continue

        if s["dependency"] == leader_id:
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# PARTICLE / ROBOT STATE
# ─────────────────────────────────────────────────────────────────────────────

class Particle:
    def __init__(self, ip: str, idx: int):
        self.ip  = ip
        self.idx = idx

        self.x = 0.0
        self.y = 9999.0

        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(-0.05, 0.05)

        self.best_x = self.x
        self.best_y = self.y
        self.best_dist = 9999.0

        self.target_visible = False
        self.last_detection = None

        self._sweep_dir = random.choice([-1, 1])
        self._sweep_cycles = 0
        self._sweep_max = 12
        self._just_saw = 0

        self.state = "SEARCHING"
        self.dependency = None

    def update_from_detection(self, det):
        if det is None:
            self.target_visible = False
            return

        self.last_detection = det
        target = det.get("target")

        if target:
            self.target_visible = True
            self.x = target.get("direction_deg", 0.0)
            self.y = target.get("distance_cm", 9999.0)

            dist = math.sqrt(self.x ** 2 + self.y ** 2)

            if dist < self.best_dist:
                self.best_dist = dist
                self.best_x = self.x
                self.best_y = self.y
        else:
            self.target_visible = False

    def update_pso_velocity(self, global_best_x, global_best_y):
        r1x, r1y = random.random(), random.random()
        r2x, r2y = random.random(), random.random()

        self.vx = (
            W * self.vx
            + C1 * r1x * (self.best_x - self.x)
            + C2 * r2x * (global_best_x - self.x)
        )

        self.vy = (
            W * self.vy
            + C1 * r1y * (self.best_y - self.y)
            + C2 * r2y * (global_best_y - self.y)
        )

        self.vx = _clamp(self.vx, -MAX_VX, MAX_VX)
        self.vy = _clamp(self.vy, -MAX_VY, MAX_VY)

    def pso_update(self, global_best_x, global_best_y, global_target_visible, states):
        self.update_pso_velocity(global_best_x, global_best_y)

        # ─────────────────────────────
        # PRIORITY 1: RED BOX
        # ─────────────────────────────
        if self.target_visible:
            self.dependency = None

            if self.y < 20.0:
                self.state = "GOAL"
                print(f"Bot{self.idx} GOAL -> stop")
                return 0.0, 0.0

            self.state = "LEADER"

            print(
                f"Bot{self.idx} LEADER sees red "
                f"angle={self.x:.1f}°, dist={self.y:.1f}cm"
            )

            if leader_can_move(self.idx, states):
                angle_deg = self.x

                left_speed = 0.15 + (angle_deg * 0.0005)
                right_speed = 0.15 - (angle_deg * 0.0005)

                print(f"Bot{self.idx} LEADER -> move toward red")

            else:
                left_speed = 0.0
                right_speed = 0.0

                print(f"Bot{self.idx} LEADER -> waiting, someone depends on me")

        # ─────────────────────────────
        # SOMEONE ELSE SEES RED
        # ─────────────────────────────
        elif global_target_visible:
            self.state = "DEPENDENT"
            self.dependency = find_best_leader(states)

            forward = _clamp(-self.vy * 1.5, -MAX_SPEED, MAX_SPEED)
            turn = _clamp(self.vx * 0.25, -MAX_SPEED, MAX_SPEED)

            left_speed = _clamp(forward - turn, -MAX_SPEED, MAX_SPEED)
            right_speed = _clamp(forward + turn, -MAX_SPEED, MAX_SPEED)

            print(
                f"Bot{self.idx} DEPENDENT on Bot{self.dependency} "
                f"-> PSO move"
            )

        # ─────────────────────────────
        # NO RED SEEN BY ANYONE
        # ─────────────────────────────
        else:
            self.state = "SEARCHING"
            self.dependency = None

            left_speed = SEARCH_TURN_SPEED
            right_speed = 0.0

            print(f"Bot{self.idx} SEARCHING")

        print(
            f"Bot{self.idx}: state={self.state}, "
            f"dep={self.dependency}, "
            f"motors=({left_speed:.2f}, {right_speed:.2f})"
        )

        return left_speed, right_speed


# ─────────────────────────────────────────────────────────────────────────────
# SWARM CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class SwarmController:
    def __init__(self, ips: List[str]):
        self.particles = [Particle(ip, i) for i, ip in enumerate(ips)]

        self.global_best_x = 0.0
        self.global_best_y = 9999.0
        self.global_best_dist = 9999.0

        self.global_target_visible = False
        self._running = False

    def _fetch_all(self):
        results = [None] * len(self.particles)

        def _fetch(idx, p):
            results[idx] = get_detection(p.ip)

        threads = [
            threading.Thread(target=_fetch, args=(i, p))
            for i, p in enumerate(self.particles)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        return results

    def _send_all(self, commands):
        def _send(p, l, r):
            send_motors(p.ip, l, r)

        threads = [
            threading.Thread(target=_send, args=(p, l, r))
            for p, (l, r) in zip(self.particles, commands)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    def build_states(self):
        return [
            {
                "id": p.idx,
                "state": p.state,
                "dependency": p.dependency,
                "best_dist": p.best_dist,
                "target_visible": p.target_visible
            }
            for p in self.particles
        ]

    def step(self):
        detections = self._fetch_all()

        for p, det in zip(self.particles, detections):
            p.update_from_detection(det)

        for p in self.particles:
            if p.best_dist < self.global_best_dist:
                self.global_best_dist = p.best_dist
                self.global_best_x = p.best_x
                self.global_best_y = p.best_y

        self.global_target_visible = any(p.target_visible for p in self.particles)

        states = self.build_states()

        commands = []

        for p in self.particles:
            l, r = p.pso_update(
                self.global_best_x,
                self.global_best_y,
                self.global_target_visible,
                states
            )

            commands.append((l, r))

        self._send_all(commands)

    def run(self):
        self._running = True

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

        threads = [
            threading.Thread(target=stop_robot, args=(p.ip,))
            for p in self.particles
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        print("[PSO] All robots stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL DRIVE
# ─────────────────────────────────────────────────────────────────────────────

def manual_drive(ip: str):
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
                move(ip, "left", speed)
            elif kb.is_pressed("d") or kb.is_pressed("right"):
                move(ip, "right", speed)
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
    parser = argparse.ArgumentParser(description="JetBot Leader/Dependency Swarm Client")

    parser.add_argument(
        "--manual",
        type=int,
        default=-1,
        help="Index of robot to drive manually"
    )

    parser.add_argument(
        "--ping",
        action="store_true",
        help="Ping all robots and print detection status"
    )

    args = parser.parse_args()

    print_stream_urls()

    if args.ping:
        print("[PING] Checking all robots...")

        for i, ip in enumerate(ROBOT_IPS):
            det = get_detection(ip)

            if det:
                print(
                    "  Bot{} ({}) OK – target_visible={}".format(
                        i,
                        ip,
                        det.get("target") is not None
                    )
                )
            else:
                print("  Bot{} ({}) UNREACHABLE".format(i, ip))

    elif args.manual >= 0:
        ip = ROBOT_IPS[args.manual]
        print("[MANUAL] Controlling Bot{} at {}".format(args.manual, ip))
        manual_drive(ip)

    else:
        swarm = SwarmController(ROBOT_IPS)
        swarm.run()