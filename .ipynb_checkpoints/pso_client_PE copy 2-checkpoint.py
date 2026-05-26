#!/usr/bin/env python3

import requests
import threading
import time
import math
import argparse
from typing import Optional, Dict, List

ROBOT_IPS = [
    "194.47.156.43",
    "194.47.156.201",
    "194.47.156.39",
    "194.47.156.213"
]

colors = {
    "194.47.156.43": "purple",
    "194.47.156.201": "green",
    "194.47.156.39": "blue",
    "194.47.156.213": "yellow"
}

STREAM_PORT = 8080
API_PORT = 8081
TIMEOUT = 1.5
PSO_INTERVAL = 0.4

MAX_SPEED = 0.15
SEARCH_TURN_SPEED = 0.15

SIDE_OFFSET_DEG = 16.0
FOLLOW_STOP_CM = 25.0
GOAL_DISTANCE_CM = 40

FOLLOW_BASE_SPEED = 0.12
LEADER_SPEED = 0.13

SEARCH_TURN_GAIN = 0.005       # only search keeps high turning behavior
FOLLOW_TURN_GAIN_FAR = 0.0025  # lower turning when far away
FOLLOW_TURN_GAIN_CLOSE = 0.0008
LEADER_TURN_GAIN_FAR = 0.0015
LEADER_TURN_GAIN_CLOSE = 0.0006

DIST_NEAR_CM = 30.0
DIST_FAR_CM = 120.0


def api(ip, path, method="GET", payload=None, timeout=TIMEOUT):
    url = f"http://{ip}:{API_PORT}/{path.lstrip('/')}"
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
    return api(ip, "/motors", method="POST", payload={
        "left": float(left),
        "right": float(right)
    })


def stop_robot(ip):
    return api(ip, "/stop", method="POST", payload={})


def move(ip, action: str, speed: float = 0.2):
    return api(ip, "/move", method="POST", payload={
        "action": action,
        "speed": speed
    })


def stream_url(ip) -> str:
    return f"http://{ip}:{STREAM_PORT}/stream"


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def turn_gain_from_distance(distance_cm, close_gain, far_gain):
    """
    Close = lower turning.
    Far = higher turning.
    """
    d = _clamp(distance_cm, DIST_NEAR_CM, DIST_FAR_CM)
    t = (d - DIST_NEAR_CM) / (DIST_FAR_CM - DIST_NEAR_CM)
    return close_gain + t * (far_gain - close_gain)


def color_key(color):
    return color + "s"


def find_visible_leader(det, leader_states):
    if det is None:
        return None

    closest = None
    closest_dist = 9999

    for leader in leader_states:
        key = color_key(leader["color"])
        seen_robots = det.get(key, [])

        for obj in seen_robots:
            dist = obj.get("distance_cm", 9999)

            if dist < closest_dist:
                closest_dist = dist
                closest = {
                    "id": leader["id"],
                    "color": leader["color"],
                    "direction_deg": obj.get("direction_deg", 0.0),
                    "distance_cm": dist
                }

    return closest


def all_robots_ready(states):
    red_seen_robot_ids = {
        s["id"]
        for s in states
        if s["state"] in ["LEADER", "GOAL"]
    }

    for s in states:
        if s["state"] in ["LEADER", "GOAL"]:
            continue

        if s["state"] in ["DEPENDENT", "CHAIN_LEADER"]:
            if s["dependency"] in red_seen_robot_ids:
                continue

        return False

    return True


class Particle:
    def __init__(self, ip: str, idx: int):
        self.ip = ip
        self.idx = idx
        self.color = colors[ip]

        self.x = 0.0
        self.y = 9999.0
        self.best_dist = 9999.0

        self.target_visible = False
        self.last_detection = None

        self.state = "SEARCHING"
        self.dependency = None
        self.visible_leader = None

    def update_from_detection(self, det):
        self.last_detection = det

        if det is None:
            self.target_visible = False
            return

        target = det.get("target")

        if target:
            self.target_visible = True
            self.x = target.get("direction_deg", 0.0)
            self.y = target.get("distance_cm", 9999.0)

            dist = math.sqrt(self.x ** 2 + self.y ** 2)

            if dist < self.best_dist:
                self.best_dist = dist
        else:
            self.target_visible = False

    def command(self, states):
        if self.state == "GOAL":
            print(f"Bot{self.idx} {self.color} GOAL -> stop")
            return 0.0, 0.0

        if self.state == "LEADER":
            print(
                f"Bot{self.idx} {self.color} LEADER sees red "
                f"angle={self.x:.1f}°, dist={self.y:.1f}cm"
            )

            if all_robots_ready(states):
                gain = turn_gain_from_distance(
                    self.y,
                    LEADER_TURN_GAIN_CLOSE,
                    LEADER_TURN_GAIN_FAR
                )

                leader_turn = self.x * gain

                left_speed = LEADER_SPEED + leader_turn
                right_speed = LEADER_SPEED - leader_turn

                left_speed = _clamp(left_speed, -MAX_SPEED, MAX_SPEED)
                right_speed = _clamp(right_speed, -MAX_SPEED, MAX_SPEED)

                print(
                    f"Bot{self.idx} {self.color} LEADER -> move, "
                    f"gain={gain:.4f}"
                )
            else:
                left_speed = 0.0
                right_speed = 0.0

                print(f"Bot{self.idx} {self.color} LEADER -> waiting for others")

            return left_speed, right_speed

        if self.state in ["DEPENDENT", "CHAIN_LEADER"] and self.visible_leader is not None:
            angle_deg = self.visible_leader["direction_deg"]
            distance = self.visible_leader["distance_cm"]

            if angle_deg >= 0:
                side_angle = angle_deg + SIDE_OFFSET_DEG
            else:
                side_angle = angle_deg - SIDE_OFFSET_DEG

            if distance < FOLLOW_STOP_CM:
                left_speed = 0.0
                right_speed = 0.0
            else:
                gain = turn_gain_from_distance(
                    distance,
                    FOLLOW_TURN_GAIN_CLOSE,
                    FOLLOW_TURN_GAIN_FAR
                )

                left_speed = FOLLOW_BASE_SPEED + (side_angle * gain)
                right_speed = FOLLOW_BASE_SPEED - (side_angle * gain)

                left_speed = _clamp(left_speed, -MAX_SPEED, MAX_SPEED)
                right_speed = _clamp(right_speed, -MAX_SPEED, MAX_SPEED)

            print(
                f"Bot{self.idx} {self.color} {self.state} -> follows "
                f"{self.visible_leader['color']} side_angle={side_angle:.1f}°, "
                f"dist={distance:.1f}cm"
            )

            return left_speed, right_speed

        print(f"Bot{self.idx} {self.color} SEARCHING -> spin")
        return SEARCH_TURN_SPEED, 0.0


class SwarmController:
    def __init__(self, ips: List[str]):
        self.particles = [Particle(ip, i) for i, ip in enumerate(ips)]
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
                "ip": p.ip,
                "color": p.color,
                "state": p.state,
                "dependency": p.dependency,
                "best_dist": p.best_dist,
                "target_visible": p.target_visible,
            }
            for p in self.particles
        ]

    def step(self):
        detections = self._fetch_all()

        for p, det in zip(self.particles, detections):
            p.update_from_detection(det)

        for p in self.particles:
            p.dependency = None
            p.visible_leader = None

            if p.target_visible:
                if p.y < GOAL_DISTANCE_CM:
                    p.state = "GOAL"
                else:
                    p.state = "LEADER"
            else:
                p.state = "SEARCHING"

        leader_states = [
            {
                "id": p.idx,
                "ip": p.ip,
                "color": p.color,
                "state": p.state,
                "best_dist": p.best_dist
            }
            for p in self.particles
            if p.state in ["LEADER", "GOAL"]
        ]

        if leader_states:
            print("\nRed box seen by:", [s["color"] for s in leader_states])
        else:
            print("\nNo robot sees red box")

        for p in self.particles:
            if p.state in ["LEADER", "GOAL"]:
                continue

            visible_leader = find_visible_leader(p.last_detection, leader_states)

            if visible_leader is not None:
                p.state = "DEPENDENT"
                p.dependency = visible_leader["id"]
                p.visible_leader = visible_leader
            else:
                p.state = "SEARCHING"
                p.dependency = None
                p.visible_leader = None

        for p in self.particles:
            if p.state == "DEPENDENT":
                has_follower = any(
                    q.dependency == p.idx
                    for q in self.particles
                    if q.idx != p.idx
                )

                if has_follower:
                    p.state = "CHAIN_LEADER"

        states = self.build_states()
        ready = all_robots_ready(states)

        print("--- STATES ---")
        print(f"All robots ready: {ready}")

        for s in states:
            print(
                f"Bot{s['id']} {s['color']}: "
                f"{s['state']} dep={s['dependency']}"
            )

        commands = []

        for p in self.particles:
            l, r = p.command(states)
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
            print("\nStopping all robots...")
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

        print("All robots stopped.")


def manual_drive(ip: str):
    try:
        import keyboard as kb

        print(f"[MANUAL] Driving {ip}  (WASD / arrows, Q to quit)")
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


def print_stream_urls():
    print("\n  ── Live stream URLs ───────────────────────────────")

    for i, ip in enumerate(ROBOT_IPS):
        print(f"  Bot{i} {colors[ip]}: {stream_url(ip)}")

    print("  ───────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JetBot Local Leader-Follower Client")

    parser.add_argument("--manual", type=int, default=-1)
    parser.add_argument("--ping", action="store_true")

    args = parser.parse_args()

    print_stream_urls()

    if args.ping:
        print("[PING] Checking all robots...")

        for i, ip in enumerate(ROBOT_IPS):
            det = get_detection(ip)

            if det:
                print(
                    f"  Bot{i} {colors[ip]} ({ip}) OK "
                    f"target_visible={det.get('target') is not None}"
                )
            else:
                print(f"  Bot{i} {colors[ip]} ({ip}) UNREACHABLE")

    elif args.manual >= 0:
        ip = ROBOT_IPS[args.manual]
        print(f"[MANUAL] Controlling Bot{args.manual} {colors[ip]} at {ip}")
        manual_drive(ip)

    else:
        swarm = SwarmController(ROBOT_IPS)
        swarm.run()