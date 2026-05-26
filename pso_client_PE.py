#!/usr/bin/env python3

import requests
import threading
import time
import argparse
import random
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

INTERVAL = 0.6
MOTOR_SEND_INTERVAL = 0.8
MOTOR_CHANGE_EPS = 0.025

MAX_SPEED = 0.2
SEARCH_TURN_SPEED = 0.12

GOAL_DISTANCE_CM = 25
GOAL_CONFIRM_FRAMES = 2
GOAL_CONFIRM_TIME = 0.30

LEADER_WAIT_SPEED = 0.06
RED_CONFIRM_FRAMES = 2
RED_CONFIRM_TIME = 0.50
MAX_FAKE_RED_DISTANCE_CM = 300.0

LEADER_SPEED = 0.12
LEADER_TURN_GAIN_CLOSE = 0.0005
LEADER_TURN_GAIN_FAR = 0.0015

FOLLOW_SPEED = 0.15
FOLLOW_TURN_GAIN_CLOSE = 0.001
FOLLOW_TURN_GAIN_FAR = 0.0020

SIDE_OFFSET_DEG = 15.0
FOLLOW_SLOW_CM = 45.0
FOLLOW_STOP_CM = 20.0

BACKUP_TIME = 3.0
BACKUP_FAST = -0.15
BACKUP_SLOW = -0.10

STUCK_STILL_TIME = 2.5
STUCK_ESCAPE_TIME = 1.5
STUCK_EPS = 0.025
STUCK_BACK_FAST = -0.12
STUCK_BACK_SLOW = -0.06

DIST_NEAR_CM = 30.0
DIST_FAR_CM = 100.0


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
    return api(ip, "/motors", method="POST",
               payload={"left": float(left), "right": float(right)})


def stop_robot(ip):
    return api(ip, "/stop", method="POST", payload={})


def stream_url(ip):
    return f"http://{ip}:{STREAM_PORT}/stream"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def gain_from_distance(distance_cm, close_gain, far_gain):
    d = clamp(distance_cm, DIST_NEAR_CM, DIST_FAR_CM)
    t = (d - DIST_NEAR_CM) / (DIST_FAR_CM - DIST_NEAR_CM)
    return close_gain + t * (far_gain - close_gain)


def color_key(color):
    return color + "s"


def all_robots_ready(states):
    for s in states:
        if s["state"] in ["LEADER", "GOAL"]:
            continue

        if s["state"] in ["DEPENDENT", "CHAIN_LEADER"]:
            if s.get("leader_distance", 9999) <= FOLLOW_SLOW_CM:
                continue

        return False

    return True


def find_visible_red_robot(det, red_robot_states):
    if det is None:
        return None

    closest = None
    closest_dist = 9999.0

    for leader in red_robot_states:
        key = color_key(leader["color"])
        seen_objects = det.get(key, [])

        for obj in seen_objects:
            dist = obj.get("distance_cm", 9999.0)

            if dist < closest_dist:
                closest_dist = dist
                closest = {
                    "id": leader["id"],
                    "color": leader["color"],
                    "direction_deg": obj.get("direction_deg", 0.0),
                    "distance_cm": dist
                }

    return closest


class RobotState:
    def __init__(self, ip, idx):
        self.ip = ip
        self.idx = idx
        self.color = colors[ip]

        self.state = "SEARCHING"
        self.dependency = None
        self.visible_leader = None

        self.target_visible = False
        self.red_confirmed = False

        self.red_angle = 0.0
        self.red_distance = 9999.0

        self.red_seen_count = 0
        self.first_red_time = None

        self.goal_seen_count = 0
        self.goal_first_seen_time = None
        self.reached_goal = False

        self.backup_until = 0.0

        self.still_since = None
        self.stuck_escape_until = 0.0
        self.stuck_escape_dir = random.choice([-1, 1])

        self.last_detection = None

        self.last_red_seen_time = 0
        self.last_red_angle = 0.0

        self.last_left = None
        self.last_right = None
        self.last_motor_send_time = 0.0

    def update_detection(self, det):
        self.last_detection = det

        if self.reached_goal:
            self.target_visible = True
            self.red_confirmed = True
            return

        target = None if det is None else det.get("target")

        if target:
            self.target_visible = True
            self.red_angle = target.get("direction_deg", 0.0)
            self.last_red_seen_time = time.time()
            self.last_red_angle = self.red_angle
            self.red_distance = target.get("distance_cm", 9999.0)

            if self.red_distance > MAX_FAKE_RED_DISTANCE_CM:
                self.red_seen_count = 0
                self.first_red_time = None
                self.red_confirmed = False
                self.goal_seen_count = 0
                self.goal_first_seen_time = None
                return

            if self.first_red_time is None:
                self.first_red_time = time.time()

            self.red_seen_count += 1
            seen_time = time.time() - self.first_red_time

            if self.red_seen_count >= RED_CONFIRM_FRAMES or seen_time >= RED_CONFIRM_TIME:
                self.red_confirmed = True

            if self.red_distance <= GOAL_DISTANCE_CM:
                if self.goal_first_seen_time is None:
                    self.goal_first_seen_time = time.time()

                self.goal_seen_count += 1
                goal_seen_time = time.time() - self.goal_first_seen_time

                if (
                    self.goal_seen_count >= GOAL_CONFIRM_FRAMES
                    or goal_seen_time >= GOAL_CONFIRM_TIME
                ):
                    self.reached_goal = True
                    self.red_confirmed = True
            else:
                self.goal_seen_count = 0
                self.goal_first_seen_time = None

        else:
            self.target_visible = False
            self.red_confirmed = False
            self.red_seen_count = 0
            self.first_red_time = None
            self.goal_seen_count = 0
            self.goal_first_seen_time = None

    def decide_base_state(self):
        self.dependency = None
        self.visible_leader = None

        if self.reached_goal:
            self.state = "GOAL"
        elif self.red_confirmed:
            self.state = "LEADER"
        elif self.target_visible and not self.red_confirmed:
            self.state = "VERIFYING_RED"
        else:
            self.state = "SEARCHING"

    def stuck_escape(self, left, right, allow_escape=True):
        now = time.time()

        if not allow_escape:
            self.still_since = None
            return left, right

        if now < self.stuck_escape_until:
            print(f"Bot{self.idx} {self.color}: stuck escape -> back up and turn")

            if self.stuck_escape_dir > 0:
                return STUCK_BACK_FAST, STUCK_BACK_SLOW
            else:
                return STUCK_BACK_SLOW, STUCK_BACK_FAST

        is_still = abs(left) < STUCK_EPS and abs(right) < STUCK_EPS

        if is_still:
            if self.still_since is None:
                self.still_since = now

            if now - self.still_since >= STUCK_STILL_TIME:
                self.stuck_escape_until = now + STUCK_ESCAPE_TIME
                self.stuck_escape_dir *= -1
                self.still_since = None

                print(f"Bot{self.idx} {self.color}: stood still too long -> escape")

                if self.stuck_escape_dir > 0:
                    return STUCK_BACK_FAST, STUCK_BACK_SLOW
                else:
                    return STUCK_BACK_SLOW, STUCK_BACK_FAST
        else:
            self.still_since = None

        return left, right

    def command(self, states):
        if self.state == "GOAL":
            print(f"Bot{self.idx} {self.color}: GOAL -> stand still")
            return 0.0, 0.0

        if self.state == "VERIFYING_RED":
            print(
                f"Bot{self.idx} {self.color}: VERIFYING_RED "
                f"{self.red_seen_count}/{RED_CONFIRM_FRAMES} -> stop"
            )
            return 0.0, 0.0

        if self.state == "LEADER":
            ready = all_robots_ready(states)

            gain = gain_from_distance(
                self.red_distance,
                LEADER_TURN_GAIN_CLOSE,
                LEADER_TURN_GAIN_FAR
            )

            turn = self.red_angle * gain

            if ready:
                speed = LEADER_SPEED
                print(f"Bot{self.idx} {self.color}: LEADER -> all ready, move to red")
            else:
                speed = LEADER_WAIT_SPEED
                print(f"Bot{self.idx} {self.color}: LEADER -> slow creep while waiting")

            left = speed + turn
            right = speed - turn

            left = clamp(left, -MAX_SPEED, MAX_SPEED)
            right = clamp(right, -MAX_SPEED, MAX_SPEED)

            return self.stuck_escape(left, right, allow_escape=True)

        if self.state in ["DEPENDENT", "CHAIN_LEADER"] and self.visible_leader is not None:
            angle = self.visible_leader["direction_deg"]
            distance = self.visible_leader["distance_cm"]

            if angle >= 0:
                side_angle = angle + SIDE_OFFSET_DEG
            else:
                side_angle = angle - SIDE_OFFSET_DEG

            now = time.time()

            if distance < FOLLOW_STOP_CM:
                self.backup_until = now + BACKUP_TIME

            if now < self.backup_until:
                print(
                    f"Bot{self.idx} {self.color}: too close to "
                    f"{self.visible_leader['color']} -> backing up"
                )

                if side_angle >= 0:
                    return BACKUP_FAST, BACKUP_SLOW
                else:
                    return BACKUP_SLOW, BACKUP_FAST

            if distance < FOLLOW_SLOW_CM:
                speed = FOLLOW_SPEED * 0.55
            else:
                speed = FOLLOW_SPEED

            gain = gain_from_distance(
                distance,
                FOLLOW_TURN_GAIN_CLOSE,
                FOLLOW_TURN_GAIN_FAR
            )

            turn = side_angle * gain

            left = speed + turn
            right = speed - turn

            left = clamp(left, -MAX_SPEED, MAX_SPEED)
            right = clamp(right, -MAX_SPEED, MAX_SPEED)

            print(
                f"Bot{self.idx} {self.color}: {self.state} -> follow "
                f"{self.visible_leader['color']} side_angle={side_angle:.1f}, "
                f"dist={distance:.1f}, gain={gain:.4f}"
            )

            return self.stuck_escape(left, right, allow_escape=True)

        now = time.time()

        if now - self.last_red_seen_time < 0.7:
            print(
                f"Bot{self.idx} {self.color}: SEARCHING "
                f"(recent red memory) -> slow tracking"
            )

            angle = self.last_red_angle
            turn = angle * 0.002

            left = 0.07 + turn
            right = 0.07 - turn

            left = clamp(left, -MAX_SPEED, MAX_SPEED)
            right = clamp(right, -MAX_SPEED, MAX_SPEED)

            return self.stuck_escape(left, right, allow_escape=True)

        print(f"Bot{self.idx} {self.color}: SEARCHING -> fast spin")
        return SEARCH_TURN_SPEED, 0.0


class Controller:
    def __init__(self, ips: List[str]):
        self.robots = [RobotState(ip, i) for i, ip in enumerate(ips)]

    def fetch_all(self):
        results = [None] * len(self.robots)

        def fetch(idx, robot):
            results[idx] = get_detection(robot.ip)

        threads = [
            threading.Thread(target=fetch, args=(i, robot))
            for i, robot in enumerate(self.robots)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        return results

    def send_all(self, commands):
        now = time.time()

        def should_send(robot, left, right):
            if robot.last_left is None or robot.last_right is None:
                return True

            if now - robot.last_motor_send_time >= MOTOR_SEND_INTERVAL:
                return True

            if abs(left - robot.last_left) >= MOTOR_CHANGE_EPS:
                return True

            if abs(right - robot.last_right) >= MOTOR_CHANGE_EPS:
                return True

            return False

        send_jobs = []

        for robot, (left, right) in zip(self.robots, commands):
            if should_send(robot, left, right):
                robot.last_left = left
                robot.last_right = right
                robot.last_motor_send_time = now
                send_jobs.append((robot, left, right))

        if not send_jobs:
            return

        threads = [
            threading.Thread(target=send_motors, args=(robot.ip, left, right))
            for robot, left, right in send_jobs
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    def build_states(self):
        return [
            {
                "id": r.idx,
                "ip": r.ip,
                "color": r.color,
                "state": r.state,
                "dependency": r.dependency,
                "target_visible": r.target_visible,
                "red_confirmed": r.red_confirmed,
                "leader_distance": (
                    r.visible_leader["distance_cm"]
                    if r.visible_leader is not None
                    else 9999
                ),
            }
            for r in self.robots
        ]

    def step(self):
        detections = self.fetch_all()

        for robot, det in zip(self.robots, detections):
            robot.update_detection(det)

        for robot in self.robots:
            robot.decide_base_state()

        red_robot_states = [
            {
                "id": r.idx,
                "ip": r.ip,
                "color": r.color,
                "state": r.state
            }
            for r in self.robots
            if r.state in ["LEADER", "GOAL"]
        ]

        for robot in self.robots:
            if robot.state in ["LEADER", "GOAL", "VERIFYING_RED"]:
                continue

            visible = find_visible_red_robot(
                robot.last_detection, red_robot_states
            )

            if visible is not None:
                robot.state = "DEPENDENT"
                robot.dependency = visible["id"]
                robot.visible_leader = visible
            else:
                robot.state = "SEARCHING"
                robot.dependency = None
                robot.visible_leader = None

        for robot in self.robots:
            if robot.state == "DEPENDENT":
                has_follower = any(
                    other.dependency == robot.idx
                    for other in self.robots
                    if other.idx != robot.idx
                )

                if has_follower:
                    robot.state = "CHAIN_LEADER"

        states = self.build_states()
        ready = all_robots_ready(states)

        print("\n--- STATES ---")
        print(
            f"Red seen by: {[r.color for r in self.robots if r.state in ['LEADER', 'GOAL']]}"
        )
        print(f"All robots ready: {ready}")

        for s in states:
            print(
                f"Bot{s['id']} {s['color']}: "
                f"{s['state']} dep={s['dependency']} "
                f"red={s['target_visible']} confirmed={s['red_confirmed']}"
            )

        commands = []

        for robot in self.robots:
            left, right = robot.command(states)
            commands.append((left, right))

        self.send_all(commands)

    def stop_all(self):
        threads = [
            threading.Thread(target=stop_robot, args=(robot.ip,))
            for robot in self.robots
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        print("All robots stopped.")

    def run(self):
        try:
            while True:
                t0 = time.time()
                self.step()
                elapsed = time.time() - t0
                time.sleep(max(0.0, INTERVAL - elapsed))

        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop_all()


def print_stream_urls():
    print("\n── Live stream URLs ──")
    for i, ip in enumerate(ROBOT_IPS):
        print(f"Bot{i} {colors[ip]}: {stream_url(ip)}")
    print("──────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple Goal-Seeking JetBot Controller"
    )
    parser.add_argument("--ping", action="store_true")
    args = parser.parse_args()

    print_stream_urls()

    if args.ping:
        print("[PING] Checking robots...")

        for i, ip in enumerate(ROBOT_IPS):
            det = get_detection(ip)

            if det:
                print(
                    f"Bot{i} {colors[ip]} OK "
                    f"target_visible={det.get('target') is not None}"
                )
            else:
                print(f"Bot{i} {colors[ip]} UNREACHABLE")

    else:
        controller = Controller(ROBOT_IPS)
        controller.run()