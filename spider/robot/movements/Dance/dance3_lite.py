# -*- coding: utf-8 -*-
# myCode.py - smooth dance for NumPy 2.0 (no np.mat, offline, no server)

import sys
import math
import time
import numpy as np
import os

SERVER = os.environ.get(
    "FREENOVE_SERVER",
    "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server",
)
os.chdir(SERVER)
# Add the Server directory to Python path
sys.path.append(SERVER)

from control import Control
from servo import Servo
c = Control()
s = Servo()

TILT, PAN = 1, 0          # head channels (swap if yours differ)

# Footpoint layout copied from control.py (same numbers it uses)
FOOTPOINTS = np.array([
    [137.1, 189.4, 0],
    [225,   0,     0],
    [137.1, -189.4, 0],
    [-137.1, -189.4, 0],
    [-225,  0,     0],
    [-137.1, 189.4, 0],
]).T   # shape (3, 6)

def posture(roll, pitch, yaw):
    """NumPy-2-safe version of calculate_posture_balance."""
    position = np.array([0.0, 0.0, c.body_height]).reshape(3, 1)
    r, p, y = np.radians([roll, pitch, yaw])
    rot_x = np.array([[1, 0, 0],
                      [0, math.cos(p), -math.sin(p)],
                      [0, math.sin(p),  math.cos(p)]])
    rot_y = np.array([[math.cos(r), 0, -math.sin(r)],
                      [0, 1, 0],
                      [math.sin(r), 0,  math.cos(r)]])
    rot_z = np.array([[math.cos(y), -math.sin(y), 0],
                      [math.sin(y),  math.cos(y), 0],
                      [0, 0, 1]])
    rotation = rot_x @ rot_y @ rot_z
    ab = position + rotation @ FOOTPOINTS        # (3, 6)
    foot = [[ab[0, i], ab[1, i], ab[2, i]] for i in range(6)]
    return foot

def set_attitude(roll, pitch, yaw):
    pts = posture(roll, pitch, yaw)
    c.transform_coordinates(pts)
    c.set_leg_angles()

# ---------- smooth helpers ----------
def smooth_head(channel, start, end, steps=25, dt=0.02):
    for i in range(steps + 1):
        s.set_servo_angle(channel, start + (end - start) * i / steps)
        time.sleep(dt)

def smooth_attitude(start, end, steps=30, dt=0.02):
    for i in range(steps + 1):
        r = start[0] + (end[0] - start[0]) * i / steps
        p = start[1] + (end[1] - start[1]) * i / steps
        y = start[2] + (end[2] - start[2]) * i / steps
        set_attitude(r, p, y)
        time.sleep(dt)

def smooth_position(start, end, steps=30, dt=0.02):
    for i in range(steps + 1):
        x = start[0] + (end[0] - start[0]) * i / steps
        y = start[1] + (end[1] - start[1]) * i / steps
        z = start[2] + (end[2] - start[2]) * i / steps
        c.move_position(x, y, z)
        time.sleep(dt)

def head_center():
    s.set_servo_angle(PAN, 90)
    s.set_servo_angle(TILT, 90)

# ---------- routine ----------
print("Smooth dance (NumPy 2 safe). Ctrl+C to stop.")
head_center()
set_attitude(0, 0, 0)
time.sleep(0.5)


try:
    # 1) Roll side to side, head follows
    smooth_attitude((0,0,0), (12,0,0));   smooth_head(PAN, 90, 110)
    smooth_attitude((12,0,0), (-12,0,0)); smooth_head(PAN, 110, 70)
    smooth_attitude((-12,0,0), (0,0,0));  smooth_head(PAN, 70, 90)

    # 2) Pitch dip forward/back, head tilts with it
    smooth_attitude((0,0,0), (0,12,0));   smooth_head(TILT, 90, 75)
    smooth_attitude((0,12,0), (0,-12,0)); smooth_head(TILT, 75, 105)
    smooth_attitude((0,-12,0), (0,0,0));  smooth_head(TILT, 105, 90)

    # 3) Yaw twist in place
    smooth_attitude((0,0,0), (0,0,14))
    smooth_attitude((0,0,14), (0,0,-14))
    smooth_attitude((0,0,-14), (0,0,0))

    # 4) Slow roll-around (combine roll + pitch)
    smooth_attitude((0,0,0),   (12,0,0))
    smooth_attitude((12,0,0),  (0,12,0))
    smooth_attitude((0,12,0),  (-12,0,0))
    smooth_attitude((-12,0,0), (0,-12,0))
    smooth_attitude((0,-12,0), (0,0,0))

    # 5) Glide side to side (no stepping), head leads
    smooth_head(PAN, 90, 65);  smooth_position((0,0,0), (25,0,0))
    smooth_head(PAN, 65, 115); smooth_position((25,0,0), (-25,0,0))
    smooth_head(PAN, 115, 90); smooth_position((-25,0,0), (0,0,0))

    # 6) Gentle up/down bob with a nod
    smooth_position((0,0,0), (0,0,12));   smooth_head(TILT, 90, 80)
    smooth_position((0,0,12), (0,0,-12)); smooth_head(TILT, 80, 100)
    smooth_position((0,0,-12), (0,0,0));  smooth_head(TILT, 100, 90)

    set_attitude(0, 0, 0)
    head_center()
    print("Done!")

except KeyboardInterrupt:
    head_center()
    set_attitude(0, 0, 0)
    print("\nStopped.")
