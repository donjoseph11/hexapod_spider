# -*- coding: utf-8 -*-
import os
import sys

SERVER = os.environ.get(
    "FREENOVE_SERVER",
    "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server",
)
os.chdir(SERVER)
sys.path.append(SERVER)

from control import Control
from servo import Servo
import time

c = Control()
s = Servo()

# head channels: 1 = tilt, 0 = pan (swap if yours differ)
TILT, PAN = 1, 0

def move(gait, x, y, speed, angle, reps=1):
    for _ in range(reps):
        c.run_gait(['CMD_MOVE', str(gait), str(x), str(y), str(speed), str(angle)])

def head(channel, angle, hold=0.4):
    s.set_servo_angle(channel, angle)
    time.sleep(hold)

def head_center():
    s.set_servo_angle(PAN, 90)
    s.set_servo_angle(TILT, 90)

print("Gentle dance. Ctrl+C to stop.")
head_center()
time.sleep(0.6)

try:
    # 1) Slow nod hello - small, soft
    for _ in range(2):
        head(TILT, 80)
        head(TILT, 100)
    head(TILT, 90)

    # 2) Gentle head sway side to side
    for _ in range(2):
        head(PAN, 75)
        head(PAN, 105)
    head(PAN, 90)

    # 3) Slow sway in place - small turn left, using the calm ripple gait
    move(gait=2, x=0, y=0, speed=4, angle=12, reps=1)
    head(PAN, 80)
    # then back the other way
    move(gait=2, x=0, y=0, speed=4, angle=-12, reps=1)
    head(PAN, 100)
    head(PAN, 90)

    # 4) Soft side step right, then left (small, slow)
    move(gait=2, x=18, y=0, speed=4, angle=0, reps=1)
    head(TILT, 85)
    move(gait=2, x=-18, y=0, speed=4, angle=0, reps=1)
    head(TILT, 95)
    head(TILT, 90)

    # 5) Easy little forward-and-back rock with a slow nod
    move(gait=2, x=0, y=15, speed=4, angle=0, reps=1)
    head(TILT, 80)
    move(gait=2, x=0, y=-15, speed=4, angle=0, reps=1)
    head(TILT, 100)
    head(TILT, 90)

    # 6) Final slow sway and settle
    head(PAN, 78)
    move(gait=2, x=0, y=0, speed=4, angle=10, reps=1)
    head(PAN, 102)
    move(gait=2, x=0, y=0, speed=4, angle=-10, reps=1)
    head_center()
    print("Done!")

except KeyboardInterrupt:
    head_center()
    c.run_gait(['CMD_MOVE', '1', '0', '0', '8', '0'])
    print("\nStopped.")
