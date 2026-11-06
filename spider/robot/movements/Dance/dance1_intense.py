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

# ---- head channels: 1 = tilt, 0 = pan (swap if yours differ) ----
TILT, PAN = 1, 0

def move(gait, x, y, speed, angle, reps=1):
    for _ in range(reps):
        c.run_gait(['CMD_MOVE', str(gait), str(x), str(y), str(speed), str(angle)])

def head(channel, angle):
    s.set_servo_angle(channel, angle)

def head_center():
    head(PAN, 90); head(TILT, 90)

def beat(t=0.25):
    time.sleep(t)

print("Dancing! Ctrl+C to stop.")
head_center()
time.sleep(0.5)

try:
    # 1) Head bob intro  nod yes 3 times
    for _ in range(3):
        head(TILT, 70); beat()
        head(TILT, 110); beat()
    head(TILT, 90)

    # 2) Head shakelook side to side
    for _ in range(3):
        head(PAN, 60); beat()
        head(PAN, 120); beat()
    head(PAN, 90)

    # 3) Spin left (turn in place via angle field)
    move(gait=1, x=0, y=0, speed=8, angle=30, reps=4)

    # 4) Spin right
    move(gait=1, x=0, y=0, speed=8, angle=-30, reps=4)

    # 5) Side shuffle  strafe right then left (x axis)
    move(gait=1, x=30, y=0, speed=8, angle=0, reps=2)
    move(gait=1, x=-30, y=0, speed=8, angle=0, reps=2)

    # 6) "The wiggle"  quick forward/back hops while head bobs
    for _ in range(2):
        head(TILT, 75)
        move(gait=1, x=0, y=25, speed=10, angle=0, reps=1)
        head(TILT, 105)
        move(gait=1, x=0, y=-25, speed=10, angle=0, reps=1)
    head(TILT, 90)

    # 7) Slow groove  ripple gait spin with head pan
    head(PAN, 60)
    move(gait=2, x=0, y=0, speed=4, angle=20, reps=1)
    head(PAN, 120)
    move(gait=2, x=0, y=0, speed=4, angle=-20, reps=1)
    head(PAN, 90)

    # 8) Big finish fast spin + head nod
    for _ in range(2):
        move(gait=1, x=0, y=0, speed=10, angle=35, reps=2)
        head(TILT, 70); beat(0.15)
        head(TILT, 110); beat(0.15)

    # Settle
    head_center()
    move(gait=1, x=0, y=0, speed=8, angle=0, reps=1)  # neutral stance
    print("Done!")

except KeyboardInterrupt:
    head_center()
    c.run_gait(['CMD_MOVE', '1', '0', '0', '8', '0'])  # stop/neutral
    print("\nStopped.")
