# -*- coding: utf-8 -*-
# hey.py - quick head nod to acknowledge the "hey spider" wake word.
# Run on its own to test:  python3 hey.py
# Or import it:            from hey import nod ; nod()

import os
import sys
import time

# Same setup your dance code uses, so the servos behave identically.
os.chdir("/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server")
sys.path.append("/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server")

from servo import Servo

s = Servo()

# --- Head config (same channels as your dance code) ---
PAN, TILT = 1, 0     # PAN = left/right, TILT = up/down
CENTER = 90          # neutral angle for the head servos

# Smaller TILT angle = head up, larger = head down (matches your dance code).
# Keep these close to CENTER for a *slight* nod; widen them for a bigger one.
UP = 80              # how far up the nod goes
DOWN = 100           # how far down the nod goes


def smooth_head(channel, start, end, steps=30, dt=0.005):
    for i in range(steps + 1):
        angle = start + (end - start) * i / steps
        s.set_servo_angle(channel, angle)
        time.sleep(dt)

    """Nod the head slightly up and down `cycles` times, then re-center.
    Chained so it never jumps between angles."""
def nod(cycles=2):
    smooth_head(TILT, CENTER, UP)

    for _ in range(cycles):
        smooth_head(TILT, UP, DOWN)
        smooth_head(TILT, DOWN, UP)

    smooth_head(TILT, UP, CENTER)


if __name__ == "__main__":
    try:
        nod()
    except KeyboardInterrupt:
        pass
    finally:
        # Always leave the head looking straight ahead.
        s.set_servo_angle(TILT, CENTER)
