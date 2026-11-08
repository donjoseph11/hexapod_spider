# -*- coding: utf-8 -*-
# hi.py - the hexapod says hello: lifts a front leg, waves it, and chirps.
#
#   Test it:   python3 hi.py
import os
import sys
import time
import copy

SERVER = os.environ.get(
    "FREENOVE_SERVER",
    "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server",
)

WAVE_LEG = 0       # front leg to wave (0 and 5 are the two front legs)
LIFT_Z   = 45      # foot height while waving (body frame). Lower if out of range.
REACH_Y  = 230     # how far forward the lifted foot reaches
SWING    = 60      # how far the foot swings side to side
WAVES    = 3       # number of back-and-forth waves
STEPS    = 15      # smoothness of each swing
PAUSE    = 0.03


def setup():
    os.chdir(SERVER)
    sys.path.append(SERVER)
    from control import Control
    c = Control()
    time.sleep(0.5)
    try:
        from buzzer import Buzzer
        b = Buzzer()
    except Exception:
        b = None            # no buzzer? just skip the chirp
    return c, b


def _apply(c, foot):
    """Command one foot position (body frame) for WAVE_LEG; others stay planted."""
    points = copy.deepcopy(c.body_points)     # standing positions for all 6 legs
    points[WAVE_LEG] = foot
    c.transform_coordinates(points)
    c.set_leg_angles()


def _glide_foot(c, start, end):
    for i in range(STEPS + 1):
        t = i / STEPS
        foot = [start[k] + (end[k] - start[k]) * t for k in range(3)]
        _apply(c, foot)
        time.sleep(PAUSE)


def beep(b, on, off, times=1):
    if b is None:
        return
    for _ in range(times):
        b.set_state(True);  time.sleep(on)
        b.set_state(False); time.sleep(off)


def hello(c, b):
    home = list(c.body_points[WAVE_LEG])      # standing foot, e.g. [137.1, 189.4, -25]
    base_x = home[0]
    up     = [base_x, REACH_Y, LIFT_Z]
    left   = [base_x - SWING, REACH_Y, LIFT_Z]
    right  = [base_x + SWING, REACH_Y, LIFT_Z]

    beep(b, 0.08, 0.08, times=2)              # chirp hello
    _glide_foot(c, home, up)                  # raise the leg
    time.sleep(0.2)

    pos = up                                  # wave it
    for _ in range(WAVES):
        _glide_foot(c, pos, left);  pos = left
        _glide_foot(c, pos, right); pos = right
    _glide_foot(c, pos, up)

    _glide_foot(c, up, home)                  # put the leg back down
    c.move_position(0, 0, 0)                  # settle level


if __name__ == "__main__":
    print("Saying hi...")
    c, b = setup()
    try:
        hello(c, b)
        print("Done.")
    except KeyboardInterrupt:
        c.move_position(0, 0, 0)
        print("\nStopped.")
    finally:
        if b is not None:
            b.set_state(False)
            b.close()
