# -*- coding: utf-8 -*-
# rest.py - the hexapod does a slow "stretch", then settles down and relaxes
# (powers off) its leg servos.
#
# "rest" / "sit" / "relax" command: it stretches like it's waking up, lowers
# the body, then cuts servo power to save the servos and battery.
#
#   Test it:   python3 rest.py
#   To get up again, just run any movement (forward.py, etc.) - it re-engages
#   the servos automatically.

import os
import sys
import time

# --- Where the Freenove server code lives ---
SERVER = "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server"

# --- Stretch shape (degrees of body tilt). Keep small (<= ~12).
#     If the robot prints "out of active range", lower these. ---
PITCH  = 10       # lean nose-down then tail-down (forward/back stretch)
ROLL   = 10       # lean left then right (side stretch)
RAISE  = -10      # how tall to stand at the peak (negative z = up)
CROUCH = 12       # how far to sink before relaxing (positive z = down)

# --- Speed (bigger STEPS + bigger PAUSE = slower, smoother) ---
STEPS = 30        # interpolation steps per move
PAUSE = 0.04      # seconds between steps  (40 * 0.04 ~= 1.6s per move)
HOLD  = 0.4       # pause held at each stretched pose


def get_controller():
    """Start the Freenove Control object (does its own calibration + stance)."""
    os.chdir(SERVER)
    sys.path.append(SERVER)
    from control import Control
    c = Control()
    time.sleep(0.5)        # let it settle into its standing stance
    return c


def _posture(c, roll, pitch, yaw):
    """Apply one body tilt (degrees)."""
    points = c.calculate_posture_balance(roll, pitch, yaw)
    c.transform_coordinates(points)
    c.set_leg_angles()


def _glide_posture(c, start, end):
    """Smoothly tilt from one (roll, pitch, yaw) to another."""
    for i in range(STEPS + 1):
        t = i / STEPS
        r = start[0] + (end[0] - start[0]) * t
        p = start[1] + (end[1] - start[1]) * t
        y = start[2] + (end[2] - start[2]) * t
        _posture(c, r, p, y)
        time.sleep(PAUSE)


def _glide_height(c, z0, z1):
    """Smoothly raise/lower the body (negative z = up, positive z = down)."""
    for i in range(STEPS + 1):
        t = i / STEPS
        c.move_position(0, 0, z0 + (z1 - z0) * t)
        time.sleep(PAUSE)


def stretch(c):
    """A slow wake-up stretch, then sink down and power off the servos."""
    level = (0, 0, 0)

    print("  stretching forward/back...")
    _glide_posture(c, level, (0, PITCH, 0));            time.sleep(HOLD)
    _glide_posture(c, (0, PITCH, 0), (0, -PITCH, 0));   time.sleep(HOLD)
    _glide_posture(c, (0, -PITCH, 0), level)

    print("  stretching side to side...")
    _glide_posture(c, level, (ROLL, 0, 0));             time.sleep(HOLD)
    _glide_posture(c, (ROLL, 0, 0), (-ROLL, 0, 0));     time.sleep(HOLD)
    _glide_posture(c, (-ROLL, 0, 0), level)

    print("  reaching up tall...")
    _glide_height(c, 0, RAISE);                         time.sleep(HOLD)

    print("  sinking down...")
    _glide_height(c, RAISE, CROUCH)
    time.sleep(0.3)

    print("  relaxing servos")
    c.relax(True)          # power down the leg servos


if __name__ == "__main__":
    print("Resting...")
    c = get_controller()
    try:
        stretch(c)
        print("Done. (run any movement to stand back up)")
    except KeyboardInterrupt:
        c.relax(True)
        print("\nStopped.")
