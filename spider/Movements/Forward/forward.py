# -*- coding: utf-8 -*-
# forward.py - walk the Freenove hexapod forward, stopping early if the
# ultrasonic sensor sees an obstacle ahead.
#
#   Test it:   python3 forward.py          # default number of strides
#              python3 forward.py 8         # walk up to 8 strides
#   To just watch the sensor, run the kit's own:  python3 ultrasonic.py

import os
import sys
import time

# --- Settings ---
GAIT  = "1"     # "1" = tripod (faster), "2" = wave (slower, steadier)
SPEED = "8"     # 2 (slow) .. 10 (fast)
STRIDE = 35     # forward amount per stride (max 35)
DEFAULT_STRIDES = 3
FORWARD_SIGN = 1   # if this walks BACKWARD on your robot, change to -1

STOP_DISTANCE_CM = 30    # stop if something is closer than this (cm). Tune it.

SERVER = "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server"


def setup():
    """Start the Control object and the ultrasonic sensor."""
    os.chdir(SERVER)
    sys.path.append(SERVER)
    from control import Control
    from ultrasonic import Ultrasonic
    c = Control()
    time.sleep(0.5)        # let it settle into its standing stance
    u = Ultrasonic()
    return c, u


def obstacle_ahead(u):
    """Return the distance (cm) if something is too close, else None."""
    d = u.get_distance()              # cm, or None if the reading failed
    if d is not None and 0 < d < STOP_DISTANCE_CM:
        return d
    return None


def forward(c, u, strides):
    """Walk forward up to `strides`, stopping early if an obstacle appears."""
    y = FORWARD_SIGN * STRIDE
    for i in range(strides):
        blocked = obstacle_ahead(u)
        if blocked is not None:
            print(f"  obstacle {blocked} cm ahead -- stopping at stride {i}")
            break
        print(f"  stride {i + 1}/{strides}")
        c.run_gait(["CMD_MOVE", GAIT, "0", str(y), SPEED, "0"])
    c.run_gait(["CMD_MOVE", GAIT, "0", "0", SPEED, "0"])   # neutral stance


def strides_from_args(argv, default):
    """Pull the first number out of the command-line args."""
    for a in argv:
        try:
            return max(1, int(round(float(a))))
        except ValueError:
            continue
    return default


if __name__ == "__main__":
    strides = strides_from_args(sys.argv[1:], DEFAULT_STRIDES)
    print(f"Walking forward up to {strides} strides (stop < {STOP_DISTANCE_CM} cm)...")
    c, u = setup()
    try:
        forward(c, u, strides)
        print("Done.")
    except KeyboardInterrupt:
        c.run_gait(["CMD_MOVE", GAIT, "0", "0", SPEED, "0"])   # settle on stop
        print("\nStopped.")
    finally:
        u.close()
