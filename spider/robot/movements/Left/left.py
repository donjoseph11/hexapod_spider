# -*- coding: utf-8 -*-
# left.py - turn the Freenove hexapod left, in place.
#
#   Test it:   python3 left.py            # default number of turn strides
#              python3 left.py 8           # turn for 8 strides
#   Also tolerates: python3 left.py left 90 degrees  (grabs the first number)

import os
import sys
import time

# --- Settings ---
GAIT  = "1"     # "1" = tripod (faster), "2" = wave (slower, steadier)
SPEED = "8"     # 2 (slow) .. 10 (fast)
TURN  = 8     # turn amount per stride
DEFAULT_STRIDES = 5
TURN_LEFT_SIGN = 1   # if this turns the robot RIGHT, change to -1

SERVER = os.environ.get(
    "FREENOVE_SERVER",
    "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server",
)


def get_controller():
    """Start the Freenove Control object (does its own calibration + stance)."""
    os.chdir(SERVER)
    sys.path.append(SERVER)
    from control import Control
    c = Control()
    time.sleep(0.5)        # let it settle into its standing stance
    return c


def turn_left(c, strides):
    """Turn left for `strides` gait cycles, then settle back to a stance."""
    angle = TURN_LEFT_SIGN * TURN
    for i in range(strides):
        print(f"  stride {i + 1}/{strides}")
        c.run_gait(["CMD_MOVE", GAIT, "0", "0", SPEED, str(angle)])
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
    print(f"Turning left for {strides} strides...")
    c = get_controller()
    try:
        turn_left(c, strides)
        print("Done.")
    except KeyboardInterrupt:
        c.run_gait(["CMD_MOVE", GAIT, "0", "0", SPEED, "0"])   # settle on stop
        print("\nStopped.")
