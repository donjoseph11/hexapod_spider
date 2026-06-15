# -*- coding: utf-8 -*-
# backward.py - walk the Freenove hexapod straight backward.
#
#   Test it:   python3 backward.py          # default number of strides
#              python3 backward.py 8         # walk 8 strides back
#   Also tolerates: python3 backward.py backward 8 feet  (grabs the first number)

import os
import sys
import time

# --- Settings ---
GAIT  = "1"     # "1" = tripod (faster), "2" = wave (slower, steadier)
SPEED = "8"     # 2 (slow) .. 10 (fast)
STRIDE = 35     # amount per stride (max 35)
DEFAULT_STRIDES = 3
FORWARD_SIGN = 1   # keep this the SAME as in forward.py (set to -1 if forward was reversed)

SERVER = "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server"


def get_controller():
    """Start the Freenove Control object (does its own calibration + stance)."""
    os.chdir(SERVER)
    sys.path.append(SERVER)
    from control import Control
    c = Control()
    time.sleep(0.5)        # let it settle into its standing stance
    return c


def backward(c, strides):
    """Take `strides` backward gait cycles, then settle back to a stance."""
    y = -FORWARD_SIGN * STRIDE        # opposite of forward
    for i in range(strides):
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
    print(f"Walking backward {strides} strides...")
    c = get_controller()
    try:
        backward(c, strides)
        print("Done.")
    except KeyboardInterrupt:
        c.run_gait(["CMD_MOVE", GAIT, "0", "0", SPEED, "0"])   # settle on stop
        print("\nStopped.")
