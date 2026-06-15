# -*- coding: utf-8 -*-
# right.py - turn the Freenove hexapod right, in place.
#
#   Test it:   python3 right.py           # default number of turn strides
#              python3 right.py 8          # turn for 8 strides
#   Also tolerates: python3 right.py right 90 degrees  (grabs the first number)

import os
import sys
import time

# --- Settings ---
GAIT  = "1"     # "1" = tripod (faster), "2" = wave (slower, steadier)
SPEED = "8"     # 2 (slow) .. 10 (fast)
TURN  = 8      # turn amount per stride
DEFAULT_STRIDES = 5
TURN_LEFT_SIGN = 1   # keep this the SAME as in left.py (set to -1 if left turned the wrong way)

SERVER = "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server"


def get_controller():
    """Start the Freenove Control object (does its own calibration + stance)."""
    os.chdir(SERVER)
    sys.path.append(SERVER)
    from control import Control
    c = Control()
    time.sleep(0.5)        # let it settle into its standing stance
    return c


def turn_right(c, strides):
    """Turn right for `strides` gait cycles, then settle back to a stance."""
    angle = -TURN_LEFT_SIGN * TURN        # opposite of left
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
    print(f"Turning right for {strides} strides...")
    c = get_controller()
    try:
        turn_right(c, strides)
        print("Done.")
    except KeyboardInterrupt:
        c.run_gait(["CMD_MOVE", GAIT, "0", "0", SPEED, "0"])   # settle on stop
        print("\nStopped.")
