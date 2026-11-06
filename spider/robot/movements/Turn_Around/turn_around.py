# -*- coding: utf-8 -*-
# turn_around.py - rotate the Freenove hexapod in place (default ~180 degrees).
#
#   Test it:   python3 turn_around.py            # turn 180
#              python3 turn_around.py 90          # turn 90
#              python3 turn_around.py 180 right   # 180 to the right
import os
import sys
import time

GAIT  = "1"          # "1" tripod (faster), "2" wave (steadier)
SPEED = "8"          # 2 (slow) .. 10 (fast)
TURN_ANGLE = 13          # leg swing per stride (physical; keep ~8..15)
DEGREES_PER_STRIDE = 60  # how much the BODY actually rotates per stride (measured)
TURN_SIGN = 1
DEFAULT_DEGREES = 260    # now this means real degrees again
SERVER = os.environ.get(
    "FREENOVE_SERVER",
    "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server",
)


def setup():
    os.chdir(SERVER)
    sys.path.append(SERVER)
    from control import Control
    c = Control()
    time.sleep(0.5)        # let it settle into its standing stance
    return c


def turn(c, degrees, sign):
    strides = max(1, int(round(degrees / DEGREES_PER_STRIDE)))
    angle = sign * TURN_ANGLE
    for i in range(strides):
        print(f"  turn stride {i + 1}/{strides}")
        c.run_gait(["CMD_MOVE", GAIT, "0", "0", SPEED, str(angle)])
    c.run_gait(["CMD_MOVE", GAIT, "0", "0", SPEED, "0"])


def parse_args(argv):
    degrees, sign = DEFAULT_DEGREES, TURN_SIGN
    for a in argv:
        al = a.lower()
        if al == "left":
            sign = TURN_SIGN
        elif al == "right":
            sign = -TURN_SIGN
        else:
            try:
                degrees = abs(float(a))
            except ValueError:
                pass
    return degrees, sign


if __name__ == "__main__":
    degrees, sign = parse_args(sys.argv[1:])
    print(f"Turning {degrees:g} degrees...")
    c = setup()
    try:
        turn(c, degrees, sign)
        print("Done.")
    except KeyboardInterrupt:
        c.run_gait(["CMD_MOVE", GAIT, "0", "0", SPEED, "0"])
        print("\nStopped.")
