"""
run_all.py - one command to run the whole benchmark, unattended.

WHAT IT DOES
    1. Unpacks every speaker .zip under the audio root (cached - reruns are free).
    2. Checks every model path exists and reports sizes before committing hours.
    3. Sets the CPU governor to 'performance' if it can (frequency scaling
       otherwise shows up as variance you'd misread as a model difference).
    4. Runs a dry-run first and refuses to start if files can't be identified.
    5. Runs the sweep, restarting automatically if it crashes. The benchmark's
       append-only log means a restart resumes instead of starting over.
    6. Tees everything to a timestamped log next to the results.

WHY A LOOP
    A 600-run sweep on a Pi takes hours. Thermal shutdowns, SD hiccups and OOM
    kills happen. Without auto-restart you come back to a dead terminal and an
    afternoon lost; with it you come back to a finished workbook.

USAGE
    python3 tools/benchmark/run_all.py --audio-root ./recordings --out ./results

    # laptop side - same data, different tag, so the two concatenate cleanly
    python3 tools/benchmark/run_all.py --audio-root ./recordings --out ./results --tag laptop

    # check everything without transcribing
    python3 tools/benchmark/run_all.py --audio-root ./recordings --out ./results --check-only

LAYOUT IT EXPECTS
    audio/
        speakers.csv                        optional
        male/male_1_recordings.zip          -> male_1/cmd01_take1.wav ...
        female/female_1_recordings.zip      -> female_1/cmd01_take1.wav ...
"""

import argparse
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "stt_benchmark.py")


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(path) for f in fs)


def check_models():
    """Import the benchmark's MODELS list and report what's actually present."""
    sys.path.insert(0, HERE)
    import stt_benchmark as B

    present, missing = [], []
    for m in B.MODELS:
        if os.path.exists(m["path"]):
            present.append((m["name"], human_size(dir_size(m["path"]))))
        else:
            missing.append((m["name"], m["path"]))

    print("models:")
    for name, size in present:
        print(f"    OK      {name:20s} {size}")
    for name, path in missing:
        print(f"    MISSING {name:20s} {path}")
    return [n for n, _ in present], missing


def set_governor():
    """Best-effort 'performance' governor. Needs root; warns rather than fails."""
    if platform.system() != "Linux":
        return
    gov_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if not os.path.exists(gov_path):
        return
    with open(gov_path) as f:
        current = f.read().strip()
    if current == "performance":
        print("cpu governor: performance")
        return
    print(f"cpu governor: {current} - trying to switch to performance")
    try:
        subprocess.run(
            "for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; "
            "do echo performance | sudo tee $c >/dev/null; done",
            shell=True, check=True, timeout=15)
        with open(gov_path) as f:
            print(f"cpu governor: now {f.read().strip()}")
    except Exception:
        print("!! could not set governor (needs sudo). Results will carry extra\n"
              "   timing variance from frequency scaling - note it in your writeup.")


def run_bench(extra_args, log_file):
    """Run the benchmark once, streaming output to console and log."""
    cmd = [sys.executable, "-u", BENCH] + extra_args
    print(f"\n$ {' '.join(cmd)}\n")
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 70}\n{time.strftime('%Y-%m-%d %H:%M:%S')}  "
                  f"{' '.join(cmd)}\n{'=' * 70}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--tag", default=None,
                    help="label for outputs (default: pi5 or laptop by hostname)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats on the --repeat-cmds subset (default 1)")
    ap.add_argument("--repeat-cmds", default="",
                    help="commands that get repeats; '' means every command")
    ap.add_argument("--rest", type=float, default=30.0,
                    help="grace period before the ceiling counts as good enough")
    ap.add_argument("--soft-cool", action="store_true",
                    help="old behaviour: run regardless after --rest")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure idle temp floor before starting")
    ap.add_argument("--cool-to", type=float, default=60.0,
                    help="ideal temp - run immediately at or below this")
    ap.add_argument("--cool-ceiling", type=float, default=65.0,
                    help="hard limit - never run above this")
    ap.add_argument("--cool-timeout", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--only", nargs="*", help="restrict to these model names")
    ap.add_argument("--speakers", nargs="*", help="restrict to these speaker ids")
    ap.add_argument("--match", default="auto", choices=["auto", "index", "text"],
                    help="filename -> command mapping (default auto)")
    ap.add_argument("--cmd-regex", default=None,
                    help="override the command-number regex")
    ap.add_argument("--text-threshold", type=float, default=None)
    ap.add_argument("--max-restarts", type=int, default=5)
    ap.add_argument("--check-only", action="store_true",
                    help="verify setup and print the plan, then stop")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    tag = args.tag or ("pi5" if "arm" in platform.machine().lower()
                       or "aarch64" in platform.machine().lower() else "laptop")
    log_file = os.path.join(out, f"{tag}_run.log")

    print("=" * 70)
    print(f"  STT benchmark runner")
    print(f"  host      {platform.node()}  ({platform.machine()})")
    print(f"  tag       {tag}")
    print(f"  audio     {os.path.abspath(args.audio_root)}")
    print(f"  results   {out}")
    print(f"  log       {log_file}")
    print("=" * 70 + "\n")

    if not os.path.exists(BENCH):
        sys.exit(f"cannot find stt_benchmark.py next to this script ({BENCH})")

    names, missing = check_models()
    if not names:
        sys.exit("\nNo models found. Fix the paths in stt_benchmark.py CONFIG.")
    if missing:
        print(f"\n!! {len(missing)} model(s) missing - continuing with "
              f"{len(names)}. Download them first if you want the full sweep.")

    base = ["--audio-root", args.audio_root, "--out", out, "--tag", tag,
            "--repeats", str(args.repeats), "--repeat-cmds", args.repeat_cmds,
            "--rest", str(args.rest), "--cool-to", str(args.cool_to),
            "--cool-ceiling", str(args.cool_ceiling),
            "--cool-timeout", str(args.cool_timeout),
            "--seed", str(args.seed)]
    base += ["--match", args.match]
    if args.soft_cool:
        base += ["--soft-cool"]
    if args.calibrate:
        base += ["--calibrate"]
    if args.cmd_regex:
        base += ["--cmd-regex", args.cmd_regex]
    if args.text_threshold is not None:
        base += ["--text-threshold", str(args.text_threshold)]
    if args.only:
        base += ["--only"] + args.only
    if args.speakers:
        base += ["--speakers"] + args.speakers

    # dry run first: unpacks archives, validates naming, prints the plan
    print("\n--- pre-flight ---")
    if run_bench(base + ["--dry-run"], log_file) != 0:
        sys.exit("\npre-flight failed - fix the errors above before running.")

    if args.check_only:
        print("\ncheck-only: stopping here. Drop --check-only to run for real.")
        return

    try:
        input("\nPress Enter to start the sweep (Ctrl-C to abort)... ")
    except (EOFError, KeyboardInterrupt):
        print("\naborted")
        return

    set_governor()

    started = time.time()
    for attempt in range(1, args.max_restarts + 1):
        rc = run_bench(base, log_file)
        if rc == 0:
            break
        print(f"\n!! benchmark exited with code {rc} "
              f"(attempt {attempt}/{args.max_restarts})")
        if attempt < args.max_restarts:
            print("   resuming in 30s - completed runs are already saved")
            time.sleep(30)
    else:
        print("\n!! gave up after repeated failures. Check the log:", log_file)
        sys.exit(1)

    mins = (time.time() - started) / 60
    print(f"\n{'=' * 70}")
    print(f"  done in {mins:.0f} min ({mins / 60:.1f} h)")
    print(f"  workbook + raw csv: {out}")
    print(f"  log:                {log_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
