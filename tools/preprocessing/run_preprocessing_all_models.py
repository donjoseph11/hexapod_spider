#!/usr/bin/env python3
"""
run_preprocessing_two_full_sweeps.py

Runs TWO complete 6-model benchmark sweeps unattended:

  1) amplitude_adaptive
     600 recordings x 6 models = 3600 runs
     -> ONE raw CSV
     -> ONE Excel workbook

  2) denoise100_mix020
     600 recordings x 6 models = 3600 runs
     -> ONE raw CSV
     -> ONE Excel workbook

The existing run_all.py calls stt_benchmark.py.
No --only argument is used, so stt_benchmark.py runs ALL 6 configured models.

The Enter prompt in run_all.py is answered automatically.

Defaults:
    cool-to       = 63 C
    cool-ceiling  = 67 C
    rest          = 30 s
    cool-timeout  = 1800 s
    seed          = 1337
    max-restarts  = 5

Usage:
    python3 run_preprocessing_two_full_sweeps.py

Pre-flight only:
    python3 run_preprocessing_two_full_sweeps.py --check-only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def count_wavs(path: Path) -> int:
    """Count WAV files recursively."""
    return sum(
        1
        for p in path.rglob("*.wav")
        if p.is_file()
    )


def run_condition(
    run_all: Path,
    audio_root: Path,
    out_dir: Path,
    tag: str,
    common: list[str],
    check_only: bool,
) -> int:
    """
    Run one FULL six-model sweep.

    IMPORTANT:
    There is intentionally NO --only argument here.
    That makes run_all.py / stt_benchmark.py use all models
    configured in stt_benchmark.py.
    """

    cmd = [
        sys.executable,
        str(run_all),

        "--audio-root",
        str(audio_root),

        "--out",
        str(out_dir),

        "--tag",
        tag,

        *common,
    ]

    if check_only:
        cmd.append("--check-only")

    print()
    print("=" * 82)
    print(f"FULL SWEEP TAG : {tag}")
    print(f"AUDIO ROOT     : {audio_root}")
    print(f"OUTPUT         : {out_dir}")
    print("=" * 82)
    print("$ " + " ".join(cmd))
    print()

    if check_only:
        return subprocess.run(cmd).returncode

    # run_all.py asks:
    #     Press Enter to start the sweep...
    #
    # Automatically answer that prompt.
    return subprocess.run(
        cmd,
        input="\n",
        text=True,
    ).returncode


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Run one complete six-model sweep for amplitude preprocessing "
            "and one complete six-model sweep for denoise preprocessing."
        )
    )

    ap.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root (default: detected from this script location).",
    )

    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Results directory. Default: "
            "<project-root>/results_preprocessing_full_sweeps"
        ),
    )

    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Run pre-flight checks only.",
    )

    ap.add_argument(
        "--rest",
        type=float,
        default=30.0,
    )

    ap.add_argument(
        "--cool-to",
        type=float,
        default=63.0,
    )

    ap.add_argument(
        "--cool-ceiling",
        type=float,
        default=67.0,
    )

    ap.add_argument(
        "--cool-timeout",
        type=float,
        default=1800.0,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=1337,
    )

    ap.add_argument(
        "--max-restarts",
        type=int,
        default=5,
    )

    args = ap.parse_args()

    project = (
        Path(args.project_root)
        .expanduser()
        .resolve()
    )

    run_all = project / "tools" / "benchmark" / "run_all.py"

    if not run_all.exists():
        sys.exit(
            f"run_all.py not found: {run_all}"
        )

    variants = project / "variants"

    amplitude_root = (
        variants / "amplitude_adaptive"
    )

    denoise_root = (
        variants / "denoise100_mix020"
    )

    if args.out:
        out_dir = (
            Path(args.out)
            .expanduser()
            .resolve()
        )
    else:
        out_dir = (
            project
            / "results_preprocessing_full_sweeps"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validate both datasets
    # --------------------------------------------------------

    conditions = [
        (
            "AMPLITUDE",
            amplitude_root,
            "pi5_amplitude_adaptive",
        ),
        (
            "DENOISE",
            denoise_root,
            "pi5_denoise100_mix020",
        ),
    ]

    print()
    print("=" * 82)
    print("PREPROCESSING BENCHMARK: TWO COMPLETE 6-MODEL SWEEPS")
    print("=" * 82)
    print(f"Project      : {project}")
    print(f"Results      : {out_dir}")
    print(f"Cool to      : {args.cool_to} C")
    print(f"Cool ceiling : {args.cool_ceiling} C")
    print(f"Seed         : {args.seed}")
    print("=" * 82)

    for name, audio_root, tag in conditions:

        if not audio_root.exists():
            sys.exit(
                f"{name} dataset not found: {audio_root}"
            )

        wav_count = count_wavs(
            audio_root
        )

        print(
            f"{name:10s}: "
            f"{wav_count} WAV files | "
            f"{audio_root}"
        )

        if wav_count != 600:

            print(
                f"WARNING: {name} expected 600 WAV files, "
                f"but found {wav_count}."
            )

    print()
    print(
        "Expected per condition: "
        "600 recordings x 6 models = 3600 runs"
    )

    print(
        "Expected total: "
        "2 x 3600 = 7200 runs"
    )

    # --------------------------------------------------------
    # Common run_all.py settings
    # --------------------------------------------------------

    common = [
        "--rest",
        str(args.rest),

        "--cool-to",
        str(args.cool_to),

        "--cool-ceiling",
        str(args.cool_ceiling),

        "--cool-timeout",
        str(args.cool_timeout),

        "--seed",
        str(args.seed),

        "--max-restarts",
        str(args.max_restarts),
    ]

    started = time.time()

    # --------------------------------------------------------
    # Run amplitude full sweep, then denoise full sweep
    # --------------------------------------------------------

    for index, (
        name,
        audio_root,
        tag,
    ) in enumerate(
        conditions,
        start=1,
    ):

        print()
        print("#" * 82)
        print(
            f"CONDITION {index}/2: {name}"
        )
        print(
            "Running ALL SIX models over the full dataset."
        )
        print("#" * 82)

        rc = run_condition(
            run_all=run_all,
            audio_root=audio_root,
            out_dir=out_dir,
            tag=tag,
            common=common,
            check_only=args.check_only,
        )

        if rc != 0:

            sys.exit(
                f"\n{name} full sweep failed "
                f"with exit code {rc}. "
                "The next condition was not started. "
                "Completed benchmark rows remain saved and "
                "the same command can be rerun to resume."
            )

        if args.check_only:
            print(
                f"{name}: pre-flight passed."
            )
        else:
            print(
                f"{name}: complete six-model sweep finished."
            )

    elapsed_min = (
        time.time() - started
    ) / 60.0

    print()
    print("=" * 82)

    if args.check_only:
        print(
            "BOTH FULL-SWEEP PRE-FLIGHT CHECKS PASSED"
        )
    else:
        print(
            "BOTH COMPLETE PREPROCESSING SWEEPS FINISHED"
        )

        print(
            f"Elapsed: {elapsed_min:.1f} min "
            f"({elapsed_min / 60.0:.1f} h)"
        )

        print(
            f"Results: {out_dir}"
        )

        print()
        print("Expected main files:")
        print(
            "  pi5_amplitude_adaptive_6m_raw.csv"
        )
        print(
            "  pi5_amplitude_adaptive_6m_<timestamp>.xlsx"
        )
        print(
            "  pi5_denoise100_mix020_6m_raw.csv"
        )
        print(
            "  pi5_denoise100_mix020_6m_<timestamp>.xlsx"
        )

    print("=" * 82)


if __name__ == "__main__":
    main()
