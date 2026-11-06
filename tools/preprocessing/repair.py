"""
repair.py - find and regenerate clean_slow90 files that ffmpeg failed on.

THE BUG
    ffmpeg 6.1 hits 'Assertion best_input >= 0 failed' on some inputs when
    loudnorm sits mid-chain with other buffering filters. It is input
    dependent - about a quarter of the corpus here. Worse, ffmpeg often
    writes a PARTIAL file before dying, so the file exists on disk and a
    plain file count looks correct while the audio is truncated.

THE FIX
    Split the chain into two ffmpeg passes. Stage 1 does the spectral work
    and loudness; stage 2 does the time-stretch and final resample. Verified
    to produce byte-identical output to the single-chain version on inputs
    where the single chain does not crash.

WHAT IT DOES
    1. Verifies every clean_slow90 file against its raw counterpart by
       duration ratio (expect ~1.111). Anything missing, unreadable or
       off-ratio is marked bad.
    2. Regenerates the bad ones via the two-stage path.
    3. Re-verifies and rewrites the affected rows in preprocess_log.csv.

USAGE
    python3 tools/preprocessing/repair.py --variants ./variants
    python3 tools/preprocessing/repair.py --variants ./variants --check-only
"""

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
import time
import wave

TEMPO = 0.9
EXPECTED_RATIO = 1.0 / TEMPO          # 1.1111
RATIO_TOL = 0.02                      # WSOLA quantisation is ~0.4%

STAGE1 = "highpass=f=80,afftdn=nr=10:nf=-25,loudnorm=I=-23:TP=-2:LRA=7"
STAGE2 = (f"atempo={TEMPO},aresample=16000:resampler=soxr,"
          "aformat=sample_fmts=s16:channel_layouts=mono")


def wav_duration(path):
    """Duration via the wave module - also proves the file is readable and
    not truncated mid-frame, which ffprobe will happily overlook."""
    try:
        with wave.open(path, "rb") as w:
            if w.getnframes() == 0:
                return None
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def convert_two_stage(src, dst):
    """Two ffmpeg passes via a temp file. Returns (ok, seconds, error)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    t0 = time.perf_counter()
    try:
        r1 = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", src, "-af", STAGE1, "-ar", "48000", "-ac", "1",
             "-c:a", "pcm_s16le", tmp],
            capture_output=True, text=True, timeout=300)
        if r1.returncode != 0:
            return False, time.perf_counter() - t0, "stage1: " + r1.stderr.strip()[:150]

        r2 = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", tmp, "-af", STAGE2, "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", dst],
            capture_output=True, text=True, timeout=300)
        if r2.returncode != 0:
            return False, time.perf_counter() - t0, "stage2: " + r2.stderr.strip()[:150]

        return True, time.perf_counter() - t0, ""
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - t0, "timeout"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def audit(variants, cond):
    """Compare every condition file against its raw counterpart."""
    raw_root = os.path.join(variants, "raw")
    cond_root = os.path.join(variants, cond)

    good, bad = [], []
    for dirpath, _, files in os.walk(raw_root):
        for fn in sorted(files):
            if not fn.lower().endswith(".wav"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), raw_root)
            rawp = os.path.join(raw_root, rel)
            outp = os.path.join(cond_root, rel)

            d_raw = wav_duration(rawp)
            d_out = wav_duration(outp)

            if d_raw is None:
                bad.append((rel, "raw unreadable"))
            elif not os.path.exists(outp):
                bad.append((rel, "missing"))
            elif d_out is None:
                bad.append((rel, "unreadable/empty"))
            else:
                ratio = d_out / d_raw
                if abs(ratio - EXPECTED_RATIO) > RATIO_TOL:
                    bad.append((rel, f"ratio {ratio:.3f} (want {EXPECTED_RATIO:.3f}) "
                                     f"- likely truncated"))
                else:
                    good.append(rel)
    return good, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--condition", default="clean_slow90")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")

    print(f"auditing {args.condition} against raw ...")
    good, bad = audit(args.variants, args.condition)
    print(f"  ok      {len(good)}")
    print(f"  bad     {len(bad)}")
    for rel, why in bad[:5]:
        print(f"          {rel[:60]}  [{why}]")
    if len(bad) > 5:
        print(f"          ... and {len(bad)-5} more")

    if not bad:
        print("\nnothing to repair - all files verified.")
        return
    if args.check_only:
        print("\ncheck-only: stopping. Rerun without --check-only to repair.")
        return

    raw_root = os.path.join(args.variants, "raw")
    cond_root = os.path.join(args.variants, args.condition)

    print(f"\nregenerating {len(bad)} file(s) via two-stage chain ...")
    fixed, still_bad, times = [], [], []
    for i, (rel, _) in enumerate(bad, 1):
        src = os.path.join(raw_root, rel)
        dst = os.path.join(cond_root, rel)
        ok, secs, err = convert_two_stage(src, dst)
        if ok:
            d_raw, d_out = wav_duration(src), wav_duration(dst)
            ratio = (d_out / d_raw) if (d_raw and d_out) else 0
            if abs(ratio - EXPECTED_RATIO) <= RATIO_TOL:
                fixed.append((rel, secs, d_out))
                times.append(secs)
            else:
                still_bad.append((rel, f"ratio still {ratio:.3f}"))
        else:
            still_bad.append((rel, err))
        if i % 25 == 0 or i == len(bad):
            print(f"  [{i}/{len(bad)}]", flush=True)

    print(f"\nrepaired    {len(fixed)}")
    print(f"still bad   {len(still_bad)}")
    for rel, why in still_bad[:5]:
        print(f"            {rel[:60]}  [{why}]")

    # ---- update the log so cost stats cover all 600 -----------------------
    log_path = os.path.join(args.variants, "preprocess_log.csv")
    if fixed and os.path.exists(log_path):
        by_rel = {rel: (secs, dur) for rel, secs, dur in fixed}
        with open(log_path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            fields, rows = rd.fieldnames, list(rd)
        n = 0
        for row in rows:
            if row["condition"] == args.condition and row["rel_path"] in by_rel:
                secs, dur = by_rel[row["rel_path"]]
                row["ok"] = "1"
                row["error"] = "repaired: two-stage"
                row["proc_sec"] = round(secs, 4)
                row["proc_sec_net"] = round(max(secs - 0.123, 0.0), 4)
                row["dur_out"] = round(dur, 3)
                if row.get("dur_in"):
                    row["rtf_prep"] = round(float(row["proc_sec_net"]) / dur, 4)
                n += 1
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nupdated {n} row(s) in preprocess_log.csv")

    print("\nre-auditing ...")
    good2, bad2 = audit(args.variants, args.condition)
    print(f"  ok {len(good2)}   bad {len(bad2)}")
    if not bad2:
        print("\nALL FILES VERIFIED - safe to run the benchmark.")
    else:
        print("\n!! still failing - do not start the sweep, send this output.")


if __name__ == "__main__":
    main()
