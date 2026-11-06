"""
preprocess.py - generate preprocessed variants of a speech corpus, one output
tree per condition, and log the CPU cost of producing each file.

WHY IT IS BUILT THIS WAY

    Each condition is written to its own directory tree that MIRRORS the input
    layout exactly:

        variants/raw/male_1/cmd01_take1.wav
        variants/slow90/male_1/cmd01_take1.wav

    That means stt_benchmark.py needs no changes at all. You point it at each
    tree in turn with --audio-root and --tag, and speaker_id / cmd_index / take
    keep parsing exactly as before. Condition becomes a column via the tag, and
    every comparison stays paired: same speaker, same utterance, same model.

    Preprocessing is done ONCE and cached, not inside the benchmark loop, so
    denoising cost never contaminates your inference timings. The cost is
    measured separately and deliberately (--cost-repeats), because on the Pi
    that cost is part of your latency budget and belongs in the results.

METRICS LOGGED PER FILE (preprocess_log.csv)
    proc_sec        wall time for this file's filter chain
    proc_sec_net    proc_sec minus measured ffmpeg startup+decode overhead
    rtf_prep        proc_sec_net / audio_sec  (Pi budget: must be << 1)
    snr_db_est      percentile-based SNR of the ORIGINAL recording
    rms_dbfs, peak_dbfs, clip_pct
    dur_in, dur_out (out != in for time-stretch conditions)

USAGE
    python3 tools/preprocessing/preprocess.py --audio-root ./recordings --out ./variants --list
    python3 tools/preprocessing/preprocess.py --audio-root ./recordings --out ./variants \
        --conditions raw hp80 denoise_light slow90 --cost-repeats 3

    Run with the SAME conditions on the Pi and the laptop if you want to
    report preprocessing cost on both. The audio output is identical either
    way (ffmpeg is deterministic here), so quality results transfer.
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import wave

import numpy as np

AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".opus", ".ogg", ".flac", ".aac", ".3gp", ".amr", ".webm"}

# ----------------------------------------------------------------------------
# Conditions. Each is an ffmpeg filter chain, applied BEFORE the final
# resample to 16 kHz mono s16 (which every model needs anyway).
#
# Filtering before downsampling matters: highpass/denoise work on the full
# original bandwidth, which is what you want. atempo is pitch-preserving
# (WSOLA-style) - never use asetrate for slowing, it shifts formants and will
# destroy accented-speech accuracy.
# ----------------------------------------------------------------------------
CONDITIONS = {
    # --- baseline: format conversion only, no enhancement -------------------
    "raw": [],

    # --- frequency shaping --------------------------------------------------
    # 80 Hz is below the male fundamental (~85-180 Hz), so this removes handling
    # rumble, AC hum and desk thumps without touching voiced speech.
    "hp80": ["highpass=f=80"],
    # Telephony-ish band limit. Speech intelligibility lives in 300-3400 Hz, but
    # fricatives (/s/, /f/) extend to 8 kHz - expect this to hurt. Include it as
    # a negative control that shows the models DO use high-frequency cues.
    "band300_3400": ["highpass=f=300", "lowpass=f=3400"],
    # The honest middle: kill rumble, keep everything the 16 kHz mel bank sees.
    "hp80_lp7800": ["highpass=f=80", "lowpass=f=7800"],

    # --- level normalisation ------------------------------------------------
    # EBU R128 to -23 LUFS. Two-pass would be more accurate; single pass is
    # fine here and much cheaper on the Pi. Report which you used.
    "loudnorm": ["loudnorm=I=-23:TP=-2:LRA=7"],

    # --- noise reduction ----------------------------------------------------
    # afftdn = FFT spectral gating. nr is reduction in dB, nf the noise floor.
    "denoise_light": ["afftdn=nr=10:nf=-25"],
    "denoise_strong": ["afftdn=nr=20:nf=-20"],
    # Non-local means. Much better artefact behaviour than spectral gating,
    # noticeably more expensive - a good cost/quality contrast for the Pi.
    "denoise_nlm": ["anlmdn=s=0.0001:p=0.002:r=0.006"],
    # RNNoise. Needs a model file: --rnnoise-model path/to/std.rnnn
    # (get one from github.com/GregorR/rnnoise-models). Enabled only if given.
    "denoise_rnn": ["arnndn=m={rnnoise_model}"],

    # --- time stretch (pitch preserving) ------------------------------------
    "slow90": ["atempo=0.9"],
    "slow80": ["atempo=0.8"],
    # Speeding up is the cheap direction: shorter audio = faster inference.
    # If WER holds at 1.1x you have a free latency win. Worth testing.
    "fast110": ["atempo=1.1"],

    # --- combinations you would actually deploy -----------------------------
    "clean": ["highpass=f=80", "afftdn=nr=10:nf=-25", "loudnorm=I=-23:TP=-2:LRA=7"],
    "clean_slow90": ["highpass=f=80", "afftdn=nr=10:nf=-25",
                     "loudnorm=I=-23:TP=-2:LRA=7", "atempo=0.9"],
}

# Applied to every condition, last. Mono because both Whisper and Vosk expect
# it; 16 kHz because that is the training sample rate of both.
TAIL_FILTERS = ["aresample=16000:resampler=soxr", "aformat=sample_fmts=s16:channel_layouts=mono"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def run(cmd, timeout=300):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path])
    if r.returncode != 0:
        return None
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return None


def read_wav_mono16(path):
    """Read a 16 kHz mono s16 wav into a float32 array in [-1, 1)."""
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        sw, ch = w.getsampwidth(), w.getnchannels()
    if sw != 2:
        raise ValueError(f"expected 16-bit pcm, got {sw*8}-bit")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a


def signal_stats(x, sr=16000):
    """Cheap, defensible recording-quality stats.

    SNR is estimated by framing the signal and comparing a high energy
    percentile (speech) against a low one (noise floor). This is not a true
    SNR - there is no clean reference - so report it as an estimate and use it
    only to RANK recordings, e.g. 'denoising helped most on the noisiest third'.
    """
    if x.size == 0:
        return dict(rms_dbfs=None, peak_dbfs=None, clip_pct=None, snr_db_est=None)

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x ** 2)))
    clip_pct = float(np.mean(np.abs(x) >= 0.999) * 100.0)

    win, hop = int(0.025 * sr), int(0.010 * sr)
    if x.size < win:
        frames_db = np.array([20 * np.log10(rms + 1e-12)])
    else:
        idx = np.arange(0, x.size - win, hop)
        frames = np.stack([x[i:i + win] for i in idx])
        fe = np.sqrt(np.mean(frames ** 2, axis=1))
        frames_db = 20 * np.log10(fe + 1e-12)

    speech_db = float(np.percentile(frames_db, 90))
    noise_db = float(np.percentile(frames_db, 10))

    def db(v):
        return round(20 * np.log10(v + 1e-12), 2)

    return dict(rms_dbfs=db(rms), peak_dbfs=db(peak), clip_pct=round(clip_pct, 4),
                snr_db_est=round(speech_db - noise_db, 2))


def build_filter_chain(name, args):
    chain = []
    for f in CONDITIONS[name]:
        if "{rnnoise_model}" in f:
            if not args.rnnoise_model:
                raise ValueError(f"{name} needs --rnnoise-model")
            f = f.format(rnnoise_model=args.rnnoise_model)
        chain.append(f)
    return ",".join(chain + TAIL_FILTERS)


def convert(src, dst, filter_chain):
    """Run one ffmpeg pass. Returns (wall_seconds, returncode, stderr)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-i", src, "-af", filter_chain, "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", dst]
    t0 = time.perf_counter()
    r = run(cmd)
    return time.perf_counter() - t0, r.returncode, r.stderr.strip()


def measure_overhead(sample_src, repeats=5):
    """Time a decode-only null run.

    ffmpeg process startup plus decode is 30-80 ms on a Pi 5 and is NOT part of
    your filter's real cost - in deployment you would filter in-process, not
    spawn ffmpeg per utterance. Subtracting it stops you overstating how
    expensive preprocessing is by an order of magnitude on short commands.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
             "-i", sample_src, "-af", ",".join(TAIL_FILTERS),
             "-f", "null", "-"])
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def find_audio(root):
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS and not fn.startswith("."):
                p = os.path.join(dirpath, fn)
                out.append((p, os.path.relpath(p, root)))
    return sorted(out, key=lambda t: t[1])


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", required=True, help="corpus root (walked recursively)")
    ap.add_argument("--out", default="./variants", help="where condition trees are written")
    ap.add_argument("--conditions", nargs="+", default=["raw", "hp80", "denoise_light", "slow90"])
    ap.add_argument("--all", action="store_true", help="run every defined condition")
    ap.add_argument("--list", action="store_true", help="print conditions and exit")
    ap.add_argument("--limit", type=int, default=0, help="only first N files (piloting)")
    ap.add_argument("--speakers", nargs="+", help="restrict to these top-level folders")
    ap.add_argument("--cost-repeats", type=int, default=1,
                    help="repeat each conversion N times, keep median timing")
    ap.add_argument("--rnnoise-model", help="path to .rnnn model for denoise_rnn")
    ap.add_argument("--force", action="store_true", help="rebuild files that already exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'condition':<16} filter chain")
        print("-" * 78)
        for k, v in CONDITIONS.items():
            print(f"{k:<16} {','.join(v) if v else '(passthrough)'}")
        print(f"\nappended to all: {','.join(TAIL_FILTERS)}")
        return

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found - sudo apt install ffmpeg")

    conditions = list(CONDITIONS) if args.all else args.conditions
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        sys.exit(f"unknown condition(s): {unknown}\nrun --list to see valid names")
    if "raw" not in conditions:
        print("!! 'raw' is not in your list. You need it as the baseline that every\n"
              "   other condition is compared against. Adding it.")
        conditions = ["raw"] + conditions

    files = find_audio(args.audio_root)
    if args.speakers:
        keep = set(args.speakers)
        files = [(p, r) for p, r in files if r.split(os.sep)[0] in keep]
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no audio found under {args.audio_root}")

    chains = {c: build_filter_chain(c, args) for c in conditions}

    print(f"files      : {len(files)}")
    print(f"conditions : {len(conditions)} -> {len(files) * len(conditions)} outputs")
    for c in conditions:
        print(f"  {c:<16} {chains[c]}")
    if args.dry_run:
        for _, rel in files[:10]:
            print("  e.g.", rel)
        return

    overhead = measure_overhead(files[0][0])
    print(f"\nffmpeg startup+decode overhead: {overhead*1000:.0f} ms (subtracted from proc_sec_net)\n")

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "preprocess_log.csv")
    fields = ["condition", "rel_path", "speaker_folder", "src_ext", "dur_in", "dur_out",
              "proc_sec", "proc_sec_net", "rtf_prep", "rms_dbfs", "peak_dbfs",
              "clip_pct", "snr_db_est", "ok", "error"]

    done = set()
    if os.path.exists(log_path) and not args.force:
        with open(log_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["condition"], row["rel_path"]))

    new_log = not os.path.exists(log_path) or args.force
    src_stats = {}
    failures = []

    with open(log_path, "w" if args.force else "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new_log:
            w.writeheader()

        total = len(files) * len(conditions)
        n = 0
        t_start = time.time()

        for cond in conditions:
            for src, rel in files:
                n += 1
                if (cond, rel) in done:
                    continue

                dst = os.path.join(args.out, cond, os.path.splitext(rel)[0] + ".wav")
                if os.path.exists(dst) and not args.force:
                    continue

                times = []
                rc, err = 0, ""
                for _ in range(max(1, args.cost_repeats)):
                    t, rc, err = convert(src, dst, chains[cond])
                    times.append(t)
                    if rc != 0:
                        break
                proc = statistics.median(times)

                dur_in = probe_duration(src)
                row = dict(condition=cond, rel_path=rel,
                           speaker_folder=rel.split(os.sep)[0],
                           src_ext=os.path.splitext(src)[1].lower(),
                           dur_in=round(dur_in, 3) if dur_in else None,
                           proc_sec=round(proc, 4),
                           proc_sec_net=round(max(proc - overhead, 0.0), 4),
                           ok=int(rc == 0), error=err[:200])

                if rc != 0:
                    failures.append((cond, rel, err[:120]))
                else:
                    try:
                        x = read_wav_mono16(dst)
                        dur_out = x.size / 16000.0
                        row["dur_out"] = round(dur_out, 3)
                        row["rtf_prep"] = round(row["proc_sec_net"] / dur_out, 4) if dur_out else None
                        # Quality stats describe the SOURCE recording, so compute
                        # them once from 'raw' and reuse - they characterise the
                        # speaker's environment, not the condition.
                        if cond == "raw":
                            src_stats[rel] = signal_stats(x)
                        row.update(src_stats.get(rel, {}))
                    except Exception as e:
                        row["error"] = f"stats failed: {e}"

                w.writerow(row)
                fh.flush()

                if n % 25 == 0 or n == total:
                    el = time.time() - t_start
                    print(f"  [{n}/{total}] {cond:<14} {el/60:.1f} min elapsed", flush=True)

    print(f"\nlog: {log_path}")
    for cond in conditions:
        print(f"  {os.path.join(args.out, cond)}")

    if failures:
        print(f"\n!! {len(failures)} conversion(s) failed:")
        for c, r, e in failures[:10]:
            print(f"   {c} {r}: {e}")

    print("\nnext:")
    for cond in conditions:
        print(f"  python3 tools/benchmark/stt_benchmark.py --audio-root {os.path.join(args.out, cond)} "
              f"--tag prep_{cond} --out results/")


if __name__ == "__main__":
    main()
