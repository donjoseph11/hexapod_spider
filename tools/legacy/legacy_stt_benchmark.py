#!/usr/bin/env python3
"""
stt_benchmark.py - compare whisper.cpp and Vosk models on a folder of audio.

OUTPUT
    ONE workbook, named <audio folder>_<number of models>.xlsx
        --audio-dir ./recording1 with 4 models -> recording1_4.xlsx
    Sheets:
        results      one row per recording x model x repeat.
                     Wrong transcripts are highlighted RED.
        per_command  one row per command: how long each model took on it,
                     plus total_wall_sec (all models added together).
        summary      one row per model.
        environment  machine + run info.

NUMBERS
    Word-numbers are converted to digits before scoring, including compounds,
    so "one hundred eighty" == "180", "ninety" == "90", "fifteen" == "15".
    Without this, whisper (writes digits) and vosk (writes words) get marked
    wrong for saying the same thing.

GROUND TRUTH
    Best: keep your command list in COMMANDS below. Each recording is matched
    to its closest command, so numbers survive intact.
    Also supported: ./audio/<name>.txt, ./audio/ground_truth.csv (filename,text),
    or --gt-from-filename as a fallback.

COOLDOWN
    Waits until the SoC is at or below --cool-to, OR until --rest seconds pass -
    whichever happens FIRST. A cool Pi means no wait at all.

USAGE
    python3 stt_benchmark.py \
      --audio-dir ./recordings/recording1 \
      --out ./results \
      --gt-from-filename

    python3 stt_benchmark.py --audio-dir ./recording1 --only whisper-base.en --rest 0

FOR PARTICULAR MODEL

python3 2stt_benchmark.py \
  --audio-dir ./recordings/Female_1_recordings/Female_1 \
  --out ./results \
  --gt-from-filename \
  --only whisper-base.en

DEPENDENCIES
    pip install psutil pandas openpyxl vosk
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import wave

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WHISPER_BIN = os.environ.get(
    "WHISPER_BIN",
    os.path.join(PROJECT_ROOT, "models", "whisper.cpp", "build", "bin", "whisper-cli"),
)
WHISPER_DIR = os.environ.get(
    "WHISPER_DIR",
    os.path.join(PROJECT_ROOT, "models", "whisper.cpp", "models"),
)
VOSK_DIR = os.environ.get("VOSK_DIR", os.path.join(PROJECT_ROOT, "models", "vosk"))

WHISPER_THREADS = 4

MODELS = [
    {"name": "whisper-tiny.en",   "type": "whisper", "path": f"{WHISPER_DIR}/ggml-tiny.en.bin"},
    {"name": "whisper-base.en",   "type": "whisper", "path": f"{WHISPER_DIR}/ggml-base.en.bin"},
    {"name": "whisper-small.en",  "type": "whisper", "path": f"{WHISPER_DIR}/ggml-small.en.bin"},
    {"name": "whisper-medium.en", "type": "whisper", "path": f"{WHISPER_DIR}/ggml-medium.en.bin"},
    {"name": "vosk-small-0.15",   "type": "vosk", "path": f"{VOSK_DIR}/vosk-model-small-en-us-0.15"},
    {"name": "vosk-0.22",         "type": "vosk", "path": f"{VOSK_DIR}/vosk-model-en-us-0.22"},
]

# The exact prompts the recordings were generated from. Each audio file is
# matched to its closest entry, which keeps numbers exact.
COMMANDS = [
    "Could you move forward for me?",
    "Please move forward 10 feet.",
    "Can you go backward 3 steps?",
    "Would you take fifteen steps forward?",
    "Please go back two steps.",
    "Could you turn around?",
    "Can you move backward for me?",
    "Please turn left 90 degrees.",
    "Could you turn left?",
    "Can you turn right 180 degrees?",
    "Please turn right.",
    "Can you dance for me?",
    "Hi, could you move forward 2 feet, turn right, and then dance for me?",
    "Please go ahead 3 steps, turn left, stop for a moment, and then turn around.",
    "Can you move backward 1 meter, turn right 90 degrees, and continue forward 2 feet?",
    "Hi, I'd like you to turn around, move forward 5 feet, and then enter rest mode.",
    "Please turn right, move forward 3 steps, turn left, move backward another 2 steps, "
    "and then reset yourself.",
    "Could you go backward 2 feet, rotate 180 degrees, move forward, return to your "
    "starting position, and stop.",
    "Please walk forward 4 feet, turn right, dance for a few seconds, and then relax.",
    "Please turn around, take 2 steps backward, turn left, move forward 2 feet, return "
    "to the starting position, and then reset yourself.",
]

SAMPLE_INTERVAL = 0.1
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

COLUMNS = ["model", "file", "repeat", "audio_sec", "wall_sec", "rtf",
           "wer", "cer", "exact_match", "reference", "hypothesis", "reference_source",
           "temp_start_c", "temp_end_c", "temp_peak_c",
           "cpu_mean_pct", "cpu_peak_pct", "mem_peak_mb",
           "gpu_mem_mb", "gpu_freq_mhz", "throttled", "exit_code"]

# ============================================================================
# TEXT + SCORING
# ============================================================================

_TAG_RE = re.compile(r"[\[\(<][^\]\)>]*[\]\)>]")   # [BLANK_AUDIO], (music), <noise>
_PUNCT_RE = re.compile(r"[^\w\s']")

_ONES = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}

# harmless spelling/plural differences between the two engines
_SYNONYMS = {"ft": "feet", "foot": "feet", "metre": "meters", "metres": "meters",
             "meter": "meters", "degree": "degrees", "step": "steps",
             "second": "seconds", "ok": "okay"}


def words_to_digits(words):
    """['one','hundred','eighty'] -> ['180'];  ['ninety'] -> ['90'].

    Whisper writes numerals, vosk writes words. Collapsing both to digits stops
    that difference from being scored as an error.
    """
    out, cur = [], None
    for w in words:
        if w in _ONES or w in _TENS:
            cur = (cur or 0) + _ONES.get(w, _TENS.get(w))
        elif w in _SCALES and cur is not None:
            cur *= _SCALES[w]
        elif w == "and" and cur is not None:          # "one hundred and eighty"
            continue
        else:
            if cur is not None:
                out.append(str(cur))
                cur = None
            out.append(w)
    if cur is not None:
        out.append(str(cur))
    return out


def normalise(text):
    """Lowercase, strip tags/punctuation, canonicalise numbers and plurals."""
    t = _PUNCT_RE.sub(" ", _TAG_RE.sub(" ", text.lower()))
    words = [_SYNONYMS.get(w, w) for w in t.split()]
    return " ".join(words_to_digits(words))


def edit_distance(ref, hyp):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(prev[j - 1] if r == h else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def error_rate(ref, hyp, unit="word"):
    """WER (unit='word') or CER (unit='char'). None if there's no reference."""
    if not ref:
        return None
    if unit == "word":
        r, h = ref.split(), hyp.split()
    else:
        r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    return edit_distance(r, h) / len(r) if r else None


# ============================================================================
# SYSTEM METRICS
# ============================================================================

def soc_temp():
    """SoC temperature in Celsius, or None."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        return None


def _vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def gpu_stats():
    """(gpu_mem_mb, v3d_clock_mhz) for the VideoCore.

    NOTE: whisper.cpp and vosk both run on the CPU, so these describe the GPU's
    allocation and clock, not work being done on it. They will be identical in
    every row. There is no per-process GPU utilisation counter on the Pi.
    """
    mem = clock = None
    m = re.search(r"([\d.]+)", _vcgencmd("get_mem", "gpu"))
    if m:
        mem = float(m.group(1))
    m = re.search(r"=(\d+)", _vcgencmd("measure_clock", "v3d"))
    if m:
        clock = round(int(m.group(1)) / 1e6, 1)
    return mem, clock


def throttled_now():
    """True if undervolted, freq-capped, throttled or at temp limit right now."""
    try:
        return bool(int(_vcgencmd("get_throttled").split("=")[-1], 16) & 0xF)
    except (ValueError, IndexError):
        return None


def cooldown(max_seconds, target_c):
    """Idle until temp <= target_c OR max_seconds elapse - whichever comes first."""
    if max_seconds <= 0:
        return
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        t = soc_temp()
        if t is None or t <= target_c:
            break
        print(f"\r    cooling {t:.0f}C -> {target_c:.0f}C "
              f"({deadline - time.time():3.0f}s max)   ", end="", flush=True)
        time.sleep(2)
    print("\r" + " " * 50 + "\r", end="")


class ProcMonitor(threading.Thread):
    """Poll a subprocess and its children for CPU%, RSS and SoC temp."""

    def __init__(self, pid, interval=SAMPLE_INTERVAL):
        super().__init__(daemon=True)
        import psutil
        self.psutil = psutil
        self.proc = psutil.Process(pid)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.cpu, self.rss, self.temps = [], [], []

    def run(self):
        p = self.proc
        try:
            p.cpu_percent(None)          # prime the counter
        except Exception:
            return
        while not self.stop_flag.is_set():
            try:
                cpu = rss = 0
                for q in [p] + p.children(recursive=True):
                    try:
                        cpu += q.cpu_percent(None)
                        rss += q.memory_info().rss
                    except Exception:
                        pass
                if rss:
                    self.cpu.append(cpu)
                    self.rss.append(rss)
                t = soc_temp()
                if t is not None:
                    self.temps.append(t)
            except self.psutil.NoSuchProcess:
                break
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2)

    def stats(self):
        return {
            "cpu_mean_pct": round(statistics.mean(self.cpu), 1) if self.cpu else None,
            "cpu_peak_pct": round(max(self.cpu), 1) if self.cpu else None,
            "mem_peak_mb": round(max(self.rss) / 1e6, 1) if self.rss else None,
            "temp_peak_c": max(self.temps) if self.temps else None,
        }


# ============================================================================
# AUDIO
# ============================================================================

def ensure_16k_mono(path, tmpdir):
    """Both engines want 16 kHz mono 16-bit wav. Convert with ffmpeg if needed."""
    try:
        with wave.open(path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) == (16000, 1, 2):
                return path, dur
    except Exception:
        pass
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"{path} is not 16k mono 16-bit wav and ffmpeg is missing")
    out = os.path.join(tmpdir, os.path.splitext(os.path.basename(path))[0] + "_16k.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", out], check=True)
    with wave.open(out, "rb") as wf:
        return out, wf.getnframes() / wf.getframerate()


# ============================================================================
# ENGINES (both run as subprocesses so the measurements are comparable)
# ============================================================================

def transcribe(model, wav_path):
    """Run one model on one file. Returns (text, elapsed, stats, returncode)."""
    if model["type"] == "whisper":
        cmd = [WHISPER_BIN, "-m", model["path"], "-f", wav_path,
               "-nt", "-l", "en", "-t", str(WHISPER_THREADS), "-np"]
    else:
        cmd = [sys.executable, os.path.abspath(__file__),
               "--vosk-worker", model["path"], wav_path]

    import psutil
    t0 = time.perf_counter()
    proc = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mon = ProcMonitor(proc.pid)
    mon.start()
    try:
        out, err = proc.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    elapsed = time.perf_counter() - t0
    mon.stop()
    if proc.returncode != 0:
        print(f"    ! exit {proc.returncode}: {err.strip()[-300:]}")
    return out.strip(), elapsed, mon.stats(), proc.returncode


def vosk_worker(model_path, wav_path):
    """Child process: transcribe with vosk, print the transcript to stdout."""
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    model = Model(model_path)
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(False)
    pieces = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            pieces.append(json.loads(rec.Result()).get("text", ""))
    pieces.append(json.loads(rec.FinalResult()).get("text", ""))
    print(" ".join(p for p in pieces if p).strip())


# ============================================================================
# GROUND TRUTH
# ============================================================================

_TRAILING_INDEX_RE = re.compile(r"_\d+$")


def match_command(stem, commands):
    """Closest COMMANDS entry for a filename stem. Returns (command, badness)."""
    probe = normalise(stem.replace("_", " ")).split()
    best, best_ratio = None, 1e9
    for cmd in commands:
        ref = normalise(cmd).split()
        ratio = edit_distance(ref, probe) / max(len(ref), 1)
        if ratio < best_ratio:
            best, best_ratio = cmd, ratio
    return best, best_ratio


def load_ground_truth(audio_dir, files, from_filename, commands):
    """stem -> (reference text, source). Prefers explicit files, then COMMANDS."""
    gt = {}
    csv_path = os.path.join(audio_dir, "ground_truth.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row.get("filename") or row.get("file")
                txt = row.get("text") or row.get("transcript") or row.get("reference")
                if key and txt:
                    gt[os.path.splitext(os.path.basename(key))[0]] = (txt.strip(), "csv")

    for name in os.listdir(audio_dir):
        if name.endswith(".txt"):
            with open(os.path.join(audio_dir, name), encoding="utf-8") as f:
                gt.setdefault(os.path.splitext(name)[0], (f.read().strip(), "txt"))

    loose = []
    for fname in files:
        stem = os.path.splitext(fname)[0]
        if stem in gt:
            continue
        if commands:
            cmd, badness = match_command(stem, commands)
            if badness <= 0.5:
                gt[stem] = (cmd, "commands")
                if badness > 0.2:
                    loose.append((fname, cmd, badness))
                continue
        if from_filename:
            # last resort: strip one trailing _<take number>, underscores -> spaces
            gt[stem] = (_TRAILING_INDEX_RE.sub("", stem).replace("_", " ").strip(),
                        "filename")
    if loose:
        print("  loose command matches - check these in the results sheet:")
        for fname, cmd, b in loose:
            print(f"    {fname} -> {cmd!r} (badness {b:.2f})")
    return gt


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default=".",
                    help="output directory (the workbook is auto-named)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="model names to run (default: all)")
    ap.add_argument("--rest", type=float, default=60.0,
                    help="max seconds to wait between runs (default 60, 0 = never wait)")
    ap.add_argument("--cool-to", type=float, default=60.0,
                    help="stop waiting once the SoC is at or below this temp (default 60)")
    ap.add_argument("--gt-from-filename", action="store_true",
                    help="fall back to the filename when no COMMANDS entry matches")
    ap.add_argument("--no-commands", action="store_true",
                    help="ignore the COMMANDS list in the CONFIG block")
    ap.add_argument("--warmup", action="store_true",
                    help="discard one untimed run per model (avoids cold-cache bias)")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.audio_dir)
                   if f.lower().endswith(AUDIO_EXTS))
    if not files:
        sys.exit(f"No audio files in {args.audio_dir}")

    models = [m for m in MODELS if not args.only or m["name"] in args.only]
    for m in models:
        if not os.path.exists(m["path"]):
            print(f"SKIP (path not found): {m['name']}")
    models = [m for m in models if os.path.exists(m["path"])]
    if not models:
        sys.exit("No usable models. Check the paths in the CONFIG block.")

    commands = [] if args.no_commands else COMMANDS
    gt = load_ground_truth(args.audio_dir, files, args.gt_from_filename, commands)
    print(f"{len(files)} file(s) x {len(models)} model(s) x {args.repeats} repeat(s) "
          f"= {len(files) * len(models) * args.repeats} runs, "
          f"{len(gt)} reference(s)")
    print(f"cooldown: up to {args.rest:.0f}s, or until {args.cool_to:.0f}C, "
          f"whichever comes first\n")

    rows = []
    tmpdir = tempfile.mkdtemp(prefix="sttbench_")
    started = time.time()
    try:
        for m in models:
            print(f"=== {m['name']} ===")
            if args.warmup:
                try:
                    wav, _ = ensure_16k_mono(os.path.join(args.audio_dir, files[0]), tmpdir)
                    print("  warm-up run (discarded)...")
                    transcribe(m, wav)
                except Exception as e:
                    print(f"  warm-up failed: {e}")

            for fname in files:
                try:
                    wav, dur = ensure_16k_mono(os.path.join(args.audio_dir, fname), tmpdir)
                except Exception as e:
                    print(f"  {fname}: conversion failed ({e})")
                    continue
                ref, ref_src = gt.get(os.path.splitext(fname)[0], (None, "none"))
                ref_n = normalise(ref) if ref else None

                for rep in range(1, args.repeats + 1):
                    if rows:                       # no wait before the very first run
                        cooldown(args.rest, args.cool_to)

                    temp_start = soc_temp()
                    text, elapsed, stats, rc = transcribe(m, wav)
                    temp_end = soc_temp()
                    gpu_mem, gpu_freq = gpu_stats()

                    hyp_n = normalise(text)
                    wer = error_rate(ref_n, hyp_n, "word")
                    cer = error_rate(ref_n, hyp_n, "char")

                    rows.append({
                        "model": m["name"],
                        "file": fname,
                        "repeat": rep,
                        "audio_sec": round(dur, 2),
                        "wall_sec": round(elapsed, 3),
                        "rtf": round(elapsed / dur, 3) if dur else None,
                        "wer": round(wer, 4) if wer is not None else None,
                        "cer": round(cer, 4) if cer is not None else None,
                        "exact_match": (hyp_n == ref_n) if ref_n else None,
                        "reference": ref or "",
                        "hypothesis": text,
                        "reference_source": ref_src,
                        "temp_start_c": temp_start,
                        "temp_end_c": temp_end,
                        "gpu_mem_mb": gpu_mem,
                        "gpu_freq_mhz": gpu_freq,
                        "throttled": throttled_now(),
                        "exit_code": rc,
                        **stats,
                    })

                    r = rows[-1]
                    wer_s = f"WER {r['wer']:.1%}" if r["wer"] is not None else "WER n/a"
                    mark = "" if r["exact_match"] in (True, None) else "  <-- MISMATCH"
                    print(f"  {fname} [{rep}/{args.repeats}] {elapsed:6.2f}s "
                          f"RTF {r['rtf']}  {wer_s}  mem {r['mem_peak_mb']}MB  "
                          f"temp {temp_start}->{temp_end}C{mark}")
                    print(f"      -> {text[:100]!r}")
            print()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"total elapsed {(time.time() - started) / 60:.1f} min")
    write_excel(rows, output_path(args.out, args.audio_dir, len(models)))


def output_path(outdir, audio_dir, n_models):
    """<output dir>/<audio folder name>_<number of models>.xlsx"""
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    folder = os.path.basename(os.path.normpath(os.path.abspath(audio_dir)))
    safe = re.sub(r"[^\w.-]+", "_", folder)
    return os.path.join(outdir, f"{safe}_{n_models}.xlsx")


def write_excel(rows, out_path):
    import pandas as pd
    from openpyxl.styles import Font, PatternFill

    df = pd.DataFrame(rows)
    df = df[[c for c in COLUMNS if c in df.columns]]

    # --- per_command: how long each model needed for each spoken command ------
    per_cmd = df.groupby("file", as_index=False).agg(
        reference=("reference", "first"),
        audio_sec=("audio_sec", "first"),
        total_wall_sec=("wall_sec", "sum"),
        mean_wall_sec=("wall_sec", "mean"),
        models=("model", "nunique"),
        mean_wer=("wer", "mean"),
        mismatches=("exact_match", lambda s: int((s == False).sum())),  # noqa: E712
    )
    by_model = df.pivot_table(index="file", columns="model", values="wall_sec",
                              aggfunc="mean").round(2)
    by_model.columns = [f"{c}_sec" for c in by_model.columns]
    per_cmd = per_cmd.merge(by_model.reset_index(), on="file").round(3)

    agg = {"wall_sec": "mean", "rtf": "mean", "wer": "mean", "cer": "mean",
           "cpu_mean_pct": "mean", "cpu_peak_pct": "max", "mem_peak_mb": "max",
           "temp_start_c": "mean", "temp_end_c": "mean", "temp_peak_c": "max"}
    summary = df.groupby("model", as_index=False).agg(
        {k: v for k, v in agg.items() if k in df.columns}).round(3)
    if df["exact_match"].notna().any():
        acc = df.groupby("model")["exact_match"].mean().reset_index(name="exact_match_rate")
        summary = summary.merge(acc, on="model")
    summary = summary.sort_values("wer" if "wer" in summary else "rtf")

    env = pd.DataFrame({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": " ".join(os.uname()),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "whisper_threads": WHISPER_THREADS,
        "temp_at_end_c": soc_temp(),
        "note": "wall_sec = seconds to transcribe that one command, model load "
                "included (fresh process per run). GPU columns are VideoCore "
                "state only - both engines are CPU-only.",
    }.items(), columns=["key", "value"])

    red_fill = PatternFill("solid", fgColor="FFC7CE")
    red_font = Font(color="9C0006")

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for sheet, frame in (("results", df), ("per_command", per_cmd),
                             ("summary", summary), ("environment", env)):
            frame.to_excel(xw, sheet_name=sheet, index=False)
            ws = xw.sheets[sheet]
            ws.freeze_panes = "A2"
            for i, col in enumerate(frame.columns, start=1):
                longest = frame[col].astype(str).str.len().max() if len(frame) else 10
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
                    min(max(len(str(col)) + 2, longest + 2), 60)

        # red-fill every wrong transcript on the results sheet
        ws = xw.sheets["results"]
        cols = {c: i + 1 for i, c in enumerate(df.columns)}
        for r, ok in enumerate(df["exact_match"], start=2):
            if ok is False or ok == 0:
                for name in ("hypothesis", "wer", "cer", "exact_match"):
                    if name in cols:
                        cell = ws.cell(row=r, column=cols[name])
                        cell.fill = red_fill
                        cell.font = red_font

    n_bad = int((df["exact_match"] == False).sum())  # noqa: E712
    print(f"wrote {out_path}  ({n_bad} mismatched transcript(s) highlighted red)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--vosk-worker":
        vosk_worker(sys.argv[2], sys.argv[3])
    else:
        main()
