"""
stt_benchmark.py - benchmark whisper.cpp and Vosk models across many speakers.

WHAT CHANGED vs the previous version
    * speaker_id and gender are parsed from the folder path and stored as
      first-class columns, so fairness analysis works without post-hoc parsing.
    * Ground truth comes from a command INDEX in the filename, not fuzzy
      matching on filename text. Deterministic, no silent wrong references.
    * The run order is a single randomised plan (fixed seed), so thermal drift
      is decorrelated from model instead of confounded with it.
    * Whisper model-load time is separated from inference time, so RTF means
      steady-state inference the way the hexapod will actually run it.
    * Every finished run is appended to a CSV immediately and skipped on
      restart. A crash at hour 9 costs you one run, not the whole sweep.
    * Adds per-model std / CV / p95 (reproducibility) and a critical-token
      error rate (direction words, numbers, units - what the robot acts on).

EXPECTED LAYOUT
    <root>/
        speakers.csv                 optional: speaker_id,l1,state,age_band,...
        male_1/    cmd01_take1.wav  cmd01_take2.wav  cmd02_take1.wav ...
        female_1/  cmd01_take1.wav  ...

    Speaker id and gender are read from the folder name (male_1 -> male_1, male).
    Command index is read from the filename (cmd07 -> COMMANDS[6]).
    If your filenames differ, set --cmd-regex or supply ground_truth.csv.

USAGE
    python3 tools/benchmark/stt_benchmark.py --audio-root ./recordings \
                             --out ./results

    # resume after a crash - just run the same command again
    # dry run to see the plan and time estimate without transcribing:
    python3 stt_benchmark.py --audio-root ./recordings --dry-run

DEPENDENCIES
    pip install psutil pandas openpyxl vosk
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WHISPER_BIN = os.environ.get(
    "WHISPER_BIN",
    str(PROJECT_ROOT / "models" / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
)
WHISPER_DIR = os.environ.get(
    "WHISPER_DIR",
    str(PROJECT_ROOT / "models" / "whisper.cpp" / "models"),
)
VOSK_DIR = os.environ.get("VOSK_DIR", str(PROJECT_ROOT / "models" / "vosk"))

WHISPER_THREADS = 4

MODELS = [
    {"name": "whisper-tiny.en",   "type": "whisper", "path": f"{WHISPER_DIR}/ggml-tiny.en.bin"},
    {"name": "whisper-base.en",   "type": "whisper", "path": f"{WHISPER_DIR}/ggml-base.en.bin"},
    {"name": "whisper-small.en",  "type": "whisper", "path": f"{WHISPER_DIR}/ggml-small.en.bin"},
    {"name": "whisper-medium.en", "type": "whisper", "path": f"{WHISPER_DIR}/ggml-medium.en.bin"},
    {"name": "vosk-small-0.15",   "type": "vosk", "path": f"{VOSK_DIR}/vosk-model-small-en-us-0.15"},
    {"name": "vosk-0.22",         "type": "vosk", "path": f"{VOSK_DIR}/vosk-model-en-us-0.22"},
]

# Index in this list == the number in the filename (cmd01 -> index 0).
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

# Tokens the hexapod actually acts on. An error here changes robot behaviour;
# an error anywhere else is usually cosmetic.
CRITICAL_TOKENS = {
    "left", "right", "forward", "backward", "back", "ahead", "around",
    "rotate", "turn", "move", "walk", "dance", "stop", "reset", "relax",
    "rest", "return", "feet", "meters", "steps", "degrees",
}

SAMPLE_INTERVAL = 0.1
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

COLUMNS = [
    "speaker_id", "gender", "cmd_index", "take", "model", "model_type", "repeat",
    "audio_sec", "load_sec", "infer_sec", "wall_sec", "rtf",
    "wer", "cer", "cter", "exact_match",
    "reference", "hypothesis", "reference_source",
    "cool_sec", "cool_reason",
    "temp_start_c", "temp_end_c", "temp_peak_c", "throttled_during",
    "cpu_mean_pct", "cpu_peak_pct", "mem_peak_mb",
    "exit_code", "match_badness", "rel_path", "timestamp",
]

# ============================================================================
# TEXT + SCORING
# ============================================================================

_TAG_RE = re.compile(r"[\[\(<][^\]\)>]*[\]\)>]")   # [BLANK_AUDIO], (music), <noise>
_PUNCT_RE = re.compile(r"[^\w\s']")

# NOTE: "oh" deliberately excluded. Speakers say it as a filler and mapping it
# to 0 injects a phantom token, inflating WER for hesitant speakers - exactly
# the population the fairness analysis looks at.
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}

_SYNONYMS = {"ft": "feet", "foot": "feet", "metre": "meters", "metres": "meters",
             "meter": "meters", "degree": "degrees", "step": "steps",
             "second": "seconds", "ok": "okay"}


def words_to_digits(words):
    """['one','hundred','eighty'] -> ['180'].  Whisper writes numerals, vosk
    writes words; collapsing both to digits stops that being scored as error."""
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
    t = _PUNCT_RE.sub(" ", _TAG_RE.sub(" ", (text or "").lower()))
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


def _critical_seq(norm_text):
    """Keep only tokens the robot acts on, in order."""
    return [w for w in norm_text.split()
            if w in CRITICAL_TOKENS or w.isdigit()]


def critical_token_error_rate(ref_n, hyp_n):
    """Error rate over direction words, numbers and units only.

    'move forward 10 feed' scores badly on WER but the intent survives;
    'turn left' -> 'turn right' scores the same WER and walks into a wall.
    This metric separates those two cases."""
    if not ref_n:
        return None
    r = _critical_seq(ref_n)
    if not r:
        return None
    return edit_distance(r, _critical_seq(hyp_n)) / len(r)


# ============================================================================
# SYSTEM METRICS
# ============================================================================

def soc_temp():
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


def throttled_now():
    """True if undervolted / freq-capped / throttled right now (live bits)."""
    try:
        return bool(int(_vcgencmd("get_throttled").split("=")[-1], 16) & 0xF)
    except (ValueError, IndexError):
        return None


def cpu_governor():
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except Exception:
        return None


def sha256_of(path, limit_mb=None):
    """Hash a model file (or the first N MB of a big one) for provenance."""
    h = hashlib.sha256()
    budget = limit_mb * 1024 * 1024 if limit_mb else None
    try:
        if os.path.isdir(path):
            return "dir"
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                if budget is not None:
                    budget -= len(chunk)
                    if budget <= 0:
                        break
        return h.hexdigest()[:16]
    except Exception:
        return None


def measure_idle_floor(seconds=45, settle_below=0.3):
    """Sit idle and watch the temperature bottom out.

    A target below the achievable floor makes every cooldown hit its cap, which
    silently turns a 'strict' gate back into a fixed delay. Measuring first
    means the target is chosen against what this board, in this room, with this
    cooling, can actually reach."""
    t0 = soc_temp()
    if t0 is None:
        return None
    print(f"calibrating idle temperature (up to {seconds}s, starting {t0:.0f}C)...")
    readings, last, stable = [t0], t0, 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        time.sleep(3)
        t = soc_temp()
        if t is None:
            break
        readings.append(t)
        # stop early once it stops falling
        stable = stable + 1 if (last - t) < settle_below else 0
        last = t
        print(f"\r    {t:.0f}C   ", end="", flush=True)
        if stable >= 3:
            break
    floor = min(readings)
    print(f"\r    idle floor ~{floor:.0f}C after "
          f"{len(readings) * 3}s          ")
    return floor


class CoolTimeout(Exception):
    """The board would not drop below the hard ceiling. Better to stop than to
    record a run at a temperature the rest of the sweep never saw."""


def cooldown(target_c, ceiling_c, grace_sec, timeout_sec, strict=True):
    """Gate a run on SoC temperature.

    Three outcomes, in priority order:
      1. temp <= target_c            -> run immediately ('target')
      2. waited >= grace_sec AND temp <= ceiling_c
                                     -> good enough, run ('ceiling')
      3. otherwise                   -> keep waiting

    The board is NEVER allowed to run above ceiling_c. If it will not drop
    below the ceiling within timeout_sec, CoolTimeout is raised and the sweep
    stops - the completed runs are already saved, so it resumes cleanly once
    cooling is fixed. Running hot 'just this once' is worse than stopping: it
    puts one contaminated point into a dataset whose whole claim is that
    thermal state was controlled.

    Returns (seconds_waited, reason)."""
    if not strict:
        # legacy behaviour: soft nudge, gives up at grace_sec
        start = time.time()
        while time.time() - start < grace_sec:
            t = soc_temp()
            if t is None or t <= target_c:
                break
            print(f"\r    cooling {t:.1f}C -> {target_c:.0f}C   ", end="", flush=True)
            time.sleep(2)
        print("\r" + " " * 60 + "\r", end="")
        return time.time() - start, "soft"

    start = time.time()
    warned = False
    while True:
        t = soc_temp()
        if t is None:
            return time.time() - start, "no_sensor"

        waited = time.time() - start
        if t <= target_c:
            if waited:
                print("\r" + " " * 70 + "\r", end="")
            return waited, "target"
        if waited >= grace_sec and t <= ceiling_c:
            print("\r" + " " * 70 + "\r", end="")
            return waited, "ceiling"

        if waited >= timeout_sec:
            raise CoolTimeout(
                f"SoC stuck at {t:.1f}C after {waited:.0f}s; ceiling is "
                f"{ceiling_c:.0f}C. Stopping rather than recording a hot run.")
        if waited > 120 and not warned:
            warned = True
            print(f"\n    !! {waited:.0f}s at {t:.1f}C and still above the "
                  f"{ceiling_c:.0f}C ceiling - check airflow/ambient.")

        goal = target_c if waited < grace_sec else ceiling_c
        label = "target" if waited < grace_sec else "CEILING"
        print(f"\r    cooling {t:.1f}C -> {goal:.0f}C ({label})  "
              f"[{waited:4.0f}s]   ", end="", flush=True)
        time.sleep(2)


class ProcMonitor(threading.Thread):
    """Poll a subprocess tree for CPU%, RSS, SoC temp and throttle events.

    Child Process objects are cached by pid: psutil's first cpu_percent() call
    on a new object always returns 0.0, so recreating them each loop reported
    zero CPU for every child."""

    def __init__(self, pid, interval=SAMPLE_INTERVAL):
        super().__init__(daemon=True)
        import psutil
        self.psutil = psutil
        self.proc = psutil.Process(pid)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.cpu, self.rss, self.temps = [], [], []
        self.throttled = False
        self._kids = {}

    def _tracked(self):
        procs = [self.proc]
        try:
            for k in self.proc.children(recursive=True):
                if k.pid not in self._kids:
                    self._kids[k.pid] = k
                    try:
                        k.cpu_percent(None)      # prime, discard the 0.0
                    except Exception:
                        pass
        except Exception:
            pass
        procs.extend(self._kids.values())
        return procs

    def run(self):
        try:
            self.proc.cpu_percent(None)
        except Exception:
            return
        ticks = 0
        while not self.stop_flag.is_set():
            try:
                cpu = rss = 0.0
                for q in self._tracked():
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
                # throttle is a live bit - sampling only after exit misses it
                ticks += 1
                if ticks % 10 == 0 and throttled_now():
                    self.throttled = True
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
            "throttled_during": self.throttled,
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
    out = os.path.join(tmpdir, hashlib.md5(path.encode()).hexdigest() + "_16k.wav")
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", out], check=True)
    with wave.open(out, "rb") as wf:
        return out, wf.getnframes() / wf.getframerate()


# ============================================================================
# ENGINES
# ============================================================================

_WHISPER_LOAD_RE = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
_WHISPER_TOTAL_RE = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")


def transcribe(model, wav_path):
    """Run one model on one file.

    Returns (text, load_sec, infer_sec, wall_sec, stats, returncode).

    load_sec is separated out because the deployed hexapod loads the model once
    and keeps it resident - charging 1.5 GB of disk read to every whisper-medium
    inference would overstate its cost by a factor unrelated to inference."""
    if model["type"] == "whisper":
        # -np was removed: whisper.cpp prints its own load/encode/decode timings
        # to stderr and we need them.
        cmd = [WHISPER_BIN, "-m", model["path"], "-f", wav_path,
               "-nt", "-l", "en", "-t", str(WHISPER_THREADS)]
    else:
        cmd = [sys.executable, os.path.abspath(__file__),
               "--vosk-worker", model["path"], wav_path]

    import psutil
    t0 = time.perf_counter()
    proc = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mon = ProcMonitor(proc.pid)
    mon.start()
    try:
        out, err = proc.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    wall = time.perf_counter() - t0
    mon.stop()

    load_sec = None
    text = out.strip()

    if model["type"] == "whisper":
        m = _WHISPER_LOAD_RE.search(err or "")
        if m:
            load_sec = float(m.group(1)) / 1000.0
        # strip any stray timing lines that leaked into stdout
        text = "\n".join(l for l in text.splitlines()
                         if "whisper_" not in l and "timings" not in l).strip()
    else:
        try:
            payload = json.loads(text)
            load_sec = payload.get("load_sec")
            text = payload.get("text", "")
        except Exception:
            pass

    infer_sec = (wall - load_sec) if load_sec is not None else None
    if infer_sec is not None and infer_sec <= 0:
        # Parsed load time exceeded wall clock - the timing parse is wrong for
        # this build. Fall back to wall time rather than emitting a negative RTF.
        print(f"    ! load_sec {load_sec:.3f} >= wall {wall:.3f}; "
              f"timing parse looks wrong, falling back to wall time")
        load_sec, infer_sec = None, wall
    if proc.returncode != 0:
        print(f"    ! exit {proc.returncode}: {(err or '').strip()[-300:]}")
    return text, load_sec, infer_sec, wall, mon.stats(), proc.returncode


def vosk_worker(model_path, wav_path):
    """Child process: transcribe with vosk, emit JSON so load time survives."""
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)
    t0 = time.perf_counter()
    model = Model(model_path)
    load_sec = time.perf_counter() - t0

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
    print(json.dumps({"text": " ".join(p for p in pieces if p).strip(),
                      "load_sec": round(load_sec, 4)}))


# ============================================================================
# DISCOVERY: speakers, commands, ground truth
# ============================================================================

_SPEAKER_RE = re.compile(r"(male|female)[_-]?(\d+)", re.I)
_DEFAULT_CMD_RE = r"cmd[_-]?(\d+)"
_TAKE_RE = re.compile(r"take[_-]?(\d+)|_(\d+)$", re.I)


def parse_speaker(rel_path):
    """'male_1/cmd07_take1.wav' -> ('male_1', 'male'). Scans path parts
    right-to-left so a wrapper folder like male_1_recording/ still works."""
    for part in reversed(os.path.dirname(rel_path).split(os.sep)):
        m = _SPEAKER_RE.search(part)
        if m:
            return f"{m.group(1).lower()}_{int(m.group(2))}", m.group(1).lower()
    return None, None


def parse_take(stem):
    m = _TAKE_RE.search(stem)
    if not m:
        return 1
    return int(m.group(1) or m.group(2) or 1)


def match_by_text(stem, commands):
    """Filename IS the command text (e.g. 'Can you dance for me.wav').

    Returns (1-based index, badness). Badness is normalised edit distance, so
    0.0 is a perfect match and higher is worse. Punctuation, case, trailing
    spaces and word-vs-digit numbers are all neutralised first, so
    'Can you turn right 180 degrees' matches 'Can you turn right 180 degrees?'
    exactly."""
    probe = normalise(stem.replace("_", " ")).split()
    best_i, best_bad = None, 1e9
    for i, cmd in enumerate(commands, 1):
        ref = normalise(cmd).split()
        bad = edit_distance(ref, probe) / max(len(ref), 1)
        if bad < best_bad:
            best_i, best_bad = i, bad
    return best_i, best_bad


def discover(audio_root, cmd_regex, match_mode="auto", text_threshold=0.35):
    """Walk the tree, return (recordings, problems).

    match_mode:
        'index' - command number must be in the filename (cmd07_take1.wav)
        'text'  - filename is the command text itself
        'auto'  - try index, fall back to text (default)
    """
    cmd_re = re.compile(cmd_regex, re.I)
    recs, problems = [], []
    for dirpath, _, filenames in os.walk(audio_root):
        for fname in sorted(filenames):
            if not fname.lower().endswith(AUDIO_EXTS):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, audio_root)
            stem = os.path.splitext(fname)[0].strip()

            speaker_id, gender = parse_speaker(rel)
            if speaker_id is None:
                problems.append((rel, "no male_N / female_N folder in path"))
                continue

            cmd_index, badness, how = None, None, None
            if match_mode in ("auto", "index"):
                m = cmd_re.search(stem)
                if m:
                    cmd_index, badness, how = int(m.group(1)), 0.0, "cmd_index"
            if cmd_index is None and match_mode in ("auto", "text"):
                cmd_index, badness = match_by_text(stem, COMMANDS)
                how = "filename_text"
                if badness > text_threshold:
                    problems.append((rel, f"no command matches the filename text "
                                          f"(closest was #{cmd_index}, "
                                          f"badness {badness:.2f})"))
                    continue
            if cmd_index is None:
                problems.append((rel, f"filename has no {cmd_regex!r} match"))
                continue
            if not (1 <= cmd_index <= len(COMMANDS)):
                problems.append((rel, f"cmd index {cmd_index} out of range"))
                continue

            recs.append({
                "path": full, "rel_path": rel, "speaker_id": speaker_id,
                "gender": gender, "cmd_index": cmd_index,
                "take": parse_take(stem),
                "match_badness": round(badness, 3) if badness is not None else None,
                "reference": COMMANDS[cmd_index - 1],
                "reference_source": how,
            })
    return recs, problems


def audit_matches(recs, loose_above=0.05):
    """Sanity-check the filename -> command mapping before hours are spent.

    Two things go wrong silently: a filename that matched the wrong command,
    and a speaker whose 20 files collapsed onto fewer than 20 commands. Both
    produce a plausible-looking WER table built on wrong references."""
    loose = sorted((r for r in recs
                    if (r.get("match_badness") or 0) > loose_above),
                   key=lambda r: -r["match_badness"])
    if loose:
        print(f"\n  {len(loose)} file(s) matched loosely - check these:")
        for r in loose[:10]:
            print(f"    [{r['match_badness']:.2f}] {r['rel_path']}")
            print(f"           -> #{r['cmd_index']} {r['reference']!r}")
        if len(loose) > 10:
            print(f"    ... and {len(loose) - 10} more")

    by_speaker = {}
    for r in recs:
        by_speaker.setdefault(r["speaker_id"], []).append(r)
    collisions = []
    for sp, rows in sorted(by_speaker.items()):
        seen = {}
        for r in rows:
            seen.setdefault(r["cmd_index"], []).append(r["rel_path"])
        dupes = {k: v for k, v in seen.items() if len(v) > 1}
        if dupes:
            collisions.append((sp, dupes))
    if collisions:
        print(f"\n  !! {len(collisions)} speaker(s) have two files mapped to the "
              f"same command:")
        for sp, dupes in collisions[:5]:
            for cmd, paths in list(dupes.items())[:3]:
                print(f"    {sp} cmd#{cmd}: {', '.join(os.path.basename(p) for p in paths)}")
        print("    If these are repeat takes that's fine. If not, two different "
              "commands matched the same reference and the WER for both is wrong.")

    counts = {sp: len({r["cmd_index"] for r in rows})
              for sp, rows in by_speaker.items()}
    expected = max(counts.values()) if counts else 0
    short = {sp: n for sp, n in counts.items() if n < expected}
    if short:
        print(f"\n  !! speaker(s) missing commands (expected {expected} distinct):")
        for sp, n in sorted(short.items())[:10]:
            print(f"    {sp}: {n}")
    return len(loose), len(collisions)


def extract_zips(audio_root, extract_dir, force=False):
    """Unpack any speaker archives found under audio_root.

    Expected: audio/male/male_1_recordings.zip  ->  <extract>/male_1_recordings/male_1/*.wav
    Extraction is cached: an archive already unpacked is skipped, so rerunning
    the sweep costs nothing. A .done marker records completion, so an archive
    interrupted mid-unpack is redone rather than silently half-used."""
    import zipfile

    zips = []
    for dirpath, _, filenames in os.walk(audio_root):
        if os.path.abspath(dirpath).startswith(os.path.abspath(extract_dir)):
            continue                      # don't re-scan our own output
        for f in sorted(filenames):
            if f.lower().endswith(".zip"):
                zips.append(os.path.join(dirpath, f))
    if not zips:
        return []

    os.makedirs(extract_dir, exist_ok=True)
    done = []
    for zpath in zips:
        stem = os.path.splitext(os.path.basename(zpath))[0]
        target = os.path.join(extract_dir, stem)
        marker = target + ".done"
        if os.path.exists(marker) and not force:
            done.append((stem, "cached"))
            continue
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target, exist_ok=True)
        try:
            with zipfile.ZipFile(zpath) as z:
                members = [m for m in z.namelist()
                           if not m.startswith("__MACOSX")
                           and not os.path.basename(m).startswith("._")]
                z.extractall(target, members=members)
            n = sum(1 for _, _, fs in os.walk(target)
                    for f in fs if f.lower().endswith(AUDIO_EXTS))
            open(marker, "w").close()
            done.append((stem, f"extracted {n} audio file(s)"))
        except zipfile.BadZipFile:
            done.append((stem, "!! CORRUPT ZIP - skipped"))
    return done


# ============================================================================
# DISPLAY NAMES (Excel only - the raw CSV keeps machine-readable names)
# ============================================================================

DISPLAY = {
    "speaker_id": "Speaker ID", "gender": "Gender", "cmd_index": "Command No.",
    "take": "Take", "model": "Model", "model_type": "Engine", "repeat": "Repeat",
    "audio_sec": "Audio Length (s)",
    "load_sec": "Model Load Time (s)",
    "infer_sec": "Execution Time per Audio (s)",
    "wall_sec": "Total Process Time (s)",
    "rtf": "Real-Time Factor",
    "wer": "Word Error Rate", "cer": "Character Error Rate",
    "cter": "Critical Token Error Rate", "exact_match": "Exact Match",
    "reference": "Reference (Ground Truth)",
    "hypothesis": "Transcript (Model Output)",
    "reference_source": "Reference Source",
    "match_badness": "Filename Match Badness",
    "cool_sec": "Cooldown Wait (s)",
    "cool_reason": "Cooldown Exit Reason",
    "temp_start_c": "CPU Temp at Start (C)",
    "temp_end_c": "CPU Temp at End (C)",
    "temp_peak_c": "CPU Temp Peak (C)",
    "throttled_during": "Throttled During Run",
    "cpu_mean_pct": "CPU Usage Mean (%)", "cpu_peak_pct": "CPU Usage Peak (%)",
    "mem_peak_mb": "Memory Peak (MB)", "exit_code": "Exit Code",
    "rel_path": "File Path", "timestamp": "Run Timestamp",
    # aggregate sheets
    "runs": "Runs", "n": "Recordings", "speakers": "Speakers",
    "wer_mean": "Mean WER", "wer_std": "WER Std Dev",
    "cter_mean": "Mean Critical Token Error Rate",
    "exact_match_rate": "Exact Match Rate",
    "infer_mean_sec": "Mean Execution Time (s)",
    "infer_std_sec": "Execution Time Std Dev (s)",
    "infer_p95_sec": "Execution Time p95 (s)",
    "infer_cv": "Execution Time CV",
    "load_mean_sec": "Mean Model Load Time (s)",
    "rtf_mean": "Mean Real-Time Factor",
    "throttled_runs": "Throttled Runs",
    "temp_start_mean_c": "Mean Start Temp (C)",
    "temp_start_std_c": "Start Temp Std Dev (C)",
    "cool_mean_sec": "Mean Cooldown Wait (s)",
    "cool_at_target_rate": "Started at Ideal Temp (rate)",
    "wer_std_across_speakers": "WER Std Dev Across Speakers",
    "wer_best_speaker": "Best Speaker WER",
    "wer_worst_speaker": "Worst Speaker WER",
    "wer_p90_speaker": "90th Percentile Speaker WER",
    "worst_minus_best": "Worst - Best Gap",
    "ref_words": "Reference Word Count",
    "mean_infer_sec": "Mean Execution Time (s)",
    "total_infer_sec": "Total Execution Time (s)",
    "total_wall_sec": "Total Process Time (s)",
    "commands": "Commands Covered",
    "key": "Setting", "value": "Value",
}


def prettify(frame):
    """Rename columns to human-readable headers for the workbook."""
    return frame.rename(columns=lambda c: DISPLAY.get(c, _titleise(c)))


def _titleise(col):
    """Fallback for columns we didn't name explicitly (e.g. speakers.csv extras)."""
    out = str(col).replace("_", " ").strip()
    return " ".join(w[:1].upper() + w[1:] for w in out.split()) if out else col



def apply_ground_truth_csv(audio_root, recs):
    """Optional override: ground_truth.csv with columns filename,text."""
    path = os.path.join(audio_root, "ground_truth.csv")
    if not os.path.exists(path):
        return 0
    table = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("filename") or row.get("file")
            txt = row.get("text") or row.get("transcript") or row.get("reference")
            if key and txt:
                table[key.strip()] = txt.strip()
    n = 0
    for r in recs:
        for key in (r["rel_path"], os.path.basename(r["rel_path"])):
            if key in table:
                r["reference"] = table[key]
                r["reference_source"] = "csv"
                n += 1
                break
    return n


def load_speaker_meta(audio_root):
    """speakers.csv -> {speaker_id: {l1, state, age_band, ...}}. Optional."""
    path = os.path.join(audio_root, "speakers.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["speaker_id"].strip(): r for r in csv.DictReader(f)
                if r.get("speaker_id")}


# ============================================================================
# RESUME LOG
# ============================================================================

def load_done(csv_path):
    """Set of (model, rel_path, repeat) already completed."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add((row["model"], row["rel_path"], int(row["repeat"])))
            except (KeyError, ValueError):
                continue
    return done


def append_row(csv_path, row, fieldnames):
    """Append one finished run. fieldnames must stay stable for the whole sweep,
    otherwise the header and the rows drift apart."""
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", required=True,
                    help="folder containing male_N/ and female_N/ subfolders")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--tag", default="run", help="label for the output files")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeats for hardware variance (default 1)")
    ap.add_argument("--repeat-cmds", default="",
                    help="comma-separated cmd indices that get --repeats runs; "
                         "everything else runs once. e.g. 1,4,9,13,17")
    ap.add_argument("--only", nargs="*", help="model names to run (default: all)")
    ap.add_argument("--speakers", nargs="*", help="restrict to these speaker ids")
    ap.add_argument("--rest", type=float, default=20.0,
                    help="max seconds to wait between runs (default 20)")
    ap.add_argument("--cool-to", type=float, default=60.0,
                    help="ideal temperature: at or below this, run immediately")
    ap.add_argument("--cool-ceiling", type=float, default=65.0,
                    help="hard limit - a run NEVER starts above this (default 65)")
    ap.add_argument("--cool-timeout", type=float, default=1800.0,
                    help="stop the sweep if the board won't drop below the "
                         "ceiling within this many seconds (default 1800)")
    ap.add_argument("--soft-cool", action="store_true",
                    help="old behaviour: nudge toward --cool-to for --rest "
                         "seconds then run regardless. Not recommended.")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the idle temperature floor first and warn if "
                         "--cool-to is unreachable")
    ap.add_argument("--seed", type=int, default=1337,
                    help="run-order shuffle seed (logged; keep it fixed)")
    ap.add_argument("--cmd-regex", default=_DEFAULT_CMD_RE,
                    help=f"regex with one capture group for the command number "
                         f"(default {_DEFAULT_CMD_RE!r})")
    ap.add_argument("--warmup", action="store_true",
                    help="one untimed run per model before the plan starts")
    ap.add_argument("--match", default="auto", choices=["auto", "index", "text"],
                    help="how to map a file to a command: 'index' uses a number "
                         "in the filename, 'text' matches the filename against "
                         "the COMMANDS list, 'auto' tries index then text")
    ap.add_argument("--text-threshold", type=float, default=0.35,
                    help="max normalised edit distance for a text match (0.35)")
    ap.add_argument("--extract-dir", default=None,
                    help="where to unpack speaker .zip archives "
                         "(default <out>/_extracted)")
    ap.add_argument("--force-extract", action="store_true",
                    help="re-unpack archives even if already cached")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan and exit without transcribing")
    args = ap.parse_args()

    root = os.path.abspath(args.audio_root)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    extract_dir = os.path.abspath(args.extract_dir or os.path.join(out_dir, "_extracted"))

    unpacked = extract_zips(root, extract_dir, force=args.force_extract)
    if unpacked:
        print(f"archives ({extract_dir}):")
        for stem, status in unpacked:
            print(f"    {stem}: {status}")
        print()

    # search both the original tree (loose files) and the unpacked archives
    recs, problems = discover(root, args.cmd_regex, args.match, args.text_threshold)
    if os.path.isdir(extract_dir):
        e_recs, e_problems = discover(extract_dir, args.cmd_regex,
                                      args.match, args.text_threshold)
        seen = {r["rel_path"] for r in recs}
        recs += [r for r in e_recs if r["rel_path"] not in seen]
        problems += e_problems
    if problems:
        print(f"!! {len(problems)} file(s) could not be identified:")
        for rel, why in problems[:15]:
            print(f"    {rel}: {why}")
        if len(problems) > 15:
            print(f"    ... and {len(problems) - 15} more")
        print("   Fix the naming or pass --cmd-regex. These are SKIPPED.\n")
    if not recs:
        sys.exit("No usable recordings found.")

    n_csv = apply_ground_truth_csv(root, recs)
    if n_csv:
        print(f"ground_truth.csv overrode {n_csv} reference(s)")
    meta = load_speaker_meta(root)

    if args.speakers:
        recs = [r for r in recs if r["speaker_id"] in set(args.speakers)]

    audit_matches(recs)

    how = {}
    for r in recs:
        how[r["reference_source"]] = how.get(r["reference_source"], 0) + 1
    print("\nmatched by: " + ", ".join(f"{k} {v}" for k, v in sorted(how.items())))

    speakers = sorted({r["speaker_id"] for r in recs})
    by_gender = {}
    for r in recs:
        by_gender.setdefault(r["gender"], set()).add(r["speaker_id"])
    print(f"{len(recs)} recording(s) from {len(speakers)} speaker(s): " +
          ", ".join(f"{g} {len(s)}" for g, s in sorted(by_gender.items())))
    if meta:
        missing = [s for s in speakers if s not in meta]
        print(f"speakers.csv: {len(meta)} row(s)" +
              (f", MISSING {len(missing)}: {missing[:5]}" if missing else ""))
    else:
        print("speakers.csv: not found (l1/age/state will be blank)")

    models = [m for m in MODELS if not args.only or m["name"] in args.only]
    for m in models:
        if not os.path.exists(m["path"]):
            print(f"SKIP (path not found): {m['name']}")
    models = [m for m in models if os.path.exists(m["path"])]
    if not models:
        sys.exit("No usable models. Check the paths in the CONFIG block.")

    repeat_cmds = {int(x) for x in args.repeat_cmds.split(",") if x.strip()}

    # ---- build the plan, then shuffle it -----------------------------------
    # Model must NOT be the outer loop: the Pi heats up over the sweep, so an
    # ordered run makes "later model" indistinguishable from "hotter board".
    plan = []
    for m in models:
        for r in recs:
            n = args.repeats if (not repeat_cmds or r["cmd_index"] in repeat_cmds) else 1
            for rep in range(1, n + 1):
                plan.append((m, r, rep))
    random.Random(args.seed).shuffle(plan)

    os.makedirs(os.path.abspath(args.out), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(os.path.abspath(args.out), f"{args.tag}_{len(models)}m")
    csv_path = base + "_raw.csv"          # stable name so resume works
    xlsx_path = f"{base}_{stamp}.xlsx"    # timestamped so nothing is overwritten

    # speakers.csv adds columns (l1, state, age_band...) - they must be in the
    # header or DictWriter silently drops them
    meta_cols = []
    for row in meta.values():
        for k in row:
            if k != "speaker_id" and k not in COLUMNS and k not in meta_cols:
                meta_cols.append(k)
    fieldnames = COLUMNS + meta_cols
    if meta_cols:
        print(f"speaker metadata columns: {', '.join(meta_cols)}")

    done = load_done(csv_path)
    todo = [p for p in plan if (p[0]["name"], p[1]["rel_path"], p[2]) not in done]
    print(f"\nplan: {len(plan)} runs, {len(done)} already done, {len(todo)} to go")
    print(f"order: shuffled, seed {args.seed}")
    print(f"raw log: {csv_path}  (append-only, safe to resume)")

    if args.dry_run:
        print("\nfirst 10 runs:")
        for m, r, rep in todo[:10]:
            print(f"  {m['name']:20s} {r['rel_path']} rep{rep}")
        est = len(todo) * (args.rest * 0.5 + 8) / 3600
        print(f"\nvery rough estimate: {est:.1f} h (assumes ~8s/run + partial cooldown)")
        return

    if args.cool_ceiling < args.cool_to:
        sys.exit(f"--cool-ceiling ({args.cool_ceiling:.0f}C) must be >= "
                 f"--cool-to ({args.cool_to:.0f}C)")

    if not args.soft_cool:
        floor = measure_idle_floor() if args.calibrate else None
        if floor is not None and args.cool_ceiling < floor + 1:
            print(f"\n!! --cool-ceiling {args.cool_ceiling:.0f}C is at or below "
                  f"the idle floor ({floor:.0f}C).\n"
                  f"   The board can never get there, so every run would hit "
                  f"the timeout and stop the sweep.\n"
                  f"   Use --cool-ceiling {floor + 4:.0f} or higher "
                  f"(and --cool-to {floor + 2:.0f}).")
            sys.exit(1)
        if floor is not None and args.cool_to < floor + 1:
            print(f"\n   note: --cool-to {args.cool_to:.0f}C is below the idle "
                  f"floor ({floor:.0f}C), so the {args.rest:.0f}s grace period "
                  f"will apply to every run and\n"
                  f"   the {args.cool_ceiling:.0f}C ceiling becomes the real "
                  f"gate. That is fine, just slower.")
        print(f"\nthermal gate: run at <={args.cool_to:.0f}C, or "
              f"<={args.cool_ceiling:.0f}C after {args.rest:.0f}s. "
              f"Never above {args.cool_ceiling:.0f}C.")

    if cpu_governor() not in (None, "performance"):
        print(f"!! CPU governor is {cpu_governor()!r}, not 'performance'. "
              f"Frequency scaling adds variance you'll misread as model effects.\n"
              f"   sudo cpufreq-set -g performance   (or echo to scaling_governor)")

    tmpdir = tempfile.mkdtemp(prefix="sttbench_")
    started = time.time()
    try:
        if args.warmup:
            for m in models:
                try:
                    wav, _ = ensure_16k_mono(recs[0]["path"], tmpdir)
                    print(f"warm-up {m['name']} (discarded)...")
                    transcribe(m, wav)
                except Exception as e:
                    print(f"  warm-up failed: {e}")

        for i, (m, rec, rep) in enumerate(todo, 1):
            try:
                wav, dur = ensure_16k_mono(rec["path"], tmpdir)
            except Exception as e:
                print(f"  {rec['rel_path']}: conversion failed ({e})")
                continue

            cool_sec, cool_reason = 0.0, "first_run"
            if i > 1:
                cool_sec, cool_reason = cooldown(
                    args.cool_to, args.cool_ceiling, args.rest,
                    args.cool_timeout, strict=not args.soft_cool)

            temp_start = soc_temp()
            text, load_sec, infer_sec, wall, stats, rc = transcribe(m, wav)
            temp_end = soc_temp()

            ref_n = normalise(rec["reference"])
            hyp_n = normalise(text)
            sp = meta.get(rec["speaker_id"], {})

            row = {
                "speaker_id": rec["speaker_id"],
                "gender": rec["gender"],
                "cmd_index": rec["cmd_index"],
                "take": rec["take"],
                "model": m["name"],
                "model_type": m["type"],
                "repeat": rep,
                "audio_sec": round(dur, 2),
                "load_sec": round(load_sec, 3) if load_sec is not None else None,
                "infer_sec": round(infer_sec, 3) if infer_sec is not None else None,
                "wall_sec": round(wall, 3),
                "rtf": round(infer_sec / dur, 3) if (infer_sec and dur) else None,
                "wer": _r(error_rate(ref_n, hyp_n, "word")),
                "cer": _r(error_rate(ref_n, hyp_n, "char")),
                "cter": _r(critical_token_error_rate(ref_n, hyp_n)),
                "exact_match": hyp_n == ref_n,
                "reference": rec["reference"],
                "hypothesis": text,
                "reference_source": rec["reference_source"],
                "match_badness": rec.get("match_badness"),
                "cool_sec": round(cool_sec, 1),
                "cool_reason": cool_reason,
                "temp_start_c": temp_start,
                "temp_end_c": temp_end,
                "rel_path": rec["rel_path"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_code": rc,
                **stats,
            }
            # carry any speakers.csv columns through untouched
            for k, v in sp.items():
                if k != "speaker_id" and k not in row:
                    row[k] = v

            append_row(csv_path, row, fieldnames)

            wer_s = f"WER {row['wer']:.1%}" if row["wer"] is not None else "WER n/a"
            mark = "" if row["exact_match"] else "  <-- MISMATCH"
            eta = (time.time() - started) / i * (len(todo) - i) / 3600
            print(f"[{i}/{len(todo)}] {m['name']:18s} {rec['rel_path']:32s} "
                  f"rep{rep} {wall:6.2f}s RTF {row['rtf']} {wer_s} "
                  f"CTER {row['cter']} {temp_start}->{temp_end}C "
                  f"ETA {eta:.1f}h{mark}")
            if not row["exact_match"]:
                print(f"      -> {text[:110]!r}")
    except CoolTimeout as e:
        print(f"\n\n!! THERMAL STOP: {e}")
        print("   Nothing was run above the ceiling. Completed runs are saved.")
        print("   Improve cooling (or raise --cool-ceiling) and rerun the same")
        print("   command - it resumes from here.")
    except KeyboardInterrupt:
        print("\ninterrupted - progress is saved, rerun the same command to resume")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\ntotal elapsed {(time.time() - started) / 60:.1f} min")
    write_excel(csv_path, xlsx_path, models, args)


def _r(x, nd=4):
    return round(x, nd) if x is not None else None


# ============================================================================
# EXCEL OUTPUT
# ============================================================================

def write_excel(csv_path, out_path, models, args):
    import pandas as pd
    from openpyxl.styles import Font, PatternFill

    if not os.path.exists(csv_path):
        print("no raw csv to summarise")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("raw csv is empty")
        return

    ordered = [c for c in COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    df = df[ordered + extra]

    # Sort for READING, not for running. Runs execute shuffled (so thermal drift
    # can't be confused with a model effect), but the sheet is laid out
    # command -> speaker -> model so it can be scanned top to bottom: all 30
    # speakers for command 1, then command 2, and so on.
    # Speakers sort numerically (male_2 before male_10), and males before
    # females within each command.
    df["_sp_num"] = df["speaker_id"].astype(str).str.extract(r"(\d+)").astype(float)
    df["_sp_grp"] = (df["gender"].astype(str) != "male").astype(int)
    df = (df.sort_values(["cmd_index", "_sp_grp", "_sp_num", "model", "repeat"])
            .drop(columns=["_sp_num", "_sp_grp"])
            .reset_index(drop=True))

    # ---- per model: central tendency AND spread (reproducibility) -----------
    g = df.groupby("model")
    summary = pd.DataFrame({
        "runs": g.size(),
        "wer_mean": g["wer"].mean(),
        "wer_std": g["wer"].std(),
        "cter_mean": g["cter"].mean(),
        "exact_match_rate": g["exact_match"].mean(),
        "infer_mean_sec": g["infer_sec"].mean(),
        "infer_std_sec": g["infer_sec"].std(),
        "infer_p95_sec": g["infer_sec"].quantile(0.95),
        "load_mean_sec": g["load_sec"].mean(),
        "rtf_mean": g["rtf"].mean(),
        "mem_peak_mb": g["mem_peak_mb"].max(),
        "cpu_mean_pct": g["cpu_mean_pct"].mean(),
        "temp_peak_c": g["temp_peak_c"].max(),
        "throttled_runs": g["throttled_during"].sum(),
        "temp_start_mean_c": g["temp_start_c"].mean(),
        "temp_start_std_c": g["temp_start_c"].std(),
        "cool_mean_sec": g["cool_sec"].mean() if "cool_sec" in df else None,
        "cool_at_target_rate": (g["cool_reason"].apply(lambda s: (s == "target").mean())
                                if "cool_reason" in df else None),
    }).reset_index()
    # CV is the reproducibility number: comparable across models with very
    # different absolute runtimes, unlike raw std.
    summary["infer_cv"] = (summary.infer_std_sec / summary.infer_mean_sec).round(4)
    summary = summary.round(4).sort_values("wer_mean")

    # ---- per speaker x model: the unit of analysis for fairness -------------
    # One row per speaker per model. Utterances from the same speaker are NOT
    # independent, so any group comparison must be done on these means, not on
    # the raw utterance rows.
    keep = ["speaker_id", "gender", "model"]
    per_speaker = df.groupby(keep).agg(
        n=("wer", "size"),
        wer=("wer", "mean"),
        cter=("cter", "mean"),
        exact_match_rate=("exact_match", "mean"),
        infer_sec=("infer_sec", "mean"),
    ).reset_index()
    per_speaker["_n"] = per_speaker["speaker_id"].str.extract(r"(\d+)").astype(float)
    per_speaker["_g"] = (per_speaker["gender"] != "male").astype(int)
    per_speaker = (per_speaker.sort_values(["model", "_g", "_n"])
                   .drop(columns=["_n", "_g"]).reset_index(drop=True).round(4))

    # ---- spread across speakers: the compression-fairness headline ----------
    sp = per_speaker.groupby("model")["wer"]
    spread = pd.DataFrame({
        "wer_mean": sp.mean(),
        "wer_std_across_speakers": sp.std(),
        "wer_best_speaker": sp.min(),
        "wer_worst_speaker": sp.max(),
        "wer_p90_speaker": sp.quantile(0.90),
    }).reset_index().round(4)
    spread["worst_minus_best"] = (spread.wer_worst_speaker - spread.wer_best_speaker).round(4)

    # ---- gender, computed correctly on speaker means ------------------------
    gender = per_speaker.groupby(["model", "gender"]).agg(
        speakers=("speaker_id", "nunique"),
        wer_mean=("wer", "mean"),
        wer_std=("wer", "std"),
        cter_mean=("cter", "mean"),
    ).reset_index().round(4)

    # ---- per command: length / complexity effect ----------------------------
    per_cmd = df.groupby("cmd_index").agg(
        reference=("reference", "first"),
        audio_sec=("audio_sec", "mean"),
        ref_words=("reference", lambda s: len(normalise(s.iloc[0]).split())),
        wer_mean=("wer", "mean"),
        cter_mean=("cter", "mean"),
        exact_match_rate=("exact_match", "mean"),
        mean_infer_sec=("infer_sec", "mean"),
    ).reset_index().round(4)
    by_model = df.pivot_table(index="cmd_index", columns="model",
                              values="wer", aggfunc="mean").round(4)
    by_model.columns = [f"wer_{c}" for c in by_model.columns]
    per_cmd = per_cmd.merge(by_model.reset_index(), on="cmd_index", how="left")

    # ---- totals: total time to get through the whole command set -----------
    # One row per speaker x model, plus a TOTAL row per model. This is the
    # "how long does one full pass of the 20 commands take" number.
    totals = df.groupby(["model", "speaker_id", "gender"]).agg(
        commands=("cmd_index", "nunique"),
        runs=("wall_sec", "size"),
        total_infer_sec=("infer_sec", "sum"),
        total_wall_sec=("wall_sec", "sum"),
        mean_infer_sec=("infer_sec", "mean"),
        wer_mean=("wer", "mean"),
    ).reset_index().round(3)

    grand = df.groupby("model").agg(
        commands=("cmd_index", "nunique"),
        runs=("wall_sec", "size"),
        total_infer_sec=("infer_sec", "sum"),
        total_wall_sec=("wall_sec", "sum"),
        mean_infer_sec=("infer_sec", "mean"),
        wer_mean=("wer", "mean"),
    ).reset_index().round(3)
    grand["speaker_id"] = "ALL SPEAKERS"
    grand["gender"] = "-"
    totals = pd.concat([totals, grand[totals.columns]], ignore_index=True)

    env = pd.DataFrame(list({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": " ".join(os.uname()),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cpu_governor": cpu_governor(),
        "whisper_threads": WHISPER_THREADS,
        "whisper_commit": _git_commit(os.path.dirname(WHISPER_BIN)),
        "vosk_version": _pkg_version("vosk"),
        "shuffle_seed": args.seed,
        "rest_max_sec": args.rest,
        "cool_to_c": args.cool_to,
        "repeats": args.repeats,
        "repeat_cmds": args.repeat_cmds or "all",
        "models": "; ".join(f"{m['name']}={sha256_of(m['path'], limit_mb=64)}"
                            for m in models),
        "temp_at_end_c": soc_temp(),
        "note": "infer_sec excludes model load; load_sec is reported separately. "
                "Run order was shuffled with the seed above so thermal drift is "
                "not confounded with model. Group comparisons must use the "
                "per_speaker sheet, not raw utterance rows.",
    }.items()), columns=["key", "value"])

    red_fill = PatternFill("solid", fgColor="FFC7CE")
    red_font = Font(color="9C0006")

    # a TOTAL row so the per-command sheet foots to the full-set time
    total_row = {c: None for c in per_cmd.columns}
    total_row["cmd_index"] = "TOTAL"
    total_row["reference"] = f"All {per_cmd['cmd_index'].nunique()} commands"
    for c in ("audio_sec", "ref_words", "mean_infer_sec"):
        if c in per_cmd:
            total_row[c] = round(per_cmd[c].sum(), 3)
    for c in ("wer_mean", "cter_mean", "exact_match_rate"):
        if c in per_cmd:
            total_row[c] = round(per_cmd[c].mean(), 4)
    per_cmd = pd.concat([per_cmd, pd.DataFrame([total_row])], ignore_index=True)

    sheets = (("results", df), ("summary", summary), ("totals", totals),
              ("per_speaker", per_speaker), ("speaker_spread", spread),
              ("gender", gender), ("per_command", per_cmd), ("environment", env))

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for name, raw_frame in sheets:
            frame = prettify(raw_frame)
            frame.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for i, col in enumerate(frame.columns, start=1):
                # an all-NaN column (Pi-only temp fields when run on the laptop)
                # makes .max() return NaN, which int() refuses
                longest = frame[col].astype(str).str.len().max() if len(frame) else 10
                longest = 10 if longest != longest or longest is None else int(longest)
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
                    min(max(len(str(col)) + 2, longest + 2), 55)

        ws = xw.sheets["results"]
        cols = {c: i + 1 for i, c in enumerate(df.columns)}
        for r, ok in enumerate(df["exact_match"], start=2):
            if ok in (False, 0, "False"):
                for nm in ("hypothesis", "wer", "cer", "cter", "exact_match"):
                    if nm in cols:
                        cell = ws.cell(row=r, column=cols[nm])
                        cell.fill = red_fill
                        cell.font = red_font

    n_bad = int((df["exact_match"].astype(str) == "False").sum())
    print(f"wrote {out_path}  ({n_bad} mismatched transcript(s) highlighted)")


def _git_commit(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--vosk-worker":
        vosk_worker(sys.argv[2], sys.argv[3])
    else:
        main()
