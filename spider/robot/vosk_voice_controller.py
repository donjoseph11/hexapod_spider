import os
import sys
import re
import time
import wave
import subprocess
import tempfile

import pyaudio
import numpy as np
from openwakeword.model import Model
from openwakeword.utils import download_models
from collections import deque
import json
from vosk import Model as VoskModel, KaldiRecognizer
from runtime_config import WAKEWORD_MODEL, VOSK_MODEL, movement
# Fetch base feature models (melspectrogram + embedding) if not already present
download_models()
# ============================================================================
# HOW THIS WORKS
#   1. Listens for the wake word "hey spider".
#   2. Nods the head (nod.py) to show it's listening.
#   3. Records what you say, transcribes it with whisper.cpp.
#   4. parse_command() pulls out a direction + number + unit.
#   5. Routes to the matching script:
#        "dance"                  -> dance3_lite.py
#        "forward 10 feet"        -> forward.py 60     (10 ft x CYCLES_PER_FOOT)
#        "back 5 steps"           -> backward.py 5
#        "turn left 90 degrees"   -> left.py 8
#        "turn right"             -> right.py 8        (defaults to ~90 deg)
#        "rest" / "sit" / "stop"  -> rest.py
#
# PUT THE FIVE MOVEMENT SCRIPTS in the MOVEMENTS folder below (or change paths).
# CALIBRATE the two numbers under "CALIBRATION" so feet/degrees come out right.
# ============================================================================

# --- Paths ---
VOSK_MODEL_PATH = VOSK_MODEL

NOD_SCRIPT = movement("Interactions/nod.py")
DANCE_SCRIPT = movement("Dance/dance3_lite.py")
FORWARD_SCRIPT = movement("Forward/forward.py")
BACKWARD_SCRIPT = movement("Backward/backward.py")
LEFT_SCRIPT = movement("Left/left.py")
RIGHT_SCRIPT = movement("Right/right.py")
REST_SCRIPT = movement("Rest/rest.py")
TURNAROUND_SCRIPT = movement("turn_around/turn_around.py")
HI_SCRIPT = movement("Interactions/say_hi.py")
NOPE_SCRIPT = movement("Interactions/say_no.py")

# --- Command names (what parse_command returns) ---
DANCE    = "dance"
FORWARD  = "forward"
BACKWARD = "backward"
LEFT     = "left"
RIGHT    = "right"
REST     = "rest"

# Inverse of each move, used by reset to retrace the path.
_OPPOSITE = {
    FORWARD:  BACKWARD,
    BACKWARD: FORWARD,
    LEFT:     RIGHT,
    RIGHT:    LEFT,
}
# Map each movement command to the script that performs it.
MOVEMENT_SCRIPTS = {
    FORWARD:  FORWARD_SCRIPT,
    BACKWARD: BACKWARD_SCRIPT,
    LEFT:     LEFT_SCRIPT,
    RIGHT:    RIGHT_SCRIPT,
    REST:     REST_SCRIPT,
}

MAX_HISTORY = 5     # reset only undoes the last 5 moves

# --- CALIBRATION (measure these once on your robot) ---
CYCLES_PER_FOOT = 6.0    # strides to travel 1 foot   (run forward.py, measure, adjust)
CYCLES_PER_90   = 3      # strides to turn 90 degrees  (run left.py, measure, adjust)
DEFAULT_FWD_STRIDES  = 5 # "forward" with no number
DEFAULT_TURN_STRIDES = 3 # "turn left" with no number (~90 deg)
METERS_TO_FEET = 3.28084

# --- Audio ---
CHUNK = 1280              # 80ms at 16kHz, required by openwakeword
RATE = 16000
SAMPLE_WIDTH = 2          # int16 = 2 bytes

# --- Wake word ---
WAKE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 2.0

# --- Command recording (after wake word) ---
SILENCE_RMS_THRESHOLD = 400    # below this = silence; tune for your mic/room
SILENCE_DURATION = 1.0         # stop after this many seconds of silence
MAX_COMMAND_SECONDS = 10       # hard cap on recording length
MIN_COMMAND_SECONDS = 0.5      # ignore blips shorter than this

# --- Sanity checks (wake word + whisper required; scripts only warned about) ---
for path in (WAKEWORD_MODEL, VOSK_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
for path in (NOD_SCRIPT, DANCE_SCRIPT, *MOVEMENT_SCRIPTS.values()):
    if not os.path.exists(path):
        print(f"WARNING: script not found (commands using it will be skipped): {path}")


# --- Load wake word model ---
model = Model(
    wakeword_models=[WAKEWORD_MODEL],
    inference_framework="tflite",
)

print("Loading Vosk model...")
vosk_model = VoskModel(VOSK_MODEL_PATH)
print("Vosk model loaded.")

print(f"Loaded wake word model: {list(model.models.keys())}")

# --- Audio stream ---
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

def rms(arr):
    """Loudness of an int16 buffer (cast up first to avoid overflow)."""
    return int(np.sqrt(np.mean(arr.astype(np.int32) ** 2)))


def record_command():
    """Record after wake word until silence. Returns raw int16 bytes or None."""
    print("Listening for command...")
    frames = []
    silence_chunks = 0
    speech_chunks = 0
    silence_chunks_needed = int(SILENCE_DURATION * RATE / CHUNK)
    max_chunks = int(MAX_COMMAND_SECONDS * RATE / CHUNK)
    started = False

    for _ in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        arr = np.frombuffer(data, dtype=np.int16)
        frames.append(data)

        if rms(arr) > SILENCE_RMS_THRESHOLD:
            silence_chunks = 0
            speech_chunks += 1
            if not started:
                started = True
                print("  (heard you)")
        else:
            silence_chunks += 1
            if started and silence_chunks >= silence_chunks_needed:
                break

    if not started:
        print("  (no speech)")
        return None
    if speech_chunks * CHUNK / RATE < MIN_COMMAND_SECONDS:
        print("  (too short)")
        return None
    return b"".join(frames)


def transcribe(pcm_bytes):
    """Run Vosk on raw int16 PCM bytes. Returns transcribed text."""
    rec = KaldiRecognizer(vosk_model, RATE)
    rec.SetWords(False)

    # Feed in chunks (Vosk handles arbitrary sizes, but chunking avoids
    # large single AcceptWaveform calls on bigger recordings)
    chunk_size = 4000  # bytes
    for i in range(0, len(pcm_bytes), chunk_size):
        rec.AcceptWaveform(pcm_bytes[i:i + chunk_size])

    result = json.loads(rec.FinalResult())
    text = result.get("text", "").strip()
    print("  vosk result:", repr(text))
    return text

'''
def transcribe(pcm_bytes):
    """Save audio to a temp WAV and run whisper.cpp. Returns transcribed text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(RATE)
            wf.writeframes(pcm_bytes)

        result = subprocess.run(
            [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav_path, "-nt", "-l", "en"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
'''


# ======================  COMMAND PARSING  ======================
# Turn what you said into (direction, number, unit), e.g.
#   "move forward 10 feet"   -> ("forward", 10.0, "feet")
#   "go back two meters"     -> ("backward", 2.0, "meters")
#   "turn left"              -> ("left", None, None)
#   "rest"                   -> ("rest", None, None)

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _number(tokens, raw):
    """Get a number from the command, digits ('10') or words ('ten')."""
    digits = re.findall(r"\d+\.?\d*", raw)
    if digits:
        return float(digits[0])
    value, found = 0, False
    for w in tokens:
        if w in _ONES:
            value += _ONES[w]; found = True
        elif w in _TENS:
            value += _TENS[w]; found = True
        elif w == "hundred":
            value = (value or 1) * 100; found = True
    return float(value) if found else None


def _unit(tokens):
    for w in tokens:
        if w in ("feet", "foot", "ft"):                 return "feet"
        if w in ("meter", "meters", "metre", "metres"): return "meters"
        if w == "m":                                    return "meters"
        if w in ("step", "steps", "stride", "strides"): return "steps"
        if w in ("degree", "degrees", "deg"):           return "degrees"
        if w in ("second", "seconds", "sec", "secs"):   return "seconds"
    return None


def parse_command(text):
    """Return (direction, number, unit). direction is forward / backward /
    left / right / rest, or None if no movement word was heard."""
    tokens = re.findall(r"[a-z]+", text.lower())
    has = lambda *ws: any(w in tokens for w in ws)

    if has("stop", "halt", "freeze", "stay", "rest", "relax", "sit"):
        direction = REST
    elif has("back", "backward", "backwards", "reverse"):
        direction = BACKWARD
    elif has("forward", "ahead", "straight", "front"):
        direction = FORWARD
    elif has("left"):
        direction = LEFT
    elif has("right"):
        direction = RIGHT
    else:
        direction = None

    return (direction, _number(tokens, text.lower()), _unit(tokens))


def to_strides(direction, number, unit):
    """Convert the spoken amount into a number of gait strides for the script.
    Returns None for rest (no amount needed)."""
    if direction == REST:
        return None

    if direction in (FORWARD, BACKWARD):
        if number is None:
            return DEFAULT_FWD_STRIDES
        if unit == "meters":
            return max(1, round(number * METERS_TO_FEET * CYCLES_PER_FOOT))
        if unit == "feet":
            return max(1, round(number * CYCLES_PER_FOOT))
        return max(1, round(number))          # steps or a bare number

    if direction in (LEFT, RIGHT):
        if number is None:
            return DEFAULT_TURN_STRIDES
        if unit in (None, "degrees"):
            return max(1, round(number / 90 * CYCLES_PER_90))
        return max(1, round(number))          # steps or a bare number

    return None

def run_nope():
    """Shake head to show the command wasn't understood."""
    if not os.path.exists(NOPE_SCRIPT):
        print(f"  Nope script not found: {NOPE_SCRIPT}")
        return
    print("  (didn't understand - shaking head)")
    try:
        subprocess.run([sys.executable, NOPE_SCRIPT], timeout=30)
    except Exception as e:
        print(f"  error running nope: {e}")

def run_dance(transcribed_text):
    print(f"Running dance: {DANCE_SCRIPT}")
    try:
        result = subprocess.run([sys.executable, DANCE_SCRIPT, transcribed_text], timeout=60)
        print(f"  dance exited with code {result.returncode}")
    except Exception as e:
        print(f"  error running dance: {e}")


def run_movement(direction, number, unit):
    """Run the script for this direction, passing the stride count as a parameter."""
    script = MOVEMENT_SCRIPTS[direction]
    if not os.path.exists(script):
        print(f"  script for '{direction}' not found: {script}")
        return

    strides = to_strides(direction, number, unit)
    args = [sys.executable, script]
    if strides is not None:
        args.append(str(strides))

    print(f"Running {direction}: {os.path.basename(script)} "
          f"({'rest' if strides is None else f'{strides} strides'})")
    try:
        result = subprocess.run(args, timeout=120)
        print(f"  {direction} exited with code {result.returncode}")
    except Exception as e:
        print(f"  error running {direction}: {e}")

def run_reset():
    """Replay the recorded moves in reverse, inverted, to return to start."""
    if not move_history:
        print("  nothing to reset")
        return
    print(f"Resetting {len(move_history)} step(s)...")
    for direction, number, unit in reversed(move_history):
        opposite = _OPPOSITE.get(direction)
        if opposite is None:
            continue                       # rest/dance have no inverse
        run_movement(opposite, number, unit)
    move_history.clear()
    print("  back to start")

# words/punctuation that separate chained commands in one sentence
_SPLIT_RE = re.compile(r"\s*,\s*|\b(?:and then|after that|then|and|next)\b")

def split_commands(text):
    """'forward 2 feet and then turn right' -> ['forward 2 feet', 'turn right']."""
    parts = _SPLIT_RE.split(text.lower())
    return [p.strip() for p in parts if p.strip()]


def handle_command(segment):
    """Run one command segment. Returns True if understood, False otherwise."""
    words = re.findall(r"[a-z]+", segment.lower())
    if "reset" in words or "undo" in words:
        run_reset()
    elif "around" in words or "about" in words:        # "turn around" / "about face"
        subprocess.run([sys.executable, TURNAROUND_SCRIPT], timeout=60)
    elif "hi" in words or "hello" in words:
        subprocess.run([sys.executable, HI_SCRIPT], timeout=30)
    elif DANCE in words:
        print("  command: dance")
        run_dance(segment)
    else:
        direction, number, unit = parse_command(segment)
        if direction is not None:
            run_movement(direction, number, unit)
            if direction in _OPPOSITE:
                move_history.append((direction, number, unit))
        else:
            print(f"  no known command in: {segment!r}")
            return False
    return True

# ======================  MAIN LOOP  ======================
print("Listening for wake word... (Ctrl+C to stop)")
last_detection = 0.0
move_history = deque(maxlen=MAX_HISTORY)   # was []  -> auto-drops the oldest past 5
try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        arr = np.frombuffer(data, dtype=np.int16)

        prediction = model.predict(arr)
        now = time.time()

        for wakeword, score in prediction.items():
            if score > WAKE_THRESHOLD and (now - last_detection) > COOLDOWN_SECONDS:
                print(f"\n=== WAKE WORD: {wakeword} ({score:.2f}) ===")
                last_detection = now
                subprocess.run([sys.executable, NOD_SCRIPT], timeout=10)  # nod hello

                # discard audio captured while the head was nodding
                avail = stream.get_read_available()
                if avail > 0:
                    stream.read(avail, exception_on_overflow=False)

                command_audio = record_command()
                if command_audio:
                    print("Transcribing...")
                    text = transcribe(command_audio)
                    print(f"You said: {text!r}")

                    segments = split_commands(text)
                    if not segments:
                        print("  no known command, ignoring")
                        run_nope()
                    else:
                        print(f"  {len(segments)} command(s): {segments}")
                        understood_any = False
                        for seg in segments:
                            if handle_command(seg):
                                understood_any = True
                        if not understood_any:
                            run_nope()   # shake head if nothing made sense
                else:
                    run_nope()   # heard the wake word but no/garbled speech

                model.reset()
                print("\nListening for wake word...")

except KeyboardInterrupt:
    print("\nStopping.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
