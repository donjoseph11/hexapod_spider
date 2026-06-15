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

# Fetch base feature models (melspectrogram + embedding) if not already present
download_models()
# --- Paths ---
WAKEWORD_MODEL = "hey_spider_modified_openwakeword/hey_spider.tflite"
WHISPER_BIN = "/home/pi/heyspider/models/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/pi/heyspider/models/whisper.cpp/models/ggml-tiny.en.bin"
NOD_SCRIPT = "/home/pi/heyspider/wake_word/Movements/Head/Nod.py"
DANCE_SCRIPT = "/home/pi/heyspider/spider/Movements/Dance/dance3_lite.py"
MOVE_SCRIPT = "/home/pi/heyspider/spider/Movements/move.py"   # walking script (parses direction+number)
 
# --- Keyword that triggers the dance ---
DANCE = "dance"
FORWARD = "forward"
BACKWARD = "backward"
LEFT = "left"
RIGHT = "right"
REST = "rest"
 
# --- Audio ---
CHUNK = 1280              # 80ms at 16kHz, required by openwakeword
RATE = 16000
SAMPLE_WIDTH = 2          # int16 = 2 bytes
 
# --- Wake word ---
WAKE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 2.0
 
# --- Command recording (after wake word) ---
SILENCE_RMS_THRESHOLD = 500    # below this = silence; tune for your mic/room
SILENCE_DURATION = 1.0         # stop after this many seconds of silence
MAX_COMMAND_SECONDS = 10       # hard cap on recording length
MIN_COMMAND_SECONDS = 0.5      # ignore blips shorter than this
 
# --- Sanity checks (wake word + whisper required; scripts only warned about) ---
for path in (WAKEWORD_MODEL, WHISPER_BIN, WHISPER_MODEL):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
for path in (NOD_SCRIPT, DANCE_SCRIPT, MOVE_SCRIPT):
    if not os.path.exists(path):
        print(f"WARNING: script not found (commands using it will be skipped): {path}")
 
# --- Load wake word model ---
model = Model(
    wakeword_models=[WAKEWORD_MODEL],
    inference_framework="tflite",
)
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
            

# ======================  COMMAND PARSING  ======================
# Turn what you said into (direction, number, unit), e.g.
#   "move forward 10 feet"   -> ("forward", 10.0, "feet")
#   "go back two meters"     -> ("backward", 2.0, "meters")
#   "turn left"              -> ("left", None, None)
#   "rest"                   -> ("stop", None, None)
 
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
    left / right / stop, or None if no movement word was heard."""
    tokens = re.findall(r"[a-z]+", text.lower())
    has = lambda *ws: any(w in tokens for w in ws)
 
    if has("stop", "halt", "freeze", "stay", "rest", "relax", "sit"):
        direction = "stop"
    elif has("back", "backward", "backwards", "reverse"):
        direction = "backward"
    elif has("forward", "ahead", "straight", "front"):
        direction = "forward"
    elif has("left"):
        direction = "left"
    elif has("right"):
        direction = "right"
    else:
        direction = None
 
    return (direction, _number(tokens, text.lower()), _unit(tokens))
 
 
def run_dance(transcribed_text):
    print(f"Running dance: {DANCE_SCRIPT}")
    try:
        result = subprocess.run([sys.executable, DANCE_SCRIPT, transcribed_text], timeout=60)
        print(f"  dance exited with code {result.returncode}")
    except Exception as e:
        print(f"  error running dance: {e}")
 
 
 
def run_move(direction, number, unit):
    """Send the parsed pieces to move.py as parameters."""
    # build argv from the parts that exist: e.g. ["forward", "10", "feet"]
    args = [direction]
    if number is not None:
        args.append(f"{number:g}")
    if unit is not None:
        args.append(unit)
    print(f"Running move: {direction} number={number} unit={unit}")
    try:
        result = subprocess.run([sys.executable, MOVE_SCRIPT, *args], timeout=120)
        print(f"  move exited with code {result.returncode}")
    except Exception as e:
        print(f"  error running move: {e}")
 
 
# ======================  MAIN LOOP  ======================
print("Listening for wake word... (Ctrl+C to stop)")
last_detection = 0.0
 
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
 
                command_audio = record_command()
                if command_audio:
                    print("Transcribing...")
                    text = transcribe(command_audio)
                    print(f"You said: {text!r}")
 
                    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
                    direction, number, unit = parse_command(text)
 
                    if KEYWORD in cleaned.split():
                        print(f"  keyword '{KEYWORD}' detected!")
                        run_dance(text)
                    elif direction == FORWARD:
                        run_move(direction, number, unit)
                    elif direction == BACKWARD:
                        run_move(direction, number, unit)
                    elif direction == LEFT:
                        run_move(direction, number, unit)
                    elif direction == RIGHT:
                        run_move(direction, number, unit)
                    elif direction == REST:
                        run_move(direction, number, unit)
                    else:
                        print("  no known command, ignoring")
 
                model.reset()
                print("\nListening for wake word...")
 
except KeyboardInterrupt:
    print("\nStopping.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
 
 
