import os
import sys
import time
import wave
import subprocess
import tempfile

import pyaudio
import numpy as np
from openwakeword.model import Model
from runtime_config import WAKEWORD_MODEL, WHISPER_BIN, WHISPER_DIR, movement

# --- Paths ---
WHISPER_MODEL = WHISPER_DIR / "ggml-tiny.en.bin"
SCRIPT_TO_RUN = movement("Dance/dance3_lite.py")

# --- Keyword ---
KEYWORD = "dance"

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

# --- Sanity checks ---
for path in (WAKEWORD_MODEL, WHISPER_BIN, WHISPER_MODEL, SCRIPT_TO_RUN):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

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


def matches_keyword(text):
    """True if KEYWORD appears as a standalone word, ignoring case/punctuation."""
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return KEYWORD in cleaned.split()


def run_script(transcribed_text):
    """Run the target Python script with the same interpreter, passing the text."""
    print(f"Running {SCRIPT_TO_RUN} ...")
    try:
        result = subprocess.run(
            [sys.executable, SCRIPT_TO_RUN, transcribed_text],
            timeout=60,
        )
        print(f"  script exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        print("  script timed out")
    except Exception as e:
        print(f"  error running script: {e}")


# --- Main loop ---
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
                subprocess.run([sys.executable, movement("Interactions/nod.py")], timeout=10)
                command_audio = record_command()
                #command_audio=None
                if command_audio:
                    print("Transcribing...")
                    text = transcribe(command_audio)
                    print(f"You said: {text!r}")

                    if matches_keyword(text):
                        print(f"  keyword '{KEYWORD}' detected!")
                        run_script(text)
                    else:
                        print(f"  no '{KEYWORD}' keyword, ignoring")

                model.reset()
                print("\nListening for wake word...")

except KeyboardInterrupt:
    print("\nStopping.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
