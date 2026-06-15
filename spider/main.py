import os
import time
import wave
import subprocess
import tempfile
import collections

import pyaudio
import numpy as np
from openwakeword.model import Model

# --- Paths ---
#WAKEWORD_MODEL = "hey_spider_openwakeword/hey_spider.tflite"
WAKEWORD_MODEL = "hey_spider_modified_openwakeword/hey_spider.tflite"
WHISPER_BIN = "/home/pi/whisper_cpp/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/pi/whisper_cpp/whisper.cpp/models/ggml-tiny.en.bin"

# --- Audio ---
CHUNK = 1280            # 80ms at 16kHz, required by openwakeword
RATE = 16000
SAMPLE_WIDTH = 2        # int16 = 2 bytes

# --- Wake word ---
WAKE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 2.0

# --- Command recording (after wake word) ---
PRE_ROLL_SECONDS = 0.5        # keep this much audio from before silence ended
SILENCE_RMS_THRESHOLD = 500   # below this, considered silence (tune for your mic)
SILENCE_DURATION = 1.0        # stop after this many seconds of continuous silence
MAX_COMMAND_SECONDS = 10      # hard cap, prevents runaway recordings
MIN_COMMAND_SECONDS = 0.5     # ignore noise blips shorter than this

# --- Sanity checks ---
for path in (WAKEWORD_MODEL, WHISPER_BIN, WHISPER_MODEL):
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


def rms(audio_array):
    """Root mean square loudness of an int16 audio buffer."""
    # cast to int32 first to avoid overflow on squaring
    return int(np.sqrt(np.mean(audio_array.astype(np.int32) ** 2)))


def record_command():
    """Record audio after the wake word until silence is detected.
    Returns raw int16 bytes, or None if nothing meaningful was captured."""
    print("Listening for command...")
    frames = []
    silence_chunks = 0
    speech_chunks = 0
    silence_chunks_needed = int(SILENCE_DURATION * RATE / CHUNK)
    max_chunks = int(MAX_COMMAND_SECONDS * RATE / CHUNK)
    started_speaking = False

    for _ in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        arr = np.frombuffer(data, dtype=np.int16)
        frames.append(data)

        level = rms(arr)
        if level > SILENCE_RMS_THRESHOLD:
            silence_chunks = 0
            speech_chunks += 1
            if not started_speaking:
                started_speaking = True
                print("  (heard you)")
        else:
            silence_chunks += 1
            if started_speaking and silence_chunks >= silence_chunks_needed:
                break

    if not started_speaking:
        print("  (no speech detected)")
        return None

    duration = speech_chunks * CHUNK / RATE
    if duration < MIN_COMMAND_SECONDS:
        print("  (too short, ignoring)")
        return None

    return b"".join(frames)
    
def transcribe(pcm_bytes):
    """Save audio to a temp WAV and run whisper.cpp on it. Returns text."""
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
            capture_output=True,
            text=True,
            timeout=30,
        )
        # whisper-cli prints the transcript to stdout
        return result.stdout.strip()
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


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
                print(f"\n=== WAKE WORD: {wakeword} (score={score:.3f}) ===")
                last_detection = now
                level = rms(arr)
                print(f"score={score:.3f}  level={level}")
                #command_audio = record_command()
                command_audio = None
                if command_audio:
                    print("Transcribing...")
                    text = transcribe(command_audio)
                    if text:
                        print(f"You said: {text}")
                    else:
                        print("(no transcription)")

                # Reset wake word model state so the command audio
                # doesn't influence the next prediction
                model.reset()
                print("\nListening for wake word...")

except KeyboardInterrupt:
    print("\nStopping.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
