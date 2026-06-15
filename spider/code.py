import os
import time
import pyaudio
import numpy as np
from openwakeword.model import Model

# Config
MODEL_PATH = "hey_spider.tflite"
THRESHOLD = 0.4          # raise to ~0.7 if you get false triggers, lower to ~0.3 if it misses
COOLDOWN_SECONDS = 1.0   # ignore repeat detections for this long after a hit
CHUNK = 1280             # 80ms at 16kHz - openwakeword expects this
RATE = 16000

# Sanity check
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {os.path.abspath(MODEL_PATH)}")

# Load model
# NOTE: keyword arg name depends on openwakeword version.
# Try wakeword_models first; if it errors, swap to wakeword_model_paths or models.
model = Model(
    wakeword_models=[MODEL_PATH],
    inference_framework="tflite",
)

print(f"Loaded model. Wake words: {list(model.models.keys())}")

# Audio stream
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

print("Listening... (Ctrl+C to stop)")

last_detection = 0.0

try:
    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        prediction = model.predict(audio_array)

        now = time.time()
        for wakeword, score in prediction.items():
            #if score > 0.05:  # only print when there's any signal at all
                #print(f"  {wakeword}: {score:.3f}")
            if score > THRESHOLD and (now - last_detection) > COOLDOWN_SECONDS:
                print(f"WAKE WORD DETECTED: {wakeword} (score={score:.3f})")
                last_detection = now
except KeyboardInterrupt:
    print("\nStopping.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
