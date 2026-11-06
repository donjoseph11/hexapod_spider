import os
import sys
import time
from pathlib import Path
import numpy as np
import pyaudio
import matplotlib
matplotlib.use("Agg")  # no GUI, just file output
import matplotlib.pyplot as plt
from collections import deque
from openwakeword.model import Model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import WAKEWORD_MODEL

# Config
CHUNK = 1280
RATE = 16000
THRESHOLD = 0.4            # lowered so near-misses also get captured
COOLDOWN_SECONDS = 2.0
HISTORY_SECONDS = 3        # how much past audio each snapshot shows
SAVE_DIR = str(Path(__file__).resolve().parent / "detections")

N_CHUNKS = int(HISTORY_SECONDS * RATE / CHUNK)
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
model = Model(
    wakeword_models=[WAKEWORD_MODEL],
    inference_framework="tflite",
)
wakeword_name = list(model.models.keys())[0]
print(f"Loaded: {wakeword_name}")

# Audio
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

# Rolling buffers
waveform_buf = deque([np.zeros(CHUNK, dtype=np.int16)] * N_CHUNKS, maxlen=N_CHUNKS)
score_buf = deque([0.0] * N_CHUNKS, maxlen=N_CHUNKS)


def save_snapshot(score, peak_score):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"{timestamp}_score{score:.2f}.png")

    waveform = np.concatenate(list(waveform_buf))
    scores = np.array(list(score_buf))

    # Compute a mel spectrogram for display from the captured waveform
    # Using a simple STFT-based spectrogram (no librosa dependency)
    from numpy.fft import rfft
    win_size = 512
    hop = 160
    n_frames = (len(waveform) - win_size) // hop
    window = np.hanning(win_size)
    spec = np.zeros((win_size // 2 + 1, n_frames))
    for i in range(n_frames):
        frame = waveform[i * hop : i * hop + win_size].astype(np.float32) * window
        spec[:, i] = np.abs(rfft(frame))
    spec_db = 20 * np.log10(spec + 1e-6)

    fig, (ax_wave, ax_spec, ax_score) = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f"Detection: {wakeword_name}  |  score={score:.2f}  |  peak={peak_score:.2f}")

    t_audio = np.arange(len(waveform)) / RATE
    ax_wave.plot(t_audio, waveform, linewidth=0.5)
    ax_wave.set_ylim(-32768, 32767)
    ax_wave.set_xlim(0, HISTORY_SECONDS)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Waveform")
    ax_wave.grid(alpha=0.3)

    ax_spec.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, HISTORY_SECONDS, 0, RATE / 2],
    )
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_ylim(0, 4000)  # speech lives here
    ax_spec.set_title("Spectrogram")

    t_score = np.linspace(0, HISTORY_SECONDS, len(scores))
    ax_score.plot(t_score, scores, color="green", linewidth=2)
    ax_score.axhline(THRESHOLD, color="orange", linestyle="--", label=f"threshold {THRESHOLD}")
    ax_score.set_ylim(0, 1.05)
    ax_score.set_xlim(0, HISTORY_SECONDS)
    ax_score.set_ylabel("Score")
    ax_score.set_xlabel("Time (s)")
    ax_score.set_title("Wake word score")
    ax_score.legend(loc="upper right")
    ax_score.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=80)
    plt.close()
    print(f"  saved: {filename}")


print(f"Listening... saving detections to ./{SAVE_DIR}/")
last_detection = 0.0
peak_since_last = 0.0

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        arr = np.frombuffer(data, dtype=np.int16)
        waveform_buf.append(arr)

        prediction = model.predict(arr)
        score = float(prediction.get(wakeword_name, 0.0))
        score_buf.append(score)
        peak_since_last = max(peak_since_last, score)

        now = time.time()
        if score > THRESHOLD and (now - last_detection) > COOLDOWN_SECONDS:
            print(f"DETECTION: score={score:.3f}  peak={peak_since_last:.3f}")
            save_snapshot(score, peak_since_last)
            last_detection = now
            peak_since_last = 0.0

except KeyboardInterrupt:
    print("\nStopping.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
