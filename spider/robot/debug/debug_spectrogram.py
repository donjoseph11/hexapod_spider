import sys
from pathlib import Path
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from openwakeword.model import Model
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import WAKEWORD_MODEL

# Config
CHUNK = 1280            # 80ms at 16kHz
RATE = 16000
WINDOW_SECONDS = 3      # how much history to show
N_CHUNKS = int(WINDOW_SECONDS * RATE / CHUNK)

# Load wake word model
model = Model(
    wakeword_models=[WAKEWORD_MODEL],
    inference_framework="tflite",
)
wakeword_name = list(model.models.keys())[0]
print(f"Loaded: {wakeword_name}")

# Audio stream
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
spectrogram_buf = deque([np.zeros(64)] * N_CHUNKS, maxlen=N_CHUNKS)

# Plot setup
fig, (ax_wave, ax_spec, ax_score) = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle(f"Wake word debug: '{wakeword_name}'  -  speak now")

# Waveform
x_time = np.arange(N_CHUNKS * CHUNK) / RATE
line_wave, = ax_wave.plot(x_time, np.zeros(N_CHUNKS * CHUNK))
ax_wave.set_ylim(-32768, 32767)
ax_wave.set_xlim(0, WINDOW_SECONDS)
ax_wave.set_ylabel("Amplitude")
ax_wave.set_title("Waveform (mic input)")
ax_wave.grid(alpha=0.3)

# Spectrogram
im = ax_spec.imshow(
    np.zeros((64, N_CHUNKS)),
    aspect="auto",
    origin="lower",
    cmap="magma",
    vmin=-2, vmax=2,
    extent=[0, WINDOW_SECONDS, 0, 8000],
)
ax_spec.set_ylabel("Mel freq (Hz)")
ax_spec.set_title("Mel spectrogram (what the model sees)")

# Wake word score
x_chunks = np.linspace(0, WINDOW_SECONDS, N_CHUNKS)
line_score, = ax_score.plot(x_chunks, np.zeros(N_CHUNKS), color="lime", linewidth=2)
ax_score.axhline(0.5, color="orange", linestyle="--", label="threshold 0.5")
ax_score.set_ylim(0, 1.05)
ax_score.set_xlim(0, WINDOW_SECONDS)
ax_score.set_ylabel("Score")
ax_score.set_xlabel("Time (s)")
ax_score.set_title("Wake word score")
ax_score.legend(loc="upper right")
ax_score.grid(alpha=0.3)

current_score = 0.0
peak_score = 0.0


def update(_frame):
    global current_score, peak_score

    # Read one chunk
    data = stream.read(CHUNK, exception_on_overflow=False)
    arr = np.frombuffer(data, dtype=np.int16)
    waveform_buf.append(arr)

    # Predict
    prediction = model.predict(arr)
    current_score = float(prediction.get(wakeword_name, 0.0))
    peak_score = max(peak_score, current_score)
    score_buf.append(current_score)

    # Get mel features for display (last frame from model preprocessor)
    # The preprocessor exposes the rolling mel buffer
    try:
        mel = model.preprocessor.melspectrogram_buffer[-32:]  # last ~256ms
        # Take mean across the time chunks to get one column per CHUNK
        mel_col = mel.mean(axis=0) if mel.ndim > 1 else mel
        # Normalize for display
        mel_col = (mel_col - mel_col.mean()) / (mel_col.std() + 1e-6)
        if len(mel_col) >= 64:
            spectrogram_buf.append(mel_col[:64])
        else:
            padded = np.zeros(64)
            padded[:len(mel_col)] = mel_col
            spectrogram_buf.append(padded)
    except Exception:
        spectrogram_buf.append(np.zeros(64))

    # Update plots
    line_wave.set_ydata(np.concatenate(list(waveform_buf)))
    im.set_data(np.array(list(spectrogram_buf)).T)
    line_score.set_ydata(list(score_buf))

    # Update title with live score
    ax_score.set_title(f"Wake word score  |  now: {current_score:.2f}  |  peak: {peak_score:.2f}")

    return line_wave, im, line_score


ani = FuncAnimation(fig, update, interval=10, blit=False, cache_frame_data=False)

plt.tight_layout()
try:
    plt.show()
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
