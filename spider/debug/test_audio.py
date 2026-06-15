import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Config
CHUNK = 2048
RATE = 16000
MIN_FREQ = 80          # hides the low-frequency rumble (the constant 55 Hz)

# Audio setup
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

# Frequency axis for the FFT bins
freqs = np.fft.rfftfreq(CHUNK, d=1.0 / RATE)
window = np.hanning(CHUNK)
times = np.arange(CHUNK) / RATE  # time axis for waveform

# Set up the figure: top = waveform, bottom = spectrum
fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(10, 6))

line_wave, = ax_wave.plot(times, np.zeros(CHUNK))
ax_wave.set_ylim(-32768, 32767)
ax_wave.set_xlim(0, CHUNK / RATE)
ax_wave.set_title("Waveform (amplitude)")
ax_wave.set_xlabel("Time (s)")
ax_wave.set_ylabel("Amplitude")
ax_wave.grid(alpha=0.3)

line_spec, = ax_spec.plot(freqs, np.zeros(len(freqs)))
ax_spec.set_xlim(0, RATE / 2)        # 0 .. 8000 Hz
ax_spec.set_ylim(0, 2_000_000)       # tweak if bars are too tall/short
ax_spec.set_title("Spectrum (frequency)")
ax_spec.set_xlabel("Frequency (Hz)")
ax_spec.set_ylabel("Magnitude")
ax_spec.grid(alpha=0.3)

plt.tight_layout()


def update(frame):
    data = stream.read(CHUNK, exception_on_overflow=False)
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)

    # waveform
    line_wave.set_ydata(arr)

    # spectrum
    spectrum = np.abs(np.fft.rfft(arr * window))
    spectrum[freqs < MIN_FREQ] = 0   # cut the rumble
    line_spec.set_ydata(spectrum)

    return line_wave, line_spec


ani = FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)

try:
    plt.show()
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
