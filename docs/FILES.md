# Project file guide

## Robot control and hardware testing

These files are for running or testing the hexapod. They are not evaluation
scripts and require Raspberry Pi audio hardware plus the Freenove robot server.

| File | Purpose |
| --- | --- |
| [`spider/robot/voice_controller.py`](../spider/robot/voice_controller.py) | Main wake-word, speech transcription, command parsing, and movement controller |
| [`spider/robot/vosk_voice_controller.py`](../spider/robot/vosk_voice_controller.py) | Vosk-based alternative voice controller |
| [`spider/robot/wakeword_transcriber_demo.py`](../spider/robot/wakeword_transcriber_demo.py) | Simple wake-word and Whisper transcription demo |
| [`spider/robot/wakeword_listener.py`](../spider/robot/wakeword_listener.py) | Minimal wake-word detection listener |
| [`spider/robot/command_parser_demo.py`](../spider/robot/command_parser_demo.py) | Earlier command parser and movement-routing demo |
| [`spider/robot/dance_command_demo.py`](../spider/robot/dance_command_demo.py) | Detects a dance command and runs the dance movement |
| [`spider/robot/runtime_config.py`](../spider/robot/runtime_config.py) | Repository-relative paths and environment-variable overrides |
| [`spider/robot/movements/`](../spider/robot/movements/) | Forward, backward, turning, rest, dance, and interaction commands |
| [`spider/robot/debug/`](../spider/robot/debug/) | Audio capture and wake-word visualization diagnostics |

## Evaluation and research tools

These scripts process datasets or measure speech-recognition performance. They
do not directly control the hexapod.

| File | Purpose |
| --- | --- |
| [`tools/benchmark/stt_benchmark.py`](../tools/benchmark/stt_benchmark.py) | Multi-speaker Whisper/Vosk accuracy, latency, thermal, and resource benchmark |
| [`tools/benchmark/run_all.py`](../tools/benchmark/run_all.py) | Resumable benchmark runner with thermal checks |
| [`tools/benchmark/net_latency_bench.py`](../tools/benchmark/net_latency_bench.py) | Network/cloud speech latency experiment |
| [`tools/preprocessing/preprocess.py`](../tools/preprocessing/preprocess.py) | ffmpeg preprocessing conditions and timing logs |
| [`tools/preprocessing/preprocess_amplitude.py`](../tools/preprocessing/preprocess_amplitude.py) | Adaptive amplitude normalization experiment |
| [`tools/preprocessing/preprocess_denoise.py`](../tools/preprocessing/preprocess_denoise.py) | Denoising and resampling experiment |
| [`tools/preprocessing/run_preprocessing_all_models.py`](../tools/preprocessing/run_preprocessing_all_models.py) | Runs complete preprocessing benchmark sweeps |
| [`tools/preprocessing/repair.py`](../tools/preprocessing/repair.py) | Repairs failed preprocessing outputs |
| [`tools/preprocessing/retry_failed_preprocessing.py`](../tools/preprocessing/retry_failed_preprocessing.py) | Fallback repair strategies for time-stretch failures |
| [`tools/dataset/extract_audio.py`](../tools/dataset/extract_audio.py) | Extracts supported audio files from ZIP archives |
| [`tools/legacy/legacy_stt_benchmark.py`](../tools/legacy/legacy_stt_benchmark.py) | Older benchmark design retained for historical comparison |

## Configuration and setup

| File | Purpose |
| --- | --- |
| [`requirements-robot.txt`](../requirements-robot.txt) | Robot runtime and hardware-related Python packages |
| [`requirements-benchmark.txt`](../requirements-benchmark.txt) | Dataset, preprocessing, and evaluation packages |
| [`scripts/install_dependencies_and_models.py`](../scripts/install_dependencies_and_models.py) | Installs both requirement sets and downloads configured speech models |
| [`README.md`](../README.md) | Setup, layout, commands, and publication guidance |

## Local-only assets

The following are intentionally not committed:

- `models/` downloaded model files and Whisper build output
- `Audio/`, `recordings/`, and `variants/`
- `results/`, CSV files, Excel workbooks, and logs
- Python virtual environments and caches
