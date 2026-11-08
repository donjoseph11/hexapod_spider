# Hey Spider

Speech-controlled Raspberry Pi hexapod experiments and benchmark tooling.

This repository contains two separate parts:

- `spider/robot/` contains code that tests or controls the hexapod. It is
  hardware-dependent and is not evaluation code.
- `tools/` contains the newer speech-to-text benchmark, dataset preparation,
  and preprocessing runners used for the thesis measurements.

Large models, recordings, benchmark outputs, and the Raspberry Pi virtual
environment are intentionally local-only. They are ignored by Git; see
the local-only assets section below.

## Repository map

| Path | Purpose | Status |
| --- | --- | --- |
| [`spider/robot/voice_controller.py`](spider/robot/voice_controller.py) | Main wake-word, transcription, command parsing, and movement routing | Primary robot test |
| [`spider/robot/wakeword_transcriber_demo.py`](spider/robot/wakeword_transcriber_demo.py) | Wake-word and transcription smoke test | Robot audio test |
| [`spider/robot/wakeword_listener.py`](spider/robot/wakeword_listener.py) | Wake-word-only listener | Minimal robot audio test |
| [`spider/robot/command_parser_demo.py`](spider/robot/command_parser_demo.py) | Earlier command parser and movement-routing test | Robot test |
| [`spider/robot/dance_command_demo.py`](spider/robot/dance_command_demo.py) | Wake-word dance command test | Robot test |
| [`spider/robot/movements/`](spider/robot/movements/) | Robot movement and interaction commands | Hardware-specific runtime |
| [`spider/robot/debug/`](spider/robot/debug/) | Audio capture and spectrogram diagnostics | Hardware/debug tools |
| [`tools/benchmark/stt_benchmark.py`](tools/benchmark/stt_benchmark.py) | Current multi-speaker Whisper/Vosk benchmark | Canonical benchmark |
| [`tools/benchmark/run_all.py`](tools/benchmark/run_all.py) | Resumable benchmark runner with thermal checks | Canonical benchmark runner |
| [`tools/preprocessing/preprocess.py`](tools/preprocessing/preprocess.py) | ffmpeg-based preprocessing sweep | Canonical preprocessing tool |
| [`tools/preprocessing/preprocess_amplitude.py`](tools/preprocessing/preprocess_amplitude.py) | Adaptive amplitude preprocessing | Research experiment |
| [`tools/preprocessing/preprocess_denoise.py`](tools/preprocessing/preprocess_denoise.py) | Denoising preprocessing | Research experiment |
| [`tools/preprocessing/run_preprocessing_all_models.py`](tools/preprocessing/run_preprocessing_all_models.py) | Runs the two preprocessing benchmark conditions | Research experiment |
| [`tools/dataset/extract_audio.py`](tools/dataset/extract_audio.py) | Extracts audio from source archives | Dataset preparation |
| [`tools/preprocessing/repair.py`](tools/preprocessing/repair.py), [`retry_failed_preprocessing.py`](tools/preprocessing/retry_failed_preprocessing.py) | Recovery tools for failed preprocessing files | One-off recovery tools |
| [`tools/benchmark/net_latency_bench.py`](tools/benchmark/net_latency_bench.py) | Network/cloud speech latency experiment | Separate experiment |
| [`tools/legacy/legacy_stt_benchmark.py`](tools/legacy/legacy_stt_benchmark.py) | Earlier benchmark design | Legacy; do not use for new runs |

The `models/` tree contains downloaded model assets and an upstream
Whisper.cpp checkout/build tree. It is local-only and should not be committed.

## Setup

The robot runtime targets a Raspberry Pi and hardware-specific libraries.
Create a fresh environment instead of using the old local environment
directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-robot.txt
```

Benchmark and preprocessing scripts have additional dependencies:

```bash
python3 -m pip install -r requirements-benchmark.txt
```

The movement scripts also require the Freenove robot server modules (`control`,
`servo`, `ultrasonic`, and `buzzer`) from the robot kit. Those modules are
hardware/vendor dependencies and are not part of this repository.

## Runtime configuration

The robot movement scripts require the Freenove server directory. Set
`FREENOVE_SERVER` if it is not at the default Raspberry Pi location.

The wake-word model files belong under `models/wakeword/`. The setup script
clones the upstream Whisper.cpp source into `models/whisper.cpp/` when needed
and downloads the configured Whisper models. The Whisper executable and model
files are expected there. The benchmark supports `WHISPER_BIN`,
`WHISPER_DIR`, and `VOSK_DIR` environment variables; prefer those over editing
the benchmark source. The robot runtime uses the same repository-relative
defaults through [`spider/robot/runtime_config.py`](spider/robot/runtime_config.py)
and supports `WAKEWORD_MODEL`, `VOSK_MODEL`, `SPIDER_MOVEMENTS`,
`LAST_COMMAND_WAV`, and `FREENOVE_SERVER`.

## Install dependencies and models

The setup script installs both requirement files and downloads the Vosk and
Whisper models used by the benchmark:

```bash
python3 scripts/install_dependencies_and_models.py
```

It also asks openWakeWord to download its base feature models. The custom
`hey_spider.tflite` model is project-specific and is not hosted in this
repository. Copy it to `models/wakeword/modified/hey_spider.tflite`, or provide
its private/authorized URL:

```bash
python3 scripts/install_dependencies_and_models.py --wakeword-url "$WAKEWORD_URL"
```

Use `--skip-dependencies` or `--skip-models` when only one setup phase is
needed. Preprocessing additionally requires the system `ffmpeg` executable.

## Running benchmarks

First prepare a local audio tree and model installations, then perform a
non-destructive preflight:

```bash
python3 tools/benchmark/run_all.py \
  --audio-root ./recordings \
  --out ./results \
  --check-only
```

Start a benchmark only after the preflight identifies the intended recordings
and models:

```bash
python3 tools/benchmark/run_all.py --audio-root ./recordings --out ./results
```

Preprocessing outputs belong under `variants/` and benchmark workbooks/logs
belong under `results/`. Both are intentionally excluded from commits.

## Validation

The repository currently has hardware-dependent scripts rather than a
portable automated test suite. A safe baseline check is:

```bash
python3 -m compileall -q \
  tools/dataset/extract_audio.py \
  tools/benchmark/net_latency_bench.py \
  tools/preprocessing/preprocess.py \
  tools/preprocessing/preprocess_amplitude.py \
  tools/preprocessing/preprocess_denoise.py \
  tools/preprocessing/repair.py tools/preprocessing/retry_failed_preprocessing.py \
  tools/benchmark/run_all.py \
  tools/preprocessing/run_preprocessing_all_models.py \
  tools/benchmark/stt_benchmark.py spider
```

Compilation checks syntax only; it does not access microphones, servos,
Whisper/Vosk models, or the robot server.

## Before publishing

1. Keep recordings, models, `spider/env/`, `results/`, and logs out of Git.
2. Choose `stt_benchmark.py` as the only benchmark entry point for new runs.
3. Replace machine-specific runtime paths with configuration before sharing the
   robot runtime with others.
4. Do not commit credentials, private recordings, or generated model binaries.
