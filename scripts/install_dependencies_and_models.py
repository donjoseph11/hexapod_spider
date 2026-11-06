#!/usr/bin/env python3
"""Install project dependencies and download the speech models used locally."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"
VOSK_ROOT = MODEL_ROOT / "vosk"
WHISPER_ROOT = MODEL_ROOT / "whisper.cpp"
WAKEWORD_ROOT = MODEL_ROOT / "wakeword"
WHISPER_REPOSITORY = "https://github.com/ggerganov/whisper.cpp.git"

VOSK_MODELS = {
    "vosk-model-small-en-us-0.15": (
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    ),
    "vosk-model-en-us-0.22": (
        "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
    ),
    "vosk-model-en-us-0.22-lgraph": (
        "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
    ),
}
WHISPER_MODELS = ("tiny.en", "base.en", "small.en", "medium.en")


def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def install_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "-r", "requirements-robot.txt"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements-benchmark.txt"])


def download_zip(url: str, destination: Path) -> None:
    archive = destination.with_suffix(".zip")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive)
    try:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = (destination / member.filename).resolve()
                if not str(target).startswith(str(destination.resolve()) + os.sep):
                    raise RuntimeError(f"Unsafe archive member: {member.filename}")
            source.extractall(destination)
    finally:
        archive.unlink(missing_ok=True)


def install_vosk_models() -> None:
    VOSK_ROOT.mkdir(parents=True, exist_ok=True)
    for name, url in VOSK_MODELS.items():
        target = VOSK_ROOT / name
        if target.exists():
            print(f"Already present: {target}")
            continue
        download_zip(url, VOSK_ROOT)


def install_whisper_models() -> None:
    downloader = WHISPER_ROOT / "models" / "download-ggml-model.sh"
    if not downloader.exists():
        if WHISPER_ROOT.exists() and any(WHISPER_ROOT.iterdir()):
            raise RuntimeError(
                f"Whisper directory exists but is not a usable checkout: {WHISPER_ROOT}"
            )
        if shutil.which("git") is None:
            raise RuntimeError("git is required to download the Whisper.cpp checkout")
        run(["git", "clone", "--depth", "1", WHISPER_REPOSITORY, str(WHISPER_ROOT)])
        downloader = WHISPER_ROOT / "models" / "download-ggml-model.sh"
    for model in WHISPER_MODELS:
        target = WHISPER_ROOT / "models" / f"ggml-{model}.bin"
        if target.exists():
            print(f"Already present: {target}")
            continue
        run(["bash", str(downloader), model])


def install_openwakeword_models(wakeword_url: str | None) -> None:
    try:
        run([sys.executable, "-c", "from openwakeword.utils import download_models; download_models()"])
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not download openWakeWord base models: {error}")

    custom = WAKEWORD_ROOT / "modified" / "hey_spider.tflite"
    if custom.exists():
        print(f"Already present: {custom}")
    elif wakeword_url:
        custom.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading custom wake-word model to {custom}")
        urllib.request.urlretrieve(wakeword_url, custom)
    else:
        print(
            "Custom hey_spider.tflite is not hosted by this repository. "
            "Provide --wakeword-url or copy it to models/wakeword/modified/."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dependencies", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--wakeword-url", default=os.environ.get("WAKEWORD_URL"))
    args = parser.parse_args()

    if not args.skip_dependencies:
        install_dependencies()
    if not args.skip_models:
        install_vosk_models()
        install_whisper_models()
        install_openwakeword_models(args.wakeword_url)

    print("\nSetup finished.")
    print(f"Python: {sys.executable}")
    print(f"Models: {MODEL_ROOT}")
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg is not installed; preprocessing requires it.")


if __name__ == "__main__":
    main()
