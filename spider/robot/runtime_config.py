"""Shared paths for the Raspberry Pi runtime experiments."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPIDER_ROOT = PROJECT_ROOT / "spider"
ROBOT_ROOT = SPIDER_ROOT / "robot"
MOVEMENTS = Path(os.environ.get("SPIDER_MOVEMENTS", ROBOT_ROOT / "movements"))
WAKEWORD_MODEL = Path(
    os.environ.get(
        "WAKEWORD_MODEL",
        PROJECT_ROOT / "models" / "wakeword" / "modified" / "hey_spider.tflite",
    )
)
WHISPER_DIR = Path(
    os.environ.get("WHISPER_DIR", PROJECT_ROOT / "models" / "whisper.cpp" / "models")
)
WHISPER_BIN = Path(
    os.environ.get(
        "WHISPER_BIN",
        PROJECT_ROOT / "models" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
    )
)
VOSK_MODEL = Path(
    os.environ.get(
        "VOSK_MODEL",
        PROJECT_ROOT / "models" / "vosk" / "vosk-model-en-us-0.22",
    )
)
LAST_COMMAND_WAV = Path(
    os.environ.get("LAST_COMMAND_WAV", PROJECT_ROOT / "spider" / "last_command.wav")
)
FREENOVE_SERVER = Path(
    os.environ.get(
        "FREENOVE_SERVER",
        "/home/pi/freenove_Big_Hexapod_Robot_Kit_for_Raspberry_Pi/Code/Server",
    )
)


def movement(name: str) -> str:
    return str(MOVEMENTS / name)
