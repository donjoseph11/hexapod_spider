#!/usr/bin/env python3

"""
preprocess_denoise.py

Preprocess all extracted WAV files using:

    Denoise strength = 1.00
    Original mix     = 0.20

Input:
    ./Audio/extracted/

Output:
    ./variants/denoise100_mix020/

Original WAV files are NOT modified.
"""

from pathlib import Path
import os

import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("AUDIO_INPUT_ROOT", PROJECT_ROOT / "Audio" / "extracted"))
OUTPUT_ROOT = Path(
    os.environ.get(
        "PREPROCESS_OUTPUT_ROOT",
        PROJECT_ROOT / "variants" / "denoise100_mix020",
    )
)

TARGET_SR = 16000

# ------------------------------------------------------------
# DENOISING PARAMETERS
# ------------------------------------------------------------

DENOISE_STRENGTH = 1.00
ORIGINAL_MIX = 0.20


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    audio, sr = sf.read(
        path,
        always_2d=False
    )

    # Stereo -> mono
    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    return (
        audio.astype(np.float32),
        sr
    )


# ============================================================
# RESAMPLE TO 16 kHz
# ============================================================

def resample_audio(
    audio,
    original_sr,
    target_sr=16000
):

    if original_sr == target_sr:
        return audio

    gcd = np.gcd(
        original_sr,
        target_sr
    )

    up = target_sr // gcd
    down = original_sr // gcd

    audio = resample_poly(
        audio,
        up,
        down
    )

    return audio.astype(np.float32)


# ============================================================
# DC OFFSET REMOVAL
# ============================================================

def remove_dc_offset(audio):

    if len(audio) == 0:
        return audio

    return (
        audio - np.mean(audio)
    ).astype(np.float32)


# ============================================================
# DENOISING
# ============================================================

def denoise_audio(
    audio,
    sr,
    strength=1.00,
    original_mix=0.20
):
    """
    Denoise audio and mix some original signal back in.

    strength = 1.00
        Maximum prop_decrease used by noisereduce.

    original_mix = 0.20
        Final signal is approximately:

        80% denoised
        +
        20% original
    """

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:
        return audio

    cleaned = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=strength
    )

    cleaned = np.asarray(
        cleaned,
        dtype=np.float32
    )

    processed = (
        (1.0 - original_mix)
        * cleaned
        +
        original_mix
        * audio
    )

    return processed.astype(
        np.float32
    )


# ============================================================
# AUDIO MEASUREMENTS
# ============================================================

def peak_dbfs(audio):

    if len(audio) == 0:
        return np.nan

    peak = np.max(
        np.abs(audio)
    )

    if peak <= 1e-12:
        return -np.inf

    return float(
        20 * np.log10(peak)
    )


def rms_dbfs(audio):

    if len(audio) == 0:
        return np.nan

    rms = np.sqrt(
        np.mean(
            np.square(audio)
        )
    )

    if rms <= 1e-12:
        return -np.inf

    return float(
        20 * np.log10(rms)
    )



# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    input_file,
    output_file
):

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    audio, original_sr = load_audio(
        input_file
    )

    original_peak = peak_dbfs(audio)
    original_rms = rms_dbfs(audio)

    # --------------------------------------------------------
    # Convert to 16 kHz
    # --------------------------------------------------------

    audio = resample_audio(
        audio,
        original_sr,
        TARGET_SR
    )

    # --------------------------------------------------------
    # Remove DC offset
    # --------------------------------------------------------

    audio = remove_dc_offset(
        audio
    )

    # Keep a copy of the signal that goes into denoising
    original_audio = audio.copy()

    # --------------------------------------------------------
    # Denoising
    # --------------------------------------------------------

    processed = denoise_audio(
        original_audio,
        sr=TARGET_SR,
        strength=DENOISE_STRENGTH,
        original_mix=ORIGINAL_MIX
    )

    # --------------------------------------------------------
    # Safety
    #
    # IMPORTANT:
    # We are NOT amplitude-normalizing here.
    # This step only protects against numerical overflow.
    # --------------------------------------------------------

    processed = np.clip(
        processed,
        -1.0,
        1.0
    ).astype(np.float32)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save 16 kHz PCM16
    # --------------------------------------------------------

    sf.write(
        output_file,
        processed,
        TARGET_SR,
        subtype="PCM_16"
    )

    return {
        "original_peak": original_peak,
        "processed_peak": peak_dbfs(processed),

        "original_rms": original_rms,
        "processed_rms": rms_dbfs(processed),

        "duration": (
            len(processed)
            / TARGET_SR
        )
    }


# ============================================================
# PROCESS FULL DATASET
# ============================================================

def main():

    print()
    print("=" * 76)
    print("DENOISE PREPROCESSING")
    print("=" * 76)

    print(
        f"Input            : {INPUT_ROOT}"
    )

    print(
        f"Output           : {OUTPUT_ROOT}"
    )

    print(
        f"Denoise strength : {DENOISE_STRENGTH}"
    )

    print(
        f"Original mix     : {ORIGINAL_MIX}"
    )

    print(
        f"Sample rate      : {TARGET_SR} Hz"
    )

    print("=" * 76)


    # --------------------------------------------------------
    # Find every WAV recursively
    # --------------------------------------------------------

    wav_files = sorted(
        INPUT_ROOT.rglob("*.wav")
    )

    print()
    print(
        f"WAV files found: {len(wav_files)}"
    )


    if not wav_files:

        raise SystemExit(
            f"No WAV files found in {INPUT_ROOT}"
        )


    successful = 0
    failed = 0



    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for index, input_file in enumerate(
        wav_files,
        start=1
    ):

        relative_path = (
            input_file.relative_to(
                INPUT_ROOT
            )
        )

        output_file = (
            OUTPUT_ROOT
            / relative_path
        )


        try:

            result = process_file(
                input_file,
                output_file
            )

            successful += 1


            print(
                f"[{index:03d}/{len(wav_files):03d}] "
                f"{relative_path}"
            )

            print(
                "    "
                f"Peak: "
                f"{result['original_peak']:.2f} "
                f"-> "
                f"{result['processed_peak']:.2f} dBFS"
            )

            print(
                "    "
                f"RMS : "
                f"{result['original_rms']:.2f} "
                f"-> "
                f"{result['processed_rms']:.2f} dBFS"
            )


        except Exception as exc:

            failed += 1

            print()
            print(
                f"ERROR: {input_file}"
            )

            print(
                f"       {exc}"
            )


    # ========================================================
    # FINAL CHECK
    # ========================================================

    output_wavs = sorted(
        OUTPUT_ROOT.rglob("*.wav")
    )


    print()
    print("=" * 76)
    print("PREPROCESSING COMPLETE")
    print("=" * 76)

    print(
        f"Input WAV files : {len(wav_files)}"
    )

    print(
        f"Successful      : {successful}"
    )

    print(
        f"Failed          : {failed}"
    )

    print(
        f"Output WAV files: {len(output_wavs)}"
    )

    print(
        f"Output folder   : {OUTPUT_ROOT}"
    )

    print("=" * 76)


    # --------------------------------------------------------
    # Dataset integrity warning
    # --------------------------------------------------------

    if len(output_wavs) != len(wav_files):

        print()
        print(
            "WARNING: Input and output WAV counts "
            "do not match."
        )

    else:

        print()
        print(
            "SUCCESS: Every input WAV has a "
            "processed output WAV."
        )


if __name__ == "__main__":
    main()
