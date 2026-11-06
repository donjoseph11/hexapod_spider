#!/usr/bin/env python3

"""
preprocess_amplitude.py

Adaptive amplitude boosting for the extracted speech-command dataset.

Rules:
1. Detect active speech.
2. Measure speech RMS only.
3. If speech RMS >= MIN_SPEECH_RMS_DBFS:
       leave audio unchanged.
4. If speech is too quiet:
       boost toward TARGET_SPEECH_RMS_DBFS.
5. Limit boost so the peak does not exceed PEAK_LIMIT_DBFS.
6. Never reduce an already-loud recording.
7. Save a CSV log showing exactly which files were adjusted.

Input:
    ./Audio/extracted/

Output:
    ./variants/amplitude_adaptive/
"""

from pathlib import Path
import os
import csv

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("AUDIO_INPUT_ROOT", PROJECT_ROOT / "Audio" / "extracted"))
OUTPUT_ROOT = Path(
    os.environ.get("PREPROCESS_OUTPUT_ROOT", PROJECT_ROOT / "variants" / "amplitude_adaptive")
)

TARGET_SR = 16000


# ============================================================
# AMPLITUDE PARAMETERS
# ============================================================

# If speech is louder than this, do nothing.
MIN_SPEECH_RMS_DBFS = -22.0

# Quiet recordings are boosted toward this.
TARGET_SPEECH_RMS_DBFS = -18.0

# Maximum peak allowed after boosting.
# This does NOT reduce already-loud original files.
PEAK_LIMIT_DBFS = -3.0

# Used only for finding active speech.
SPEECH_TOP_DB = 30


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    audio, sr = sf.read(
        path,
        always_2d=False
    )

    # Convert stereo/multichannel to mono
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
# RESAMPLE
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

    return audio.astype(
        np.float32
    )


# ============================================================
# MEASUREMENT FUNCTIONS
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
# ACTIVE SPEECH RMS
# ============================================================

def get_speech_rms_dbfs(
    audio,
    top_db=30
):

    """
    Measure RMS using active/non-silent speech only.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:
        return np.nan

    intervals = librosa.effects.split(
        audio,
        top_db=top_db
    )

    if len(intervals) == 0:
        return np.nan

    active_parts = []

    for start, end in intervals:

        if end > start:

            active_parts.append(
                audio[start:end]
            )

    if not active_parts:
        return np.nan

    active_audio = np.concatenate(
        active_parts
    )

    rms = np.sqrt(
        np.mean(
            np.square(active_audio)
        )
    )

    if rms <= 1e-12:
        return np.nan

    return float(
        20 * np.log10(rms)
    )


# ============================================================
# ADAPTIVE AMPLITUDE BOOST
# ============================================================

def boost_speech_amplitude(
    audio,
    min_speech_rms_dbfs=-22.0,
    target_speech_rms_dbfs=-18.0,
    peak_limit_dbfs=-3.0,
    top_db=30
):

    """
    Boost quiet speech only.

    Returns:
        processed_audio
        adjusted
        speech_rms_before
        speech_rms_after
        gain_db
        reason
    """

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:

        return (
            audio,
            False,
            np.nan,
            np.nan,
            0.0,
            "empty_audio"
        )


    # --------------------------------------------------------
    # Measure speech RMS
    # --------------------------------------------------------

    speech_rms_before = get_speech_rms_dbfs(
        audio,
        top_db=top_db
    )


    if not np.isfinite(
        speech_rms_before
    ):

        return (
            audio.copy(),
            False,
            speech_rms_before,
            speech_rms_before,
            0.0,
            "no_active_speech"
        )


    # --------------------------------------------------------
    # Already loud enough -> DO NOTHING
    # --------------------------------------------------------

    if (
        speech_rms_before
        >= min_speech_rms_dbfs
    ):

        return (
            audio.copy(),
            False,
            speech_rms_before,
            speech_rms_before,
            0.0,
            "already_loud_enough"
        )


    # --------------------------------------------------------
    # Desired gain
    # --------------------------------------------------------

    desired_gain_db = (
        target_speech_rms_dbfs
        - speech_rms_before
    )

    desired_gain_linear = (
        10 ** (
            desired_gain_db / 20.0
        )
    )


    # --------------------------------------------------------
    # Calculate maximum safe boost
    # --------------------------------------------------------

    current_peak = np.max(
        np.abs(audio)
    )

    if current_peak <= 1e-12:

        return (
            audio.copy(),
            False,
            speech_rms_before,
            speech_rms_before,
            0.0,
            "silent_audio"
        )


    peak_limit_linear = (
        10 ** (
            peak_limit_dbfs / 20.0
        )
    )


    maximum_safe_gain = (
        peak_limit_linear
        / current_peak
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Never reduce an original recording.
    # --------------------------------------------------------

    if maximum_safe_gain <= 1.0:

        return (
            audio.copy(),
            False,
            speech_rms_before,
            speech_rms_before,
            0.0,
            "no_safe_boost_available"
        )


    # --------------------------------------------------------
    # Use smaller of desired gain and safe gain
    # --------------------------------------------------------

    actual_gain_linear = min(
        desired_gain_linear,
        maximum_safe_gain
    )


    if actual_gain_linear <= 1.0:

        return (
            audio.copy(),
            False,
            speech_rms_before,
            speech_rms_before,
            0.0,
            "no_boost_needed"
        )


    # --------------------------------------------------------
    # Apply boost
    # --------------------------------------------------------

    processed = (
        audio
        * actual_gain_linear
    ).astype(np.float32)


    # Numerical safety only
    processed = np.clip(
        processed,
        -1.0,
        1.0
    )


    actual_gain_db = float(
        20 * np.log10(
            actual_gain_linear
        )
    )


    # --------------------------------------------------------
    # Measure speech RMS after
    # --------------------------------------------------------

    speech_rms_after = get_speech_rms_dbfs(
        processed,
        top_db=top_db
    )


    return (
        processed,
        True,
        speech_rms_before,
        speech_rms_after,
        actual_gain_db,
        "boosted"
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


    # --------------------------------------------------------
    # Resample to 16 kHz
    # --------------------------------------------------------

    audio = resample_audio(
        audio,
        original_sr,
        TARGET_SR
    )


    # --------------------------------------------------------
    # Original measurements
    # --------------------------------------------------------

    original_peak = peak_dbfs(
        audio
    )

    original_rms = rms_dbfs(
        audio
    )


    # --------------------------------------------------------
    # Adaptive amplitude
    # --------------------------------------------------------

    (
        processed,
        adjusted,
        speech_rms_before,
        speech_rms_after,
        gain_db,
        reason
    ) = boost_speech_amplitude(

        audio,

        min_speech_rms_dbfs=(
            MIN_SPEECH_RMS_DBFS
        ),

        target_speech_rms_dbfs=(
            TARGET_SPEECH_RMS_DBFS
        ),

        peak_limit_dbfs=(
            PEAK_LIMIT_DBFS
        ),

        top_db=(
            SPEECH_TOP_DB
        )
    )


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    sf.write(
        output_file,
        processed,
        TARGET_SR,
        subtype="PCM_16"
    )


    # --------------------------------------------------------
    # Final measurements
    # --------------------------------------------------------

    processed_peak = peak_dbfs(
        processed
    )

    processed_rms = rms_dbfs(
        processed
    )


    clipped_samples = int(
        np.sum(
            np.abs(processed)
            >= 0.999
        )
    )


    return {

        "input_file":
            str(input_file),

        "output_file":
            str(output_file),

        "adjusted":
            adjusted,

        "reason":
            reason,

        "gain_db":
            gain_db,

        "speech_rms_before_dbfs":
            speech_rms_before,

        "speech_rms_after_dbfs":
            speech_rms_after,

        "original_peak_dbfs":
            original_peak,

        "processed_peak_dbfs":
            processed_peak,

        "original_full_rms_dbfs":
            original_rms,

        "processed_full_rms_dbfs":
            processed_rms,

        "clipped_samples":
            clipped_samples
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("ADAPTIVE AMPLITUDE PREPROCESSING")
    print("=" * 78)

    print(
        f"Input              : {INPUT_ROOT}"
    )

    print(
        f"Output             : {OUTPUT_ROOT}"
    )

    print(
        f"Min speech RMS     : "
        f"{MIN_SPEECH_RMS_DBFS} dBFS"
    )

    print(
        f"Target speech RMS  : "
        f"{TARGET_SPEECH_RMS_DBFS} dBFS"
    )

    print(
        f"Boost peak ceiling : "
        f"{PEAK_LIMIT_DBFS} dBFS"
    )

    print("=" * 78)


    # --------------------------------------------------------
    # Find WAVs
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


    results = []

    successful = 0
    failed = 0


    # --------------------------------------------------------
    # Process every file
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

            result["status"] = "success"
            result["error"] = ""

            results.append(
                result
            )

            successful += 1


            status_text = (
                "BOOSTED"
                if result["adjusted"]
                else "UNCHANGED"
            )


            print(
                f"[{index:03d}/{len(wav_files):03d}] "
                f"{status_text:9s} "
                f"{relative_path}"
            )


            print(
                "    "
                f"Speech RMS: "
                f"{result['speech_rms_before_dbfs']:.2f} "
                f"-> "
                f"{result['speech_rms_after_dbfs']:.2f} dBFS"
            )


            print(
                "    "
                f"Gain: "
                f"{result['gain_db']:.2f} dB"
            )


        except Exception as exc:

            failed += 1

            results.append({

                "input_file":
                    str(input_file),

                "output_file":
                    str(output_file),

                "status":
                    "error",

                "error":
                    str(exc)
            })


            print()
            print(
                f"ERROR: {input_file}"
            )

            print(
                f"       {exc}"
            )


    # ========================================================
    # WRITE CSV LOG
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    log_file = (
        OUTPUT_ROOT
        / "amplitude_preprocessing_log.csv"
    )


    if results:

        fieldnames = sorted(
            {
                key
                for row in results
                for key in row.keys()
            }
        )


        with open(
            log_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                results
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    successful_rows = [
        row
        for row in results
        if row.get("status") == "success"
    ]


    adjusted_rows = [
        row
        for row in successful_rows
        if row.get("adjusted") is True
    ]


    unchanged_rows = [
        row
        for row in successful_rows
        if row.get("adjusted") is False
    ]


    clipped_files = [
        row
        for row in successful_rows
        if row.get(
            "clipped_samples",
            0
        ) > 0
    ]


    output_wavs = sorted(
        OUTPUT_ROOT.rglob("*.wav")
    )


    print()
    print("=" * 78)
    print("AMPLITUDE SUMMARY")
    print("=" * 78)

    print(
        f"Input WAV files : "
        f"{len(wav_files)}"
    )

    print(
        f"Successful      : "
        f"{successful}"
    )

    print(
        f"Failed          : "
        f"{failed}"
    )

    print()

    print(
        f"Adjusted files  : "
        f"{len(adjusted_rows)}"
    )

    print(
        f"Unchanged files : "
        f"{len(unchanged_rows)}"
    )


    if successful_rows:

        percentage = (
            len(adjusted_rows)
            / len(successful_rows)
            * 100
        )

        print(
            f"Adjusted %      : "
            f"{percentage:.2f}%"
        )


    print()

    print(
        f"Files clipping  : "
        f"{len(clipped_files)}"
    )

    print(
        f"Output WAV files: "
        f"{len(output_wavs)}"
    )

    print(
        f"CSV log         : "
        f"{log_file}"
    )

    print(
        f"Output folder   : "
        f"{OUTPUT_ROOT}"
    )

    print("=" * 78)


    # ========================================================
    # PRINT ADJUSTED FILES
    # ========================================================

    if adjusted_rows:

        print()
        print("FILES WHOSE AMPLITUDE WAS BOOSTED")
        print("-" * 78)

        for row in adjusted_rows:

            relative = Path(
                row["input_file"]
            ).relative_to(
                INPUT_ROOT
            )

            print(
                f"{relative}"
            )

            print(
                "    "
                f"{row['speech_rms_before_dbfs']:.2f} "
                f"-> "
                f"{row['speech_rms_after_dbfs']:.2f} dBFS, "
                f"gain={row['gain_db']:.2f} dB"
            )


if __name__ == "__main__":
    main()