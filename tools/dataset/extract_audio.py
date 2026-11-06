#!/usr/bin/env python3

from pathlib import Path
import os
import zipfile
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_ROOT = Path(
    os.environ.get("AUDIO_ROOT", PROJECT_ROOT / "Audio")
)

FEMALE_DIR = AUDIO_ROOT / "Female"
MALE_DIR = AUDIO_ROOT / "Male"

# All extracted audio will go here
OUTPUT_ROOT = AUDIO_ROOT / "extracted"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".aac",
    ".wma"
}


# ============================================================
# EXTRACT ONE ZIP
# ============================================================

def extract_audio_from_zip(zip_path, gender):
    """
    Extract only audio files from one ZIP.

    Example:

    female_1_recordings.zip

    becomes:

    Audio/extracted/Female/female_1_recordings/
    """

    zip_path = Path(zip_path)

    # Folder name without .zip
    participant_folder = zip_path.stem

    output_dir = (
        OUTPUT_ROOT
        / gender
        / participant_folder
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    extracted = 0
    skipped = 0

    print()
    print("=" * 70)
    print(f"ZIP    : {zip_path}")
    print(f"OUTPUT : {output_dir}")
    print("=" * 70)

    with zipfile.ZipFile(zip_path, "r") as zf:

        for member in zf.infolist():

            # Ignore folders inside ZIP
            if member.is_dir():
                continue

            member_path = Path(member.filename)

            extension = member_path.suffix.lower()

            # Ignore non-audio files
            if extension not in AUDIO_EXTENSIONS:
                skipped += 1
                continue

            # Keep only filename, avoiding unnecessary nested folders
            filename = member_path.name

            destination = output_dir / filename

            # Avoid accidentally overwriting duplicate filenames
            if destination.exists():

                stem = destination.stem
                suffix = destination.suffix

                counter = 1

                while destination.exists():

                    destination = (
                        output_dir
                        / f"{stem}_{counter}{suffix}"
                    )

                    counter += 1

            # Extract file safely
            with zf.open(member) as source:
                with open(destination, "wb") as target:
                    shutil.copyfileobj(
                        source,
                        target
                    )

            extracted += 1

            print(
                f"  [{extracted:03d}] "
                f"{destination.name}"
            )

    print()
    print(
        f"Extracted audio files : {extracted}"
    )

    print(
        f"Skipped non-audio     : {skipped}"
    )

    return extracted



# ============================================================
# PROCESS ONE GENDER
# ============================================================

def process_gender(folder, gender):

    folder = Path(folder)

    if not folder.exists():

        print(
            f"ERROR: Folder does not exist: "
            f"{folder}"
        )

        return 0, 0

    zip_files = sorted(
        folder.glob("*.zip")
    )

    print()
    print("#" * 70)
    print(f"{gender.upper()}")
    print("#" * 70)

    print(
        f"Folder    : {folder}"
    )

    print(
        f"ZIP files : {len(zip_files)}"
    )

    if not zip_files:

        print(
            f"No ZIP files found in {folder}"
        )

        return 0, 0

    total_audio = 0

    for index, zip_path in enumerate(
        zip_files,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(zip_files)}] "
            f"{zip_path.name}"
        )

        try:

            count = extract_audio_from_zip(
                zip_path,
                gender
            )

            total_audio += count

        except zipfile.BadZipFile:

            print(
                f"ERROR: Invalid ZIP file: "
                f"{zip_path}"
            )

    return len(zip_files), total_audio


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AUDIO DATASET EXTRACTION")
    print("=" * 70)

    print(
        f"Audio root : {AUDIO_ROOT}"
    )

    print(
        f"Output     : {OUTPUT_ROOT}"
    )


    # --------------------------------------------------------
    # FEMALE
    # --------------------------------------------------------

    female_zips, female_audio = (
        process_gender(
            FEMALE_DIR,
            "Female"
        )
    )


    # --------------------------------------------------------
    # MALE
    # --------------------------------------------------------

    male_zips, male_audio = (
        process_gender(
            MALE_DIR,
            "Male"
        )
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Female ZIP files    : {female_zips}"
    )

    print(
        f"Female audio files  : {female_audio}"
    )

    print()

    print(
        f"Male ZIP files      : {male_zips}"
    )

    print(
        f"Male audio files    : {male_audio}"
    )

    print()

    print(
        f"TOTAL ZIP files     : "
        f"{female_zips + male_zips}"
    )

    print(
        f"TOTAL audio files   : "
        f"{female_audio + male_audio}"
    )

    print()

    print(
        f"Extracted dataset   : "
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
