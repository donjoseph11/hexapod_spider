"""
retry_failed_preprocessing.py - last-resort repair for files still hitting ffmpeg 6.1
'best_input >= 0' filtergraph assertion, which is triggered by atempo.

STRATEGIES, tried in order per file:
  A  -filter_complex instead of -af  (different code path in ffmpeg)
  B  atempo alone, then aresample as a separate process
  C  sox for the tempo change, ffmpeg for the resample (if sox installed)

Anything that survives all three is DELETED, not left on disk. A truncated
wav that stays in place would be transcribed as clipped audio and produce
silently wrong WER - far worse than a missing file, which simply drops out
of the paired comparison. The deleted list is written to excluded.txt so the
same utterances can be removed from the baseline arm at analysis time.

USAGE
    python3 tools/preprocessing/retry_failed_preprocessing.py --variants ./variants
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import wave

TEMPO = 0.9
EXPECTED = 1.0 / TEMPO
TOL = 0.02
STAGE1 = "highpass=f=80,afftdn=nr=10:nf=-25,loudnorm=I=-23:TP=-2:LRA=7"


def dur(p):
    try:
        with wave.open(p, "rb") as w:
            n = w.getnframes()
            return n / w.getframerate() if n else None
    except Exception:
        return None


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return r.returncode == 0, r.stderr.strip()[:120]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def strat_a(mid, dst):
    return sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-i", mid, "-filter_complex",
               f"[0:a]atempo={TEMPO},aresample=16000,"
               "aformat=sample_fmts=s16:channel_layouts=mono[out]",
               "-map", "[out]", "-c:a", "pcm_s16le", dst])


def strat_b(mid, dst):
    fd, t2 = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        ok, err = sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                      "-y", "-i", mid, "-af", f"atempo={TEMPO}",
                      "-c:a", "pcm_s16le", t2])
        if not ok:
            return False, "b/atempo: " + err
        return sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-y", "-i", t2, "-ar", "16000", "-ac", "1",
                   "-c:a", "pcm_s16le", dst])
    finally:
        if os.path.exists(t2):
            os.remove(t2)


def strat_c(mid, dst):
    if not shutil.which("sox"):
        return False, "sox not installed"
    fd, t2 = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        ok, err = sh(["sox", mid, t2, "tempo", "-s", str(TEMPO)])
        if not ok:
            return False, "c/sox: " + err
        return sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-y", "-i", t2, "-ar", "16000", "-ac", "1",
                   "-c:a", "pcm_s16le", dst])
    finally:
        if os.path.exists(t2):
            os.remove(t2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--condition", default="clean_slow90")
    args = ap.parse_args()

    raw_root = os.path.join(args.variants, "raw")
    cond_root = os.path.join(args.variants, args.condition)

    bad = []
    for dp, _, fs in os.walk(raw_root):
        for fn in sorted(fs):
            if not fn.lower().endswith(".wav"):
                continue
            rel = os.path.relpath(os.path.join(dp, fn), raw_root)
            dr, do = dur(os.path.join(raw_root, rel)), dur(os.path.join(cond_root, rel))
            if dr is None:
                continue
            if do is None or abs(do / dr - EXPECTED) > TOL:
                bad.append(rel)

    print(f"still bad: {len(bad)}")
    if not bad:
        print("nothing to do - run the sweep.")
        return

    fixed, dead = [], []
    for i, rel in enumerate(bad, 1):
        src = os.path.join(raw_root, rel)
        dst = os.path.join(cond_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        fd, mid = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        ok, err = sh(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                      "-y", "-i", src, "-af", STAGE1, "-ar", "48000", "-ac", "1",
                      "-c:a", "pcm_s16le", mid])
        if not ok:
            dead.append((rel, "stage1: " + err))
            os.path.exists(mid) and os.remove(mid)
            continue

        won = None
        for name, fn in (("A", strat_a), ("B", strat_b), ("C", strat_c)):
            ok, err = fn(mid, dst)
            if ok:
                d = dur(dst)
                if d and abs(d / dur(src) - EXPECTED) <= TOL:
                    won = name
                    break
        os.path.exists(mid) and os.remove(mid)

        if won:
            fixed.append((rel, won))
        else:
            dead.append((rel, "all strategies failed"))
            if os.path.exists(dst):
                os.remove(dst)          # never leave truncated audio behind
        if i % 10 == 0 or i == len(bad):
            print(f"  [{i}/{len(bad)}]", flush=True)

    print(f"\nfixed   {len(fixed)}")
    for rel, w in fixed[:5]:
        print(f"        [{w}] {rel[:60]}")
    print(f"deleted {len(dead)}")

    if dead:
        p = os.path.join(args.variants, "excluded.txt")
        with open(p, "w", encoding="utf-8") as f:
            for rel, why in dead:
                f.write(rel + "\n")
        print(f"\nexcluded list: {p}")
        print("These files no longer exist in the condition tree, so the benchmark\n"
              "will skip them. Remove the same rel_paths from the baseline arm when\n"
              "you compare, so both sides cover an identical file set.")

    print(f"\n{len(bad) - len(dead)} of {len(bad)} recovered. "
          f"Corpus now {600 - len(dead)}/600 usable "
          f"({(600 - len(dead)) / 6:.1f}% coverage).")
    print("\nSafe to start the sweep.")


if __name__ == "__main__":
    main()
