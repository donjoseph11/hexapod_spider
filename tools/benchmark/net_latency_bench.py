#!/usr/bin/env python3
"""
net_latency_bench_scored.py

Raspberry Pi 5 client: on-device vs network-assisted STT latency.

This version integrates the FINAL thesis scoring normalisation directly into
the benchmark, so new CSV files are scored correctly at collection time.

Integrated scoring rules
------------------------
Original benchmark normalisation plus:
    1. "I'd"      == "I would"
    2. "backward" == "backwards"
    3. "180"      == "one eighty" == "one hundred eighty"
                    == "one hundred and eighty"

Outputs:
    WER
    CER
    CTER
    Exact Match
    Normalised Reference
    Normalised Transcript

The network timing design is otherwise kept the same:
    local   Pi transcribes locally
    remote  Pi sends audio to laptop STT server and receives the result

Jobs are shuffled to reduce thermal/link drift bias. Re-running with --resume
skips successful jobs already present in the same output CSV.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, asdict, fields
from pathlib import Path


# =============================================================================
# FINAL THESIS SCORING
# =============================================================================

_TAG_RE = re.compile(r"[\[\(<][^\]\)>]*[\]\)>]")
_PUNCT_RE = re.compile(r"[^\w\s']")

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}

_SCALES = {"hundred": 100, "thousand": 1000}

_SYNONYMS = {
    "ft": "feet",
    "foot": "feet",
    "metre": "meters",
    "metres": "meters",
    "meter": "meters",
    "degree": "degrees",
    "step": "steps",
    "second": "seconds",
    "ok": "okay",
}

CRITICAL_TOKENS = {
    "left", "right", "forward", "backward", "back", "ahead", "around",
    "rotate", "turn", "move", "walk", "dance", "stop", "reset", "relax",
    "rest", "return", "feet", "meters", "steps", "degrees",
}

_APOSTROPHES = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "\u00b4": "'",
    "`": "'",
})


def words_to_digits(words: list[str]) -> list[str]:
    """Original number handling plus 'one eighty' -> '180'."""
    fixed = []
    i = 0
    while i < len(words):
        if i + 1 < len(words) and words[i] == "one" and words[i + 1] == "eighty":
            fixed.append("180")
            i += 2
        else:
            fixed.append(words[i])
            i += 1

    out = []
    cur = None

    for w in fixed:
        if w in _ONES or w in _TENS:
            cur = (cur or 0) + _ONES.get(w, _TENS.get(w))
        elif w in _SCALES and cur is not None:
            cur *= _SCALES[w]
        elif w == "and" and cur is not None:
            continue
        else:
            if cur is not None:
                out.append(str(cur))
                cur = None
            out.append(w)

    if cur is not None:
        out.append(str(cur))

    return out


def normalise(text: str | None) -> str:
    """
    Final benchmark normalisation:
      I'd -> I would
      backwards -> backward
      spoken number forms -> digits
      benchmark unit/synonym normalisation
    """
    t = "" if text is None else str(text)
    t = t.translate(_APOSTROPHES).lower()
    t = _TAG_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)

    words = t.split()

    expanded = []
    for w in words:
        if w == "i'd":
            expanded.extend(["i", "would"])
        else:
            expanded.append(w)
    words = expanded

    words = [_SYNONYMS.get(w, w) for w in words]
    words = ["backward" if w == "backwards" else w for w in words]
    words = words_to_digits(words)

    return " ".join(words)


def edit_distance(ref, hyp) -> int:
    prev = list(range(len(hyp) + 1))

    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            if r == h:
                cur.append(prev[j - 1])
            else:
                cur.append(
                    1 + min(
                        prev[j - 1],  # substitution
                        prev[j],      # deletion
                        cur[j - 1],   # insertion
                    )
                )
        prev = cur

    return prev[-1]


def error_rate(ref: str, hyp: str, unit: str = "word") -> float:
    if not ref:
        return -1.0

    if unit == "word":
        r = ref.split()
        h = hyp.split()
    else:
        # Thesis benchmark CER ignores spaces.
        r = list(ref.replace(" ", ""))
        h = list(hyp.replace(" ", ""))

    if not r:
        return -1.0

    return edit_distance(r, h) / len(r)


def critical_sequence(text: str) -> list[str]:
    return [
        w for w in text.split()
        if w in CRITICAL_TOKENS or w.isdigit()
    ]


def critical_token_error_rate(ref_n: str, hyp_n: str) -> float:
    if not ref_n:
        return -1.0

    r = critical_sequence(ref_n)
    if not r:
        return -1.0

    h = critical_sequence(hyp_n)
    return edit_distance(r, h) / len(r)


def score(reference: str, hypothesis: str):
    ref_n = normalise(reference)
    hyp_n = normalise(hypothesis)

    wer = error_rate(ref_n, hyp_n, "word")
    cer = error_rate(ref_n, hyp_n, "char")
    cter = critical_token_error_rate(ref_n, hyp_n)
    exact = 1 if ref_n == hyp_n else 0

    return (
        round(wer, 5) if wer >= 0 else -1.0,
        round(cer, 5) if cer >= 0 else -1.0,
        round(cter, 5) if cter >= 0 else -1.0,
        exact,
        ref_n,
        hyp_n,
    )


# =============================================================================
# ROW SCHEMA
# =============================================================================

@dataclass
class Row:
    ts: str = ""
    condition: str = ""
    mode: str = ""            # local | remote
    model: str = ""
    codec: str = ""
    repeat: int = 0
    file: str = ""
    speaker: str = ""
    ref: str = ""
    hyp: str = ""
    ok: int = 1
    error: str = ""

    audio_sec: float = 0.0
    src_bytes: int = 0
    sent_bytes: int = 0

    encode_ms: float = 0.0
    read_ms: float = 0.0
    upload_ms: float = 0.0
    wait_ms: float = 0.0
    download_ms: float = 0.0
    total_ms: float = 0.0
    server_ms: float = 0.0
    server_recv_ms: float = 0.0
    server_io_ms: float = 0.0
    infer_ms: float = 0.0
    network_ms: float = 0.0
    rtf: float = 0.0

    rtt_pre_ms: float = 0.0
    rtt_post_ms: float = 0.0
    clock_offset_ms: float = 0.0
    up_kbps: float = 0.0

    wifi_signal_dbm: float = 0.0
    wifi_tx_mbps: float = 0.0
    cpu_temp_c: float = 0.0
    throttled: str = ""

    # Final thesis scoring
    wer: float = -1.0
    cer: float = -1.0
    cter: float = -1.0
    exact_match: int = 0
    normalised_ref: str = ""
    normalised_hyp: str = ""


HEADER = [f.name for f in fields(Row)]


# =============================================================================
# PI TELEMETRY
# =============================================================================

def cpu_temp() -> float:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        return 0.0


def throttled() -> str:
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        return out.split("=")[-1] if "=" in out else out
    except Exception:
        return ""


def wifi_link(iface: str) -> tuple[float, float]:
    """Return (signal dBm, tx bitrate Mb/s) from iw."""
    try:
        out = subprocess.run(
            ["iw", "dev", iface, "link"],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout

        sig = re.search(r"signal:\s*(-?\d+)\s*dBm", out)
        txr = re.search(r"tx bitrate:\s*([\d.]+)\s*MBit/s", out)

        return (
            float(sig.group(1)) if sig else 0.0,
            float(txr.group(1)) if txr else 0.0,
        )
    except Exception:
        return (0.0, 0.0)


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


# =============================================================================
# CODEC
# =============================================================================

def encode(path: Path, codec: str) -> tuple[bytes, float]:
    """Return (payload, encode_ms). WAV is passthrough."""
    t0 = time.perf_counter()

    if codec == "wav":
        data = path.read_bytes()
        return data, (time.perf_counter() - t0) * 1000.0

    if codec == "flac":
        args = ["-c:a", "flac", "-f", "flac"]
    elif codec == "opus":
        args = ["-c:a", "libopus", "-b:a", "24k", "-f", "ogg"]
    else:
        raise ValueError(f"unsupported codec {codec!r}")

    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-ac", "1", "-ar", "16000",
            *args, "pipe:1",
        ],
        capture_output=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[:400])

    return proc.stdout, (time.perf_counter() - t0) * 1000.0


CODEC_FMT = {"wav": "wav", "flac": "flac", "opus": "ogg"}


# =============================================================================
# REMOTE PATH
# =============================================================================

class RemoteClient:
    """Persistent keep-alive connection; TCP setup excluded from each run."""

    def __init__(self, host: str, port: int, timeout: float = 300.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.conn: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self.conn is None:
            self.conn = http.client.HTTPConnection(
                self.host,
                self.port,
                timeout=self.timeout,
            )
            self.conn.connect()
        return self.conn

    def reset(self) -> None:
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        self.conn = None

    def ping(self) -> tuple[float, float]:
        """Return (RTT ms, clock offset ms)."""
        conn = self._connect()

        t1 = time.time()
        conn.request("GET", "/ping")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        t4 = time.time()

        t2 = body["t_recv"]
        t3 = body["t_send"]

        rtt = (t4 - t1) - (t3 - t2)
        offset = ((t2 - t1) + (t3 - t4)) / 2.0

        return rtt * 1000.0, offset * 1000.0

    def transcribe(self, payload: bytes, spec: str, fmt: str, tag: str) -> dict:
        conn = self._connect()

        path = f"/transcribe?model={spec}&fmt={fmt}&tag={tag}"
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(payload)),
        }

        t0 = time.perf_counter()
        conn.request("POST", path, body=payload, headers=headers)
        t_sent = time.perf_counter()

        resp = conn.getresponse()
        t_hdr = time.perf_counter()

        raw = resp.read()
        t_done = time.perf_counter()

        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {raw[:300]!r}")

        body = json.loads(raw)

        body["_upload_ms"] = (t_sent - t0) * 1000.0
        body["_wait_ms"] = (t_hdr - t_sent) * 1000.0
        body["_download_ms"] = (t_done - t_hdr) * 1000.0
        body["_total_ms"] = (t_done - t0) * 1000.0

        return body


# =============================================================================
# LOCAL PATH
# =============================================================================

class LocalModel:
    def __init__(self, name: str):
        self.name = name
        self.compute = ""

        if name.startswith("faster:"):
            parts = name.split(":")
            self.kind = "faster"
            self.model_id = parts[1]
            self.compute = parts[2] if len(parts) > 2 else "int8"
        elif name.lower().startswith("vosk"):
            self.kind = "vosk"
        else:
            self.kind = "whisper"

        self._impl = None

    def load(self):
        if self._impl is not None:
            return

        if self.kind == "faster":
            from faster_whisper import WhisperModel
            self._impl = WhisperModel(
                self.model_id,
                device="cpu",
                compute_type=self.compute,
            )

        elif self.kind == "whisper":
            from transformers import pipeline
            repo = self.name if "/" in self.name else f"openai/{self.name}"
            self._impl = pipeline(
                "automatic-speech-recognition",
                model=repo,
            )

        else:
            from vosk import Model
            path = os.environ.get("VOSK_MODEL_DIR", ".")
            self._impl = Model(os.path.join(path, self.name))

        self._warm()

    def _warm(self):
        buf = io.BytesIO()

        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)

        tmp = Path("/tmp/_warm.wav")
        tmp.write_bytes(buf.getvalue())

        try:
            self.transcribe(tmp)
        except Exception:
            pass
        finally:
            tmp.unlink(missing_ok=True)

    def transcribe(self, path: Path) -> tuple[str, float]:
        self.load()

        t0 = time.perf_counter()

        if self.kind == "faster":
            segments, _ = self._impl.transcribe(
                str(path),
                language="en",
                beam_size=1,
                vad_filter=False,
            )
            text = " ".join(s.text for s in segments).strip()

        elif self.kind == "whisper":
            out = self._impl(str(path))
            text = (
                out.get("text")
                if isinstance(out, dict)
                else str(out)
            ).strip()

        else:
            from vosk import KaldiRecognizer

            with wave.open(str(path), "rb") as wf:
                rec = KaldiRecognizer(
                    self._impl,
                    wf.getframerate(),
                )
                rec.SetWords(False)

                while True:
                    frames = wf.readframes(4000)
                    if not frames:
                        break
                    rec.AcceptWaveform(frames)

                text = (
                    json.loads(rec.FinalResult()).get("text") or ""
                ).strip()

        return text, (time.perf_counter() - t0) * 1000.0


# =============================================================================
# REFERENCE / JOB HANDLING
# =============================================================================

def ref_from_filename(path: Path) -> str:
    """Filenames are the command text itself."""
    return path.stem.replace("_", " ").strip()


def build_jobs(files, modes, local_models, remote_models, codecs, repeats):
    jobs = []

    for f in files:
        for r in range(repeats):

            if "local" in modes:
                for m in local_models:
                    jobs.append(("local", m, "wav", r, f))

            if "remote" in modes:
                for m in remote_models:
                    for c in codecs:
                        jobs.append(("remote", m, c, r, f))

    return jobs


def load_done(out: Path) -> set[tuple]:
    if not out.exists():
        return set()

    done = set()

    with out.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("ok") == "1":
                done.add(
                    (
                        row.get("condition", ""),
                        row.get("mode", ""),
                        row.get("model", ""),
                        row.get("codec", ""),
                        row.get("repeat", ""),
                        row.get("file", ""),
                    )
                )

    return done


def open_writer(out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)

    new = not out.exists() or out.stat().st_size == 0
    fh = out.open("a", newline="", encoding="utf-8")

    writer = csv.DictWriter(
        fh,
        fieldnames=HEADER,
        extrasaction="ignore",
    )

    if new:
        writer.writeheader()
        fh.flush()

    return fh, writer


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="On-device vs network-assisted STT latency with final scoring"
    )

    ap.add_argument("--audio", required=True)
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--out", default="results/net_latency.csv")
    ap.add_argument("--server", default="192.168.137.1:8420")
    ap.add_argument(
        "--mode",
        default="both",
        choices=["local", "remote", "both"],
    )
    ap.add_argument("--local-models", default="whisper-base.en")
    ap.add_argument(
        "--remote-models",
        default="faster:small.en:float16",
    )
    ap.add_argument(
        "--codec",
        default="wav",
        help="comma list: wav,flac,opus",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap number of files (0 = all)",
    )
    ap.add_argument("--condition", default="default")
    ap.add_argument("--iface", default="wlan0")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ping-every", type=int, default=1)

    args = ap.parse_args()

    root = Path(args.audio).expanduser()
    files = sorted(root.rglob(args.glob))

    if not files:
        print(
            f"[fatal] no files matching {args.glob} under {root}",
            file=sys.stderr,
        )
        return 2

    if args.limit:
        files = files[:args.limit]

    modes = (
        ["local", "remote"]
        if args.mode == "both"
        else [args.mode]
    )

    local_models = [
        s.strip()
        for s in args.local_models.split(",")
        if s.strip()
    ]

    remote_models = [
        s.strip()
        for s in args.remote_models.split(",")
        if s.strip()
    ]

    codecs = [
        s.strip()
        for s in args.codec.split(",")
        if s.strip()
    ]

    for c in codecs:
        if c != "wav" and shutil.which("ffmpeg") is None:
            print(
                "[fatal] ffmpeg required for non-wav codecs",
                file=sys.stderr,
            )
            return 2

    jobs = build_jobs(
        files,
        modes,
        local_models,
        remote_models,
        codecs,
        args.repeats,
    )

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(jobs)

    out = Path(args.out)

    done = load_done(out) if args.resume else set()

    if done:
        before = len(jobs)

        jobs = [
            j for j in jobs
            if (
                args.condition,
                j[0],
                j[1],
                j[2],
                str(j[3]),
                str(j[4]),
            ) not in done
        ]

        print(
            f"[resume] skipping "
            f"{before - len(jobs)} completed runs"
        )

    host, _, port = args.server.partition(":")
    client = RemoteClient(
        host,
        int(port or 8420),
    )

    loaded: dict[str, LocalModel] = {}

    # IMPORTANT: preload/warm local models BEFORE timed benchmark jobs.
    # This prevents the first local sample from including model-load time.
    if "local" in modes:
        for model in local_models:
            print(f"[warmup] local model: {model}")
            lm = LocalModel(model)
            lm.load()
            loaded[model] = lm

    if "remote" in modes:
        try:
            rtt, off = client.ping()
            print(
                f"[link] rtt={rtt:.1f} ms  "
                f"clock_offset={off:.1f} ms"
            )
        except Exception as exc:
            print(
                f"[fatal] cannot reach {args.server}: {exc}",
                file=sys.stderr,
            )
            return 2

    fh, writer = open_writer(out)

    total = len(jobs)

    print(
        f"[run] {total} jobs  "
        f"condition={args.condition}"
    )

    t_start = time.time()

    rtt_pre = 0.0
    off = 0.0
    remote_n = 0

    try:
        for i, (mode, model, codec, rep, path) in enumerate(jobs, 1):

            row = Row(
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                condition=args.condition,
                mode=mode,
                model=model,
                codec=codec,
                repeat=rep,
                file=str(path),
                speaker=path.parent.name,
                ref=ref_from_filename(path),
                audio_sec=round(wav_duration(path), 3),
                src_bytes=path.stat().st_size,
                cpu_temp_c=round(cpu_temp(), 2),
                throttled=throttled(),
            )

            (
                row.wifi_signal_dbm,
                row.wifi_tx_mbps,
            ) = wifi_link(args.iface)

            try:
                if mode == "local":
                    lm = loaded[model]

                    t0 = time.perf_counter()
                    hyp, infer_ms = lm.transcribe(path)

                    row.total_ms = round(
                        (time.perf_counter() - t0) * 1000.0,
                        3,
                    )
                    row.infer_ms = round(infer_ms, 3)
                    row.hyp = hyp
                    row.sent_bytes = 0

                else:
                    if (
                        args.ping_every
                        and remote_n % args.ping_every == 0
                    ):
                        rtt_pre, off = client.ping()

                    payload, enc_ms = encode(path, codec)

                    row.encode_ms = round(enc_ms, 3)
                    row.sent_bytes = len(payload)

                    body = client.transcribe(
                        payload,
                        model,
                        CODEC_FMT[codec],
                        args.condition,
                    )

                    row.hyp = body.get("text", "")

                    row.upload_ms = round(
                        body["_upload_ms"],
                        3,
                    )
                    row.wait_ms = round(
                        body["_wait_ms"],
                        3,
                    )
                    row.download_ms = round(
                        body["_download_ms"],
                        3,
                    )
                    row.total_ms = round(
                        body["_total_ms"],
                        3,
                    )

                    row.server_ms = round(
                        body.get("server_ms", 0.0),
                        3,
                    )
                    row.server_recv_ms = round(
                        body.get("recv_ms", 0.0),
                        3,
                    )
                    row.server_io_ms = round(
                        body.get("io_ms", 0.0),
                        3,
                    )
                    row.infer_ms = round(
                        body.get("infer_ms", 0.0),
                        3,
                    )

                    row.network_ms = round(
                        row.total_ms - row.server_ms,
                        3,
                    )

                    row.rtt_pre_ms = round(
                        rtt_pre,
                        3,
                    )
                    row.clock_offset_ms = round(
                        off,
                        3,
                    )

                    if row.upload_ms > 0:
                        row.up_kbps = round(
                            (
                                row.sent_bytes
                                * 8
                                / 1000.0
                            )
                            / (
                                row.upload_ms
                                / 1000.0
                            ),
                            1,
                        )

                    try:
                        row.rtt_post_ms = round(
                            client.ping()[0],
                            3,
                        )
                    except Exception:
                        pass

                    remote_n += 1

                if row.audio_sec > 0:
                    row.rtf = round(
                        (
                            row.total_ms / 1000.0
                        )
                        / row.audio_sec,
                        5,
                    )

                (
                    row.wer,
                    row.cer,
                    row.cter,
                    row.exact_match,
                    row.normalised_ref,
                    row.normalised_hyp,
                ) = score(
                    row.ref,
                    row.hyp,
                )

            except Exception as exc:
                row.ok = 0
                row.error = (
                    f"{type(exc).__name__}: {exc}"
                )[:400]

                if mode == "remote":
                    client.reset()

            writer.writerow(asdict(row))
            fh.flush()

            if i % 10 == 0 or i == total:
                rate = i / max(
                    time.time() - t_start,
                    1e-6,
                )

                eta = (
                    total - i
                ) / max(rate, 1e-9)

                print(
                    f"[{i}/{total}] "
                    f"{mode:6s} "
                    f"{model:28s} "
                    f"{row.total_ms:8.1f} ms  "
                    f"WER={row.wer:7.4f}  "
                    f"Exact={row.exact_match}  "
                    f"eta={eta/60:.1f} min"
                )

    except KeyboardInterrupt:
        print(
            "\n[interrupt] partial results "
            "are already on disk"
        )

    finally:
        fh.close()
        client.reset()

    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
