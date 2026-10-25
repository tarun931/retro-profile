"""Read TensorBoard scalars without installing tensorboard.

TensorBoard is not installed on the Mac, and the per-epoch loss components
(e.g. ``train/ordcon/age/magnitude_epoch``) exist only in the event files. This
parses the event files directly: TFRecord framing (8-byte length, 4-byte CRC,
payload, 4-byte CRC) around Event protobufs, reading ``Summary.Value`` entries
with a ``simple_value``. Verified 2026-10-10 against a run's checkpoint name:
the parsed minimum of ``val/combined_mae`` matched ``val=1.3223`` exactly.

    python scripts/read_tb_scalars.py runs/<run> --grep ordcon --at 16
"""

from __future__ import annotations

import argparse
import collections
import struct
from pathlib import Path


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return r, i


def _fields(b: bytes):
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 1:
            v, i = b[i:i + 8], i + 8
        elif w == 5:
            v, i = b[i:i + 4], i + 4
        elif w == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        else:
            raise ValueError(f"unsupported protobuf wire type {w}")
        yield f, w, v


def scalars(path: Path) -> dict[str, list[tuple[int, float]]]:
    """{tag: [(step, value), ...]} from one event file."""
    out: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    data = Path(path).read_bytes()
    i = 0
    while i + 12 <= len(data):
        n = struct.unpack("<Q", data[i:i + 8])[0]
        rec = data[i + 12:i + 12 + n]
        i += 12 + n + 4
        step, summ = 0, None
        for f, w, v in _fields(rec):
            if f == 2 and w == 0:
                step = v
            if f == 5 and w == 2:
                summ = v
        if summ is None:
            continue
        for f, w, val in _fields(summ):
            if f != 1 or w != 2:
                continue
            tag = sv = None
            for g, ww, x in _fields(val):
                if g == 1 and ww == 2:
                    tag = x.decode()
                if g == 2 and ww == 5:
                    sv = struct.unpack("<f", x)[0]
            if tag is not None and sv is not None:
                out[tag].append((step, sv))
    return out


def run_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """All scalars of a run (every event file under runs/<run>/tb)."""
    merged: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    for p in sorted(Path(run_dir).glob("tb/version_*/events.out.tfevents.*")):
        for k, v in scalars(p).items():
            merged[k].extend(v)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--grep", default="", help="only tags containing this text")
    ap.add_argument("--at", type=int, default=None, help="print the value at this epoch index (epoch-level tags)")
    args = ap.parse_args()
    for tag, vals in sorted(run_scalars(args.run).items()):
        if args.grep not in tag:
            continue
        v = [x for _, x in vals]
        at = f"  @{args.at}={v[args.at]:.4f}" if args.at is not None and args.at < len(v) else ""
        print(f"{tag:45s} n={len(v):4d}  first={v[0]:.4f}  last={v[-1]:.4f}  min={min(v):.4f}{at}")


if __name__ == "__main__":
    main()
