"""Bounded apparent-byte and entry counts over one live task root.

A vanished directory entry invalidates that pass: restart instead of returning
an undercount. All other filesystem errors propagate. The same operation can
capture a future attempt's baseline before its ledger and sample it during work.
"""

from __future__ import annotations

import argparse
import os
import stat
import time
from pathlib import Path


def _sample_once(root: Path, deadline: float) -> tuple[int, int]:
    root_stat = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("task root is not a directory")
    device = root_stat.st_dev
    stack = [root]
    seen_links: set[tuple[int, int]] = set()
    apparent_bytes = 0
    entries = 0
    while stack:
        if time.monotonic() >= deadline:
            raise TimeoutError("task-root sampling deadline exhausted")
        path = stack.pop()
        info = os.stat(path, follow_symlinks=False)
        if info.st_dev != device:
            continue
        entries += 1
        identity = (info.st_dev, info.st_ino)
        if not stat.S_ISDIR(info.st_mode) and (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink <= 1
            or identity not in seen_links
        ):
            apparent_bytes += info.st_size
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            seen_links.add(identity)
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as listing:
                stack.extend(Path(item.path) for item in listing)
    return apparent_bytes, entries


def sample_tree(root: Path, *, seconds: float = 4.0) -> tuple[int, int]:
    if not 0 < seconds <= 5:
        raise ValueError("sampling deadline must be in (0, 5] seconds")
    deadline = time.monotonic() + seconds
    while True:
        try:
            return _sample_once(root, deadline)
        except FileNotFoundError as error:
            # Only a vanished path is transient. A fresh pass must observe all
            # currently live entries, including a recreated WAL/SHM file.
            if time.monotonic() >= deadline:
                raise TimeoutError("task-root sampling deadline exhausted") from error
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("bytes", "inodes", "both"), required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    counts = sample_tree(args.root, seconds=args.seconds)
    if args.kind == "both":
        print(*counts)
    else:
        print(counts[0 if args.kind == "bytes" else 1])


if __name__ == "__main__":
    main()
