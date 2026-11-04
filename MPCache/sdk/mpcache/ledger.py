"""Append results to a JSONL file the moment they exist.

Machines get reclaimed between arms; a number that is not on disk when the box
disappears is gone. Every evaluated arm therefore goes through `record`.
"""

from __future__ import annotations
import json, os, time, socket


def record(path: str, tag: str, **fields) -> None:
    row = {"tag": tag, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname(), **fields}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def read(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
