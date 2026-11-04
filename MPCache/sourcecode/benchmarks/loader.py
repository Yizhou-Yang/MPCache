"""Unified loader: the only place that knows which benchmarks exist.

Mirrors the role of `benchmarks/loader.py` in the CuratorMem layout: the method
package stays free of benchmark names, and this file maps a name to an adapter.
"""

from __future__ import annotations
import importlib, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

MODULES = {"cogvideox": "latest.cogvideox", "wan": "latest.wan", "hunyuan": "latest.hunyuan"}


def available() -> list[str]:
    return sorted(MODULES)


def get(name: str):
    if name not in MODULES:
        raise KeyError(f"unknown benchmark {name!r}; have {available()}")
    mod = importlib.import_module(MODULES[name])
    return mod.backend
