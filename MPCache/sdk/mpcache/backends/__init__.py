from .base import Backend
from . import cogvideox, wan, hunyuan

REGISTRY = {b.backend.name: b.backend for b in (cogvideox, wan, hunyuan)}


def get(name: str) -> Backend:
    if name not in REGISTRY:
        raise KeyError(f"unknown backend {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]
