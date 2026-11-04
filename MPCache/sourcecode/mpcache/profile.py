"""The sensitivity measurement, as a protocol rather than a script.

`measure(run)` takes a callable that renders one prompt and optionally skips one
step, and returns the per-step cost. Every benchmark supplies its own `run`; the
protocol - skip exactly one step, compare against the full run, normalise by the
maximum - is shared.
"""

from __future__ import annotations


def measure(run, steps: int) -> list[float]:
    """run(skip: int | None) -> list of frames. Returns the normalised profile."""
    full = run(None)
    profile = []
    for k in range(steps):
        frames = run(None if k == 0 else k)
        profile.append(score(full, frames))
    hi = max(profile) or 1.0
    return [p / hi for p in profile]


def score(a, b) -> float:
    """LPIPS between two frame lists; imported lazily so the protocol is testable."""
    import numpy as np, torch, lpips
    L = lpips.LPIPS(net="vgg").cuda()
    ta = torch.from_numpy(np.stack(a).astype("float32") / 127.5 - 1).permute(0, 3, 1, 2).cuda()
    tb = torch.from_numpy(np.stack(b).astype("float32") / 127.5 - 1).permute(0, 3, 1, 2).cuda()
    with torch.no_grad():
        return float(L(ta, tb).mean().item())


def head_length(profile, frac: float = 0.5) -> int:
    hi = max(profile)
    h = 0
    for i, w in enumerate(profile):
        if i == 0:
            continue
        if w >= frac * hi:
            h = i + 1
        else:
            break
    return h
